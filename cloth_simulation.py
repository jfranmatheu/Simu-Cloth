import numpy as np
from typing import List, Set, Tuple
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom timing decorator
def time_it(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"[TIME-IT]: {func.__name__} took: {end_time - start_time:.6f} sec")
        return result
    return wrapper

@time_it
def update_positions(
    positions: np.ndarray,
    prev_positions: np.ndarray,
    velocities: np.ndarray,
    pinned_vertices: Set[int],
    dt: float,
    damping: float
) -> Tuple[np.ndarray, np.ndarray]:
    gravity = np.array([0.0, 0.0, -9.8])
    for i in range(len(positions)):
        if i not in pinned_vertices:
            velocity = (positions[i] - prev_positions[i]) / dt
            velocity += gravity * dt
            velocity *= (1.0 - damping)
            prev_positions[i] = positions[i].copy()
            positions[i] += velocity * dt
    return positions, prev_positions

@time_it
def solve_constraints(
    positions: np.ndarray,
    constraints: List[Tuple[int, int, float]],
    pinned_vertices: Set[int],
    stiffness: float,
    iterations: int
) -> np.ndarray:
    for _ in range(iterations):
        for v1, v2, rest_length in constraints:
            if v1 not in pinned_vertices or v2 not in pinned_vertices:
                p1, p2 = positions[v1], positions[v2]
                delta = p2 - p1
                dist = np.linalg.norm(delta)
                correction = delta * (1 - rest_length / dist) * 0.5 * stiffness
                if v1 not in pinned_vertices:
                    positions[v1] += correction
                if v2 not in pinned_vertices:
                    positions[v2] -= correction
    return positions

@time_it
def update_velocities(
    positions: np.ndarray,
    prev_positions: np.ndarray,
    velocities: np.ndarray,
    pinned_vertices: Set[int],
    dt: float
) -> np.ndarray:
    for i in range(len(positions)):
        if i not in pinned_vertices:
            velocities[i] = (positions[i] - prev_positions[i]) / dt
    return velocities

@time_it
def apply_drag(
    positions: np.ndarray,
    drag_index: int,
    drag_position: np.ndarray,
    drag_radius: float
) -> np.ndarray:
    if drag_index >= 0:
        for i in range(len(positions)):
            if i != drag_index:
                delta = positions[i] - drag_position
                distance = np.linalg.norm(delta)
                if distance < drag_radius:
                    influence = 1.0 - distance / drag_radius
                    positions[i] += (drag_position - positions[drag_index]) * influence
    return positions