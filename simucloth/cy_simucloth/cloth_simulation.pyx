# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: embedsignature=True
# cython: binding=False

import logging
from math import isnan, isinf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import numpy as np
cimport numpy as cnp

cnp.import_array()

DTYPE = np.float32
ctypedef cnp.float32_t DTYPE_t

from libc.math cimport sqrt, fabs, copysign
from cython.parallel import prange

from .types cimport SimulationObject, Particle, Constraint, Mesh, ObjectType, Vector3
from .sim_objects cimport get_simulation_objects, get_object_count


cdef int update_positions(float dt, float damping, Vector3 gravity) nogil:
    cdef:
        SimulationObject** simulation_objects = get_simulation_objects()
        Py_ssize_t object_count = get_object_count()
        SimulationObject* sim_obj
        Particle* particle
        int i, j

    for i in range(object_count):
        sim_obj = simulation_objects[i]
        if sim_obj.type != ObjectType.CLOTH:
            continue
        
        for j in range(sim_obj.tot_particles):
            particle = &sim_obj.particles[j]
            if not particle.pinned:
                particle.vel.x += gravity.x * dt
                particle.vel.y += gravity.y * dt
                particle.vel.z += gravity.z * dt
                
                particle.vel.x *= (<float>1.0 - damping)
                particle.vel.y *= (<float>1.0 - damping)
                particle.vel.z *= (<float>1.0 - damping)
                
                particle.prev_pos = particle.pos
                
                particle.pos.x += particle.vel.x * dt
                particle.pos.y += particle.vel.y * dt
                particle.pos.z += particle.vel.z * dt
    
    return 0

def update_positions_py(float dt, float damping, list gravity):
    # logger.info(f"Updating positions with dt={dt} and damping={damping}")
    update_positions(dt, damping, Vector3(gravity[0], gravity[1], gravity[2]))
    # logger.info("Positions updated")
    # print_particle_stats(simulation_objects[0], "After update_positions")

cdef void solve_constraints(float stiffness, int iterations) nogil:
    cdef:
        SimulationObject** simulation_objects = get_simulation_objects()
        Py_ssize_t object_count = get_object_count()
        int obj_index, i, j, k
        SimulationObject* sim_obj
        float dx, dy, dz, distance, difference, factor
        Vector3 correction
        float max_correction = 1.0  # Maximum allowed correction per iteration
        Particle* p1
        Particle* p2
        Constraint* constraint

    for obj_index in range(object_count):
        sim_obj = simulation_objects[obj_index]
        if sim_obj.type != ObjectType.CLOTH:
            continue

        for i in range(iterations):
            for j in range(sim_obj.tot_constraints):
                constraint = &sim_obj.constraints[j]
                p1 = &sim_obj.particles[constraint.v1]
                p2 = &sim_obj.particles[constraint.v2]

                dx = p2.pos.x - p1.pos.x
                dy = p2.pos.y - p1.pos.y
                dz = p2.pos.z - p1.pos.z
                distance = sqrt(dx*dx + dy*dy + dz*dz)

                if distance > 0:
                    difference = (distance - constraint.rest_length) / distance
                    factor = stiffness * difference * 0.5  # Reduce the factor by half

                    # Limit the correction
                    if factor > max_correction:
                        factor = max_correction
                    elif factor < -max_correction:
                        factor = -max_correction

                    correction.x = dx * factor
                    correction.y = dy * factor
                    correction.z = dz * factor

                    if not p1.pinned:
                        p1.pos.x += correction.x
                        p1.pos.y += correction.y
                        p1.pos.z += correction.z

                    if not p2.pinned:
                        p2.pos.x -= correction.x
                        p2.pos.y -= correction.y
                        p2.pos.z -= correction.z

            # Add a global position check after each iteration
            for k in prange(sim_obj.tot_particles, nogil=True):
                if not sim_obj.particles[k].pinned:
                    if fabs(sim_obj.particles[k].pos.x) > 100.0:
                        sim_obj.particles[k].pos.x = copysign(100.0, sim_obj.particles[k].pos.x)
                    if fabs(sim_obj.particles[k].pos.y) > 100.0:
                        sim_obj.particles[k].pos.y = copysign(100.0, sim_obj.particles[k].pos.y)
                    if fabs(sim_obj.particles[k].pos.z) > 100.0:
                        sim_obj.particles[k].pos.z = copysign(100.0, sim_obj.particles[k].pos.z)

def solve_constraints_py(
    float stiffness,
    int iterations
):
    # logger.info(f"Solving constraints with stiffness={stiffness} and iterations={iterations}")
    solve_constraints(stiffness, iterations)
    # logger.info("Constraints solved")
    # print_particle_stats(simulation_objects[0], "After solve_constraints")

cdef int update_velocities(float dt) nogil:
    cdef:
        SimulationObject** simulation_objects = get_simulation_objects()
        Py_ssize_t object_count = get_object_count()
        int i, j
        SimulationObject* sim_obj
        Particle* particle

    for i in range(object_count):
        sim_obj = simulation_objects[i]
        if sim_obj.type != ObjectType.CLOTH:
            continue
        
        for j in range(sim_obj.tot_particles):
            particle = &sim_obj.particles[j]
            if not particle.pinned:
                particle.vel.x = (particle.pos.x - particle.prev_pos.x) / dt
                particle.vel.y = (particle.pos.y - particle.prev_pos.y) / dt
                particle.vel.z = (particle.pos.z - particle.prev_pos.z) / dt
    
    return 0

def update_velocities_py(float dt):
    # logger.info(f"Updating velocities with dt={dt}")
    update_velocities(dt)
    # logger.info("Velocities updated")
    # print_particle_stats(simulation_objects[0], "After update_velocities")


cdef int _get_positions_parallel(SimulationObject* sim_obj, DTYPE_t[:, ::1] positions) nogil:
    cdef int i
    for i in prange(sim_obj.tot_particles, nogil=True):
        positions[i, 0] = sim_obj.particles[i].pos.x
        positions[i, 1] = sim_obj.particles[i].pos.y
        positions[i, 2] = sim_obj.particles[i].pos.z
    
    return 0

def get_positions_py(int object_index):
    # logger.info(f"Getting positions for object index: {object_index}")
    cdef:
        SimulationObject** simulation_objects = get_simulation_objects()
        Py_ssize_t object_count = get_object_count()
        SimulationObject* sim_obj
        cnp.ndarray[DTYPE_t, ndim=2] positions
        int i

    if object_index < 0 or object_index >= object_count:
        logger.error(f"Invalid object index: {object_index}, object_count: {object_count}")
        raise IndexError("Invalid object index")

    sim_obj = simulation_objects[object_index]
    if sim_obj == NULL:
        logger.error(f"Simulation object at index {object_index} is NULL")
        raise ValueError(f"Simulation object at index {object_index} is NULL")

    logger.info(f"Total particles: {sim_obj.tot_particles}")
    
    if sim_obj.tot_particles <= 0:
        logger.error(f"Invalid number of particles: {sim_obj.tot_particles}")
        return np.empty((0, 3), dtype=DTYPE)

    positions = np.empty((sim_obj.tot_particles, 3), dtype=DTYPE)

    if sim_obj.particles == NULL:
        logger.error(f"Particles array is NULL for object at index {object_index}")
        return np.empty((0, 3), dtype=DTYPE)

    if sim_obj.tot_particles > 1000:
        _get_positions_parallel(sim_obj, positions)
    else:
        for i in range(sim_obj.tot_particles):
            if &sim_obj.particles[i] == NULL:
                logger.error(f"Null particle pointer at index: {i}")
                continue
            positions[i, 0] = sim_obj.particles[i].pos.x
            positions[i, 1] = sim_obj.particles[i].pos.y
            positions[i, 2] = sim_obj.particles[i].pos.z

    # Add a check for NaN values
    if np.isnan(positions).any():
        logger.error("NaN values detected in positions array")
        # Optionally, print the indices of NaN values
        nan_indices = np.where(np.isnan(positions))
        logger.error(f"NaN indices: {nan_indices}")

    # logger.info(f"Positions retrieved for {sim_obj.tot_particles} particles")
    return positions


cdef void print_particle_stats(SimulationObject* sim_obj, const char* step_name):
    cdef int i
    cdef float min_x = 1e38, max_x = -1e38
    cdef float min_y = 1e38, max_y = -1e38
    cdef float min_z = 1e38, max_z = -1e38
    cdef int nan_count = 0
    cdef int inf_count = 0

    for i in range(sim_obj.tot_particles):
        if isnan(sim_obj.particles[i].pos.x) or isnan(sim_obj.particles[i].pos.y) or isnan(sim_obj.particles[i].pos.z):
            nan_count += 1
        elif isinf(sim_obj.particles[i].pos.x) or isinf(sim_obj.particles[i].pos.y) or isinf(sim_obj.particles[i].pos.z):
            inf_count += 1
        else:
            min_x = min(min_x, sim_obj.particles[i].pos.x)
            max_x = max(max_x, sim_obj.particles[i].pos.x)
            min_y = min(min_y, sim_obj.particles[i].pos.y)
            max_y = max(max_y, sim_obj.particles[i].pos.y)
            min_z = min(min_z, sim_obj.particles[i].pos.z)
            max_z = max(max_z, sim_obj.particles[i].pos.z)

    logger.info("%s:\n", step_name)
    logger.info("  NaN count: %d\n", nan_count)
    logger.info("  Inf count: %d\n", inf_count)
    logger.info("  X range: [%f, %f]\n", min_x, max_x)
    logger.info("  Y range: [%f, %f]\n", min_y, max_y)
    logger.info("  Z range: [%f, %f]\n", min_z, max_z)
