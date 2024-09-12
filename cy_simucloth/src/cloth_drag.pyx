# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: embedsignature=True
# cython: binding=False

import numpy as np
cimport numpy as cnp
from cython.parallel import prange
from libc.stdlib cimport malloc, free
from libc.math cimport sqrt

from .types cimport Particle, Triangle, DragData
from .geom_utils cimport ray_triangle_intersection
from .sim_objects cimport get_simulation_objects, get_object_count

cnp.import_array()

DTYPE = np.float32
ctypedef cnp.float32_t DTYPE_t


################################################################################
# DRAGGING.
################################################################################

cdef DragData* create_drag_data(DTYPE_t drag_influence_radius):
    cdef DragData* drag_data = <DragData*>malloc(sizeof(DragData))
    if drag_data is NULL:
        raise MemoryError("Failed to allocate memory for DragData")
    drag_data.dragging = False
    drag_data.drag_vertex_index = -1
    drag_data.drag_start_x = 0
    drag_data.drag_start_y = 0
    drag_data.drag_start_z = 0
    drag_data.drag_current_x = 0
    drag_data.drag_current_y = 0
    drag_data.drag_current_z = 0
    drag_data.drag_influence_radius = drag_influence_radius
    return drag_data

def create_drag_data_py(DTYPE_t drag_influence_radius):
    return <size_t>create_drag_data(drag_influence_radius)

def free_drag_data_py(size_t drag_data_ptr):
    free(<DragData*>drag_data_ptr)

cdef bint start_drag(
    DragData* drag_data,
    Particle* particles,
    Py_ssize_t pcount,
    Triangle* triangles,
    Py_ssize_t tcount,
    DTYPE_t ray_origin_x,
    DTYPE_t ray_origin_y,
    DTYPE_t ray_origin_z,
    DTYPE_t ray_direction_x,
    DTYPE_t ray_direction_y,
    DTYPE_t ray_direction_z
) nogil:
    cdef:
        Py_ssize_t i, i1, i2, i3
        DTYPE_t closest_distance = 1e30
        Py_ssize_t closest_vertex = -1
        DTYPE_t intersection_x, intersection_y, intersection_z
        DTYPE_t dx, dy, dz, distance

    for i in range(tcount):
        i1, i2, i3 = triangles[i].i1, triangles[i].i2, triangles[i].i3
        if ray_triangle_intersection(
            ray_origin_x, ray_origin_y, ray_origin_z,
            ray_direction_x, ray_direction_y, ray_direction_z,
            particles[i1].x, particles[i1].y, particles[i1].z,
            particles[i2].x, particles[i2].y, particles[i2].z,
            particles[i3].x, particles[i3].y, particles[i3].z,
            &intersection_x, &intersection_y, &intersection_z
        ):
            dx = intersection_x - ray_origin_x
            dy = intersection_y - ray_origin_y
            dz = intersection_z - ray_origin_z
            distance = dx * dx + dy * dy + dz * dz
            if distance < closest_distance:
                closest_distance = distance
                closest_vertex = i1  # You might want to find the closest vertex of the triangle here
                drag_data.plane_point_x = intersection_x
                drag_data.plane_point_y = intersection_y
                drag_data.plane_point_z = intersection_z

    if closest_vertex != -1:
        drag_data.dragging = True
        drag_data.drag_vertex_index = closest_vertex
        drag_data.drag_start_x = particles[closest_vertex].x
        drag_data.drag_start_y = particles[closest_vertex].y
        drag_data.drag_start_z = particles[closest_vertex].z
        drag_data.drag_current_x = drag_data.drag_start_x
        drag_data.drag_current_y = drag_data.drag_start_y
        drag_data.drag_current_z = drag_data.drag_start_z

        # Calculate plane normal (using ray direction as an approximation)
        drag_data.plane_normal_x = ray_direction_x
        drag_data.plane_normal_y = ray_direction_y
        drag_data.plane_normal_z = ray_direction_z
        return True
    return False

# Update the Python wrapper for start_drag
def start_drag_py(
    size_t drag_data_ptr,
    size_t particles_ptr,
    Py_ssize_t pcount,
    size_t triangles_ptr,
    Py_ssize_t tcount,
    cnp.ndarray[DTYPE_t, ndim=1] ray_origin,
    cnp.ndarray[DTYPE_t, ndim=1] ray_direction
):
    return start_drag(
        <DragData*>drag_data_ptr,
        <Particle*>particles_ptr,
        pcount,
        <Triangle*>triangles_ptr,
        tcount,
        ray_origin[0], ray_origin[1], ray_origin[2],
        ray_direction[0], ray_direction[1], ray_direction[2]
    )

cdef void update_drag(
    DragData* drag_data,
    Particle* particles,
    Py_ssize_t pcount,
    DTYPE_t new_x,
    DTYPE_t new_y,
    DTYPE_t new_z
) nogil:
    if not drag_data.dragging:
        return

    cdef:
        Py_ssize_t i
        DTYPE_t dx, dy, dz, distance, influence
        DTYPE_t plane_dot, t, projected_x, projected_y, projected_z
        DTYPE_t move_x, move_y, move_z

    # Project the new point onto the drag plane
    dx = new_x - drag_data.plane_point_x
    dy = new_y - drag_data.plane_point_y
    dz = new_z - drag_data.plane_point_z
    plane_dot = dx * drag_data.plane_normal_x + dy * drag_data.plane_normal_y + dz * drag_data.plane_normal_z
    projected_x = new_x - plane_dot * drag_data.plane_normal_x
    projected_y = new_y - plane_dot * drag_data.plane_normal_y
    projected_z = new_z - plane_dot * drag_data.plane_normal_z

    move_x = projected_x - drag_data.drag_current_x
    move_y = projected_y - drag_data.drag_current_y
    move_z = projected_z - drag_data.drag_current_z

    drag_data.drag_current_x = projected_x
    drag_data.drag_current_y = projected_y
    drag_data.drag_current_z = projected_z

    for i in range(pcount):
        dx = particles[i].x - drag_data.drag_start_x
        dy = particles[i].y - drag_data.drag_start_y
        dz = particles[i].z - drag_data.drag_start_z
        distance = sqrt(dx*dx + dy*dy + dz*dz)
        
        if distance < drag_data.drag_influence_radius:
            influence = 1.0 - distance / drag_data.drag_influence_radius
            particles[i].x += move_x * influence
            particles[i].y += move_y * influence
            particles[i].z += move_z * influence

    particles[drag_data.drag_vertex_index].x = projected_x
    particles[drag_data.drag_vertex_index].y = projected_y
    particles[drag_data.drag_vertex_index].z = projected_z

def update_drag_py(
    size_t drag_data_ptr,
    size_t particles_ptr,
    Py_ssize_t pcount,
    cnp.ndarray[DTYPE_t, ndim=1] new_position
):
    update_drag(
        <DragData*>drag_data_ptr,
        <Particle*>particles_ptr,
        pcount,
        new_position[0], new_position[1], new_position[2]
    )

cdef void end_drag(DragData* drag_data) nogil:
    drag_data.dragging = False
    drag_data.drag_vertex_index = -1

def end_drag_py(size_t drag_data_ptr):
    end_drag(<DragData*>drag_data_ptr)


cdef void apply_drag(
    DragData* drag_data,
    Particle* particles,
    Py_ssize_t pcount
) nogil:
    if not drag_data.dragging:
        return

    cdef:
        Py_ssize_t i
        DTYPE_t dx, dy, dz, distance, influence
        DTYPE_t move_x, move_y, move_z

    move_x = drag_data.drag_current_x - drag_data.drag_start_x
    move_y = drag_data.drag_current_y - drag_data.drag_start_y
    move_z = drag_data.drag_current_z - drag_data.drag_start_z

    for i in range(pcount):
        dx = particles[i].x - drag_data.drag_start_x
        dy = particles[i].y - drag_data.drag_start_y
        dz = particles[i].z - drag_data.drag_start_z
        distance = sqrt(dx*dx + dy*dy + dz*dz)
        
        if distance < drag_data.drag_influence_radius:
            influence = 1.0 - distance / drag_data.drag_influence_radius
            particles[i].x += move_x * influence
            particles[i].y += move_y * influence
            particles[i].z += move_z * influence

    # Update the start position for the next frame
    drag_data.drag_start_x = drag_data.drag_current_x
    drag_data.drag_start_y = drag_data.drag_current_y
    drag_data.drag_start_z = drag_data.drag_current_z

def apply_drag_py(
    size_t drag_data_ptr,
    size_t particles_ptr,
    Py_ssize_t pcount
):
    apply_drag(<DragData*>drag_data_ptr, <Particle*>particles_ptr, pcount)
