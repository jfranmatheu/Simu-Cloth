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

cnp.import_array()

DTYPE = np.float64
ctypedef cnp.float64_t DTYPE_t


cdef struct Triangle:
    Py_ssize_t i1, i2, i3

cdef struct Particle:
    DTYPE_t x, y, z
    DTYPE_t prev_x, prev_y, prev_z
    DTYPE_t vx, vy, vz
    bint pinned

cdef struct Constraint:
    Py_ssize_t v1, v2
    DTYPE_t rest_length


cdef Triangle* create_triangles(list triangles_list):
    cdef:
        Py_ssize_t tcount = len(triangles_list)
        Triangle* triangles = <Triangle*>malloc(tcount * sizeof(Triangle))
        Py_ssize_t i

    if triangles is NULL:
        raise MemoryError("Failed to allocate memory for triangles")

    for i in range(tcount):
        triangles[i].i1 = triangles_list[i][0]
        triangles[i].i2 = triangles_list[i][1]
        triangles[i].i3 = triangles_list[i][2]

    return triangles

def create_triangles_py(list triangles_list):
    return <size_t>create_triangles(triangles_list)


cdef Particle* create_particles(DTYPE_t[:, ::1] positions, set pinned_vertices):
    cdef:
        Py_ssize_t pcount = positions.shape[0]
        Particle* particles = <Particle*>malloc(pcount * sizeof(Particle))
        Py_ssize_t i
        Particle* particle

    if particles is NULL:
        raise MemoryError("Failed to allocate memory for particles")

    for i in range(pcount):
        particle = &particles[i]
        particle.x = positions[i, 0]
        particle.y = positions[i, 1]
        particle.z = positions[i, 2]
        particle.prev_x = positions[i, 0]
        particle.prev_y = positions[i, 1]
        particle.prev_z = positions[i, 2]
        particle.vx = 0
        particle.vy = 0
        particle.vz = 0
        particle.pinned = i in pinned_vertices

    return particles

def create_particles_py(cnp.ndarray[DTYPE_t, ndim=2] positions, set pinned_vertices):
    return <size_t>create_particles(positions, pinned_vertices)


cdef Constraint* create_constraints(list constraints_list):
    cdef:
        Py_ssize_t ccount = len(constraints_list)
        Constraint* constraints = <Constraint*>malloc(ccount * sizeof(Constraint))
        Py_ssize_t i
        Constraint* constraint

    if constraints is NULL:
        raise MemoryError("Failed to allocate memory for constraints")

    for i in range(ccount):
        constraint = &constraints[i]
        constraint.v1 = constraints_list[i][0]
        constraint.v2 = constraints_list[i][1]
        constraint.rest_length = constraints_list[i][2]

    return constraints

def create_constraints_py(list constraints_list):
    return <size_t>create_constraints(constraints_list)


def free_triangles_py(size_t triangles_ptr):
    free(<void*>triangles_ptr)

def free_particles_py(size_t particles_ptr):
    free(<void*>particles_ptr)

def free_constraints_py(size_t constraints_ptr):
    free(<void*>constraints_ptr)


cdef void update_positions(
    Particle* particles,
    Py_ssize_t pcount,
    double dt,
    double damping
) nogil:
    cdef:
        int i
        double gravity_x = 0.0
        double gravity_y = 0.0
        double gravity_z = -9.8
        double* velocities = <double*>malloc(pcount * 3 * sizeof(double))
        Particle* particle
    
    if velocities == NULL:
        with gil:
            raise MemoryError("Failed to allocate memory for velocities")

    for i in prange(pcount, nogil=True):
        particle = &particles[i]
        if not particle.pinned:
            velocities[i*3] = (particle.x - particle.prev_x) / dt
            velocities[i*3 + 1] = (particle.y - particle.prev_y) / dt
            velocities[i*3 + 2] = (particle.z - particle.prev_z) / dt
            
            velocities[i*3] += gravity_x * dt
            velocities[i*3 + 1] += gravity_y * dt
            velocities[i*3 + 2] += gravity_z * dt
            
            velocities[i*3] *= (1.0 - damping)
            velocities[i*3 + 1] *= (1.0 - damping)
            velocities[i*3 + 2] *= (1.0 - damping)
            
            particle.prev_x = particle.x
            particle.prev_y = particle.y
            particle.prev_z = particle.z
            
            particle.x += velocities[i*3] * dt
            particle.y += velocities[i*3 + 1] * dt
            particle.z += velocities[i*3 + 2] * dt

            particle.vx = velocities[i*3]
            particle.vy = velocities[i*3 + 1]
            particles[i].vz = velocities[i*3 + 2]
    
    free(velocities)

def update_positions_py(size_t particles_ptr, Py_ssize_t pcount, double dt, double damping):
    update_positions(<Particle*>particles_ptr, pcount, dt, damping)



cdef void solve_constraints(
    Particle* particles,
    Constraint* constraints,
    Py_ssize_t ccount,
    DTYPE_t stiffness,
    int iterations
) nogil:
    cdef:
        int iter, i
        Py_ssize_t v1, v2
        DTYPE_t rest_length, dx, dy, dz, dist, correction_factor
        DTYPE_t correction_x, correction_y, correction_z

    for iter in range(iterations):
        for i in range(ccount):
            v1 = constraints[i].v1
            v2 = constraints[i].v2
            rest_length = constraints[i].rest_length

            if not particles[v1].pinned or not particles[v2].pinned:
                dx = particles[v2].x - particles[v1].x
                dy = particles[v2].y - particles[v1].y
                dz = particles[v2].z - particles[v1].z
                dist = sqrt(dx*dx + dy*dy + dz*dz)
                
                correction_factor = (1 - rest_length / dist) * 0.5 * stiffness
                correction_x = dx * correction_factor
                correction_y = dy * correction_factor
                correction_z = dz * correction_factor

                if not particles[v1].pinned:
                    particles[v1].x += correction_x
                    particles[v1].y += correction_y
                    particles[v1].z += correction_z
                if not particles[v2].pinned:
                    particles[v2].x -= correction_x
                    particles[v2].y -= correction_y
                    particles[v2].z -= correction_z

def solve_constraints_py(
    size_t particles_ptr,
    size_t constraints_ptr,
    Py_ssize_t pcount,
    Py_ssize_t ccount,
    DTYPE_t stiffness,
    int iterations
):
    solve_constraints(<Particle*>particles_ptr, <Constraint*>constraints_ptr, ccount, stiffness, iterations)


cdef void update_velocities(
    Particle* particles,
    Py_ssize_t pcount,
    DTYPE_t dt
) nogil:
    cdef:
        int i
        Particle* particle

    for i in prange(pcount, nogil=True):
        particle = &particles[i]
        if not particle.pinned:
            particle.vx = (particle.x - particle.prev_x) / dt
            particle.vy = (particle.y - particle.prev_y) / dt
            particle.vz = (particle.z - particles[i].prev_z) / dt

def update_velocities_py(size_t particles_ptr, Py_ssize_t pcount, DTYPE_t dt):
    update_velocities(<Particle*>particles_ptr, pcount, dt)


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


# UTILTIES.
def get_positions_py(size_t particles_ptr, Py_ssize_t pcount):
    cdef:
        Particle* particles = <Particle*>particles_ptr
        cnp.ndarray[DTYPE_t, ndim=2] positions = np.empty((pcount, 3), dtype=DTYPE)
        Particle* particle

    for i in range(pcount):
        particle = &particles[i]
        positions[i, 0] = particle.x
        positions[i, 1] = particle.y
        positions[i, 2] = particle.z

    return positions

def get_velocities_py(size_t particles_ptr, Py_ssize_t pcount):
    cdef:
        Particle* particles = <Particle*>particles_ptr
        cnp.ndarray[DTYPE_t, ndim=2] velocities = np.empty((pcount, 3), dtype=DTYPE)
        Particle* particle

    for i in range(pcount):
        particle = &particles[i]
        velocities[i, 0] = particle.vx
        velocities[i, 1] = particle.vy
        velocities[i, 2] = particle.vz

    return velocities


cdef bint ray_triangle_intersection(
    DTYPE_t ray_origin_x, DTYPE_t ray_origin_y, DTYPE_t ray_origin_z,
    DTYPE_t ray_direction_x, DTYPE_t ray_direction_y, DTYPE_t ray_direction_z,
    DTYPE_t v0_x, DTYPE_t v0_y, DTYPE_t v0_z,
    DTYPE_t v1_x, DTYPE_t v1_y, DTYPE_t v1_z,
    DTYPE_t v2_x, DTYPE_t v2_y, DTYPE_t v2_z,
    DTYPE_t* intersection_x, DTYPE_t* intersection_y, DTYPE_t* intersection_z
) nogil:
    cdef:
        DTYPE_t epsilon = 1e-6
        DTYPE_t edge1_x = v1_x - v0_x, edge1_y = v1_y - v0_y, edge1_z = v1_z - v0_z
        DTYPE_t edge2_x = v2_x - v0_x, edge2_y = v2_y - v0_y, edge2_z = v2_z - v0_z
        DTYPE_t h_x, h_y, h_z, s_x, s_y, s_z, q_x, q_y, q_z
        DTYPE_t a, f, u, v, t

    h_x = ray_direction_y * edge2_z - ray_direction_z * edge2_y
    h_y = ray_direction_z * edge2_x - ray_direction_x * edge2_z
    h_z = ray_direction_x * edge2_y - ray_direction_y * edge2_x
    a = edge1_x * h_x + edge1_y * h_y + edge1_z * h_z

    if -epsilon < a < epsilon:
        return False

    f = 1.0 / a
    s_x = ray_origin_x - v0_x
    s_y = ray_origin_y - v0_y
    s_z = ray_origin_z - v0_z
    u = f * (s_x * h_x + s_y * h_y + s_z * h_z)

    if u < 0.0 or u > 1.0:
        return False

    q_x = s_y * edge1_z - s_z * edge1_y
    q_y = s_z * edge1_x - s_x * edge1_z
    q_z = s_x * edge1_y - s_y * edge1_x
    v = f * (ray_direction_x * q_x + ray_direction_y * q_y + ray_direction_z * q_z)

    if v < 0.0 or u + v > 1.0:
        return False

    t = f * (edge2_x * q_x + edge2_y * q_y + edge2_z * q_z)

    if t > epsilon:
        intersection_x[0] = ray_origin_x + t * ray_direction_x
        intersection_y[0] = ray_origin_y + t * ray_direction_y
        intersection_z[0] = ray_origin_z + t * ray_direction_z
        return True

    return False


# DRAGGING.

cdef struct DragData:
    bint dragging
    Py_ssize_t drag_vertex_index
    DTYPE_t drag_start_x, drag_start_y, drag_start_z
    DTYPE_t drag_current_x, drag_current_y, drag_current_z
    DTYPE_t drag_influence_radius
    DTYPE_t plane_normal_x, plane_normal_y, plane_normal_z
    DTYPE_t plane_point_x, plane_point_y, plane_point_z

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
