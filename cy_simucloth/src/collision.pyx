################################################################################
# COLLISION.
################################################################################

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: initializedcheck=False

from libc.stdlib cimport malloc, free
from libc.math cimport sqrt, fabs

from .types cimport SimulationObject, Particle, Vector3, ObjectType
from .sim_objects cimport get_simulation_objects, get_object_count

cdef extern from "math.h":
    float INFINITY

cdef struct Triangle:
    Vector3 v1
    Vector3 v2
    Vector3 v3

cdef float point_triangle_distance(Vector3 p, Triangle triangle, Vector3* closest_point, Vector3* normal) nogil:
    cdef:
        Vector3 edge0, edge1, v0
        float a, b, c, d, e, f, det, s, t, inv_det
        Vector3 tmp

    edge0.x = triangle.v2.x - triangle.v1.x
    edge0.y = triangle.v2.y - triangle.v1.y
    edge0.z = triangle.v2.z - triangle.v1.z

    edge1.x = triangle.v3.x - triangle.v1.x
    edge1.y = triangle.v3.y - triangle.v1.y
    edge1.z = triangle.v3.z - triangle.v1.z

    v0.x = triangle.v1.x - p.x
    v0.y = triangle.v1.y - p.y
    v0.z = triangle.v1.z - p.z

    a = edge0.x * edge0.x + edge0.y * edge0.y + edge0.z * edge0.z
    b = edge0.x * edge1.x + edge0.y * edge1.y + edge0.z * edge1.z
    c = edge1.x * edge1.x + edge1.y * edge1.y + edge1.z * edge1.z
    d = edge0.x * v0.x + edge0.y * v0.y + edge0.z * v0.z
    e = edge1.x * v0.x + edge1.y * v0.y + edge1.z * v0.z

    det = a * c - b * b
    s = b * e - c * d
    t = b * d - a * e

    if s + t <= det:
        if s < 0:
            if t < 0:
                if d < 0:
                    s = -d / a if -d / a <= 1 else 1
                    t = 0
                else:
                    s = 0
                    t = -e / c if -e / c <= 1 else 1
            else:
                s = 0
                t = -e / c if -e / c <= 1 else 1
        elif t < 0:
            s = -d / a if -d / a <= 1 else 1
            t = 0
        else:
            inv_det = 1.0 / det
            s *= inv_det
            t *= inv_det
    else:
        if s < 0:
            tmp.x = b + d
            tmp.y = c + e
            if tmp.y > tmp.x:
                s = -tmp.x / a if -tmp.x / a >= 0 and -tmp.x / a <= 1 else 1
                t = 0
            else:
                t = -tmp.y / c if -tmp.y / c >= 0 and -tmp.y / c <= 1 else 1
                s = 0
        elif t < 0:
            tmp.x = b + e
            tmp.y = a + d
            if tmp.y > tmp.x:
                s = -tmp.x / c if -tmp.x / c >= 0 and -tmp.x / c <= 1 else 1
                t = 0
            else:
                s = -tmp.y / a if -tmp.y / a >= 0 and -tmp.y / a <= 1 else 1
                t = 0
        else:
            inv_det = 1.0 / det
            s = (b * e - c * d) * inv_det
            t = (b * d - a * e) * inv_det

    closest_point.x = triangle.v1.x + edge0.x * s + edge1.x * t
    closest_point.y = triangle.v1.y + edge0.y * s + edge1.y * t
    closest_point.z = triangle.v1.z + edge0.z * s + edge1.z * t

    normal.x = edge0.y * edge1.z - edge0.z * edge1.y
    normal.y = edge0.z * edge1.x - edge0.x * edge1.z
    normal.z = edge0.x * edge1.y - edge0.y * edge1.x

    # Normalize the normal vector
    normal_length = sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z)
    if normal_length > 0:
        normal.x /= normal_length
        normal.y /= normal_length
        normal.z /= normal_length

    tmp.x = p.x - closest_point.x
    tmp.y = p.y - closest_point.y
    tmp.z = p.z - closest_point.z

    return sqrt(tmp.x * tmp.x + tmp.y * tmp.y + tmp.z * tmp.z)

cdef void handle_collisions(float friction) nogil:
    cdef:
        SimulationObject** simulation_objects = get_simulation_objects()
        Py_ssize_t object_count = get_object_count()
        SimulationObject* cloth_obj
        SimulationObject* collider_obj
        Particle* particle
        Triangle triangle
        Vector3 closest_point, normal, correction, friction_force, tangent
        float distance, normal_velocity, tangent_velocity
        int i, j, k, l
        bint collision_occurred

    for i in range(object_count):
        cloth_obj = simulation_objects[i]
        if cloth_obj.type != ObjectType.CLOTH:
            continue

        for j in range(cloth_obj.tot_particles):
            particle = &cloth_obj.particles[j]
            collision_occurred = False

            for k in range(object_count):
                collider_obj = simulation_objects[k]
                if collider_obj.type != ObjectType.COLLIDER:
                    continue

                # Bounding box check
                if (particle.pos.x < collider_obj.bbox_min.x or
                    particle.pos.x > collider_obj.bbox_max.x or
                    particle.pos.y < collider_obj.bbox_min.y or
                    particle.pos.y > collider_obj.bbox_max.y or
                    particle.pos.z < collider_obj.bbox_min.z or
                    particle.pos.z > collider_obj.bbox_max.z):
                    continue

                for l in range(collider_obj.mesh.tot_tris):
                    triangle.v1 = collider_obj.mesh.vert_positions[collider_obj.mesh.loop_tri[l].i1]
                    triangle.v2 = collider_obj.mesh.vert_positions[collider_obj.mesh.loop_tri[l].i2]
                    triangle.v3 = collider_obj.mesh.vert_positions[collider_obj.mesh.loop_tri[l].i3]

                    distance = point_triangle_distance(particle.pos, triangle, &closest_point, &normal)

                    if distance < 0.01:  # Collision threshold
                        collision_occurred = True
                        # Collision response
                        correction.x = normal.x * (0.01 - distance)
                        correction.y = normal.y * (0.01 - distance)
                        correction.z = normal.z * (0.01 - distance)

                        particle.pos.x += correction.x
                        particle.pos.y += correction.y
                        particle.pos.z += correction.z

                        # Friction
                        normal_velocity = (
                            particle.vel.x * normal.x +
                            particle.vel.y * normal.y +
                            particle.vel.z * normal.z
                        )

                        tangent.x = particle.vel.x - normal_velocity * normal.x
                        tangent.y = particle.vel.y - normal_velocity * normal.y
                        tangent.z = particle.vel.z - normal_velocity * normal.z

                        tangent_velocity = sqrt(
                            tangent.x * tangent.x +
                            tangent.y * tangent.y +
                            tangent.z * tangent.z
                        )

                        if tangent_velocity > 1e-6:
                            tangent.x /= tangent_velocity
                            tangent.y /= tangent_velocity
                            tangent.z /= tangent_velocity

                            friction_force.x = -friction * fabs(normal_velocity) * tangent.x
                            friction_force.y = -friction * fabs(normal_velocity) * tangent.y
                            friction_force.z = -friction * fabs(normal_velocity) * tangent.z

                            particle.vel.x += friction_force.x
                            particle.vel.y += friction_force.y
                            particle.vel.z += friction_force.z

                        break  # Exit the triangle loop if collision occurred

                if collision_occurred:
                    break  # Exit the object loop if collision occurred

def handle_collisions_py(float friction):
    handle_collisions(friction)

'''
cdef void handle_collisions(
    Particle* particles,
    Py_ssize_t pcount,
    CollisionObject** collision_objects,
    Py_ssize_t object_count,
    Vector3 cloth_min_point,
    Vector3 cloth_max_point,
    DTYPE_t friction
):
    cdef:
        Py_ssize_t i, j, k
        DTYPE_t* closest_point = <DTYPE_t*>malloc(3 * sizeof(DTYPE_t))
        DTYPE_t* normal = <DTYPE_t*>malloc(3 * sizeof(DTYPE_t))
        Vector3 correction, friction_force, tangent
        DTYPE_t distance, normal_velocity, tangent_velocity
        bint collision_occurred

    if closest_point == NULL or normal == NULL:
        raise MemoryError("Failed to allocate memory for collision detection")

    for i in range(pcount):
        for j in range(object_count):
            # Bounding box check
            if not (particles[i].x >= collision_objects[j].min_point.x and
                    particles[i].x <= collision_objects[j].max_point.x and
                    particles[i].y >= collision_objects[j].min_point.y and
                    particles[i].y <= collision_objects[j].max_point.y and
                    particles[i].z >= collision_objects[j].min_point.z and
                    particles[i].z <= collision_objects[j].max_point.z):
                continue

            collision_occurred = False

            for k in range(collision_objects[j].tot_tris):
                # NOTE: To get real triangle vertices (indices), do the following:
                # v1 index: corner_vert[loop_tri[i].i1]
                # v2 index: corner_vert[loop_tri[i].i2]
                # v3 index: corner_vert[loop_tri[i].i3]

                distance = point_triangle_distance(
                    Vector3(particles[i].x, particles[i].y, particles[i].z),
                    collision_objects[j].vert_positions[collision_objects[j].corner_vert[k*3].i1],
                    collision_objects[j].vert_positions[collision_objects[j].corner_vert[k*3+1].i2],
                    collision_objects[j].vert_positions[collision_objects[j].corner_vert[k*3+2].i3],
                    closest_point,
                    normal
                )

                if distance < 0.01:  # Collision threshold
                    collision_occurred = True
                    # Collision response
                    correction.x = normal[0] * (0.01 - distance)
                    correction.y = normal[1] * (0.01 - distance)
                    correction.z = normal[2] * (0.01 - distance)

                    particles[i].x += correction.x
                    particles[i].y += correction.y
                    particles[i].z += correction.z

                    # Friction
                    normal_velocity = (
                        particles[i].vx * normal[0] +
                        particles[i].vy * normal[1] +
                        particles[i].vz * normal[2]
                    )

                    tangent.x = particles[i].vx - normal_velocity * normal[0]
                    tangent.y = particles[i].vy - normal_velocity * normal[1]
                    tangent.z = particles[i].vz - normal_velocity * normal[2]

                    tangent_velocity = sqrt(
                        tangent.x * tangent.x +
                        tangent.y * tangent.y +
                        tangent.z * tangent.z
                    )

                    if tangent_velocity > 1e-6:
                        tangent.x /= tangent_velocity
                        tangent.y /= tangent_velocity
                        tangent.z /= tangent_velocity

                        friction_force.x = -friction * fabs(normal_velocity) * tangent.x
                        friction_force.y = -friction * fabs(normal_velocity) * tangent.y
                        friction_force.z = -friction * fabs(normal_velocity) * tangent.z

                        particles[i].vx += friction_force.x
                        particles[i].vy += friction_force.y
                        particles[i].vz += friction_force.z

                    break  # Exit the triangle loop if collision occurred

            if collision_occurred:
                break  # Exit the object loop if collision occurred
    
    free(closest_point)
    free(normal)


def handle_collisions_py(
    size_t particles_ptr,
    Py_ssize_t pcount,
    list collision_object_data,
    DTYPE_t friction,
    list cloth_min_point,
    list cloth_max_point
):
    cdef:
        CollisionObject** collision_objects
        Py_ssize_t object_count = len(collision_object_data)
        Py_ssize_t i
        DTYPE_t[16] c_matrix_world
        Vector3 c_min_point, c_max_point, c_cloth_min_point, c_cloth_max_point

    collision_objects = <CollisionObject**>malloc(object_count * sizeof(CollisionObject*))
    if collision_objects == NULL:
        raise MemoryError("Failed to allocate memory for collision objects")
    
    # Create Vector3 structs for min_point and max_point
    c_cloth_min_point.x, c_cloth_min_point.y, c_cloth_min_point.z = cloth_min_point
    c_cloth_max_point.x, c_cloth_max_point.y, c_cloth_max_point.z = cloth_max_point

    try:
        for i in range(object_count):
            data = collision_object_data[i]
            
            # Convert matrix_world to C array
            for j in range(16):
                c_matrix_world[j] = data['matrix_world'][j]
            
            # Create Vector3 structs for min_point and max_point
            c_min_point.x, c_min_point.y, c_min_point.z = data['min_point']
            c_max_point.x, c_max_point.y, c_max_point.z = data['max_point']

            collision_objects[i] = create_collision_object(
                data['loop_tri_ptr'],
                data['corner_vert_ptr'],
                data['tri_normals_ptr'],
                data['vert_positions_ptr'],
                data['vert_normals_ptr'],
                data['tot_tris'],
                data['tot_verts'],
                c_matrix_world,
                c_min_point,
                c_max_point
            )

        handle_collisions(
            <Particle*>particles_ptr,
            pcount,
            collision_objects,
            object_count,
            c_min_point,  # Use cloth's min_point here
            c_max_point,  # Use cloth's max_point here
            friction
        )

    finally:
        for i in range(object_count):
            if collision_objects[i] != NULL:
                free(collision_objects[i])
        free(collision_objects)
'''
