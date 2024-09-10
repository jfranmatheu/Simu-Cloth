################################################################################
# COLLISION.
################################################################################
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
