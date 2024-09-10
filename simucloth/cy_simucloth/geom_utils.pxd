from .types cimport Vector3

cdef bint ray_triangle_intersection(
    float ray_origin_x, float ray_origin_y, float ray_origin_z,
    float ray_direction_x, float ray_direction_y, float ray_direction_z,
    float v0_x, float v0_y, float v0_z,
    float v1_x, float v1_y, float v1_z,
    float v2_x, float v2_y, float v2_z,
    float* intersection_x, float* intersection_y, float* intersection_z
) nogil

cdef float point_triangle_distance(
    Vector3 p,
    Vector3 v0,
    Vector3 v1,
    Vector3 v2,
    float* closest_point,
    float* normal
) nogil