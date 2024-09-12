# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: embedsignature=True
# cython: binding=False

from libc.math cimport sqrt

from .types cimport Vector3


cdef bint ray_triangle_intersection(
    float ray_origin_x, float ray_origin_y, float ray_origin_z,
    float ray_direction_x, float ray_direction_y, float ray_direction_z,
    float v0_x, float v0_y, float v0_z,
    float v1_x, float v1_y, float v1_z,
    float v2_x, float v2_y, float v2_z,
    float* intersection_x, float* intersection_y, float* intersection_z
) nogil:
    cdef:
        float epsilon = 1e-6
        float edge1_x = v1_x - v0_x, edge1_y = v1_y - v0_y, edge1_z = v1_z - v0_z
        float edge2_x = v2_x - v0_x, edge2_y = v2_y - v0_y, edge2_z = v2_z - v0_z
        float h_x, h_y, h_z, s_x, s_y, s_z, q_x, q_y, q_z
        float a, f, u, v, t

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


cdef float point_triangle_distance(
    Vector3 p,
    Vector3 v0,
    Vector3 v1,
    Vector3 v2,
    float* closest_point,
    float* normal
) nogil:
    cdef:
        Vector3 edge1, edge2, v0p
        float dot00, dot01, dot02, dot11, dot12
        float inv_denom, u, v
        float length
    
    edge1.x = v1.x - v0.x
    edge1.y = v1.y - v0.y
    edge1.z = v1.z - v0.z
    
    edge2.x = v2.x - v0.x
    edge2.y = v2.y - v0.y
    edge2.z = v2.z - v0.z
    
    v0p.x = p.x - v0.x
    v0p.y = p.y - v0.y
    v0p.z = p.z - v0.z
    
    dot00 = edge1.x*edge1.x + edge1.y*edge1.y + edge1.z*edge1.z
    dot01 = edge1.x*edge2.x + edge1.y*edge2.y + edge1.z*edge2.z
    dot02 = edge1.x*v0p.x + edge1.y*v0p.y + edge1.z*v0p.z
    dot11 = edge2.x*edge2.x + edge2.y*edge2.y + edge2.z*edge2.z
    dot12 = edge2.x*v0p.x + edge2.y*v0p.y + edge2.z*v0p.z
    
    inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    
    if u >= 0 and v >= 0 and u + v <= 1:
        # Point is inside the triangle
        closest_point[0] = v0.x + u*edge1.x + v*edge2.x
        closest_point[1] = v0.y + u*edge1.y + v*edge2.y
        closest_point[2] = v0.z + u*edge1.z + v*edge2.z
        
        # Calculate normal
        normal[0] = edge1.y*edge2.z - edge1.z*edge2.y
        normal[1] = edge1.z*edge2.x - edge1.x*edge2.z
        normal[2] = edge1.x*edge2.y - edge1.y*edge2.x
        
        # Normalize normal
        length = sqrt(normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2])
        if length > 0:
            normal[0] /= length
            normal[1] /= length
            normal[2] /= length
        
        return sqrt(
            (p.x - closest_point[0])*(p.x - closest_point[0]) +
            (p.y - closest_point[1])*(p.y - closest_point[1]) +
            (p.z - closest_point[2])*(p.z - closest_point[2])
        )
    else:
        # Point is outside the triangle, you might want to implement edge and vertex cases here
        return 1e6  # Return a large distance
