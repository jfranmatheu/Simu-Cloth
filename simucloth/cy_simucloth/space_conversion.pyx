from libc.math cimport sqrt

from .types cimport Vector3, SimulationObject


cdef void transform_point(Vector3* point, float[4][4] matrix_world):
    cdef float x = point.x
    cdef float y = point.y
    cdef float z = point.z
    point.x = matrix_world[0][0] * x + matrix_world[0][1] * y + matrix_world[0][2] * z + matrix_world[0][3]
    point.y = matrix_world[1][0] * x + matrix_world[1][1] * y + matrix_world[1][2] * z + matrix_world[1][3]
    point.z = matrix_world[2][0] * x + matrix_world[2][1] * y + matrix_world[2][2] * z + matrix_world[2][3]

cdef void transform_normal(Vector3* normal, float[4][4] matrix_world):
    cdef float x = normal.x
    cdef float y = normal.y
    cdef float z = normal.z
    normal.x = matrix_world[0][0] * x + matrix_world[0][1] * y + matrix_world[0][2] * z
    normal.y = matrix_world[1][0] * x + matrix_world[1][1] * y + matrix_world[1][2] * z
    normal.z = matrix_world[2][0] * x + matrix_world[2][1] * y + matrix_world[2][2] * z
    
    # Normalize the transformed normal
    cdef float length = sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z)
    if length > 0:
        normal.x /= length
        normal.y /= length
        normal.z /= length

cdef void get_world_coordinate(SimulationObject* sim_obj, int vertex_index, Vector3* result):
    if vertex_index < 0 or vertex_index >= sim_obj.mesh.tot_verts:
        result.x = result.y = result.z = 0
        return
    
    result[0] = sim_obj.mesh.vert_positions[vertex_index]
    transform_point(result, sim_obj.matrix_world)

cdef void get_world_normal(SimulationObject* sim_obj, int normal_index, Vector3* result):
    if normal_index < 0 or normal_index >= sim_obj.mesh.tot_verts:
        result.x = result.y = result.z = 0
        return
    
    result[0] = sim_obj.mesh.vert_normals[normal_index]
    transform_normal(result, sim_obj.matrix_world)
