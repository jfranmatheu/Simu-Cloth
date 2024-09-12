from .types cimport Vector3, SimulationObject


cdef void transform_point(Vector3* point, float[4][4] matrix_world)

cdef void transform_normal(Vector3* normal, float[4][4] matrix_world)

cdef void get_world_coordinate(SimulationObject* sim_obj, int vertex_index, Vector3* result)

cdef void get_world_normal(SimulationObject* sim_obj, int normal_index, Vector3* result)