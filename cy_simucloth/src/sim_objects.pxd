from libc.stdint cimport uintptr_t
from .types cimport SimulationObject

cdef:
    SimulationObject** simulation_objects
    Py_ssize_t object_count

cdef SimulationObject** get_simulation_objects() nogil
cdef Py_ssize_t get_object_count() nogil

# Python-accessible functions
cpdef create_simulation_objects_py(sim_objects)
cpdef update_simulation_object_py(int index, sim_ob)
cpdef free_simulation_objects_py()