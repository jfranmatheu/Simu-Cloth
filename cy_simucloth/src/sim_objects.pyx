# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: embedsignature=True
# cython: binding=False

import logging
import ctypes

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from libc.stdlib cimport malloc, free
from libc.stdint cimport uintptr_t
from .types cimport SimulationObject

cdef SimulationObject** simulation_objects
cdef Py_ssize_t object_count

cdef SimulationObject** get_simulation_objects() nogil:
    return simulation_objects

cdef Py_ssize_t get_object_count() nogil:
    return object_count

cpdef create_simulation_objects_py(sim_objects):
    logger.info(f"Creating simulation objects, count: {len(sim_objects)}")
    global simulation_objects, object_count
    object_count = len(sim_objects)
    simulation_objects = <SimulationObject**>malloc(object_count * sizeof(SimulationObject*))
    if simulation_objects == NULL:
        logger.error("Failed to allocate memory for simulation objects")
        raise MemoryError("Failed to allocate memory for simulation objects")
    for i, sim_ob in enumerate(sim_objects):
        simulation_objects[i] = <SimulationObject*><uintptr_t>ctypes.addressof(sim_ob)
        logger.info(f"Object {i} created with {simulation_objects[i].tot_particles} particles")
    logger.info("Simulation objects created")

cpdef update_simulation_object_py(int index, sim_ob):
    logger.info(f"Updating specific simulation object")
    simulation_objects[index] = <SimulationObject*><uintptr_t>ctypes.addressof(sim_ob)
    logger.info("Simulation object updated")

cpdef free_simulation_objects_py():
    logger.info("Freeing simulation objects")
    global simulation_objects, object_count
    if simulation_objects != NULL:
        free(simulation_objects)
        simulation_objects = NULL
        object_count = 0
    logger.info("Simulation objects freed")