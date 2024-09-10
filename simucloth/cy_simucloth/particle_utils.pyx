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

from .types cimport Particle

cnp.import_array()

DTYPE = np.float32
ctypedef cnp.float32_t DTYPE_t


def get_positions_py(size_t particles_ptr, Py_ssize_t pcount):
    cdef:
        Particle* particles = <Particle*>particles_ptr
        cnp.ndarray[DTYPE_t, ndim=2] positions = np.empty((pcount, 3), dtype=DTYPE)
        Particle* particle

    for i in range(pcount):
        particle = &particles[i]
        positions[i, 0] = particle.pos[0]
        positions[i, 1] = particle.pos[1]
        positions[i, 2] = particle.pos[2]

    return positions

def get_velocities_py(size_t particles_ptr, Py_ssize_t pcount):
    cdef:
        Particle* particles = <Particle*>particles_ptr
        cnp.ndarray[DTYPE_t, ndim=2] velocities = np.empty((pcount, 3), dtype=DTYPE)
        Particle* particle

    for i in range(pcount):
        particle = &particles[i]
        velocities[i, 0] = particle.vel[0]
        velocities[i, 1] = particle.vel[1]
        velocities[i, 2] = particle.vel[2]

    return velocities
