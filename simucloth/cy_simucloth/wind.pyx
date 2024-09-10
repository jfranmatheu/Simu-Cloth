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

from .types cimport Particle
from .perlin_noise cimport noise

cnp.import_array()

DTYPE = np.float32
ctypedef cnp.float32_t DTYPE_t


################################################################################
# WIND FORCE.
################################################################################

cdef void apply_wind_force(
    Particle* particles,
    Py_ssize_t pcount,
    DTYPE_t wind_force_x,
    DTYPE_t wind_force_y,
    DTYPE_t wind_force_z,
    DTYPE_t turbulence,
    DTYPE_t time,
    DTYPE_t dt
) nogil:
    cdef:
        Py_ssize_t i
        DTYPE_t force_x, force_y, force_z
        DTYPE_t noise_x, noise_y, noise_z
        DTYPE_t scale = 0.1  # Scale factor for Perlin noise
        DTYPE_t time_scale = 0.5  # Time scale for wind variation

    for i in prange(pcount, nogil=True):
        if not particles[i].pinned:
            # Base wind force
            force_x = wind_force_x
            force_y = wind_force_y
            force_z = wind_force_z

            # Add Perlin noise-based turbulence
            noise_x = noise(particles[i].x * scale, particles[i].y * scale, time * time_scale) * 2 - 1
            noise_y = noise(particles[i].y * scale, particles[i].z * scale, time * time_scale) * 2 - 1
            noise_z = noise(particles[i].z * scale, particles[i].x * scale, time * time_scale) * 2 - 1

            force_x += noise_x * turbulence
            force_y += noise_y * turbulence
            force_z += noise_z * turbulence

            # Apply force to velocity
            particles[i].vx += force_x * dt
            particles[i].vy += force_y * dt
            particles[i].vz += force_z * dt

# Update the Python wrapper
def apply_wind_force_py(
    size_t particles_ptr,
    Py_ssize_t pcount,
    cnp.ndarray[DTYPE_t, ndim=1] wind_force,
    DTYPE_t turbulence,
    DTYPE_t time,
    DTYPE_t dt
):
    apply_wind_force(
        <Particle*>particles_ptr,
        pcount,
        wind_force[0],
        wind_force[1],
        wind_force[2],
        turbulence,
        time,
        dt
    )

