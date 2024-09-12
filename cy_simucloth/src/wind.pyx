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

from .types cimport Particle, SimulationObject, ObjectType
from .perlin_noise cimport noise
from .sim_objects cimport get_simulation_objects, get_object_count

cnp.import_array()

DTYPE = np.float32
ctypedef cnp.float32_t DTYPE_t


################################################################################
# WIND FORCE.
################################################################################

cdef void apply_wind_force(
    DTYPE_t wind_force_x,
    DTYPE_t wind_force_y,
    DTYPE_t wind_force_z,
    DTYPE_t turbulence,
    DTYPE_t time,
    DTYPE_t dt
) nogil:
    cdef:
        SimulationObject** simulation_objects = get_simulation_objects()
        Py_ssize_t object_count = get_object_count()
        SimulationObject* sim_obj
        Particle* particle
        Py_ssize_t i
        DTYPE_t force_x, force_y, force_z
        DTYPE_t noise_x, noise_y, noise_z
        DTYPE_t scale = 0.1  # Scale factor for Perlin noise
        DTYPE_t time_scale = 0.5  # Time scale for wind variation

    for i in range(object_count):
        sim_obj = simulation_objects[i]
        if sim_obj.type != ObjectType.CLOTH:
            continue
        for j in range(sim_obj.tot_particles):
            particle = &sim_obj.particles[j]
            if not particle.pinned:
                # Base wind force
                force_x = wind_force_x
                force_y = wind_force_y
                force_z = wind_force_z

                # Add Perlin noise-based turbulence
                noise_x = noise(particle.pos.x * scale, particle.pos.y * scale, time * time_scale) * 2 - 1
                noise_y = noise(particle.pos.y * scale, particle.pos.z * scale, time * time_scale) * 2 - 1
                noise_z = noise(particle.pos.z * scale, particle.pos.x * scale, time * time_scale) * 2 - 1

                force_x += noise_x * turbulence
                force_y += noise_y * turbulence
                force_z += noise_z * turbulence

                # Apply force to velocity
                particle.vel.x += force_x * dt
                particle.vel.y += force_y * dt
                particle.vel.z += force_z * dt

# Update the Python wrapper
def apply_wind_force_py(
    cnp.ndarray[DTYPE_t, ndim=1] wind_force,
    DTYPE_t turbulence,
    DTYPE_t time,
    DTYPE_t dt
):
    apply_wind_force(
        wind_force[0],
        wind_force[1],
        wind_force[2],
        turbulence,
        time,
        dt
    )
