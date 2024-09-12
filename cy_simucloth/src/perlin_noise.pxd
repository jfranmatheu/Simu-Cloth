cdef void init_perlin() nogil

cdef float noise(float x, float y, float z) nogil

cdef float fade(float t) nogil

cdef float lerp(float t, float a, float b) nogil

cdef float grad(int hash, float x, float y, float z) nogil

cpdef void init_perlin_py()

cpdef void free_perlin_py()