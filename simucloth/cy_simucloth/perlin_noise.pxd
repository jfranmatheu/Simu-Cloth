cdef void init_perlin() nogil

cdef double noise(double x, double y, double z) nogil

cdef double fade(double t) nogil

cdef double lerp(double t, double a, double b) nogil

cdef double grad(int hash, double x, double y, double z) nogil

cpdef void init_perlin_py()

cpdef void free_perlin_py()