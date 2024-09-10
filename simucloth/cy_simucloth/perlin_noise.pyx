# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: embedsignature=True
# cython: binding=False


from cython.parallel import prange
from libc.stdlib cimport malloc, free, rand
from libc.math cimport floor


##############################################################################
# PERLIN NOISE
###########################################################################

# Define constants for Perlin noise
cdef int PERLIN_YWRAPB = 4
cdef int PERLIN_YWRAP = 1 << PERLIN_YWRAPB
cdef int PERLIN_ZWRAPB = 8
cdef int PERLIN_ZWRAP = 1 << PERLIN_ZWRAPB
cdef int PERLIN_SIZE = 4095

cdef int* perlin

# Initialize Perlin noise
cdef void init_perlin() nogil:
    global perlin
    cdef int i
    perlin = <int*>malloc(PERLIN_SIZE * sizeof(int))
    for i in prange(PERLIN_SIZE, nogil=True):
        perlin[i] = <int>(rand() % 256)

# Perlin noise function
cdef double noise(double x, double y, double z) nogil:
    cdef int X = <int>floor(x) & 255
    cdef int Y = <int>floor(y) & 255
    cdef int Z = <int>floor(z) & 255
    x -= floor(x)
    y -= floor(y)
    z -= floor(z)
    cdef double u = fade(x)
    cdef double v = fade(y)
    cdef double w = fade(z)
    cdef int A = perlin[X] + Y
    cdef int AA = perlin[A & 255] + Z
    cdef int AB = perlin[(A + 1) & 255] + Z
    cdef int B = perlin[(X + 1) & 255] + Y
    cdef int BA = perlin[B & 255] + Z
    cdef int BB = perlin[(B + 1) & 255] + Z
    return lerp(w, lerp(v, lerp(u, grad(perlin[AA & 255], x, y, z),
                                   grad(perlin[BA & 255], x - 1, y, z)),
                           lerp(u, grad(perlin[AB & 255], x, y - 1, z),
                                   grad(perlin[BB & 255], x - 1, y - 1, z))),
                   lerp(v, lerp(u, grad(perlin[(AA + 1) & 255], x, y, z - 1),
                                   grad(perlin[(BA + 1) & 255], x - 1, y, z - 1)),
                           lerp(u, grad(perlin[(AB + 1) & 255], x, y - 1, z - 1),
                                   grad(perlin[(BB + 1) & 255], x - 1, y - 1, z - 1))))

cdef double fade(double t) nogil:
    return t * t * t * (t * (t * 6 - 15) + 10)

cdef double lerp(double t, double a, double b) nogil:
    return a + t * (b - a)

cdef double grad(int hash, double x, double y, double z) nogil:
    cdef int h = hash & 15
    cdef double u = x if h < 8 else y
    cdef double v = y if h < 4 else (x if h == 12 or h == 14 else z)
    return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)

# Don't forget to update these Python wrapper functions
def init_perlin_py():
    init_perlin()

def free_perlin_py():
    global perlin
    if perlin != NULL:
        free(perlin)
        perlin = NULL
