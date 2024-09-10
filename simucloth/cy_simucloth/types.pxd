from libcpp cimport bool

ctypedef enum ObjectType:
    COLLIDER = 0
    CLOTH = 1
    FLUID = 2
    SOLID = 3

ctypedef enum ConstraintType:
    SPRING = 0
    BENDING = 1
    SHEARING = 2
    PIN = 3

cdef struct Vector3:
    float x
    float y
    float z

cdef struct Triangle:
    int i1
    int i2
    int i3

cdef struct DragData:
    bool dragging
    int drag_vertex_index
    Vector3 drag_start_pos
    Vector3 drag_current_pos
    float drag_influence_radius
    Vector3 plane_normal
    Vector3 plane_point

cdef struct Particle:
    Vector3 pos
    Vector3 prev_pos
    Vector3 vel
    bool pinned

cdef struct Constraint:
    ConstraintType type
    int v1
    int v2
    float rest_length
    float distance

cdef struct Mesh:
    Triangle* loop_tri
    Triangle* corner_vert
    Vector3* tri_normals
    Vector3* vert_positions
    Vector3* vert_normals
    int tot_tris
    int tot_verts

cdef struct SimulationObject:
    ObjectType type
    Mesh mesh
    Particle* particles
    Constraint* constraints
    int tot_particles
    int tot_constraints
    Vector3 location
    Vector3 rotation
    Vector3 scale
    float[4][4] matrix_world
    Vector3 bbox_min
    Vector3 bbox_max
