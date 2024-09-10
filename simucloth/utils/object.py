import bpy
from mathutils import Vector
from ctypes import Structure, POINTER, c_float, cast, addressof, c_int32

class Vector3(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("z", c_float)]

class IVector3(Structure):
    _fields_ = [("x", c_int32),
                ("y", c_int32),
                ("z", c_int32)]


def get_local_axis_direction(obj: bpy.types.Object, axis: str):
    """ Gets object direction in given local axis.
        - `obj`: bpy.types.Object
        - `axis`: string value in {'X', '-X', 'Y', '-Y', 'Z', '-Z'}
    """
    # Ensure the input is valid
    valid_inputs = {'X', '-X', 'Y', '-Y', 'Z', '-Z'}
    if axis not in valid_inputs:
        raise ValueError(f"Invalid axis. Must be one of {valid_inputs}")

    # Get the object's local-to-world matrix
    local_to_world = obj.matrix_world

    # Determine the axis index and sign
    axis_index = 'XYZ'.index(axis[-1])
    sign = -1 if axis.startswith('-') else 1

    # Extract the axis direction from the matrix
    axis_vector = local_to_world.to_3x3().col[axis_index].normalized()

    # Apply the sign
    return axis_vector * sign


def get_min_max_bound_box(obj: bpy.types.Object):
    # Ensure the bounding box is up to date
    # obj.update_from_editmode()
    # obj.matrix_world.identity()  # Reset to identity matrix

    # Initialize min and max vectors
    min_point = Vector((float('inf'),) * 3)
    max_point = Vector((float('-inf'),) * 3)

    # Iterate through all points in the bounding box
    for point in obj.bound_box:
        for i in range(3):  # For each x, y, z
            min_point[i] = min(min_point[i], point[i])
            max_point[i] = max(max_point[i], point[i])

    return min_point, max_point


def get_mesh_vert_positions_ptr(mesh: bpy.types.Mesh):
    return mesh.attributes['position'].data[0].as_pointer()
    vert_count = len(mesh.vertices)
    return (Vector3 * vert_count).from_address(p_position)

def get_mesh_vert_normals_ptr(mesh: bpy.types.Mesh):
    return mesh.vertex_normals[0].as_pointer()
    vert_count = len(mesh.vertices)
    return (Vector3 * vert_count).from_address(p_position)

def get_mesh_loop_triangles_ptr(mesh: bpy.types.Mesh):
    return mesh.loop_triangles[0].as_pointer()
    vert_count = len(mesh.polygons)
    return (IVector3 * vert_count).from_address(p_position)

def get_mesh_polygon_normals_ptr(mesh: bpy.types.Mesh):
    return mesh.polygon_normals[0].as_pointer()
    vert_count = len(mesh.polygons)
    return (Vector3 * vert_count).from_address(p_position)

def get_mesh_corner_vert_ptr(mesh: bpy.types.Mesh):
    return mesh.attributes['.corner_vert'].data[0].as_pointer()
    vert_count = len(mesh.polygons)
    return (Vector3 * vert_count).from_address(p_position)

def get_matrix_world_ptr(ob: bpy.types.Object):
    # Convert the matrix to a ctypes array
    matrix_data = (c_float * 16)(*[v for row in ob.matrix_world for v in row])
    return cast(matrix_data, POINTER(c_float)).contents.address

def get_min_max_bound_box_ptr(ob: bpy.types.Object):
    min_point, max_point = get_min_max_bound_box(ob)
    return (
        (c_float * 3)(min_point.x, min_point.y, min_point.z),
        (c_float * 3)(max_point.x, max_point.y, max_point.z)
    )
