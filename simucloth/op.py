import os
import sys
import numpy as np

import bpy
import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Vector
from bpy_extras import view3d_utils

from .shaders import mesh_vertex_shader, mesh_fragment_shader
from .utils.time_it import time_it

# Add the directory containing the simucloth folder to Python's path
# addon_dir = os.path.dirname(os.path.abspath(__file__))
# if addon_dir not in sys.path:
#     sys.path.append(addon_dir)

try:
    from .cy_simucloth.cloth_simulation import (
        create_particles_py,
        create_constraints_py,
        free_particles_py,
        free_constraints_py,
        update_positions_py,
        solve_constraints_py,
        update_velocities_py,
        apply_drag_py,
        get_positions_py,
        get_velocities_py,
        create_drag_data_py,
        free_drag_data_py,
        start_drag_py,
        update_drag_py,
        end_drag_py,
        create_triangles_py,
        free_triangles_py
    )
    print("cloth_simulation imported successfully")
except ImportError as e:
    print("Failed to import cloth_simulation:", str(e))
    import traceback
    traceback.print_exc()


class SimpleClothOperator(bpy.types.Operator):
    bl_idname = "object.simple_cloth_operator"
    bl_label = "Simple Cloth Simulation"
    
    _timer = None
    _handler = None

    def __init__(self):
        self.mesh = None
        self.particles_ptr = None
        self.constraints_ptr = None
        self.pcount = 0
        self.ccount = 0
        self.pinned_vertices = None
        self.point_shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        self.mesh_shader = gpu.types.GPUShader(mesh_vertex_shader, mesh_fragment_shader)
        self.point_batch = None
        self.mesh_batch = None
        self.dragging = False
        self.drag_data_ptr = None
        self._drag_influence_radius = 1.0  # Adjust this value as needed
        self.triangles_ptr = None
        self.tcount = 0

    def __del__(self):
        if hasattr(self, 'particles_ptr') and self.particles_ptr is not None:
            free_particles_py(self.particles_ptr)
        if hasattr(self, 'constraints_ptr') and self.constraints_ptr is not None:
            free_constraints_py(self.constraints_ptr)
        if hasattr(self, 'drag_data_ptr') and self.drag_data_ptr is not None:
            free_drag_data_py(self.drag_data_ptr)
        if hasattr(self, 'triangles_ptr') and self.triangles_ptr is not None:
            free_triangles_py(self.triangles_ptr)

    def execute(self, context):
        if context.active_object and context.active_object.type == 'MESH':
            self.drag_data_ptr = create_drag_data_py(self._drag_influence_radius)
            self.mesh = context.active_object.data
            self.initialize_simulation(context)
            wm = context.window_manager
            self._timer = wm.event_timer_add(context.scene.simple_cloth.dt, window=context.window)
            wm.modal_handler_add(self)
            self._handler = bpy.types.SpaceView3D.draw_handler_add(self.draw_callback, (context,), 'WINDOW', 'POST_VIEW')
            return {'RUNNING_MODAL'}
        else:
            self.report({'ERROR'}, "No active mesh object selected")
            return {'CANCELLED'}

    def modal(self, context, event):
        if event.type == 'TIMER':
            self.update_simulation(context)
            context.area.tag_redraw()
        elif event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                if self.start_drag(context, event):
                    self._last_drag_update_pos = Vector((event.mouse_region_x, event.mouse_region_y))
                    self.dragging = True
                    return {'RUNNING_MODAL'}
            elif event.value == 'RELEASE':
                self.end_drag()
                self.dragging = False
                self._last_drag_update_pos = None
                return {'RUNNING_MODAL'}
        elif event.type == 'MOUSEMOVE':
            if self.dragging:
                current_pos = Vector((event.mouse_region_x, event.mouse_region_y))
                if (current_pos - self._last_drag_update_pos).length >= 5:
                    self.update_drag(context, event)
                    self._last_drag_update_pos = current_pos
        elif event.type in {'ESC'}:
            self.cancel(context)
            return {'CANCELLED'}
        return {'PASS_THROUGH'}

    @time_it
    def start_drag(self, context, event):
        mouse_pos = Vector((event.mouse_region_x, event.mouse_region_y))
        ray_origin = np.array(view3d_utils.region_2d_to_origin_3d(context.region, context.region_data, mouse_pos))
        ray_direction = np.array(view3d_utils.region_2d_to_vector_3d(context.region, context.region_data, mouse_pos))

        return start_drag_py(self.drag_data_ptr, self.particles_ptr, self.pcount, self.triangles_ptr, self.tcount, ray_origin, ray_direction)

    @time_it
    def update_drag(self, context, event):
        mouse_pos = Vector((event.mouse_region_x, event.mouse_region_y))
        ray_origin = np.array(view3d_utils.region_2d_to_origin_3d(context.region, context.region_data, mouse_pos))
        ray_direction = np.array(view3d_utils.region_2d_to_vector_3d(context.region, context.region_data, mouse_pos))
        
        # Project the ray onto a plane perpendicular to the view direction
        plane_normal = ray_direction
        plane_point = ray_origin + ray_direction  # Any point on the plane

        # Calculate the intersection of the ray with the plane
        t = np.dot(plane_point - ray_origin, plane_normal) / np.dot(ray_direction, plane_normal)
        new_position = ray_origin + t * ray_direction

        update_drag_py(self.drag_data_ptr, self.particles_ptr, self.pcount, new_position)

    def end_drag(self):
        end_drag_py(self.drag_data_ptr)

    def ray_triangle_intersection(self, ray_origin, ray_direction, v0, v1, v2):
        epsilon = 1e-6

        edge1 = v1 - v0
        edge2 = v2 - v0
        h = np.cross(ray_direction, edge2)
        a = np.dot(edge1, h)

        if abs(a) < epsilon:
            return None  # Ray is parallel to the triangle

        f = 1.0 / a
        s = ray_origin - v0
        u = f * np.dot(s, h)

        if u < 0.0 or u > 1.0:
            return None

        q = np.cross(s, edge1)
        v = f * np.dot(ray_direction, q)

        if v < 0.0 or u + v > 1.0:
            return None

        t = f * np.dot(edge2, q)

        if t > epsilon:
            return ray_origin + t * ray_direction

        return None

    def cancel(self, context):
        if hasattr(self, 'particles_ptr') and self.particles_ptr is not None:
            free_particles_py(self.particles_ptr)
            del self.particles_ptr
        if hasattr(self, 'constraints_ptr') and self.constraints_ptr is not None:
            free_constraints_py(self.constraints_ptr)
            del self.constraints_ptr
        if hasattr(self, 'drag_data_ptr') and self.drag_data_ptr is not None:
            free_drag_data_py(self.drag_data_ptr)
            del self.drag_data_ptr
        wm = context.window_manager
        wm.event_timer_remove(self._timer)
        bpy.types.SpaceView3D.draw_handler_remove(self._handler, 'WINDOW')
        context.area.tag_redraw()

    @time_it
    def initialize_simulation(self, context):
        props = context.scene.simple_cloth
        sim_ob = context.active_object

        # Get the evaluated mesh data
        depsgraph = context.evaluated_depsgraph_get()
        ob = sim_ob.evaluated_get(depsgraph)
        eval_mesh = ob.data

        # Get vertex positions with world matrix applied
        verts = np.empty(shape=(len(eval_mesh.vertices) * 3), dtype=np.float64)
        eval_mesh.attributes['position'].data.foreach_get('vector', verts)
        verts = verts.reshape(-1, 3)

        matrix_world = np.array(sim_ob.matrix_world)
        rotation_scale = matrix_world[:3, :3]
        translation = matrix_world[:3, 3]

        verts = np.dot(verts, rotation_scale.T)
        verts += translation

        # Initialize particles
        pinned_group = sim_ob.vertex_groups.get('PINNED')
        if pinned_group:
            self.pinned_vertices = set(v.index for v in eval_mesh.vertices if pinned_group.index in [g.group for g in v.groups])
        else:
            self.pinned_vertices = set()

        self.particles_ptr = create_particles_py(verts, self.pinned_vertices)
        self.pcount = len(eval_mesh.vertices)

        # Create distance constraints
        constraints = []
        for edge in eval_mesh.edges:
            v1, v2 = edge.vertices
            rest_length = np.linalg.norm(verts[v1] - verts[v2])
            constraints.append((v1, v2, rest_length))

        self.constraints_ptr = create_constraints_py(constraints)
        self.ccount = len(constraints)

        # Store mesh indices for triangle faces
        # Create triangles
        triangles = []
        for poly in eval_mesh.polygons:
            if len(poly.vertices) == 3:
                triangles.append(tuple(poly.vertices))
            elif len(poly.vertices) == 4:
                # Split quad into two triangles
                triangles.append((poly.vertices[0], poly.vertices[1], poly.vertices[2]))
                triangles.append((poly.vertices[2], poly.vertices[3], poly.vertices[0]))

        self.triangles_ptr = create_triangles_py(triangles)
        self.tcount = len(triangles)
        self.indices = triangles

        # If the mesh is not triangulated, you might want to add a warning or triangulate it
        if any(len(poly.vertices) > 4 for poly in eval_mesh.polygons):
            self.report({'WARNING'}, "Mesh contains n-gons (polygons with more than 4 vertices). These will be skipped.")

    # @time_it
    def update_simulation(self, context):
        props = context.scene.simple_cloth
        dt = props.dt

        update_positions_py(self.particles_ptr, self.pcount, dt, props.damping)
        solve_constraints_py(self.particles_ptr, self.constraints_ptr, self.pcount, self.ccount, props.stiffness, props.iterations)
        update_velocities_py(self.particles_ptr, self.pcount, dt)

        # Apply drag using the new DragData structure
        apply_drag_py(self.drag_data_ptr, self.particles_ptr, self.pcount)

        self.update_batches()

    # @time_it
    def update_batches(self):
        positions = get_positions_py(self.particles_ptr, self.pcount)
        vertices = [Vector(p) for p in positions]
        
        # Calculate normals
        normals = [Vector() for _ in vertices]
        for i1, i2, i3 in self.indices:
            v1, v2, v3 = vertices[i1], vertices[i2], vertices[i3]
            normal = (v2 - v1).cross(v3 - v1).normalized()
            normals[i1] += normal
            normals[i2] += normal
            normals[i3] += normal
        
        normals = [n.normalized() for n in normals]

        # self.point_batch = batch_for_shader(self.point_shader, 'POINTS', {"pos": vertices})
        self.mesh_batch = batch_for_shader(self.mesh_shader, 'TRIS', {
            "position": vertices,
            "normal": normals
        }, indices=self.indices)

    # @time_it
    def draw_callback(self, context):
        # Draw points
        # gpu.state.point_size_set(5)
        # gpu.state.blend_set('ALPHA')
        # gpu.state.depth_test_set('EQUAL')
        # self.point_shader.bind()
        # self.point_shader.uniform_float("color", (1, 1, 1, 1))
        # self.point_batch.draw(self.point_shader)

        # Draw mesh
        gpu.state.depth_test_set('LESS')
        gpu.state.depth_mask_set(True)
        # Remove face culling to render both sides
        # gpu.state.face_culling_set('BACK')
        self.mesh_shader.bind()
        
        model_view_matrix = context.region_data.view_matrix
        projection_matrix = context.region_data.window_matrix
        view_position = context.region_data.view_matrix.inverted().translation

        self.mesh_shader.uniform_float("modelViewMatrix", model_view_matrix)
        self.mesh_shader.uniform_float("projectionMatrix", projection_matrix)
        self.mesh_shader.uniform_float("lightDirection", (1, 1, 1))
        self.mesh_shader.uniform_float("viewPosition", view_position)
        self.mesh_shader.uniform_float("color", (0.2, 0.4, 0.8))  # Adjust color as needed
        self.mesh_shader.uniform_float("shininess", 32.0)
        self.mesh_shader.uniform_float("fresnelBias", 0.1)
        self.mesh_shader.uniform_float("fresnelScale", 1.0)
        self.mesh_shader.uniform_float("fresnelPower", 2.0)
        
        self.mesh_batch.draw(self.mesh_shader)

        gpu.state.depth_test_set('NONE')
        gpu.state.depth_mask_set(False)
        gpu.state.blend_set('NONE')
