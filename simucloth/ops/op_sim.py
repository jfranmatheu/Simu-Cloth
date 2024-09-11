import os
import sys
import numpy as np
import ctypes

import bpy
import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Vector
from bpy_extras import view3d_utils
from mathutils import Matrix

from ..shaders import mesh_vertex_shader, mesh_fragment_shader
from ..utils.time_it import time_it
from ..utils.object import (
    get_local_axis_direction,
    get_matrix_world_ptr,
    get_mesh_corner_vert_ptr,
    get_mesh_loop_triangles_ptr,
    get_mesh_polygon_normals_ptr,
    get_mesh_vert_positions_ptr,
    get_mesh_vert_normals_ptr,
    get_matrix_world_ptr,
    get_min_max_bound_box
)
from ..cy_types import SimulationObject, ObjectType


# Add the directory containing the simucloth folder to Python's path
# addon_dir = os.path.dirname(os.path.abspath(__file__))
# if addon_dir not in sys.path:
#     sys.path.append(addon_dir)

try:
    from ..cy_simucloth.sim_objects import (
        create_simulation_objects_py,
        update_simulation_object_py,
        free_simulation_objects_py,
    )
    from ..cy_simucloth.cloth_simulation import (
        get_positions_py,
        update_positions_py,
        solve_constraints_py,
        update_velocities_py,
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
        self.point_shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        self.mesh_shader = gpu.types.GPUShader(mesh_vertex_shader, mesh_fragment_shader)
        self.point_batches = []
        self.mesh_batches = []
        self.dragging = False
        self.drag_data_ptr = None
        self._drag_influence_radius = 1.0  # Adjust this value as needed
        self.simulation_time = 0.0

    def __del__(self):
        # if hasattr(self, 'drag_data_ptr') and self.drag_data_ptr is not None:
        #     free_drag_data_py(self.drag_data_ptr)
        pass

    def execute(self, context):
        # self.drag_data_ptr = create_drag_data_py(self._drag_influence_radius)
        # init_perlin_py()  # Initialize Perlin noise
        self.initialize_simulation(context)
        wm = context.window_manager
        self._timer = wm.event_timer_add(context.scene.simple_cloth.dt, window=context.window)
        wm.modal_handler_add(self)
        self._handler = bpy.types.SpaceView3D.draw_handler_add(self.draw_callback, (context,), 'WINDOW', 'POST_VIEW')
        context.area.tag_redraw()
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'TIMER':
            self.update_simulation(context)
            context.area.tag_redraw()
        # elif event.type == 'LEFTMOUSE':
        #     if event.value == 'PRESS':
        #         if self.start_drag(context, event):
        #             self._last_drag_update_pos = Vector((event.mouse_region_x, event.mouse_region_y))
        #             self.dragging = True
        #             return {'RUNNING_MODAL'}
        #     elif event.value == 'RELEASE':
        #         self.end_drag()
        #         self.dragging = False
        #         self._last_drag_update_pos = None
        #         return {'RUNNING_MODAL'}
        # elif event.type == 'MOUSEMOVE':
        #     if self.dragging:
        #         current_pos = Vector((event.mouse_region_x, event.mouse_region_y))
        #         if (current_pos - self._last_drag_update_pos).length >= 5:
        #             self.update_drag(context, event)
        #             self._last_drag_update_pos = current_pos
        elif event.type in {'ESC'}:
            self.cancel(context)
            return {'CANCELLED'}
        return {'PASS_THROUGH'}

    '''
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
    '''

    def cancel(self, context):
        # free_perlin_py()  # Clean up Perlin noise
        # if hasattr(self, 'drag_data_ptr') and self.drag_data_ptr is not None:
        #     free_drag_data_py(self.drag_data_ptr)
        #     del self.drag_data_ptr
        wm = context.window_manager
        wm.event_timer_remove(self._timer)
        bpy.types.SpaceView3D.draw_handler_remove(self._handler, 'WINDOW')
        context.area.tag_redraw()
        # Free simulation objects after each update
        free_simulation_objects_py()
        self.simulation_objects.clear()
        # Clear batches.
        self.point_batches.clear()
        self.mesh_batches.clear()

    def prepare_objects(self, context):
        USE_EVAL = False
        OBJECT_TYPE = ObjectType.CLOTH.value

        self.simulation_objects = {}
        self.mesh_draw_data = {}
        depsgraph = context.evaluated_depsgraph_get()
        for ob in context.scene.collection.children['SimuCloth-Cloths'].all_objects:
            if ob.type != 'MESH':
                continue
            IS_CLOTH = OBJECT_TYPE == ObjectType.CLOTH.value
            eval_ob = ob.evaluated_get(depsgraph) if USE_EVAL else ob
            if IS_CLOTH:
                sim_obj, mesh_draw_data = SimulationObject.from_object(eval_ob, return_mesh_draw_data=True)
                self.mesh_draw_data[ob['simucloth_ob_id']] = mesh_draw_data
            else:
                sim_obj = SimulationObject.from_object(eval_ob)
            sim_obj.type = OBJECT_TYPE

            self.simulation_objects[ob['simucloth_ob_id']] = sim_obj
        
        for obj in self.simulation_objects.values():
            print(f"SimulationObject: type={obj.type}, tot_particles={obj.tot_particles}, tot_constraints={obj.tot_constraints}")
            # Print a few particle positions for verification
            # for i in range(min(5, obj.tot_particles)):
            #     particle = obj.particles[i]
            #     print(f"Particle {i}: pos=({particle.pos.x}, {particle.pos.y}, {particle.pos.z})")
            # for i in range(min(5, obj.tot_constraints)):
            #     constraint = obj.constraints[i]
            #     print(f"Constraint {i}: v1={constraint.v1}, v2={constraint.v2}, rest_length={constraint.rest_length}")


        # Pass the simulation objects to the Cython functions
        create_simulation_objects_py(list(self.simulation_objects.values()))

    # @time_it
    def initialize_simulation(self, context: bpy.types.Context):
        # Initialize simulation objects
        self.prepare_objects(context)

    # @time_it
    def update_simulation(self, context):
        props = context.scene.simple_cloth
        dt = props.dt
        self.simulation_time += dt  # Increment simulation time

        update_positions_py(dt, props.damping, list(props.gravity))
        solve_constraints_py(props.stiffness, props.iterations)
        update_velocities_py(dt)
        
        # for i in range(len(self.simulation_objects)):
        #     print(f"SimulationObject {i}: tot_particles={self.simulation_objects[i].tot_particles}, tot_constraints={self.simulation_objects[i].tot_constraints}")
        #     for j in range(min(5, self.simulation_objects[i].tot_constraints)):
        #         constraint = self.simulation_objects[i].constraints[j]
        #         print(f"Constraint {j}: v1={constraint.v1}, v2={constraint.v2}, rest_length={constraint.rest_length}, distance={constraint.distance}")

        # apply_drag_py(self.drag_data_ptr)

        self.update_batches()

        '''wind_force = Vector((0, 0, 0))
        if props.wind_object and props.wind_force > 0:
            wind_direction = get_local_axis_direction(props.wind_object, 'Y')
            wind_force = wind_direction * props.wind_force

        wind_force_np = np.array([wind_force.x, wind_force.y, wind_force.z], dtype=np.float64)

        apply_wind_force_py(
            wind_force_np,
            props.wind_turbulence,
            self.simulation_time,
            dt
        )'''

        # Prepare collision object data
        '''collision_object_data = []
        depsgraph = context.evaluated_depsgraph_get()
        for ob in context.scene.collection.children['SimuCloth-Colliders'].all_objects:
            if ob.type != 'MESH':
                continue
            eval_ob = ob.evaluated_get(depsgraph)
            mesh = eval_ob.data
            min_bb, max_bb = get_min_max_bound_box(eval_ob)
            collision_object_data.append({
                'loop_tri_ptr': get_mesh_loop_triangles_ptr(mesh),
                'corner_vert_ptr': get_mesh_corner_vert_ptr(mesh),
                'tri_normals_ptr': get_mesh_polygon_normals_ptr(mesh),
                'vert_positions_ptr': get_mesh_vert_positions_ptr(mesh),
                'vert_normals_ptr': get_mesh_vert_normals_ptr(mesh),
                'tot_tris': len(mesh.polygons),
                'tot_verts': len(mesh.vertices),
                'matrix_world': [v for row in eval_ob.matrix_world for v in row],
                'min_point': list(min_bb),
                'max_point': list(max_bb)
            })

        # Handle collisions
        handle_collisions_py(
            props.friction
        )'''


    # @time_it
    def update_batches(self):
        self.point_batches = []
        self.mesh_batches = []

        index = 0
        for ob_id in self.simulation_objects.keys():
            try:
                positions = get_positions_py(index)
                if positions.size > 0:
                    if mesh_draw_data := self.mesh_draw_data.get(ob_id):
                        mesh_draw_data['vert_positions'] = positions
                    # print(f"Info: GPUBatch created. Numpy size: {positions.size}. Numpy shape: {positions.shape}")
                    # print(f"POSITIONS at time {self.simulation_time}: ", positions)
                    self.point_batches.append((
                        batch_for_shader(self.point_shader, 'POINTS', {"pos": positions}),
                        self.simulation_objects[ob_id]
                    ))
                else:
                    print(f"Warning: No positions returned for object {ob_id}")
            except Exception as e:
                print(f"Error getting positions for object {ob_id}: {str(e)}")
            index += 1

        '''# Calculate normals
        normals = [Vector() for _ in vertices]
        for i1, i2, i3 in self.indices:
            v1, v2, v3 = vertices[i1], vertices[i2], vertices[i3]
            normal = (v2 - v1).cross(v3 - v1).normalized()
            normals[i1] += normal
            normals[i2] += normal
            normals[i3] += normal
        
        normals = [n.normalized() for n in normals]'''

        for ob_id in self.simulation_objects.keys():
            if mesh_draw_data := self.mesh_draw_data.get(ob_id):
                batch = batch_for_shader(self.mesh_shader, 'TRIS', {
                    "position": mesh_draw_data['vert_positions'],
                    "normal": mesh_draw_data['vert_normals']
                }, indices=mesh_draw_data['tri_indices'])
                self.mesh_batches.append((
                    batch,
                    self.simulation_objects[ob_id]
                ))

    # @time_it
    def draw_callback(self, context):
        # Draw points
        gpu.state.point_size_set(5)
        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set('LESS')  # 'EQUAL' (if draw mesh), 'LESS'
        batch_count = len(self.point_batches)
        for batch_index, (batch, sim_obj) in enumerate(self.point_batches):
            color = (
                1 - batch_index / batch_count,
                batch_index / batch_count,
                1,
                1
            )
            self.point_shader.uniform_float("color", color)
            with gpu.matrix.push_pop():
                model_view_mat = gpu.matrix.get_model_view_matrix()
                gpu.matrix.load_matrix(model_view_mat @ Matrix(sim_obj.matrix_world))
                batch.draw(self.point_shader)

        # Draw mesh
        gpu.state.depth_test_set('LESS')
        gpu.state.depth_mask_set(True)
        # Remove face culling to render both sides
        # gpu.state.face_culling_set('BACK')

        model_view_matrix = context.region_data.view_matrix
        projection_matrix = context.region_data.window_matrix
        view_position = context.region_data.view_matrix.inverted().translation

        self.mesh_shader.uniform_float("projectionMatrix", projection_matrix)
        self.mesh_shader.uniform_float("lightDirection", (1, 1, 1))
        self.mesh_shader.uniform_float("viewPosition", view_position)
        self.mesh_shader.uniform_float("shininess", 32.0)
        self.mesh_shader.uniform_float("fresnelBias", 0.1)
        self.mesh_shader.uniform_float("fresnelScale", 1.0)
        self.mesh_shader.uniform_float("fresnelPower", 2.0)

        batch_count = len(self.mesh_batches)
        for batch_index, (batch, sim_obj) in enumerate(self.mesh_batches):
            color = (
                1 - batch_index / batch_count,
                batch_index / batch_count,
                1,
                1
            )
            self.mesh_shader.uniform_float("color", color)
            self.mesh_shader.uniform_float("modelViewMatrix", model_view_matrix @ Matrix(sim_obj.matrix_world))
            batch.draw(self.mesh_shader)

        gpu.state.depth_test_set('NONE')
        gpu.state.depth_mask_set(False)
        gpu.state.blend_set('NONE')
