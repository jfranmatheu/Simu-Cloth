import bpy
from .ops.op_sim import SimpleClothOperator
from .ops.op_wind import CreateWindObjectOperator
from .ops.op_collision import AddColliderOperator
from .ops.op_sim_object import AddClothOperator


class SimuClothPanel_SimulationSettings(bpy.types.Panel):
    bl_label = "Simulation Settings"
    bl_idname = "VIEW3D_PT_simu_cloth_settings"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Simu-Cloth"

    def draw(self, context):
        layout = self.layout
        props = context.scene.simple_cloth
        layout.prop(props, "mass")
        layout.prop(props, "stiffness")
        layout.prop(props, "damping")
        layout.prop(props, "iterations")
        layout.prop(props, "dt")
        layout.separator()
        layout.prop(props, "gravity")
        layout.label(text="Wind Force:")
        layout.prop(props, "wind_force")
        layout.prop(props, "wind_turbulence")
        layout.prop(props, "wind_object")
        layout.operator(CreateWindObjectOperator.bl_idname)
        layout.label(text="Collision:")
        layout.prop(props, "friction")
        layout.operator(AddColliderOperator.bl_idname)
        layout.separator()
        layout.operator(SimpleClothOperator.bl_idname)


class SimuClothPanel_SimulationObjects(bpy.types.Panel):
    bl_label = "Simulation Objects"
    bl_idname = "VIEW3D_PT_simu_cloth_objects"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Simu-Cloth"

    def draw(self, context):
        layout = self.layout
        layout.operator(AddClothOperator.bl_idname)
        layout.operator(AddColliderOperator.bl_idname)
