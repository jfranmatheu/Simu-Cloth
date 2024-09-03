import bpy
from .ops.op_sim import SimpleClothOperator
from .ops.op_wind import CreateWindObjectOperator


class SimpleClothPanel(bpy.types.Panel):
    bl_label = "Simple Cloth Simulation"
    bl_idname = "VIEW3D_PT_simple_cloth"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "SimpleCloth"

    def draw(self, context):
        layout = self.layout
        props = context.scene.simple_cloth
        layout.prop(props, "mass")
        layout.prop(props, "stiffness")
        layout.prop(props, "damping")
        layout.prop(props, "iterations")
        layout.prop(props, "dt")
        layout.label(text="Wind Force:")
        layout.prop(props, "wind_force")
        layout.prop(props, "wind_turbulence")
        layout.prop(props, "wind_object")
        layout.operator(CreateWindObjectOperator.bl_idname)
        layout.operator(SimpleClothOperator.bl_idname)
