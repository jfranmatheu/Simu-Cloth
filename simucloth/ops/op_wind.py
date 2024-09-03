import bpy

class CreateWindObjectOperator(bpy.types.Operator):
    bl_idname = "object.create_wind_object"
    bl_label = "Create Wind Object"
    
    def execute(self, context):
        bpy.ops.object.empty_add(type='SINGLE_ARROW')
        wind_object = context.active_object
        wind_object.name = "Wind Force"
        context.scene.simple_cloth.wind_object = wind_object
        return {'FINISHED'}
