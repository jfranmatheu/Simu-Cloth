import bpy
import uuid


class AddColliderOperator(bpy.types.Operator):
    bl_idname = "object.add_collider"
    bl_label = "Add Collider"
    bl_description = "Add selected object as a collider for the cloth simulation"

    def execute(self, context):
        colliders_collection = bpy.data.collections.get('SimuCloth-Colliders')
        if colliders_collection is None:
            colliders_collection = bpy.data.collections.new('SimuCloth-Colliders')
            context.scene.collection.children.link(colliders_collection)

        for obj in context.selected_objects:
            if obj.type == 'MESH':
                if 'simucloth_ob_uid' in obj:
                    continue
                
                obj['simucloth_ob_uid'] = str(uuid.uuid4().hex)
                
                # Add triangulate modifier if not already present
                if not any(mod.type == 'TRIANGULATE' for mod in obj.modifiers):
                    triangulate_mod = obj.modifiers.new(name="SimuClothTriangulate", type='TRIANGULATE')
                    triangulate_mod.quad_method = 'BEAUTY'
                    triangulate_mod.ngon_method = 'BEAUTY'

                # Link object to the colliders collection
                if obj.name not in colliders_collection.objects:
                    colliders_collection.objects.link(obj)

        return {'FINISHED'}
