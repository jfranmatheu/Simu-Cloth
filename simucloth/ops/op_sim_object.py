import bpy
import uuid


class AddClothOperator(bpy.types.Operator):
    bl_idname = "object.add_cloth"
    bl_label = "Add Cloth"
    bl_description = "Add selected object as a cloth for the cloth simulation"

    def execute(self, context):
        colliders_collection = bpy.data.collections.get('SimuCloth-Cloths')
        if colliders_collection is None:
            colliders_collection = bpy.data.collections.new('SimuCloth-Cloths')
            context.scene.collection.children.link(colliders_collection)

        for obj in context.selected_objects:
            if obj.type == 'MESH':
                if 'simucloth_ob_id' in obj:
                    continue
                
                obj['simucloth_ob_id'] = str(uuid.uuid4().hex)
                
                # Add 'PINNED' vertedx group.
                obj.vertex_groups.new(name='PINNED')

                # Link object to the colliders collection
                if obj.name not in colliders_collection.objects:
                    colliders_collection.objects.link(obj)

        return {'FINISHED'}
