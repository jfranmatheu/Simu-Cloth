import bpy


class SimpleClothProperties(bpy.types.PropertyGroup):
    mass: bpy.props.FloatProperty(name="Mass", default=1.0, min=0.01, max=10.0)
    stiffness: bpy.props.FloatProperty(name="Stiffness", default=0.9, min=0.0, max=1.0)
    damping: bpy.props.FloatProperty(name="Damping", default=0.01, min=0.0, max=1.0)
    iterations: bpy.props.IntProperty(name="Solver Iterations", default=5, min=1, max=50)
    dt: bpy.props.FloatProperty(name="Simulation Speed", default=0.03, min=0.001, max=0.1)
    
    # Wind Force attributes.
    wind_force: bpy.props.FloatProperty(name="Wind Force", default=0.0, min=0.0, max=10.0)
    wind_turbulence: bpy.props.FloatProperty(name="Wind Turbulence", default=0.0, min=0.0, max=1.0)
    wind_object: bpy.props.PointerProperty(name="Wind Object", type=bpy.types.Object)


def register():
    bpy.types.Scene.simple_cloth = bpy.props.PointerProperty(type=SimpleClothProperties)

def unregister():
    del bpy.types.Scene.simple_cloth
