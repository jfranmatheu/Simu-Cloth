import ctypes
from typing import List, Tuple
import numpy as np
from enum import IntEnum

import bpy


class ObjectType(IntEnum):
    COLLIDER = 0
    CLOTH = 1
    FLUID = 2
    SOLID = 3

class ConstraintType(IntEnum):
    SPRING = 0
    BENDING = 1
    SHEARING = 2
    PIN = 3


class Vector3(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float),
                ("y", ctypes.c_float),
                ("z", ctypes.c_float)]

class Triangle(ctypes.Structure):
    _fields_ = [("i1", ctypes.c_int),
                ("i2", ctypes.c_int),
                ("i3", ctypes.c_int)]

class Particle(ctypes.Structure):
    _fields_ = [("pos", Vector3),
                ("prev_pos", Vector3),
                ("vel", Vector3),
                ("pinned", ctypes.c_bool)]

class Constraint(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int),
                ("v1", ctypes.c_int),
                ("v2", ctypes.c_int),
                ("rest_length", ctypes.c_float),
                ("distance", ctypes.c_float)]

class Mesh(ctypes.Structure):
    _fields_ = [("loop_tri", ctypes.POINTER(Triangle)),
                ("corner_vert", ctypes.POINTER(Triangle)),
                ("tri_normals", ctypes.POINTER(Vector3)),
                ("vert_positions", ctypes.POINTER(Vector3)),
                ("vert_normals", ctypes.POINTER(Vector3)),
                ("tot_tris", ctypes.c_int),
                ("tot_verts", ctypes.c_int)]
    
    @classmethod
    def from_python(cls,
                    loop_tri_ptr: int,
                    corner_vert_ptr: int,
                    tri_normals_ptr: int,
                    vert_positions_ptr: int,
                    vert_normals_ptr: int,
                    tot_tris: int,
                    tot_verts: int) -> 'Mesh':
        obj = cls()
        obj.tot_tris = tot_tris
        obj.tot_verts = tot_verts
        obj.loop_tri = ctypes.cast(loop_tri_ptr, ctypes.POINTER(Triangle))
        obj.corner_vert = ctypes.cast(corner_vert_ptr, ctypes.POINTER(Triangle))
        obj.tri_normals = ctypes.cast(tri_normals_ptr, ctypes.POINTER(Vector3))
        obj.vert_positions = ctypes.cast(vert_positions_ptr, ctypes.POINTER(Vector3))
        obj.vert_normals = ctypes.cast(vert_normals_ptr, ctypes.POINTER(Vector3))
        return obj

    @classmethod
    def from_mesh(cls, eval_mesh: bpy.types.Mesh) -> 'Mesh':
        return cls.from_python(
            loop_tri_ptr=eval_mesh.loop_triangles[0].as_pointer(),
            corner_vert_ptr=eval_mesh.attributes['.corner_vert'].data[0].as_pointer(),
            tri_normals_ptr=eval_mesh.polygon_normals[0].as_pointer(),
            vert_positions_ptr=eval_mesh.attributes['position'].data[0].as_pointer(),
            vert_normals_ptr=eval_mesh.vertex_normals[0].as_pointer(),
            tot_tris=len(eval_mesh.polygons),
            tot_verts=len(eval_mesh.vertices)
        )

    def free(self):
        if self.loop_tri:
            ctypes.pythonapi.PyMem_Free(ctypes.c_void_p(self.loop_tri))
        if self.corner_vert:
            ctypes.pythonapi.PyMem_Free(ctypes.c_void_p(self.corner_vert))
        if self.tri_normals:
            ctypes.pythonapi.PyMem_Free(ctypes.c_void_p(self.tri_normals))
        if self.vert_positions:
            ctypes.pythonapi.PyMem_Free(ctypes.c_void_p(self.vert_positions))
        if self.vert_normals:
            ctypes.pythonapi.PyMem_Free(ctypes.c_void_p(self.vert_normals))


class SimulationObject(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int),
                ("mesh", Mesh),
                ("particles", ctypes.POINTER(Particle)),
                ("constraints", ctypes.POINTER(Constraint)),
                ("tot_particles", ctypes.c_int),
                ("tot_constraints", ctypes.c_int),
                ("location", Vector3),
                ("rotation", Vector3),
                ("scale", Vector3),
                ("matrix_world", (ctypes.c_float  * 4) * 4),
                ("bbox_min", Vector3),
                ("bbox_max", Vector3)]

    @classmethod
    def from_python(cls,
                    type: ObjectType,
                    mesh: Mesh,
                    particles: List[Particle],
                    constraints: List[Constraint],
                    location: Tuple[float, float, float],
                    rotation: Tuple[float, float, float],
                    scale: Tuple[float, float, float],
                    matrix_world: List[List[float]],
                    bbox_min: Tuple[float, float, float],
                    bbox_max: Tuple[float, float, float]) -> 'SimulationObject':
        obj = cls()
        obj.type = ctypes.c_int(type.value)
        obj.mesh = mesh
        obj.tot_particles = ctypes.c_int(len(particles))
        obj.particles = (Particle * obj.tot_particles)(*particles)
        obj.tot_constraints = ctypes.c_int(len(constraints))
        obj.constraints = (Constraint * obj.tot_constraints)(*constraints)
        obj.location = Vector3(*location)
        obj.rotation = Vector3(*rotation)
        obj.scale = Vector3(*scale)
        obj.matrix_world = ((ctypes.c_float * 4) * 4)(*[tuple(row) for row in matrix_world])
        obj.bbox_min = Vector3(*bbox_min)
        obj.bbox_max = Vector3(*bbox_max)
        return obj

    @classmethod
    def from_object(cls, eval_obj: bpy.types.Object) -> 'SimulationObject':
        # Get the evaluated mesh data
        eval_mesh: bpy.types.Mesh = eval_obj.data

        # Create particles
        verts = np.empty(shape=(len(eval_mesh.vertices) * 3), dtype=np.float32)
        eval_mesh.attributes['position'].data.foreach_get('vector', verts)

        # Check for invalid vertex positions
        if np.isnan(verts).any() or np.isinf(verts).any():
            raise ValueError("Invalid vertex positions detected in the input mesh")

        verts = verts.reshape(-1, 3)

        particles = []
        pinned_group = eval_obj.vertex_groups.get('PINNED')
        for i, v in enumerate(verts):
            pinned = pinned_group.index in {g.group for g in eval_mesh.vertices[i].groups}
            particle = Particle(pos=Vector3(*v), prev_pos=Vector3(*v), vel=Vector3(0, 0, 0), pinned=pinned)
            particles.append(particle)
            print(f"Particle {i}: pos=({particle.pos.x}, {particle.pos.y}, {particle.pos.z}), pinned={pinned}")

        # Create constraints
        constraints = []
        for edge in eval_mesh.edges:
            v1, v2 = edge.vertices
            rest_length = np.linalg.norm(verts[v1] - verts[v2])
            constraint = Constraint(type=0, v1=v1, v2=v2, rest_length=rest_length, distance=rest_length)
            constraints.append(constraint)
            print(f"Constraint {i}: v1={constraint.v1}, v2={constraint.v2}, rest_length={constraint.rest_length}")

        # Create mesh
        # triangles = []
        # for poly in eval_mesh.polygons:
        #     if len(poly.vertices) == 3:
        #         triangles.append(Triangle(*poly.vertices))
        #     elif len(poly.vertices) == 4:
        #         triangles.append(Triangle(poly.vertices[0], poly.vertices[1], poly.vertices[2]))
        #         triangles.append(Triangle(poly.vertices[2], poly.vertices[3], poly.vertices[0]))

        # Create SimulationObject
        sim_obj = cls.from_python(
            type=ObjectType.CLOTH,
            mesh=Mesh.from_mesh(eval_mesh),
            particles=particles,
            constraints=constraints,
            location=tuple(eval_obj.location),
            rotation=tuple(eval_obj.rotation_euler),
            scale=tuple(eval_obj.scale),
            matrix_world=list(eval_obj.matrix_world.row),
            bbox_min=tuple(eval_obj.bound_box[0]),
            bbox_max=tuple(eval_obj.bound_box[6])
        )

        return sim_obj

    def free(self):
        if self.mesh:
            self.mesh.contents.free()
            ctypes.pythonapi.PyMem_Free(ctypes.c_void_p(self.mesh))
        if self.particles:
            ctypes.pythonapi.PyMem_Free(ctypes.c_void_p(self.particles))
        if self.constraints:
            ctypes.pythonapi.PyMem_Free(ctypes.c_void_p(self.constraints))
