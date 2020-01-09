import bpy, bmesh
from bpy_extras import view3d_utils
from mathutils import Vector, Matrix
from mathutils.geometry import intersect_point_line
from datetime import datetime

# Author: Alexander Nedovizin, Andrei Rusnak
# ref:
# https://github.com/Cfyzzz/MetaSculptBlender.git

valid_types = ('MESH', 'META')
valid_types_mesh = ('MESH')


def main_active(context, event):
    """Run this function on left mouse, execute the ray cast"""
    # get the context arguments
    scene = context.scene
    region = context.region
    rv3d = context.region_data
    coord = event.mouse_region_x, event.mouse_region_y

    # get the ray from the viewport and mouse
    view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
    ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)

    ray_target = ray_origin + view_vector

    def visible_objects_and_duplis():
        """Loop over (object, matrix) pairs (mesh only)"""

        # for obj in context.visible_objects:
        for obj in context.scene.objects:
            if obj.type in valid_types:
                yield (obj, obj.matrix_world.copy())

            if obj.dupli_type != 'NONE':
                obj.dupli_list_create(scene)
                for dob in obj.dupli_list:
                    obj_dupli = dob.object
                    if obj_dupli.type in valid_types:
                        yield (obj_dupli, dob.matrix.copy())

            obj.dupli_list_clear()

    def obj_ray_cast(obj, matrix):
        """Wrapper for ray casting that moves the ray into object space"""

        # get the ray relative to the object
        matrix_inv = matrix.inverted()
        ray_origin_obj = matrix_inv * ray_origin
        ray_target_obj = matrix_inv * ray_target
        ray_direction_obj = ray_target_obj - ray_origin_obj

        # cast the ray
        success, location, normal, face_index = obj.ray_cast(ray_origin_obj, ray_direction_obj)
        if success:
            return location, normal, face_index
        else:
            return None, None, None

    # cast rays and find the closest object
    best_length_squared = -1.0
    best_obj = None

    for obj, matrix in visible_objects_and_duplis():
        if obj.type in valid_types_mesh:
            hit, normal, face_index = obj_ray_cast(obj, matrix)
            if hit is not None:
                hit_world = matrix * hit
                scene.cursor_location = hit_world
                length_squared = (hit_world - ray_origin).length_squared
                if best_obj is None or length_squared < best_length_squared:
                    best_length_squared = length_squared
                    best_obj = obj

    # now we have the object under the mouse cursor,
    # we could do lots of stuff but for the example just select.
    if best_obj is not None:
        bpy.ops.object.select_all(action='DESELECT')
        best_obj.select = True
        context.scene.objects.active = best_obj
        return True
    else:
        return False


class SculptExtrudeOperator(bpy.types.Operator):
    """Modal object selection with a ray cast"""
    bl_idname = "view3d.modal_operator_sculpt_extrude"
    bl_label = "SculptExtrude Operator"

    doPick = False
    vector = None
    obj_to = None
    matrix = Matrix()
    moved = False
    arr_mesh = []
    arr_meta = []
    arr_mesh_all = []
    actObj = None
    brushObj = None
    sculptObj = None
    actPos = None
    start_meta = True
    spheres = []  # vers 16. Сферы для темповой геометрии

    def clearObjs(self):
        self.brushObj = None
        self.actObj = None
        self.actPos = None
        self.start_meta = True

    def objFromCopy(self, object, location):
        tobj = len(bpy.data.objects)
        t1 = datetime.now()
        obj_new = object.copy()
        obj_new.location = location
        bpy.context.scene.objects.link(obj_new)
        t2 = datetime.now()
        t1_ = t1.second * 1e5 + t1.microsecond
        t2_ = t2.second * 1e5 + t2.microsecond
        dt = t2_ - t1_
        # print("dt=  %.4f sec"%dt)

        if object.type == 'META':
            bpy.ops.mesh.primitive_uv_sphere_add(segments=4, ring_count=2, size=3, location=location)
            new_sphere = bpy.context.active_object
            new_sphere.hide = True
            bpy.context.scene.objects.active = self.actObj
            self.spheres.append(new_sphere)

        return obj_new

    def addArr(self, obj):
        if obj is None:
            return False
        if obj.type == 'MESH':
            self.arr_mesh.append(obj)
        elif obj.type == 'META':
            self.arr_meta.append(obj)

    def start_point_drawing(self, context, event):
        config = bpy.context.scene.sculptextrude_manager
        if self.actPos is None:
            self.actPos = context.scene.cursor_location
            origin = self.getOrigin(context, event, True)
            self.actPos = origin
        else:
            origin = self.getOrigin(context, event, True)
        new_obj = self.objFromCopy(self.brushObj, origin)
        self.actObj = new_obj
        self.actPos = origin

        new_obj = self.objFromCopy(self.brushObj, origin)
        self.drawObj = new_obj
        self.drawPos = origin
        context.scene.objects.active = self.drawObj
        self.dist = config.dist
        return True

    def finishDraw(self, context):
        self.moved = True
        self.start_meta = True

        self.vector = None

        if len(self.spheres) > 1:
            sph = self.spheres[0]
            sph.hide = False
            context.scene.objects.active = sph
            sph.select = True
            for sph_ in self.spheres[1:]:
                sph_.hide = False
                sph_.select = True
            bpy.ops.object.join()
            sph.hide = True
            self.spheres = [sph]

        if len(self.arr_mesh) > 1:
            bpy.ops.object.make_single_user(object=True, obdata=True)
            bpy.ops.object.select_all(action='DESELECT')
            if self.sculptObj is None:
                self.sculptObj = self.arr_mesh[0]
                arm_tmp = self.arr_mesh[1:]
            else:
                arm_tmp = self.arr_mesh[:]

            context.scene.objects.active = self.sculptObj
            self.sculptObj.select = True
            for msh in arm_tmp:
                if msh.name == self.sculptObj.name:
                    continue

                msh.select = True
                self.arr_mesh.pop(self.arr_mesh.index(msh))

            if len(arm_tmp) > 0:
                bpy.ops.object.booltron_union()

        bpy.ops.object.select_all(action='DESELECT')
        context.scene.objects.active = self.brushObj
        self.brushObj.select = True
        self.obj_to = None
        self.doPick = False

    def move(self, context):
        config = bpy.context.scene.sculptextrude_manager
        view_user = config.user_view
        delta = self.drawPos - self.actPos
        if delta.length < self.dist: return

        if view_user:
            self.matrix = self.getViewMatrix()
            self.drawObj.matrix_local = self.matrix
            self.drawObj.location = self.drawPos
            tmp_loc = self.actObj.location.copy()
            self.actObj.matrix_local = self.matrix
            self.actObj.location = tmp_loc

        numb = int(delta.length // self.dist)
        if numb > 1:
            numb += 1
        norm = delta.normalized()
        tmp_pos = self.actPos.copy()
        if view_user:
            norm = norm * self.matrix
        if self.vector is None:
            self.vector = norm.copy()
            if not view_user:
                self.matrix = self.brushObj.matrix_local.copy()

        for u in range(numb):
            if view_user:
                new_loc = norm * self.matrix.inverted() * self.dist * (u + 1)
            else:
                new_loc = norm * self.dist * (u + 1)
            obj_new = self.objFromCopy(self.brushObj, tmp_pos + new_loc)

            q_rot = self.vector.rotation_difference(norm)
            obj_new.matrix_local = self.matrix * q_rot.to_matrix().to_4x4()
            obj_new.location = tmp_pos + new_loc
            self.drawObj.matrix_local = self.matrix * q_rot.to_matrix().to_4x4()
            self.drawObj.location = self.drawPos
            self.addArr(obj_new)

        if numb > 0:
            self.actPos = obj_new.location
        return {'FINISHED'}

    def getViewMatrix(self):
        mat = bpy.context.region_data.view_matrix.inverted().copy()
        return mat

    def getOrigin(self, context, event, mode=False):
        scene = context.scene
        region = context.region
        rv3d = context.region_data
        coord = event.mouse_region_x, event.mouse_region_y

        # get the ray from the viewport and mouse
        view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)

        origin_ = intersect_point_line(self.actPos, ray_origin, ray_origin + view_vector)
        origin = origin_[0]
        return origin

    def modal(self, context, event):
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE'} and not event.shift:
            # allow navigation
            return {'PASS_THROUGH'}
        elif event.type == 'WHEELUPMOUSE' and event.shift:
            if context.active_object.type == 'META':
                bpy.data.metaballs[0].resolution += 0.1
        elif event.type == 'WHEELDOWNMOUSE' and event.shift:
            if context.active_object.type == 'META':
                bpy.data.metaballs[0].resolution -= 0.1
        elif event.type == 'MOUSEMOVE' and self.doPick: # and not self.moved:
            self.moved = True
            origin = self.getOrigin(context, event, False)
            self.drawObj.location = origin
            self.drawPos = origin
            self.move(context)
            self.moved = False
        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS' \
                and self.doPick is False and self.brushObj is not None: 
            self.doPick = True
            self.moved = False
            ma = main_active(context, event)
            self.arr_mesh = []
            self.arr_meta = []
            if ma:
                self.actPos = context.scene.cursor_location
            else:
                self.actPos = None

            self.start_point_drawing(context, event)
            self.addArr(self.actObj)
            self.addArr(self.drawObj)
            self.vector = None
            return {'PASS_THROUGH'}
        elif event.type == 'LEFTMOUSE' and event.value == 'RELEASE' and self.doPick:
            self.finishDraw(context)
            self.pushArrMesh()
            self.arr_meta = []
            return {'PASS_THROUGH'}
        elif event.type in {'RIGHTMOUSE'}:
            # print(event.mouse_region_x , event.mouse_region_y)
            ma = main_active(context, event)
            if ma:
                self.brushObj = context.active_object
            else:
                tmp_obj = context.active_object
                if tmp_obj.type in valid_types:
                    self.brushObj = tmp_obj
                else:
                    bpy.ops.object.select_all(action='DESELECT')
                    self.clearObjs()
            return {'PASS_THROUGH'}


        elif event.type in {'ESC'}: 
            # print(event.mouse_region_x , event.mouse_region_y)
            return {'FINISHED'}
        elif event.type in {'SPACE'}:
            # Объеднить геометрию
            self.arr_mesh = self.arr_mesh_all
            self.finishDraw(context)
            self.arr_mesh_all = []

        elif event.type in {'Z'}:
            # print('KEY_Z')
            if self.actObj.type == 'META':
                bpy.ops.object.convert(target='MESH')

        return {'RUNNING_MODAL'}

    def pushArrMesh(self):
        # сбрасываем всё в общий массив и обнуляем мазок
        self.arr_mesh_all.extend(self.arr_mesh)
        self.arr_mesh = []
        if self.sculptObj and self.sculptObj not in self.arr_mesh_all:
            self.arr_mesh_all.append(self.sculptObj)
        self.sculptObj = None
        # print([m.name for m in self.arr_mesh_all])

    def invoke(self, context, event):
        if context.space_data.type == 'VIEW_3D':
            self.sculptObj = None
            self.clearObjs()
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            self.report({'WARNING'}, "Active space must be a View3d")
            return {'CANCELLED'}


class AndreyPanel(bpy.types.Panel):
    bl_label = "Sculpt extrude"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'TOOLS'
    bl_context = "objectmode"

    def draw(self, context):
        config = bpy.context.scene.sculptextrude_manager
        act = context.active_object
        layout = self.layout
        layout.label(text="Draw Clones:")
        layout.operator("view3d.modal_operator_sculpt_extrude", text="DrawClones")
        layout.prop(config, "dist", text='dist')
        layout.prop(config, "user_view", text="User view")
        # print(act, act.type)

        if act and act.type == "META":
            meta = bpy.data.metaballs[0]
            layout.prop(meta, "resolution")


class ScExProperties(bpy.types.PropertyGroup):
    dist = bpy.props.FloatProperty(
        default=0.5,
        min=0.001,
        max=500.0,
        name='dist'
    )

    user_view = bpy.props.BoolProperty(name='user_view', default=False)


classes = [SculptExtrudeOperator, ScExProperties, AndreyPanel]


def register():
    for c in classes:
        bpy.utils.register_class(c)

    bpy.types.Scene.sculptextrude_manager = \
        bpy.props.PointerProperty(type=ScExProperties)


def unregister():
    del bpy.types.Scene.sculptextrude_manager
    for c in reversed(classes):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()
    if 'bpy' in locals():
        from importlib import reload
        import booltronoperator

        reload(booltronoperator)
        del reload
        bpy.utils.register_class(booltronoperator.UNION)
