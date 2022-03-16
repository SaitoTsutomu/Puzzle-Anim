from collections import Counter
from dataclasses import dataclass
from typing import Any

import bpy
import numpy as np

bl_info = {
    "name": "Puzzle Anim",  # プラグイン名
    "author": "tsutomu",  # 制作者名
    "version": (1, 0),  # バージョン
    "blender": (3, 1, 0),  # 動作可能なBlenderバージョン
    "support": "COMMUNITY",  # サポートレベル
    "category": "3D View",  # カテゴリ名
    "description": "Puzzle Anim",  # 説明文
    "location": "3D View: Puzzle Anim",  # 機能の位置付け
    "warning": "",  # 注意点やバグ情報
    "doc_url": "https://github.com/SaitoTsutomu/Puzzle-Anim",  # ドキュメントURL
}


def kmeans(data, ngroup, seed=0, center_ini=None, just=False):
    """K-Meansによるグループ化

    :param data: データ
    :param ngroup: グループ数
    :param seed: 乱数シード, defaults to 0
    :param center_ini: 初期中心, defaults to None
    :param just: ちょうどにするか, defaults to False
    :return: 各データのグループ番号
    """
    data = np.array(data)
    n, ndim = data.shape
    assert ngroup <= n
    rnd = np.random.default_rng(seed=seed)
    for _ in range(16):
        if center_ini is None:
            group = np.zeros(n)
            while np.unique(group).size < ngroup:
                group = rnd.choice(ngroup, n)
        else:
            group = np.array(
                [np.nanargmin(np.sum((center_ini - data[i]) ** 2, 1)) for i in range(n)]
            )
            center_ini = None
        for _ in range(32):
            center = [data[group == i].mean(0) for i in range(ngroup)]
            new = [np.nanargmin(np.sum((center - data[i]) ** 2, 1)) for i in range(n)]
            if just and np.unique(new).size < ngroup:
                break
            if (group == new).all():
                just = False
                break
            group = np.array(new)
        if not just:
            break
    return group


@dataclass
class Group:
    """連（同じ種類の並び）"""

    typ: int  # 種類
    objs: list[Any]  # 下から連続するオブジェクト

    def size(self):
        return len(self.objs)

    def __str__(self):
        return str(self.typ) * self.size()


def to_groups(items: list[Any], mtids: list[int]):
    """連に分割"""
    n = len(items)
    last = n - 1
    for i in range(n - 1, -1, -1):
        if i < n - 1 and mtids[i] != mtids[i + 1]:
            yield Group(mtids[i + 1], items[last:i:-1])
            last = i
    if n:
        yield Group(mtids[0], items[last::-1])


@dataclass
class Column:
    """列（1本の串に相当）"""

    pos: int  # 棒番号
    groups: list[Group]  # 下からの連のリスト
    space: int  # 空き

    def __init__(self, pos: int, items: list[Any], mat2idx):
        self.pos = pos
        mtids = [mat2idx[item.material_slots[0].material] for item in items]
        self.groups = list(to_groups(items, mtids))
        self.space = 4 - sum(g.size() for g in self.groups)

    def __str__(self):
        return "".join(map(str, self.groups)) + " " * self.space

    def ok(self):
        return self.space == 4 or (self.space == 0 and self.groups[0].size() == 4)

    def can_push(self, group: Group):
        sp = self.space >= group.size()
        return sp and (not self.groups or self.groups[-1].typ == group.typ)

    def push(self, column: "Column"):
        group = column.groups.pop()
        column.space += group.size()
        if not self.groups:
            self.groups.append(group)
        else:
            self.groups[-1].objs.extend(group.objs)
        self.space -= group.size()
        return group

    def restore(self, column: "Column", group: Group):
        if self.groups[-1].size() == group.size():
            self.groups.pop()
        else:
            del self.groups[-1].objs[-group.size() :]  # noqa E203
        self.space += group.size()
        column.groups.append(group)
        column.space -= group.size()


@dataclass
class Table:
    columns: list[Column]

    def __init__(self, plit):
        mat2idx = {}
        for items in plit:
            for item in items:
                mat = item.material_slots[0].material
                if mat not in mat2idx:
                    mat2idx[mat] = len(mat2idx)
        self.columns = [Column(pos, list(items), mat2idx) for pos, items in enumerate(plit)]
        c = Counter("".join(str(g) for c in self.columns for g in c.groups))
        assert all(n == 4 for _, n in c.items())

    def __hash__(self):
        """To be managed by set."""
        return hash(" ".join(sorted(map(str, self.columns))))

    def __str__(self):
        return "\n".join(map(str, self.columns))


def solve_logic(table: Table, cache: set[Table], result: list[Any]):
    if all(column.ok() for column in table.columns):
        return result
    for column1 in table.columns:
        if column1.ok():
            continue
        for column2 in table.columns:
            if column1 is column2 or not column2.can_push(column1.groups[-1]):
                continue
            group = column2.push(column1)
            if table not in cache:
                cache.add(table)
                if r := solve_logic(table, cache, result + [(column1.pos, column2.pos)]):
                    return r
            column2.restore(column1, group)


def negaz(obj):
    return -obj.location[2]


def posix(obj):
    return obj.location[0]


def make_move(objs):
    heights = [[obj.bound_box[1][2] - obj.bound_box[0][2]] for obj in objs]
    try:
        group1 = kmeans(heights, 2, just=True)
    except AssertionError:
        return "Create sticks and items.", None
    stcks = objs[group1 == (group1.sum() * 2 < group1.size)]
    stcks = sorted(stcks, key=posix)
    items = objs[group1 != (group1.sum() * 2 < group1.size)]
    locp = np.array([stck.location for stck in stcks])
    loc = np.array([item.location for item in items])
    nstcks = len(stcks)
    try:
        group2 = kmeans(loc[:, :2], nstcks, center_ini=locp[:, :2])
    except AssertionError:
        return "Create items.", None
    plit = [sorted(items[group2 == i], key=negaz) for i in range(nstcks)]
    try:
        table = Table(plit)
    except AssertionError:
        return "Create 4 items of each type.", None
    mov = solve_logic(table, set(), [])
    return ("" if mov else "Could not create."), (stcks, Table(plit), mov)


class CMA_OT_sample(bpy.types.Operator):
    bl_idname = "object.sample_operator"
    bl_label = "Make sample"

    def execute(self, context):
        if "puzzle" not in bpy.data.collections:
            col = bpy.data.collections.new("puzzle")
            bpy.context.scene.collection.children.link(col)
        col = bpy.data.collections["puzzle"]
        lc = bpy.context.view_layer.layer_collection.children["puzzle"]
        bpy.context.view_layer.active_layer_collection = lc
        mats = [None] * 4
        cols = [(0.8, 0.7, 0), (0.1, 0.8, 0.1), (0.8, 0.1, 0.1)]
        for i, col in enumerate(cols):
            mats[i] = bpy.data.materials.get(f"mat{i}") or bpy.data.materials.new(name=f"mat{i}")
            mats[i].diffuse_color = *col, 1
        for i in range(3):
            bpy.ops.mesh.primitive_cylinder_add(scale=(0.05, 0.05, 1.2))
            obj = bpy.context.selected_objects[0]
            obj.location = i * 1.5, 0, 0.1
            obj.active_material = mats[0]
        for i in range(4):
            bpy.ops.mesh.primitive_ico_sphere_add(scale=(0.25, 0.25, 0.25))
            obj = bpy.context.selected_objects[0]
            obj.location = (i % 3 == 0) * 1.5, 0, i * 0.5 - 0.7
            obj.active_material = mats[1]
        for i in range(4):
            bpy.ops.mesh.primitive_cone_add(scale=(0.25, 0.25, 0.25))
            obj = bpy.context.selected_objects[0]
            obj.location = (i % 3 != 0) * 1.5, 0, i * 0.5 - 0.7
            obj.active_material = mats[2]
        return {"FINISHED"}


class CMA_OT_make(bpy.types.Operator):
    bl_idname = "object.make_animation_operator"
    bl_label = "Make animation"

    def execute(self, context):
        clc = bpy.data.collections.get("puzzle")
        if not clc:
            self.report({"WARNING"}, "There is no collection named 'puzzle'.")
            return {"CANCELLED"}
        objs = np.array(clc.objects)
        bpy.ops.object.select_all(action="DESELECT")
        for obj in objs:
            obj.select_set(state=True)
        bpy.ops.object.transform_apply(location=False)
        # bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY")
        bpy.ops.object.select_all(action="DESELECT")
        msg, result = make_move(objs)
        if msg:
            self.report({"WARNING"}, msg)
            return {"CANCELLED"}
        tm = 1
        for obj in objs:
            if obj.animation_data:
                obj.animation_data.action = None
            obj.keyframe_insert(data_path="location", frame=tm)
        stcks, table, mov = result
        stdf = (stcks[0].bound_box[1][2] - stcks[0].bound_box[0][2]) / 4.8
        stbs = stcks[0].bound_box[0][2] + stdf
        tm += 20
        for pos1, pos2 in mov:
            column1 = table.columns[pos1]
            column2 = table.columns[pos2]
            stck = stcks[pos2]
            group = column1.groups[-1]
            for i, obj in enumerate(group.objs):
                pr = obj.location[2]
                nw = stbs + stdf * (i + 4 - column2.space)
                obj.keyframe_insert(data_path="location", frame=tm)
                mid = ((obj.location + stck.location) / 2)[:2]
                obj.location = *mid, (pr + nw) / 2 + stdf * 2
                obj.keyframe_insert(data_path="location", frame=tm + 20 // 2)
                obj.location = *stck.location[:2], nw
                obj.keyframe_insert(data_path="location", frame=tm + 20)
            tm += 20
            column2.push(column1)
        # アニメーション開始
        bpy.context.scene.frame_end = tm + 20 * 3
        bpy.context.scene.frame_set(1)
        bpy.ops.screen.animation_cancel()
        bpy.ops.screen.animation_play()
        return {"FINISHED"}


class CMA_PT_make(bpy.types.Panel):
    bl_label = "Puzzle Anim"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Edit"

    def draw(self, context):
        self.layout.operator(CMA_OT_sample.bl_idname, text=CMA_OT_sample.bl_label)
        self.layout.operator(CMA_OT_make.bl_idname, text=CMA_OT_make.bl_label)


ui_classes = (
    CMA_OT_sample,
    CMA_OT_make,
    CMA_PT_make,
)


def register():
    for ui_class in ui_classes:
        bpy.utils.register_class(ui_class)


def unregister():
    for ui_class in ui_classes:
        bpy.utils.unregister_class(ui_class)


if __name__ == "__main__":
    register()
