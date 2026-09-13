"""The selected node: its local transform, its world transform beside it,
and what it is.

**Local, editable; world, read-only, off ``scene.resolved_for``.** That
pairing is the entire answer to Clay's argument against parenting a mesh
document -- ``clay/document.py``'s own docstring calls a hierarchy "a
transform that is not the one you typed", and it is right about a *modeller*,
where every number a user enters should be the number that ends up on the
mesh. Mason is not that: a scene is an arrangement of many placed things, some
of them nested under groups and prefab instances on purpose, and the number a
user typed into "position" is correctly a *local* one -- the world figure is
what the resolver, the same one the viewport draws from, says the whole
ancestry adds up to. Showing both rather than picking one is what lets a user
who moved a group ask "where did that prop actually end up" without doing the
matrix multiply themselves, and it is computed through ``scene.resolved_for``
rather than a second walk here, so this panel can never disagree with what is
on screen.

The rest follows ``clay_props``'s shape: identity fields, then transform,
then a per-kind block keyed by ``gltfout.kind_of`` (the one naming rule this
package asks of a caller, restated in that module's own docstring), then the
``properties`` table that reaches the engine manifest untouched.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from .. import controls, icons, mason_mode, tokens, widgets
from ..manual import render as manual_render
from ..mason import gltfout
from ..mason import scene as mscene
from ..tokens import sp
from ..viewer import math3d as m3


def draw(ctx: Any) -> None:
    with widgets.section_blocks():
        _body(ctx)


def _body(ctx: Any) -> None:
    state = mason_mode.ensure(ctx)
    tab = mason_mode.active(ctx)
    widgets.section("Properties")
    manual_render.help_button(ctx, "mason-props")
    if tab is None:
        return
    doc = tab.doc
    node = _selected(doc)
    if node is None:
        count = len(doc.selection)
        if count > 1:
            widgets.empty_state(icons.BOX, f"{count} nodes selected", "Select one to edit it.")
        else:
            widgets.empty_state(icons.BOX, "Nothing selected", "Click a node in the outliner.")
        return

    imgui.begin_disabled(tab.saving)
    _identity(doc, node)
    imgui.dummy((0, sp(tokens.SP_2)))
    _transform(doc, node)
    imgui.dummy((0, sp(tokens.SP_2)))
    _world_transform(doc, node)
    imgui.dummy((0, sp(tokens.SP_2)))
    _kind_block(doc, node)
    imgui.dummy((0, sp(tokens.SP_2)))
    _properties(doc, node)
    imgui.end_disabled()
    del state


def _selected(doc: Any) -> Any:
    """The one selected node, or ``None`` -- ``clay_props._selected``'s
    reasoning, verbatim: a panel that silently edited whichever node happened
    to sort first under a multi-selection would be worse than one that says
    it cannot."""
    if len(doc.selection) != 1:
        return None
    return doc.node(next(iter(doc.selection)))


def _identity(doc: Any, node: Any) -> None:
    name = widgets.input_text("name##masonname", node.name, max_length=120, commit=True)
    if name != node.name:
        doc.set_props(node.uid, name=name)
    changed, value = widgets.toggle(f"{icons.EYE} Visible", node.visible, tag=str(node.uid))
    if changed:
        doc.set_props(node.uid, visible=value)
    lock_icon = icons.LOCK if node.locked else icons.LOCK_OPEN
    changed, value = widgets.toggle(f"{lock_icon} Locked", node.locked, tag=f"lock{node.uid}")
    if changed:
        doc.set_props(node.uid, locked=value)
    widgets.help_marker(
        "A lock stops a drag in the viewport, not an edit here -- "
        "reported by the resolver, never enforced by the document, so an "
        "undo can always put back what was there before the lock was set."
    )
    changed, value = widgets.toggle("Static", node.static, tag=f"static{node.uid}")
    if changed:
        doc.set_props(node.uid, static=value)
    widgets.help_marker("Reaches the engine manifest: a static node skips runtime updates.")


def _transform(doc: Any, node: Any) -> None:
    widgets.field_label("local transform")
    was = tuple(v.copy() for v in node.trs())
    changed = False
    edited, translation = controls.input_float3("position##mt", list(node.translation))
    controls.fold_undo(doc.history)
    changed |= edited
    edited, scale = controls.input_float3("scale##ms", list(node.scale))
    controls.fold_undo(doc.history)
    changed |= edited
    edited, rotation = controls.input_float4("rotation##mr", list(node.rotation))
    controls.fold_undo(doc.history)
    widgets.help_marker("A quaternion, XYZW -- the gizmo is the way to set one by eye.")
    changed |= edited
    if changed:
        doc.set_transform(
            node.uid, translation=translation, rotation=rotation, scale=scale, was=was
        )


def _world_transform(doc: Any, node: Any) -> None:
    """The same node's transform after its whole ancestry has had its say --
    see the module docstring for why this is the resolver's answer, read
    only, and never a second computation of its own."""
    widgets.field_label("world transform")
    try:
        placed = mscene.resolved_for(doc, node.uid)
    except ValueError:
        # mason-03, the 2026-09-13 audit: ``resolved_for`` walks the whole
        # document looking for one uid, so a document past ``MAX_PLACED``
        # raises the same refusal ``resolve`` always has -- caught here
        # rather than left to crash the frame thread, since a properties
        # panel asking "where did this end up" is not the caller that should
        # be the one to discover a corrupt/absurd document.
        placed = None
    if placed is None:
        # ``muted_wrapped`` and not ``muted``: this is a sentence rather than
        # a status line, and ``muted`` does not wrap in a 300 dp sidebar --
        # the pairing the refreshed screenshots found cutting four of
        # Settings' sentences off mid-word with no ellipsis and no scrollbar.
        widgets.muted_wrapped(
            "Not currently drawable: a hidden ancestor, or a broken prefab path."
        )
        return
    translation, rotation, scale = m3.decompose(placed.world)
    widgets.muted(
        f"position  {translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}"
    )
    widgets.muted(f"scale     {scale[0]:.3f}, {scale[1]:.3f}, {scale[2]:.3f}")
    widgets.muted(
        f"rotation  {rotation[0]:.3f}, {rotation[1]:.3f}, {rotation[2]:.3f}, {rotation[3]:.3f}"
    )
    if placed.prefab:
        widgets.muted(f"reached through prefab '{placed.prefab}'")


def _kind_block(doc: Any, node: Any) -> None:
    kind = gltfout.kind_of(node)
    widgets.field_label(kind)
    if kind == "mesh":
        _mesh_block(doc, node)
    elif kind == "light":
        _light_block(doc, node)
    elif kind == "camera":
        _camera_block(doc, node)
    elif kind == "prefab":
        _prefab_block(doc, node)
    elif kind == "terrain":
        _terrain_block(doc)
    else:
        widgets.muted("a group -- holds other nodes, nothing of its own")


def _mesh_block(doc: Any, node: Any) -> None:
    if node.ref is None:
        widgets.muted("no source yet")
        return
    widgets.muted(f"source: {node.ref}")
    if (node.uid, node.ref) in {(n.uid, r) for n, r in doc.missing_refs()}:
        widgets.secondary("Missing -- the source could not be resolved.")


def _light_block(doc: Any, node: Any) -> None:
    widgets.muted(f"kind: {node.kind}")
    changed, colour = controls.color_edit3("colour##mlight", list(node.color))
    controls.fold_undo(doc.history)
    if changed:
        doc.set_props(node.uid, color=tuple(float(c) for c in colour))
    widgets.field_label("intensity")
    changed, intensity = controls.input_float("##mlightintensity", float(node.intensity), 0.1, 0.0)
    # The 2026-09-13 audit's mason-01: this field called the undoable
    # ``set_props`` on every changed frame with no ``controls.fold_undo``,
    # so typing "2000" digit by digit (or a drag) pushed one step per
    # keystroke/report -- one Ctrl+Z left the value mid-edit rather than
    # undoing the whole gesture. Folded here as every other door does:
    # draw, fold, act.
    controls.fold_undo(doc.history)
    if changed:
        doc.set_props(node.uid, intensity=max(0.0, intensity))
    if node.kind in ("point", "spot"):
        # The 2026-09-12 audit's docs-06: Chapter 17 says a selected light's
        # Properties panel shows "colour, intensity, and range," but this
        # block drew no control for it even though ``LightNode.range`` exists
        # and reaches glTF export (``gltfout._light``) -- a reader looking for
        # the field the chapter names had no way to set it from this panel.
        # Guarded to point/spot, the two kinds ``KHR_lights_punctual`` gives a
        # range at all (``viewer.gltf.Light``'s own docstring: a directional
        # light ignores it entirely and never writes it).
        widgets.field_label("range (m)")
        changed, light_range = controls.input_float(
            "##mlightrange", float(node.range), 0.5, 0.0
        )
        widgets.help_marker("Distance the light reaches. 0 means no limit.")
        # mason-01: same unfolded door as intensity above.
        controls.fold_undo(doc.history)
        if changed:
            doc.set_props(node.uid, range=max(0.0, light_range))
    if node.kind == "spot":
        widgets.field_label("inner cone (rad)")
        changed, inner = controls.input_float(
            "##mlightinner", float(node.inner_cone_angle), 0.05, 0.0
        )
        # mason-01: same unfolded door as intensity above.
        controls.fold_undo(doc.history)
        if changed:
            doc.set_props(node.uid, inner_cone_angle=max(0.0, inner))
        widgets.field_label("outer cone (rad)")
        changed, outer = controls.input_float(
            "##mlightouter", float(node.outer_cone_angle), 0.05, 0.0
        )
        # mason-01: same unfolded door as intensity above.
        controls.fold_undo(doc.history)
        if changed:
            doc.set_props(node.uid, outer_cone_angle=max(0.0, outer))


def _camera_block(doc: Any, node: Any) -> None:
    widgets.field_label("vertical fov (rad)")
    changed, yfov = controls.input_float("##mcamfov", float(node.yfov), 0.05, 0.01)
    # mason-01: same unfolded door as the light block above.
    controls.fold_undo(doc.history)
    if changed:
        doc.set_props(node.uid, yfov=max(0.01, yfov))
    widgets.field_label("near / far")
    changed, near = controls.input_float("##mcamnear", float(node.znear), 0.01, 0.001)
    controls.fold_undo(doc.history)
    if changed:
        doc.set_props(node.uid, znear=max(0.001, near))
    changed, far = controls.input_float("##mcamfar", float(node.zfar), 1.0, 0.01)
    controls.fold_undo(doc.history)
    if changed:
        doc.set_props(node.uid, zfar=max(node.znear + 0.01, far))


def _prefab_block(doc: Any, node: Any) -> None:
    """Which template this instance follows, and whether that template exists.

    No per-child override editor, and that is the decision rather than a gap --
    ``document.unpack_instance``'s own docstring carries the argument. What this
    block can honestly say is the one thing a user needs from it: the name, and
    whether anything is behind the name, because an instance of a removed
    template is legal, resolves as dangling, and would otherwise be an empty
    space in the viewport with no explanation anywhere in the UI.
    """
    widgets.muted(f"instance of '{node.template}'")
    if node.template in doc.prefabs:
        widgets.muted("edit the template through any instance of it")
        return
    widgets.secondary("No template of that name -- this instance draws nothing.")


def _terrain_block(doc: Any) -> None:
    """The ground's extent, which belongs to the ``Terrain`` rather than to its
    node -- so it is set through ``set_terrain_config`` and not ``set_props``.

    The *resolution* is deliberately not editable here. Changing a height
    field's side means resampling every height, which is a different operation
    from configuring one (and a lossy one), and offering it as a spinbox beside
    two sizes would make it look like the three are the same kind of edit.
    """
    terrain = doc.terrain
    if terrain is None:
        # A terrain node with no height field behind it: reachable for one frame
        # mid-undo, and a real state in a hand-edited file.
        widgets.secondary("No height field -- this node draws nothing.")
        return
    widgets.muted(f"{terrain.side} x {terrain.side} cells")
    widgets.field_label("size x / z (m)")
    changed_x, size_x = controls.input_float("##mterrainx", float(terrain.size_x), 1.0, 0.01)
    controls.fold_undo(doc.history)
    changed_z, size_z = controls.input_float("##mterrainz", float(terrain.size_z), 1.0, 0.01)
    # mason-01: same unfolded door as the transform/light/camera fields --
    # ``set_terrain_config`` is the undoable write and it fired once per
    # changed frame with no fold.
    controls.fold_undo(doc.history)
    if changed_x or changed_z:
        # ``max`` and not a refusal: ``set_terrain_config`` raises on a
        # non-positive size, and a spinbox that can be dragged to zero must not
        # be able to raise out of a draw call.
        doc.set_terrain_config(size_x=max(0.01, size_x), size_z=max(0.01, size_z))


def _properties(doc: Any, node: Any) -> None:
    """The user ``properties`` table that reaches the engine manifest
    untouched -- name/value rows, added and removed here rather than typed
    into a generic dict editor, since a per-key widget for an open-ended
    string-keyed dict is a door this pane does not need to build twice."""
    widgets.field_label("properties")
    if not node.properties:
        widgets.muted("none")
    remove_key = None
    for key, value in sorted(node.properties.items()):
        imgui.text(f"{key}:")
        imgui.same_line()
        widgets.muted(str(value))
        imgui.same_line()
        if controls.small_button(f"{icons.TRASH}##mpropdel{key}"):
            remove_key = key
    if remove_key is not None:
        props = dict(node.properties)
        was = dict(node.properties)
        del props[remove_key]
        doc.set_props(node.uid, was={"properties": was}, properties=props)
