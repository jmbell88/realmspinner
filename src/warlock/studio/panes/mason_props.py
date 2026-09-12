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
    placed = mscene.resolved_for(doc, node.uid)
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
        widgets.muted(f"instance of '{node.template}'")
    elif kind == "terrain":
        widgets.muted("the document's one terrain node")
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
    if changed:
        doc.set_props(node.uid, intensity=max(0.0, intensity))
    if node.kind == "spot":
        widgets.field_label("inner cone (rad)")
        changed, inner = controls.input_float(
            "##mlightinner", float(node.inner_cone_angle), 0.05, 0.0
        )
        if changed:
            doc.set_props(node.uid, inner_cone_angle=max(0.0, inner))
        widgets.field_label("outer cone (rad)")
        changed, outer = controls.input_float(
            "##mlightouter", float(node.outer_cone_angle), 0.05, 0.0
        )
        if changed:
            doc.set_props(node.uid, outer_cone_angle=max(0.0, outer))


def _camera_block(doc: Any, node: Any) -> None:
    widgets.field_label("vertical fov (rad)")
    changed, yfov = controls.input_float("##mcamfov", float(node.yfov), 0.05, 0.01)
    if changed:
        doc.set_props(node.uid, yfov=max(0.01, yfov))
    widgets.field_label("near / far")
    changed, near = controls.input_float("##mcamnear", float(node.znear), 0.01, 0.001)
    if changed:
        doc.set_props(node.uid, znear=max(0.001, near))
    changed, far = controls.input_float("##mcamfar", float(node.zfar), 1.0, 0.01)
    if changed:
        doc.set_props(node.uid, zfar=max(node.znear + 0.01, far))


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
