"""The templates this scene defines, and how many times each one is placed.

**A conditional slot**: this pane is in the right column only while the document
actually has a template (``skeletons.mason``'s ``when``), and it is the first
Mason slot to use that mechanism. A permanently-empty panel in a three-panel
column costs the outliner and Properties the height it sits in, on every scene
that never authors a prefab -- which is most of them.

That is also why **"Make prefab" is not here.** A pane that only exists once a
template does cannot be where the first one is made; the gesture lives on the
selection instead, in the viewport's context menu and the outliner's row menu,
beside Group and Duplicate where the rest of the scene-shaping verbs are.

**Two mechanisms, not conflated** -- the plan's own instruction, and this pane is
where a user meets the difference. Sharing one GPU upload and one glTF mesh
between two placements of one asset is automatic and invisible; it falls out of
keying on the ref and nothing in the document says so. A *prefab* is authored: a
template subtree that is not in the scene tree, instanced by
:class:`~..mason.nodes.PrefabNode`\\ s that carry their own transform and **no
per-child overrides**. A template edit reaches every instance on the next frame
because the walk reads through ``doc.prefabs`` every time -- there is no
propagation step and no "apply to instances" button, because the propagation
step is what an editor gets wrong. The escape hatch for the one instance that
has to diverge is Unpack, which turns it into an ordinary independent subtree.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from .. import controls, icons, mason_mode, widgets
from ..manual import render as manual_render
from ..mason import nodes as nd
from ..tokens import sp


def has_prefab(ctx: Any) -> bool:
    """Whether the active scene defines a template -- this pane's own ``when``.

    Written here rather than as a lambda in ``skeletons.py`` for the reason
    ``plotter_objects.has_object_layer`` is: the predicate and the pane it
    governs belong to each other, and a lambda over there would be a second
    place that has to know what this pane is for.
    """
    tab = mason_mode.active(ctx)
    return tab is not None and bool(tab.doc.prefabs)


def draw(ctx: Any) -> None:
    with widgets.section_blocks():
        _body(ctx)


def _body(ctx: Any) -> None:
    state = mason_mode.ensure(ctx)
    tab = mason_mode.active(ctx)
    widgets.section("Prefabs")
    manual_render.help_button(ctx, "mason-prefabs")
    if tab is None:
        return
    doc = tab.doc
    if not doc.prefabs:
        # Reachable for one frame after the last template is removed, and by
        # anything that draws this pane directly: the slot's ``when`` is
        # evaluated before the column is laid out, not between these two lines.
        widgets.muted("No templates yet.")
        return

    imgui.begin_disabled(tab.saving)
    counts = _instance_counts(doc)
    for name in sorted(doc.prefabs):
        _row(ctx, state, doc, name, counts.get(name, 0))
    imgui.dummy((0, sp(8)))
    _unpack(ctx, doc)
    imgui.end_disabled()


def _instance_counts(doc: Any) -> dict[str, int]:
    """How many instances each template has, by name.

    Counted over ``all_nodes`` -- the *scene* tree -- so an instance that sits
    inside another template's subtree is deliberately not counted here: it is
    placed once per instance of that outer template, and a number that changed
    when the outer one was placed again would be answering a different question
    than the one a row asks ("how many of these are in my scene").
    """
    counts: dict[str, int] = {}
    for node in doc.all_nodes():
        if isinstance(node, nd.PrefabNode):
            counts[node.template] = counts.get(node.template, 0) + 1
    return counts


def _row(ctx: Any, state: Any, doc: Any, name: str, count: int) -> None:
    imgui.push_id(f"masonprefab{name}")
    armed = state.place_prefab == name
    width = widgets.grid_width(2)
    if controls.button(
        f"{icons.COPY} {name}##place",
        (width, sp(26)),
        selected=armed,
        tooltip="Arm it, then click in the viewport to place an instance",
    ):
        # Toggling off rather than only arming, so the button that armed it is
        # also the one that puts it down -- and ``place_kind`` is cleared in the
        # same breath, because two armed placements would make the next click
        # ambiguous and ``place_armed`` would silently prefer this one.
        state.place_prefab = "" if armed else name
        state.place_kind = ""
    imgui.same_line()
    widgets.muted(f"{count} placed")
    if controls.small_button(f"{icons.TRASH}##remove"):
        if armed:
            state.place_prefab = ""
        mason_mode.remove_prefab(ctx, name)
    if imgui.is_item_hovered():
        imgui.set_tooltip(
            "Drop the template. Instances already in the scene stay where they "
            "are and stop resolving until a template of that name comes back."
        )
    imgui.pop_id()
    del doc


def _unpack(ctx: Any, doc: Any) -> None:
    selected = [uid for uid in doc.selection if isinstance(doc.node(uid), nd.PrefabNode)]
    if widgets.disabled_button(
        f"{icons.UNLINK} Unpack instance##masonunpack",
        bool(selected),
        reason="Select a prefab instance in the scene first.",
    ):
        mason_mode.unpack_selected(ctx)
    widgets.help_marker(
        "Replace the selected instance with an independent copy of its "
        "template. The copy stops tracking the template -- which is the point: "
        "an instance carries no per-child overrides, and this is the one way "
        "out for the one that has to differ."
    )
