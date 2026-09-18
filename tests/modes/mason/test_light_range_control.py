"""Chapter 17: "Select one and look at the Properties panel: colour,
intensity, and range."

The 2026-09-12 audit's docs-06 found ``LightNode.range`` reaches glTF export
(``mason/gltfout.py``'s ``_light``) but ``modes/mason/ui/panes/props.py``'s
``_light_block`` drew no control for it at all -- a reader following the
chapter word for word would not find the field it names. Guarded to point and
spot, the two kinds ``KHR_lights_punctual`` gives a range at all
(``viewer/gltf.py``'s ``Light`` docstring: a directional light ignores it
entirely and never writes it).

Driven through a real (headless, no-GL) imgui context and the control census
in ``warlock.studio.probe`` -- ``tests/modes/mason/test_mason_panes.py``'s own house
pattern for "does this pane draw the control it claims to" -- rather than
reading the pane's source text, since the point of the claim is what actually
reaches the screen.
"""

from __future__ import annotations

from _ui_context import imgui_context

from warlock.studio import probe
from warlock.studio.modes.mason.engine import document as md
from warlock.studio.modes.mason.engine import nodes as nd
from warlock.studio.modes.mason.ui.panes import props as mason_props

RANGE_LABEL = "##mlightrange"


def _drawn_labels(monkeypatch, node: nd.LightNode) -> list[str]:
    doc = md.MasonDoc(roots=[node])
    with imgui_context(monkeypatch) as imgui:
        probe.begin_frame()
        imgui.new_frame()
        imgui.set_next_window_size((320.0, 900.0))
        imgui.begin("##host")
        mason_props._light_block(doc, node)
        imgui.end()
        imgui.end_frame()
        return [c.label for c in probe.FRAME_CONTROLS]


def test_the_light_properties_panel_exposes_a_range_control(monkeypatch) -> None:
    """A point light -- the common case, and the one the tutorial places --
    must draw the range field Chapter 17 promises."""
    node = nd.LightNode(uid=nd.new_uid(), name="Lamp", kind="point")

    labels = _drawn_labels(monkeypatch, node)

    assert RANGE_LABEL in labels


def test_a_spot_lights_range_control_sits_alongside_its_cone_angles(monkeypatch) -> None:
    """Range is not spot-only, but a spot must keep it -- both apply per
    ``KHR_lights_punctual``, and the guard must not have accidentally traded
    one for the other."""
    node = nd.LightNode(uid=nd.new_uid(), name="Spot", kind="spot")

    labels = _drawn_labels(monkeypatch, node)

    assert RANGE_LABEL in labels
    assert "##mlightinner" in labels
    assert "##mlightouter" in labels


def test_a_directional_lights_panel_draws_no_range_control(monkeypatch) -> None:
    """A directional light has no position that a range could be measured
    from -- ``viewer/gltf.py``'s ``Light`` docstring says this kind "ignores
    it entirely and never writes it" -- so the panel must not offer a control
    for a field the export silently drops."""
    node = nd.LightNode(uid=nd.new_uid(), name="Sun", kind="directional")

    labels = _drawn_labels(monkeypatch, node)

    assert RANGE_LABEL not in labels
