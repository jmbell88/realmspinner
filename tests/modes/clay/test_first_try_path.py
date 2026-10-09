"""The first-try path: an empty Clay to an exported model, driven by real input.

A first-time user should get an object quickly, and "quickly" is a number of
gestures. This test drives the real Clay workspace -- the viewport pane with its
header, rail, palette strip and empty state, and Properties beside it -- through
imgui's own input queue (``_ui_driver``), the way a person would, and counts
the pointer presses and key presses it took:

    empty Clay -> Start with a box -> type a height into Size -> add a colour
    and click it -> Export (the toolbar's split button, whose main click is the
    library export).

It asserts what the person ends up with (a shape standing on the ground, selected,
inside the camera's view, added in one undo step, in the colour they clicked, and
a row in the library) and that no control was pressed while disabled.

**The gesture budget is a ratchet.** :data:`FIRST_TRY_BUDGET` is what the path
costs *today*, measured, and may only go down: a landing that makes the path
dearer fails here, and one that makes it cheaper lowers the number in the same
commit. The redesign this belongs to reaches 4 (Start with a box, Size, a colour,
Export) once the inspector and toolbar land.

Needs a GL context like the rest of the Clay viewport tests (skipped without one).
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from _ui_driver import UiDriver, app_context, gl_imgui

from realmspinner.kernels.mesh import ops as mesh_ops
from realmspinner.studio import icons, probe
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay import placement

#: Gestures (pointer presses + keys typed) the path costs today, measured:
#: Start with a box (1), click Size's H box (2), type the height (3), + on the
#: palette strip (4), click the new swatch (5), Export on the toolbar (6). It was
#: 7 while Export lived behind Properties' Document tab (a tab hop first).
#: **It may only decrease** -- raising it to fit a change is the one edit this
#: number refuses -- and a later landing brings it to 4 (the Size and Colour
#: cards take the + away and the height box's click).
FIRST_TRY_BUDGET = 6


@pytest.fixture
def workspace(gl, svc, tmp_path, monkeypatch):
    """The real Clay viewport pane and Properties, over a real ``App``."""
    from imgui_bundle import imgui

    from realmspinner.studio import widgets
    from realmspinner.studio.main import App
    from realmspinner.studio.modes.clay.ui.panes import props as clay_props

    with app_context(gl, svc, tmp_path) as app_ctx, gl_imgui(gl) as (_imgui, renderer):
        app = App(app_ctx.runtime)
        app.app_ctx = app_ctx
        app.ctx = gl
        # A fresh checkout has no recent documents, and the recents read the
        # settings store the real app has; nothing here is about them.
        monkeypatch.setattr(clay_mode, "recent_paths", lambda ctx: [])
        # The census is env-gated and sampled at import; the pane tests turn it on
        # the same way (``_ui_context.imgui_context``).
        monkeypatch.setattr(probe, "ENABLED", True)

        flags = (
            imgui.WindowFlags_.no_title_bar.value
            | imgui.WindowFlags_.no_move.value
            | imgui.WindowFlags_.no_resize.value
            | imgui.WindowFlags_.no_saved_settings.value
            | imgui.WindowFlags_.no_scrollbar.value
        )

        def draw():
            imgui.new_frame()
            imgui.set_next_window_pos((0, 0))
            imgui.set_next_window_size((1280, 900))
            imgui.begin("##centre", None, flags)
            app._clay_viewport(app_ctx, clay_mode, widgets)
            imgui.end()
            imgui.set_next_window_pos((1280, 0))
            imgui.set_next_window_size((300, 900))
            imgui.begin("##props", None, flags)
            clay_props.draw(app_ctx)
            imgui.end()
            imgui.render()
            renderer.render(imgui.get_draw_data())

        driver = UiDriver(imgui, draw)
        try:
            yield SimpleNamespace(
                imgui=imgui, app=app, ctx=app_ctx, driver=driver, svc=svc
            )
        finally:
            if app.clay_view is not None:
                app.clay_view.release()


def _wait_for_export(workspace, tab, seconds: float = 20.0) -> None:
    """Drain the task runner the way the frame loop does until the export lands."""
    app, ctx = workspace.app, workspace.ctx
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline and not tab.job_id:
        for done in ctx.tasks.poll():
            app._on_task_done(done)
        time.sleep(0.01)
        workspace.driver.frame()
    assert tab.job_id, "the export never landed"


def _in_view(workspace, obj, doc) -> bool:
    camera = workspace.app.clay_view.camera
    box = mesh_ops.world_box(obj, world=doc.world_matrix(obj.uid))
    view_proj = camera.projection() @ camera.view()
    return placement.in_frustum(box[0], box[1], view_proj, camera.position)


def _start_with_a_box(workspace):
    d = workspace.driver
    d.frame()
    d.settle()
    d.click(d.find(text="Start with a box"))
    d.settle(3)
    tab = clay_mode.active(workspace.ctx)
    return tab, tab.doc.objects[0]


def test_the_first_try_path_ends_with_a_standing_framed_coloured_model_in_the_library(workspace):
    d = workspace.driver
    tab, obj = _start_with_a_box(workspace)
    doc = tab.doc

    # What one press on "Start with a box" must already have given them.
    assert len(doc.objects) == 1 and doc.selection == {obj.uid}
    assert len(doc.history) == 1, "starting took more than one undo step"
    lowest = mesh_ops.world_box(obj, world=doc.world_matrix(obj.uid))[0][1]
    assert lowest == 0.0, "the first box is not standing on the ground"
    assert _in_view(workspace, obj, doc), "the first box is outside the camera's view"

    # A height, typed into the middle (H) box of Size.
    d.click(d.find(label="size##bz"))
    d.type_text("2")
    d.settle(1)
    assert obj.scale[1] == pytest.approx(2.0), "the typed height never reached the shape"

    # A colour: the strip's + adds a slot, a click on it paints the shape. The
    # first press after typing must land on its own target (a text field that
    # still holds focus swallowing it is the lost click this guards).
    d.click(d.find(label="##clay-palette/add"))
    d.settle(2)
    assert len(doc.materials) == 2, "the click straight after typing was lost"
    d.click(d.find(label="##palsw1"))
    d.settle(2)
    assert obj.material == 1 and set(obj.mesh.material.tolist()) == {1}

    # Into the library, from the toolbar: no tab to hop to first.
    d.click(d.find(label=f"{icons.DOWNLOAD} Export##clay-header/export"))
    d.settle(2)
    _wait_for_export(workspace, tab)
    assert workspace.svc.store.get(tab.job_id)["status"] == "done"

    assert d.disabled_presses == [], f"a disabled control was pressed: {d.disabled_presses}"
    assert d.gestures <= FIRST_TRY_BUDGET, (
        f"the first-try path now costs {d.gestures} gestures "
        f"({d.presses} presses, {d.keys} keys); the budget is {FIRST_TRY_BUDGET} and "
        "may only go down"
    )


def test_a_click_straight_after_typing_a_height_is_not_swallowed(workspace):
    """The mockup's lost-click bug: the first click after typing into a field only
    took focus away from it. Here the + must add its slot on that one press."""
    d = workspace.driver
    tab, obj = _start_with_a_box(workspace)
    d.click(d.find(label="size##bz"))
    d.type_text("3")
    before = len(tab.doc.materials)
    d.click(d.find(label="##clay-palette/add"))
    d.settle(1)
    assert len(tab.doc.materials) == before + 1
    assert obj.scale[1] == pytest.approx(3.0)


def test_start_with_a_box_is_the_primary_action_and_new_model_still_starts_empty(workspace):
    d = workspace.driver
    d.frame()
    d.settle()
    drawn = [c.text for c in d.controls if c.kind == "button"]
    assert drawn[:3] == ["Start with a box", "New model", "Open a file..."], drawn
    d.click(d.find(text="New model"))
    d.settle(2)
    tab = clay_mode.active(workspace.ctx)
    assert tab is not None and tab.doc.objects == [], "New model must stay an empty page"
    assert not workspace.ctx.state.clay.frame_pending
