"""The full-window Library grid's own defects -- shell, not the sidebar.

:mod:`library` already carries the exhaustive query/sort/trash coverage
(``tests/modes/library/test_library_browsing.py``); this file is for the composition
``library_full.py`` adds on top of it, one real-imgui-frame at a time so a
defect in *where a click lands* is caught rather than only what a handler
does once called directly.
"""

from __future__ import annotations

from types import SimpleNamespace

from _ui_context import imgui_context

from warlock.studio.modes.library.ui.panes import full as library_full
from warlock.studio.modes.library.ui.panes import library


def _job(job_id: str) -> dict:
    return {
        "id": job_id,
        "kind": "text",
        "status": "done",
        # Not "reference": ``library.can_drag`` must stay false so
        # ``draggable_source`` does not push a second item onto the stack
        # this test would then have to account for.
        "stage": "model",
        "name": job_id,
        "prompt": "",
        "files": [],
        "params": {},
    }


def _ctx(jobs: dict[str, dict]):
    state = SimpleNamespace(
        selected=None,
        compare_baseline=None,
        library_scroll_to=None,
    )

    def select(job_id):
        state.selected = job_id
        state.compare_baseline = None

    state.select = select
    return SimpleNamespace(
        state=state,
        cache=SimpleNamespace(get=lambda jid: jobs.get(jid)),
        textures=None,
        job_dir=lambda jid: None,
    )


def test_library_full_grid_right_click_opens_the_menu_for_the_cell_under_the_cursor_not_the_last_one_drawn(  # noqa: E501
    monkeypatch,
):
    """shell finding, 2026-09-16 audit: ``library_full._cell`` called
    ``library._card_context`` *after* the cell's own ``widgets.card`` child
    window had already closed via ``end_child()``. ``_card_context``'s
    right-click test is ``imgui.is_window_hovered()``, and by the time it ran
    the *current* window was the shared scrolling grid again -- so a
    right-click anywhere in the grid's own area that is not a deeper child
    window (the gaps between cards, the margins, the space below the last
    row) registered as the grid being hovered on *every* cell's check that
    frame, and whichever job was processed last (drawn last) won the
    selection and the popup, regardless of where the pointer actually was.

    Two cells, "a" then "b", side by side. First: a right-click squarely in
    the gap *between* them -- under neither card's own picture -- must act on
    neither (proving the shared-grid false positive is gone). Second: a
    right-click on "b"'s own picture must act on "b" and not "a" (proving the
    right-click test now runs against the cell actually under the cursor,
    which only ``is_window_hovered()`` read from *inside* that cell's own
    still-open child window can tell apart from its neighbour -- imgui's
    default ``IsWindowHovered()`` returns false for a window whose child is
    the one actually hovered, which is what makes the gap case go quiet
    rather than pick a winner).
    """
    calls: list[str] = []

    jobs = {"a": _job("a"), "b": _job("b")}
    ctx = _ctx(jobs)
    ordered = [jobs["a"], jobs["b"]]

    with imgui_context(monkeypatch) as imgui:
        # A stub, not a no-op: ``_card_context`` calls ``_overflow`` every
        # frame regardless of whether the popup is open (``_overflow``'s own
        # ``imgui.begin_popup("more")`` is the guard) -- a no-op stub would
        # record a call every frame this runs at all, which is every frame,
        # and prove nothing. This keeps that same guard so a call is only
        # recorded on the frame the popup this job's cell opened is actually
        # showing.
        def stub_overflow(_ctx, job):
            if imgui.begin_popup("more"):
                calls.append(job["id"])
                imgui.end_popup()

        monkeypatch.setattr(library, "_overflow", stub_overflow)

        rects: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {}

        def build():
            imgui.begin_child("library-full/cells", (400.0, 200.0))
            pad = imgui.get_style().window_padding
            for i, job in enumerate(ordered):
                if i:
                    imgui.same_line()
                library_full._cell(ctx, job, (150.0, 150.0), 130.0, pad)
                mn = imgui.get_item_rect_min()
                mx = imgui.get_item_rect_max()
                rects[job["id"]] = ((mn.x, mn.y), (mx.x, mx.y))
            imgui.end_child()

        io = imgui.get_io()

        def frame(pos, *, right_down):
            io.add_mouse_pos_event(pos[0], pos[1])
            io.add_mouse_button_event(1, right_down)
            imgui.new_frame()
            imgui.set_next_window_pos((0.0, 0.0))
            imgui.set_next_window_size((500.0, 300.0))
            imgui.begin("##host")
            build()
            imgui.end()
            imgui.end_frame()

        # Two off-screen warm-up frames: the window rects are stable from the
        # first (sizes/positions are fixed, not mouse-dependent), but imgui's
        # hovered-window resolution is computed at ``new_frame()`` from the
        # *previous* frame's window stack, so a window's hover state is only
        # trustworthy once it has existed for a frame (``_ui_context``'s
        # neighbour ``test_context_controls.py`` hits the same thing with
        # popups and answers it the same way: lay out twice before reading).
        frame((-100.0, -100.0), right_down=False)
        frame((-100.0, -100.0), right_down=False)

        (a_min, a_max) = rects["a"]
        (b_min, b_max) = rects["b"]

        # Squarely between the two cards -- inside the grid's own scroll
        # area, over neither card's picture.
        gap = ((a_max[0] + b_min[0]) / 2, (a_min[1] + a_max[1]) / 2)
        frame(gap, right_down=True)
        assert calls == [], f"a click in the gap between cells acted on {calls}"
        assert ctx.state.selected is None, ctx.state.selected

        frame(gap, right_down=False)  # release, so the next press is a fresh one
        calls.clear()

        centre_b = ((b_min[0] + b_max[0]) / 2, (b_min[1] + b_max[1]) / 2)
        frame(centre_b, right_down=True)

    assert calls == ["b"], calls
    assert ctx.state.selected == "b", ctx.state.selected
