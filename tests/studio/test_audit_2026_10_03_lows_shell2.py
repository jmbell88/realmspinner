"""The 2026-10-03 audit's Low findings owned by fixer ``shell2``.

One file for the whole batch, so each test's name is the claim it pins and a
reader can find the finding from the name. Findings that are pure wording or
that live in the (gitignored) ledger carry no test here: nothing under
``tests/`` may read ``dev/``.
"""

from __future__ import annotations

import re
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from realmspinner.studio import layout_skeleton as skeleton
from realmspinner.studio import layouts

ROOT = Path(__file__).resolve().parents[2]


def _manual(chapter: str) -> str:
    """A manual chapter with hand-wrapped line breaks collapsed to spaces."""
    path = ROOT / "docs" / "manual" / f"{chapter}.md"
    return re.sub(r"\s+", " ", path.read_text(encoding="utf-8"))


class _Settings:
    def __init__(self, data=None):
        self.data = dict(data or {})

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value


# --- shell-41: a request document must not undo the flat loop's coercion ------


def test_copying_settings_from_a_junk_generation_request_keeps_numeric_fields_numeric():
    """``GenerationRequest.from_dict`` leaves an unconvertible scalar unchanged
    (so ``validate_request`` can refuse it by type), and the request pass copied
    that over the flat loop's already-coerced fields: ``form["seed"] ==
    "banana"`` in a numeric field. It also wrote ``sprite_candidates``, a key
    ``default_form_2d`` does not declare -- hidden state no control shows."""
    from realmspinner.studio.state import default_form_2d, form_from_params

    form = form_from_params(
        {
            "generation_request": {
                "generation_type": "sprite_sheet",
                "prompt": "a rogue",
                "seed": "banana",
                "count": "lots",
                "lora_weight": "heavy",
                "sprite": {"candidate_count": 2, "target_cell_px": 48},
            }
        }
    )

    assert isinstance(form["seed"], int) and not isinstance(form["seed"], bool)
    assert isinstance(form["count"], int)
    assert isinstance(form["lora_weight"], float)
    assert form["prompt"] == "a rogue"
    assert set(form) <= set(default_form_2d()), "the form is the allowlist"


def test_a_good_generation_request_still_restores_its_numbers():
    from realmspinner.studio.state import form_from_params

    form = form_from_params(
        {"generation_request": {"seed": 7, "count": 3, "lora_weight": 0.5, "prompt": "x"}}
    )

    assert (form["seed"], form["count"], form["lora_weight"]) == (7, 3, 0.5)


# --- shell-42: a stored column with a repeat draws one pane twice ------------


def test_reconcile_drops_a_duplicated_id_in_a_stored_column():
    assert skeleton.reconcile(["a", "b"], ["a", "a", "b"]) == ["a", "b"]
    # The first place wins, so a hand edit that repeats an id keeps the
    # position the user most plausibly meant.
    assert skeleton.reconcile(["a", "b", "c"], ["b", "a", "b", "c", "a"]) == ["b", "a", "c"]


# --- shell-43: an unrelated edit forgets a pane that was not drawn ------------


def test_a_drag_while_a_conditional_pane_is_absent_keeps_its_saved_place(monkeypatch):
    """``_commit`` wrote each column from the panes live at that moment, so a
    pane absent through ``Slot.when`` lost its saved position and came back
    wherever ``reconcile`` re-inserted it."""
    from realmspinner.studio import layout as layout_mod
    from realmspinner.studio import layout_edit, skeletons

    shown = {"on": False}
    slots = (
        skeleton.Slot(id="a", label="A", draw=lambda ctx: None),
        skeleton.Slot(id="b", label="B", draw=lambda ctx: None),
        skeleton.Slot(id="cond", label="Cond", draw=lambda ctx: None, when=lambda ctx: shown["on"]),
    )
    column = skeleton.Column("left", slots)
    monkeypatch.setattr(skeletons, "for_mode", lambda ctx, mode: {"left": column})

    library = layouts.Library(_Settings())
    # The user kept the conditional pane *first*, which is not its built-in
    # place: only a place that differs from it can tell a kept position from a
    # reset one.
    library.record("inker", {"left": ["cond", "a", "b"]}, set())
    app = SimpleNamespace(layouts=library)
    ctx = SimpleNamespace(state=SimpleNamespace(mode="inker"))
    edit = layout_edit.ensure(ctx.state)
    edit.hidden = set()
    edit.dragging = "b"
    edit.dragging_column = "left"
    monkeypatch.setattr(
        layout_mod,
        "FRAME_PANES",
        {"a": (0.0, 0.0, 200.0, 100.0), "b": (0.0, 100.0, 200.0, 100.0)},
    )

    # b dropped above a, while ``cond`` is not showing.
    layout_edit._commit(app, ctx, {"left": column}, edit, SimpleNamespace(x=10.0, y=5.0))

    shown["on"] = True
    assert library.order("inker", "left", ["a", "b", "cond"]) == ["cond", "b", "a"]


# --- shell-46: the window's X pressed repeatedly queues one confirm ----------


def test_a_second_window_close_while_the_quit_confirm_is_up_does_not_queue_another():
    from realmspinner.studio import dialogs
    from realmspinner.studio import main as main_mod

    app = main_mod.App.__new__(main_mod.App)
    app.runtime = SimpleNamespace(current_job_id=None)
    queue = dialogs.ConfirmQueue()
    app.app_ctx = SimpleNamespace(
        cache=SimpleNamespace(active=None),
        tasks=SimpleNamespace(
            busy_keys={"clay-exportfile:tab-1"}, commit_busy=lambda prefix: False
        ),
        confirms=queue,
        toast=lambda *a, **k: None,
    )

    app._ask_quit()
    app._ask_quit()
    app._ask_quit()

    assert queue.pending is not None and queue.pending.title == "Quit Realmspinner?"
    assert queue.waiting == 0, "answering Stay once must settle the question"

    # And once it is answered, the next press asks again.
    queue.dismiss()
    app._ask_quit()
    assert queue.pending is not None


# --- shell-47: status copy measured on every surface and on its own wash ------

# The shortfalls measured on 2026-10-03, each a real (palette, role, surface)
# where the 12 px status copy falls under 4.5:1. Fixing one is a palette change
# (an art-direction decision, TODO), and ``xfail(strict=True)`` is what makes
# that fix remove its row here rather than leave a stale allowance behind.
_KNOWN_STATUS_SHORT = {
    ("dark", "ERR", "BG", "wash"),
    ("dark", "ERR", "PANEL", "wash"),
    ("dark", "ERR", "ELEV_1", "text"),
    ("dark", "ERR", "ELEV_1", "wash"),
    ("dark", "ERR", "ELEV_2", "text"),
    ("dark", "ERR", "ELEV_2", "wash"),
    ("dark", "ACCENT", "BG", "wash"),
    ("dark", "ACCENT", "PANEL", "text"),
    ("dark", "ACCENT", "PANEL", "wash"),
    ("dark", "ACCENT", "ELEV_1", "text"),
    ("dark", "ACCENT", "ELEV_1", "wash"),
    ("dark", "ACCENT", "ELEV_2", "text"),
    ("dark", "ACCENT", "ELEV_2", "wash"),
    ("light", "OK", "BG", "wash"),
    ("light", "OK", "PANEL", "wash"),
    ("light", "OK", "ELEV_1", "text"),
    ("light", "OK", "ELEV_1", "wash"),
    ("light", "OK", "ELEV_2", "text"),
    ("light", "OK", "ELEV_2", "wash"),
    ("light", "ERR", "BG", "wash"),
    ("light", "ERR", "PANEL", "wash"),
    ("light", "ERR", "ELEV_1", "wash"),
    ("light", "ERR", "ELEV_2", "text"),
    ("light", "ERR", "ELEV_2", "wash"),
    ("light", "WARN", "BG", "wash"),
    ("light", "WARN", "PANEL", "wash"),
    ("light", "WARN", "ELEV_1", "text"),
    ("light", "WARN", "ELEV_1", "wash"),
    ("light", "WARN", "ELEV_2", "text"),
    ("light", "WARN", "ELEV_2", "wash"),
    ("light", "ACCENT", "ELEV_2", "wash"),
    ("pixel", "ERR", "PANEL", "wash"),
    ("pixel", "ERR", "ELEV_1", "wash"),
    ("pixel", "ERR", "ELEV_2", "text"),
    ("pixel", "ERR", "ELEV_2", "wash"),
    ("pixel", "ACCENT", "ELEV_2", "wash"),
    # Familiar's chat bubbles (restored 2026-10-06) sit at ELEV_1/ELEV_2's tonal level,
    # so they inherit exactly the shortfalls those surfaces already record above.
    # Same palette decision, same strict xfail: fixing the palette flips these too.
    ("dark", "ERR", "BUBBLE_USER", "text"),
    ("dark", "ERR", "BUBBLE_USER", "wash"),
    ("dark", "ERR", "BUBBLE_ASSISTANT", "text"),
    ("dark", "ERR", "BUBBLE_ASSISTANT", "wash"),
    ("dark", "ACCENT", "BUBBLE_USER", "text"),
    ("dark", "ACCENT", "BUBBLE_USER", "wash"),
    ("dark", "ACCENT", "BUBBLE_ASSISTANT", "text"),
    ("dark", "ACCENT", "BUBBLE_ASSISTANT", "wash"),
    ("light", "OK", "BUBBLE_USER", "text"),
    ("light", "OK", "BUBBLE_USER", "wash"),
    ("light", "OK", "BUBBLE_ASSISTANT", "text"),
    ("light", "OK", "BUBBLE_ASSISTANT", "wash"),
    ("light", "ERR", "BUBBLE_USER", "wash"),
    ("light", "ERR", "BUBBLE_ASSISTANT", "wash"),
    ("light", "WARN", "BUBBLE_USER", "text"),
    ("light", "WARN", "BUBBLE_USER", "wash"),
    ("light", "WARN", "BUBBLE_ASSISTANT", "text"),
    ("light", "WARN", "BUBBLE_ASSISTANT", "wash"),
    ("light", "ACCENT", "BUBBLE_USER", "wash"),
    ("pixel", "ERR", "BUBBLE_USER", "wash"),
    ("pixel", "ERR", "BUBBLE_ASSISTANT", "wash"),
    ("pixel", "ACCENT", "BUBBLE_USER", "wash"),
}


def _status_cases():
    from realmspinner.studio import tokens

    cases = []
    for palette in sorted(tokens.PALETTES):
        for role in ("OK", "ERR", "WARN", "ACCENT"):
            for surface in tokens.COPY_SURFACES:
                for kind in ("text", "wash"):
                    key = (palette, role, surface, kind)
                    marks = (
                        [pytest.mark.xfail(strict=True, reason="palette decision, see TODO")]
                        if key in _KNOWN_STATUS_SHORT
                        else []
                    )
                    cases.append(pytest.param(*key, marks=marks, id="-".join(key)))
    return cases


@pytest.mark.parametrize(("palette", "role", "surface", "kind"), _status_cases())
def test_status_copy_is_readable_on_every_surface_and_on_its_own_wash(
    palette: str, role: str, surface: str, kind: str
) -> None:
    """``test_status_colours_are_readable`` measured PANEL only, but status copy
    is drawn on cards (ELEV_1, ELEV_2 on hover), on section blocks and on
    ``status_pill``'s own 0.16 wash of its colour."""
    from realmspinner.studio import tokens

    colours = tokens.PALETTES[palette]
    ground = colours[surface]
    if kind == "wash":
        ground = tokens.composite(colours[role], ground, 0.16)
    ratio = tokens.contrast(colours[role], ground)
    assert ratio >= tokens.CONTRAST_TEXT, f"{palette}/{role} on {surface} {kind}: {ratio:.2f}:1"


# --- shell-50: fit_text measures once per trimmed character ------------------


def test_fit_text_measures_a_long_name_a_bounded_number_of_times(monkeypatch):
    from realmspinner.studio import widgets

    calls = []

    def measure(text):
        calls.append(text)
        return SimpleNamespace(x=7.0 * len(text), y=14.0)

    monkeypatch.setattr(widgets.imgui, "calc_text_size", measure)

    text = "a long prompt that goes on " * 40
    out = widgets.fit_text(text, 136.0)

    assert out.endswith("-")
    assert 7.0 * len(out) <= 136.0
    assert len(out) == int(136.0 // 7.0), "the longest prefix that fits, plus the dash"
    assert len(calls) <= 16, f"{len(calls)} measurements for {len(text)} characters"


def test_fit_text_agrees_with_trimming_one_character_at_a_time(monkeypatch):
    from realmspinner.studio import widgets

    monkeypatch.setattr(
        widgets.imgui,
        "calc_text_size",
        lambda text: SimpleNamespace(x=sum(3.0 if ch == "i" else 8.0 for ch in text), y=14.0),
    )

    def linear(text, width):
        def size(s):
            return sum(3.0 if ch == "i" else 8.0 for ch in s)

        if size(text) <= width:
            return text
        trimmed = text
        while trimmed and size(trimmed + "-") > width:
            trimmed = trimmed[:-1]
        return trimmed + "-"

    for text in ("short", "iiiiiiiiiiiiiiiiiiiiii", "WWWWWWWWWWWWWWWWWWWW", "mixed iiWW iWiW text"):
        for width in (0.0, 5.0, 8.0, 40.0, 100.0, 1000.0):
            assert widgets.fit_text(text, width) == linear(text, width), (text, width)


# --- shell-51: a heading inside a child must not close the pane's block ------


def test_a_section_inside_a_child_window_does_not_move_the_outer_blocks_edge(monkeypatch):
    from _ui_context import imgui_context

    from realmspinner.studio import widgets

    with imgui_context(monkeypatch) as imgui:
        imgui.new_frame()
        imgui.set_next_window_size((400.0, 800.0))
        imgui.begin("##shell51")
        with widgets.section_blocks() as outer:
            widgets.section("Outer")
            widgets.muted("row")
            imgui.begin_child("##shell51-child", (0.0, 200.0))
            widgets.section("Inside the child")
            painted_in_child = len(outer.blocks)
            still_open = outer.start is not None
            widgets.end_section()
            ended_in_child = len(outer.blocks)
            imgui.end_child()
        imgui.end()
        imgui.end_frame()

    assert painted_in_child == 0, "the child's heading closed the outer block"
    assert still_open, "and the outer block is still the pane's"
    assert ended_in_child == 0, "end_section inside a child is the child's, not the pane's"
    assert len(outer.blocks) == 1


# --- shell-62: manual 21 on Not now and on how an asset row opens ------------


def test_manual_21_describes_not_now_as_per_tour():
    text = _manual("21-home")

    assert "puts it away for good" not in text
    assert "puts that tour away" in text
    assert "2D for a reference or a tile and 3D for anything else" not in text
    assert "Reference stage" in text and "Mesh stage" in text


# --- shell-64: every axis the sweep form offers has help and is counted -------


_NUMBER_WORDS = {
    19: "nineteen",
    20: "twenty",
    21: "twenty-one",
    22: "twenty-two",
    23: "twenty-three",
    24: "twenty-four",
    25: "twenty-five",
    26: "twenty-six",
}


def test_every_axis_the_sweep_form_offers_has_help_and_is_counted_in_the_manual():
    from realmspinner.service import sweeps
    from realmspinner.studio.modes.review.mode import AXIS_HELP

    offered = set(sweeps.axis_params())
    assert offered <= set(AXIS_HELP), sorted(offered - set(AXIS_HELP))
    assert all(len(AXIS_HELP[axis]) > 30 for axis in offered)

    text = _manual("37-review")
    assert f"There are {_NUMBER_WORDS[len(offered)]}," in text, len(offered)
    for phrase in ("Base model", "Platform", "Style LoRA", "IP-Adapter", "Control"):
        assert phrase in text, phrase


# --- shell-65: the score ordering lands when the scores do -------------------


def _review_ctx(job_ids):
    from realmspinner.studio.modes.review import mode as review_mode
    from realmspinner.studio.state import AppState

    ctx = SimpleNamespace(state=AppState())
    ctx.state.review = None
    state = review_mode.ensure(ctx)
    state.sweeps = [
        {
            "id": "s1",
            "units": [
                {"job_id": job_id, "verdict": None, "status": "done"} for job_id in job_ids
            ],
        }
    ]
    return ctx, state


def test_a_sweep_opened_before_its_scores_land_is_ordered_best_first_once_they_do():
    from realmspinner.studio.modes.review import mode as review_mode

    ctx, state = _review_ctx(["j0", "j1", "j2"])
    review_mode.open_sweep(ctx, "s1")
    assert [u["job_id"] for u in state.units] == ["j0", "j1", "j2"], "no scores yet"
    watching = review_mode.current(state)

    state.score_request = ["j0", "j1", "j2"]  # what ``pump_scores`` records
    review_mode.adopt_scores(state, {"j0": 0.1, "j1": 0.9, "j2": 0.5})

    assert [u["job_id"] for u in state.units] == ["j1", "j2", "j0"]
    assert review_mode.current(state) is watching, "the cursor stays on the unit on screen"


def test_scores_that_land_after_the_reviewer_moved_on_reorder_nothing():
    """``adopt_scores``' standing promise: a list that resorts under the cursor
    is how the wrong thing gets judged. The one reorder is for a reviewer who
    has not yet moved off the unit the sweep opened on."""
    from realmspinner.studio.modes.review import mode as review_mode

    ctx, state = _review_ctx(["j0", "j1", "j2"])
    review_mode.open_sweep(ctx, "s1")
    review_mode.step(state, 1)

    state.score_request = ["j0", "j1", "j2"]
    review_mode.adopt_scores(state, {"j0": 0.1, "j1": 0.9, "j2": 0.5})

    assert [u["job_id"] for u in state.units] == ["j0", "j1", "j2"]


# --- shell-66: labels on two questions each queue their own retrain ----------


def test_labels_on_two_questions_each_queue_their_own_retrain(monkeypatch):
    from realmspinner.studio.modes.review import mode as review_mode
    from realmspinner.studio.state import AppState

    monkeypatch.setattr(review_mode.verdicts_mod, "record_verdict", lambda *a, **k: None)
    submitted = []
    ctx = SimpleNamespace(
        state=AppState(),
        svc=None,
        toast=lambda *a, **k: None,
        submit=lambda key, run, *args, **kw: submitted.append(args[-1]) or True,
    )
    ctx.state.review = None
    state = review_mode.ensure(ctx)
    for stage in ("reference", "blank"):
        state.labels = review_mode.LabelPass(
            stage=stage, rows=[{"job_id": f"{stage}-1", "verdict": None}]
        )
        assert review_mode.record_label(ctx, "accept")

    review_mode.pump_judge(ctx)
    review_mode.pump_judge(ctx)
    review_mode.pump_judge(ctx)

    assert sorted(submitted) == ["blank", "reference"]
    assert not ctx.state.judge_dirty


# --- shell-73: the quit chain covers every document mode ---------------------


def test_every_doc_mode_has_a_guard_in_the_quit_chain(monkeypatch):
    """``_request_quit``'s guards were a hand list of eight beside a manifest
    that says which modes are documents; a mode added to ``DOC_MODES`` but not
    to the tuple would be journalled and counted in the caption but never asked
    about on quit."""
    from realmspinner.studio import main as main_mod
    from realmspinner.studio import mode_manifest

    asked = []
    fake = types.ModuleType("realmspinner.studio.modes.fakedoc.mode")
    fake.guard = lambda ctx, why, proceed: (asked.append("fakedoc"), proceed())
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    monkeypatch.setattr(
        mode_manifest,
        "DOC_MODES",
        (
            *mode_manifest.DOC_MODES,
            mode_manifest.ModeManifest("fakedoc", "modes.fakedoc.mode", "fakedoc", "", None),
        ),
    )
    for entry in mode_manifest.DOC_MODES[:-1]:
        module = mode_manifest.module_of(entry)
        monkeypatch.setattr(
            module, "guard", lambda ctx, why, proceed, key=entry.key: (asked.append(key), proceed())
        )
    from realmspinner.studio.panes import pose_panel

    monkeypatch.setattr(
        pose_panel, "guard", lambda ctx, why, proceed: (asked.append("pose-panel"), proceed())
    )

    app = main_mod.App.__new__(main_mod.App)
    app.app_ctx = SimpleNamespace()
    app._running = True
    app._request_quit()

    assert app._running is False, "the chain ran to the end"
    # Manifest order, with the inspector's pose guard just before Poser's own.
    expected = []
    for entry in mode_manifest.DOC_MODES:
        if entry.key == "poser":
            expected.append("pose-panel")
        expected.append(entry.key)
    assert asked == expected
    assert "fakedoc" in asked


# --- shell-74: a colon-less save key still warns on quit ---------------------


def test_quit_summary_warns_while_a_poser_clips_save_is_busy():
    from realmspinner.studio import main as main_mod
    from realmspinner.studio.modes.poser import mode as poser_mode

    app = main_mod.App.__new__(main_mod.App)
    app.runtime = SimpleNamespace(current_job_id=None)
    for key in (
        poser_mode.SAVE_KEY,
        poser_mode.CLIPS_SAVE_KEY,
        poser_mode.RENAME_KEY,
        poser_mode.DUPLICATE_KEY,
    ):
        app.app_ctx = SimpleNamespace(
            cache=SimpleNamespace(active=None),
            tasks=SimpleNamespace(busy_keys={key}),
        )
        assert app._quit_summary() == "An export is still being written.", key


def test_a_read_only_poser_key_does_not_raise_the_export_warning():
    from realmspinner.studio import main as main_mod
    from realmspinner.studio.modes.poser import mode as poser_mode

    app = main_mod.App.__new__(main_mod.App)
    app.runtime = SimpleNamespace(current_job_id=None)
    app.app_ctx = SimpleNamespace(
        cache=SimpleNamespace(active=None),
        tasks=SimpleNamespace(busy_keys={poser_mode.LIST_KEY, poser_mode.CLIPS_KEY}),
    )
    assert app._quit_summary() == ""


# --- shell-77: layout.pane opens no section scope; panes opt in --------------


def test_the_section_block_comment_names_the_opt_in_rule_not_layout_pane():
    import inspect

    from realmspinner.studio import layout, widgets

    assert "section_blocks" not in inspect.getsource(layout.pane)
    # Comment lines are wrapped by hand, so flatten the line breaks and the
    # comment markers before matching a phrase.
    source = re.sub(r"\s*\n#?\s*", " ", inspect.getsource(widgets))
    assert "safe to hang off ``layout.pane``" not in source
    assert "A pane opts in." in source


# --- shell-78: one step count for the button and the popover -----------------


def test_the_history_button_and_the_popover_report_the_same_step_count(monkeypatch):
    from realmspinner.core.undo import UndoStack
    from realmspinner.studio import widgets

    class _Edit:
        cost = 1

        def undo(self, doc):  # pragma: no cover - the stack only moves heads here
            pass

        def redo(self, doc):  # pragma: no cover
            pass

    stack = UndoStack()
    for _ in range(8):
        stack._done.append(_Edit())
    for _ in range(3):
        stack._undone.append(stack._done.pop())
    assert len(stack) == 5 and len(stack.history()) == 8

    labels = []
    monkeypatch.setattr(widgets, "disabled_button", lambda *a, **k: False)
    monkeypatch.setattr(widgets.imgui, "same_line", lambda *a, **k: None)
    monkeypatch.setattr(widgets, "muted", lambda text, *a, **k: labels.append(text))
    monkeypatch.setattr(widgets, "grid_width", lambda *a, **k: 100.0)
    tab = SimpleNamespace(doc=SimpleNamespace(history=stack), busy=False)

    widgets.history_block(ctx=None, tab=tab, key="t", undo=lambda: None, redo=lambda: None)

    assert labels == ["8 step(s)"], "the button counted done steps only"
