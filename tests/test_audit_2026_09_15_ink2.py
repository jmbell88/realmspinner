"""Regression tests for the 2026-09-15 audit's inker-05, inker-06, inker-07
and inker-09 findings (the "ink2" fixer brief).

Each test's name is the claim; each failed against the code as the audit
found it before the fix beside it landed.
"""

from __future__ import annotations

import ast
import dataclasses
from types import SimpleNamespace
from typing import Any

import numpy as np

from warlock.studio import inker, inker_mode, inker_open, inker_ops, inker_state
from warlock.studio.inker import filters
from warlock.studio.inker.flourish import bake as B
from warlock.studio.inker.flourish import engines, presets
from warlock.studio.state import AppState

# --- shared harness: runs a submitted task inline, exactly test_flourish_ops.py's own ------


class _FlourishCtx:
    def __init__(self) -> None:
        self.state = SimpleNamespace(inker=inker_state.InkerState())
        self.toasts: list[tuple[str, str]] = []
        self.tasks = SimpleNamespace(set_progress=lambda *a, **k: None)

    def toast(self, text: str, level: str = "info", **_: Any) -> None:
        self.toasts.append((text, level))

    def busy(self, key: str) -> bool:
        return False

    def progress(self, key: str):
        return None

    def submit(self, key: str, fn: Any, *args: Any, **kwargs: Any) -> bool:
        from warlock.studio.tasks import Done

        try:
            done = Done(key=key, result=fn(*args, **kwargs))
        except Exception as exc:  # noqa: BLE001 -- the runner reports, never raises
            done = Done(key=key, error=exc)
        inker_mode.on_task_done(self, done)
        return True


def _open_with_effect(ctx: _FlourishCtx, size: tuple[int, int] = (32, 32)) -> Any:
    tab = inker_state.InkerDoc(doc=inker.Document.blank(*size))
    ctx.state.inker.docs.append(tab)
    ctx.state.inker.active_uid = tab.uid
    rec = dataclasses.replace(
        presets.load("sword_impact"), width=size[0], height=size[1], supersample=2
    )
    tab.doc.insert_flourish(B.bake(rec))
    return tab


# --- inker-05: Detach / Use selection as texture while busy ------------------


def test_flourish_detach_and_texture_selection_stay_disabled_while_the_document_is_busy():
    """``has_effect`` (Detach's ``enabled``) and ``can_texture_selection``
    omitted the ``not busy`` every sibling predicate here carries
    (``can_regenerate``, ``can_prompt``, ``can_restyle``...), and neither
    handler checked it either -- so both stayed clickable, and pressing
    either while a save or an export was writing the document mutated
    ``state``/the document underneath it.
    """
    ctx = _FlourishCtx()
    tab = _open_with_effect(ctx)
    (group,) = tab.doc.flourish
    # ``sword_impact``'s last layer ("Glow") has no ``texture`` parameter at
    # all, which would grey the button for an unrelated reason (inker-06,
    # 2026-09-14) -- pin the inspector at "Sparks" (kind "particles"), which
    # has one, and give the document a selection, so what disables the
    # button below is only ever the busy question this test is about. Found
    # by kind rather than a hard-coded uid: layer uids are allocated from a
    # running counter, so the literal number depends on what else has opened
    # a Flourish document earlier in the same process.
    recipe = tab.doc.flourish[group].recipe
    sparks_uid = next(layer.uid for layer in recipe.layers if layer.kind == "particles")
    ctx.state.inker.flourish_layer[group] = sparks_uid
    tab.doc.mask = SimpleNamespace(bounds=(0, 0, 1, 1))

    detach_op = inker_ops.get("flourish_detach")
    texture_op = inker_ops.get("flourish_texture_selection")

    # Not busy: both are live, as they always were.
    assert detach_op.enabled(ctx.state.inker, tab)
    assert texture_op.enabled(ctx.state.inker, tab)

    tab.saving = True
    assert tab.busy

    assert not detach_op.enabled(ctx.state.inker, tab), "Detach stayed enabled while busy"
    assert not texture_op.enabled(ctx.state.inker, tab), (
        "Use selection as texture stayed enabled while busy"
    )

    groups_before = dict(tab.doc.flourish)
    pending_before = dict(ctx.state.inker.flourish_pending)

    assert inker_mode.flourish_detach(ctx, tab) is False
    assert tab.doc.flourish == groups_before, "Detach ran on a busy document"

    assert inker_mode.flourish_texture_selection(ctx, tab) is False
    assert ctx.state.inker.flourish_pending == pending_before, (
        "Use selection as texture pushed a pending edit onto a busy document"
    )


# --- inker-06: engine snippet header ------------------------------------------


def test_a_newline_in_the_effect_name_does_not_splice_code_into_any_engine_snippet():
    """Every engine's header comment embedded the raw effect name with no
    sanitizing; the 2026-09-14 audit's inker-10 escaped only Godot's quoted
    string literals, so a newline in the name still closed the header
    comment early and spliced whatever followed in as live code, in every
    engine -- Godot's own header included, since it read the raw name rather
    than the already-escaped one right beside it.
    """
    base = dict(
        image="sheet.png", frame_width=32, frame_height=32, frames=4, fps=12,
        loop=False, origin=(16, 16),
    )
    clean = engines.describe(name="Evil Spell", **base)
    injected = engines.describe(name='Evil\nSPLICED = "gotcha"', **base)

    for engine in engines.ENGINES:
        clean_text = engines.snippet(engine, clean)
        injected_text = engines.snippet(engine, injected)
        # A newline that reached the output is a line the template did not
        # put there -- the line count must not move just because the name
        # carried one.
        assert injected_text.count("\n") == clean_text.count("\n"), engine
        # And the smuggled statement must never appear as a line of its own:
        # only ever folded into the (now space-joined) header line.
        assert "\nSPLICED" not in injected_text, engine
        assert '\n"gotcha"' not in injected_text, engine

    # Before the fix, ``SPLICED = "gotcha"`` landed as a real top-level
    # statement in the Pygame snippet and this parsed too -- parsing alone
    # does not prove safety, the two asserts above do; this just confirms
    # the fixed snippet is still ordinary, valid Python.
    tree = ast.parse(engines.snippet("pygame-ce", injected))
    assert not any(
        isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "SPLICED" for t in node.targets)
        for node in ast.walk(tree)
    ), "the injected name became a real assignment in the parsed module"


# --- inker-07: Defringe / Matte grow step ceiling -----------------------------


def _rim_canvas(size: int = 25) -> np.ndarray:
    """One opaque source pixel in a corner and one half-transparent pixel far
    enough away (Chebyshev distance 20) that reaching it needs more than
    ``DEFRINGE_MAX``/``MATTE_GROW_MAX`` steps of 8-neighbour dilation --
    exactly what makes an uncapped step count observably different from a
    capped one, and a timing-free way to prove the cap actually runs.
    """
    out = np.zeros((size, size, 4), dtype=np.uint8)
    out[0, 0] = (10, 20, 30, 255)
    out[20, 20] = (0, 0, 0, 128)
    return out


def test_defringe_and_matte_grow_cap_their_step_count_independent_of_the_slider():
    """``fringe``/``grow`` became step counts with no internal ceiling
    (unlike :func:`filters.blur`, :func:`filters.despeckle`,
    :func:`filters.outline`), so a value past the slider (``RANGES`` tops out
    at 8) drove that many full-canvas passes -- the 2026-09-15 audit measured
    2048x2048 at 250s for a value of 3000. Capped, a call with a value far
    past the slider must behave exactly like a call at the cap itself.
    """
    assert filters.DEFRINGE_MAX == 8
    assert filters.MATTE_GROW_MAX == 8

    rim = _rim_canvas()
    capped = filters.defringe(rim, fringe=float(filters.DEFRINGE_MAX))
    over = filters.defringe(rim, fringe=3000.0)
    assert np.array_equal(over, capped), "defringe(fringe=3000) outran the cap"
    # The far rim pixel is unreachable within the cap (distance 20 > 8), so a
    # correct cap leaves it exactly as it started -- an uncapped call would
    # have coloured it in, which is what made ``over`` differ from ``capped``
    # before the fix.
    assert tuple(capped[20, 20, :3]) == (0, 0, 0)

    blob = np.zeros((25, 25, 4), dtype=np.uint8)
    blob[2:23, 2:23] = (10, 20, 30, 255)

    grown_capped = filters.matte_grow(rim, grow=float(filters.MATTE_GROW_MAX))
    grown_over = filters.matte_grow(rim, grow=3000.0)
    assert np.array_equal(grown_over, grown_capped), "matte_grow(grow=3000) outran the cap"

    eroded_capped = filters.matte_grow(blob, grow=-float(filters.MATTE_GROW_MAX))
    eroded_over = filters.matte_grow(blob, grow=-3000.0)
    assert np.array_equal(eroded_over, eroded_capped), "matte_grow(grow=-3000) outran the cap"
    # A blob wider than twice the cap cannot be eroded away entirely by a
    # capped call -- an uncapped one would have eaten it down to nothing.
    assert eroded_capped[..., 3].any()


# --- inker-09: Open in Inker pressed twice ------------------------------------


class _OpenCtx:
    """``submit`` that always refuses, the way a duplicate in-flight key
    does -- ``tests/test_inker_audit_2026_09_14.py``'s own harness for this
    exact family of doors."""

    def __init__(self, state: inker_state.InkerState) -> None:
        self.state = AppState()
        self.state.inker = state
        self.settings = None  # unused: ``ensure`` short-circuits on ``state``
        self.svc = object()
        self.submitted: list[str] = []
        self.toasts: list[tuple[str, str]] = []

    def submit(self, key: str, fn: Any, *args: Any, **kwargs: Any) -> bool:
        self.submitted.append(key)
        return False

    def toast(self, text: str, level: str = "info", **_kw: Any) -> None:
        self.toasts.append((text, level))


def test_open_job_reference_second_press_while_loading_toasts_instead_of_silently_dropping():
    """``open_job_reference`` ignored ``ctx.submit``'s return, so a second
    press while the first decode was still running did nothing a user could
    see -- inker-12's class (2026-09-14 audit), which every other opener in
    this module was fixed for, except this one (2026-09-15 audit, inker-09).
    """
    ctx = _OpenCtx(inker_state.InkerState())
    inker_open.open_job_reference(ctx, {"id": "job1"})
    assert ctx.submitted == ["inker-open:job1"]
    assert ctx.toasts, "a second press while job1 was still opening was silent"
