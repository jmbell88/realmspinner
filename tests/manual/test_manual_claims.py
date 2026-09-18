"""One chapter's claims, pinned against the code that makes them true or false.

Created for the 2026-09-08 audit's finding inker-03:
``docs/manual/28-inker.md``'s Zooming section promises "The wheel zooms in
steps of 5% (`Ctrl` and the wheel does the same)" with no qualification, but
above ``inker_state.FINE_ZOOM_MAX`` the wheel switches to walking
``ZOOM_LADDER`` rungs instead -- rungs far larger than 5% per notch (one
notch at 800% jumps straight to 1000%, a 200-percentage-point step). A user
zooming past 800% sees jumps the manual told them not to expect, with nothing
in the chapter to say why the increment changed.

``tests/manual/test_manual_promises.py`` already does this for four other
2026-09-06 findings (a chapter's claim read against the module that decides
it, rather than a second hand-written copy); this file is for the same shape
of pin, kept separate because that one belongs to a different fixer's file
list.
"""

from __future__ import annotations

import re
from pathlib import Path

MANUAL = Path(__file__).resolve().parents[2] / "docs" / "manual"


def _chapter(name: str) -> str:
    return (MANUAL / name).read_text(encoding="utf-8")


def _section(text: str, heading: str) -> str:
    """The prose under one ``##`` heading, up to the next ``##`` (or EOF)."""
    pattern = rf"^## {re.escape(heading)}\s*$"
    match = re.search(pattern, text, re.MULTILINE)
    assert match, f"no '## {heading}' heading found"
    rest = text[match.end():]
    nxt = re.search(r"^## ", rest, re.MULTILINE)
    return rest[: nxt.start()] if nxt else rest


def _wheel_paragraph(section: str) -> str:
    """The Zooming section's *first* paragraph -- the one about the wheel.

    Scoped this narrowly on purpose: the section's second paragraph (about
    `+`/`-`) already mentions "800%" as one rung of its own whole-scale
    ladder list, and the fifth uses the word "stops" for the zoom *floor* --
    either would make a whole-section keyword search pass without the wheel's
    own paragraph ever having been touched, which is exactly the gap finding
    inker-03 is about.
    """
    return section.strip().split("\n\n", 1)[0]


def test_manual_zooming_section_names_the_wheels_gear_change_above_800_percent():
    """The 2026-09-08 audit, finding inker-03.

    ``shell.paintview.FINE_ZOOM_MAX`` (800%) is where ``zoom_step`` switches the
    wheel from additive 5% notches to walking ``ZOOM_LADDER`` rungs -- the
    same jump size the `+`/`-` keys use everywhere. The chapter's *wheel*
    paragraph (not the section as a whole -- see ``_wheel_paragraph``) must
    name that ceiling and say the wheel's own step size changes there, the
    way the code comment on ``FINE_ZOOM_MAX`` already does ("5% of 1x is a
    meaningful step and 5% of 64x is a twentieth of a source pixel").
    """
    from warlock.studio.modes.inker import state as inker_state
    from warlock.studio.shell import paintview

    gear_change_pct = inker_state.zoom_key(paintview.FINE_ZOOM_MAX)
    wheel = _wheel_paragraph(_section(_chapter("28-inker.md"), "Zooming"))

    assert gear_change_pct in wheel, (
        f"docs/manual/28-inker.md's Zooming section's wheel paragraph never "
        f"names {gear_change_pct}% (shell.paintview.FINE_ZOOM_MAX), the zoom "
        "above which the wheel stops taking 5% notches and starts walking "
        "ZOOM_LADDER rungs instead"
    )
    gear_change_words = ("rung", "ladder", "jump", "no longer")
    assert any(word in wheel.lower() for word in gear_change_words), (
        "docs/manual/28-inker.md's Zooming section's wheel paragraph names "
        f"{gear_change_pct}% but does not say the wheel's own step size "
        "changes there -- a reader still expects 5% notches past it"
    )


#: Small numerals spelled out, the way this chapter and its neighbours write
#: them ("four minutes", not "4 minutes"). Covers the plausible range for a
#: sample ceiling stated in minutes; a value landing outside it is a sign the
#: constant moved to something this test should be widened to say, not that
#: the manual's prose style changed.
_NUMBER_WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
    7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven",
    12: "twelve", 15: "fifteen", 20: "twenty", 30: "thirty",
}


def test_sirens_manual_sample_ceiling_matches_max_sample_frames():
    """The 2026-09-08 audit, finding sirens-03.

    ``wavout.MAX_SAMPLE_FRAMES`` was raised from four minutes to ten
    (``48_000 * 600``) on 2026-09-07 ("Muse ... reaches ten minutes"), and
    that commit updated ``docs/manual/36-muse.md`` and
    ``docs/manual/16-generating-a-soundtrack.md`` but never touched
    ``docs/manual/35-sirens.md``'s own Samples section, which still promised
    "one sample runs to four minutes ... and past that the import is
    refused" -- telling a reader that a 5-9 minute Muse take would be refused
    on import when the code has accepted it since the day before.

    The number is checked spelled out ("ten minutes"), not as a digit: this
    chapter's prose spells small numbers ("four minutes" was the phrasing
    replaced), and a test that only accepted "10 minutes" would fail against
    the correct fix for no reason but its own choice of numeral style.
    """
    from warlock.kernels.audio import wavout

    seconds = wavout.MAX_SAMPLE_FRAMES / 48_000
    minutes = seconds / 60
    assert minutes == int(minutes), (
        f"wavout.MAX_SAMPLE_FRAMES ({wavout.MAX_SAMPLE_FRAMES}) is not a"
        " whole number of minutes at 48 kHz -- the manual sentence this"
        " test pins assumes it is"
    )
    minutes = int(minutes)
    word = _NUMBER_WORDS.get(minutes)
    assert word is not None, (
        f"{minutes} minutes has no spelled-out form in _NUMBER_WORDS -- widen"
        " the map (or confirm the chapter now uses digits) before trusting"
        " this pin"
    )
    # Line-wrapped in the source markdown ("... and one\nsample runs to ..."),
    # so whitespace is normalised before the substring check. Either spelling
    # of the number passes -- the chapter's own style (spelled out) and the
    # digit form both say the same true thing, and this pin cares which
    # *number* the chapter states, not which numeral style it is written in.
    normalized = re.sub(r"\s+", " ", _chapter("35-sirens.md"))
    phrasings = (
        f"one sample runs to {word} minutes",
        f"one sample runs to {minutes} minutes",
    )
    assert any(phrasing in normalized for phrasing in phrasings), (
        "docs/manual/35-sirens.md's Samples section does not say 'one sample"
        f" runs to {word} minutes' or '...{minutes} minutes', which is what"
        f" wavout.MAX_SAMPLE_FRAMES ({wavout.MAX_SAMPLE_FRAMES}) is"
    )


def _element_drag_paragraph(section: str) -> str:
    """The Transforming section's paragraph about element-mode gizmo drags.

    Scoped to the paragraph beginning "The gizmos work on elements too." --
    the object-mode drag paragraph two above it in the same section already
    says "one undo step" with no per-object qualifier, so a search over the
    whole section would pass whether or not *this* paragraph agreed with it.
    """
    marker = "The gizmos work on elements too."
    start = section.index(marker)
    return section[start:].strip().split("\n\n", 1)[0]


def test_manual_clay_chapter_does_not_claim_one_undo_step_per_object_for_element_drags():
    """The 2026-09-09 audit, finding clay-02.

    ``docs/manual/30-clay.md``'s Transforming section says an element-mode
    gizmo drag across several objects is "one undo step per object per
    drag". ``_view_drag.DragOps._commit_element_drag`` folds the whole
    multi-object gesture into a single history step instead
    (``history.mark()`` / ``history.collapse_since(mark)``), which is exactly
    what its own docstring and
    ``tests/test_clay_history.py::test_a_multiobject_element_drag_is_one_undo_step``
    both pin (three dragged objects, ``len(doc.history) == depth + 1``). A
    reader who follows the manual expects one Ctrl+Z per object; one Ctrl+Z
    already restores all of them.

    Keyed on the claim itself ("per object" inside the element-drag
    paragraph) rather than an exact quoted sentence, so a reword that keeps
    the false claim still fails this test and one that drops it still
    passes.
    """
    from types import SimpleNamespace

    import numpy as np

    from warlock.kernels.mesh import document as bd
    from warlock.kernels.mesh import elements as el
    from warlock.kernels.mesh import primitives as bp
    from warlock.studio.modes.clay.ui._view_drag import DragOps, _ElementDrag

    doc = bd.ClayDoc()
    uids = [
        doc.add_object(bd.Obj(uid=bd.new_uid(), name=f"Box{i}", mesh=bp.box())).uid
        for i in range(3)
    ]
    doc.select(uids)
    doc.element_mode = "vertex"
    drags = {}
    for uid in uids:
        obj = doc.by_uid(uid)
        doc.set_element_sel(uid, el.ElementSel(verts=[0, 1]))
        drags[uid] = _ElementDrag(
            before=obj.mesh,
            verts=np.array([0, 1]),
            local=obj.mesh.positions[[0, 1]].astype("f8"),
            matrix=np.eye(4),
            inverse=np.eye(4),
            preview=obj.mesh.positions.copy() + 1.0,
        )
    view = SimpleNamespace(_element_drags=drags, _cache={})
    depth = len(doc.history)
    DragOps._commit_element_drag(view, doc)
    steps_per_drag = len(doc.history) - depth

    paragraph = _element_drag_paragraph(_section(_chapter("30-clay.md"), "Transforming"))
    assert "per object" not in paragraph.lower(), (
        "docs/manual/30-clay.md's Transforming section still claims an "
        "element-mode gizmo drag across several objects is undone one "
        f"object at a time, but _commit_element_drag folds it into "
        f"{steps_per_drag} undo step regardless of how many objects it "
        "touched -- see tests/test_clay_history.py::"
        "test_a_multiobject_element_drag_is_one_undo_step"
    )
