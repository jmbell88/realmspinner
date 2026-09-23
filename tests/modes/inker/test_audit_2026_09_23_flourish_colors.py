"""Three findings from the 2026-09-23 audit: two in the Flourish keyword
mapper (``kernels/pixel/flourish/keywords.py``), one in the Colour pane's
palette-cap doors (``studio/modes/inker/ui/panes/colors.py``).

inker-03: "brighter"/"stronger"/"dimmer"/"softer" scale every parameter
literally named ``strength``, including ``distortion``'s -- which is a warp
displacement in pixels, not a light level.

inker-06: a colour word aimed at ``distortion`` ("green shimmer") fell all
the way through to "No words I know" because ``distortion`` was in neither
``_COLOUR_SLOTS`` nor ``_NO_COLOUR_SLOT``.

inker-04: the palette pane's "+ from colour" and ramp "Insert" call
``doc.add_slot``/``doc.insert_ramp`` with no ``try`` around them, so a click
at or near the 256-colour cap raises ``ValueError`` out of the pane's draw
call instead of saying the palette is full.
"""

from __future__ import annotations

from types import SimpleNamespace

from realmspinner.kernels import pixel as inker
from realmspinner.kernels.pixel.flourish import keywords
from realmspinner.kernels.pixel.flourish import recipe as R
from realmspinner.studio.modes.inker.ui.panes import colors as inker_colors

# --- inker-03: "brighter" must not also scale distortion's warp -------------


def test_brighter_does_not_also_scale_a_distortion_layers_warp_strength():
    """Reproduced with ``inker-flourish-01.py``: distortion strength 3.0 ->
    3.9 while the toast said only "brighter: x1.3", because ``strength`` is
    in ``_BRIGHT_PARAMS`` and distortion happens to have a param of that
    name -- a warp displacement in pixels, not a light level."""
    rec = R.Recipe(
        layers=(
            R.Layer(uid=1, kind="core", params={}),
            R.Layer(uid=2, kind="distortion", params={}),
        )
    )
    rec = R.clamp(rec)
    before_intensity = rec.layer(1).params["intensity"]
    before_strength = rec.layer(2).params["strength"]

    changed, notes = keywords.apply(rec, "brighter")

    after_intensity = changed.layer(1).params["intensity"]
    after_strength = changed.layer(2).params["strength"]
    assert after_intensity != before_intensity, "the core's brightness must still change"
    assert after_strength == before_strength, (
        "distortion's warp strength is not a brightness param and must be untouched"
    )
    assert any("brighter" in n for n in notes)


# --- inker-06: a colour word naming distortion --------------------------------


def test_a_colour_word_naming_distortion_reports_it_has_no_colour_rather_than_no_words_known():
    """"green shimmer" names both a known colour word and a known kind word
    (``shimmer`` -> distortion, see ``_KIND_WORDS``), but before the fix
    ``distortion`` was missing from ``_NO_COLOUR_SLOT``, so the clause fell
    through identically to a genuinely unrecognised word."""
    rec = R.Recipe(
        layers=(
            R.Layer(uid=1, kind="core", params={}),
            R.Layer(uid=2, kind="distortion", params={}),
        )
    )
    rec = R.clamp(rec)

    _changed, notes = keywords.apply(rec, "green shimmer")

    assert any("distortion has no colour" in n for n in notes)
    assert not any("No words I know" in n for n in notes)


def test_every_primitive_kind_has_a_colour_slot_or_is_listed_as_colourless():
    """The gate the fix adds beside ``_NO_COLOUR_SLOT``: every kind in
    ``prims.KINDS`` must be reachable through ``_COLOUR_SLOTS`` or
    ``_NO_COLOUR_SLOT`` -- the shape of gap that produced inker-06 for
    ``smoke`` (2026-09-18) and again for ``distortion`` (2026-09-23)."""
    from realmspinner.kernels.pixel.flourish import prims

    assert set(keywords._COLOUR_SLOTS) | keywords._NO_COLOUR_SLOT == set(prims.KINDS)


# --- inker-04: the palette-cap doors must refuse, not raise ------------------


class _Sayer:
    def __init__(self) -> None:
        self.said: list[str] = []

    def say(self, text: str) -> None:
        self.said.append(text)


def _full_indexed_doc(count: int) -> inker.Document:
    doc = inker.Document.blank(8, 8)
    palette = [(i % 256, (i * 3) % 256, (i * 7) % 256, 255) for i in range(count)]
    doc.convert_to_indexed(palette)
    return doc


def test_add_slot_and_insert_ramp_refuse_with_a_toast_at_the_palette_cap_instead_of_raising():
    """Reproduced with ``inker-panes-01.py``/``-02.py``: at 256 colours,
    ``doc.add_slot`` raises ``ValueError`` straight out of the unguarded
    call in ``colors.py``'s "+ from colour" button; near the cap,
    ``doc.insert_ramp`` does the same from the ramp "Insert" button. Both
    doors must instead behave like ``remove_slot_or_say`` (2026-09-13,
    inker-06): catch, toast, return False."""
    doc = _full_indexed_doc(256)
    sayer = _Sayer()
    state = SimpleNamespace(say=sayer.say)

    result = inker_colors.add_slot_or_say(state, doc, (250, 1, 2, 255))

    assert result is False
    assert len(sayer.said) == 1
    assert "256" in sayer.said[0]
    assert len(doc.palette) == 256

    doc2 = _full_indexed_doc(250)
    sayer2 = _Sayer()
    state2 = SimpleNamespace(say=sayer2.say)
    ctx = SimpleNamespace(toast=lambda *_a, **_k: None)

    result2 = inker_colors.insert_ramp_or_say(ctx, state2, doc2, 0, 249, 16)

    assert result2 is False
    assert len(sayer2.said) == 1
    assert "256" in sayer2.said[0]
    assert len(doc2.palette) == 250


def test_add_slot_or_say_still_adds_when_there_is_room():
    doc = inker.Document.blank(8, 8)
    doc.convert_to_indexed([(0, 0, 0, 255), (255, 255, 255, 255)])
    sayer = _Sayer()
    state = SimpleNamespace(say=sayer.say)

    result = inker_colors.add_slot_or_say(state, doc, (10, 20, 30, 255))

    assert result is True
    assert sayer.said == []
    assert (10, 20, 30, 255) in doc.palette


def test_insert_ramp_or_say_still_inserts_when_there_is_room():
    doc = inker.Document.blank(8, 8)
    doc.convert_to_indexed([(0, 0, 0, 255), (255, 255, 255, 255)])
    sayer = _Sayer()
    state = SimpleNamespace(say=sayer.say)
    before = len(doc.palette)
    ctx = SimpleNamespace(toast=lambda *_a, **_k: None)

    result = inker_colors.insert_ramp_or_say(ctx, state, doc, 0, 1, 3)

    assert result is True
    assert sayer.said == []
    assert len(doc.palette) > before
