"""Regression tests for the 2026-10-14 audit's F17 slice (kernels/pixel).

Flourish: the diff validator's containers (inker-flourish-08), "lots of"
(-13), the scale-word loop (-14), seed 0 (-07), phase-name dedupe (-10), the
palette the pixel pass reads (-11), a layer limited to a dropped phase (-12)
and the painterly held-bytes ceiling (-18). Codecs: the GIF held-frame merge
(inker-codecs-14) and the lazy palette reader (inker-codecs-05). Document:
two isolated groups with equal extents (inker-document-01) and the stale
"v1 gap" docstrings (inker-document-12).
"""

from __future__ import annotations

import io
import tracemalloc
import zipfile
from pathlib import Path

import numpy as np
import pytest

from realmspinner.kernels.pixel import gpl, ora
from realmspinner.kernels.pixel import groups as gp
from realmspinner.kernels.pixel.document import Document
from realmspinner.kernels.pixel.flourish import keywords, presets
from realmspinner.kernels.pixel.flourish import recipe as R

PIXEL_DIR = Path(__file__).resolve().parents[3] / "src" / "realmspinner" / "kernels" / "pixel"


def _layer(rec, kind):
    return next(each for each in rec.layers if each.kind == kind)


# --- inker-flourish-08: a wrongly shaped container is named, not raised ----------


@pytest.mark.parametrize(
    "diff",
    [
        {"layers": ["Core"]},
        {"phases": ["main"]},
        {"hide": 3},
        {"show": 3.5},
        {"hide": "Core"},
        {"show": "Core"},
        {"layers": 7},
    ],
)
def test_apply_diff_names_a_wrongly_shaped_container_instead_of_raising(diff):
    """The 2026-10-14 audit, finding inker-flourish-08: ``layers`` as a list
    raised AttributeError, ``hide`` as a number raised TypeError and a string
    ``hide`` became one "no layer called" note per character."""
    before = presets.load("fireball")
    after, notes = keywords.apply_diff(before, diff)
    assert after == R.clamp(before)
    key = next(iter(diff))
    assert any(key in note and "not" in note for note in notes), notes
    assert len(notes) <= 2, "a string must not become one note per character"


# --- inker-flourish-13: "lots of" -------------------------------------------------


def test_lots_of_sparks_adds_particles():
    """The 2026-10-14 audit, finding inker-flourish-13: ``_MORE`` held the
    two-word phrase "lots of", which a one-word tokenizer can never yield."""
    before = presets.load("fireball")
    after, notes = keywords.apply(before, "lots of sparks")
    assert not any("No words I know" in note for note in notes), notes
    grew = [
        a.params["count"] > b.params["count"]
        for a, b in zip(after.layers, before.layers, strict=True)
        if a.kind == "particles"
    ]
    assert grew and all(grew)


# --- inker-flourish-14: a scale word binds to the kind after it -------------------


def test_a_scale_word_names_only_the_layers_it_scaled():
    """The 2026-10-14 audit, finding inker-flourish-14: "bigger sparks" scaled
    all twelve layers of ``fireball``, and "wilder" on a recipe with no
    turbulence-like parameter still toasted "wilder: x1.4"."""
    before = presets.load("fireball")
    after, _notes = keywords.apply(before, "bigger sparks")
    for a, b in zip(after.layers, before.layers, strict=True):
        if a.kind == "particles":
            assert a.params != b.params, a.name
        else:
            assert a.params == b.params, f"{a.name} was not named"

    bolt = presets.load("lightning_bolt")
    from realmspinner.kernels.pixel.flourish import prims

    assert not any(
        {"turbulence", "noise", "raggedness", "unevenness"} & set(prims.params_of(each.kind))
        for each in bolt.layers
    ), "the premise: no turbulence-like parameter on this recipe"
    same, notes = keywords.apply(bolt, "wilder")
    assert same.layers == R.clamp(bolt).layers
    assert not any("x1.4" in note for note in notes), notes


# --- inker-flourish-07: seed 0 -----------------------------------------------------


def test_a_recipe_with_seed_zero_round_trips_to_seed_zero():
    """The 2026-10-14 audit, finding inker-flourish-07: ``raw.get("seed") or 1``
    loaded seed 0 as seed 1 while the inspector's Seed field allows 0."""
    rec = R.from_dict({"seed": 0, "layers": []})
    assert rec.seed == 0
    assert R.loads(R.dumps(rec)).seed == 0
    assert R.from_dict({"layers": []}).seed == 1
    assert R.from_dict({"seed": None, "layers": []}).seed == 1


# --- inker-flourish-10: phase names stay unique ------------------------------------


def test_clamp_phase_names_stay_unique_when_a_renumbered_name_matches_a_later_one():
    """The 2026-10-14 audit, finding inker-flourish-10: ``["a", "a", "a 2"]``
    clamped to ``["a", "a 2", "a 2"]``, the collision the dedupe exists to
    prevent."""
    rec = R.clamp(
        R.Recipe(phases=(R.Phase("a", 3), R.Phase("a", 3), R.Phase("a 2", 3)))
    )
    names = [p.name for p in rec.phases]
    assert len(set(names)) == len(names), names
    assert names[0] == "a" and names[2] == "a 2", "explicit names keep their name"


# --- inker-flourish-11: the palette the pixel pass can read ------------------------


def test_a_loaded_recipe_palette_is_one_the_pixel_pass_can_read():
    """The 2026-10-14 audit, finding inker-flourish-11: ``clamp`` kept
    ``#112233ff`` (``prims.parse_color`` takes eight digits) and then every
    pixel-mode render died in ``pixel.parse_hex`` ("not a hex colour")."""
    from realmspinner.pipelines import pixel

    rec = R.from_dict(
        {"palette": ["#112233ff", "#445566", "778899", "nonsense"], "mode": "pixel", "layers": []}
    )
    colours = pixel.parse_hex("\n".join(rec.palette))  # must not raise
    assert colours[0] == (0x11, 0x22, 0x33)
    assert len(colours) == 3


# --- inker-flourish-12: a layer limited to a dropped phase -------------------------


def test_a_layer_limited_to_a_dropped_phase_stays_inactive():
    """The 2026-10-14 audit, finding inker-flourish-12: filtering a layer's
    ``phases`` down to the survivors left an empty tuple, which means "every
    phase", so a layer for ``p12`` (past ``MAX_PHASES``) rendered in ``p0``."""
    raw = {
        "phases": [{"name": f"p{i}", "frames": 2} for i in range(R.MAX_PHASES + 1)],
        "layers": [{"kind": "core", "phases": [f"p{R.MAX_PHASES}"]}],
    }
    rec = R.from_dict(raw)
    assert len(rec.phases) == R.MAX_PHASES
    layer = rec.layers[0]
    assert not any(layer.active_in(p.name) for p in rec.phases)
    # a layer with no restriction at all still means "every phase"
    free = R.from_dict({**raw, "layers": [{"kind": "core"}]}).layers[0]
    assert all(free.active_in(p.name) for p in rec.phases)


# --- inker-flourish-18: the painterly held-bytes ceiling ---------------------------


def test_check_bake_cost_refuses_a_painterly_bake_whose_held_cels_exceed_the_ceiling():
    """The 2026-10-14 audit, finding inker-flourish-18: only pixel mode had a
    byte ceiling; 1024x1024, two layers, 100 frames at ``supersample=1`` passed
    the time budget and held about 1.2 GiB of cels."""
    rec = R.clamp(
        R.Recipe(
            width=1024,
            height=1024,
            supersample=1,
            mode="painterly",
            phases=(R.Phase("main", 100, True),),
            layers=(R.Layer(R.new_uid(), "core"), R.Layer(R.new_uid(), "glow")),
        )
    )
    assert R.bake_cost(rec) <= R.MAX_BAKE_COST, "the premise: the time budget passes it"
    with pytest.raises(ValueError, match="bytes"):
        R.check_bake_cost(rec)


def test_every_shipped_preset_clears_the_painterly_ceiling_at_its_own_size():
    for name in presets.names():
        R.check_bake_cost(presets.load(name))


# --- inker-codecs-14: held frames merge on GIF export ------------------------------


def test_gif_round_trip_frame_count_is_stated_or_preserved_for_held_frames():
    """The 2026-10-14 audit, finding inker-codecs-14: four frames, three
    identical, reopen as two (Pillow sums the durations of identical
    consecutive frames) and nothing said so."""
    from PIL import Image

    from realmspinner.kernels.pixel import gifout

    a = np.zeros((4, 4, 4), dtype=np.uint8)
    a[..., 3] = 255
    a[1:3, 1:3, 0] = 255
    b = a.copy()
    b[..., 1] = 200
    frames = [a, a.copy(), a.copy(), b]
    out = io.BytesIO()
    gifout.write_gif(out, frames, [100, 100, 100, 100])
    out.seek(0)
    with Image.open(out) as im:
        count = im.n_frames
    doc = (gifout.__doc__ or "").lower()
    stated = "identical" in doc and "merge" in doc
    assert count == len(frames) or stated, (count, "the module must say held frames merge")


# --- inker-codecs-05: palette.gpl is read lazily and bounded ----------------------


def test_a_compressible_palette_member_is_refused_before_it_is_split_into_lines(
    tmp_path: Path,
):
    """The 2026-10-14 audit, finding inker-codecs-05: ``_read_palette`` decoded
    the whole member and ``gpl.parse`` ran ``splitlines()`` over it, a list of
    one string per line before ``MAX_PALETTE_ROWS`` was consulted. A 40 MB
    member of newlines peaked at 387 MiB on open."""
    doc = Document.blank(4, 4)
    path = tmp_path / "a.ora"
    ora.write_ora(doc, path)
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        items = {name: zf.read(name) for name in names}
    items[ora.PALETTE_MEMBER] = b"\n" * (24 * 1024 * 1024)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in names:
            zf.writestr(name, items[name])
        if ora.PALETTE_MEMBER not in names:
            zf.writestr(ora.PALETTE_MEMBER, items[ora.PALETTE_MEMBER])
    del items
    assert path.stat().st_size < 1 << 20

    tracemalloc.start()
    try:
        back = ora.read_ora(path)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert back.palette is None
    assert peak < 40 * 1024 * 1024, f"opening the file peaked at {peak / 2**20:.0f} MiB"


def test_gpl_parse_does_not_build_a_list_of_every_line():
    text = "\n" * (16 * 1024 * 1024) + "10 20 30 swatch\n"
    tracemalloc.start()
    try:
        colours = gpl.parse(text)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert colours == [(10, 20, 30, 255)]
    assert peak < 16 * 1024 * 1024, f"parse peaked at {peak / 2**20:.0f} MiB"


def test_the_other_palette_readers_agree_on_unusual_line_breaks():
    """The lazy line walk must split where ``str.splitlines`` did."""
    assert gpl.parse("GIMP Palette\r1 2 3 a\x0b4 5 6 b 7 8 9 c") == [
        (1, 2, 3, 255),
        (4, 5, 6, 255),
        (7, 8, 9, 255),
    ]
    assert gpl.parse_hex("112233\r\n445566\n") == [(17, 34, 51, 255), (68, 85, 102, 255)]


# --- inker-document-01: equal-extent isolated groups -------------------------------


def _wrapped_isolated_pair(second: str):
    doc = Document.blank(4, 4)
    doc.add_layer("a")
    doc.add_layer("b")
    doc.stack[1].pixels[:, :] = (255, 0, 0, 255)
    doc.stack[2].pixels[:, :] = (0, 255, 0, 255)
    doc.invalidate_all()
    outer = doc.group_layers([1, 2])
    inner = doc.group_layers([1, 2])
    assert doc.group_of[inner.uid] == outer.uid
    doc.set_group_props(outer.uid, blend="multiply")
    doc.set_group_props(inner.uid, **{second: True if second == "isolate" else "multiply"})
    return doc, outer, inner


def test_a_group_wrapped_around_exactly_the_rows_of_an_isolated_group_composites_without_recursing():  # noqa: E501
    """The 2026-10-14 audit, finding inker-document-01: ``spans()`` listed the
    inner group first, so ``top_level`` accepted it and dropped the outer;
    entering it accepted the outer, and so on until RecursionError -- out of
    ``set_group_props`` after the node was mutated, and out of every later
    ``invalidate_all`` too."""
    for second in ("isolate", "blend"):
        doc, outer, inner = _wrapped_isolated_pair(second)
        doc.invalidate_all()
        spans = gp.spans(doc.groups, doc.group_of, doc.member_uids())
        assert [s.guid for s in spans] == [outer.uid, inner.uid], "outer first on a tie"
        top = doc.composite
        assert tuple(top[1, 1]) == (0, 255, 0, 255)
        # undo is still one step and the document stays usable
        doc.undo()
        doc.invalidate_all()


# --- inker-document-12: the "v1 gap" docstrings ------------------------------------


@pytest.mark.parametrize("name", ["layers.py", "document.py"])
def test_group_docstrings_do_not_call_isolated_compositing_a_v1_gap(name: str):
    """The 2026-10-14 audit, finding inker-document-12: isolated compositing
    landed on 2026-08-31, but these comments still called it a v1 gap and cited
    a module (``inker/groups.py``) that does not exist."""
    text = (PIXEL_DIR / name).read_text(encoding="utf-8")
    assert "v1 gap" not in text
    assert "inker/groups.py" not in text
    assert (PIXEL_DIR / "groups.py").is_file()

