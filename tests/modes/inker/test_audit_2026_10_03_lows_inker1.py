"""The 2026-10-03 audit's Low findings, fixer inker1 (codecs, paint, sheets,
document, flourish, mode, panes). One file per fixer; every test's name is the
claim and was run red against the unfixed code first.
"""

from __future__ import annotations

import numpy as np
import pytest

from realmspinner.kernels.pixel import aseout
from realmspinner.kernels.pixel.animation import Note, Tag
from realmspinner.kernels.pixel.document import Document

BLUE = (0, 0, 255, 255)


def _animated() -> Document:
    doc = Document.blank(4, 4)
    doc.stack[0].name = "Background"
    doc.add_layer("Ink")
    doc.invalidate_all()
    doc.ensure_animation()
    doc.add_frame(link=True)
    return doc


# --- inker-67 ----------------------------------------------------------------


@pytest.mark.parametrize("role", ["layer", "tag", "note"])
def test_a_name_past_a_word_is_refused_by_name_when_writing_aseprite(role):
    """``_Writer.string`` packed ``len(raw)`` into a WORD, so a layer name, tag
    name or note text past 65535 UTF-8 bytes (an .ora accepts any length) died
    with a bare ``struct.error`` naming neither the field nor the layer."""
    long = "x" * 70_000
    doc = _animated()
    if role == "layer":
        doc.anim.tracks[1].name = long
    elif role == "tag":
        doc.anim.tags.append(Tag(name=long, start=0, end=1))
    else:
        doc.anim.tracks[1].note = Note(text=long)
    with pytest.raises(ValueError, match="65535") as caught:
        aseout.aseprite_bytes(doc)
    assert not isinstance(caught.value, __import__("struct").error)


# --- inker-68 ----------------------------------------------------------------


def _reference_slices_for(sprite, frames):
    """The pre-fix loop, kept as the oracle for what the fast one must answer."""
    out = []
    for entry in sprite.slices:
        keys = sorted(entry.keys, key=lambda pair: pair[0])
        _, base = keys[0]
        overrides = {}
        applies = base
        pos = 0
        for index, frame in enumerate(frames):
            while pos < len(keys) and keys[pos][0] <= index:
                applies = keys[pos][1]
                pos += 1
            if applies is not base:
                overrides[frame.uid] = applies
        out.append((entry.name, base.bounds, overrides))
    return out


def test_many_slices_over_many_frames_open_in_time_proportional_to_the_keys():
    """4,000 one-key slices over 65,535 frames took 10 s to open: the loop was
    slices x frames though no frame can change a one-key slice's answer."""
    import time

    from realmspinner.kernels.pixel import asein
    from realmspinner.kernels.pixel.animation import Frame
    from realmspinner.kernels.pixel.slices import SliceKey

    frames = [Frame() for _ in range(65_535)]
    sprite = asein.Sprite(
        slices=[
            asein.AseSlice(name=f"s{i}", keys=[(0, SliceKey(bounds=(0, 0, 2, 2)))])
            for i in range(4_000)
        ]
    )
    started = time.perf_counter()
    result = asein._slices_for(sprite, frames, lambda text: None)
    elapsed = time.perf_counter() - started
    assert len(result) == 4_000
    assert all(not s.keys for s in result)
    assert elapsed < 1.0, f"{elapsed:.1f}s"


def test_slices_for_resolves_run_to_the_end_keys_like_the_per_frame_scan():
    from realmspinner.kernels.pixel import asein
    from realmspinner.kernels.pixel.animation import Frame
    from realmspinner.kernels.pixel.slices import SliceKey

    def key(n):
        return SliceKey(bounds=(0, 0, n, n))

    frames = [Frame() for _ in range(10)]
    sprite = asein.Sprite(
        slices=[
            asein.AseSlice(name="a", keys=[(0, key(1)), (3, key(2)), (3, key(3)), (7, key(1))]),
            asein.AseSlice(name="late", keys=[(2, key(4)), (5, key(5))]),
            asein.AseSlice(name="past", keys=[(0, key(1)), (12, key(6))]),
            asein.AseSlice(name="one", keys=[(0, key(1))]),
        ]
    )
    got = asein._slices_for(sprite, frames, lambda text: None)
    want = _reference_slices_for(sprite, frames)
    assert [(s.name, s.bounds, s.keys) for s in got] == want


# --- inker-69 ----------------------------------------------------------------


class _TaskCtx:
    """Enough of ``Ctx`` to run a ``ctx.submit`` task inline and keep toasts."""

    def __init__(self, state):
        from types import SimpleNamespace

        self.state = SimpleNamespace(inker=state)
        self.toasts: list[tuple[str, str]] = []
        self.submitted: list[tuple[str, object]] = []

    def toast(self, message, level="info", **_kw):
        self.toasts.append((message, level))

    def submit(self, key, fn, *args, **kwargs):
        self.submitted.append((key, fn))
        return True


def test_a_palette_from_an_image_with_no_opaque_pixel_is_refused(monkeypatch):
    """``build_palette`` answers ``[(0, 0, 0, 255)]`` for a picture with nothing
    visible, and ``_done_palimg`` indexed the document to that one black colour
    with a success toast, on a pick that carried no palette at all."""
    from realmspinner.core.safeio import pixelguard
    from realmspinner.studio import dialogs
    from realmspinner.studio.modes.inker import mode as inker_mode
    from realmspinner.studio.modes.inker import palette_io
    from realmspinner.studio.modes.inker import state as inker_state
    from realmspinner.studio.tasks import Done

    doc = Document.blank(8, 8)
    doc.stack[0].pixels[:, :] = (200, 30, 30, 255)
    doc.invalidate_all()
    tab = inker_state.InkerDoc(doc=doc, uid="t1", title="Untitled")
    state = inker_state.InkerState()
    state.add(tab)
    ctx = _TaskCtx(state)
    monkeypatch.setattr(dialogs, "open_file", lambda *a, **k: "ghost.png")
    monkeypatch.setattr(
        pixelguard, "decode_rgba", lambda *a, **k: np.zeros((6, 6, 4), dtype=np.uint8)
    )
    before = [tuple(p) for p in doc.stack[0].pixels[0, :2]]

    palette_io.palette_from_image(ctx)
    key, run = ctx.submitted[-1]
    ctx.toasts.clear()
    inker_mode._done_palimg(ctx, state, Done(key=key, result=run()))

    assert [tuple(p) for p in doc.stack[0].pixels[0, :2]] == before
    assert doc.palette in (None, [])
    assert len(ctx.toasts) == 1
    message, level = ctx.toasts[0]
    assert level != "success" and "visible" in message


# --- inker-70 ----------------------------------------------------------------


def test_remove_orphans_does_not_recolour_a_pixel_whose_match_is_outside_the_selection_crop():
    """The filter sees only the selection's crop and padded it with 'no colour',
    so a deliberate two-pixel mark straddling the crop's edge lost the half
    inside it: its same-coloured partner was off-crop, not absent."""
    from realmspinner.kernels.pixel import filters

    field = np.zeros((7, 7, 4), dtype=np.uint8)
    field[..., :3] = (40, 40, 40)
    field[..., 3] = 255
    field[3, 3, :3] = (250, 10, 10)
    field[3, 4, :3] = (250, 10, 10)
    crop = field[:, 4:].copy()  # the selection starts at the pair's second pixel

    out = filters.remove_orphans(crop, orphans=1.0)

    assert tuple(int(v) for v in out[3, 0, :3]) == (250, 10, 10)
    # ...and a friendless pixel well inside the crop is still cleaned.
    interior = np.zeros((7, 7, 4), dtype=np.uint8)
    interior[..., :3] = (40, 40, 40)
    interior[..., 3] = 255
    interior[3, 3, :3] = (250, 10, 10)
    assert tuple(int(v) for v in filters.remove_orphans(interior, orphans=1.0)[3, 3, :3]) == (
        40,
        40,
        40,
    )


# --- inker-72 ----------------------------------------------------------------


def _ora_with_animation(tmp_path, edit):
    import json
    import zipfile

    from realmspinner.kernels.pixel import ora

    doc = Document.blank(2, 2)
    doc.ensure_animation()
    doc.add_frame(link=True)
    doc.anim.tags.append(Tag(name="walk", start=0, end=1))
    path = tmp_path / "inf.ora"
    ora.write_ora(doc, path)
    with zipfile.ZipFile(path) as zf:
        members = [(info, zf.read(info.filename)) for info in zf.infolist()]
    out = []
    for info, body in members:
        if info.filename == ora.ANIMATION_MEMBER:
            payload = json.loads(body)
            edit(payload)
            body = json.dumps(payload).encode("utf-8")  # emits bare Infinity
        out.append((info, body))
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for info, body in out:
            zf.writestr(info, body)
    return path


@pytest.mark.parametrize("where", ["duration", "repeat"])
def test_an_animation_json_duration_of_infinity_falls_back_to_the_flat_read(tmp_path, where):
    """JSON accepts a bare ``Infinity``; ``int(inf)`` raises OverflowError, which
    nothing caught, so a file whose pixels were all intact failed to open."""
    from realmspinner.kernels.pixel import ora

    def edit(payload):
        if where == "duration":
            payload["frames"][1]["duration_ms"] = float("inf")
        else:
            payload["tags"][0]["repeat"] = float("inf")

    path = _ora_with_animation(tmp_path, edit)
    back = ora.read_ora(path)  # raised OverflowError
    assert back.size == (2, 2)


def test_clamp_duration_and_tag_repeat_survive_infinity():
    from realmspinner.kernels.pixel import animation

    assert animation.clamp_duration(float("inf")) == animation.DEFAULT_DURATION_MS
    assert Tag(name="t", start=0, end=0, repeat=float("inf")).repeat == 0


# --- inker-74 ----------------------------------------------------------------


def test_despeckle_leaves_a_flat_low_alpha_region_exactly_as_it_was():
    """Premultiplied RGB narrowed to 8 bits cannot carry colour at alpha below
    ~32, so the round trip through the median moved the hue of a faint
    antialiased fringe even where the median changed nothing."""
    from realmspinner.kernels.pixel import filters

    field = np.zeros((9, 9, 4), dtype=np.uint8)
    field[..., :3] = (200, 77, 31)
    field[..., 3] = 9
    out = filters.despeckle(field, speck=1.0)
    assert np.array_equal(out, field)


# --- inker-76 / inker-77 -----------------------------------------------------


def _three_frames_with(*tags):
    doc = Document.blank(4, 4)
    doc.ensure_animation()
    doc.add_frame()
    doc.add_frame()
    assert len(doc.anim.frames) == 3
    # Appended straight, as a file reader or a frame delete leaves them: the
    # editing funnel would have clamped and ordered these on the way in.
    doc.anim.tags.extend(tags)
    return doc


def test_sheetout_tag_span_orders_an_inverted_tag_like_the_timeline_does():
    """Both readers build tags unordered; the timeline plays 2..0 as 0-2 but the
    sidecar wrote start 2, end 0, which this app's own ``sheetin.span_tags``
    refuses ("covers frames 2-0")."""
    from realmspinner.kernels.pixel import sheetout

    doc = _three_frames_with(Tag(name="flip", start=2, end=0))
    tag = doc.anim.tags[0]
    assert doc.anim.tag_span(tag) == (0, 2)
    assert sheetout.tag_span(doc.anim, tag) == (0, 2)
    _durations, tags, _layout = sheetout.timing(doc)
    assert (tags[0].start, tags[0].end) == (0, 2)


def test_a_tag_wholly_past_the_end_is_exported_the_same_whole_and_in_part():
    """A frame delete leaves tag B 3-5 on a 3-frame clip. The timeline and the
    whole export clamp it onto the last frame (B 2-2); a range export that
    includes that cell dropped it, and ``remap_tags`` dropped it too."""
    from realmspinner.kernels.pixel import sheetout

    doc = _three_frames_with(Tag(name="B", start=3, end=5))
    tag = doc.anim.tags[0]

    _d, whole, _l = sheetout.timing(doc)
    assert [(t.name, t.start, t.end) for t in whole] == [("B", 2, 2)]

    _d, part, _l = sheetout.timing(doc, (1, 2))
    assert [(t.name, t.start, t.end) for t in part] == [("B", 1, 1)]

    _d, elsewhere, _l = sheetout.timing(doc, (0, 1))
    assert elsewhere == []  # the last cell is not in this range

    remapped = sheetout.remap_tags([tag], [0, 1, 2])
    assert [(t.start, t.end) for t in remapped] == [(2, 2)]


# --- inker-78 ----------------------------------------------------------------


def test_dropped_by_aseprite_reports_a_sheet_merge_base():
    """The recorded render digests travel in .ora (animation.json's ``sheet``)
    and are dropped by aseout, yet no line said so: after Save As .aseprite a
    Merge against a re-rendered sheet was silently off."""
    from realmspinner.kernels.pixel import sheetmerge

    doc = Document.blank(4, 4)
    doc.ensure_animation()
    assert not any("merge" in line for line in aseout.dropped_by_aseprite(doc))
    doc.sheet_base = sheetmerge.SheetBase(digests={1: "abc"})
    lines = aseout.dropped_by_aseprite(doc)
    assert any("sheet merge base" in line for line in lines), lines


# --- inker-79 ----------------------------------------------------------------


def test_the_compat_refusal_row_names_every_asein_refusal_class():
    """The ledger row that claims to list everything the reader refuses omitted
    seven classes, so a reader of COMPAT.md could not tell those refusals from
    a bug."""
    from pathlib import Path

    text = (Path(__file__).resolve().parents[3] / "docs" / "COMPAT.md").read_text(
        encoding="utf-8"
    )
    (row,) = [
        line for line in text.splitlines() if line.startswith("| Everything refused outright")
    ]
    row = row.lower()
    for needle in (
        "layer of a kind",  # asein.py: "is of a kind this build does not know"
        "external file",  # a linked tileset
        "no embedded pixel",  # a tileset holding nothing readable
        "no tiles",
        "names a layer the file",  # a cel naming a layer past the layer list
        "tileset the file does not define",  # a tilemap layer's dangling binding
        "no palette",  # an indexed file with nothing to read its pixels with
        "palette entry past",  # an indexed plane naming a slot beyond the table
        "size ceiling",  # the decompressed / pixel / keys / palette ceilings
    ):
        assert needle in row, needle


# --- inker-81 ----------------------------------------------------------------


def test_changing_layer_or_frame_commits_the_floating_buffer():
    """Three comments said a float 'survives selecting another layer, another
    frame'; both setters commit it first. Pinned so the comments (and the
    unguarded ``by_uid(floating.layer_uid)`` in ``commit_floating``) stay true
    to what the code does."""
    doc = Document.blank(4, 4)
    doc.add_layer("Ink")
    doc.ensure_animation()
    doc.add_frame()
    doc.set_current_frame(0)
    doc.set_active_layer(0)
    doc.select_all()
    assert doc.lift()
    assert doc.floating is not None
    doc.set_active_layer(1)
    assert doc.floating is None

    doc.set_active_layer(0)
    doc.select_all()
    assert doc.lift()
    assert doc.floating is not None
    doc.set_current_frame(1)
    assert doc.floating is None


# --- inker-83 ----------------------------------------------------------------


def test_the_texture_and_restyle_refusals_describe_the_scope_they_enforce():
    """``flourish_texture_pending`` and ``flourish_restyle_pending`` are single
    slots on the InkerState shared by every open document, so the refusal must
    not say 'of this effect' and the manual must not say 'per document': a user
    generating in one tab was told another tab's effect had one running."""
    from pathlib import Path

    from realmspinner.studio.modes.inker import flourish

    assert "this effect" not in flourish.RESTYLE_PENDING
    assert "this effect" not in flourish.TEXTURE_PENDING
    chapter = Path(__file__).resolve().parents[3] / "docs" / "manual" / "29-inker-animation.md"
    manual = chapter.read_text(encoding="utf-8")
    assert "one runs at a time per document" not in manual


# --- inker-85 ----------------------------------------------------------------


def test_a_model_diff_that_changes_nothing_falls_back_to_the_keyword_mapper(monkeypatch, tmp_path):
    """Any answer that parsed as a JSON object landed as source 'model' even
    when ``apply_diff`` dropped everything it named, and the keyword mapper was
    never tried on those words -- where 'bigger' would have been understood."""
    from realmspinner.kernels.pixel.flourish import keywords, presets
    from realmspinner.studio.modes.inker import flourish as inker_flourish

    rec = presets.load("fireball")
    monkeypatch.setattr(
        inker_flourish,
        "run_text_model",
        lambda recipe, text, model_dir: ({"layers": {"no such layer": {"size": 2}}}, ""),
    )
    changed, notes, source = inker_flourish.ask_words(rec, "bigger", model_dir=tmp_path)
    expected, _ = keywords.apply(rec, "bigger")
    assert expected != rec, "fixture: the mapper understands 'bigger'"
    assert source == "keywords"
    assert changed == expected
    assert any("used the keyword mapper" in note for note in notes)


def test_a_model_diff_that_changes_something_is_still_the_model(monkeypatch, tmp_path):
    from realmspinner.kernels.pixel.flourish import presets
    from realmspinner.studio.modes.inker import flourish as inker_flourish

    rec = presets.load("fireball")
    name = rec.layers[0].name
    monkeypatch.setattr(
        inker_flourish,
        "run_text_model",
        lambda recipe, text, model_dir: ({"hide": [name]}, ""),
    )
    changed, _notes, source = inker_flourish.ask_words(rec, "no flames", model_dir=tmp_path)
    assert source == "model" and changed != rec


# --- inker-89 ----------------------------------------------------------------


def test_docstrings_name_no_inker_state_view_function_that_has_moved():
    """``pan_limits``/``clamp_pan``/``scroll_thumb``/``ROTATIONS``/``basis``/
    ``to_screen``/``view_extent`` live in ``studio/shell/paintview.py`` since the
    P5 promotion; two docstrings still sent a reader to ``inker_state.*``."""
    import re
    from pathlib import Path

    from realmspinner.studio.modes.inker import state as inker_state
    from realmspinner.studio.shell import paintview

    moved = (
        "pan_limits",
        "clamp_pan",
        "scroll_thumb",
        "ROTATIONS",
        "basis",
        "to_screen",
        "view_extent",
    )
    for name in moved:
        assert hasattr(paintview, name), name
        assert not hasattr(inker_state, name), f"{name} is back in inker state"
    root = Path(inker_state.__file__).resolve().parent
    pattern = re.compile(r"\binker_state\.(" + "|".join(moved) + r")\b")
    hits = [
        f"{path.relative_to(root)}:{n}"
        for path in root.rglob("*.py")
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if pattern.search(line)
    ]
    assert hits == []


# --- inker-93 / inker-94 -----------------------------------------------------


def _timeline_tab(doc):
    from realmspinner.studio.modes.inker import state as inker_state

    return inker_state.InkerDoc(doc=doc, title="t", saved_head=doc.history.head)


def test_row_menu_move_up_is_greyed_with_a_reason_on_the_top_row():
    """The row menu's Move up/down stayed enabled on the end rows (and where a
    background layer refuses the move) while ``move_layer`` returned False in
    silence; ``_frame_menu`` already greys Move left/right at its ends."""
    from realmspinner.studio import widgets
    from realmspinner.studio.modes.inker.ui.panes import timeline

    doc = Document.blank(4, 4)
    doc.add_layer("Mid")
    doc.add_layer("Top")
    tab = _timeline_tab(doc)

    enabled, why = timeline._move_gate(tab, doc, 2, +1)
    assert not enabled and "top" in why.lower()
    enabled, why = timeline._move_gate(tab, doc, 0, -1)
    assert not enabled and "bottom" in why.lower()
    assert timeline._move_gate(tab, doc, 1, +1) == (True, "")
    assert timeline._move_gate(tab, doc, 1, -1) == (True, "")

    # A background layer keeps the bottom row: it cannot go up and nothing can
    # go beneath it -- the same refusal ``LayerStack.move`` raises for.
    doc.stack[0].background = True
    enabled, why = timeline._move_gate(tab, doc, 0, +1)
    assert not enabled and "background" in why.lower()
    enabled, why = timeline._move_gate(tab, doc, 1, -1)
    assert not enabled and "background" in why.lower()
    assert doc.move_layer(1, 0) is False  # the gate agrees with the engine

    tab.saving = True
    assert timeline._move_gate(tab, doc, 1, +1) == (False, widgets.DOCUMENT_SAVING_WHY)


def test_row_menu_uses_the_move_gate():
    import inspect

    from realmspinner.studio.modes.inker.ui.panes import timeline

    source = inspect.getsource(timeline._row_menu)
    assert source.count("_move_gate(") == 2


def test_timeline_cells_say_why_they_ignore_a_click_while_busy(monkeypatch):
    """Frame numbers, cels, the layer name and the tag name swallowed a press
    with ``and not tab.busy`` and said nothing, next to a transport greyed with
    a sentence."""
    import inspect
    from types import SimpleNamespace

    from realmspinner.studio import widgets
    from realmspinner.studio.modes.inker.ui.panes import timeline

    tips = []
    monkeypatch.setattr(
        timeline,
        "imgui",
        SimpleNamespace(is_item_hovered=lambda: True, set_tooltip=tips.append),
    )
    doc = Document.blank(4, 4)
    tab = _timeline_tab(doc)

    timeline._busy_tip(tab)
    assert tips == []
    tab.saving = True
    timeline._busy_tip(tab)
    assert tips == [widgets.DOCUMENT_SAVING_WHY]

    # And every control that swallows the press asks for the tip.
    assert inspect.getsource(timeline._frame_headers).count("_busy_tip(tab)") == 1
    assert inspect.getsource(timeline._cell).count("_busy_tip(tab)") == 1
    assert inspect.getsource(timeline._tag_row).count("_busy_tip(tab)") == 1
    assert "DOCUMENT_SAVING_WHY" in inspect.getsource(timeline._track_row)
