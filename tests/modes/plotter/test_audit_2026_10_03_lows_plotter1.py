"""Closing the Low findings plotter-25 through plotter-33 of the 2026-10-03 audit.

One test per claim, named for it.
"""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest

from realmspinner.kernels.grid2d.tileset import Tileset
from realmspinner.studio.modes.plotter.engine import rmap, tmx
from realmspinner.studio.modes.plotter.engine.tilemap import MapDoc


def _pixels(w: int = 8, h: int = 8) -> np.ndarray:
    array = np.zeros((h, w, 4), dtype=np.uint8)
    array[..., 3] = 255
    return array


def _doc() -> MapDoc:
    doc = MapDoc(4, 4, 8, 8)
    doc.add_tileset(Tileset(name="t", pixels=_pixels(), tile_w=8, tile_h=8))
    return doc


def _members(data: bytes) -> dict[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        return {name: zf.read(name) for name in zf.namelist()}


def _pack(members: dict[str, bytes]) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as zf:
        for name, blob in members.items():
            zf.writestr(name, blob)
    return out.getvalue()


def _rewrite(doc: MapDoc, mutate) -> bytes:
    members = _members(rmap.rmap_bytes(doc))
    manifest = json.loads(members[rmap.MANIFEST])
    mutate(manifest)
    members[rmap.MANIFEST] = json.dumps(manifest).encode()
    return _pack(members)


def _npy(array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, array, allow_pickle=False)
    return buf.getvalue()


# --- plotter-25 -----------------------------------------------------------------


@pytest.mark.parametrize("bad", [7.5, -0.1, float("nan"), float("inf")])
def test_a_layer_opacity_outside_zero_to_one_is_refused(bad):
    doc = _doc()
    layer = doc.add_tile_layer("Ground")
    head = doc.history.head
    with pytest.raises(ValueError, match="opacity"):
        doc.set_layer_props(layer.uid, opacity=bad)
    assert doc.history.head == head, "a refused opacity pushed an undo step"
    assert layer.opacity == 1.0

    def mutate(manifest):
        manifest["layers"][-1]["opacity"] = bad

    # json.dumps writes NaN/Infinity as bare tokens, which json.loads reads back.
    with pytest.raises(ValueError, match="opacity"):
        rmap.read_rmap(_rewrite(doc, mutate))


def test_a_layer_opacity_outside_zero_to_one_is_refused_by_both_tiled_readers():
    def _img(_p):
        return _pixels(16, 16)

    def _tsx(_p):
        return Tileset(name="t", pixels=_pixels(16, 16), tile_w=16, tile_h=16)

    xml = (
        b'<map version="1.10" orientation="orthogonal" width="1" height="1" '
        b'tilewidth="16" tileheight="16"><layer id="1" name="a" width="1" height="1" '
        b'opacity="7.5"><data encoding="csv">0</data></layer></map>'
    )
    with pytest.raises(ValueError, match="opacity"):
        tmx.read_tmx(xml, image_loader=_img, tsx_loader=_tsx)
    payload = {
        "type": "map",
        "orientation": "orthogonal",
        "width": 1,
        "height": 1,
        "tilewidth": 16,
        "tileheight": 16,
        "layers": [
            {
                "type": "tilelayer",
                "id": 1,
                "name": "a",
                "width": 1,
                "height": 1,
                "data": [0],
                "opacity": 7.5,
            }
        ],
    }
    with pytest.raises(ValueError, match="opacity"):
        tmx.read_tmj(json.dumps(payload).encode(), image_loader=_img, tsx_loader=_tsx)


# --- plotter-26 -----------------------------------------------------------------


def test_read_rmap_falls_back_on_an_unknown_stagger_axis():
    doc = _doc()
    data = _rewrite(doc, lambda m: m.__setitem__("stagger", ["z", "weird", -5]))
    opened = rmap.read_rmap(data)
    assert (opened.stagger_axis, opened.stagger_index) == ("y", "odd")
    assert opened.hex_side == 0
    # The repair-blocking shape: an unrelated settings change must still save.
    opened.set_map_settings(class_name="repaired")
    assert opened.map_settings()["class_name"] == "repaired"


# --- plotter-31 -----------------------------------------------------------------


def _doc_with_stamp_entries(entries: list[dict], blobs: dict[str, bytes]) -> bytes:
    members = _members(rmap.rmap_bytes(_doc()))
    manifest = json.loads(members[rmap.MANIFEST])
    manifest["stamps"] = entries
    members[rmap.MANIFEST] = json.dumps(manifest).encode()
    members.update(blobs)
    return _pack(members)


def test_a_manifest_naming_one_stamp_member_many_times_is_charged_to_the_read_budget(
    monkeypatch,
):
    blob = _npy(np.ones((2, 2), dtype=np.uint32))
    entries = [{"slot": 1, "name": f"s{i}", "member": "stamps/one.npy"} for i in range(60)]
    data = _doc_with_stamp_entries(entries, {"stamps/one.npy": blob})

    calls = []
    real = rmap.npyguard.read_array

    def counting(raw, what):
        calls.append(what)
        return real(raw, what)

    monkeypatch.setattr(rmap.npyguard, "read_array", counting)
    rmap.read_rmap(data)
    assert len(calls) == 1, f"one member was decoded {len(calls)} times"

    # And what is decoded is charged: two distinct 16-byte members cannot fit
    # under a 20-byte ceiling.
    other = _npy(np.ones((2, 2), dtype=np.uint32))
    entries = [
        {"slot": 1, "member": "stamps/one.npy"},
        {"slot": 2, "member": "stamps/two.npy"},
    ]
    data = _doc_with_stamp_entries(
        entries, {"stamps/one.npy": blob, "stamps/two.npy": other}
    )
    monkeypatch.setattr(rmap.npyguard, "read_array", real)
    monkeypatch.setattr(rmap, "MAX_DECOMPRESSED_BYTES", 20)
    with pytest.raises(ValueError, match="decode"):
        rmap._stamps_from(zipfile.ZipFile(io.BytesIO(data)), entries)


def test_a_stamp_stored_as_a_non_gid_dtype_is_refused_like_a_layer():
    blob = _npy(np.ones((2, 2), dtype=np.float64))
    data = _doc_with_stamp_entries(
        [{"slot": 1, "member": "stamps/one.npy"}], {"stamps/one.npy": blob}
    )
    with pytest.raises(ValueError, match="stamp"):
        rmap.read_rmap(data)


# --- plotter-27 -----------------------------------------------------------------


def test_the_status_line_reports_the_clamped_selection():
    from types import SimpleNamespace

    from realmspinner.studio.modes.plotter.state import PlotterState
    from realmspinner.studio.modes.plotter.ui.panes.canvas import status_bits

    doc = MapDoc(4, 4, 8, 8)
    doc.add_tile_layer()
    tab = SimpleNamespace(doc=doc, busy=False, view=SimpleNamespace(zoom=1.0))
    state = PlotterState()
    state.hover_cell = None
    state.tool = "stamp"
    state.terrain = None

    # A marquee drawn on a bigger map, then the map shrank under it.
    state.select = (0, 0, 11, 11)
    assert "sel 4 x 4" in status_bits(state, tab)
    assert "sel 12 x 12" not in status_bits(state, tab)

    # Wholly outside the map: it constrains nothing, so it must read as nothing.
    state.select = (8, 8, 11, 11)
    assert not any(bit.startswith("sel ") for bit in status_bits(state, tab))


# --- plotter-30 -----------------------------------------------------------------


def test_a_stamp_edit_costs_the_bytes_of_the_blocks_it_holds():
    doc = MapDoc(4, 4, 8, 8)
    first = np.ones((50, 50), dtype=np.uint32)
    second = np.full((50, 50), 2, dtype=np.uint32)
    assert doc.set_stamp(1, first)
    assert doc.history.top.cost == first.nbytes  # nothing before, one block after
    assert doc.set_stamp(1, second)
    assert doc.history.top.cost == first.nbytes + second.nbytes
    assert doc.history.bytes == 2 * first.nbytes + second.nbytes
    # A rename holds one block in both halves; it is one array, counted once.
    assert doc.rename_stamp(1, "roof")
    assert doc.history.top.cost == second.nbytes
    assert doc.clear_stamp(1)
    assert doc.history.top.cost == second.nbytes


# --- plotter-34 -----------------------------------------------------------------


def _plotter_sheet_keys() -> str:
    from realmspinner.studio.shortcuts import shortcut_sections

    for title, rows in shortcut_sections():
        if title == "Plotter":
            return "\n".join(keys for keys, _action in rows)
    raise AssertionError("the shortcuts sheet has no Plotter table")


def _plotter_chapter_keys() -> str:
    from pathlib import Path

    text = (
        Path(__file__).resolve().parents[3] / "docs" / "manual" / "38-shortcuts.md"
    ).read_text(encoding="utf-8")
    section = text.split("\n## Plotter", 1)[1].split("\n## ", 1)[0]
    return "\n".join(line for line in section.splitlines() if line.startswith("|"))


def test_every_plotter_chord_in_handle_key_is_on_the_shortcuts_sheet():
    import ast
    import re
    from pathlib import Path

    from realmspinner.studio.modes.plotter import mode as plotter_mode

    # The four the audit found bound and unlisted, by name...
    for chord in ("Ctrl+R", "Ctrl+Shift+G", "Ctrl+Shift+P", "Ctrl+Shift+Up"):
        assert chord in _plotter_sheet_keys(), f"{chord} is bound but not on the sheet"
        assert chord in _plotter_chapter_keys(), f"{chord} is bound but not in chapter 38"

    # ...and every letter ``_ctrl_key`` compares ``name`` to, so the next chord
    # added without a row fails here rather than in the next audit.
    tree = ast.parse(Path(plotter_mode.__file__).read_text(encoding="utf-8"))
    func = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_ctrl_key"
    )
    letters: set[str] = set()
    for node in ast.walk(func):
        if (
            isinstance(node, ast.Compare)
            and isinstance(node.left, ast.Name)
            and node.left.id == "name"
        ):
            for comp in node.comparators:
                for const in ast.walk(comp):
                    if (
                        isinstance(const, ast.Constant)
                        and isinstance(const.value, str)
                        and len(const.value) == 1
                        and const.value.isalnum()
                    ):
                        letters.add(const.value)
    assert {"r", "g", "p", "s", "z"} <= letters, "the walk found nothing to check"
    keys = _plotter_sheet_keys()
    for letter in sorted(letters):
        pattern = rf"(Ctrl\+(Shift\+)?|/ |- ){re.escape(letter)}(?![A-Za-z])"
        assert re.search(pattern, keys, re.IGNORECASE), (
            f"Ctrl chord {letter!r} is bound in _ctrl_key but absent from the sheet"
        )


# --- plotter-35 -----------------------------------------------------------------


class _BusyScene:
    """The real ``canvas._events`` over a fake pointer, on a tab that is saving."""

    def __init__(self, monkeypatch, tool: str) -> None:
        from types import SimpleNamespace

        from imgui_bundle import imgui

        from realmspinner.studio.modes.plotter import state as plotter_state
        from realmspinner.studio.shell import paintview

        self.doc = MapDoc(4, 4, 16, 16)
        pixels = np.zeros((16, 64, 4), dtype=np.uint8)
        pixels[..., 3] = 255
        self.doc.add_tileset(Tileset(name="t", pixels=pixels, tile_w=16, tile_h=16))
        self.layer = self.doc.add_tile_layer("Ground")
        self.doc.set_active_layer(self.layer.uid)
        self.state = plotter_state.PlotterState()
        self.state.tool = tool
        self.state.minimap = False
        self.tab = SimpleNamespace(
            doc=self.doc,
            uid="tab-1",
            busy=True,
            view=paintview.PaintView(zoom=1.0, pan=(0.0, 0.0), fitted=True),
        )
        self.ctx = SimpleNamespace(toast=lambda *_a, **_k: None)
        self.at = (24.0, 24.0)  # cell (1, 1) at zoom 1, origin 0
        self.clicked = {0: False, 1: False, 2: False}
        self.down = {0: False, 1: False, 2: False}
        self.released = {0: False, 1: False, 2: False}
        self.wheel = 0.0
        self.drag = (0.0, 0.0)
        mp = monkeypatch
        mp.setattr(imgui, "get_mouse_pos", lambda: SimpleNamespace(x=self.at[0], y=self.at[1]))
        mp.setattr(imgui, "is_mouse_clicked", lambda b: self.clicked[b])
        mp.setattr(imgui, "is_mouse_down", lambda b: self.down[b])
        mp.setattr(imgui, "is_mouse_released", lambda b: self.released[b])
        mp.setattr(
            imgui, "is_mouse_dragging", lambda b: self.down[b] and self.drag != (0.0, 0.0)
        )
        mp.setattr(
            imgui,
            "get_mouse_drag_delta",
            lambda b: SimpleNamespace(x=self.drag[0], y=self.drag[1]),
        )
        mp.setattr(imgui, "reset_mouse_drag_delta", lambda b: None)
        mp.setattr(
            imgui,
            "get_io",
            lambda: SimpleNamespace(
                key_ctrl=False,
                key_alt=False,
                key_shift=False,
                mouse_wheel=self.wheel,
                mouse_wheel_h=0.0,
            ),
        )

    def frame(self) -> None:
        from realmspinner.studio.modes.plotter.ui.panes import canvas

        canvas._events(self.ctx, self.state, self.tab, (0.0, 0.0), True, (400.0, 300.0))


def test_pan_zoom_and_selection_work_while_the_map_is_being_written(monkeypatch):
    scene = _BusyScene(monkeypatch, "select")
    head = scene.doc.history.head

    # Wheel zoom.
    scene.wheel = 1.0
    scene.frame()
    assert scene.tab.view.zoom != 1.0, "the wheel did not zoom a saving map"
    scene.wheel = 0.0
    scene.tab.view.zoom = 1.0

    # Middle-drag pan.
    scene.down[2] = True
    scene.drag = (7.0, 5.0)
    pan = scene.tab.view.pan
    scene.frame()
    assert scene.tab.view.pan != pan, "a middle drag did not pan a saving map"
    scene.down[2] = False
    scene.drag = (0.0, 0.0)

    # The marquee.
    scene.clicked[0] = scene.down[0] = True
    scene.frame()
    assert scene.state.select is not None, "the marquee did not start on a saving map"
    scene.state.set_selection(None)
    scene.clicked[0] = scene.down[0] = False

    # The wand.
    scene.state.tool = "wand"
    scene.clicked[0] = scene.down[0] = True
    scene.frame()
    assert scene.state.select is not None, "the wand did not select on a saving map"
    assert scene.doc.history.head == head


def test_a_writing_gesture_still_waits_for_the_save_to_finish(monkeypatch):
    scene = _BusyScene(monkeypatch, "stamp")
    scene.state.brush = np.array([[1]], dtype=np.uint32)
    head = scene.doc.history.head
    scene.clicked[0] = scene.down[0] = True
    scene.frame()
    scene.clicked[0] = False
    scene.down[0] = False
    scene.released[0] = True
    scene.frame()
    assert scene.doc.history.head == head
    assert int(np.count_nonzero(scene.layer.data)) == 0


# --- plotter-36 -----------------------------------------------------------------


def test_ctrl_s_and_ctrl_z_still_work_with_the_tileset_sheet_open(plotter_ctx, monkeypatch):
    from types import SimpleNamespace

    import pygame

    from realmspinner.studio.modes.plotter import mode as plotter_mode

    ctx, state = plotter_ctx
    tab = state.active
    tab.doc.add_tileset(_tileset())
    state.editing_tileset = 0  # the sheet covers the canvas now

    calls: list[str] = []
    monkeypatch.setattr(plotter_mode, "save", lambda _c, _t: calls.append("save"))
    monkeypatch.setattr(plotter_mode, "save_as", lambda _c, _t: calls.append("save_as"))
    monkeypatch.setattr(plotter_mode, "undo", lambda _c, _t: calls.append("undo"))
    monkeypatch.setattr(plotter_mode, "redo", lambda _c, _t: calls.append("redo"))

    def press(key, mod):
        event = SimpleNamespace(type=pygame.KEYDOWN, key=key, mod=mod)
        return plotter_mode.handle_key(ctx, event)

    ctrl, shift = pygame.KMOD_CTRL, pygame.KMOD_SHIFT
    assert press(pygame.K_s, ctrl) is True
    assert press(pygame.K_s, ctrl | shift) is True
    assert press(pygame.K_z, ctrl) is True
    assert press(pygame.K_y, ctrl) is True
    assert calls == ["save", "save_as", "undo", "redo"]

    # The map-editing keys are still held back by the sheet.
    calls.clear()
    assert press(pygame.K_v, ctrl) is False
    assert press(pygame.K_DELETE, 0) is False
    assert calls == []


def _tileset():
    pixels = np.zeros((16, 32, 4), dtype=np.uint8)
    pixels[..., 3] = 255
    return Tileset(name="t", pixels=pixels, tile_w=16, tile_h=16)


# --- plotter-37 -----------------------------------------------------------------


def _landing(tab, name: str, **extra) -> dict:
    pixels = np.zeros((16, 16, 4), dtype=np.uint8)
    pixels[..., 3] = 255
    return {
        "tileset": Tileset(name=name, pixels=pixels, tile_w=16, tile_h=16),
        "source": "",
        "uid": tab.uid,
        **extra,
    }


def test_adding_a_tileset_keeps_the_view_unless_the_projection_changed():
    from realmspinner.studio.modes.plotter import mode as plotter_mode
    from realmspinner.studio.modes.plotter import tilesets as plotter_tilesets
    from realmspinner.studio.modes.plotter.engine import project

    from .test_plotter_mode import FakeCtx

    ctx = FakeCtx()
    tab = plotter_mode.new_document(ctx, (4, 4, 16, 16), projection=project.ORTHOGONAL)
    state = plotter_mode.ensure(ctx)

    # The first tileset of a map is the one arrival that has to frame it.
    tab.view.fitted = True
    plotter_tilesets.land_tileset(ctx, state, tab, _landing(tab, "first"))
    assert tab.view.fitted is False

    # A later one does not throw away the zoom and pan the user settled on.
    tab.view.fitted = True
    tab.view.zoom = 3.0
    tab.view.pan = (11.0, 7.0)
    plotter_tilesets.land_tileset(ctx, state, tab, _landing(tab, "fifth"))
    assert tab.view.fitted is True
    assert (tab.view.zoom, tab.view.pan) == (3.0, (11.0, 7.0))

    # A projection change moves the map's pixel extent, so that one refits --
    # even though this map already holds a tileset.
    assert tab.doc.projection == project.ORTHOGONAL
    tab.view.fitted = True
    plotter_tilesets.land_tileset(
        ctx, state, tab, _landing(tab, "iso", projection=project.ISOMETRIC)
    )
    assert tab.doc.projection == project.ISOMETRIC
    assert tab.view.fitted is False


def test_attaching_a_picture_to_an_image_layer_keeps_the_view():
    from realmspinner.studio.modes.plotter import mode as plotter_mode
    from realmspinner.studio.modes.plotter import tilesets as plotter_tilesets

    from .test_plotter_mode import FakeCtx

    ctx = FakeCtx()
    tab = plotter_mode.new_document(ctx, (4, 4, 16, 16))
    layer = tab.doc.add_image_layer("Backdrop")
    tab.view.fitted = True
    tab.view.zoom = 2.0
    picture = np.zeros((8, 8, 4), dtype=np.uint8)
    picture[..., 3] = 255
    plotter_tilesets.land_layer_image(ctx, tab, {"image": (layer.uid, "pic.png", picture)})
    assert tab.view.fitted is True
    assert tab.view.zoom == 2.0


# --- plotter-32 -----------------------------------------------------------------


def test_tmx_module_docstring_does_not_list_modelled_features_as_refused():
    doc = " ".join(tmx.__doc__.split())
    assert "Tiled's format is much larger than a finite stamp-and-fill editor" not in doc
    assert "all modelled now" in doc
    from realmspinner.studio.modes.plotter import engine

    pkg = " ".join(engine.__doc__.split())
    assert "The one outward import" not in pkg
    for name in ("core.safeio", "kernels.grid2d", "realmspinner.native"):
        assert name in pkg, f"the package docstring omits {name}"


# --- plotter-33 -----------------------------------------------------------------


def test_manual_does_not_claim_object_template_contents_load():
    from pathlib import Path

    text = (Path(__file__).resolve().parents[3] / "docs" / "manual" / "32-plotter.md").read_text(
        encoding="utf-8"
    )
    flat = " ".join(text.split())
    assert "object templates' *contents* all load" not in flat
    assert "object templates" in flat, "the refusal sentence must stay"
