"""The properties panel's material editor (tranche 6: "UV and materials").

Three things this pins:

* the new PBR fields (emissive, alpha mode/cutoff, double-sided) fold into
  the **same** one-gesture-one-step write the base colour/metallic/roughness
  fields already made, proven by pressing exactly one of the new controls
  (double-sided, the simplest to isolate in a live headless frame) and
  checking the result is a single, complete replacement -- plus a source
  check that every new field actually reaches the ``replace(...)`` call, the
  same "bidirectional" shape ``test_clay_props_undo.py``'s own fold tests use;
* a texture slot assign/clear round-trips: Clear is a live control press,
  Assign is landed through :func:`clay_props.on_task_done` directly (the
  picker itself runs off the frame thread -- see ``shell/tasks.py``'s
  ``clay-mattex:`` branch -- so there is nothing for a headless imgui frame
  to press);
* every one of those is a *replacement*, never an in-place mutation, and
  every one is exactly one undo step.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest
from _ui_context import imgui_context

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio.modes.clay.ui.panes import props as clay_props
from realmspinner.studio.tasks import Done


@pytest.fixture
def ui(monkeypatch):
    with imgui_context(monkeypatch) as imgui:
        yield imgui


def _doc_with_object() -> tuple[bd.ClayDoc, bd.Obj]:
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=bp.box()))
    return doc, doc.by_uid(obj.uid)


class _FakeTab:
    def __init__(self, uid: str, doc: bd.ClayDoc) -> None:
        self.uid = uid
        self.doc = doc


class _FakeState:
    """Only ``state.get(uid)`` -- the one thing ``on_task_done`` reads off
    what ``clay_mode.ensure`` gives back."""

    def __init__(self, tab: _FakeTab | None) -> None:
        self._tab = tab

    def get(self, uid: str):
        return self._tab if self._tab is not None and uid == self._tab.uid else None


# --- the new scalar fields fold into one replace, one step -------------------


def test_every_new_pbr_field_reaches_the_replace_call() -> None:
    """Bidirectional, the way ``test_clay_props_widget.py``'s registry gates
    are: a field added to the material editor's write must actually be
    threaded from its own changed-flag into both the ``if`` guard and the
    ``replace(...)`` call, or a press would silently do nothing."""
    source = inspect.getsource(clay_props._material)
    for field, flag in (
        ("emissive_factor=", "emissive_changed"),
        ("alpha_mode=", "alpha_changed"),
        ("alpha_cutoff=", "cutoff_changed"),
        ("double_sided=", "ds_changed"),
    ):
        assert field in source, f"{field} is never written"
        assert flag in source, f"{flag} is never read"
    guard = source[source.index("if (") : source.index("):", source.index("if ("))]
    for flag in ("emissive_changed", "alpha_changed", "cutoff_changed", "ds_changed"):
        assert flag in guard, f"{flag} does not gate the write"


def test_toggling_double_sided_is_one_undo_step_and_a_full_replacement(
    monkeypatch: pytest.MonkeyPatch, ui
) -> None:
    doc, obj = _doc_with_object()
    original = doc.materials[obj.material]
    before_history = len(doc.history)

    # Isolates the one checkbox ``_material`` draws (double-sided) without
    # touching the colour/slider fields beside it, the same "monkeypatch the
    # one control kind this function calls once" shape
    # ``test_modifier_props.py``'s own enabled-checkbox test uses.
    monkeypatch.setattr(clay_props.controls, "checkbox", lambda label, value, **kw: (True, True))

    ctx = _FakeCtxForMaterial()
    tab = _FakeTab("bd1", doc)
    ui.new_frame()
    ui.begin("##host")
    clay_props._material(ctx, tab, doc, obj)
    ui.end()
    ui.end_frame()

    assert len(doc.history) == before_history + 1, "one press, one undo step"
    updated = doc.materials[obj.material]
    assert updated is not original, "a replacement, never an in-place edit"
    assert updated.double_sided is True
    # Everything else the panel did not touch survives the replace verbatim
    # (approx: a colour field round-trips through imgui's own float32 buffer,
    # which is not bit-identical to the float64 Python literal it started
    # from even when nothing was dragged).
    assert updated.base_color_factor == pytest.approx(original.base_color_factor)
    assert updated.metallic_factor == pytest.approx(original.metallic_factor)
    assert updated.roughness_factor == pytest.approx(original.roughness_factor)
    assert updated.emissive_factor == pytest.approx(original.emissive_factor)
    assert updated.alpha_mode == original.alpha_mode

    assert doc.undo()
    assert doc.materials[obj.material] is original


class _FakeCtxForMaterial:
    """``_material`` also calls ``_material_library``, which reads
    ``ctx.svc.config.home`` -- a real, per-instance temp directory is fine
    here since nothing is saved to it in this test (no Save/Apply button is
    pressed), but it still has to resolve without raising."""

    def __init__(self) -> None:
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace

        self.svc = SimpleNamespace(config=SimpleNamespace(home=Path(tempfile.mkdtemp())))

    def busy(self, key: str) -> bool:
        return False

    def submit(self, key: str, fn, *args, **kwargs) -> bool:
        return True

    def toast(self, message: str, level: str = "info") -> None:
        pass


# --- texture slots: assign (off-thread) and clear round-trip -----------------


def test_clearing_a_texture_slot_is_one_undo_step(monkeypatch: pytest.MonkeyPatch, ui) -> None:
    doc, obj = _doc_with_object()
    image = (2, 2, bytes(16))
    material = doc.materials[obj.material]
    from dataclasses import replace

    doc.set_material(obj.material, replace(material, base_color=image))
    before_history = len(doc.history)
    with_texture = doc.materials[obj.material]
    assert with_texture.base_color == image

    def _fake_disabled_button(label, enabled, *a, **kw):
        return "texclearbase_color" in label

    monkeypatch.setattr(clay_props.widgets, "disabled_button", _fake_disabled_button)
    ctx = _FakeCtxForMaterial()
    tab = _FakeTab("bd1", doc)
    ui.new_frame()
    ui.begin("##host")
    clay_props._texture_slots(ctx, tab, doc, obj.material, with_texture)
    ui.end()
    ui.end_frame()

    assert len(doc.history) == before_history + 1, "one click, one undo step"
    cleared = doc.materials[obj.material]
    assert cleared is not with_texture
    assert cleared.base_color is None
    assert doc.undo()
    assert doc.materials[obj.material].base_color == image


def test_assigning_a_texture_lands_through_on_task_done(monkeypatch: pytest.MonkeyPatch) -> None:
    doc, obj = _doc_with_object()
    tab = _FakeTab("bd1", doc)
    monkeypatch.setattr(clay_props.clay_mode, "ensure", lambda ctx: _FakeState(tab))
    before_history = len(doc.history)

    result = {"width": 2, "height": 2, "rgba": bytes(16)}
    done = Done(key=f"clay-mattex:{tab.uid}:{obj.material}:base_color", result=result)
    clay_props.on_task_done(ctx=None, done=done)

    assert len(doc.history) == before_history + 1
    material = doc.materials[obj.material]
    assert material.base_color == (2, 2, bytes(16))
    assert doc.undo()
    assert doc.materials[obj.material].base_color is None


def test_a_cancelled_picker_does_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    doc, obj = _doc_with_object()
    tab = _FakeTab("bd1", doc)
    monkeypatch.setattr(clay_props.clay_mode, "ensure", lambda ctx: _FakeState(tab))
    before_history = len(doc.history)

    done = Done(key=f"clay-mattex:{tab.uid}:{obj.material}:base_color", result=None)
    clay_props.on_task_done(ctx=None, done=done)

    assert len(doc.history) == before_history
    assert doc.materials[obj.material].base_color is None


def test_a_stale_tab_or_material_index_is_a_quiet_no_op(monkeypatch: pytest.MonkeyPatch) -> None:
    """The document can have closed, or the palette slot removed, while the
    picker's dialog was still open -- an ordinary race, not a failure."""
    doc, obj = _doc_with_object()
    result = {"width": 1, "height": 1, "rgba": bytes(4)}

    # No tab at all for this uid.
    monkeypatch.setattr(clay_props.clay_mode, "ensure", lambda ctx: _FakeState(None))
    clay_props.on_task_done(
        ctx=None, done=Done(key="clay-mattex:gone:0:base_color", result=result)
    )  # must not raise

    # A material index the palette no longer has.
    tab = _FakeTab("bd1", doc)
    monkeypatch.setattr(clay_props.clay_mode, "ensure", lambda ctx: _FakeState(tab))
    before_history = len(doc.history)
    clay_props.on_task_done(
        ctx=None, done=Done(key=f"clay-mattex:{tab.uid}:99:base_color", result=result)
    )
    assert len(doc.history) == before_history


def test_pick_texture_returns_none_on_a_cancelled_dialog(monkeypatch: pytest.MonkeyPatch) -> None:
    from realmspinner.studio import dialogs

    monkeypatch.setattr(dialogs, "open_file", lambda *a, **kw: None)
    assert clay_props._pick_texture("base_color") is None


def test_pick_texture_decodes_through_the_pixel_guard(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    import numpy as np
    from PIL import Image

    from realmspinner.studio import dialogs

    path = tmp_path / "swatch.png"
    Image.frombytes("RGBA", (3, 2), bytes(np.full((2, 3, 4), 128, dtype=np.uint8))).save(path)
    monkeypatch.setattr(dialogs, "open_file", lambda *a, **kw: path)

    result = clay_props._pick_texture("normal")
    expected_rgba = bytes(np.full((2, 3, 4), 128, dtype=np.uint8))
    assert result == {"width": 3, "height": 2, "rgba": expected_rgba}


# --- the material shelf listing is cached, not re-read every frame ----------


def test_material_library_listing_is_memoised_across_frames(
    monkeypatch: pytest.MonkeyPatch, ui
) -> None:
    """The 2026-09-19 audit, finding clay-35: ``clay_matlib.list_materials``
    -- a ``Path.glob`` plus one JSON parse per entry -- was called with no
    memoisation on every single frame ``_material_library`` draws, which is
    nearly every frame the properties panel shows an object with any
    material at all. Cost scales with how many materials the user has ever
    saved.
    """
    doc, obj = _doc_with_object()
    ctx = _FakeCtxForMaterial()
    tab = _FakeTab("bd1", doc)
    home = ctx.svc.config.home
    clay_props._invalidate_material_library(home)  # a clean slate for this home

    calls = {"n": 0}
    real_list = clay_props.clay_matlib.list_materials

    def _counting_list(path):
        calls["n"] += 1
        return real_list(path)

    monkeypatch.setattr(clay_props.clay_matlib, "list_materials", _counting_list)

    def _run_frame() -> None:
        ui.new_frame()
        ui.begin("##host")
        clay_props._material_library(ctx, tab, doc, obj)
        ui.end()
        ui.end_frame()

    _run_frame()
    assert calls["n"] == 1, "the first frame must read the shelf once"

    for _ in range(4):
        _run_frame()
    assert calls["n"] == 1, "unchanged frames must not re-read the shelf"

    def _typed_values():
        yield "Rusty Metal"
        while True:
            yield ""

    typed = _typed_values()
    monkeypatch.setattr(clay_props.widgets, "input_text", lambda *a, **kw: next(typed))

    _run_frame()  # types and *submits* a save (async, off the frame thread -- clay-09)
    assert calls["n"] == 1, "submitting a save must not itself force a re-read"

    # The save lands later, the way ``shell/tasks.py`` would deliver it --
    # only landing (not submitting) is what must invalidate the cache.
    clay_props.on_task_done(
        ctx=None,
        done=Done(key=clay_props._matlib_save_key(tab.uid), result=object(), tag=home),
    )
    _run_frame()
    assert calls["n"] == 2, "landing a save must invalidate the cache and force a re-read"

    for _ in range(4):
        _run_frame()
    assert calls["n"] == 2, "frames after the save, with nothing new changed, hit the cache again"


# --- the material shelf's Save/Apply run off the frame thread ---------------


def test_save_material_with_a_realistic_multi_slot_texture_set_is_submitted_through_ctx_submit(
    monkeypatch: pytest.MonkeyPatch, ui
) -> None:
    """The 2026-09-22 audit's clay-09: ``matlib.save_material``/``load_material``
    encode/decode PNGs (up to five slots, each potentially 4096^2) and were
    called inline from ``_material_library``, on the frame thread -- the
    audit's own reproduction measured 11.1 s for a save and 1.8 s for a load
    at that size. Asserted here as "submitted through ``ctx.submit``, never
    run inline" rather than a wall-clock budget, since a timing assertion is
    flaky under this suite's own parallel dist (``-n 8 --dist loadfile``).

    ``clay_matlib.save_material`` is monkeypatched to *raise* if ever called
    directly -- the only way it can be reached is inline, since the fake
    ``ctx.submit`` below records the call rather than invoking it, the same
    "prove it never runs on this thread" shape ``test_pick_texture_...``
    already gives the sibling texture-assign door.
    """
    doc, obj = _doc_with_object()

    def _must_not_run_inline(home, name, material):
        raise AssertionError("save_material must not be called inline on the frame thread")

    monkeypatch.setattr(clay_props.clay_matlib, "save_material", _must_not_run_inline)

    submitted: dict[str, Any] = {}

    class _RecordingCtx(_FakeCtxForMaterial):
        def submit(self, key: str, fn, *args, **kwargs) -> bool:
            submitted["key"] = key
            submitted["fn"] = fn
            submitted["args"] = args
            return True

    ctx = _RecordingCtx()
    tab = _FakeTab("bd1", doc)
    typed = iter(["Rusty Metal"])
    monkeypatch.setattr(clay_props.widgets, "input_text", lambda *a, **kw: next(typed, ""))

    ui.new_frame()
    ui.begin("##host")
    clay_props._material_library(ctx, tab, doc, obj)
    ui.end()
    ui.end_frame()

    assert submitted.get("fn") is clay_props.clay_matlib.save_material, (
        "save_material must be handed to ctx.submit, not called directly"
    )
    assert submitted["args"][1] == "Rusty Metal"
    assert submitted["key"].startswith(clay_props.MATLIB_SAVE_TASK_PREFIX)


def test_applying_a_library_material_is_submitted_through_ctx_submit_not_run_inline(
    monkeypatch: pytest.MonkeyPatch, ui, tmp_path
) -> None:
    """clay-09's Apply half: ``load_material`` decodes the same PNGs on the
    way back in, and was likewise called inline from the Apply button."""
    doc, obj = _doc_with_object()
    home = tmp_path
    entry = clay_props.clay_matlib.save_material(home, "Rusty Metal", doc.materials[obj.material])
    clay_props._invalidate_material_library(home)

    def _must_not_run_inline(h, entry_id):
        raise AssertionError("load_material must not be called inline on the frame thread")

    monkeypatch.setattr(clay_props.clay_matlib, "load_material", _must_not_run_inline)

    submitted: dict[str, Any] = {}

    class _RecordingCtx(_FakeCtxForMaterial):
        def __init__(self) -> None:
            super().__init__()
            self.svc.config.home = home

        def submit(self, key: str, fn, *args, **kwargs) -> bool:
            submitted["key"] = key
            submitted["fn"] = fn
            submitted["args"] = args
            return True

    ctx = _RecordingCtx()
    tab = _FakeTab("bd1", doc)
    monkeypatch.setattr(clay_props.widgets, "input_text", lambda *a, **kw: "")
    monkeypatch.setattr(
        clay_props.controls, "small_button", lambda label, **kw: "matlibapply" in label
    )

    ui.new_frame()
    ui.begin("##host")
    clay_props._material_library(ctx, tab, doc, obj)
    ui.end()
    ui.end_frame()

    assert submitted.get("fn") is clay_props.clay_matlib.load_material, (
        "load_material must be handed to ctx.submit, not called directly"
    )
    assert submitted["args"] == (home, entry.id)
    assert submitted["key"].startswith(clay_props.MATLIB_APPLY_TASK_PREFIX)


def test_matlib_save_task_invalidates_the_listing_cache_only_once_it_lands(tmp_path) -> None:
    home = tmp_path
    clay_props._cached_materials(home)  # populate the cache with an empty read
    assert home in clay_props._matlib_cache

    done = Done(key=clay_props._matlib_save_key("bd1"), result=object(), tag=home)
    clay_props.on_task_done(ctx=None, done=done)

    assert home not in clay_props._matlib_cache


def test_matlib_apply_task_applies_the_loaded_material_to_the_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace

    doc, obj = _doc_with_object()
    tab = _FakeTab("bd1", doc)
    monkeypatch.setattr(clay_props.clay_mode, "ensure", lambda ctx: _FakeState(tab))
    before_history = len(doc.history)

    material = replace(doc.materials[obj.material], name="Rusty Metal")
    key = clay_props._matlib_apply_key(tab.uid, obj.uid, "rusty-metal-abcd1234")
    clay_props.on_task_done(ctx=None, done=Done(key=key, result=material))

    assert len(doc.history) == before_history + 1
    landed = doc.by_uid(obj.uid)
    assert doc.materials[landed.material].name == "Rusty Metal"

    assert doc.undo()
    landed = doc.by_uid(obj.uid)
    assert doc.materials[landed.material].name != "Rusty Metal"
