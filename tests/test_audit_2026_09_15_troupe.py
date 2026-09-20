"""Regression tests for the troupe fix-brief of the 2026-09-15 audit.

One module for the findings whose fix is a behaviour change with no existing
test file of its own to sit beside (troupe-03, troupe-05, troupe-08,
service-07). troupe-01, troupe-02 and troupe-04 have their regressions beside
the code they fix (``tests/modes/troupe/test_ulpc.py``,
``tests/service/test_pixel_sheet_service.py`` and ``tests/modes/troupe/test_troupe_imports.py``
respectively); troupe-06, troupe-07 and troupe-09 are docstring-only and carry
no regression test, per the orchestrator's note for this batch.
"""

from __future__ import annotations

import io
import json
import struct
import zlib

import pytest
from PIL import Image

from realmspinner.kernels.rig import store
from realmspinner.service import sprites as svc_sprites
from realmspinner.service import troupe as svc_troupe
from realmspinner.service.errors import Conflict, Invalid

# --- troupe-03: sheet_preview_png must validate before it allocates ---------


def _mesh_with_sheet(svc, *, frame_size=64, frames=2):
    """A finished character mesh plus a hand-built character sheet, with one
    movement ("walk") of *frames* frames -- the minimum ``sheet_preview_png``
    needs to compose a movement strip, the same shape
    ``tests/test_character_agent_doors.py``'s own ``_build_sheet`` uses."""
    from realmspinner.service import jobs as svc_jobs

    mesh_id = svc_jobs.create_job(svc, kind="text", prompt="a knight")["id"]
    svc.store.set_status(mesh_id, "done")
    job_dir = svc.job_dir(mesh_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"glb")

    sheet_id = store.new_id()
    columns, rows = 2, 1
    atlas = Image.new("RGBA", (frame_size * columns, frame_size * rows), (0, 0, 0, 255))
    png_path = store.sheet_png_path(job_dir, sheet_id)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    atlas.save(png_path, format="PNG")

    movements = [
        {
            "key": "walk", "label": "Walk", "frames": frames, "loop": False,
            "duration_ms": 100,
            "directions": [{"key": "front", "label": "Front", "yaw": 0.0}],
        }
    ]
    # ``start``/``end`` name real cells (0, 1); the corrupted claim in the
    # tests below is ``movements[0]["frames"]``, which is what
    # ``sheet_preview_png`` reads for the "whole movement" branch that has no
    # single ``chosen_run`` to bound it -- exactly the sidecar field troupe-03
    # is about.
    runs = [{"movement": "walk", "direction": "front", "yaw": 0.0, "start": 0, "end": 1}]
    cells = [{"index": i, "x": (i % columns) * frame_size, "y": 0, "w": frame_size, "h": frame_size}
             for i in range(columns * rows)]
    sidecar = {
        "version": 1, "id": sheet_id, "name": "preview test", "source_job": mesh_id,
        "created": 0.0, "image": png_path.name, "frame_size": frame_size,
        "columns": columns, "rows": rows, "width": frame_size * columns,
        "height": frame_size * rows, "yaws": [0.0], "poses": [], "cells": cells,
        "troupe": {"version": 3, "columns": columns, "movements": movements, "runs": runs,
                   "cell_count": len(cells)},
    }
    store.sheet_path(job_dir, sheet_id).write_text(json.dumps(sidecar), encoding="utf-8")
    return mesh_id, sheet_id


def test_sheet_preview_png_refuses_before_allocating_from_a_corrupted_sidecars_frame_count(
    svc, monkeypatch
):
    """The 2026-09-15 audit, finding troupe-03: everything ``sheet_preview_png``
    does to compose a movement strip used to allocate a fresh ``Image.new``
    sized straight from the sidecar's own ``frames``/``frame_size`` -- with
    ``max_side`` only downscaling the *finished* composite well after that
    allocation had already happened. The sidecar is not this door's own
    write: an agent names a job id it does not own, or the file is simply
    corrupted, and a claimed frame count this door never asked for should
    cost nothing before it is checked.

    ``PIL.Image.new`` is spied on rather than actually handed the corrupted
    size (which would either allocate a very large buffer or trip Pillow's
    own decompression-bomb guard, neither of which is safe or fast to run in
    a suite): the claim under test is squarely "never called before the
    refusal", which the spy proves directly.
    """
    mesh_id, sheet_id = _mesh_with_sheet(svc, frame_size=64, frames=2)
    job_dir = svc.job_dir(mesh_id)
    sidecar_path = store.sheet_path(job_dir, sheet_id)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    # Corrupted after the fact, the way a hand-edited or truncated sidecar
    # would be -- ``create_charsheet`` would never itself write a movement
    # this large.
    sidecar["troupe"]["movements"][0]["frames"] = 999_999
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    calls: list[tuple] = []
    real_new = Image.new

    def spy_new(mode, size, *a, **k):
        calls.append(size)
        return real_new(mode, size, *a, **k)

    monkeypatch.setattr(Image, "new", spy_new)

    from realmspinner.service import characters as svc_characters

    with pytest.raises(Invalid) as excinfo:
        svc_characters.sheet_preview_png(
            svc, mesh_id, sheet_id, max_side=64, movement="walk", max_bytes=200_000,
        )
    assert excinfo.value.field == "sheet_id"
    assert not calls, f"Image.new was reached before the refusal: {calls!r}"


def _giant_header_png() -> bytes:
    """A structurally valid PNG whose IHDR claims a huge width/height, built
    by patching the header of a genuine 1x1 PNG -- the same trick
    ``tests/test_agent_refs.py``'s own ``_giant_header_png`` uses for
    ``files.to_png``'s identical header-only guard. ``Image.open`` alone
    parses only this header, so a fake one is enough to exercise the refusal
    without the cost the refusal exists to avoid.
    """
    buf = io.BytesIO()
    Image.new("RGB", (1, 1), (0, 0, 0)).save(buf, "PNG")
    data = bytearray(buf.getvalue())
    assert data[12:16] == b"IHDR"
    width = height = 9000  # > kernels.sheet.MAX_ATLAS_PX (8192)
    struct.pack_into(">II", data, 16, width, height)
    crc = zlib.crc32(bytes(data[12:29])) & 0xFFFFFFFF
    struct.pack_into(">I", data, 29, crc)
    return bytes(data)


def test_sheet_preview_png_refuses_before_decoding_an_oversized_atlas_png(svc, monkeypatch):
    """service-queue-03 (the 2026-09-16 audit): every ceiling troupe-03 added
    above (``check_atlas_size`` against the sidecar's own numbers) bounds the
    *composed* preview -- and the ``movement is None`` (whole-atlas) branch
    never calls it at all -- but none of them ever reads ``png_path``'s own
    declared dimensions. A hand-edited, corrupted, or otherwise oversized
    ``sheet.png`` sitting beside a small, honest sidecar used to be decoded
    in full (``opened.load()``) before any ceiling on the file itself
    applied, in every branch including ``movement is None``.

    ``PIL.Image.Image.load`` is spied on rather than actually handed the
    fake 9000x9000 header (which would either allocate a very large buffer
    or trip Pillow's own decompression-bomb guard): the claim under test is
    squarely "never decoded before the refusal", which the spy proves
    directly -- the same shape the sidecar-corruption test above uses on
    ``Image.new``.
    """
    mesh_id, sheet_id = _mesh_with_sheet(svc, frame_size=64, frames=2)
    job_dir = svc.job_dir(mesh_id)
    png_path = store.sheet_png_path(job_dir, sheet_id)
    png_path.write_bytes(_giant_header_png())

    calls: list[bool] = []
    real_load = Image.Image.load

    def spy_load(self, *a, **k):
        calls.append(True)
        return real_load(self, *a, **k)

    monkeypatch.setattr(Image.Image, "load", spy_load)

    from realmspinner.service import characters as svc_characters

    with pytest.raises(Invalid) as excinfo:
        svc_characters.sheet_preview_png(
            svc, mesh_id, sheet_id, max_side=64, movement=None, max_bytes=200_000,
        )
    assert excinfo.value.field == "sheet_id"
    assert not calls, f"Image.load was reached before the refusal: {calls!r}"


# --- troupe-05: a sprite draft must not be deletable mid-publish ------------


def _reference(svc):
    job_id = svc.store.create("text", "a knight", {"seed": 1}, stage="reference")
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), (200, 200, 200)).save(job_dir / "input.png")
    svc.store.set_status(job_id, "done")
    return job_id


def _draft_mid_publish(svc, job_id):
    """The worker's own publish order, stopped halfway: both PNGs written,
    the sidecar (``list_sprite_drafts``' completion marker) not yet."""
    draft_id = store.new_id()
    job_dir = svc.job_dir(job_id)
    for letter in store.SPRITE_CANDIDATES:
        path = store.sprite_draft_png_path(job_dir, draft_id, letter)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGBA", (32, 32), (0, 0, 0, 0)).save(path)
    return draft_id


def test_deleting_a_sprite_draft_while_its_synthesis_is_still_publishing_leaves_no_orphan_files(
    svc,
):
    """The 2026-09-15 audit, finding troupe-05: ``delete_sprite_draft`` took
    no lock and made no in-flight check, unlike its ``sheets.py`` siblings.
    Expressed as state, the way
    ``test_deleting_a_sheet_while_its_restyle_is_running_does_not_resurrect_it``
    already does for the pixel-sheet door: a queued/running
    ``sprite_synthesis`` row targeting this draft is exactly the state a
    delete arriving mid-publish would see, and the delete must refuse rather
    than unlink whatever the worker has written so far.
    """
    job_id = _reference(svc)
    draft_id = _draft_mid_publish(svc, job_id)
    svc.store.create(
        "sprite_synthesis",
        "a knight",
        {"source_job": job_id, "draft_id": draft_id},
        status="running",
    )

    with pytest.raises(Conflict):
        svc_sprites.delete_sprite_draft(svc, job_id, draft_id)

    job_dir = svc.job_dir(job_id)
    assert all(
        store.sprite_draft_png_path(job_dir, draft_id, c).exists()
        for c in store.SPRITE_CANDIDATES
    )


# --- troupe-08: dead kin under an already-populated winged archetype -------


def test_imp_gargoyle_angel_and_harpy_never_offer_their_declared_kin():
    """The 2026-09-15 audit, finding troupe-08: each of these four is a
    "winged" archetype creature, and the registry already ships winged
    species (dragon, wyvern, bat, bird, raven, griffin) -- so
    ``resolve._plan_for`` never falls through to humanoid for them, and
    ``resolve._offer_for``'s pool is winged-only. Each one used to declare a
    humanoid ``kin`` (goblin, ogre, human) that ``add(creature.kin)`` could
    never add to that pool: dead data no offer sentence could ever surface.
    Checked directly against the source of truth -- no ``kin`` left to be
    unreachable -- and, for the general shape of the mistake, that whatever
    ``kin`` a creature declares must actually be inside the pool
    ``_offer_for`` computes for it.
    """
    from realmspinner.characters import family as family_mod
    from realmspinner.characters.resolve import KNOWN_CREATURES, _offer_for

    registry = family_mod.families()
    for name in ("imp", "gargoyle", "angel", "harpy"):
        creature = KNOWN_CREATURES[name]
        assert creature.kin == (), f"{name} still declares dead kin {creature.kin!r}"
        offer = _offer_for(creature, registry)
        assert not (set(creature.kin) & set(offer))


# --- service-07: Troupe HD mode must drop colors/palette/dither/outline -----


def test_hd_mode_drops_colors_palette_dither_outline_but_not_reduce_mode(svc):
    """The 2026-09-15 audit, finding service-07, was itself corrected by the
    2026-09-18 audit, finding troupe-01: this test used to assert
    ``reduce_mode`` was stripped too, on the belief that ``_q_troupe``'s HD
    branch never reads it. It does -- ``_render_charsheet`` (``_q_troupe.py``)
    calls ``pixelize.reduce_frames(..., mode=reduce_mode)`` to take the 512px
    render down to the logical size *before* the ``if not pixel_art`` branch
    in ``_quantise`` is ever reached, so stripping ``reduce_mode`` from HD
    rows silently forced every HD sheet through the "box" default regardless
    of what the user picked. ``colors``/``palette``/``dither``/``outline``
    really are unread on the HD path (no quantise, no palette, no outline
    pass), so those four still drop.
    """
    options = svc_troupe._check_options(svc, {"pixel_art": False})
    assert options["pixel_art"] is False
    for key in ("colors", "palette", "dither", "outline"):
        assert key not in options, f"{key!r} should not survive HD mode: {options!r}"
    assert "reduce_mode" in options, (
        "reduce_mode must survive HD mode: _render_charsheet reduces the "
        f"512px render with it before the pixel-art branch runs: {options!r}"
    )


def test_hd_mode_keeps_the_users_reduce_mode_because_render_charsheet_reduces_with_it_before_the_pixel_art_branch_runs(  # noqa: E501
    svc,
):
    """The 2026-09-18 audit, finding troupe-01: ``_check_options`` used to
    strip ``reduce_mode`` from every HD (``pixel_art=False``) row on the
    belief that ``_q_troupe``'s ``if not pixel_art`` branch never reads it.
    It doesn't need to -- ``_render_charsheet`` already reduced the 512px
    Blender render to the logical size with ``params["reduce_mode"]`` (or the
    "box" default) *before* that branch runs, on every Troupe sheet, HD or
    not. Stripping the field meant an HD request that asked for "point" (or
    any non-default mode) silently got "box" instead, with no error and
    nothing in the row to say so.
    """
    options = svc_troupe._check_options(
        svc, {"pixel_art": False, "reduce_mode": "point"}
    )
    assert options["reduce_mode"] == "point", (
        f"the user's reduce_mode should survive HD mode, not be silently "
        f"replaced with the 'box' default: {options!r}"
    )
