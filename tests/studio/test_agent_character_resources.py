"""``agent_character_resources``: the one static vocabulary resource and the
per-sheet dynamic pair.

``cliplib.shipped_clip_*``/``clips.shipped_clip_timing`` and
``service.characters.ASSET_FILTERS`` are real on this branch now -- see
``tests/studio/test_agent_character.py``'s own module docstring for the same
finding -- so nothing in this file stubs them any more."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from warlock.mcp import rpc
from warlock.studio import agent_character_resources as acr


def test_the_vocabulary_resource_is_process_stable() -> None:
    """Same call, same bytes -- a pure function of the shipped registries,
    never of wall-clock time, randomness or a mutable module-level cache
    that could drift between two reads in the same process."""
    from warlock.service import troupe as svc_troupe

    mime1, body1 = acr.read_static(acr.VOCABULARY_URI)
    mime2, body2 = acr.read_static(acr.VOCABULARY_URI)
    assert mime1 == mime2 == "application/json"
    assert body1 == body2
    payload = json.loads(body1)
    assert "movements" in payload
    assert "families" in payload
    assert payload["sheet_uri_pattern"] == acr.SHEET_URI_RE.pattern
    # The custom-size door's own bound (master's 8b091e98 Send to Troupe),
    # alongside "sizes"' preset ladder -- both read fresh off the same
    # process-stable constant agent_character's own "size" schema uses, so
    # a client reading this resource once sees the exact range a size
    # outside "sizes" is still checked against.
    assert payload["size_range"] == list(svc_troupe.TROUPE_CUSTOM_SIZE_RANGE)


def test_a_uri_outside_the_strict_sheet_pattern_is_not_owned() -> None:
    assert acr.owns_uri(acr.VOCABULARY_URI)
    assert acr.owns_uri("warlock://character/sheet/" + "a" * 12 + "/" + "b" * 12 + "/atlas.png")
    assert acr.owns_uri("warlock://character/sheet/" + "a" * 12 + "/" + "b" * 12 + "/sidecar.json")
    # Not a sheet id shape, not the vocabulary uri, not a known part name.
    assert not acr.owns_uri("warlock://character/sheet/short/short/atlas.png")
    assert not acr.owns_uri("warlock://character/sheet/" + "a" * 12 + "/" + "b" * 12 + "/model.glb")
    assert not acr.owns_uri("warlock://clay/scene")
    assert not acr.owns_uri("warlock://character/vocabulary/")


def test_a_sheet_resource_reads_the_published_sidecar_and_atlas(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A real file on disk for the atlas, not a hand-rolled stand-in: fix 3
    added a ``.stat().st_size`` read before the PNG bytes are ever touched
    (see ``read_dynamic``'s own ``ATLAS_RESOURCE_MAX_BYTES`` check), and a
    fake ``Path`` that only implements ``read_bytes`` fails that call
    outright rather than reading a real (small, well-under-the-cap) size.
    """
    from warlock.service import sheets as svc_sheets

    job_id = "a" * 12
    sheet_id = "b" * 12
    record = {"frame_size": 64, "cells": []}
    png_bytes = b"\x89PNG\r\n\x1a\n"
    png_path = tmp_path / "atlas.png"
    png_path.write_bytes(png_bytes)

    def fake_get_sheet(svc, jid, sid):
        assert (jid, sid) == (job_id, sheet_id)
        return record

    def fake_sheet_png(svc, jid, sid):
        assert (jid, sid) == (job_id, sheet_id)
        return png_path

    monkeypatch.setattr(svc_sheets, "get_sheet", fake_get_sheet)
    monkeypatch.setattr(svc_sheets, "sheet_png", fake_sheet_png)

    uris = acr.sheet_uris(job_id, sheet_id)
    mime, body = acr.read_dynamic(object(), uris["sidecar"])
    assert mime == "application/json"
    assert json.loads(body) == record

    mime, body = acr.read_dynamic(object(), uris["atlas"])
    assert mime == "image/png"
    assert body == png_bytes


def test_an_unpublished_sheet_reads_as_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    from warlock.service import sheets as svc_sheets
    from warlock.service.errors import NotFound

    def raise_not_found(svc, jid, sid):
        raise NotFound("no such sheet")

    monkeypatch.setattr(svc_sheets, "get_sheet", raise_not_found)
    monkeypatch.setattr(svc_sheets, "sheet_png", raise_not_found)

    uris = acr.sheet_uris("a" * 12, "b" * 12)
    assert acr.read_dynamic(object(), uris["sidecar"]) is None
    assert acr.read_dynamic(object(), uris["atlas"]) is None
    # And a uri this module does not own at all is also None, never a raise.
    assert acr.read_dynamic(object(), "warlock://clay/scene") is None


def test_a_huge_atlas_resource_is_refused_and_points_to_the_preview(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Fix 3: an atlas over :data:`acr.ATLAS_RESOURCE_MAX_BYTES` reads as a
    small JSON error naming ``character_sheet_preview`` -- never the raw
    PNG, and never a bare ``None`` (which ``agent_host._read_resource``
    would report as ``not_found``, indistinguishable from a sheet that was
    never published at all -- see ``read_dynamic``'s own docstring for why
    that would be dishonest here). A real oversized file on disk, not a
    monkeypatched ``.stat()``, so the size this test relies on is the same
    one production code would actually see.
    """
    from warlock.service import sheets as svc_sheets

    job_id = "a" * 12
    sheet_id = "b" * 12
    png_path = tmp_path / "atlas.png"
    with open(png_path, "wb") as fh:
        fh.seek(acr.ATLAS_RESOURCE_MAX_BYTES + 1)
        fh.write(b"\x00")
    assert png_path.stat().st_size > acr.ATLAS_RESOURCE_MAX_BYTES

    def fake_sheet_png(svc, jid, sid):
        assert (jid, sid) == (job_id, sheet_id)
        return png_path

    monkeypatch.setattr(svc_sheets, "sheet_png", fake_sheet_png)

    uris = acr.sheet_uris(job_id, sheet_id)
    mime, body = acr.read_dynamic(object(), uris["atlas"])
    assert mime == "application/json"
    payload = json.loads(body)
    assert payload["use"] == "character_sheet_preview"
    assert payload["size"] == png_path.stat().st_size
    assert payload["size"] > payload["max_bytes"]

    # And a small atlas -- the ordinary case -- still reads as the image.
    small_path = tmp_path / "small.png"
    small_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    monkeypatch.setattr(svc_sheets, "sheet_png", lambda svc, jid, sid: small_path)
    mime2, body2 = acr.read_dynamic(object(), uris["atlas"])
    assert mime2 == "image/png"
    assert body2 == small_path.read_bytes()


def test_the_atlas_bound_is_one_rpc_frame() -> None:
    """Pins :data:`acr.ATLAS_RESOURCE_MAX_BYTES` to the exact formula fix 3
    specifies -- the same one-frame budget ``character_sheet_preview``
    already computes inline -- so the two cannot silently drift apart."""
    assert acr.ATLAS_RESOURCE_MAX_BYTES == (rpc.MAX_FRAME - 64 * 1024) * 3 // 4
