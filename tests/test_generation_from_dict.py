"""``GenerationRequest.from_dict``'s sub-document coercion.

The module's docstring calls the request "safe to use from settings
migration, API validation, and worker planning code" -- but until the
2026-09-06 audit (finding create2-07), ``from_dict`` only cast its own
top-level scalar fields; ``TileSettings``, ``SpriteSettings`` and
``ModelSettings`` were built from raw sub-dict values with no casting at
all. A string-typed numeric field arriving from JSON (a form field, a
migrated settings row, an external caller) survived construction and then
made ``validate_request`` crash with an uncaught ``TypeError`` instead of
returning a ``CompatibilityIssue``. ``from_dict`` has no test home before
this file -- part of why the gap survived undetected.
"""

from __future__ import annotations

from realmspinner import generation


def test_from_dict_coerces_string_typed_tile_and_sprite_numerics():
    """The two working reproductions from the audit, as one regression."""
    tile_raw = {
        "generation_type": "tileset",
        "prompt": "a knight",
        "tile": {"mode": "collection", "prompt_items": ["grass"], "variants": "2"},
    }
    tile_req = generation.GenerationRequest.from_dict(tile_raw)
    assert tile_req.tile.variants == 2
    assert isinstance(tile_req.tile.variants, int)
    tile_issues = generation.validate_request(tile_req)
    assert all(issue.field != "tile.variants" for issue in tile_issues)

    sprite_raw = {
        "generation_type": "sprite_sheet",
        "prompt": "a knight",
        "sprite": {
            "mode": "action",
            "action": "walk",
            "directions": "8",
            "candidate_count": "1",
        },
    }
    sprite_req = generation.GenerationRequest.from_dict(sprite_raw)
    assert sprite_req.sprite.directions == 8
    assert isinstance(sprite_req.sprite.directions, int)
    assert sprite_req.sprite.candidate_count == 1
    assert isinstance(sprite_req.sprite.candidate_count, int)
    issues = generation.validate_request(sprite_req)
    assert all(
        issue.field not in ("sprite.directions", "sprite.candidate_count")
        for issue in issues
    )


def test_from_dict_coerces_model_settings_custom_triangles():
    """The finding's ``model`` sub-document, uncovered by either probe."""
    raw = {
        "generation_type": "3d_model",
        "prompt": "a knight",
        "model": {"output_profile": "raw", "custom_triangles": "5000"},
    }
    req = generation.GenerationRequest.from_dict(raw)
    assert req.model.custom_triangles == 5000
    assert isinstance(req.model.custom_triangles, int)


def test_a_non_numeric_tile_variants_refuses_instead_of_crashing():
    """Coercion is not acceptance: ``{"variants": "banana"}`` must reach
    ``validate_request`` as a ``CompatibilityIssue``, not an uncaught
    ``ValueError`` from ``int()`` -- that would just move the crash the
    audit found, not fix it."""
    raw = {
        "generation_type": "tileset",
        "prompt": "a knight",
        "tile": {"mode": "collection", "prompt_items": ["grass"], "variants": "banana"},
    }
    req = generation.GenerationRequest.from_dict(raw)
    issues = generation.validate_request(req)
    assert any(issue.field == "tile.variants" for issue in issues)


def test_a_non_numeric_sprite_directions_refuses_instead_of_crashing():
    raw = {
        "generation_type": "sprite_sheet",
        "prompt": "a knight",
        "sprite": {"mode": "action", "action": "walk", "directions": "banana"},
    }
    req = generation.GenerationRequest.from_dict(raw)
    issues = generation.validate_request(req)
    assert any(issue.field == "sprite.directions" for issue in issues)


def test_from_dict_round_trips_init_image_and_init_strength():
    """The 2026-09-11 audit, finding create-05.

    ``to_dict`` (a plain ``asdict``) faithfully wrote both fields, but
    ``from_dict``'s explicit ``cls(...)`` call never read them back out -- so
    a request round-tripped through ``to_dict()``/``from_dict()``, or any raw
    dict handed to ``service.jobs.create_generation_request`` (which coerces
    through this same door), silently lost its img2img intent and fell back
    to ``init_image=False, init_strength=None`` with no error and no visible
    signal, changing the actual output.
    """
    req = generation.GenerationRequest(
        generation_type="3d_model",
        prompt="a knight",
        references=("some/ref.png",),
        reference_mode="single",
        init_image=True,
        init_strength=0.55,
    )
    round_tripped = generation.GenerationRequest.from_dict(req.to_dict())
    assert round_tripped.init_image is True
    assert round_tripped.init_strength == 0.55

    # And the raw-dict door itself, which is the other half of the finding:
    # a plain dict with these keys set, never having been a GenerationRequest
    # at all, must not lose them either.
    raw = {
        "generation_type": "3d_model",
        "prompt": "a knight",
        "init_image": True,
        "init_strength": 0.4,
    }
    assert generation.GenerationRequest.from_dict(raw).init_image is True
    assert generation.GenerationRequest.from_dict(raw).init_strength == 0.4

    # False/unset must stay honest too -- a coercion bug that always turned
    # init_image on would be exactly as silent as one that always turned it
    # off.
    off = generation.GenerationRequest.from_dict({"prompt": "x"})
    assert off.init_image is False
    assert off.init_strength is None


def test_a_non_numeric_top_level_count_refuses_instead_of_crashing():
    """The 2026-09-13 audit, finding create-01.

    ``from_dict`` cast ``seed``, ``count`` and ``init_strength`` with a bare
    ``int()``/``float()``, so a non-numeric top-level scalar raised
    ``ValueError`` out of the constructor instead of surviving to be
    refused by ``validate_request`` as a ``CompatibilityIssue`` -- the same
    crash create2-07 fixed one level down, in ``TileSettings`` and friends.
    """
    raw = {"generation_type": "3d_model", "prompt": "a knight", "count": "banana"}
    req = generation.GenerationRequest.from_dict(raw)  # must not raise
    issues = generation.validate_request(req)
    assert any(issue.field == "count" for issue in issues)


def test_from_dict_does_not_explode_a_bare_string_reference_into_characters():
    """The 2026-09-13 audit, finding create-02.

    ``references`` and ``tile.prompt_items`` were built with
    ``tuple(str(x) for x in value)``, and ``str`` is itself iterable, so a
    bare string turned into one entry per character -- a short tileset
    prompt would pass validation and queue per-character materials.
    """
    raw = {
        "generation_type": "3d_model",
        "prompt": "a knight",
        "references": "some/ref.png",
    }
    req = generation.GenerationRequest.from_dict(raw)
    assert req.references == ("some/ref.png",)

    tile_raw = {
        "generation_type": "tileset",
        "prompt": "a knight",
        "tile": {"mode": "collection", "prompt_items": "grass"},
    }
    tile_req = generation.GenerationRequest.from_dict(tile_raw)
    assert tile_req.tile.prompt_items == ("grass",)


def test_validate_request_refuses_a_count_above_the_doors_own_ceiling():
    """The 2026-09-13 audit, finding create-07.

    ``validate_request`` refused ``count < 1`` but had no upper bound,
    though ``service._jobs_create.create_job`` enforces
    ``MAX_REFERENCE_COUNT`` -- so a request could clear this door and still
    be refused two steps later with no field pointed at until it did.
    """
    from realmspinner.service.validation import MAX_REFERENCE_COUNT

    req = generation.GenerationRequest(
        generation_type="3d_model", prompt="a knight", count=MAX_REFERENCE_COUNT + 1
    )
    issues = generation.validate_request(req)
    assert any(issue.field == "count" for issue in issues)


def test_cell_dimensions_is_gone_now_that_nothing_ever_called_it():
    """The 2026-09-16 audit, finding create-panes-03.

    ``cell_dimensions`` (docstring: "Return output dimensions; a blank
    target never reduces or upscales", enforcing the isometric-parity and
    no-upscale rules) had no caller anywhere in ``src/``, ``scripts/``,
    ``docs/`` or ``tests/`` -- the actual pixel-size resolution for a
    tileset/sprite request happens in
    ``pipelines/tilesheet.py``'s ``reduce_cell``/``reduce_sheet``
    (its own no-upscale check: "cannot be reduced to ...; generate it
    larger") and ``geometry`` (its own isometric-parity halving), an
    entirely separate implementation that has superseded this one. Deleted
    rather than wired in: the pipeline that would need to call it
    (``pipelines/tilesheet.py``, ``_q_tilesheet.py``) is outside this
    finding's owned files, and wiring a second, competing implementation
    into a path that already has one is not a fix.
    """
    assert not hasattr(generation, "cell_dimensions"), (
        "generation.cell_dimensions still exists; it was found unreachable "
        "from every real code path (superseded by pipelines/tilesheet.py's "
        "reduce_cell/reduce_sheet) and should have been deleted"
    )
