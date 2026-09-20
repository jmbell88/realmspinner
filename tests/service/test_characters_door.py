"""``service.troupe``'s doors, seen from the character rather than the mesh.

What a character sheet actually needs is a rig with clips authored for it. The
door used to ask a narrower question -- "is this the humanoid template?" -- and
that answer was right only for as long as ``humanoid`` was the one template with
a clip library.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from realmspinner import doctor
from realmspinner.characters import family as family_mod
from realmspinner.kernels.rig import cliplib, store, templates
from realmspinner.service import characters as svc_characters
from realmspinner.service import export as svc_export
from realmspinner.service import jobs as svc_jobs
from realmspinner.service import troupe as svc_troupe
from realmspinner.service.errors import Invalid, NotFound


def _rigged_mesh(svc, template):
    job_id = svc.store.create("image", "a ranger", {}, stage="model")
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"fake-glb")
    (job_dir / "rig.glb").write_bytes(b"fake-glb")
    (job_dir / "rig.json").write_text(json.dumps({"template": template}), "utf-8")
    svc.store.set_status(job_id, "done")
    return job_id


def test_create_charsheet_accepts_any_template_with_a_clip_library_and_refuses_one_without(
    svc, monkeypatch
):
    """**The refusal is about clips, not about ``humanoid``.**

    A family that ships its own skeleton and its own walk cycle was turned away
    for not being the one template that happened to have a library first -- and
    the message told it so, which sent the reader to change the rig rather than
    to author the clips. Now the door asks ``cliplib.clip_library`` (which
    answers "no clips" rather than failing, so somebody has to ask) and the
    sheet is expanded from the rig's *own* template.
    """
    # Nothing is authored for a fish, so the sheet is refused by what is
    # missing -- and the sentence names the rig rather than naming humanoid.
    #
    # ``fish`` and not ``quadruped``: the quadruped, bird and blob skeletons
    # gained authored clip libraries on 2026-09-05 for the character families,
    # so the example this test was written around stopped being an example of
    # anything. The claim is unchanged; only the template that still has no
    # clips is.
    fish = _rigged_mesh(svc, "fish")
    with pytest.raises(Invalid) as refused:
        svc_troupe.create_charsheet(svc, fish)
    assert "clip library" in str(refused.value)
    assert "fish" in str(refused.value)

    # The same rig, once clips exist for it, is accepted -- and the row is
    # minted on its own template, because that is the skeleton on disk.
    library = cliplib.clip_library(svc_troupe.TROUPE_TEMPLATE)
    monkeypatch.setattr(
        cliplib,
        "clip_library",
        lambda key: library if key == "fish" else {"poses": {}, "clips": []},
    )
    made = svc_troupe.create_charsheet(svc, fish)
    row = svc.store.get(made["id"])
    assert row["params"]["template"] == "fish"

    # And a rig whose template is not recorded at all is still refused, rather
    # than raising the ``ValueError`` ``get_template`` answers an unknown key
    # with -- from the user's side it is the same missing clip library.
    monkeypatch.undo()
    nameless = _rigged_mesh(svc, "")
    with pytest.raises(Invalid, match="clip library"):
        svc_troupe.create_charsheet(svc, nameless)


# --- the character door ------------------------------------------------------
#
# ``service.characters`` is one press that produces a body, a skeleton and a
# queued sheet. What these pin is the *order* it does that in and the fact that
# nothing in it is humanoid-shaped.

@pytest.fixture
def blender(monkeypatch):
    """Blender answered present. Every character needs a rig, so without this
    every test below would be testing the refusal instead of the door."""

    class _Ok:
        ok = True
        detail = ""

    monkeypatch.setattr(doctor, "blender_check", lambda *a, **k: _Ok())
    return _Ok()


#: One species per archetype, and the point of the list. A door that reads the
#: rig template off a constant passes the first row and fails the other three.
ONE_PER_ARCHETYPE = ["human", "wolf", "dragon", "slime"]


def _recipe(family_key, **changes):
    """The smallest recipe that renders: one animation, one direction.

    Deliberately not ``DEFAULT_RECIPE`` -- that is 144 cells, and the plan this
    door runs up front walks every one of them. The claims here are about rows
    and params, none of which change with the cell count.
    """
    fam = family_mod.get_family(family_key)
    return {
        "family": family_key,
        "theme": fam.themes[0].key,
        "animations": {"idle": 2},
        "directions": 1,
        "logical_size": 32,
        "colors": 16,
        **changes,
    }


@pytest.mark.parametrize("family_key", ONE_PER_ARCHETYPE)
def test_a_character_mints_a_built_done_model_row_and_one_rig_row(svc, blender, family_key):
    """**One press, two rows -- for every archetype, not just humanoids.**

    The mesh row is minted finished the way ``import_mesh`` mints one: there is
    no reconstruction behind a generated body, so queueing an image job for it
    would spend two minutes of GPU reproducing what this door just built.

    Parameterised across all four body plans because the first draft of a door
    like this reads its rig template and its clip library off
    ``TROUPE_TEMPLATE``, passes for a human and mints a wolf on a humanoid
    skeleton -- which fails an hour later in the worker as a frame-count error.
    """
    made = svc_characters.create_character(svc, _recipe(family_key))
    assert made["kind"] == "character"

    row = svc.store.get(made["id"])
    assert row["stage"] == "model"
    assert row["status"] == "done"
    assert row["params"]["built"] is True
    assert row["params"]["asset_intent"] == "character"
    assert row["params"]["family"] == family_key
    job_dir = svc.job_dir(made["id"])
    assert (job_dir / "model.glb").exists()
    assert (job_dir / "source.glb").exists()
    assert (job_dir / "character.json").exists()

    rig = svc.store.get(made["rig"])
    assert rig["kind"] == "rig"
    assert rig["params"]["source_job"] == made["id"]
    # The species' own skeleton and the species' own clips, read off the
    # archetype. This is the assertion the parameterisation exists for.
    arch = family_mod.get_family(family_key).arch
    assert rig["params"]["template"] == arch.template
    assert rig["params"]["troupe_sheet"]["template"] == arch.template

    # And exactly two rows: no third row minted eagerly. The sheet is the
    # worker's to mint on the finished rig, so it cancels on its own.
    assert len(svc.store.list(limit=50)) == 2


def test_a_character_recipe_asking_for_hit_and_death_creates_its_sheet_row(svc, blender):
    """A recipe may name any clip its archetype's own library defines, per
    ``dev/measurements/2026-09-12-troupe-open-clip-vocabulary.md`` -- not
    only the closed legacy five (``idle``/``walk``/``run``/``attack``/``jump``)
    ``charsheet.ANIMATIONS`` used to enumerate. Naming ``hit``/``death`` --
    real clips the humanoid archetype's library defines -- gets a sheet row
    whose layout names them; at HEAD, ``Recipe``'s closed vocabulary check
    refused either name before a character could even be built.
    """
    made = svc_characters.create_character(
        svc, _recipe("human", animations={"hit": 4, "death": 6})
    )
    assert made["kind"] == "character"
    row = svc.store.get(made["id"])
    assert row["status"] == "done"
    rig = svc.store.get(made["rig"])
    spec = rig["params"]["troupe_sheet"]
    assert {m["key"] for m in spec["layout"]["movements"]} == {"hit", "death"}


def test_an_hd_recipe_sends_an_hd_sheet(svc, blender):
    """``Recipe.pixel_art`` (D5 HD mode) used to be dropped on the floor
    between the recipe and ``send_to_troupe``: colours, outline and dither
    rode along regardless of the switch, and an HD request refuses any of
    them outright (``service.troupe._check_options``: "pixel_art is off, so
    there is no outline mode to set"). The recipe's own ``outline`` default
    is "outer", not "none", so an HD recipe that touched nothing else used to
    fail the very call ``create_character`` made to build it.
    """
    made = svc_characters.create_character(svc, _recipe("human", pixel_art=False))
    spec = svc.store.get(made["rig"])["params"]["troupe_sheet"]
    assert spec["pixel_art"] is False
    for key in ("colors", "palette", "dither", "outline"):
        assert key not in spec
    # The size ladder still applies -- an HD sheet is still laid out at a
    # chosen cell size, it is just never reduced into one.
    assert spec["logical_size"] == 32


def test_the_rig_row_carries_the_exact_joints_and_never_measures_them(svc, blender):
    """A family states its skeleton exactly; measuring would guess it again.

    ``joints="measured"`` reads joint positions off a *reference image*, which
    this character never had -- there is no reference, only a recipe. Passing
    ``bones`` withholds that flag in ``send_to_troupe``, and the pin is here as
    well as there because this is the one door that always has an exact answer.
    """
    made = svc_characters.create_character(svc, _recipe("human"))
    params = svc.store.get(made["rig"])["params"]
    assert "joints" not in params, "the exact skeleton was thrown away and re-measured"
    stored = params["bones"]
    bones = stored["bones"] if isinstance(stored, dict) else stored
    names = [b["name"] for b in bones]
    template = templates.get_template("humanoid")
    assert names == [b["name"] for b in template.bones]
    # Real coordinates, not a stub: a skeleton collapsed to the origin would
    # satisfy every structural assertion above and skin the whole mesh to one
    # point.
    assert any(abs(v) > 1e-6 for b in bones for v in list(b["head"]) + list(b["tail"]))


def test_a_character_refuses_a_rerun_and_reroll_character_is_the_door_instead(svc, blender):
    """``rerun_job`` has nothing to re-run: ``built`` says so.

    The compensation is :func:`reroll_character`, and it is a door that copies
    ``params`` -- which puts it under
    ``test_rerun_regressions::test_every_door_that_copies_params_rerolls_the_seeds``.
    A reroll that reproduced the recipe's seed would look like it ran, take the
    time, and hand back a byte-identical character.
    """
    made = svc_characters.create_character(svc, _recipe("human", seed=4242))
    with pytest.raises(Invalid, match="built"):
        svc_jobs.rerun_job(svc, made["id"], mode="reroll")

    again = svc_characters.reroll_character(svc, made["id"])
    assert again["id"] != made["id"]
    before = svc.store.get(made["id"])["params"]
    after = svc.store.get(again["id"])["params"]
    assert before["character"]["recipe"]["seed"] == 4242
    assert after["character"]["recipe"]["seed"] != 4242, "the recipe seed came through unchanged"
    assert after["mesh_seed"] != before["mesh_seed"], "the mesh seed came through unchanged"
    # Same character, different roll: everything else in the recipe survives.
    assert {k: v for k, v in after["character"]["recipe"].items() if k != "seed"} == {
        k: v for k, v in before["character"]["recipe"].items() if k != "seed"
    }


def test_a_re_render_of_a_subset_keeps_the_recipe_seed_byte_for_byte(svc, blender):
    """**The flames in twelve re-rendered cells must match the ones beside them.**

    ``rerender_charsheet`` copies the row's params wholesale and strips only
    what the *worker* recorded, so the nested character block -- and the recipe
    seed inside it that drives every procedural effect -- rides through
    untouched. Exactly the pinned-palette argument: a door that re-rolled it
    would produce a subset whose effects are a different draw from the cells it
    is landing among.
    """
    made = svc_characters.create_character(svc, _recipe("human"))
    block = svc.store.get(made["rig"])["params"]["troupe_sheet"]["character"]

    job_dir = svc.job_dir(made["id"])
    (job_dir / "rig.glb").write_bytes(b"fake-glb")
    (job_dir / "rig.json").write_text(json.dumps({"template": "humanoid"}), "utf-8")
    sheet = svc_troupe.create_charsheet(svc, made["id"], character=block)
    sheet_params = svc.store.get(sheet["id"])["params"]
    # The sheet has to exist on disk for the re-render door to read its layout.
    store.sheet_dir(job_dir).mkdir(parents=True, exist_ok=True)
    store.sheet_path(job_dir, sheet["sheet_id"]).write_text(
        json.dumps({"troupe": sheet_params["layout"]}), "utf-8"
    )
    store.sheet_png_path(job_dir, sheet["sheet_id"]).write_bytes(b"png")

    again = svc_troupe.rerender_charsheet(
        svc,
        made["id"],
        sheet_id=sheet["sheet_id"],
        subset=[{"animation": "idle", "direction": "front"}],
    )
    copied = svc.store.get(again["id"])["params"]["character"]
    assert copied["recipe"]["seed"] == block["recipe"]["seed"]
    assert copied == block


def test_the_character_block_rides_every_row_of_the_chain(svc, blender):
    """Model, rig and sheet all say who this is -- and it is always *nested*.

    Flattened, any of ``seed``, ``palette`` or ``colors`` inside the recipe
    would be one entry away from being picked up by ``VECTOR_PARAMS``, which is
    an allowlist of flat settings, and quietly becoming a rerun vector for a row
    that never asked for one.
    """
    from realmspinner.vectors import VECTOR_PARAMS

    made = svc_characters.create_character(
        svc, _recipe("wolf", name="Fang"), prompt="a grey wolf"
    )

    model = svc.store.get(made["id"])["params"]
    rig = svc.store.get(made["rig"])["params"]
    block = model["character"]
    assert block["version"] == svc_characters.CHARACTER_BLOCK_VERSION
    assert block["recipe"]["family"] == "wolf"
    assert block["prompt"] == "a grey wolf"
    assert block["resolution"]["family"] is None  # nothing was resolved
    assert rig["troupe_sheet"]["character"] == block

    # And the third row, minted the way the worker mints it -- from the block
    # the rig row is carrying.
    job_dir = svc.job_dir(made["id"])
    (job_dir / "rig.glb").write_bytes(b"fake-glb")
    (job_dir / "rig.json").write_text(json.dumps({"template": "quadruped"}), "utf-8")
    sheet = svc_troupe.create_charsheet(
        svc, made["id"], character=rig["troupe_sheet"]["character"]
    )
    assert svc.store.get(sheet["id"])["params"]["character"] == block

    # Nested, and provably so. ``VECTOR_PARAMS`` is an allowlist of *flat*
    # settings, and the block is not on it -- nor is any recipe setting a
    # top-level key of the rows that carry it, which is what would have to be
    # true for one of them to be picked up as a rerun vector.
    from realmspinner.service.validation import DERIVED_PARAMS

    assert "character" not in VECTOR_PARAMS
    assert not set(model) & set(VECTOR_PARAMS)
    flattened = {"palette", "colors", "logical_size", "outline", "animations", "appearance"}
    assert not flattened & set(model)
    assert not flattened & set(rig)
    # And it is the *request*, not something the worker learned: stripping it
    # on a rerun would leave a sheet that no longer knows who it depicts.
    assert "character" not in DERIVED_PARAMS


def test_a_bad_recipe_costs_the_request_and_leaves_no_directory(svc, blender):
    """The whole ordering argument, in one assertion.

    An unmakeable character must cost the request -- not a mesh, not a rig, not
    144 EEVEE frames. So every refusal that can be raised without building
    anything is raised first, and the one failure that cannot (the build itself)
    takes its directory with it on the way out.
    """
    before = sorted(p.name for p in svc.config.data_dir.iterdir())

    with pytest.raises(Invalid) as refused:
        svc_characters.create_character(svc, _recipe("human", colors=7))
    assert refused.value.field == "colors"
    with pytest.raises(Invalid) as unknown:
        svc_characters.create_character(svc, _recipe("human", theme="chartreuse"))
    assert unknown.value.field == "theme"

    assert svc.store.list(limit=50) == []
    assert sorted(p.name for p in svc.config.data_dir.iterdir()) == before


def test_a_build_that_fails_halfway_takes_its_directory_with_it(svc, blender, monkeypatch):
    """``import_mesh``'s cleanup, for the same reason: a disk-full mid-write
    must not leave a truncated orphan the library then lists."""
    from realmspinner.characters import instantiate as instantiate_mod

    real = instantiate_mod.instantiate
    seen = []

    def _explode(recipe, out_dir):
        real(recipe, out_dir)
        seen.append(Path(out_dir))
        raise OSError("no space left on device")

    monkeypatch.setattr(instantiate_mod, "instantiate", _explode)
    with pytest.raises(OSError):
        svc_characters.create_character(svc, _recipe("human"))
    assert seen and not seen[0].exists()
    assert svc.store.list(limit=50) == []


def test_create_character_leaves_no_orphaned_row_when_send_to_troupe_fails(
    svc, blender, monkeypatch
):
    """troupe-01 (2026-09-07 audit): the door's own docstring claims "the mesh
    is built only once nothing left can refuse it", but ``send_to_troupe`` was
    called after the mesh row was already committed with no cleanup around it
    -- a failure there left a permanently orphaned "done" mesh row with no rig
    and no sheet. This must cost the request exactly as an earlier refusal
    does: no row, no directory.
    """
    monkeypatch.setattr(
        svc_troupe,
        "send_to_troupe",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("rig door exploded")),
    )
    before = sorted(p.name for p in svc.config.data_dir.iterdir())
    with pytest.raises(RuntimeError, match="rig door exploded"):
        svc_characters.create_character(svc, _recipe("human"))
    assert svc.store.list(limit=50) == [], "the mesh row was left orphaned"
    assert sorted(p.name for p in svc.config.data_dir.iterdir()) == before


def test_without_blender_a_character_is_refused_in_the_rig_segments_words(svc, monkeypatch):
    """One wording for "this needs Blender", wherever the app meets it -- and
    the refusal lands *before* the mesh, because a body whose skeleton can never
    be built is the half-made asset this door's ordering exists to prevent."""

    class _Missing:
        ok = False
        detail = "no bpy"

    monkeypatch.setattr(doctor, "blender_check", lambda *a, **k: _Missing())
    with pytest.raises(Invalid) as refused:
        svc_characters.create_character(svc, _recipe("human"))
    assert str(refused.value) == "Rigging needs Blender, which is not installed."
    assert svc.store.list(limit=50) == []


# --- the exported package ----------------------------------------------------


def _sheet_on_disk(svc, job_id):
    sheet_id = store.new_id()
    job_dir = svc.job_dir(job_id)
    store.sheet_dir(job_dir).mkdir(parents=True, exist_ok=True)
    store.sheet_png_path(job_dir, sheet_id).write_bytes(b"png-bytes")
    store.sheet_path(job_dir, sheet_id).write_text(json.dumps({"cells": []}), "utf-8")
    return sheet_id


def test_an_exported_package_lands_as_a_pair_and_leaves_no_temp(svc, blender, tmp_path):
    """**Both files or neither.** The PNG is the atlas and the JSON is what says
    which cell is ``walk`` facing south-east; a folder holding one without the
    other holds an asset nothing can interpret. The copy is staged for
    ``export.staged_copy``'s reason -- the destination is a watched project
    folder, and a torn read there is a hot-reloading engine's crash."""
    made = svc_characters.create_character(svc, _recipe("human", name="Ranger"))
    sheet_id = _sheet_on_disk(svc, made["id"])

    dest = tmp_path / "project" / "assets"
    out = svc_characters.export_package(svc, made["id"], sheet_id, dest_dir=dest)

    assert Path(out["png"]).read_bytes() == b"png-bytes"
    assert json.loads(Path(out["json"]).read_text("utf-8")) == {"cells": []}
    assert Path(out["png"]).stem == Path(out["json"]).stem == "Ranger"
    assert sorted(p.name for p in dest.iterdir()) == ["Ranger.json", "Ranger.png"]

    # Neither, when the pair cannot be completed: the sidecar's copy fails and
    # the PNG must not be standing at the destination on its own.
    real_copy = shutil.copyfile

    def _half(src, dst, *a, **k):
        if str(src).endswith(".json"):
            raise OSError("no space left on device")
        return real_copy(src, dst, *a, **k)

    other = tmp_path / "second"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(svc_export.shutil, "copyfile", _half)
        with pytest.raises(OSError):
            svc_characters.export_package(svc, made["id"], sheet_id, dest_dir=other)
    assert list(other.iterdir()) == []


def test_export_refuses_without_an_export_folder_in_the_librarys_exact_words(svc, blender):
    """``export_to_folder``'s sentence verbatim. Two wordings for one condition
    is how a user comes to believe they are two different problems."""
    made = svc_characters.create_character(svc, _recipe("human"))
    sheet_id = _sheet_on_disk(svc, made["id"])
    assert svc.config.export_dir is None

    with pytest.raises(NotFound) as refused:
        svc_characters.export_package(svc, made["id"], sheet_id)
    assert str(refused.value) == "no export folder configured (set REALMSPINNER_EXPORT_DIR)"


def test_a_preview_is_a_temp_glb_and_never_a_row(svc):
    """A slider drag must not mint a library row per frame. Keyed by the whole
    recipe, so dragging back to where it was is a hit rather than a rebuild."""
    path = svc_characters.preview_character(svc, _recipe("slime"))
    assert path.exists() and path.suffix == ".glb"
    assert path.parent == svc.config.data_dir / "tmp"
    assert svc.store.list(limit=50) == []
    assert svc_characters.preview_character(svc, _recipe("slime")) == path
    assert svc_characters.preview_character(svc, _recipe("ooze")) != path
    # And no scratch directory left behind beside them.
    assert all(p.is_file() and p.suffix == ".glb" for p in path.parent.iterdir())


def test_preview_character_lands_atomically_even_when_a_concurrent_build_won_the_race(
    svc, monkeypatch
):
    """service-02 (2026-09-07 audit): ``shutil.move`` onto a destination that
    already exists degrades to non-atomic ``copy2`` on Windows -- exactly the
    concurrent-build hazard the surrounding comment claimed was handled. Two
    previews racing the same recipe both pass the ``dest.exists()`` check
    before either has built anything, so the second one's final placement
    lands on a destination the first one already created; that placement has
    to be ``os.replace``, atomic there, and never ``shutil.move``.
    """
    import pathlib

    recipe = _recipe("slime")
    spec = svc_characters._recipe(recipe)
    digest = hashlib.sha256(
        json.dumps(spec.as_dict(), sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    dest = svc.config.data_dir / "tmp" / f"character-preview-{digest}.glb"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"the earlier racer's build")

    # The exact race window: this preview's own ``dest.exists()`` check ran
    # before the other build landed (the read that lost the race), but by the
    # time it is ready to place its own file, a destination is already there.
    real_exists = pathlib.Path.exists
    monkeypatch.setattr(
        pathlib.Path,
        "exists",
        lambda self: False if self == dest else real_exists(self),
    )
    monkeypatch.setattr(
        svc_characters.shutil,
        "move",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("shutil.move is not atomic once the destination exists")
        ),
    )

    result = svc_characters.preview_character(svc, recipe)
    assert result == dest
    assert dest.read_bytes() != b"the earlier racer's build"


def test_character_preview_cache_is_swept_or_bounded(svc):
    """service-04 (the 2026-09-20 audit): ``preview_character`` writes one
    ``character-preview-<hash>.glb`` per recipe under ``data_dir/tmp`` and
    nothing in the tree ever deleted one -- unbounded growth from exactly the
    iterate-on-a-recipe workflow the preview button exists for.
    """
    tmp_dir = svc.config.data_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    # Seed more stale previews than the cache is ever meant to hold, each
    # with its own mtime so a sweep has an oldest and a newest to tell apart.
    stale_total = svc_characters._PREVIEW_CACHE_CAP + 10
    for i in range(stale_total):
        stale = tmp_dir / f"character-preview-stale{i:04d}.glb"
        stale.write_bytes(b"stale")
        os.utime(stale, (i, i))

    svc_characters.preview_character(svc, _recipe("slime"))

    remaining = list(tmp_dir.glob("character-preview-*.glb"))
    assert len(remaining) <= svc_characters._PREVIEW_CACHE_CAP + 1
    # The oldest stale files are the ones a bounded cache must have reclaimed.
    names = {p.name for p in remaining}
    assert "character-preview-stale0000.glb" not in names


def test_the_estimate_grows_with_the_cells_and_is_never_zero(svc):
    """A form has to say something before the press. It is an estimate and
    labelled one -- but a *character* reported as instant is a user who thinks
    the button did nothing."""
    assert svc_characters.estimate_minutes(0) == pytest.approx(svc_characters.RIG_MINUTES)
    assert svc_characters.estimate_minutes(144) > svc_characters.estimate_minutes(16)
    assert svc_characters.estimate_minutes(-5) == svc_characters.estimate_minutes(0)


def test_character_options_answers_for_every_archetype_and_species(svc):
    """One source for the whole form. A picker built from a subset of the
    registry is a picker that cannot offer three quarters of what ships."""
    options = svc_characters.character_options(svc)
    assert {a["key"] for a in options["archetypes"]} == set(family_mod.archetypes())
    assert len(options["families"]) == len(family_mod.families())
    assert set(options["channels"]) == set(family_mod.families())
    assert options["default_recipe"]["family"]
    assert options["troupe"]["camera_presets"]


def test_a_finished_character_offers_poser_once_it_is_rigged_and_clay_before(svc):
    """The card's one action, for the intent this door writes.

    Never "rig": the rig row was minted in the same press, and offering to make
    a second one is how a user spends Blender twice on one character. That is
    exactly what the generic ladder below the arm answers for the window
    between the mesh landing and the rig landing.
    """
    from realmspinner.studio import state

    body = {
        "status": "done",
        "stage": "model",
        "files": ["model.glb"],
        "params": {"asset_intent": "character"},
    }
    assert state.primary_action(body) == "clay"
    rigged = {**body, "files": ["model.glb", "rig.glb"]}
    assert state.primary_action(rigged) == "poser"
    # And the label exists, because the library looks it up by name.
    assert state.ACTIONS["poser"]
