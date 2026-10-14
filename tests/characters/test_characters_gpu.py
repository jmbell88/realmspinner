"""The character-families programme, exercised against a real Blender.

**Marked ``gpu``, and the marker is about the lane rather than the hardware.**
All anything here needs is ``bpy``. No checkpoint is loaded, no VRAM budget is
taken, nothing is downloaded: the rig is minutes of CPU, and the frames that
are rendered are an EEVEE rasterisation on whatever display adapter the machine
already has. A user with none of the optional weights and no network can still
press one button and get a finished, rigged character. These tests are in the
``gpu`` lane only because they cost real minutes of Blender and must not run
inside ``uv run pytest``; run them with ``uv run pytest -m gpu -n 0``.

``pytest.importorskip("bpy")`` is the skip, exactly as ``tests/test_rigging.py``
and ``tests/test_sheet.py`` do it for the same reason -- bpy is an optional
cp313-only extra.

**Two ways in, deliberately.** The end-to-end test goes through the real door
(``service.characters.create_character``) and the real worker stage
(``Worker._process`` on the rig row it minted), which runs Blender out of
process exactly as the app does. The others drive ``blender_worker`` in *this*
process, because they are about what happens inside Blender -- material slots
and weighting method.

The gpu lane's ``REALMSPINNER_HOME`` exemption is why every test here takes
``svc`` or ``tmp_path`` and never writes near the developer's real library.
"""

from __future__ import annotations

import contextlib
import io
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from realmspinner.characters import instantiate as instantiate_mod
from realmspinner.characters import recipe as recipe_mod
from realmspinner.characters.family import get_family
from realmspinner.kernels.rig import blender_spec, store

# One species per body plan. The registry has 31 and rigging all of them would
# be a different test; these four are the four *templates* -- humanoid,
# quadruped, bird, blob -- which is the axis every claim below actually varies
# along, because the template is what the weighting solve and the clip library
# are keyed on.
ARCHETYPE_SPECIES = ("ogre", "wolf", "dragon", "slime")


def _quiet(fn, *args, **kwargs):
    """Run *fn* with bpy's chatter off. bpy prints per import and per export."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def _recipe_for(species: str, **changes: Any) -> recipe_mod.Recipe:
    """``DEFAULT_RECIPE`` moved onto another species, its own look and defaults.

    Not ``replace(family=...)`` alone: the theme keys and the appearance channel
    set both belong to the species, so a recipe carrying the ogre's ``fire`` and
    the ogre's ``bulk`` would be refused by ``Recipe.from_dict`` for a wolf.
    """
    fam = get_family(species)
    return recipe_mod.DEFAULT_RECIPE.replace(
        family=species,
        theme=fam.themes[0].key,
        appearance=fam.appearance_defaults(),
        **changes,
    )


def _built_and_rigged(bpy: Any, spec: recipe_mod.Recipe, out_dir: Path) -> dict[str, Any]:
    """Instantiate the recipe into *out_dir* and rig it, in this process.

    Returns ``op_rig``'s own result dict. The joints handed to the rig are the
    instance's exact ones, which is what ``service.characters`` does at the real
    door -- a reference image a generated character never had is not asked to
    place them.
    """
    from realmspinner.pipelines import blender_worker

    fam = spec.spec
    instance = _quiet(instantiate_mod.instantiate, spec, out_dir)
    result = _quiet(
        blender_worker.op_rig,
        bpy,
        blender_spec.rig_spec(out_dir, fam.template, bones=instance.joints),
    )
    store.finalize_rig(out_dir)
    return result


def _opaque_colours(image: Image.Image) -> Counter:
    """``rgb -> pixel count`` over everything that is not fully transparent.

    Alpha is dropped rather than counted: a fully transparent pixel carries
    whatever RGB the compositor left in it, and counting those would make an
    empty cell look like it had a palette.
    """
    return Counter(px[:3] for px in image.get_flattened_data() if px[3] > 0)


def _srgb_byte(linear_byte: int) -> int:
    """The byte a glTF ``baseColorFactor`` component renders as in the PNG.

    ``instantiate._hex_to_rgba`` divides the theme's ``#rrggbb`` by 255 and
    stores it as the factor, which glTF defines as *linear*; the sheet renders
    with ``view_transform = "Standard"``, so the file gets the sRGB encoding of
    that linear value and never the hex itself. Spelled out here because a test
    that compared against the hex would fail on a correct render.
    """
    v = linear_byte / 255.0
    encoded = 12.92 * v if v <= 0.0031308 else 1.055 * v ** (1 / 2.4) - 0.055
    return round(255 * encoded)


def _expected_region_colours(materials: dict[str, str]) -> dict[str, tuple[int, int, int]]:
    return {
        region: tuple(
            _srgb_byte(int(hex_value.lstrip("#")[i : i + 2], 16)) for i in (0, 2, 4)
        )
        for region, hex_value in materials.items()
    }


# --- 1. the whole chain, through the real door -------------------------------


@pytest.mark.gpu
async def test_a_character_end_to_end_ends_at_a_rigged_mesh(svc):
    """One press, a finished mesh and a rig that runs; no sheet follows it.

    **Through the real door and the real worker.** The alternative -- driving
    ``instantiate`` and ``op_rig`` by hand -- would have skipped
    ``create_character``'s ordering (every refusal before a byte is written) and
    the exact joints it carries to the rig row. The character sheet that used
    to follow the rig left with Poser (``retired.py``): once the rig is done the
    queue is empty.

    ``svc`` pins ``REALMSPINNER_HOME`` and the data dir under ``tmp_path`` with
    ``monkeypatch.setenv``, which matters more in this lane than in any other:
    the gpu lane is exempt from the session-wide pin, so a test that built its
    own bare ``Config`` would mint a character row in the developer's real
    library.
    """
    pytest.importorskip("bpy")

    from realmspinner.queue import Worker
    from realmspinner.service.characters import create_character

    made = create_character(svc, recipe_mod.DEFAULT_RECIPE.as_dict(), name="P28 ogre")
    model = svc.store.get(made["id"])
    # The mesh row is minted finished -- ``import_mesh``'s arrangement, which is
    # what makes ``rerun_job`` refuse it.
    assert (model["stage"], model["status"]) == ("model", "done")
    assert model["params"]["built"] is True

    worker = Worker(svc.config, svc.store)
    try:
        rig_row = svc.store.get(made["rig"])
        assert rig_row["kind"] == "rig"
        await worker._process(rig_row)
        rigged = svc.store.get(made["rig"])
        assert rigged["status"] == "done", rigged.get("error")
        # Nothing is minted by the rig's completion any more.
        assert svc.store.next_queued() is None
    finally:
        await worker.shutdown()

    source_dir = svc.config.job_dir(made["id"])
    assert (source_dir / "rig.glb").is_file()
    assert (source_dir / "rig.json").is_file()


# --- 2. the import, which is where a UV-less triangle soup would surprise -----


@pytest.mark.gpu
@pytest.mark.parametrize("species", ARCHETYPE_SPECIES)
def test_a_generated_character_imports_into_blender_with_its_material_slots(
    species, tmp_path
):
    """A boolean union with per-face materials and **no UVs**, through the importer.

    Every mesh this path had ever carried was a TRELLIS reconstruction: one
    primitive, one material, a UV atlas. A generated character is the opposite
    on all three counts -- one primitive per region, one material each, and not
    a single UV layer, because ``instantiate`` writes flat colours rather than
    textures. That is the shape most likely to come back from a glTF importer as
    one merged grey object, so this asserts the three things that would go
    missing: the slots, the per-face indices into them, and the colours
    ``_make_flat`` then renders.

    The expected pixel is the *sRGB encoding* of the theme's hex, not the hex:
    see ``_srgb_byte``. A test written against the hex would fail on a correct
    render, which is exactly the kind of near-miss that gets an assertion
    weakened instead of understood.
    """
    bpy = pytest.importorskip("bpy")

    from realmspinner.pipelines import blender_worker

    spec = _recipe_for(species)
    instance = _quiet(instantiate_mod.instantiate, spec, tmp_path)

    _quiet(blender_worker._reset_scene, bpy)
    mesh = _quiet(blender_worker._import_glb, bpy, tmp_path / "model.glb")

    # The regions survive as distinct, correctly *named* slots -- the name is
    # what any exporter downstream addresses them by, so a slot that came back as
    # "Material.001" is a lost region even if the colour is right.
    #
    # A *subset* of the archetype's regions, and in its order. The tuple is the
    # body plan's whole vocabulary while the bake is one silhouette's use of it:
    # every quadruped theme paints ``mane``, and the paw, antlered and scaled
    # bakes have no mane geometry to paint. What must hold is that the slots are
    # region names, that none is duplicated, and that the order is the region
    # tuple's -- a baked mesh stores one region *index* per face, so a reordered
    # slot list repaints the body.
    slots = [material.name for material in mesh.data.materials]
    regions = list(spec.spec.regions)
    assert slots, f"{species}: the import produced no material slots at all"
    assert len(set(slots)) == len(slots), f"{species}: duplicated slots {slots}"
    assert set(slots) <= set(regions), f"{species}: slots came back as {slots}"
    assert slots == [name for name in regions if name in set(slots)], (
        f"{species}: the slots are out of region order: {slots} against {regions}"
    )
    # Per-face, not per-object: a join that dropped the material indices would
    # leave every polygon on slot 0 and paint the whole body one colour.
    assert {p.material_index for p in mesh.data.polygons} == set(range(len(slots)))
    # Stated rather than assumed. This is the property that makes the mesh
    # unusual, and if a future bake grows UVs the test that says "no UVs" should
    # be the thing that notices.
    assert list(mesh.data.uv_layers) == []

    # --- and _make_flat renders them as different colours ---------------------
    frames = tmp_path / "flat"
    _quiet(
        blender_worker.op_sheet,
        bpy,
        blender_spec.sheet_spec(
            tmp_path / "model.glb",
            frames,
            [{"index": 0, "yaw": 0.0, "pose": None, "bones": {}}],
            frame_size=256,
            elevation=spec.elevation,
            lighting="flat",
        ),
    )
    with Image.open(frames / "0000.png") as opened:
        opened.load()
        rendered = _opaque_colours(opened.convert("RGBA"))

    # Keyed on the *slots*, not on the theme's whole palette: a colour the mesh
    # has no geometry for must not appear in the render, and asserting against
    # the palette would let one through.
    expected = _expected_region_colours(
        {region: instance.materials[region] for region in slots}
    )
    # The tolerance is one byte of rounding either way: Blender's float-to-byte
    # conversion and ``_srgb_byte``'s ``round`` disagree in the last bit on some
    # channels, and an exact-equality test duly failed on a *correct* render
    # (ogre skin came back (204, 195, 168) against a predicted (204, 194, 167)).
    # It stays unambiguous because the shipped themes keep their regions much
    # further apart than that, which the next line refuses to assume.
    tolerance = 2
    values = list(expected.values())
    closest = min(
        (
            max(abs(a[i] - b[i]) for i in range(3))
            for n, a in enumerate(values)
            for b in values[n + 1 :]
        ),
        default=255,
    )
    assert closest > 2 * tolerance, (
        f"{species}: two regions are {closest} apart, too close to tell apart at +/-{tolerance}"
    )

    def region_of(rgb: tuple[int, int, int]) -> str | None:
        for region, want in expected.items():
            if max(abs(rgb[i] - want[i]) for i in range(3)) <= tolerance:
                return region
        return None

    # **The failure this is really about**: a body that comes through as one
    # flat grey, or shaded, or wearing Blender's default material. So the claim
    # is not "several colours appeared" but the strongest form available -- that
    # *every* pixel drawn is one of the theme's region colours. Measured
    # 2026-09-05: 100% of opaque pixels match, on all four body plans.
    total = sum(rendered.values())
    covered: Counter = Counter()
    for rgb, count in rendered.items():
        region = region_of(rgb)
        assert region is not None or count / total < 0.001, (
            f"{species}: {rgb} covers {count / total:.1%} of the render "
            f"and is no region colour of {sorted(expected)}"
        )
        if region is not None:
            covered[region] += count
    # And every slot reached the frame. A slot that survived the import but
    # renders as nothing is a region the user cannot recolour and will not see.
    assert set(covered) == set(slots), (
        f"{species}: {sorted(set(slots) - set(covered))} rendered no pixels at all"
    )


# --- 3. the one that matters: heat weights, not envelope ----------------------


@pytest.mark.gpu
@pytest.mark.parametrize("species", ARCHETYPE_SPECIES)
def test_a_character_rig_takes_bone_heat_weights_not_envelope(species, tmp_path):
    """Every generated body plan binds by bone heat, and none of them falls back.

    **The silent failure this exists for.** ``_skin`` tries a welded heat solve,
    then an unwelded one, then envelope -- and the envelope rung only prints a
    line on the subprocess's stdout. Envelope weights on a *fused* humanoid,
    where the arms are one welded surface with the chest, skin the arms to the
    torso: the character deforms like a sheet of dough and the rig reports
    success. A boolean union of capsules is manifold by construction, which is
    the whole argument for generating bodies rather than reconstructing them --
    but the subdivision and the weld in front of the solve can reopen a surface,
    and nothing else in the pipeline would notice.

    Asserted on what the worker *reports*, which is ``result["weighting"]`` and
    the ``weighting`` field ``_rig_meta`` writes into rig.json -- not on a proxy
    like "some vertex group has a weight", which envelope satisfies too.

    All four body plans, because the four templates differ in exactly the way
    that decides this: 19 bones over a humanoid, 19 over a quadruped, 20
    including a wing chain, 8 over a blob. About 0.5 s each.
    """
    pytest.importorskip("bpy")
    import bpy

    result = _built_and_rigged(bpy, _recipe_for(species), tmp_path)

    assert result["ok"] is True
    # ``automatic`` and ``automatic-welded`` are the two bone-heat outcomes --
    # ``_skin_steps`` names the second when the weld actually merged something,
    # which a glTF round trip guarantees it will. ``envelope`` is the fallback
    # and is a failure here.
    assert result["weighting"] in ("automatic", "automatic-welded"), (
        f"{species} fell back to {result['weighting']}: {result['weighting_reason']}"
    )
    # ``None`` on either automatic path: there is nothing to explain about a
    # solve that took, and a reason beside an automatic method would mean the
    # two halves of the report disagree.
    assert result["weighting_reason"] is None

    # And the same answer in the file, because rig.json is the only record that
    # outlives the subprocess -- the pose editor, the adjust-joints pass and the
    # UI's "this rig is degraded" notice all read it and not the result dict.
    rig = store.read_rig(tmp_path)
    assert rig["weighting"] == result["weighting"]
    assert rig["weighting_reason"] is None
    assert len(rig["bones"]) == result["bones"]


# --- 4. the union framing, and what it is worth ------------------------------
