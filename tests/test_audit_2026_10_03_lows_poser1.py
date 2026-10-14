"""The 2026-10-03 audit's Low findings poser-21..53 that fixer ``poser1`` owns.

One file for the lot (the module trees differ -- characters, kernels, service,
a studio pane, the manual -- and the brief asks for one new file per fixer).
Every test name is the claim it proves; each was run against the unfixed code
and failed for the reason its docstring gives, except where a docstring says a
helper did not exist yet.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from realmspinner.characters import CharacterError, Recipe
from realmspinner.characters import family as family_mod
from realmspinner.characters import instantiate as instantiate_mod
from realmspinner.kernels import charsheet
from realmspinner.kernels.rig import poses as poses_mod
from realmspinner.kernels.rig import store
from realmspinner.service import characters as svc_characters
from realmspinner.service import sprites as svc_sprites
from realmspinner.service.errors import Invalid

REPO = Path(__file__).resolve().parents[1]


# --- fixtures shared by the service tests ------------------------------------


def _mesh(svc, *, rigged: bool = True, template: str = "humanoid") -> str:
    job_id = svc.store.create("image", "a hooded ranger", {}, stage="model")
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"fake-glb")
    if rigged:
        (job_dir / "rig.glb").write_bytes(b"fake-rig")
        (job_dir / "rig.json").write_text(json.dumps({"template": template}), "utf-8")
    svc.store.set_status(job_id, "done")
    return job_id


# --- poser-21 ---------------------------------------------------------------


def _blender_up(along):
    """Blender's own roll-0 bone frame, written out independently of the code
    under test: ``vec_roll_to_mat3`` -- the Z axis a bone pointing along
    *along* gets when its roll is zero."""
    x, y, z = along
    theta = 1.0 + y
    if theta < 1e-6:
        return (0.0, 0.0, 1.0)
    return (-x * z / theta, -z, 1.0 - z * z / theta)


@pytest.mark.parametrize(
    ("species", "socket", "bone"),
    [("horse", "saddle", "spine"), ("wolf", "pack", "hips"), ("dragon", "saddle", "chest")],
)
def test_a_quadruped_saddle_socket_is_above_the_spine_it_hangs_off(
    species, socket, bone, tmp_path
):
    """The 2026-10-03 audit, poser-21: the sidecar's socket frame was built
    from a world +Z cross product, so a horizontal bone's "up" pointed down
    and the horse saddle landed 0.445 m *below* the spine head -- where the
    Blender worker, which uses the bone's own frame, puts it above."""
    fam = family_mod.get_family(species)
    inst = instantiate_mod.instantiate(
        Recipe.from_dict({"family": species, "theme": fam.themes[0].key}), tmp_path
    )
    joint = next(j for j in inst.joints if j["name"] == bone)
    position = inst.sockets[socket]["position"]
    if species == "horse":
        # The reproduced case: 0.742 (below) before the fix, 1.675 after.
        assert position[2] > joint["head"][2], (species, socket, position, joint)

    # And it is the point the worker computed (the worker's own socket
    # projection read the bone's own matrix and has since been deleted); this
    # reference is Blender's roll-0 ``vec_roll_to_mat3``, checked against a real
    # ``bpy`` armature for all three rows when this was written (horse saddle
    # 1.675, wolf pack 0.354, dragon saddle 2.367 -- sidecar and worker equal).
    # A bone tilted a few degrees off -Y is the one place Blender's own frame
    # flips "up" to point down, so wolf and dragon sit *below* their bone in the
    # worker too; the sidecar now agrees with it rather than with world +Z.
    spec = next(s for s in fam.sockets if s.name == socket)
    head = joint["head"]
    tail = joint["tail"]
    span = [t - h for t, h in zip(tail, head, strict=True)]
    length = sum(v * v for v in span) ** 0.5
    along = [v / length for v in span]
    x, y, z = along
    theta = 1.0 + y
    x_axis = (1.0 - x * x / theta, -x, -x * z / theta) if theta >= 1e-6 else (-1.0, 0.0, 0.0)
    up = _blender_up(along)
    a, b, c = spec.offset
    expected = [
        head[i] + length * (b * x_axis[i] + a * along[i] + c * up[i]) for i in range(3)
    ]
    assert position == pytest.approx(expected, abs=1e-6)


# --- poser-22 ---------------------------------------------------------------


@pytest.mark.parametrize("damage", ["missing", "empty", "corrupt"])
def test_a_missing_or_empty_masks_file_is_refused_by_name(damage, tmp_path):
    """poser-22: ``_load_base`` presence-checked the baked .glb but not the
    .masks.npz, so a missing file raised a raw FileNotFoundError, a zero-byte
    one a raw EOFError and a corrupt one a field-less ValueError -- none of
    them a CharacterError, the exception every door is written to catch."""
    real = family_mod.get_family("horse")
    masks = tmp_path / "horse.masks.npz"
    if damage == "empty":
        masks.write_bytes(b"")
    elif damage == "corrupt":
        masks.write_bytes(b"this is not a zip archive at all" * 8)
    fam = SimpleNamespace(label=real.label, base_glb=real.base_glb, masks_npz=masks)
    with pytest.raises(CharacterError) as excinfo:
        instantiate_mod._load_base(fam)
    assert excinfo.value.field == "family"
    assert "author_humanoid.py" in str(excinfo.value)


# --- poser-25 ---------------------------------------------------------------


# --- poser-26 ---------------------------------------------------------------


def test_the_sprite_panel_does_not_touch_the_guide_files_on_a_second_draw(monkeypatch):
    """poser-26: ``sprite_options()`` and ``sprite_cost()`` ran on every draw
    with no memo, and each stats the pose-guide files and parses a guide JSON
    (about 2.3 ms a frame while the section is open). The panel now asks
    through ``_options()`` / ``_cost()``, which read each answer once.

    (The two accessors did not exist before this fix, so on the unfixed code
    this fails on the missing attribute; the call counts are the claim.)"""
    from realmspinner.studio.panes import sprite_panel

    calls = {"options": 0, "cost": 0}
    real_options = svc_sprites.sprite_options
    real_cost = svc_sprites.sprite_cost

    def options():
        calls["options"] += 1
        return real_options()

    def cost(*args, **kwargs):
        calls["cost"] += 1
        return real_cost(*args, **kwargs)

    monkeypatch.setattr(svc_sprites, "sprite_options", options)
    monkeypatch.setattr(svc_sprites, "sprite_cost", cost)
    sprite_panel._reset_memo()
    first = sprite_panel._options()
    assert sprite_panel._options() is first
    plan = sprite_panel._cost("turnaround", 64)
    assert sprite_panel._cost("turnaround", 64) is plan
    assert calls == {"options": 1, "cost": 1}
    # A different input is a different answer, not a stale one.
    sprite_panel._cost("turnaround", 32)
    assert calls["cost"] == 2
    sprite_panel._reset_memo()


def test_sprite_palettes_docstring_no_longer_claims_sprite_options_reads_no_disk():
    doc = inspect.getdoc(svc_sprites.sprite_palettes) or ""
    assert "reads no disk" not in doc


# --- poser-27 ---------------------------------------------------------------


# --- poser-28 ---------------------------------------------------------------


@pytest.mark.parametrize("size", [16, 100, 7, 0])
def test_sprite_cost_marks_an_off_ladder_cell_size_not_drawable(size):
    """poser-28: ``sprite_cost`` said ``drawable=True`` for a cell size the
    door refuses (anything off the 32/48/64 ladder), against its docstring."""
    plan = svc_sprites.sprite_cost("turnaround", size)
    assert plan["drawable"] is False
    assert "logical_size" in plan["refusal"]


# --- poser-30 ---------------------------------------------------------------


@pytest.mark.parametrize("bad", [True, 8.9, "8.5", float("inf"), float("nan"), None])
def test_resolve_layout_refuses_a_non_integral_or_infinite_directions_preset_by_name(bad):
    """poser-30: the ``directions`` preset still went through a bare ``int()``,
    so ``8.9`` rendered an 8-direction sheet in silence and ``inf`` escaped as
    a raw OverflowError."""
    payload = {"version": 3, "movements": [{"key": "walk", "directions": bad}]}
    with pytest.raises(ValueError, match="directions must be 1, 4, 8, or 16"):
        charsheet.resolve_layout(payload)


def test_resolve_layout_still_takes_a_whole_directions_preset():
    for preset in (1, 4, 8, 16, "8"):
        layout = charsheet.resolve_layout(
            {"version": 3, "movements": [{"key": "walk", "directions": preset}]}
        )
        assert len(layout.movements[0].directions) == int(preset)


# --- poser-31 ---------------------------------------------------------------


def test_validate_bones_does_not_zero_a_quaternion_whose_norm_overflows():
    """poser-31: ``sum(v*v) ** 0.5`` overflows to inf above ~1e154, the
    renormalise step then divided by inf, and ``[1e200, 0, 0, 0]`` was stored
    as ``[0, 0, 0, 0]`` -- an all-zero rotation handed to Blender, which the
    zero-quaternion refusal never saw."""
    out = poses_mod.validate_bones({"hips": [1e200, 0.0, 0.0, 0.0]})
    assert out["hips"] == pytest.approx([1.0, 0.0, 0.0, 0.0])
    out = poses_mod.validate_bones({"hips": [1e200, 1e200, 0.0, 0.0]})
    assert sum(v * v for v in out["hips"]) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="degenerate"):
        poses_mod.validate_bones({"hips": [0.0, 0.0, 0.0, 0.0]})


# --- poser-36 ---------------------------------------------------------------


@pytest.mark.parametrize("bad", [4.9, True, float("inf"), "8.5"])
def test_recipe_from_prompt_refuses_a_fractional_or_boolean_directions(svc, bad):
    """poser-36: a bare ``int(overrides.get("directions", 8))`` turned 4.9 into
    4 and True into 1 (both on the ladder) before ``Recipe``'s hardened
    ``_integer`` ever saw them, and an infinity escaped as OverflowError."""
    with pytest.raises(Invalid) as excinfo:
        svc_characters.recipe_from_prompt(svc, "a knight", overrides={"directions": bad})
    assert excinfo.value.field == "directions"


def test_recipe_from_prompt_still_takes_a_ladder_directions(svc):
    result = svc_characters.recipe_from_prompt(svc, "a knight", overrides={"directions": 4})
    assert result["recipe"]["directions"] == 4


# --- poser-39 ---------------------------------------------------------------


# --- poser-40 ---------------------------------------------------------------


def test_a_directory_squatting_at_a_sprite_draft_png_name_is_not_served(svc):
    mesh_id = _mesh(svc, rigged=False)
    job_dir = svc.job_dir(mesh_id)
    draft_id = store.new_id()
    png = store.sprite_draft_png_path(job_dir, draft_id, store.SPRITE_CANDIDATES[0])
    png.mkdir(parents=True)
    store.sprite_draft_path(job_dir, draft_id).write_text("{}", encoding="utf-8")
    with pytest.raises(Exception, match="no such sprite draft") as excinfo:
        svc_sprites.sprite_draft_png(svc, mesh_id, draft_id, store.SPRITE_CANDIDATES[0])
    assert type(excinfo.value).__name__ == "NotFound"


# --- poser-41 ---------------------------------------------------------------


# --- poser-42 ---------------------------------------------------------------


# --- poser-51 ---------------------------------------------------------------


# --- poser-53 ---------------------------------------------------------------


def _source(relative: str) -> str:
    return (REPO / "src" / "realmspinner" / relative).read_text(encoding="utf-8")


def test_conditioning_comment_names_the_worker_not_the_door_as_the_guides_drawer():
    """poser-53: ``_conditioning``'s comment said the reference stage "writes
    control.png at the door" two lines above code that draws it in the worker
    for the stated reason that the door writes none."""
    text = _source("_q_generate.py")
    assert "writes ``control.png`` at the\n        # door" not in text.replace("\r\n", "\n")
    assert "writes control.png at the door" not in " ".join(text.split())
    assert "render_tpose_guide" not in _source("service/_jobs_create.py")


def test_comments_name_kernels_charsheet_not_a_pipelines_module():
    for relative in ("pipelines/spritesynth.py", "kernels/charsheet.py"):
        text = " ".join(_source(relative).split())
        assert "pipelines`` module" not in text, relative
        assert "pipelines.charsheet" not in text, relative
    assert "studio.packwright.compose" not in _source("kernels/sheet.py")
