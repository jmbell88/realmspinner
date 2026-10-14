"""The 2026-10-14 audit, docs fixer D2: the Clay chapter and the chapters that send a mesh to it.

``tests/manual/`` gates a chapter's structure, not its claims, so each of these reads one
sentence of a chapter against the code it describes. Each fails against the text the audit found.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_DOCS = _ROOT / "docs" / "manual"


def _chapter(name: str) -> str:
    # Joined across the hard wraps, so a phrase split over two lines still matches.
    text = (_DOCS / name).read_text(encoding="utf-8")
    return re.sub(r"\s+", " ", text)


def _section(chapter: str, heading: str) -> str:
    """The text from ``## heading`` to the next ``## `` (the chapter is whitespace-joined)."""
    start = chapter.index(f"## {heading}")
    return re.split(r" ## ", chapter[start + 3 :], maxsplit=1)[0]


# -- clay-mesh-model-07, clay-ops-tail-08: the format a save writes ------------------------------


def test_the_manual_names_the_format_number_the_writer_writes() -> None:
    from realmspinner.kernels.mesh import serialize

    clay = _chapter("27-clay.md")
    assert f"saved in format {serialize.VERSION}" in clay
    assert f"a save writes format {serialize.VERSION}" in clay
    stale = f"format {serialize.VERSION - 1}"
    assert f"saved in {stale}" not in clay
    assert f"a save writes {stale}" not in clay


# -- docs-F1-04, docs-F2-09: nothing downstream of a mesh makes a sheet --------------------------

_SHEET_PROMISES = (
    r"posing, sheets",
    r"posing, sprite sheets",
    r"sprite sheets, every export",
    r"poses and sheets",
    r"poses and its sheets",
    r"poses and its rendered sheets",
    r"rendered sheets",
    r"a sheet, or",
    r"no sprite sheets",
    r"no\s+sprite sheets",
    r"posing and sheets",
    r"sprite sheets, the triangle",
)


def test_manual_no_longer_promises_character_or_sprite_sheets_from_a_mesh() -> None:
    names = (
        "27-clay.md",
        "07-modelling.md",
        "08-rigging-and-animating.md",
        "24-the-3d-viewport.md",
        "23-generating-meshes.md",
        "43-troubleshooting.md",
        "02-your-first-asset.md",
        "11-tuning-what-you-get.md",
    )
    found = [
        f"{name}: {pattern}"
        for name in names
        for pattern in _SHEET_PROMISES
        if re.search(pattern, _chapter(name))
    ]
    assert not found


def test_clay_manual_export_inheritance_list_has_no_sprite_sheets() -> None:
    section = _section(_chapter("27-clay.md"), "The two ways out")
    assert "inherits everything" in section
    assert "sprite sheets" not in section


# -- docs-F2-08: the button is "Open in Clay" ----------------------------------------------------


def test_manual_never_names_an_edit_in_clay_button() -> None:
    from realmspinner.studio import verbs

    label = verbs.open_in("clay")
    assert label == "Open in Clay"
    offenders = [
        p.name for p in sorted(_DOCS.glob("*.md")) if "Edit in Clay" in p.read_text("utf-8")
    ]
    assert not offenders, offenders


# -- docs-F2-03: Weight with Blender is a hand-off to Blender ------------------------------------


def test_clay_manual_does_not_claim_nothing_goes_to_blender() -> None:
    clay = _chapter("27-clay.md")
    assert "Clay no longer hands anything to Blender" not in clay
    ways = _section(clay, "The two ways out")
    assert "no `.blend` export" in ways
    assert "Weight with Blender" in ways


# -- docs-F2-07: the outliner row menu -----------------------------------------------------------


def test_clay_manual_outliner_row_menu_lists_the_rig_rows() -> None:
    from realmspinner.studio.modes.clay.ui.panes import outliner

    source = inspect.getsource(outliner._context_menu)
    # The menu draws the rig rows from the registry: the two names it can pick between.
    assert "mark-bone" in source and "unmark-bone" in source and "unbind-skin" in source
    outliner_text = _section(_chapter("27-clay.md"), "The outliner")
    row_menu = outliner_text[outliner_text.index("Right-clicking a row") :][:700]
    labels = ("Rename", "Duplicate", "Solo", "Delete", "Mark as Bone", "Unmark Bone", "Unbind Skin")
    for label in labels:
        assert f"**{label}**" in row_menu, label


# -- clay-panes-07: the Inspector's folds and the Pose segment -----------------------------------


def test_manual_inspector_folds_match_the_one_shape_state() -> None:
    from realmspinner.studio.modes.clay.ui.panes import inspector

    source = inspect.getsource(inspector._one)
    folds = re.findall(r'widgets\.header\(\s*"([^"]+)"', source)
    assert "Bone & rig" in folds
    section = _section(_chapter("27-clay.md"), "The inspector")
    assert "three closed folds" not in section
    for title in folds:
        if title == "Advanced":  # a fold inside Position & turn, named where it is used
            continue
        assert f"**{title}**" in section, title
    # The Rig fold opens by default for a bone or a weighted shape, and the manual says which.
    assert "default_open=rigged" in source
    assert "starts open for a bone or a weighted shape" in section


def test_manual_mode_switch_names_pose_and_its_narrow_letter() -> None:
    header = (
        _ROOT / "src" / "realmspinner" / "studio" / "modes" / "clay" / "ui" / "panes" / "header.py"
    ).read_text(encoding="utf-8")
    short = dict(re.findall(r'"(\w+)": "(\w)"', header.split("MODE_SHORT = ")[1].split("}")[0]))
    assert short["pose"] == "O"
    text = _chapter("27-clay.md")
    bullet = text[text.index("The **mode switch**") :][:700]
    assert "**Pose**" in bullet
    for letter in short.values():
        assert f"**{letter}**" in bullet, letter


# -- docs-F2-10: Apply Pose as Rest acts on the whole pose ---------------------------------------


def test_apply_pose_as_rest_is_documented_as_whole_pose() -> None:
    from realmspinner.kernels.mesh import document as bd

    # No selection parameter: the op rebases every driven bone.
    assert list(inspect.signature(bd.ClayDoc.apply_pose_as_rest).parameters) == ["self"]
    pose = _section(_chapter("27-clay.md"), "Pose mode")
    assert "work on the selected bones in Pose mode" not in pose
    assert "acts on the whole pose" in pose
    assert "**Apply Pose as Rest** takes no selection" in pose


# -- docs-F2-01: Export GLB writes every animation, in list order --------------------------------


def test_export_glb_writes_the_animations_in_document_order_and_the_manual_says_so() -> None:
    from realmspinner.kernels.mesh import document as bd
    from realmspinner.kernels.mesh import primitives as bp

    doc = bd.ClayDoc()
    hip = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Hip", mesh=bp.box(), bone=True)).uid
    doc.add_animation("Walk")
    doc.set_key(hip, "translation", 0.0, [0, 0, 0])
    doc.add_animation("Jump")
    doc.set_key(hip, "translation", 0.0, [0, 1, 0], animation=1)
    doc.set_active_animation(1)
    assert [a.name for a in bd.to_model(doc).animations] == ["Walk", "Jump"]

    clay = _chapter("27-clay.md")
    assert "Export GLB writes first" not in clay
    assert "Export GLB writes every animation, in the order of the list" in clay


# -- docs-F2-06: an animation that moves no matched bone is left out -----------------------------


def test_import_by_name_drops_an_animation_that_moves_no_matched_bone_and_the_manual_says_so() -> (
    None
):
    from realmspinner.kernels.geom3d import glbwrite, gltf
    from realmspinner.kernels.mesh import animimport
    from realmspinner.kernels.mesh import document as bd
    from realmspinner.kernels.mesh import primitives as bp

    def bone(doc: bd.ClayDoc, name: str) -> int:
        return doc.add_object(
            bd.Obj(uid=bd.new_uid(), name=name, mesh=bp.box(), bone=True)
        ).uid

    source = bd.ClayDoc()
    hip, tail = bone(source, "Hip"), bone(source, "Tail")
    source.add_animation("Walk")
    source.set_key(hip, "translation", 0.0, [0, 0, 0])
    source.set_key(hip, "translation", 1.0, [0, 1, 0])
    source.set_key(tail, "translation", 0.0, [0, 0, 0])
    source.add_animation("Wag")
    source.set_key(tail, "translation", 0.0, [0, 0, 0], animation=1)
    source.set_key(tail, "translation", 1.0, [1, 0, 0], animation=1)
    model = gltf.load(glbwrite.write_glb(bd.to_model(source)))

    target = bd.ClayDoc()
    bone(target, "Hip")
    imported = animimport.import_by_name(target, model)
    assert [a.name for a in imported.animations] == ["Walk"]  # "Wag" moves no bone of the target

    section = _section(_chapter("27-clay.md"), "Importing an animation")
    assert "Every animation in the file comes in as its own" not in section
    assert "an animation that moves none of them is left out" in section


# -- docs-F1-09: the tutorial says which object carries the bone's name --------------------------


def test_rigging_tutorial_says_which_object_carries_the_bone_name() -> None:
    chapter = _chapter("08-rigging-and-animating.md")
    assert "belong to the *bones*" in chapter
    # No shape is named after a bone: a box and a joint with one name is what made the
    # imported animation match the wrong object.
    shapes = re.search(r"Give each part a name in the outliner: (.+?)\. The boxes", chapter)
    assert shapes
    shape_names = set(re.findall(r"`([^`]+)`", shapes.group(1)))
    bone_names = {"UpperArm.L", "UpperArm.R", "Leg.L", "Leg.R"}
    for name in bone_names:
        assert f"`{name}`" in chapter
    assert not shape_names & bone_names


# -- clay-view-01 skeleton: the viewport draws the skeleton in the pose editor -------------------


def test_viewport_chapter_says_the_skeleton_is_drawn_while_editing_a_pose() -> None:
    from realmspinner.studio import _viewer_pose

    source = inspect.getsource(_viewer_pose)
    assert "if not self.pose_mode or not self.editor.bound:" in source
    chapter = _chapter("24-the-3d-viewport.md")
    assert "while you edit a pose" in chapter
    assert "once an asset is rigged" not in chapter


# -- F01, F33: a hand-built rig is guarded, and Rig by hand survives without Blender -------------


def test_generating_meshes_chapter_matches_the_clay_rig_guards() -> None:
    from realmspinner.studio.panes import pose_panel

    chapter = _chapter("23-generating-meshes.md")
    assert "re-rigs the mesh with the corrected skeleton" in chapter
    assert "**Apply joint positions** asks first" in chapter
    rebind_warning = pose_panel.CLAY_REBIND_WARNING
    assert "re-rigs the static mesh with Blender's automatic weights" in rebind_warning
    # The library menu draws the hand-rig row without asking for Blender; the chapter says so.
    library_path = (
        _ROOT / "src" / "realmspinner" / "studio" / "modes" / "library" / "ui" / "panes"
    ) / "library.py"
    library = library_path.read_text(encoding="utf-8")
    assert 'if ctx.rigging_available and controls.menu_item("Rig"' in library
    assert 'controls.menu_item("Rig by hand in Clay"' in library
    assert "**Rig by hand in Clay** stays" in chapter
