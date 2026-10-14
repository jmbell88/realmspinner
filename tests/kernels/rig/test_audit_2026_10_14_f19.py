"""The 2026-10-14 audit's rig-kernel findings (poser-rig-05, poser-rig-06, poser-rig-07).

Dead symbols a removed mode left behind, two name patterns that let a trailing newline
through, and prose that still sends a reader to a mode, a module and a manual section that
are gone.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from realmspinner import clips, poselib
from realmspinner.kernels.rig import cliplib, skeleton, store, templates

RIG_KERNEL = Path(skeleton.__file__).parent
SRC = RIG_KERNEL.parents[1]
REPO = SRC.parents[1]


# -- poser-rig-06: `$` matches before a trailing newline ---------------------------------------


def test_a_bone_name_with_a_trailing_newline_is_refused() -> None:
    """``BONE_NAME_RE.match("hips\\n")`` succeeded, so a validated skeleton could carry a
    bone named ``'arm\\n'`` -- a different name once anything downstream strips it."""
    assert skeleton.BONE_NAME_RE.match("hips") is not None
    assert skeleton.BONE_NAME_RE.match("hips\n") is None
    with pytest.raises(store.RigError, match="not usable"):
        skeleton.check_skeleton_structure(
            [{"name": "root", "parent": None}, {"name": "arm\n", "parent": "root"}]
        )


def test_a_template_key_with_a_trailing_newline_is_refused() -> None:
    assert templates.TEMPLATE_KEY_RE.match("humanoid") is not None
    assert templates.TEMPLATE_KEY_RE.match("humanoid\n") is None


# -- poser-rig-05: symbols with no caller anywhere ----------------------------------------------


@pytest.mark.parametrize(
    ("module", "name"),
    [(cliplib, "shipped_clip_templates"), (clips, "shipped_clip_timing")],
)
def test_the_agent_catalogue_helpers_with_no_caller_are_gone(module, name) -> None:
    """Both served the agent catalogue that left with the character surface; neither has a
    caller in ``src/`` or ``tests/``."""
    assert not hasattr(module, name), f"{module.__name__}.{name} has no caller; delete it"


# -- poser-rig-07: prose that names things that no longer exist --------------------------------

#: (needle, why it is stale). Each was found in the rig kernel's own text on 2026-10-14.
_STALE = (
    ("Troupe reads", "Troupe is not a mode; the collision is Inker's filename tag"),
    ("TROUPE_DIRECTION_KEYS", "the constant carried a removed mode's name"),
    ("Rig manually", "Poser's door; the only door now is Clay's rig route"),
    ("poselib.preview_path", "poselib has no such function"),
    ("poselib.template_digest", "poselib has no such function"),
    ("chapter 08", "the A-pose trap moved to chapter 23; chapter 08 is Clay-only"),
    ("Poser", "Poser was removed with the character-sheet pipeline"),
    ("Troupe's", "Troupe is not a mode"),
)

_PROSE_FILES = (
    *sorted(RIG_KERNEL.glob("*.py")),
    SRC / "templates" / "blank.json",
    SRC / "templates" / "sprite_guides" / "walk8.json",
)


def test_rig_kernel_prose_names_no_removed_mode() -> None:
    found = [
        f"{path.relative_to(SRC)}: {needle!r} ({why})"
        for path in _PROSE_FILES
        for needle, why in _STALE
        if needle in path.read_text(encoding="utf-8")
    ]
    assert not found, found


def test_the_poselib_functions_the_prose_cites_exist() -> None:
    """Every ``poselib.<name>`` a rig-kernel file cites is a name poselib defines."""
    cited = {
        match
        for path in sorted(RIG_KERNEL.glob("*.py"))
        for match in re.findall(r"poselib\.(\w+)", path.read_text(encoding="utf-8"))
    }
    # A name wrapped across a line in a docstring is captured up to the break; skip those.
    missing = sorted(n for n in cited if not n.endswith("_") and not hasattr(poselib, n))
    assert not missing, missing


def test_the_manual_section_blender_spec_cites_exists() -> None:
    text = (RIG_KERNEL / "blender_spec.py").read_text(encoding="utf-8")
    match = re.search(r'manual chapter (\d+), "([^"]+)"', text)
    assert match, "the A-pose cross-reference is gone; update this test with it"
    [chapter] = list((REPO / "docs" / "manual").glob(f"{match.group(1)}-*.md"))
    assert f"### {match.group(2)}" in chapter.read_text(encoding="utf-8")


def test_the_direction_collision_refusal_names_the_importer_that_causes_it() -> None:
    with pytest.raises(ValueError, match="Inker") as raised:
        cliplib.reject_direction_named_clip("fall_back")
    assert "Troupe" not in str(raised.value)
    assert "'fall'" in str(raised.value) and "'back'" in str(raised.value)


def test_no_module_reads_a_cliplib_attribute_that_is_not_there() -> None:
    """Guard on the rename and the deletions: every ``cliplib.<name>`` read in ``src/``
    resolves."""
    missing: list[str] = []
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "cliplib"
                and not hasattr(cliplib, node.attr)
            ):
                missing.append(f"{path.relative_to(SRC)}: cliplib.{node.attr}")
    assert not missing, missing
