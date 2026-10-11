"""Manual findings from the 2026-10-11 Create and Clay audits, pinned against the chapters.

Every test here was written first and failed against the manual as it stood. Each reads a fact from
the code where one exists (the three shipped image models, the Simplify ceilings, the op registry)
and the sentence from the chapter, so the next change to either fails here rather than in a reader's
hands.
"""

from __future__ import annotations

import re
from pathlib import Path

from realmspinner import models
from realmspinner.kernels.mesh import ops_simplify
from realmspinner.studio.modes.clay import ops

ROOT = Path(__file__).resolve().parents[2]
MANUAL = ROOT / "docs" / "manual"


def _chapter(name: str) -> str:
    return (MANUAL / name).read_text(encoding="utf-8").replace("\r\n", "\n")


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _paragraphs(text: str) -> list[str]:
    return [_flat(p) for p in re.split(r"\n\s*\n", text) if p.strip()]


# --- create-08: the retired distilled models --------------------------------

_RETIRED_PHRASES = re.compile(r"four-step|distilled entries|guidance[ -]zero", re.IGNORECASE)


def test_manual_does_not_describe_the_retired_distilled_model_entries() -> None:
    assert sorted(models.BASE_MODELS) == ["flux_klein", "sdxl_cfg", "sdxl_cfg_pag"]
    offenders = []
    for path in sorted(MANUAL.glob("*.md")):
        for paragraph in _paragraphs(path.read_text(encoding="utf-8")):
            match = _RETIRED_PHRASES.search(paragraph)
            if not match:
                continue
            # The one paragraph allowed to name them is the one that says they were taken away.
            if re.search(r"\b(retired|removed)\b", paragraph, re.IGNORECASE):
                continue
            offenders.append(f"{path.name}: {match.group(0)!r} in {paragraph[:70]!r}")
    listing = "\n".join(offenders)
    assert not offenders, f"the manual still describes models Create no longer offers:\n{listing}"


def test_chapter_12_names_the_one_refusal_the_three_models_have() -> None:
    text = _flat(_chapter("12-tuning-what-you-get.md"))
    assert "FLUX.2" in text and "ControlNet" in text, (
        "chapter 12's Conditioning passage does not say FLUX.2 klein cannot take a "
        "structure control"
    )
    assert "Avoid" in text and "all three" in text.lower(), (
        "chapter 12 does not say the negative prompt is honoured on every shipped model"
    )


# --- create-14: the right column is History, not "recent creations" ---------


def test_manual_screen_tour_names_the_history_pane_not_recent_creations() -> None:
    offenders = [
        path.name
        for path in sorted(MANUAL.glob("*.md"))
        if "recent creations" in _flat(path.read_text(encoding="utf-8")).lower()
    ]
    assert not offenders, f"{offenders} still say the right column holds recent creations"
    for name in ("02-your-first-asset.md", "22-generating-references.md"):
        assert "History" in _flat(_chapter(name)), f"{name} never names the History pane"
    assert "inspector and History" in _flat(_chapter("02-your-first-asset.md")), (
        "chapter 02's screen tour does not name History beside the inspector"
    )


def test_chapter_23_says_make_3d_works_on_a_reference_from_a_removed_model() -> None:
    text = _flat(_chapter("23-generating-meshes.md"))
    assert "removed" in text and "Make 3D" in text
    assert re.search(r"drawn (with|on) an image model (that )?(this version|has since)", text), (
        "chapter 23 does not say a reference drawn on a since-removed model can still be made 3D"
    )


# --- Clay chapters ----------------------------------------------------------


def _section(text: str, heading: str) -> str:
    """The body of *heading* (any level), up to the next heading of the same or a higher level."""
    lines = text.split("\n")
    start = next(i for i, line in enumerate(lines) if line.strip() == heading)
    level = len(heading) - len(heading.lstrip("#"))
    end = len(lines)
    for i in range(start + 1, len(lines)):
        stripped = lines[i]
        if stripped.startswith("#") and len(stripped) - len(stripped.lstrip("#")) <= level:
            end = i
            break
    return _flat("\n".join(lines[start:end]))


def test_chapter_07_does_not_deny_the_simplify_operation_exists() -> None:
    text = _flat(_chapter("07-modelling.md"))
    assert "does not reduce triangle counts" not in text, (
        "chapter 07 says Clay cannot reduce a face count; Simplify does"
    )
    assert "Simplify" in text, "chapter 07 never names Simplify"
    assert "rebuild a mesh as quads" in text, (
        "chapter 07 should still say Clay does not rebuild a mesh as quads"
    )


def test_the_simplify_section_states_the_ceiling_the_target_and_the_triangle_result() -> None:
    section = _section(_chapter("30-clay.md"), "## Simplifying a mesh")
    assert f"{ops_simplify.MAX_SIMPLIFY_CORNERS:,} corners" in section, (
        "the Simplify section does not state the per-press corner ceiling"
    )
    assert f"{ops_simplify.MAX_SIMPLIFY_CORNERS // 2:,} with" in section, (
        "the Simplify section does not state the halved ceiling for a mesh with UVs"
    )
    assert "above the target" in section, (
        "the Simplify section does not say only objects above the target are charged"
    )
    assert "across 3 objects, each toward" in section, (
        "the Simplify section does not quote the multi-object result sentence"
    )
    assert "per closed shell" in section and "several shells" in section, (
        "the Simplify section does not say the four-face floor is per shell"
    )
    assert "counts triangles" in section and "not quads" in section, (
        "the Simplify section does not say the collapse stage's target counts triangles"
    )
    assert "refused rather than raised" in section
    assert "the dialog" in section, (
        "the section does not say the dialog refuses below the floor too"
    )


def test_the_manual_quotes_the_headers_mode_words_not_object_mode() -> None:
    for name in ("30-clay.md", "46-extending.md"):
        text = _flat(_chapter(name))
        for stale in ('"Switch to object mode first', '"Switch to face mode first'):
            assert stale not in text, f"{name} quotes {stale!r}, words the interface never uses"
    assert "Switch to Shape mode first." in _flat(_chapter("46-extending.md"))
    assert ops.mode_word("object") == "Shape"


def test_the_recalculate_normals_paragraph_says_a_flat_sheet_is_left_alone() -> None:
    section = _section(_chapter("30-clay.md"), "## Recalculating normals")
    assert "has no outside to point to" in section
    assert "leaves it exactly as it is" in section
    assert "bowl or a half-pipe" in section


def test_the_parenting_paragraphs_say_a_sheared_shape_with_children_is_refused() -> None:
    section = _section(_chapter("30-clay.md"), "## Parents and groups")
    assert "is refused when the move would shear it" in section
    assert "changes nothing" in section
    assert "own children are not carried" not in section, (
        "the delete/ungroup paragraph still says a sheared child's own children are not carried; "
        "the move is refused instead"
    )


def test_merge_faces_paragraph_says_which_material_a_mixed_block_keeps() -> None:
    text = _flat(_chapter("30-clay.md"))
    assert (
        "the merged face takes the material and smooth/flat setting of the lowest-numbered of them"
        in text
    )


def test_the_uv_view_paragraph_names_the_slot_the_colours_page_edits() -> None:
    section = _section(_chapter("30-clay.md"), "### The UV view")
    assert "the object's default slot otherwise" not in section, (
        "the UV view backdrop is the slot the Colours & textures page edits, not the default slot"
    )
    assert "the slot the Colours & textures page edits" in section
    assert "first selected face" in section, "the first-selected-face exception was lost"


def test_the_manual_names_every_parameter_of_the_pack_islands_dialog() -> None:
    section = _section(_chapter("30-clay.md"), "### The UV view")
    pack = ops.get("pack-uv")
    assert [p.label for p in pack.params] == ["margin", "rotate islands"]
    for param in pack.params:
        assert param.label in section, f"the UV view section never names Pack's {param.label!r}"
    assert "no longer lines up" in section, (
        "the manual does not warn that repacking leaves an already painted picture misaligned"
    )


def test_the_manual_defines_uv_seam_once_and_uses_it_consistently() -> None:
    text = _flat(_chapter("30-clay.md"))
    assert "there are no seams" not in text, (
        "a box unwrap makes a seam on every edge between faces of different axes"
    )
    definition = "any edge whose two faces disagree about their texture coordinates"
    assert text.count(definition) == 1, "the chapter must define a UV seam exactly once"
    assert "Clay has no seam marking" in text
    assert "a seam" in _section(_chapter("30-clay.md"), "## Simplifying a mesh")


def test_the_manual_says_a_live_gesture_refuses_the_buttons_that_would_land_in_it() -> None:
    text = _flat(_chapter("30-clay.md"))
    assert "Finish the drag in progress first" in text
    assert "Inker texture pull waits for a live Clay gesture" in text
    assert "the first of" in text and "selected shapes" in text


def test_chapter_46_describes_clay_op_replies_and_the_agent_tab() -> None:
    text = _flat(_chapter("46-extending.md"))
    assert "`clay_op` answers with the `params` it actually ran with" in text
    assert "named in `clamped`" in text or "names the clamped ones in `clamped`" in text
    assert "whole-number rounding is not a clamp" in text
    assert "refused rather than clamped" in text
    assert "`errors` list" in text and "`faces_before`" in text
    assert "`clay_select` first" in text
    assert "60 s and 20 s" in text
    assert "opens behind" in text and "never takes the front" in text
    assert "`alpha`" in text and "`compare_mode`" in text and "`overlay`" in text
    assert "`clay_uv pack`" in text and "names a field" in text
