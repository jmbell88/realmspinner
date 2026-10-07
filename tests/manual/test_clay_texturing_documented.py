"""Chapter 30's Texturing section, held against the code it describes.

Phase 3 of Clay (paint faces, textures, the Inker round trip, crisp GLB and OBJ
textures) changed what four existing sentences of the chapter said, and nothing
tied the new section to the numbers it quotes: the sizes "Add texture" offers,
the size of the PICO-8 table, the file name an OBJ texture gets and the count of
operations. Each is stated in prose and each is read from its source here, so a
fourth texture size or a seventeenth colour fails this file instead of leaving
the manual one change behind.
"""

from __future__ import annotations

import pytest

from realmspinner.kernels.manual import loader
from realmspinner.kernels.mesh import document, objexport
from realmspinner.kernels.pixel import palettes
from realmspinner.studio.modes.clay import ops

_WORDS = {16: "sixteen", 38: "thirty-eight"}


def _flat(text: str) -> str:
    return " ".join(text.split())


def _section(heading: str) -> str:
    """The chapter's section under *heading*: down to the next ``## `` heading."""
    text = loader.load("30-clay").replace("\r\n", "\n")
    start = text.index(heading)
    end = text.find("\n## ", start + 1)
    return _flat(text[start : end if end != -1 else len(text)])


def test_chapter_30_has_a_texturing_section_naming_pico8_and_map_kd() -> None:
    section = _section("\n## Texturing\n")
    assert "PICO-8" in section, "the Texturing section does not name the PICO-8 palette"
    assert "map_Kd" in section, "the Texturing section does not name the OBJ texture line"
    assert "Edit texture in Inker" in section
    assert "Take texture back from Inker" in section


def test_the_texturing_section_states_exactly_the_texture_sizes_add_texture_offers() -> None:
    sizes = document.TEXTURE_SIZES
    spelled = f"{', '.join(str(s) for s in sizes[:-1])} or {sizes[-1]} pixels a side"
    assert spelled in _section("\n## Texturing\n"), (
        f"the Texturing section does not say {spelled!r}, which is document.TEXTURE_SIZES"
    )


def test_the_texturing_section_counts_the_pico8_colours_the_code_ships() -> None:
    count = len(palettes.PICO8)
    assert count in _WORDS, f"teach this test the word for {count}"
    assert f"{_WORDS[count]} PICO-8 colours" in _section("\n## Texturing\n")


def test_the_manual_names_a_texture_png_the_way_the_obj_exporter_does() -> None:
    example = objexport.texture_name("barrel", 2)
    assert example == "barrel_2.png"
    assert f"`{example}`" in _flat(loader.load("30-clay")), (
        "chapter 30 shows an example texture file name that objexport.texture_name no "
        "longer produces"
    )


@pytest.mark.parametrize("label", ["Assign Material..."])
def test_the_operation_added_for_painting_faces_is_in_the_chapter_and_the_count(
    label: str,
) -> None:
    assert any(op.label == label for op in ops.OPS), f"no operation is labelled {label!r}"
    text = _flat(loader.load("30-clay"))
    assert label in text
    count = len(ops.OPS)
    assert count in _WORDS, f"teach this test the word for {count}"
    assert f"Clay has {_WORDS[count]} operations" in text
