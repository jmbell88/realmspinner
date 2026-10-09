"""Chapter 30 held against four behaviours the second 2026-10-09 clay audit found it misdescribing.

Each of these is a sentence that the code contradicted or left out, and none had a test that
could notice: the dialog sentence omitted an operation that has a dialog (clay-111), the texture
sentence promised a rebuild rewrites a shape's layout when it keeps the stored one (clay-79), the
OBJ paragraph never said what an OBJ cannot carry (clay-114), and the Weld rows did not say the
big-mesh pass can leave a close pair apart (clay-83). Each test reads the fact from the code where
it can and the sentence from the chapter, so the next change to either fails here.
"""

from __future__ import annotations

from realmspinner.kernels.manual import loader
from realmspinner.kernels.mesh import ops_topo
from realmspinner.studio.modes.clay import ops

#: What each parameterised operation is called in the sentence that lists the dialogs. The
#: dialog sentence says "assign material", not "Assign Material...", so the label is lowered and
#: its ellipsis dropped, then looked up here only where the sentence says it differently.
_SENTENCE_WORD = {
    "inset faces": "inset",
    "bevel edges": "bevel",
}


def _flat(text: str) -> str:
    return " ".join(text.split())


def _chapter() -> str:
    return _flat(loader.load("30-clay").replace("\r\n", "\n"))


def test_the_manual_lists_every_parameterised_clay_op_in_its_dialog_sentence() -> None:
    text = _chapter()
    start = text.index("Operations that take a number")
    sentence = text[start : text.index("dialog", start)].lower()
    missing = []
    for op in ops.OPS:
        if not op.params:
            continue
        label = op.label.rstrip(".").lower()
        word = _SENTENCE_WORD.get(label, label)
        if word not in sentence:
            missing.append(op.label)
    assert not missing, (
        f"the sentence that says which operations open a dialog of numbers leaves out {missing}"
    )


def test_the_manual_does_not_promise_a_size_edit_rewrites_the_texture_layout() -> None:
    text = _chapter()
    assert "or rebuilding a generated shape, writes the current one" not in text, (
        "a generated shape keeps its stored texture layout when its settings change without "
        "changing the face count, so a size edit does not rewrite it"
    )
    assert "press Box Unwrap again" in text, (
        "the chapter does not tell a reader who resized a textured box to run Box Unwrap again"
    )


def test_the_manual_says_obj_export_drops_cutout_and_double_sided() -> None:
    text = _chapter()
    start = text.index("**Export GLB** and **Export OBJ** write the document")
    paragraph = text[start : text.index("**Save screenshot**", start)].lower()
    assert "cutout" in paragraph and "double-sided" in paragraph, (
        "the OBJ paragraph does not say an OBJ cannot carry a cutout or a double-sided material"
    )
    assert "export a glb" in paragraph


def test_the_manual_states_the_weld_cell_grouping_caveat() -> None:
    limit = f"{ops_topo.WELD_SEARCH_LIMIT:,}"
    text = _chapter()
    assert text.count(f"more than {limit} vertices") >= 2, (
        f"the Weld row and the Merge Objects paragraph must each say that a weld of more than "
        f"{limit} vertices takes the rougher cell-grouping pass, which never welds a pair "
        "farther apart than the distance but can leave a closer pair apart"
    )
    assert "can leave a closer pair apart" in text
    assert "can leave some closer pairs apart" in text
