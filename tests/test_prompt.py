from __future__ import annotations

from pathlib import Path

import pytest

from warlock.pipelines import prompt

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "sdxl-turbo"

# Deliberately *not* a module-level pytestmark. Only chunk()/count() need a
# real CLIP tokenizer; build(), pad_pair() and everything about the templates
# is pure string assembly. A module-level skip made a checkout without the
# sdxl-turbo weights -- a worktree, say -- silently stop asserting the
# template properties. The transformers import lives in the fixture for the
# same reason: prompt.py imports it inside load_tokenizers(), so the module is
# importable without it.
needs_tokenizer = pytest.mark.skipif(
    not (MODEL_DIR / "tokenizer" / "vocab.json").exists(),
    reason="local sdxl-turbo tokenizer not downloaded",
)


@pytest.fixture(scope="module")
def tokenizers():
    pytest.importorskip("transformers")
    return prompt.load_tokenizers(MODEL_DIR)


@needs_tokenizer
def test_short_prompt_yields_exactly_one_chunk(tokenizers):
    chunks = prompt.chunk("a wooden crate, worn condition", tokenizers)
    assert chunks == ["a wooden crate, worn condition"]


@needs_tokenizer
def test_a_long_prompt_yields_more_than_one_chunk(tokenizers):
    fragment = "an ornate medieval fantasy weapon with intricate engravings"
    long_text = ", ".join([fragment] * 8)
    chunks = prompt.chunk(long_text, tokenizers)
    assert len(chunks) > 1
    for c in chunks:
        assert prompt.count(c, tokenizers) <= 77


@needs_tokenizer
def test_chunking_never_splits_a_phrase(tokenizers):
    text = "alpha one, beta two, gamma three"
    chunks = prompt.chunk(text, tokenizers, limit=5)
    for phrase in ("alpha one", "beta two", "gamma three"):
        assert any(phrase in c for c in chunks)


@needs_tokenizer
def test_a_lone_overlong_phrase_falls_back_to_whitespace_split(tokenizers):
    single_phrase = " ".join(["overlong"] * 40)
    chunks = prompt.chunk(single_phrase, tokenizers, limit=10)
    assert len(chunks) > 1
    for c in chunks:
        assert prompt.count(c, tokenizers) <= 12


@needs_tokenizer
def test_a_single_unsplittable_atom_is_hard_split_not_truncated(tokenizers):
    # One whitespace-free "word" (a pasted URL, say) that alone exceeds the
    # limit: chunk() must slice it rather than emit an over-limit chunk the
    # encoder would silently truncate.
    atom = "x/" * 40
    chunks = prompt.chunk(atom, tokenizers, limit=10)
    assert len(chunks) > 1
    for c in chunks:
        assert prompt.count(c, tokenizers) <= 12
    # No characters were lost in the split.
    assert "".join(chunks) == atom


@needs_tokenizer
def test_empty_prompt_yields_one_empty_chunk(tokenizers):
    assert prompt.chunk("", tokenizers) == [""]


def test_pad_pair_equalises_chunk_counts():
    a, b = prompt.pad_pair(["x", "y", "z"], ["only"])
    assert len(a) == len(b) == 3
    assert b == ["only", "", ""]

    a2, b2 = prompt.pad_pair(["x"], ["x"])
    assert a2 == ["x"]
    assert b2 == ["x"]


def test_build_reproduces_the_trigger_and_template_order():
    # Against the template constant, not a copy of its opening words: this test
    # is about the *order* the pieces are assembled in, and a literal made it
    # fail for a wording change that left that order exactly as it was.
    text = prompt.build("a barrel", {}, trigger="3d style, 3d render")
    body = prompt.PROMPT_TEMPLATE.format(prompt="a barrel")
    assert text == f"3d style, 3d render, {body}"


def test_build_with_no_trigger_has_no_leading_comma():
    text = prompt.build("a barrel", {})
    assert text == prompt.PROMPT_TEMPLATE.format(prompt="a barrel")
    assert not text.startswith(",")


def test_build_selects_the_sheet_template_when_asked():
    # pipelines-02 (the 2026-09-13 audit): build() claimed parity with
    # text2image.generate()'s own template assembly but had no ``sheet``
    # argument, so it could never select SHEET_TEMPLATE -- the prompt preview
    # could not show what a sheet job would actually send. tilesheet still
    # wins over sheet, mirroring generate()'s precedence.
    text = prompt.build("a knight", {}, sheet=True)
    assert text == prompt.SHEET_TEMPLATE.format(prompt="a knight")

    text = prompt.build("a knight", {}, sheet=True, tilesheet=True)
    assert text == prompt.TILESHEET_TEMPLATE.format(prompt="a knight")


def test_a_tile_prompt_does_not_ask_for_a_single_centred_object():
    out = prompt.build("mossy cobblestone", {}, tile=True)
    assert "single subject" not in out
    assert "seamless" in out and "tileable" in out


def test_a_tile_prompt_keeps_the_users_words_first():
    out = prompt.build("mossy cobblestone", {}, tile=True)
    assert out.startswith("mossy cobblestone")


def test_the_object_template_does_not_ask_for_concept_art():
    """The template rewrite. Every one of the 17 refusals in the 2026-08-07 sweep
    was the multi-object rule, and the family was concept-art layouts: character
    sheets, turnarounds, multi-view plates. "game asset concept art" is a
    request for exactly that -- a sheet is the canonical form of the genre --
    and it sat in the template that wraps every object prompt.
    """
    assert "concept art" not in prompt.PROMPT_TEMPLATE


def test_the_object_template_asks_for_one_subject_and_nothing_beside_it():
    """The positive half of the template rewrite, and positive on purpose: a user is
    free to empty ``negative_prompt``, so a constraint that lives only there is
    one the composed prompt can lose."""
    text = prompt.build("a rogue", {})
    assert "single subject" in text
    assert "no other objects" in text


def test_the_sheet_template_still_asks_for_a_grid():
    """The isolation clause must not leak across. SHEET_TEMPLATE restyles a
    contact sheet of eight real renders, so "one subject, nothing beside it" is
    the opposite of what it needs -- the two templates fight by design."""
    assert "grid of separate character poses" in prompt.SHEET_TEMPLATE
    assert "single subject" not in prompt.SHEET_TEMPLATE
    assert "no other objects" not in prompt.SHEET_TEMPLATE


# --- version 5: the taxonomy retirement ---------------------------------------

# The exact object prompt this compiler produces for the subject "a barrel"
# and no guidance at all. A literal on purpose: it is the safety argument that
# the empty-params composition is byte-identical across the taxonomy
# retirement (PROMPT_VERSION 4 -> 5) -- the view clause was re-inlined as
# exactly the fragment the deleted ``{view}`` slot used to receive. A test
# written against the template constant cannot make that argument, because it
# would follow the constant wherever it went.
DEFAULT_COMPOSITION = (
    "a barrel, a single subject centered on a plain light gray background, "
    "no other objects, 3/4 perspective view, studio lighting, game asset render, "
    "full object in frame, no cropping, no text, no watermark"
)


def test_the_default_composition_is_byte_identical_across_the_retirement():
    assert prompt.build("a barrel", {}) == DEFAULT_COMPOSITION
    # 6 was the scene-template/expansion bump, 7 the tile-sheet one and 8 the
    # deletion of the scene template with the expander that was its only
    # reader; the literal above still holding is the proof the object path did
    # not move with any of them.
    assert prompt.PROMPT_VERSION == 8


def test_the_tilesheet_template_asks_for_separate_tiles_not_one_scene():
    """The failure this template exists to prevent: handed "tile sheet" alone,
    the model draws one large top-down scene across the whole frame, which
    slices into sixty-four fragments of a picture rather than sixty-four
    tiles."""
    assert "separate tiles" in prompt.TILESHEET_TEMPLATE
    assert "each cell one complete standalone tile" in prompt.TILESHEET_TEMPLATE
    # SHEET_TEMPLATE's clause is about one character in eight poses, and is
    # exactly wrong where the cells are sixty-four different things.
    assert "character" not in prompt.TILESHEET_TEMPLATE
    assert "single subject" not in prompt.TILESHEET_TEMPLATE


def test_the_tilesheet_template_leaves_the_framing_to_the_pipeline():
    """The projection clause differs between orthogonal and isometric and sits
    under TILE_SHEET_VERSION, so it must not be duplicated here."""
    assert "isometric" not in prompt.TILESHEET_TEMPLATE
    assert "top-down" not in prompt.TILESHEET_TEMPLATE


def test_a_sheet_prompt_keeps_the_users_words_first():
    text = prompt.build("mossy dungeon floor", {}, tilesheet=True)
    assert text.startswith("mossy dungeon floor, ")
    assert "uniform grid of separate tiles" in text


def test_the_sheet_output_kind_wins_over_the_tile_one():
    """A sheet's cells are sixty-four different tiles, so the grid template
    decides which clauses may be present at all. This used to have a third
    flag -- the scene prompt mode -- which went with the expander."""
    both = prompt.build("stone", {}, tilesheet=True, tile=True)
    assert both == prompt.build("stone", {}, tilesheet=True)


def test_stale_taxonomy_params_compose_the_default():
    """A job row written before the retirement still composes -- its fragments
    are simply gone, which is the tolerance the retirement rests on."""
    old = {"framing": "front_ortho", "art_style": "nes", "category": "weapon"}
    assert prompt.build("a barrel", old) == DEFAULT_COMPOSITION


def test_a_stale_framing_is_inert_on_a_tile_too():
    plain = prompt.build("mossy cobblestone", {}, tile=True)
    assert prompt.build("mossy cobblestone", {"framing": "front_ortho"}, tile=True) == plain
    assert "orthographic" in plain  # the tile's own clause


def test_the_prompt_field_list_is_empty():
    """No stored field composes into the prompt any more; a future entry here
    is a deliberate re-opening, not an accident."""
    from warlock import guidance

    assert guidance._PROMPT_FIELDS == ()


def test_author_humanoid_report_uses_the_computed_species_count_not_a_literal():
    """pipelines-04 (2026-09-11 audit): author_humanoid.py's summary line
    hardcoded "for the same twelve characters" instead of the dynamic
    ``count`` variable its three sibling scripts (author_amorphous.py,
    author_quadruped.py, author_winged.py) already use for the identical
    sentence. The species count is silently correct today only by
    coincidence (12 humanoid species); a literal that drifts from its source
    of truth is a named defect class here. Compared against the siblings'
    own source text rather than a hardcoded expectation, so this only ever
    tests that the four scripts agree, not a copy of their wording.
    """
    import re
    from pathlib import Path

    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    sentence = re.compile(r'f"for the same .*?characters\."')

    def summary_line(name: str) -> str:
        text = (scripts_dir / name).read_text(encoding="utf-8")
        match = sentence.search(text)
        assert match, f"{name}: could not find the summary sentence to check"
        return match.group(0)

    siblings = {
        summary_line("author_amorphous.py"),
        summary_line("author_quadruped.py"),
        summary_line("author_winged.py"),
    }
    assert len(siblings) == 1, "the three siblings disagree with each other already"
    assert summary_line("author_humanoid.py") == next(iter(siblings))


def test_max_prompt_docstring_does_not_claim_truncation_the_encoder_no_longer_does():
    """pipelines-07 (2026-09-11 audit): MAX_PROMPT's comment justified the
    1000-character refusal with "the prompt ends up in an SDXL text encoder
    that truncates far earlier anyway" -- but ``prompt.chunk()`` (landed
    2026-08-02, one day before that comment was written 2026-08-03) removed
    exactly that truncation for the SDXL family, which docs/INVARIANTS.md now
    states explicitly: "The composed SDXL prompt is chunk-encoded, not
    truncated." A reader tuning MAX_PROMPT should not reason from a premise
    the code no longer has. Read from source, not a second copy of the
    wording, so this only ever tests what is actually written there.
    """
    import inspect

    from warlock.service import validation

    lines = inspect.getsource(validation).splitlines()
    target = next(
        i for i, line in enumerate(lines) if line.strip().startswith("MAX_PROMPT =")
    )
    comment_lines = []
    i = target - 1
    while i >= 0 and lines[i].strip().startswith("#"):
        comment_lines.insert(0, lines[i])
        i -= 1
    comment = "\n".join(comment_lines)
    assert comment, "MAX_PROMPT has no comment above it to check"
    assert "truncat" not in comment.lower(), comment
