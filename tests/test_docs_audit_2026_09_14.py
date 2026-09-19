"""Regression tests closing the docs/root-document findings of the 2026-09-14
audit (docs-01, docs-02, inker-03, shell-03, shell-07, pipelines-02,
packwright-02, docs-03, docs-04, docs-09, docs-10, docs-11, docs-12, tour-01).
See ``docs/audit-2026-09-14.md`` for the full findings; do not cite it from
``src/`` or ``scripts/`` (``tests/test_ux_todo_fixes.py`` refuses that).

Each test reads the document(s) from disk relative to the repo root and,
wherever the claim is about code, derives the expected truth from the code
itself (a registry, a pane's own source, a dataclass's fields) rather than
hard-coding it, so a later change that keeps the prose honest does not need a
second edit here.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(relpath: str) -> str:
    return (ROOT / relpath).read_text(encoding="utf-8")


def _normalize_ws(text: str) -> str:
    """Collapse whitespace so a phrase hand-wrapped across a line break (this
    repo's manual prose is wrapped at ~100-110 columns) is still found as one
    phrase -- the same idiom ``tests/test_licence_claims.py`` already uses."""
    return re.sub(r"\s+", " ", text)


# --- docs-01: THIRD-PARTY-NOTICES.md claims llama.cpp/the Familiar weights
# are "shown by hand in the tables above", but neither appeared in any
# table. Familiar's base moved from Gemma 4 E2B to Qwen3-VL-4B-Instruct
# (2026-09-16); the row this test looks for moved with it. -----------------


def test_third_party_notices_names_llama_cpp_and_qwen_where_it_claims_to():
    text = _read("THIRD-PARTY-NOTICES.md")
    marker = "all shown by\nhand in the tables above"
    idx = text.replace("\r\n", "\n").index(marker.replace("\r\n", "\n"))
    tables_above = text.replace("\r\n", "\n")[:idx]

    # A real table row (not just prose) for each, with the licence
    # docs/MODELS.md records (MIT for the runtime, Apache-2.0 for the
    # weights).
    llama_rows = [
        line
        for line in tables_above.splitlines()
        if line.startswith("|") and "llama.cpp" in line
    ]
    assert llama_rows, "no table row mentions llama.cpp above the claim"
    assert any("MIT" in row for row in llama_rows), llama_rows

    qwen_rows = [
        line
        for line in tables_above.splitlines()
        if line.startswith("|") and "Qwen" in line
    ]
    assert qwen_rows, "no table row mentions Qwen3-VL above the claim"
    assert any("Apache-2.0" in row for row in qwen_rows), qwen_rows


# --- docs-02: Manual 17 said an instance holds nothing of its own but a
# position; PrefabNode actually carries its own full transform. -----------


def test_manual_says_a_prefab_instance_carries_its_own_transform_not_only_position():
    from warlock.studio.modes.mason.engine import nodes as nd

    node_fields = {f.name for f in dataclasses.fields(nd.Node)}
    prefab_fields = {f.name for f in dataclasses.fields(nd.PrefabNode)}
    # PrefabNode's own fields, beyond the base Node it inherits, are exactly
    # what "and nothing else" must mean -- deliberately no per-child override.
    assert prefab_fields - node_fields == {"template"}
    assert {"translation", "rotation", "scale"} <= node_fields

    text = _normalize_ws(_read("docs/manual/17-dressing-a-scene.md"))
    assert "holds nothing of its own but a position" not in text
    # The fixed sentence must say the instance carries its own transform,
    # naming the pieces PrefabNode/Node actually carry.
    section = text[text.index("A prefab, and why it is not a copy") :]
    assert "own transform" in section
    assert "rotation" in section
    assert "scale" in section


# --- inker-03: Manual 29 said a reference layer "opens hidden"; asein.py
# keeps the file's own visibility bit and only warns about flattened export.


def test_manual_29_reference_layer_claim_matches_asein_visibility_behaviour():
    asein_src = _read("src/warlock/kernels/pixel/asein.py")
    # The literal warning text asein.py emits for a reference layer -- the
    # ground truth the chapter must agree with.
    warn_match = re.search(
        r'state\.warn\(\s*f"the reference layer \{name!r\} is an underlay;'
        r' a flattened export"\s*\n\s*" leaves it out"\s*\)',
        asein_src,
    )
    assert warn_match, "asein.py's reference-layer warning text moved or changed"

    text = _normalize_ws(_read("docs/manual/29-inker-animation.md"))
    assert "opens hidden" not in text
    assert "underlay" in text
    assert "flattened export" in text
    assert "leaves it out" in text


# --- shell-03: "an agent already running on this computer connects to it"
# reads as connecting to Familiar; must be unambiguously Warlock. ----------


def test_agent_settings_help_text_never_implies_agents_reach_familiar():
    targets = [
        "docs/manual/42-app-settings.md",
        "src/warlock/studio/modes/settings/ui/panes/app_settings.py",
        "docs/manual/46-extending.md",
    ]
    # The ambiguous old sentence, verbatim, in all three places docs-03
    # named. It must be gone from every one.
    ambiguous = re.compile(r"connects to \*?it\*?[,.]")
    for rel in targets:
        text = _read(rel)
        assert not ambiguous.search(text), f"{rel} still reads ambiguously"
        # And the fix must say plainly that the arrow points at Warlock.
        assert "connects inward to Warlock" in text, f"{rel} lost the fix"


# --- shell-07: the Models overview says "every model" but omits some
# fetch-group headings the pane actually draws. ----------------------------


def test_models_section_overview_names_every_fetch_group_heading():
    from warlock import fetch

    # Unique headings, in the order the pane groups rows under them --
    # derived from the same table the pane itself reads (fetch.GROUPS), so a
    # future registry addition that gains a new heading is caught here too.
    seen: list[str] = []
    for _kind, heading in fetch.GROUPS:
        if heading not in seen:
            seen.append(heading)

    text = _read("docs/manual/42-app-settings.md")
    start = text.index("## Models")
    end = text.index("\n## ", start + 1)
    section = _normalize_ws(text[start:end]).lower()
    for heading in seen:
        assert heading.lower() in section, f"{heading!r} missing from the Models overview"


# --- pipelines-02: chapter 22 says style LoRAs are SDXL-only, then
# describes the FLUX.2 klein pixel-art LoRA the picker offers. -------------


def test_manual_style_lora_sdxl_only_claim_does_not_contradict_the_flux2_entry():
    from warlock import models

    non_sdxl = [
        lora
        for lora in models.STYLE_LORAS.values()
        if lora.family != models.FAMILY_SDXL
    ]
    assert len(non_sdxl) == 1, non_sdxl
    exception_label = non_sdxl[0].label
    assert exception_label == "Pixel art (FLUX.2 klein)"

    text = _normalize_ws(_read("docs/manual/22-generating-references.md"))
    idx = text.index("SDXL-only")
    window = text[max(0, idx - 400) : idx + 600]
    # The sentence claiming SDXL-only must, in the same breath, name the one
    # style LoRA that is not -- not merely be contradicted pages later by the
    # per-style description.
    assert exception_label in window, window
    assert "exception" in window.lower()


# --- packwright-02: chapter 33 lists three refusal ceilings but not
# MAX_DOCUMENT_PIXELS, which also fires on an ordinary add. ----------------


def test_manual_packwright_chapter_names_the_document_wide_pixel_ceiling():
    from warlock.studio.modes.packwright.engine import wpack

    width = int(round(wpack.MAX_DOCUMENT_PIXELS**0.5))
    assert width * width == wpack.MAX_DOCUMENT_PIXELS
    million = round(wpack.MAX_DOCUMENT_PIXELS / 1_000_000)

    text = _read("docs/manual/33-packwright.md")
    start = text.index("## Starting an atlas")
    end = text.index("\n## ", start + 1)
    section = _normalize_ws(text[start:end])
    assert f"{width}×{width}" in section or f"{width}x{width}" in section
    assert str(million) in section
    assert "split" in section.lower()


# --- docs-03: SECURITY.md's scope never mentions llama-server.exe, a second
# loopback HTTP listener with an API-key file. ------------------------------


def test_security_md_names_familiars_loopback_listener():
    text = _read("SECURITY.md")
    start = text.index("## What is in scope")
    end = text.index("\n## ", start + 1)
    section = _normalize_ws(text[start:end])
    assert "llama-server.exe" in section
    assert "127.0.0.1" in section
    assert "key" in section.lower()


# --- docs-04: the pixel-art-xl pre-registration still reads as live and
# plans edits to guidance.PRESETS, deleted 2026-08-17. The document this test
# read, docs/measurements/2026-08-06-pixel-art-xl.md, moved to
# dev/measurements/ on 2026-09-16, so this whole regression moved to
# dev/tests/test_docs_audit_2026_09_14.py. ----------------------------------


# --- docs-09 / docs-10: manual bolds a control name that does not match the
# pane's own label. ----------------------------------------------------------


def _combo_label(source: str, key_literal: str) -> str:
    pattern = re.compile(
        r'form_ui\.combo\(\s*"' + re.escape(key_literal) + r'",\s*"([^"]+)"'
    )
    match = pattern.search(source)
    assert match, f"could not find the {key_literal!r} combo in the pane source"
    return match.group(1)


def test_manual_12_names_the_bg_removal_control_by_its_pane_label():
    pane_src = _read("src/warlock/studio/modes/create/ui/panes/settings_3d.py")
    label = _combo_label(pane_src, "bg_removal")
    assert label == "Background"

    text = _read("docs/manual/12-tuning-what-you-get.md")
    assert f"**{label}**" in text
    assert "**Background removal**" not in text


def test_manual_09_names_the_tileset_layout_combo_by_its_pane_label():
    pane_src = _read("src/warlock/studio/modes/create/ui/panes/settings_2d.py")
    label = _combo_label(pane_src, "mode")
    assert label == "Layout"

    text = _read("docs/manual/09-building-a-map.md")
    assert f"**{label}**" in text
    assert "**Tile layout**" not in text


# --- docs-11: chapter 24 says the toolbar carries up to twelve controls;
# overlay.py's comment said ten. Count the real controls in overlay.py. ----

_NUMBER_WORDS = {
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
}


def _toolbar_control_count() -> int:
    """The toolbar's own worst-case control count, derived from its source.

    Every control but one is drawn behind the local ``_wrap`` closure (so
    counting calls to it catches them all in one sweep); "Open in Inker" is
    the sole exception, since it is drawn first and needs no wrap of its own;
    and ``_front_yaw`` is invoked once from the toolbar but is its own
    logical control (it wraps its own one or two buttons), so it counts once
    here regardless of whether the "Reset front" button beside it is shown.
    """
    from warlock.studio.panes import overlay

    source = inspect.getsource(overlay.toolbar)
    tree = ast.parse(source)
    func = tree.body[0]
    assert isinstance(func, ast.FunctionDef)

    wrap_calls = 0
    front_yaw_calls = 0
    for node in ast.walk(func):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "_wrap":
                wrap_calls += 1
            elif node.func.id == "_front_yaw":
                front_yaw_calls += 1

    open_in_inker = 1 if "Open in Inker" in source else 0
    return wrap_calls + front_yaw_calls + open_in_inker


def _claimed_count(text: str, pattern: str) -> int:
    match = re.search(pattern, text)
    assert match, f"pattern {pattern!r} not found"
    word = match.group(1)
    assert word in _NUMBER_WORDS, f"unrecognized number word {word!r}"
    return _NUMBER_WORDS[word]


def test_viewport_toolbar_control_count_matches_manual_claim():
    real_count = _toolbar_control_count()

    manual_text = _read("docs/manual/24-the-3d-viewport.md")
    manual_claim = _claimed_count(manual_text, r"up to (\w+) controls")

    overlay_src = _read("src/warlock/studio/panes/overlay.py")
    code_claim = _claimed_count(overlay_src, r"up to (\w+) controls")

    assert manual_claim == real_count, (manual_claim, real_count)
    assert code_claim == real_count, (code_claim, real_count)


# --- docs-12: chapter 34 lists sprite sizes to 128 then Custom 8-256; 256 is
# itself a preset in charsheet.SIZES. ---------------------------------------


def test_troupe_manual_lists_256_as_a_preset_sprite_size_not_custom_only():
    from warlock.kernels import charsheet

    assert 256 in charsheet.SIZES

    text = _normalize_ws(_read("docs/manual/26-poser.md"))
    row_match = re.search(r"How many pixels tall one cell is\. (.*?) or \*\*Custom", text)
    assert row_match, "could not find the Sprite size row"
    preset_text = row_match.group(1)
    for size in charsheet.SIZES:
        assert re.search(rf"\b{size}\b", preset_text), (size, preset_text)
    # And it must not be relegated to "Custom" only.
    assert "128 or 256" in preset_text or re.search(r"128,? or 256", preset_text)


# --- tour-01: chapter 21 lists tours in a different order from TOURS'
# offer order. ---------------------------------------------------------------


def test_home_chapter_lists_tours_in_TOURS_offer_order():
    from warlock.studio.tour import scripts

    titles = [tour.title for tour in scripts.TOURS]
    assert len(titles) == 5

    text = _normalize_ws(_read("docs/manual/21-home.md"))
    positions = []
    for title in titles:
        marker = f"*{title}*"
        assert marker in text, f"{title!r} not found in chapter 21"
        positions.append(text.index(marker))

    assert positions == sorted(positions), (titles, positions)
