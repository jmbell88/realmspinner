"""The 2026-10-04 audit, Create's manual chapters and the brief/rail docstrings.

create-56  manual 12 said a distilled pick "clears both" the negative prompt and
           the structure control; ``clear_unusable`` clears only the structure
           control and the Avoid box is hidden, its text kept (as manual 22 says).
create-57  manual 23 said Budget "collapses to Raw alone" with neither Blender nor
           gltfpack; ``_budget`` draws no row at all.
create-58  manual 22 named "the References section"; the header is Conditioning.
create-59  the Character column draws Seed, Reroll and Lock seed and the chapter
           never listed them.
create-61  ``rail.py`` / ``brief.py`` docstrings described a bar row that no longer
           exists: a phantom caller of ``stage_rail_width``, a dead ``show_label``
           and its spacers in ``_count``, and "GENERATE_W on both".

The chapters are read as data and compared with what the pane's own source draws,
so a rename in either place fails here rather than in a reader's lap.
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path
from types import SimpleNamespace

from realmspinner import guidance
from realmspinner.studio.modes.create.engine import assets as create_assets
from realmspinner.studio.modes.create.engine import recipe as create_recipe
from realmspinner.studio.modes.create.ui import brief as create_brief
from realmspinner.studio.modes.create.ui import rail as create_rail
from realmspinner.studio.modes.create.ui.panes import settings_2d, settings_3d, settings_character
from realmspinner.studio.panes import remesh_panel, retarget_panel
from realmspinner.studio.state import default_form_2d

MANUAL = Path(__file__).resolve().parents[3] / "docs" / "manual"


def _chapter(name: str) -> str:
    # Whitespace folded: the chapters hard-wrap at ~100 columns and a sentence
    # this test looks for may straddle a wrap.
    return " ".join((MANUAL / f"{name}.md").read_text(encoding="utf-8").split())


def _section(text: str, heading: str, level: str = "##") -> str:
    """The body under ``heading`` up to the next heading of the same level."""
    start = text.index(f"{level} {heading}")
    nxt = re.search(rf"\s{re.escape(level)} [^#]", text[start + len(heading) + 4 :])
    end = start + len(heading) + 4 + nxt.start() if nxt else len(text)
    return text[start:end]


# --- create-56 -------------------------------------------------------------------


def test_picking_a_distilled_checkpoint_keeps_the_avoid_text_as_manual_12_says():
    """The pane behaves one way and the chapter has to say that way: the structure
    control is cleared, the Avoid text is not."""
    ctx = SimpleNamespace(guidance=guidance.catalog())
    form = default_form_2d()
    create_assets.sync_legacy_fields(form)
    form.update(
        base_model="turbo",
        negative_prompt="blurry, watermark",
        ref_path="ref.png",
        control=guidance.catalog()["fields"]["control"][0]["key"],
    )
    create_recipe.clear_unusable(ctx, form)
    assert form["control"] == ""
    assert form["negative_prompt"] == "blurry, watermark"

    para = _section(_chapter("12-tuning-what-you-get"), "Models and LoRAs")
    assert "picking one clears both" not in para
    assert "clears the structure control" in para
    # The kept text is named by the box it lives in, as chapter 22 names it.
    assert "Avoid" in para
    assert "hides" in para and "keeping its text" in para


# --- create-57 -------------------------------------------------------------------


def test_budget_row_is_absent_without_blender_and_gltfpack(monkeypatch):
    monkeypatch.setattr(remesh_panel, "blender_available", lambda _ctx: False)
    monkeypatch.setattr(retarget_panel, "gltfpack_available", lambda _ctx: False)
    drawn: list[str] = []
    monkeypatch.setattr(
        settings_3d.widgets,
        "labeled_combo",
        lambda label, value, options, **kw: drawn.append(label) or value,
    )
    form = {"profile": "standard", "lowpoly_triangles": 5000, "custom_triangles": 0}
    settings_3d._budget(SimpleNamespace(state=SimpleNamespace()), form)
    assert drawn == []
    assert form["profile"] == "raw" and form["lowpoly_triangles"] == 0

    text = _chapter("23-generating-meshes")
    assert "collapses to Raw alone" not in text
    assert "draws no **Budget** row" in text
    # The retarget panel's default, which the chapter states beside it.
    assert "Standard is this panel's default" in text
    assert "Raw when `gltfpack` is absent" in text
    assert retarget_panel.TIERS  # the panel the sentence is about still exists


def test_manual_23_states_the_custom_start_and_the_rig_rule_the_pane_applies():
    """Two behaviours another fixer's change in the same audit added: Custom seeds
    the count (``_apply_budget_choice``) and a host without Blender never sends a
    saved rig request (``promote_kwargs(rig_available=False)``)."""
    from realmspinner.pipelines import optimize
    from realmspinner.studio.modes.create.engine import mesh as create_mesh

    form = {"profile": "raw", "lowpoly_triangles": 0, "custom_triangles": 0}
    settings_3d._apply_budget_choice(form, "custom")
    assert form["custom_triangles"] == optimize.PROFILES["standard"] == 50_000
    assert (optimize.CUSTOM_MIN, optimize.CUSTOM_MAX) == (5_000, 250_000)
    assert create_mesh.promote_kwargs(_mesh_form(), rig_available=False)["rig"] is False
    assert create_mesh.promote_kwargs(_mesh_form(), rig_available=True)["rig"] is True

    text = _chapter("23-generating-meshes")
    assert "Picking Custom starts the count at 50,000" in text
    assert "outside 5,000 to 250,000 is refused in the plan footer" in text
    assert "naming the **Triangles** field" in text
    assert "Without Blender, Make 3D also never requests a rig, even if the setting was on" in text


def _mesh_form() -> dict:
    from realmspinner.studio.state import DEFAULT_FORM_3D

    form = dict(DEFAULT_FORM_3D)
    form["rig"] = True
    return form


# --- create-58 -------------------------------------------------------------------


def test_manual_22_names_the_conditioning_header_the_pane_draws():
    assert 'f"Conditioning{create_recipe.conditioning_tail(form)}##create"' in inspect.getsource(
        settings_2d
    )
    section = _section(_chapter("22-generating-references"), "Conditioning on an image")
    assert "In the **Conditioning** section" in section
    assert "**References** section" not in section


# --- create-59 -------------------------------------------------------------------


def test_character_controls_in_the_manual_match_the_column():
    """The column draws ``settings_2d._seed_row`` -- Seed, Reroll, Lock seed -- so
    the Controls subsection lists those three captions by name."""
    assert "settings_2d._seed_row(ctx, form, form_ui)" in inspect.getsource(
        settings_character.draw_block
    )
    row = inspect.getsource(settings_2d._seed_row)
    captions = ('"Seed"', '"Reroll"', '"Lock seed"')
    for caption in captions:
        assert caption in row, f"_seed_row no longer draws {caption}"

    controls = _section(_chapter("22-generating-references"), "The controls", level="###")
    for caption in captions:
        assert f"**{caption.strip(chr(34))}**" in controls, caption


# --- create-61 -------------------------------------------------------------------


def _docstrings(module) -> list[str]:
    tree = ast.parse(inspect.getsource(module))
    out = [ast.get_docstring(tree) or ""]
    out += [
        ast.get_docstring(node) or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    ]
    return out


def test_every_documented_caller_in_rail_and_brief_docstrings_exists():
    """No docstring may name a caller that does not call. ``stage_rail_width`` is a
    measuring seam only tests use, ``_count`` has one mode (no label, no spacers)
    and ``_generate``'s width is the column's, not ``GENERATE_W``."""
    brief_src = inspect.getsource(create_brief)
    rail_src = inspect.getsource(create_rail)
    assert "stage_rail_width" not in brief_src

    # The phantom: "create_brief measures the rail with stage_rail_width".
    rail_doc = ast.get_docstring(ast.parse(rail_src)) or ""
    assert not re.search(r"create_brief\W+measures", " ".join(rail_doc.split()))

    # Every ``:func:`name``` a docstring points at is a real name in one of the two.
    known = {*dir(create_rail), *dir(create_brief)}
    for module in (create_rail, create_brief):
        for doc in _docstrings(module):
            for name in re.findall(r":(?:func|data):`([A-Za-z_]\w*)`", doc):
                assert name in known, f"{module.__name__} docstring names {name}"

    # The dead parameter and the spacers that centred a label in a bar row.
    assert list(inspect.signature(create_brief._count).parameters) == [
        "ctx",
        "form",
        "counts",
        "current",
    ]
    count_src = inspect.getsource(create_brief._count)
    assert "imgui.dummy" not in count_src
    assert "show_label" not in brief_src

    # "GENERATE_W on both" was untrue: the column passes its own width.
    generate_doc = ast.get_docstring(
        next(
            n
            for n in ast.walk(ast.parse(brief_src))
            if isinstance(n, ast.FunctionDef) and n.name == "_generate"
        )
    )
    assert "on both" not in generate_doc
    assert "width=imgui.get_content_region_avail().x" in inspect.getsource(
        create_brief.submit_control
    )
