"""Landing 2 of Create's cards: the Reference and Mesh columns (concept 03).

Every heading in the two columns is a ``widgets.group`` card; the brief is the
first card, the recipe/mesh settings follow, and the plan and the press are one
more card pinned under the scrolling form, identical on both stages. These tests
draw the real columns in a headless imgui frame and measure what the design
says: where the plan card and Generate sit, how far the form stops above them,
how wide every slider, drag and combo is, what a closed fold still tells the
reader, and what the brief card carries in its header.

Each claim is a thing the pre-card columns did not do: they opened with a 20 px
``pane_header``, drew a ``section_blocks`` fill instead of a child, put the plan
in a child of its own sized by a hand-typed ``sp(52)``, drew sliders and drags at
imgui's 65 % default width, and kept "Recent prompts..." in the Recipe section.
"""

from __future__ import annotations

import contextlib
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from imgui_bundle import imgui

from realmspinner import guidance
from realmspinner.studio import anchors, layout, probe, theme, widgets
from realmspinner.studio.modes.create.engine import recipe as create_recipe
from realmspinner.studio.modes.create.ui import brief as create_brief
from realmspinner.studio.modes.create.ui import workspace
from realmspinner.studio.modes.create.ui.panes import (
    settings_2d,
    settings_3d,
    settings_character,
)
from realmspinner.studio.state import AppState

_SOURCE = {
    "id": "ref-1",
    "name": "mossy well",
    "status": "done",
    "stage": "reference",
    "files": ["input.png"],
    "prompt": "a mossy well",
}

PANE = (320.0, 800.0)

# The real functions, taken once: ``draw_column`` is called several times in a
# test and each call must wrap the original, not the previous call's spy.
_REAL = SimpleNamespace(
    group=widgets.group,
    begin_child=imgui.begin_child,
    end_child=imgui.end_child,
    collapsing_header=imgui.collapsing_header,
    input_text_multiline=imgui.input_text_multiline,
)


@pytest.fixture
def ui(monkeypatch):
    """A bare headless imgui context with the control census on."""
    previous = imgui.get_current_context()
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    # Tall enough that a 2600 px pane is not clipped by the display (a clipped
    # control is not in the census as visible).
    io.display_size = (1600, 3200)
    io.delta_time = 1 / 60
    io.fonts.add_font_default()
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures.value
    theme.apply(imgui)
    monkeypatch.setattr(probe, "ENABLED", True)
    monkeypatch.setattr(widgets, "FORCE_SECTIONS_OPEN", False)
    monkeypatch.setattr(widgets, "_SETTINGS", None)
    yield imgui
    imgui.destroy_context(context)
    if previous is not None:
        imgui.set_current_context(previous)


def _ctx(stage: str, tmp_path: Path) -> Any:
    catalog = guidance.catalog()
    state = AppState(mode="create")
    state.create.stage = stage
    jobs: dict[str, Any] = {}
    if stage == "mesh":
        jobs[_SOURCE["id"]] = _SOURCE
        state.source_job = _SOURCE["id"]
    return SimpleNamespace(
        state=state,
        svc=SimpleNamespace(
            config=SimpleNamespace(bench_dir=str(tmp_path), palette_dir=tmp_path)
        ),
        guidance=catalog,
        base_models=[(m["key"], m["label"]) for m in catalog["fields"]["base_model"]],
        style_loras=[(m["key"], m["label"]) for m in catalog["fields"]["style_lora"]],
        busy=lambda _k: False,
        confirms=SimpleNamespace(ask=lambda *_a: None),
        cache=SimpleNamespace(get=jobs.get, jobs=list(jobs.values()), active=None),
        job=lambda: None,
        textures=None,
        model_rows=[],
        rig_templates=[{"key": "biped", "label": "Biped"}],
        rig_default="biped",
        rigging_available=True,
        submit=lambda *a, **k: True,
        toast=lambda *a, **k: None,
    )


class Drawn(SimpleNamespace):
    """What one drawn column left behind (see :func:`draw_column`)."""


def draw_column(
    ui_, monkeypatch, ctx, *, stage: str, size=PANE, frames: int = 5, between=None
) -> Drawn:
    """Draw ``stage``'s settings column for ``frames`` frames in one window.

    Records, from the last frame: the census, the anchors, every ``widgets.group``
    (title, id, keyword arguments, the content width at its body and its rect once
    closed), every child window by name (rect once closed), every header label,
    and the size of every multiline text box.
    """
    draw = settings_3d.draw if stage == "mesh" else settings_2d.draw
    state = Drawn(groups=[], children={}, child_names=[], headers=[], multilines=[])
    real_group = _REAL.group
    real_begin, real_end = _REAL.begin_child, _REAL.end_child
    real_header = _REAL.collapsing_header
    real_multiline = _REAL.input_text_multiline
    names: list[str] = []

    @contextlib.contextmanager
    def group(title=None, **kw):
        entry = SimpleNamespace(title=title, kw=kw, content_w=None, rect=None)
        state.groups.append(entry)
        with real_group(title, **kw) as opened:
            entry.content_w = ui_.get_content_region_avail().x
            yield opened
        low, high = ui_.get_item_rect_min(), ui_.get_item_rect_max()
        entry.rect = (low.x, low.y, high.x, high.y)

    def begin_child(name, *a, **k):
        names.append(str(name))
        state.child_names.append(str(name))
        return real_begin(name, *a, **k)

    def end_child():
        real_end()
        low, high = ui_.get_item_rect_min(), ui_.get_item_rect_max()
        state.children[names.pop()] = (low.x, low.y, high.x, high.y)

    def collapsing_header(label, *a, **k):
        state.headers.append(label)
        return real_header(label, *a, **k)

    def multiline(label, text, size_, *a, **k):
        state.multilines.append((label, size_.y if hasattr(size_, "y") else size_[1]))
        return real_multiline(label, text, size_, *a, **k)

    monkeypatch.setattr(widgets, "group", group)
    monkeypatch.setattr(ui_, "begin_child", begin_child)
    monkeypatch.setattr(ui_, "end_child", end_child)
    monkeypatch.setattr(ui_, "collapsing_header", collapsing_header)
    monkeypatch.setattr(ui_, "input_text_multiline", multiline)
    for index in range(frames):
        state.groups.clear()
        state.child_names.clear()
        state.headers.clear()
        state.multilines.clear()
        if between is not None:
            between(index)
        ui_.new_frame()
        layout.begin_frame()
        probe.begin_frame()
        anchors.begin_frame()
        ui_.set_next_window_pos((0.0, 0.0))
        ui_.set_next_window_size(size)
        ui_.begin("settings", None, ui_.WindowFlags_.no_title_bar.value)
        try:
            draw(ctx)
        finally:
            ui_.end()
            ui_.end_frame()
    state.census = probe.census()
    state.anchors = dict(anchors.FRAME_ANCHORS)
    state.spacing = ui_.get_style().item_spacing.y
    return state


def _by_id(drawn: Drawn, ident: str) -> Any:
    found = [g for g in drawn.groups if g.kw.get("id") == ident]
    assert len(found) == 1, (ident, [(g.title, g.kw.get("id")) for g in drawn.groups])
    return found[0]


# --- both stages are cards, and only cards --------------------------------------------


@pytest.mark.parametrize("module", [create_brief, settings_2d, settings_3d, settings_character])
def test_no_heading_in_the_create_columns_is_drawn_any_way_but_a_card(module):
    """A heading is a ``widgets.group``: no 20 px ``pane_header``, no tinted
    ``section_blocks`` fill, no bare ``secondary`` line standing in for a title,
    no ``widgets.section`` (which only means something inside a block scope)."""
    source = inspect.getsource(module)
    for retired in ("pane_header(", "section_blocks", "widgets.secondary(", "widgets.section("):
        assert retired not in source, (module.__name__, retired)


@pytest.mark.parametrize("stage", ["reference", "mesh"])
def test_both_stages_draw_the_column_as_cards_and_never_as_a_pane_header(
    ui, monkeypatch, tmp_path, stage
):
    called: list[str] = []
    for name in ("pane_header", "section", "secondary", "section_blocks"):
        real = getattr(widgets, name)

        def spy(*a, _n=name, _real=real, **k):
            called.append(_n)
            return _real(*a, **k)

        monkeypatch.setattr(widgets, name, spy)
    ctx = _ctx(stage, tmp_path)
    drawn = draw_column(ui, monkeypatch, ctx, stage=stage)
    assert called == [], f"{stage} drew a heading that is not a card: {called}"
    titles = [g.title for g in drawn.groups]
    brief_title = "Build from a reference" if stage == "mesh" else "Your brief"
    assert titles[0] == brief_title
    assert titles[-1] == "Generation plan"
    assert "Advanced" in "".join(t for t in titles if t)


# --- the plan card is one card, identical on both stages ------------------------------


def test_the_plan_card_top_and_generate_are_in_the_same_place_on_both_stages(
    ui, monkeypatch, tmp_path
):
    """Reference put Generate under a divider-and-gap footer and Mesh under a bare
    divider, so the footer's top moved by the difference between two hand-built
    spacers. The plan card is the foot of the column on both, so given the same
    lines (the plans differ by stage on purpose, and are stubbed to three lines
    here) its top edge, its bottom edge and the button's rect are identical."""

    def three_lines(ctx, plan, problems, repairs, **kw):
        for line in ("one", "two", "three"):
            ui.text(line)

    monkeypatch.setattr(workspace, "plan_footer", three_lines)
    seen: dict[str, Any] = {}
    for stage in ("reference", "mesh"):
        ctx = _ctx(stage, tmp_path)
        drawn = draw_column(ui, monkeypatch, ctx, stage=stage)
        card = _by_id(drawn, "plan")
        assert card.title == "Generation plan"
        seen[stage] = (card.rect, drawn.anchors["create/generate"])
    (ref_card, ref_gen), (mesh_card, mesh_gen) = seen["reference"], seen["mesh"]
    assert ref_gen == pytest.approx(mesh_gen, abs=0.5), "Generate's rect differs between stages"
    assert ref_card == pytest.approx(mesh_card, abs=0.5), "the plan card's rect differs"


@pytest.mark.parametrize("stage", ["reference", "mesh"])
def test_generate_is_forty_tall_and_fills_the_plan_card(ui, monkeypatch, tmp_path, stage):
    ctx = _ctx(stage, tmp_path)
    drawn = draw_column(ui, monkeypatch, ctx, stage=stage)
    card = _by_id(drawn, "plan")
    x, y, w, h = drawn.anchors["create/generate"]
    pad = widgets.sp(12)
    assert h == pytest.approx(widgets.sp(create_brief.GENERATE_H), abs=0.5)
    assert x - card.rect[0] == pytest.approx(pad, abs=0.5)
    assert card.rect[2] - (x + w) == pytest.approx(pad, abs=0.5), "full card width"
    assert card.rect[3] - (y + h) == pytest.approx(pad, abs=0.5), "12 below the button"


@pytest.mark.parametrize("stage", ["reference", "mesh"])
def test_the_form_stops_exactly_one_item_spacing_above_the_plan_card(
    ui, monkeypatch, tmp_path, stage
):
    """The reserve was a hand-typed ``sp(52)`` for a button drawn outside any
    child. It is the card's measured height plus the gap imgui puts between two
    children, so the form's bottom edge and the card's top edge are one spacing
    apart, and the card's bottom edge is the pane's."""
    ctx = _ctx(stage, tmp_path)
    drawn = draw_column(ui, monkeypatch, ctx, stage=stage)
    form = drawn.children["2d-form" if stage == "reference" else "3d-form"]
    card = _by_id(drawn, "plan")
    assert card.rect[1] - form[3] == pytest.approx(drawn.spacing, abs=0.5)
    assert "sp(52)" not in inspect.getsource(settings_2d) + inspect.getsource(settings_3d)
    # And what the card measured is what the next frame reserves.
    key = "2d" if stage == "reference" else "3d"
    assert ctx.state.create.plan_heights[key] == pytest.approx(card.rect[3] - card.rect[1], abs=0.5)


def test_a_taller_plan_moves_the_form_up_and_not_the_button(ui, monkeypatch, tmp_path):
    """A refusal adds lines to the plan; the button must stay on the pane's last
    row and the form give up exactly the height the card gained."""
    ctx = _ctx("reference", tmp_path)
    ctx.state.form_2d["prompt"] = "a mossy well"
    before = draw_column(ui, monkeypatch, ctx, stage="reference")
    ctx.state.create.submit_refusal = "The door said no, and then it said why. " * 8
    after = draw_column(ui, monkeypatch, ctx, stage="reference", frames=7)
    assert after.anchors["create/generate"] == pytest.approx(
        before.anchors["create/generate"], abs=0.5
    )
    gained = (_by_id(after, "plan").rect[3] - _by_id(after, "plan").rect[1]) - (
        _by_id(before, "plan").rect[3] - _by_id(before, "plan").rect[1]
    )
    assert gained > 0
    assert before.children["2d-form"][3] - after.children["2d-form"][3] == pytest.approx(
        gained, abs=1.0
    )


# --- every full-width control is as wide as its card ----------------------------------


def _loaded_reference(ctx, tmp_path):
    ref = tmp_path / "ref.png"
    ref.write_bytes(b"x")
    form = ctx.state.form_2d
    form.update(
        prompt="a mossy well",
        ref_path=str(ref),
        ip_adapter="plus",
        control="canny",
        init_image=True,
        style_lora="pixelxl",
        lora_weight=0.9,
    )
    return form


def _wide_controls(drawn: Drawn) -> list[Any]:
    kinds = {"slider_float", "slider_int", "drag_float", "drag_int", "combo", "input_int"}
    return [c for c in drawn.census if c.kind in kinds and c.visible]


@pytest.mark.parametrize("stage", ["reference", "mesh"])
def test_no_slider_drag_or_combo_is_narrower_than_the_card_it_is_in(
    ui, monkeypatch, tmp_path, stage
):
    """imgui draws a slider, drag or combo at 65 % of the row unless told
    otherwise: Style strength, the three Conditioning sliders, Size and the
    custom Triangles box were all narrower than the combo above them, and
    Triangles a fixed ``sp(140)``."""
    monkeypatch.setattr(widgets, "FORCE_SECTIONS_OPEN", True)
    ctx = _ctx(stage, tmp_path)
    if stage == "reference":
        _loaded_reference(ctx, tmp_path)
    else:
        from realmspinner.studio.modes.create.ui.panes import settings_3d as mod

        monkeypatch.setattr(mod.remesh_panel, "blender_available", lambda c: True)
        monkeypatch.setattr(mod.retarget_panel, "gltfpack_available", lambda c: True)
        ctx.state.form_3d.update(profile="custom", lowpoly_triangles=0, custom_triangles=50000)
        ctx.state.form_3d["mesh_seed"] = 7
    drawn = draw_column(ui, monkeypatch, ctx, stage=stage, size=(320.0, 2600.0))
    cards = [g for g in drawn.groups if g.content_w is not None]
    content = max(g.content_w for g in cards)
    controls = _wide_controls(drawn)
    assert len(controls) >= 6, [c.label for c in controls]
    narrow = [
        (c.label, round((c.hit or c.rect)[2], 1))
        for c in controls
        if (c.hit or c.rect)[2] < content - 1.5
    ]
    assert not narrow, f"narrower than the {content:.1f} px card content: {narrow}"
    if stage == "reference":
        labels = {c.label for c in controls}
        assert {"##Strength", "##Strength##ip", "##Strength##init", "##Strength##cn"} <= labels
    else:
        assert {"##Size", "##Triangles"} <= {c.label for c in controls}


def test_the_text_areas_in_the_reference_column_have_two_heights_and_both_are_scaled(
    ui, monkeypatch, tmp_path
):
    """The prompt was a literal 104 and the Avoid box a literal, unscaled 54."""
    monkeypatch.setattr(widgets, "FORCE_SECTIONS_OPEN", True)
    ctx = _ctx("reference", tmp_path)
    ctx.state.form_2d["prompt"] = "a mossy well"
    drawn = draw_column(ui, monkeypatch, ctx, stage="reference", size=(320.0, 2600.0))
    heights = dict(drawn.multilines)
    assert heights["##brief-prompt"] == pytest.approx(widgets.sp(create_brief.PROMPT_H))
    assert heights["##negative"] == pytest.approx(widgets.sp(create_brief.TEXT_AREA_H))
    assert (create_brief.PROMPT_H, create_brief.TEXT_AREA_H) == (96.0, 56.0)
    assert create_brief.GENERATE_H == 40.0


# --- a closed fold still says what it holds -------------------------------------------


def test_a_closed_advanced_card_still_says_the_seed_is_locked(ui, monkeypatch, tmp_path):
    ctx = _ctx("reference", tmp_path)
    ctx.state.form_2d["seed_locked"] = True
    drawn = draw_column(ui, monkeypatch, ctx, stage="reference")
    advanced = _by_id(drawn, "advanced")
    assert advanced.title == "Advanced - seed locked"
    assert advanced.kw["collapsible"] is True and advanced.kw["default_open"] is False
    assert "Advanced - seed locked##advanced" in drawn.headers
    # Closed: nothing inside it drew, so the seed row is not on screen.
    assert not [c for c in drawn.census if "seed" in c.label.lower() and c.kind != "button"]


def test_a_closed_conditioning_card_counts_what_is_attached_even_when_a_value_is_refused(
    ui, monkeypatch, tmp_path
):
    """INVARIANTS: a closed disclosure never hides a setting that is doing
    something. A held reference with an adapter and a structure pick, one of
    them refused by name (``ip_scale``), is on the header while the card is
    shut."""
    ctx = _ctx("reference", tmp_path)
    form = _loaded_reference(ctx, tmp_path)
    form["ip_scale"] = 9.0
    ctx.state.note_field_error("ip_scale", "The adapter strength is out of range.")
    drawn = draw_column(ui, monkeypatch, ctx, stage="reference")
    tail = create_recipe.conditioning_tail(form)
    assert tail.strip() == "(4 on)"
    assert f"Conditioning{tail}##conditioning" in drawn.headers
    assert "##Strength##ip" not in {c.label for c in drawn.census}, "the fold is shut"


class _Store:
    """The slice of ``Settings`` the persisted headers read and write."""

    def __init__(self, data=None):
        self.data = data or {}

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value


def test_advanced_and_conditioning_keep_their_open_state_and_a_tail_does_not_change_the_card(
    ui, monkeypatch, tmp_path
):
    """The cards are persisted through ``widgets.header`` (``create/advanced``,
    ``create/conditioning``), and the card's child id is the ``id=`` rather than
    the title, so ' - seed locked' appearing is the same window with the same
    open state."""
    store = _Store({"panels_open": {"create/advanced": True, "create/conditioning": False}})
    monkeypatch.setattr(widgets, "_SETTINGS", store)
    ctx = _ctx("reference", tmp_path)
    form = _loaded_reference(ctx, tmp_path)

    def seeds(drawn):
        return [c for c in drawn.census if c.label.startswith("##seed")]

    plain = draw_column(ui, monkeypatch, ctx, stage="reference")
    assert seeds(plain), "the stored-open Advanced card drew its seed field"
    assert "##Strength##ip" not in {c.label for c in plain.census}, "stored-closed stays closed"
    assert "##group/advanced" in plain.child_names

    form["seed_locked"] = True
    locked = draw_column(ui, monkeypatch, ctx, stage="reference")
    assert any(h.startswith("Advanced - seed locked") for h in locked.headers)
    assert "##group/advanced" in locked.child_names, "a new tail is not a new window"
    assert "##group/advanced - seed locked" not in locked.child_names
    assert seeds(locked), "and it is still open"
    assert store.data["panels_open"]["create/advanced"] is True

    # A one-shot open request (what a dropped reference file makes) still reaches
    # the card through the header it is drawn with.
    monkeypatch.setattr(widgets, "_OPEN_REQUESTS", {"create/conditioning"})
    reopened = draw_column(ui, monkeypatch, ctx, stage="reference", frames=2)
    assert "##Strength##ip" in {c.label for c in reopened.census}


# --- the brief card -------------------------------------------------------------------


def test_the_brief_card_holds_recent_prompts_as_a_header_action(ui, monkeypatch, tmp_path):
    """The button lived in the Recipe section beside the style strength: a
    button about the prompt, drawn under a control about the model."""
    ctx = _ctx("reference", tmp_path)
    ctx.state.history = ["a mossy well", "a rusty gate"]
    drawn = draw_column(ui, monkeypatch, ctx, stage="reference")
    brief_card = _by_id(drawn, "brief")
    assert brief_card.title == "Your brief"
    assert [a[1] for a in brief_card.kw["actions"]] == ["Recent prompts..."]
    recent = [c for c in drawn.census if c.text == "Recent prompts..."]
    assert len(recent) == 1
    # Inside the brief card's rect, on its header row, and nowhere in the Recipe.
    x, y, w, h = recent[0].rect
    assert brief_card.rect[1] <= y and y + h <= brief_card.rect[1] + widgets.sp(12 + 26) + 0.5
    assert brief_card.rect[0] <= x and x + w <= brief_card.rect[2]
    recipe = _by_id(drawn, "recipe")
    assert recipe.rect[1] >= brief_card.rect[3]
    assert "_history" not in dir(settings_2d)
    assert 'controls.button("Recent prompts' not in inspect.getsource(settings_2d)
    # No history, no button: an empty list is not an action.
    quiet = draw_column(ui, monkeypatch, _ctx("reference", tmp_path), stage="reference")
    assert not [c for c in quiet.census if c.text == "Recent prompts..."]
    assert not _by_id(quiet, "brief").kw["actions"]


def test_the_recent_prompts_header_action_opens_the_list_and_a_pick_fills_the_prompt(
    ui, monkeypatch, tmp_path
):
    """The action is a header button; the popup it opens is drawn by the card's
    body, in the card's own window, which is the only place imgui can find it."""
    from realmspinner.studio import controls

    ctx = _ctx("reference", tmp_path)
    ctx.state.history = ["a mossy well", "a rusty gate"]
    ctx.state.form_2d["prompt"] = "something else"
    click = {"pending": True}
    real_small = controls.small_button

    def small_button(label, *a, **k):
        if label.startswith("Recent prompts"):
            pressed, click["pending"] = click["pending"], False
            return pressed
        return real_small(label, *a, **k)

    popups: list[tuple[str, bool]] = []
    real_popup = ui.begin_popup

    def begin_popup(name, *a, **k):
        opened = real_popup(name, *a, **k)
        popups.append((name, opened))
        return opened

    items: list[str] = []

    def menu_item(label, *a, **k):
        items.append(label.split("##")[0])
        return (label.startswith("a rusty gate"), False)

    monkeypatch.setattr(controls, "small_button", small_button)
    monkeypatch.setattr(ui, "begin_popup", begin_popup)
    monkeypatch.setattr(controls, "menu_item", menu_item)
    draw_column(ui, monkeypatch, ctx, stage="reference", frames=2)
    assert (create_brief.HISTORY_POPUP, True) in popups, popups
    assert items[:2] == ["a mossy well", "a rusty gate"]
    assert ctx.state.form_2d["prompt"] == "a rusty gate"


def test_the_mesh_source_chip_is_a_filled_control_and_its_gap_is_one_item_spacing(
    ui, monkeypatch, tmp_path
):
    """The chip was a ghost button (a transparent rectangle on a card) and left
    the cursor one spacing below itself before the dummy added another: a 16 px
    gap in a column whose rhythm is 8."""
    ctx = _ctx("mesh", tmp_path)
    roles: list[Any] = []
    real_button = create_brief.controls.button

    def button(label, *a, **k):
        if label == "##mesh-source":
            roles.append(k.get("role", "secondary"))
        return real_button(label, *a, **k)

    monkeypatch.setattr(create_brief.controls, "button", button)
    seen: dict[str, float] = {}

    for _ in range(3):
        ui.new_frame()
        ui.set_next_window_size(PANE)
        ui.begin("chip")
        create_brief._source_chip(ctx, 300.0)
        seen["after"] = ui.get_cursor_screen_pos().y
        seen["spacing"] = ui.get_style().item_spacing.y
        ui.end()
        ui.end_frame()
    chip = [c for c in probe.census() if c.label == "##mesh-source"]
    assert chip, "the chip is a real button"
    _x, y, _w, h = chip[0].rect
    assert seen["after"] - (y + h) == pytest.approx(seen["spacing"], abs=0.5)
    assert roles and str(roles[-1]) != "ghost", "a filled control, not a ghost button"


@pytest.mark.parametrize(
    ("stage", "asset", "ident", "help_key"),
    [
        ("reference", "image", "recipe", "settings-2d"),
        ("reference", "tileset", "tileset", "settings-sheet"),
        ("reference", "sprite_sheet", "sprite-sheet", "settings-sheet"),
        ("reference", "character", "character", "settings-character"),
        ("mesh", "", "mesh", "settings-3d"),
    ],
)
def test_each_settings_card_carries_its_manual_button_on_its_own_header_row(
    ui, monkeypatch, tmp_path, stage, asset, ident, help_key
):
    """The (?) is the first call in the card's body and lands on the header row,
    compact, instead of on a line of its own under a 20 px heading."""
    ctx = _ctx(stage, tmp_path)
    if stage == "reference":
        ctx.state.form_2d["asset_type"] = asset
    drawn = draw_column(ui, monkeypatch, ctx, stage=stage)
    card = _by_id(drawn, ident)
    helps = [c for c in drawn.census if c.label.endswith(f"##help-{help_key}")]
    assert len(helps) == 1, [c.label for c in drawn.census if "help" in c.label]
    x, y, w, h = helps[0].rect
    pad = widgets.sp(12)
    assert y == pytest.approx(card.rect[1] + pad, abs=1.0), "on the header row"
    assert h == pytest.approx(widgets.sp(26), abs=1.0), "compact"
    assert x + w <= card.rect[2] - pad + 0.5, "inside the card's right inset"


# --- the pane itself ------------------------------------------------------------------


def test_the_settings_pane_passes_one_gutter_of_padding_for_the_card_stages_only():
    from realmspinner.studio.shell import frame

    source = inspect.getsource(frame)
    assert 'layout_mod.pane(\n                        "settings"' in source
    start = source.index('"settings",\n')
    block = source[start : start + 700]
    assert "padding=(" in block and "tokens_mod.SP_2" in block
    assert '("rig", "pose", "export")' in block, "Rig, Pose and Export keep the default padding"
    assert workspace.plan_card is not None
