"""``widgets.group``: the card every Create pane is built from.

A card is a *child window*, not a draw-list fill, because the thing under test is
a right inset and ``section_blocks`` cannot give one: its fill bleeds outward from
the content, so ``-1`` widths, ``avail.x`` and a right-aligned (?) all ran to the
window edge. These tests measure the geometry the design (concept 03) specifies --
12 px of padding on all four sides, a 26 px header row, 8 px from the row to the
body and 8 px between cards -- off a real headless imgui layout, at two UI scales.

An ``auto_resize_y`` child is a frame late, so every measurement is taken on the
third frame of a repeated host window.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from realmspinner.studio import controls, forms, layout, theme, tokens, widgets
from realmspinner.studio.tokens import sp

HOST = "##host"


@pytest.fixture(params=[1.0, 1.5], ids=["x1", "x1.5"])
def ui(request, monkeypatch):
    """A headless imgui at the requested UI scale, theme applied at that scale."""
    from _ui_context import imgui_context

    old = tokens.SCALE
    tokens.set_scale(request.param)
    try:
        with imgui_context(monkeypatch) as imgui:
            yield imgui
    finally:
        tokens.set_scale(old)


def draw(imgui, build, *, frames=3, size=(420.0, 900.0)):
    """Run ``build`` in one host window for ``frames`` frames; -> the last result."""
    out = None
    for _ in range(frames):
        imgui.new_frame()
        imgui.set_next_window_pos((0.0, 0.0))
        imgui.set_next_window_size(size)
        imgui.begin(
            HOST,
            None,
            imgui.WindowFlags_.no_title_bar.value | imgui.WindowFlags_.no_scrollbar.value,
        )
        try:
            out = build()
        finally:
            imgui.end()
        imgui.end_frame()
    return out


def rect(imgui):
    lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
    return lo.x, lo.y, hi.x, hi.y


def owner(imgui) -> str:
    return imgui.get_window_draw_list()._owner_name


def colour(imgui, which):
    v = imgui.get_style_color_vec4(which)
    return (round(v.x, 4), round(v.y, 4), round(v.z, 4), round(v.w, 4))


def want(value, alpha=1.0):
    return tuple(round(c, 4) for c in theme.rgba(value, alpha))


def approx(value):
    return pytest.approx(value, abs=0.01)


# --- geometry ------------------------------------------------------------------


def test_a_group_insets_its_content_12_design_px_on_all_four_sides(ui):
    imgui = ui

    def build():
        out = {}
        with widgets.group(None, id="inset"):
            win, size = imgui.get_window_pos(), imgui.get_window_size()
            imgui.button("##fill", (imgui.get_content_region_avail().x, sp(30)))
            out["item"] = rect(imgui)
            out["box"] = (win.x, win.y, win.x + size.x, win.y + size.y)
        return out

    out = draw(imgui, build)
    left, top, right, bottom = out["box"]
    ix0, iy0, ix1, iy1 = out["item"]
    pad = sp(12)
    assert ix0 - left == approx(pad), "left inset"
    assert iy0 - top == approx(pad), "top inset"
    assert right - ix1 == approx(pad), "a full-width control stops 12 short of the right edge"
    assert bottom - iy1 == approx(pad), "bottom inset: no trailing spacing counted in the height"


def test_a_group_header_row_is_26_tall_and_the_body_starts_8_below_it(ui):
    imgui = ui

    def build():
        out = {}
        with widgets.group("Recipe", id="row"):
            win = imgui.get_window_pos()
            out["row"] = rect(imgui)  # the last item is the header row itself
            out["body"] = imgui.get_cursor_screen_pos().y
            out["top"] = win.y
        return out

    out = draw(imgui, build)
    _x0, y0, _x1, y1 = out["row"]
    assert y1 - y0 == approx(sp(26))
    assert y0 - out["top"] == approx(sp(12))
    assert out["body"] - y1 == approx(sp(8)), "header row to first body item"


def test_a_collapsible_groups_header_row_is_26_tall_too(ui):
    imgui = ui

    def build():
        out = {}
        with widgets.group("Advanced", id="adv", collapsible=True, default_open=True):
            out["row"] = rect(imgui)
            out["body"] = imgui.get_cursor_screen_pos().y
        return out

    out = draw(imgui, build)
    _x0, y0, _x1, y1 = out["row"]
    assert y1 - y0 == approx(sp(26))
    assert out["body"] - y1 == approx(sp(8))


def test_a_collapsed_collapsible_group_is_50_design_px_tall(ui):
    imgui = ui

    def build():
        out = {}
        with widgets.group("Advanced", id="closed", collapsible=True, default_open=False) as opened:
            out["opened"] = opened
            out["height"] = imgui.get_window_size().y
        return out

    out = draw(imgui, build)
    assert out["opened"] is False
    assert out["height"] == approx(sp(12 + 26 + 12))


def test_a_header_less_group_has_no_header_row(ui):
    imgui = ui

    def build():
        with widgets.group(None, id="bare"):
            return imgui.get_cursor_screen_pos().y - imgui.get_window_pos().y

    assert draw(imgui, build) == approx(sp(12))


def test_group_actions_are_compact_ghost_buttons_flush_with_the_right_inset(ui):
    imgui = ui
    pressed = []

    def build():
        out = {}
        with widgets.group(
            "History",
            id="acts",
            actions=(("export", "Export", lambda: pressed.append("export")),),
        ):
            win, size = imgui.get_window_pos(), imgui.get_window_size()
            out["row"] = rect(imgui)
            out["right"] = win.x + size.x
            out["top"] = win.y
        return out

    out = draw(imgui, build)
    x0, y0, x1, y1 = out["row"]
    assert x1 == approx(out["right"] - sp(12))
    assert y1 - y0 == approx(sp(26))
    assert y0 == approx(out["top"] + sp(12))
    assert not pressed, "drawing the actions must not fire them"


def test_two_groups_in_a_group_stack_are_8_design_px_apart(ui):
    imgui = ui

    def build():
        out = {}
        with widgets.group_stack():
            with widgets.group("One", id="one"):
                imgui.button("##a", (-1, sp(30)))
                out["one"] = imgui.get_window_pos().y + imgui.get_window_size().y
            with widgets.group("Two", id="two"):
                out["two"] = imgui.get_window_pos().y
        return out

    out = draw(imgui, build)
    assert out["two"] - out["one"] == approx(sp(8))


def test_the_group_stack_gutter_does_not_follow_the_global_item_spacing(ui):
    """imgui separates two children by ``ItemSpacing.y``, which is one number for
    the app. The gutter is a property of the cards, so a style that moved it must
    not move the gutter -- and, outside a stack, nothing is promised or changed."""
    imgui = ui

    def gap(stacked):
        def build():
            out = {}
            imgui.push_style_var(imgui.StyleVar_.item_spacing.value, (sp(8), sp(20)))
            try:
                scope = widgets.group_stack() if stacked else _null()
                with scope:
                    with widgets.group("One", id="g1"):
                        imgui.button("##a", (-1, sp(30)))
                        out["one"] = imgui.get_window_pos().y + imgui.get_window_size().y
                    with widgets.group("Two", id="g2"):
                        out["two"] = imgui.get_window_pos().y
                        imgui.button("##b", (-1, sp(30)))
            finally:
                imgui.pop_style_var()
            return out["two"] - out["one"]

        return draw(imgui, build)

    assert gap(True) == approx(sp(8))
    assert gap(False) == approx(sp(20))


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def test_a_group_title_tail_does_not_change_the_cards_identity(ui):
    """The Advanced fold's title grows " - seed locked"; ``##`` pins the id, or
    the card is a different window (and a one-frame height jump) each time."""
    imgui = ui
    seen = []

    def build(title):
        def inner():
            with widgets.group(title, collapsible=True):
                seen.append(owner(imgui))

        return inner

    draw(imgui, build("Advanced##create/adv"), frames=1)
    draw(imgui, build("Advanced - seed locked##create/adv"), frames=1)
    assert len(set(seen)) == 1, seen


# --- the header's persistence is still header()'s ----------------------------------


class _Settings:
    def __init__(self, panels):
        self.panels = dict(panels)

    def get(self, name, default=None):
        return self.panels if name == "panels_open" else default

    def set(self, name, value):
        if name == "panels_open":
            self.panels = dict(value)


def test_a_collapsible_group_follows_the_persisted_open_state(ui, monkeypatch):
    imgui = ui
    monkeypatch.setattr(widgets, "_SETTINGS", _Settings({"create/adv": False}))

    def build():
        with widgets.group("Advanced", id="p", key="create/adv", collapsible=True) as opened:
            return opened

    assert draw(imgui, build, frames=2) is False


def test_an_open_request_opens_a_collapsed_collapsible_group_once(ui, monkeypatch):
    imgui = ui
    monkeypatch.setattr(widgets, "_SETTINGS", _Settings({"create/cond": False}))
    widgets.request_open("create/cond")

    def build():
        with widgets.group("Conditioning", id="r", key="create/cond", collapsible=True) as opened:
            return opened

    assert draw(imgui, build, frames=1) is True
    assert not widgets.open_requested("create/cond"), "the request is one-shot"


# --- balance ---------------------------------------------------------------------


class _Boom(Exception):
    pass


def test_a_group_whose_body_raises_still_balances_the_window_and_style_stacks(ui):
    imgui = ui

    def build():
        before = (
            colour(imgui, imgui.Col_.frame_bg),
            colour(imgui, imgui.Col_.child_bg),
            tuple(imgui.get_style().window_padding),
            tuple(imgui.get_style().frame_padding),
        )
        host = owner(imgui)
        with pytest.raises(_Boom), widgets.group("Boom", id="boom", collapsible=True):
            raise _Boom
        after = (
            colour(imgui, imgui.Col_.frame_bg),
            colour(imgui, imgui.Col_.child_bg),
            tuple(imgui.get_style().window_padding),
            tuple(imgui.get_style().frame_padding),
        )
        return before, after, host, owner(imgui)

    before, after, host, back = draw(imgui, build, frames=1)
    assert before == after
    assert back == host, "the child was ended"
    assert not widgets.in_group()
    assert controls.elevation_depth() == 0
    assert widgets._PLAIN_HEADER is False


def test_card_sections_closes_its_open_card_when_the_body_raises(ui):
    imgui = ui

    def build():
        host = owner(imgui)
        with pytest.raises(_Boom), widgets.card_sections():
            widgets.section("One")
            raise _Boom
        return host, owner(imgui)

    host, back = draw(imgui, build, frames=1)
    assert back == host
    assert not widgets.in_group()
    assert widgets._CARD_SCOPES == []


# --- elevation -------------------------------------------------------------------


def test_field_frames_resolve_to_elev_2_inside_a_group_and_elev_1_outside(ui):
    imgui = ui

    def build():
        out = {"outside": colour(imgui, imgui.Col_.frame_bg)}
        with widgets.group(None, id="e"):
            out["inside"] = colour(imgui, imgui.Col_.frame_bg)
            out["hover"] = colour(imgui, imgui.Col_.frame_bg_hovered)
            out["active"] = colour(imgui, imgui.Col_.frame_bg_active)
        out["after"] = colour(imgui, imgui.Col_.frame_bg)
        return out

    out = draw(imgui, build, frames=1)
    assert out["outside"] == want(theme.ELEV_1) == out["after"]
    assert out["inside"] == want(theme.ELEV_2)
    assert out["hover"] == out["active"] == want(theme.EDGE)


def _button_fill(imgui, role, selected=False):
    with controls._button_colours(role, selected, controls.ControlState.DEFAULT):
        return (
            colour(imgui, imgui.Col_.button),
            colour(imgui, imgui.Col_.button_hovered),
        )


def test_a_secondary_button_rests_at_elev_2_inside_a_group_and_elev_1_outside(ui):
    imgui = ui

    def build():
        outside = _button_fill(imgui, controls.ButtonRole.SECONDARY)
        with widgets.group(None, id="b"):
            inside = _button_fill(imgui, controls.ButtonRole.SECONDARY)
        return outside, inside

    outside, inside = draw(imgui, build, frames=1)
    assert outside == (want(theme.ELEV_1), want(theme.ELEV_2))
    assert inside == (want(theme.ELEV_2), want(theme.EDGE))


def test_a_ghost_button_hovers_at_edge_inside_a_group_and_elev_2_outside(ui):
    imgui = ui

    def build():
        outside = _button_fill(imgui, controls.ButtonRole.GHOST)
        selected = None
        with widgets.group(None, id="gh"):
            inside = _button_fill(imgui, controls.ButtonRole.GHOST)
            selected = _button_fill(imgui, controls.ButtonRole.GHOST, selected=True)
        return outside, inside, selected

    outside, inside, selected = draw(imgui, build, frames=1)
    assert outside == ((0.0, 0.0, 0.0, 0.0), want(theme.ELEV_2))
    assert inside == ((0.0, 0.0, 0.0, 0.0), want(theme.EDGE))
    assert selected[0] == want(theme.ACCENT, tokens.SELECTION_WASH_ALPHA), "the wash stays"


def test_a_combo_cap_is_pushed_to_edge_only_inside_a_group(ui, monkeypatch):
    imgui = ui
    pushes = []
    real = imgui.push_style_color

    def spy(col, value):
        pushes.append((col, tuple(round(c, 4) for c in (value.x, value.y, value.z, value.w))))
        return real(col, value)

    monkeypatch.setattr(imgui, "push_style_color", spy)

    def build():
        controls.combo_native("##out", 0, ["a", "b"])
        outside = [p for p in pushes if p[0] == imgui.Col_.button.value]
        pushes.clear()
        with widgets.group(None, id="cb"):
            pushes.clear()
            controls.combo_native("##in", 0, ["a", "b"])
            native = [p for p in pushes if p[0] == imgui.Col_.button.value]
            pushes.clear()
            controls.combo("##menu", "a", [("a", "A")])
            menu = [p for p in pushes if p[0] == imgui.Col_.button.value]
            after = colour(imgui, imgui.Col_.button)
        return outside, native, menu, after, colour(imgui, imgui.Col_.button)

    outside, native, menu, inside_after, back = draw(imgui, build, frames=1)
    assert outside == []
    assert native == [(imgui.Col_.button.value, want(theme.EDGE))]
    assert menu == [(imgui.Col_.button.value, want(theme.EDGE))]
    assert inside_after == back == want(theme.ELEV_2), "pushed around the one call, not the card"


def test_segmented_control_and_list_row_draw_inside_a_group(ui):
    """Their hard-coded ELEV_1 fills read the depth; this pins that they still
    draw (the fill colour itself lives in a draw list a headless test cannot read)."""
    imgui = ui

    def build():
        with widgets.group(None, id="seg"):
            picked = widgets.segmented_control("seg", [("a", "A"), ("b", "B")], "a")
            with widgets.list_row("row-1", stripe=True) as clicked:
                imgui.text("row")
        return picked, clicked

    assert draw(imgui, build, frames=2) == ("a", False)


# --- Form rhythm -----------------------------------------------------------------


def _two_fields(imgui):
    out = {}
    with forms.Form("f") as form:
        with form.field("a", "Alpha"):
            out["label_a"] = rect(imgui)
            imgui.button("##c1", (-1, sp(30)))
            out["ctl_a"] = rect(imgui)
        with form.field("b", "Beta"):
            out["label_b"] = rect(imgui)
            imgui.button("##c2", (-1, sp(30)))
            out["ctl_b"] = rect(imgui)
    return out


def test_form_fields_are_4_then_12_apart_inside_a_group(ui):
    imgui = ui

    def build():
        with widgets.group(None, id="form"):
            out = _two_fields(imgui)
            win, size = imgui.get_window_pos(), imgui.get_window_size()
            out["bottom"] = win.y + size.y
        return out

    out = draw(imgui, build)
    assert out["ctl_a"][1] - out["label_a"][3] == approx(sp(4)), "label to control"
    assert out["label_b"][1] - out["ctl_a"][3] == approx(sp(12)), "field to field"
    assert out["ctl_b"][1] - out["label_b"][3] == approx(sp(4))
    assert out["bottom"] - out["ctl_b"][3] == approx(sp(12)), "no trailing field gap in the card"


def test_form_fields_keep_8_and_20_outside_a_group(ui):
    out = draw(ui, lambda: _two_fields(ui))
    assert out["ctl_a"][1] - out["label_a"][3] == approx(sp(8))
    assert out["label_b"][1] - out["ctl_a"][3] == approx(sp(20))


def test_a_helper_line_in_a_group_sits_4_under_its_control(ui):
    imgui = ui

    def build():
        out = {}
        with widgets.group(None, id="helper"), forms.Form("f") as form:
            with form.field("a", "Alpha", helper="Under the control."):
                imgui.button("##c", (-1, sp(30)))
                out["ctl"] = rect(imgui)
            out["helper"] = rect(imgui)
        return out

    out = draw(imgui, build)
    assert out["helper"][1] - out["ctl"][3] == approx(sp(4))


# --- the (?) ---------------------------------------------------------------------


def _help_pane() -> str:
    from realmspinner.kernels.manual.targets import HELP_TARGETS

    return next(iter(HELP_TARGETS))


def test_help_button_on_a_group_header_is_a_26_square_on_the_row(ui):
    from realmspinner.studio.manual import render as manual_render

    imgui = ui

    def build():
        out = {}
        with widgets.group("Recipe", id="help"):
            win, size = imgui.get_window_pos(), imgui.get_window_size()
            manual_render.help_button(SimpleNamespace(), _help_pane())
            out["btn"] = rect(imgui)
            out["box"] = (win.x, win.y, win.x + size.x)
            out["after"] = imgui.get_cursor_screen_pos().y
            out["row"] = widgets.group_header_row()
        return out

    out = draw(imgui, build)
    x0, y0, x1, y1 = out["btn"]
    left, top, right = out["box"]
    assert (x1 - x0, y1 - y0) == (approx(sp(26)), approx(sp(26)))
    assert x1 == approx(right - sp(12))
    assert y0 == approx(top + sp(12)), "centred in the row: the row is the button's height"
    assert out["after"] - y1 == approx(sp(8)), "the body still starts 8 under the row"
    assert out["row"], "a (?) does not end the header row"


def test_help_button_in_a_group_body_is_the_ordinary_one(ui):
    from realmspinner.studio.manual import render as manual_render

    imgui = ui

    def build():
        with widgets.group("Recipe", id="body"):
            imgui.button("##first", (-1, sp(30)))
            assert not widgets.group_header_row()
            manual_render.help_button(SimpleNamespace(), _help_pane())
            return rect(imgui), imgui.get_frame_height()

    (x0, y0, x1, y1), frame = draw(imgui, build)
    assert y1 - y0 == approx(frame)


def test_help_button_outside_a_group_is_unchanged(ui):
    from realmspinner.studio.manual import render as manual_render

    imgui = ui

    def build():
        manual_render.help_button(SimpleNamespace(), _help_pane())
        return rect(imgui), imgui.get_frame_height()

    (x0, y0, x1, y1), frame = draw(imgui, build)
    assert (x1 - x0, y1 - y0) == (approx(frame), approx(frame))


def test_a_help_button_sits_left_of_the_header_actions_not_on_them(ui):
    from realmspinner.studio.manual import render as manual_render

    imgui = ui

    def build():
        out = {}
        with widgets.group("History", id="both", actions=(("go", "Export", lambda: None),)):
            out["action"] = rect(imgui)
            manual_render.help_button(SimpleNamespace(), _help_pane())
            out["help"] = rect(imgui)
        return out

    out = draw(imgui, build)
    assert out["help"][2] <= out["action"][0]


# --- layout.pane -------------------------------------------------------------------


def _pane_inset(imgui, **kwargs):
    def build():
        with layout.pane("##p", (300.0, 200.0), **kwargs):
            return imgui.get_cursor_screen_pos().x - imgui.get_window_pos().x

    return draw(imgui, build, frames=1)


def test_pane_padding_override_replaces_the_default_inset(ui):
    assert _pane_inset(ui, padding=tokens.SP_2) == approx(sp(8))


def test_pane_without_a_padding_argument_keeps_pane_padding(ui):
    assert _pane_inset(ui) == approx(sp(layout.PANE_PADDING))


# --- card_sections ---------------------------------------------------------------


def test_card_sections_turns_section_and_header_into_cards(ui):
    imgui = ui

    def build():
        out = {"outside": widgets.in_group()}
        with widgets.card_sections():
            widgets.section("First")
            out["first"] = (widgets.in_group(), owner(imgui))
            imgui.text("body")
            widgets.section("Second")
            out["second"] = (widgets.in_group(), owner(imgui), len(widgets._GROUPS))
            opened = widgets.header("Fold", default_open=True, persist_key=None)
            out["fold"] = (opened, owner(imgui), len(widgets._GROUPS))
            widgets.end_section()
            out["ended"] = widgets.in_group()
        out["after"] = widgets.in_group()
        return out

    out = draw(ui, build)
    assert out["outside"] is False
    assert out["first"][0] and out["second"][0]
    assert out["first"][1] != out["second"][1], "each section is its own child window"
    assert out["second"][2] == 1, "the first card was ended before the second opened"
    assert out["fold"][0] is True and out["fold"][2] == 1
    assert out["fold"][1] not in (out["first"][1], out["second"][1])
    assert out["ended"] is False and out["after"] is False


def test_section_cards_are_real_groups_with_the_card_inset(ui):
    imgui = ui

    def build():
        with widgets.card_sections():
            widgets.section("Next step")
            imgui.button("##x", (imgui.get_content_region_avail().x, sp(30)))
            item = rect(imgui)
            win, size = imgui.get_window_pos(), imgui.get_window_size()
            return item, (win.x, win.y, win.x + size.x, win.y + size.y)

    item, box = draw(imgui, build)
    assert item[0] - box[0] == approx(sp(12))
    assert box[2] - item[2] == approx(sp(12))
    assert item[1] - box[1] == approx(sp(12 + 26 + 8))


def test_card_sections_leaves_a_section_in_a_nested_child_alone(ui):
    imgui = ui

    def build():
        with widgets.card_sections():
            widgets.section("Outer")
            card = owner(imgui)
            frames = len(widgets._GROUPS)
            imgui.begin_child("##nested", (0.0, 120.0))
            widgets.section("Inside the child")
            widgets.end_section()
            nested = (len(widgets._GROUPS), owner(imgui) != card)
            imgui.end_child()
            return card, frames, nested, owner(imgui), len(widgets._GROUPS)

    card, frames, nested, back, still = draw(imgui, build)
    assert nested == (frames, True), "no card was opened or closed from the nested child"
    assert back == card and still == 1, "and the outer card is still the one open"


def test_a_section_inside_an_open_header_fold_is_a_sub_label_not_a_card(ui):
    imgui = ui

    def build():
        with widgets.card_sections():
            opened = widgets.header("Open in...", default_open=True, persist_key=None)
            fold = owner(imgui)
            widgets.section("Take it somewhere")
            return opened, fold, owner(imgui), len(widgets._GROUPS), imgui.get_item_rect_min().x

    opened, fold, after, depth, text_x = draw(imgui, build)
    assert opened is True
    assert (after, depth) == (fold, 1), "still inside the fold's card, and no card in a card"


def test_a_section_after_a_closed_fold_starts_a_sibling_card(ui):
    imgui = ui

    def build():
        with widgets.card_sections():
            opened = widgets.header("Open in...", default_open=False, persist_key=None)
            fold = owner(imgui)
            widgets.section("Next")
            return opened, fold, owner(imgui), len(widgets._GROUPS)

    opened, fold, after, depth = draw(imgui, build)
    assert opened is False
    assert after != fold and depth == 1


def test_end_section_lets_a_section_follow_an_open_fold_as_a_card(ui):
    imgui = ui

    def build():
        with widgets.card_sections():
            widgets.header("Fold", default_open=True, persist_key=None)
            fold = owner(imgui)
            widgets.end_section()
            widgets.section("Sibling")
            return fold, owner(imgui)

    fold, sibling = draw(imgui, build)
    assert sibling != fold


def test_two_sections_with_one_title_get_two_windows(ui):
    imgui = ui

    def build():
        with widgets.card_sections():
            widgets.section("Same")
            first = owner(imgui)
            widgets.section("Same")
            return first, owner(imgui)

    first, second = draw(imgui, build)
    assert first != second


def test_card_sections_with_a_stack_spaces_its_cards_8_apart(ui):
    imgui = ui

    def build():
        with widgets.card_sections(), widgets.group_stack():
            widgets.section("A")
            imgui.text("a")
            end_a = imgui.get_window_pos().y + imgui.get_window_size().y
            widgets.section("B")
            return imgui.get_window_pos().y - end_a

    assert draw(imgui, build) == approx(sp(8))


def test_section_header_and_end_section_are_unchanged_outside_the_scope(ui):
    imgui = ui

    def build():
        host = owner(imgui)
        widgets.section("Plain")
        opened = widgets.header("Plain fold", default_open=True)
        widgets.end_section()
        return host, owner(imgui), opened, widgets.in_group(), widgets.section_block_count()

    host, still, opened, grouped, blocks = draw(imgui, build, frames=1)
    assert (still, opened, grouped, blocks) == (host, True, False, 0)


def test_a_section_scope_opened_inside_card_sections_is_not_double_drawn(ui):
    """A pane that asks for blocks and cards in one window gets cards: the block
    scope would otherwise paint a fill with the card child's cursor."""
    imgui = ui

    def build():
        with widgets.section_blocks() as blocks, widgets.card_sections():
            widgets.section("Card")
            return len(blocks.blocks), widgets.in_group()

    assert draw(imgui, build, frames=1) == (0, True)


def test_a_sub_label_is_4_design_px_above_what_it_names(ui):
    imgui = ui

    def build():
        with widgets.group(None, id="sub"):
            top = imgui.get_cursor_screen_pos().y
            widgets.sub_label("Engine")
            imgui.button("##after", (-1, sp(30)))
            return rect(imgui)[1] - top, imgui.calc_text_size("ENGINE").y

    gap, text_h = draw(imgui, build)
    assert gap == approx(text_h + sp(4))


def test_a_sub_label_that_ends_its_card_does_not_trip_imgui(ui):
    """It must not hand-place the cursor: imgui refuses a window whose cursor was
    set past its last item (a RuntimeError in this harness), and a card that ends
    on its sub-label is where that would show. The card still keeps its 12 px."""
    imgui = ui

    def build():
        with widgets.group(None, id="sub-last"):
            widgets.sub_label("Engine")
            win, size = imgui.get_window_pos(), imgui.get_window_size()
            return win.y + size.y - rect(imgui)[3]

    assert draw(imgui, build) == approx(sp(12))
