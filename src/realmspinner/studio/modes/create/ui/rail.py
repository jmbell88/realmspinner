"""Create mode's breadcrumb: the pill that draws the stage rail, and the two
helpers that measure it without drawing anything.

Split out of ``widgets`` (P4 of the restructure, ``dev/RESTRUCTURE.md``):
this is imgui-bearing and Create-only, while ``create_stages`` -- the module
next to it that decides *which* stages exist and whether an asset has
reached them -- is deliberately headless and shared by nothing that draws.
Putting the rail's own drawing into ``create_stages`` would have dragged
imgui into a module whose own docstring promises it imports nothing from it,
so the rail gets its own file instead of folding into that one.

``create_brief`` measures the rail with :func:`stage_rail_width` before
drawing anything else on its row (the rail is the row's leftmost element and
shares it with the type combo, the prompt, the count and Generate); ``main``'s
``App._stage_rail`` builds the live ``items``/``done`` from a job and calls
:func:`stage_rail` to actually draw it.
"""

from __future__ import annotations

from imgui_bundle import imgui

from .... import fonts, icons, motion, theme, tokens
from ....tokens import sp


def _rail_keys(done: str | frozenset[str] | set[str] | None) -> frozenset[str]:
    """``done`` folded to a set, the way :func:`stage_rail`'s own docstring
    explains: a lone string is one caller's "just this one" and is not given
    its own code path."""
    if done is None:
        return frozenset()
    if isinstance(done, str):
        return frozenset({done})
    return frozenset(done)


def _rail_fit(
    items: list[tuple[str, str, str, str | None]],
    current: str,
    done: str | frozenset[str] | set[str] | None,
    max_width: float | None,
) -> tuple[list[str], list[float], dict[str, str], frozenset[str]]:
    """The measure-and-degrade ladder shared by :func:`stage_rail` (which
    draws it) and :func:`stage_rail_width` (which only wants the number).

    Factored out rather than left inline, and rather than reimplemented at the
    second call site: ``create_brief._row_widths`` needs to know how wide the
    rail wants to be *before* the rail is drawn -- it is the row's leftmost
    element now, sharing one line with the type combo, the prompt, the count
    and Generate -- and a second copy of ``faces``/``measure`` is exactly how
    the two numbers would drift the day either rung changes. Must run inside
    ``fonts.label(imgui)``, as both callers already do: ``calc_text_size``
    reads the currently pushed font.

    -> ``(shown, widths, titles, done_keys)``, matching what :func:`stage_rail`
    used to compute inline.
    """
    done_keys = _rail_keys(done)
    pad_x = sp(12)

    # What each segment actually reads as, before it is measured: a check is
    # part of the width, which is why it is a rung of the ladder.
    def faces(compact: bool, ticks: bool) -> list[str]:
        out = []
        for key, label, icon, _reason in items:
            if compact:
                out.append(icon)
            elif ticks and key in done_keys and key != current:
                out.append(f"{icons.CHECK} {label}")
            else:
                out.append(label)
        return out

    def measure(labels: list[str]) -> list[float]:
        return [imgui.calc_text_size(text).x + pad_x * 2 for text in labels]

    titles: dict[str, str] = {}
    shown = faces(False, True)
    widths = measure(shown)
    for compact, ticks in ((False, False), (True, False)):
        if max_width is None or sum(widths) <= max_width:
            break
        if compact:
            titles = {key: label for key, label, _icon, _reason in items}
        shown = faces(compact, ticks)
        widths = measure(shown)
    return shown, widths, titles, done_keys


def stage_rail_width(
    items: list[tuple[str, str, str, str | None]],
    current: str = "",
    *,
    done: str | frozenset[str] | set[str] | None = None,
    max_width: float | None = None,
) -> float:
    """What :func:`stage_rail` would measure for ``items`` at this budget --
    without drawing anything.

    ``create_brief._row_widths`` calls this to learn how much of the row the
    rail wants before deciding how much of the row everyone else keeps: the
    rail used to own the whole width of its own bar and could size itself with
    no help from a caller, but sharing a row means something else now has to
    ask. Sharing :func:`_rail_fit` with :func:`stage_rail` rather than
    guessing a constant (304 was one, and wrong the moment a label changed) is
    what keeps this answer and the one actually drawn from disagreeing.

    Needs an active imgui context the way :func:`stage_rail` does --
    ``calc_text_size`` inside ``fonts.label`` -- and nothing else: no GL, no
    renderer, no draw list.
    """
    with fonts.label(imgui):
        _shown, widths, _titles, _done = _rail_fit(items, current, done, max_width)
    return sum(widths)


def stage_rail(
    rail_id: str,
    items: list[tuple[str, str, str, str | None]],
    current: str,
    *,
    done: str | frozenset[str] | set[str] | None = None,
    optional: dict[str, str] | None = None,
    max_width: float | None = None,
    row_height: float | None = None,
) -> str:
    """The Create mode's breadcrumb: where this asset is, and what is left.

    ``items`` is ``(key, label, icon, blocked_reason)`` in pipeline order;
    ``done`` names the segments the asset has actually landed on. -> the key
    the user picked, or ``current``.

    ``done`` is a **set of keys**, not the single furthest one (2026-09-07
    Create review, item 3.5): a finished prop with no rig and no pose still
    has an export grid, and "furthest reached" could only tick Reference and
    Mesh for it, leaving Export dark on every asset the app will ever finish.
    A single key is still accepted, for a caller with nothing but "the one
    thing so far" to report -- it is folded into a one-member set below.

    ``optional`` names segments a finished asset may legitimately never earn
    (``create_stages.OPTIONAL_HINTS``, keyed the same way). A segment named
    there that is open (not blocked) and not in ``done`` gets a tooltip
    saying so -- the same courtesy a blocked segment's reason already is, for
    a segment that is merely skippable rather than unreachable.

    The segmented control's idiom deliberately -- one track, a sliding pill,
    the same padding and radius -- because this *is* a switch between panels
    and inventing a second visual language for it would say it was something
    else. What it adds is a third segment state. A segment is one of:

    * **done** -- the asset has been through it. A leading check, full text.
    * **current** -- the pill is under it, wherever it is on the track. A
      stage you have reached is not the stage you are looking at: standing on
      Reference with a finished mesh in hand is the normal way to reroll.
    * **blocked** -- dimmed, unclickable, and carrying its reason as a
      tooltip. **Not hidden.** "Rig" missing entirely is a feature the user
      concludes does not exist; "Rig -- Blender is not installed" is an
      answer. The reason is the service's own sentence (see
      ``create_stages.available``), so the tooltip and the refusal it predicts
      cannot drift into two paraphrases of one rule.

    Fitting is a **ladder of three**, each rung all-or-nothing in
    ``segmented_control``'s sense -- a rail that abbreviated only the segments
    that did not fit would change what it was saying as the window was
    dragged, and a clipped segment is an unreachable stage.

    1. Labels with their checks.
    2. Labels alone. The checks go *first*, before the words, because a check
       costs a glyph and a space on every completed segment and the words are
       what make the stages findable: five labelled segments and two ticks do
       not fit a 300 dp column at 150 %, and dropping straight to icons to keep
       two ticks trades the whole rail for them. Done-ness survives as
       full-strength text against a not-yet-reached segment's 0.55.
    3. Icons, each keeping its label in a tooltip.

    ``row_height``, added when the rail moved onto Create's command bar
    (2026-09-07): the rail's own content is one line, ``get_text_line_height()
    + 2 * sp(6)`` tall, roughly 25 dp at the default font -- short beside the
    40 dp prompt and
    Generate it now shares a line with. Handing a taller ``row_height`` does
    **not** stretch the track to fill it (a pill rail the height of a text
    field reads as broken, not tall); it centres the rail's own natural-height
    content inside the reserved band instead, the same way a short glyph
    button sits centred beside a full-height field. ``None`` (every other
    caller) keeps the old behaviour: the reserved height *is* the content
    height.
    """
    draw = imgui.get_window_draw_list()
    pad_y = sp(6)
    keys = [key for key, _label, _icon, _reason in items]
    order = {key: index for index, key in enumerate(keys)}
    optional = optional or {}
    with fonts.label(imgui):
        shown, widths, titles, done_keys = _rail_fit(items, current, done, max_width)
        height = imgui.get_text_line_height() + pad_y * 2
        band = height if row_height is None else max(row_height, height)
        origin = imgui.get_cursor_screen_pos()
        # Everything painted below reads off ``paint`` rather than ``origin``
        # for its Y: ``origin`` is what the cursor is restored to and what the
        # final ``dummy`` sizes against (``band``), so the *reserved* rect
        # matches the row's height while the *drawn* pill floats centred
        # inside it. X is untouched -- only the vertical centring is new.
        paint = (origin.x, origin.y + (band - height) * 0.5)
        offsets: list[float] = []
        cursor = 0.0
        for width in widths:
            offsets.append(cursor)
            cursor += width
        total = cursor
        draw.add_rect_filled(
            (paint[0], paint[1]),
            (paint[0] + total, paint[1] + height),
            imgui.get_color_u32(theme.rgba(theme.ELEV_1)),
            height * 0.5,
        )
        index = order.get(current, 0)
        # Sprung, ``segmented_control``'s way and for its reason: the target
        # can change while the pill is still travelling (a click on Mesh
        # followed by one on Export before it has arrived), and an eased
        # approach has no velocity to carry through the re-aim.
        x = motion.spring(f"{rail_id}/x", offsets[index], duration=tokens.DUR_BASE)
        w = motion.spring(f"{rail_id}/w", widths[index], duration=tokens.DUR_BASE)
        draw.add_rect_filled(
            (paint[0] + x + sp(2), paint[1] + sp(2)),
            (paint[0] + x + w - sp(2), paint[1] + height - sp(2)),
            imgui.get_color_u32(theme.rgba(theme.ELEV_2)),
            (height - sp(4)) * 0.5,
        )
        picked = current
        for (key, label, _icon, reason), text, width, offset in zip(
            items, shown, widths, offsets, strict=True
        ):
            imgui.set_cursor_screen_pos((paint[0] + offset, paint[1]))
            # A blocked segment is still an *item*: it has to be hoverable to
            # carry its tooltip, so it is clicked and the click is dropped,
            # rather than not drawn as a button at all.
            hit = imgui.invisible_button(f"{rail_id}/{key}", (width, height))
            hovered = imgui.is_item_hovered()
            if hit and reason is None:
                picked = key
            done_here = key in done_keys
            active = key == current
            if hovered:
                tip = titles.get(key)
                if reason is not None:
                    tip = f"{label} -- {reason}" if tip is None else f"{tip} -- {reason}"
                elif not done_here and not active and key in optional:
                    # The blocked reason's courtesy, extended to a segment
                    # that is open rather than unreachable: this asset may
                    # simply never take it (2026-09-07 Create review, item
                    # 3.5), and a rail that says nothing about that reads as
                    # if the segment were merely late rather than skippable.
                    hint = optional[key]
                    tip = f"{label} -- {hint}" if tip is None else f"{tip} -- {hint}"
                if tip is not None:
                    imgui.set_tooltip(tip)
            if reason is not None:
                alpha = tokens.DISABLED_ALPHA * 0.6
            elif active:
                alpha = 1.0
            elif done_here:
                alpha = 0.85 if not hovered else 1.0
            else:
                alpha = 0.85 if hovered else 0.55
            alpha = motion.value(f"{rail_id}/{key}/text", alpha, duration=tokens.DUR_FAST)
            size = imgui.calc_text_size(text)
            draw.add_text(
                (
                    paint[0] + offset + (width - size.x) * 0.5,
                    paint[1] + (height - size.y) * 0.5,
                ),
                imgui.get_color_u32(theme.rgba(theme.TEXT, alpha)),
                text,
            )
        imgui.set_cursor_screen_pos((origin.x, origin.y))
        imgui.dummy((total, band))
    return picked
