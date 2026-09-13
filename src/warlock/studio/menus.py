"""The application menu model and its single host-level renderer.

Rows are adapters over the command palette and workspace operation registries;
the menu owns placement, never a second implementation of an action.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .tour import scripts as tour_scripts


@dataclass(frozen=True)
class MenuSpec:
    """One command wherever it is presented in the menu tree."""

    identity: str
    path: tuple[str, ...]
    order: int
    label: str
    enabled: bool
    checked: bool
    shortcut: str
    disabled_reason: str
    callback: Callable[[], None]
    separator_before: bool = False


ROOTS = ("File", "Edit", "View", "Workspace", "Window", "Help")

_COMMAND_PATHS: dict[str, tuple[str, ...]] = {
    "new-drawing": ("File",),
    "new-clay": ("File",),
    "new-map": ("File",),
    "new-atlas": ("File",),
    # The fifth, and missing for the same reason the palette's ``new-map`` was:
    # a New that is not in the File menu reads as the mode not having one, and
    # falls into the contextual per-mode menu instead.
    "new-song": ("File",),
    "save": ("File",),
    "save-as": ("File",),
    "export": ("File",),
    "quit": ("File",),
    "undo": ("Edit",),
    "redo": ("Edit",),
    "delete": ("Edit",),
    # The 2026-09-07 audit found ``reroll`` missing here (shell-06): with no
    # entry it fell into the mode-specific contextual menu instead, so it was
    # absent from Home and Library -- where selecting an asset and rerolling
    # it actually happens -- and turned up in nine unrelated workspace modes.
    # Chapter 38 states the menu bar and the palette carry the same commands.
    "reroll": ("Edit",),
    "frame": ("View",),
    "wireframe": ("View",),
    "turntable": ("View",),
    "clear-viewport": ("View",),
    "fps": ("View",),
    "workspace-layout": ("Window",),
    "show-trash": ("Window",),
    "empty-trash": ("Window",),
    "manual": ("Help",),
    "shortcuts": ("Help",),
    "open-log": ("Help",),
}

# The tours, which ``palette`` mints one command per and which had no path at
# all -- so ``roots()`` dropped every one of them and the only way to a tour was
# Ctrl+K or Home's offer card. Derived from ``TOURS`` rather than written out,
# so a fifth tour is not a second table to keep in agreement.
_COMMAND_PATHS.update({f"tour:{one.key}": ("Help",) for one in tour_scripts.TOURS})


def _checked(ctx: Any, key: str) -> bool:
    if key.startswith("go:"):
        return ctx.state.mode == key.partition(":")[2]
    return {
        "wireframe": bool(getattr(ctx.state, "wireframe", False)),
        "turntable": bool(getattr(ctx.state, "turntable", False)),
        "fps": bool(getattr(ctx.state, "show_fps", False)),
        "show-trash": bool(getattr(getattr(ctx.state, "filters", None), "trash", False)),
    }.get(key, False)


def _command_specs(
    ctx: Any, commands: list[Any], *, evaluate: bool = True
) -> list[MenuSpec]:
    from . import modes

    out: list[MenuSpec] = []
    for index, command in enumerate(commands):
        path = ("Workspace",) if command.key.startswith("go:") else _COMMAND_PATHS.get(command.key)
        if path is None:
            # Mode-specific actions form a contextual menu instead of making
            # File/Edit into miscellaneous command dumps.
            if (
                command.group in ("Actions", "Viewport")
                and ctx.state.mode not in ("home", "settings", "library")
                and ctx.state.mode != "inker"
            ):
                label = next(
                    (
                        name
                        for key, name, _icon, _purpose in modes.MODES
                        if key == ctx.state.mode
                    ),
                    "Actions",
                )
                path = (label,)
            else:
                continue
        out.append(
            MenuSpec(
                identity=f"command:{command.key}",
                path=path,
                order=index,
                label=command.label,
                enabled=bool(command.enabled(ctx)) if evaluate else True,
                checked=_checked(ctx, command.key) if evaluate else False,
                shortcut=command.hint,
                disabled_reason=command.why,
                callback=lambda command=command: command.run(ctx),
            )
        )
    return out


#: Two of Inker's five export doors were already ``inker_ops`` File rows, under
#: their own older labels ("Export sprite sheet...") and their own refusal. They
#: are suppressed there and drawn from :data:`inker_export.DOORS` instead, so
#: the File menu spells each door exactly once and in the same words the
#: bridge's button uses. The ops themselves are untouched -- the palette and the
#: shortcut table still reach them by name.
SHADOWED_BY_DOORS = frozenset({"export_sheet", "export_gif"})


def _inker_specs(ctx: Any, *, evaluate: bool = True) -> list[MenuSpec]:
    if ctx.state.mode != "inker":
        return []
    from . import inker_mode, inker_ops
    from .panes import inker_menu

    state = inker_mode.ensure(ctx)
    tab = state.active
    out = []
    shortcuts = inker_ops.shortcuts_for(state.shortcut_overrides)
    for index, op in enumerate(inker_ops.OPS):
        if not op.menu or op.name in SHADOWED_BY_DOORS:
            continue
        out.append(
            MenuSpec(
                identity=f"inker:{op.name}",
                path=(op.menu,),
                order=index,
                label=op.label,
                enabled=bool(op.enabled(state, tab)) if evaluate else True,
                checked=(
                    bool(op.checked(state, tab)) if evaluate and op.checked else False
                ),
                shortcut=shortcuts.get(op.name, ""),
                disabled_reason=inker_ops.reason_for(op, state, tab) if evaluate else "",
                callback=lambda op=op: inker_menu.activate(ctx, op),
                separator_before=bool(op.separator_before),
            )
        )
    return out


def _inker_export_specs(
    ctx: Any, commands: list[Any], *, evaluate: bool = True
) -> list[MenuSpec]:
    """Inker's five exports as File rows.

    They were toolbar buttons on the timeline's second row and nowhere else --
    a row that overflowed at 1280x800, so three of the five were only reachable
    through a ``...`` menu inside a strip. There is no export *root*: the File
    menu carries one ``export`` command and these five land beside it, ordered
    after everything the command table puts there.

    Label, enabled state and refusal all come from
    :func:`inker_export.door_state`, the same call Inker's bridge makes, so the
    menu row and the button cannot disagree about whether a door is open or
    about why it is not.

    Takes ``commands`` rather than calling ``palette.commands(ctx)`` itself --
    the 2026-09-07 audit's shell-11: with both this and ``_command_specs``
    each rebuilding the ~30-command list, and ``specs`` itself called twice a
    frame (once for the bar's shape, once for whichever menu is open), the
    same list was rebuilt up to four times a frame. ``specs`` builds it once
    and hands it down.
    """
    if ctx.state.mode != "inker":
        return []
    from . import inker_export, inker_mode

    tab = inker_mode.ensure(ctx).active
    # The same order as the File menu's own ``export`` row, so the five land
    # beside it rather than under Quit: ``sorted`` is stable and these rows are
    # appended after the command rows, so a tie puts them immediately after it.
    base = next((i for i, one in enumerate(commands) if one.key == "export"), 0)
    out = []
    for index, door in enumerate(inker_export.doors()):
        enabled, reason = (
            inker_export.door_state(door, tab) if evaluate else (True, "")
        )
        out.append(
            MenuSpec(
                identity=f"inker-export:{door.key}",
                path=("File",),
                order=base,
                label=door.label,
                enabled=enabled,
                checked=False,
                shortcut="",
                disabled_reason=reason,
                callback=lambda door=door: inker_export.open_door(
                    ctx, inker_mode.ensure(ctx).active, door.key
                ),
                separator_before=index == 0,
            )
        )
    return out


def specs(ctx: Any, layout: Any = None, *, evaluate: bool = True) -> list[MenuSpec]:
    """The current menu tree as data, rebuilt so state never goes stale.

    ``evaluate=False`` skips every ``enabled``/``checked``/reason call and
    reports the shape alone. It exists for :func:`draw`, which needs the *root
    names* on every frame and the row states only inside a menu that is open:
    evaluating them unconditionally ran every command's gate -- including a
    scan of the whole job cache for "Empty the trash" -- sixty times a second
    at a bar nobody had clicked. Nothing is memoised, so an open menu is still
    rebuilt from live state every frame, which is the freshness this docstring
    has always promised.
    """

    from . import palette

    commands = palette.commands(ctx)
    rows = (
        _command_specs(ctx, commands, evaluate=evaluate)
        + _inker_specs(ctx, evaluate=evaluate)
        + _inker_export_specs(ctx, commands, evaluate=evaluate)
    )
    if layout is not None:
        rows.append(
            MenuSpec(
                identity="window:navigation-labels",
                path=("Window",),
                order=10_000,
                label="Navigation labels",
                enabled=True,
                checked=layout.rail == "labels",
                shortcut="",
                disabled_reason="",
                callback=lambda: layout.set_rail("icons" if layout.rail == "labels" else "labels"),
            )
        )
    return rows


def roots(rows: list[MenuSpec]) -> list[str]:
    """Stable root order, with contextual workspace menus before View."""

    present = {row.path[0] for row in rows if row.path}
    contextual: list[str] = []
    for row in rows:
        root = row.path[0] if row.path else ""
        if root and root not in ROOTS and root not in contextual:
            contextual.append(root)
    ordered = ["File", "Edit", *contextual, "View", "Workspace", "Window", "Help"]
    return [name for name in ordered if name in present or name in ROOTS]


#: Reserved room for a "Familiar" entry, drawn as an ordinary menu right of
#: the workspace roots (T0: one disabled row, "Not installed", no wiring
#: behind it yet). Unlike the status group below, this is never dropped for
#: space -- it is drawn before the status group's available width is
#: measured, the same way any other root would be.
FAMILIAR_LABEL = "✦ Familiar"

#: Status keys the right-aligned menu-bar group drops, lowest priority first,
#: when the roots and the Familiar menu leave it no room. ``health`` (and the
#: leading ``workspace`` row) are deliberately absent from this tuple: they
#: are never dropped, regardless of space.
STATUS_DROP_ORDER: tuple[str, ...] = ("resources", "zoom", "tool", "document", "queue")


def status_rows(ctx: Any) -> list[Any]:
    """``status_bar.items(ctx)`` plus the resource meter, as one ordered list.

    ``menus.py`` owns none of this data -- ``status_bar`` stays the single
    account of what a status item is and says; this just folds its two
    sources (the item list and the separately-anchored meter) into the one
    sequence the menu bar's right-aligned group draws.
    """

    from . import status_bar

    rows = list(status_bar.items(ctx))
    meter = status_bar.resource_item(ctx)
    if meter is not None:
        rows.append(meter)
    return rows


def fit_status_rows(
    rows: list[Any], available: float, measure: Callable[[Any], float]
) -> list[Any]:
    """*rows* trimmed to fit *available* width, dropping the lowest-priority
    key in :data:`STATUS_DROP_ORDER` first, one key at a time, until what is
    left fits (or the order is exhausted). Any row whose key is not in that
    tuple -- ``health`` and ``workspace`` today -- is never removed here.
    """

    kept = list(rows)
    for key in STATUS_DROP_ORDER:
        if sum(measure(row) for row in kept) <= available:
            break
        kept = [row for row in kept if row.key != key]
    return kept


def _draw_status_group(ctx: Any) -> None:
    """The right-aligned, non-clickable status readouts.

    Drawn last in the menu bar, so ``imgui.get_content_region_avail()`` at
    the top of this function already reflects every root and the Familiar
    menu having been laid out -- which is how "measure the menu labels'
    width first" is honoured without a second, hand-rolled measurement of
    them: imgui's own left-to-right menu-bar layout already did it.
    """

    from imgui_bundle import imgui

    from . import fonts, theme, tokens

    avail = imgui.get_content_region_avail().x
    rows = status_rows(ctx)
    if not rows or avail <= 0:
        return
    pad_x = tokens.sp(tokens.SP_2)
    with fonts.small(imgui):

        def _text(index: int, item: Any) -> str:
            return item.text if index == 0 else f"  |  {item.text}"

        # Measured with the separator every row but the first is drawn with,
        # so the fit never passes a group that then overflows into the menus.
        def measure(item: Any) -> float:
            return imgui.calc_text_size(f"  |  {item.text}").x

        fitted = fit_status_rows(rows, max(avail - pad_x, 0.0), measure)
        if not fitted:
            return
        total = sum(imgui.calc_text_size(_text(i, item)).x for i, item in enumerate(fitted))
        x = max(imgui.get_cursor_pos_x(), imgui.get_window_width() - total - pad_x)
        imgui.same_line(x)
        for index, item in enumerate(fitted):
            text = _text(index, item)
            if index:
                imgui.same_line(0.0, 0.0)
            if item.warning:
                imgui.text_colored(imgui.ImVec4(*theme.rgba(theme.WARN)), text)
                if imgui.is_item_hovered():
                    imgui.set_tooltip("Health checks need attention")
            else:
                imgui.text_colored(imgui.ImVec4(*theme.rgba(theme.MUTED)), text)


def draw(ctx: Any, layout: Any = None) -> None:
    """Render the 26 dp global menu bar in the host window."""

    from imgui_bundle import imgui

    from . import controls, tokens

    shape = specs(ctx, layout, evaluate=False)
    live: list[MenuSpec] | None = None
    imgui.push_style_var(imgui.StyleVar_.frame_padding.value, (tokens.sp(8), tokens.sp(6)))
    opened = imgui.begin_menu_bar()
    imgui.pop_style_var()
    if not opened:
        return
    try:
        for root in roots(shape):
            with controls.menu(root) as menu_open:
                if not menu_open:
                    continue
                if live is None:
                    live = specs(ctx, layout)
                for row in sorted(
                    (one for one in live if one.path and one.path[0] == root),
                    key=lambda one: one.order,
                ):
                    if row.separator_before:
                        controls.menu_separator()
                    hit = controls.menu_item(
                        f"{row.label}##menu/{row.identity}",
                        row.shortcut,
                        row.checked,
                        row.enabled,
                        reason=row.disabled_reason,
                    )
                    clicked = hit[0] if isinstance(hit, tuple) else hit
                    if clicked and row.enabled:
                        row.callback()
        # Reserved, never dropped: T0 of the Familiar programme wires no
        # model behind this yet, so its one row stays disabled.
        with controls.menu(FAMILIAR_LABEL) as familiar_open:
            if familiar_open:
                controls.menu_item(
                    "Not installed##menu/familiar-not-installed",
                    "",
                    False,
                    False,
                    reason="Familiar isn't installed yet.",
                )
        _draw_status_group(ctx)
    finally:
        imgui.end_menu_bar()
