"""Saved workspace arrangements: the data half, with no imgui in it.

What a layout is, and deliberately is not. It captures **arrangement,
visibility and shares** -- which panes are in which column, in what order, and
how their vertical splits are divided. It does **not** capture a width: each
side column is a fixed share of the room (``layout.proportions``), the same in
every mode, and the per-workspace widths a v2 blob used to carry were never read
once that landed. A file that still has a ``widths`` key loads (the key is
ignored, :meth:`Arrangement.from_json`) and the next save drops it.  The rail,
UI scale and theme remain application preferences: a workspace switch that
collapsed navigation would still be a surprise.

**Top-level settings keys**, ``workspace_layouts`` and ``active_layout``,
rather than living inside
``settings["layout"]`` -- ``Settings.set`` replaces a whole dict and
``test_a_save_writes_no_width_key_at_all`` asserts that dict's exact key set,
so a fifth key in it would be a test failure and a silently dropped preference
the first time anything else saved.

The blob is **sparse and per workspace**; absent means "the built-in", so the
migration story is empty and ``settings._migrate`` gains nothing.
``settings.VERSION`` stays 1 -- a bump discards the whole file -- and each blob
carries its own ``v`` instead: an unknown higher one is **kept verbatim and
listed greyed** rather than coerced, because the alternative is a newer build's
layout quietly rewritten by an older one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .settings import as_dict, as_list
from .tokens import clamp_share, finite_float

#: This build's layout-blob version. Independent of ``settings.VERSION``.
VERSION = 2

#: The two layouts every profile has. ``default`` is the built-in arrangement
#: and ``mirrored`` is Aseprite's own second built-in -- the columns swapped --
#: which is one line of code and the proof the machinery works end to end.
BUILT_IN = ("default", "mirrored")

LAYOUTS_KEY = "workspace_layouts"
ACTIVE_KEY = "active_layout"


@dataclass
class Arrangement:
    """One workspace's saved shape, inside one named layout."""

    columns: dict[str, list[str]] = field(default_factory=dict)
    hidden: list[str] = field(default_factory=list)
    shares: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "columns": {key: list(value) for key, value in sorted(self.columns.items())},
            "hidden": sorted(self.hidden),
            "shares": {key: round(float(value), 3) for key, value in sorted(self.shares.items())},
        }

    @classmethod
    def from_json(cls, raw: Any) -> Arrangement:
        if not isinstance(raw, dict):
            return cls()
        columns = {}
        for key, value in as_dict(raw.get("columns")).items():
            if isinstance(value, list):
                columns[str(key)] = [str(item) for item in value]
        hidden = [item for item in as_list(raw.get("hidden")) if isinstance(item, str)]
        # ``widths`` is deliberately not read: a file a build with per-workspace
        # widths wrote still carries the key, and ignoring it is what lets that
        # file load -- the next explicit save simply omits it.
        shares: dict[str, float] = {}
        for key, value in as_dict(raw.get("shares")).items():
            # A stored NaN reads as absent (shell-20, the 2026-10-03 audit):
            # it would pass ``clamp_share`` unchanged and reach a pane height.
            number = finite_float(value)
            if number is not None:
                shares[str(key)] = number
        return cls(columns=columns, hidden=hidden, shares=shares)


@dataclass
class Layout:
    """A named layout: one arrangement per workspace that has been touched."""

    name: str
    v: int = VERSION
    workspaces: dict[str, Arrangement] = field(default_factory=dict)
    #: A blob this build does not understand, kept exactly as it was found.
    opaque: Any = None

    @property
    def readable(self) -> bool:
        """Whether this build may apply it. See the module docstring."""

        return self.opaque is None

    def to_json(self) -> Any:
        if self.opaque is not None:
            return self.opaque
        return {
            "v": VERSION,
            "workspaces": {key: value.to_json() for key, value in sorted(self.workspaces.items())},
        }

    @classmethod
    def from_json(cls, name: str, raw: Any) -> Layout:
        if not isinstance(raw, dict):
            return cls(name=name)
        try:
            # The 2026-09-26 audit, finding shell-documents-02: a stored
            # ``"v": Infinity`` parses to ``float("inf")``, and
            # ``int(float("inf"))`` raises ``OverflowError`` -- a value
            # ``TypeError``/``ValueError`` alone did not catch, so a
            # malformed layout blob crashed every launch that read it.
            version = int(raw.get("v", VERSION))
        except (TypeError, ValueError, OverflowError):
            version = VERSION
        if version > VERSION:
            # Kept verbatim rather than coerced: an older build rewriting a
            # newer layout is the one way this feature can destroy something.
            return cls(name=name, v=version, opaque=raw)
        # The 2026-09-26 audit, finding shell-documents-02: ``or {}`` is not a
        # type guard (``as_dict``'s own docstring already says so) -- a stored
        # ``"workspaces": [...]`` is truthy all the way to ``.items()``'s
        # ``AttributeError``, which crashed App construction on any launch
        # that read it. ``as_dict`` falls back to ``{}`` instead.
        spaces = {
            str(key): Arrangement.from_json(value)
            for key, value in as_dict(raw.get("workspaces")).items()
        }
        # Versions one and two are both readable.  A v1 object stays sparse in
        # memory and is upgraded only when an explicit edit calls ``save``;
        # constructing the library never rewrites settings.
        return cls(name=name, v=version, workspaces=spaces)


class Library:
    """Every saved layout, and which one is in use.

    Reads on construction and writes only when something changes -- **never on
    launch**: a start-up that rewrites the settings file is one that can lose
    it to a crash it had no reason to be exposed to.
    """

    def __init__(self, settings: Any) -> None:
        self._settings = settings
        raw = as_dict(settings.get(LAYOUTS_KEY))
        self.layouts: dict[str, Layout] = {}
        for name, blob in raw.items():
            self.layouts[str(name)] = Layout.from_json(str(name), blob)
        for name in BUILT_IN:
            self.layouts.setdefault(name, Layout(name=name))
        wanted = str(settings.get(ACTIVE_KEY) or BUILT_IN[0])
        self.active = wanted if wanted in self.layouts else BUILT_IN[0]
        legacy = as_dict(settings.get("layout"))
        self._share_seed = 0.55
        seed = finite_float(legacy.get("settings_share", 0.55))
        if seed is not None:
            self._share_seed = seed
        self._share_seeds: dict[str, float] = {}
        for key, value in as_dict(legacy.get("settings_shares")).items():
            number = finite_float(value)
            if number is not None:
                self._share_seeds[str(key)] = number

    # -- reading ------------------------------------------------------------

    def current(self) -> Layout:
        return self.layouts[self.active]

    def arrangement(self, workspace: str) -> Arrangement:
        """This layout's shape for one workspace, or an empty one."""

        return self.current().workspaces.get(workspace) or Arrangement()

    def order(self, workspace: str, column: str, builtin: list[str]) -> list[str]:
        """The slots of one column, in the order this layout wants them.

        Reconciled against the built-in list every time it is read and **never
        written back**: a retired pane disappears, a new one lands after its
        last already-placed predecessor rather than at the bottom, and a launch
        that changes nothing leaves the file alone.
        """

        from .layout_skeleton import reconcile

        if not self.current().readable:
            return list(builtin)
        stored = self.arrangement(workspace).columns.get(column)
        return list(builtin) if stored is None else reconcile(builtin, stored)

    def hidden(self, workspace: str) -> set[str]:
        if not self.current().readable:
            return set()
        return set(self.arrangement(workspace).hidden)

    def share(self, workspace: str, key: str, default: float | None = None) -> float:
        """A workspace-local vertical split, with v1 global values as seeds."""

        if self.current().readable:
            value = self.arrangement(workspace).shares.get(key)
            if value is not None:
                return clamp_share(value)
        if key in self._share_seeds:
            return clamp_share(self._share_seeds[key])
        return clamp_share(self._share_seed if default is None else default)

    # -- writing ------------------------------------------------------------

    def set_active(self, name: str) -> None:
        if name not in self.layouts:
            return
        self.active = name
        self._settings.set(ACTIVE_KEY, name)

    def record(
        self,
        workspace: str,
        columns: dict[str, list[str]],
        hidden: set[str],
        *,
        shares: dict[str, float] | None = None,
    ) -> None:
        """Store an arrangement for one workspace of the active layout."""

        layout = self.current()
        if not layout.readable:
            return
        previous = layout.workspaces.get(workspace) or Arrangement()
        layout.workspaces[workspace] = Arrangement(
            columns={key: list(value) for key, value in columns.items()},
            hidden=sorted(hidden),
            shares=dict(previous.shares if shares is None else shares),
        )
        self.save()

    def hidden_everywhere(self) -> list[tuple[str, str]]:
        """Every ``(workspace, slot id)`` the active layout hides, sorted.

        A hidden slot is dropped before it is drawn, so it has no rect for the
        layout editor to hang a badge on; this is the list Settings draws its
        way back from (shell-11), and the editor's own "Show" chips
        (``layout_edit.EditState.chips``) read the per-workspace half of it.
        """

        if not self.current().readable:
            return []
        return sorted(
            (workspace, slot)
            for workspace, arrangement in self.current().workspaces.items()
            for slot in arrangement.hidden
        )

    def unhide(self, workspace: str, slot_id: str) -> None:
        """Bring one hidden pane back, leaving the rest of the arrangement."""

        arrangement = self.arrangement(workspace)
        if slot_id not in arrangement.hidden:
            return
        self.record(workspace, arrangement.columns, set(arrangement.hidden) - {slot_id})

    def reset_sizes(self) -> None:
        """Drop every stored split, keeping the pane arrangement.

        Not :meth:`reset`: that also discards ``columns`` and ``hidden``, which
        are *which panes are where*, and "Reset pane sizes" does not claim to
        touch them.
        """

        layout = self.current()
        if not layout.readable:
            return
        for arrangement in layout.workspaces.values():
            arrangement.shares.clear()
        self.save()

    def set_share(self, workspace: str, key: str, value: float) -> None:
        layout = self.current()
        if not layout.readable:
            return
        arrangement = layout.workspaces.setdefault(workspace, Arrangement())
        arrangement.shares[str(key)] = clamp_share(value)
        self.save()

    def duplicate(self, name: str, into: str) -> bool:
        if name not in self.layouts or not into or into in self.layouts:
            return False
        source = self.layouts[name]
        self.layouts[into] = Layout.from_json(into, source.to_json())
        self.save()
        return True

    def rename(self, name: str, into: str) -> bool:
        """Rename a layout. **A built-in cannot be renamed**, because its name
        is what the reset commands and this docstring refer to."""

        if name in BUILT_IN or name not in self.layouts or not into or into in self.layouts:
            return False
        layout = self.layouts.pop(name)
        layout.name = into
        self.layouts[into] = layout
        if self.active == name:
            self.active = into
            self._settings.set(ACTIVE_KEY, into)
        self.save()
        return True

    def delete(self, name: str) -> bool:
        """Delete a layout. A built-in is **reset** instead of removed: there
        is no reachable state in which a pane cannot be got back."""

        if name not in self.layouts:
            return False
        if name in BUILT_IN:
            # The 2026-09-23 audit (shell-04): unreadable (future-version)
            # built-ins were coerced to a fresh Layout unconditionally, the
            # same rewrite-by-an-older-build bug as ``reset`` below. The
            # button that reaches this branch is disabled while the active
            # layout is built-in, so this path is presently unreachable from
            # the UI -- but the guard belongs on the data method, not on the
            # one caller that happens to exist today.
            if not self.layouts[name].readable:
                return False
            self.layouts[name] = Layout(name=name)
        else:
            del self.layouts[name]
            if self.active == name:
                self.set_active(BUILT_IN[0])
        self.save()
        return True

    def reset(self, name: str = "") -> None:
        """Put a layout back to the built-in arrangement.

        Early-returns on an unreadable (future-version) layout rather than
        replacing it: the 2026-09-23 audit (shell-01) found this method had
        no ``readable`` check at all, so pressing the always-enabled Reset
        button on a newer build's layout silently rewrote it with a fresh
        one -- the exact "older build rewrites a newer layout" case the
        module docstring's "kept verbatim" rule exists to prevent, just
        reached from a different button than the ones already guarded.
        """

        wanted = name or self.active
        if wanted not in self.layouts or not self.layouts[wanted].readable:
            return
        self.layouts[wanted] = Layout(name=wanted)
        self.save()

    def save(self) -> None:
        # An explicit edit is the migration boundary.  Readable v1 layouts are
        # emitted as v2; opaque future versions remain byte-for-byte intact.
        for layout in self.layouts.values():
            if layout.readable:
                layout.v = VERSION
        self._settings.set(
            LAYOUTS_KEY,
            {name: layout.to_json() for name, layout in sorted(self.layouts.items())},
        )


def mirrored(columns: dict[str, list[str]]) -> dict[str, list[str]]:
    """Aseprite's second built-in: the two sidebars swapped.

    One function, and it is the proof the machinery works end to end -- a
    layout that is not the default, expressible in the same data, with nothing
    special about it anywhere else.
    """

    out = dict(columns)
    left, right = out.get("left"), out.get("right")
    if left is not None and right is not None:
        out["left"], out["right"] = list(right), list(left)
    return out
