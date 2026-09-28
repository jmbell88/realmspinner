"""Regression tests for the 2026-09-26 audit's shell-documents batch (wave 7).

shell-documents-02: ``Layout.from_json`` called ``.items()`` on any truthy
non-dict ``workspaces`` value (``or {}`` is not a type guard), and its ``int()``
parse of ``"v"`` caught only ``TypeError``/``ValueError`` -- an
``OverflowError`` from ``int(float("inf"))`` (a stored ``"v": Infinity``) went
uncaught. Either one raised during ``App`` construction, so a malformed stored
layout blob failed every launch. ``layout.Layout.__init__`` had the identical
``or {}`` non-guard on ``settings_shares``.

shell-documents-03: dropping a dock pane onto the *other* sidebar column was
accepted and persisted by ``layout_edit._commit``, but ``reconcile`` (via
``layouts.Library.order``) filters a stored id against the *target* column's
own built-in set and drops anything that column never natively owned -- so the
moved pane vanished from where it landed and reappeared at its original
built-in position in the column it left on the very next load, while that
source column's saved order had already been silently overwritten to drop it.

shell-documents-05: ``JobsCache.adopt``'s own docstring claimed a read started
before ``reset_window`` could not land in the cache, but nothing enforced it.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from realmspinner.service import jobs as svc_jobs
from realmspinner.studio import layout as layout_mod
from realmspinner.studio import layout_edit, layouts
from realmspinner.studio import layout_skeleton as skeleton
from realmspinner.studio.jobs_cache import LIST_LIMIT, JobsCache

# --- shell-documents-02 -------------------------------------------------------


class _Settings:
    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self.data: dict[str, Any] = dict(data or {})

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value) -> None:
        self.data[key] = value


def test_layout_from_json_survives_a_non_dict_workspaces_value():
    """A stored ``"workspaces": [...]`` used to reach ``.items()`` and raise
    ``AttributeError`` -- ``or {}`` is truthy for any non-empty list."""

    layout = layouts.Layout.from_json("default", {"v": 2, "workspaces": ["not", "a", "dict"]})
    assert layout.workspaces == {}


def test_layout_from_json_survives_an_infinite_version():
    """A stored ``"v": Infinity`` parses (via ``json``) to ``float("inf")``,
    and ``int(float("inf"))`` raises ``OverflowError`` -- not
    ``TypeError``/``ValueError``, the only two caught before this fix."""

    layout = layouts.Layout.from_json("default", {"v": float("inf"), "workspaces": {}})
    assert layout.v == layouts.VERSION
    assert layout.opaque is None


def test_layout_settings_object_survives_a_non_dict_settings_shares():
    """``layout.Layout`` (the live per-frame proportions object, distinct from
    ``layouts.Layout``) had the identical ``(... or {}).items()`` non-guard on
    ``settings_shares``."""

    settings = _Settings({"layout": {"settings_shares": ["bad", "data"]}})
    live = layout_mod.Layout(settings)
    assert live.shares == {}


# --- shell-documents-03 -------------------------------------------------------


def _slot(name: str) -> skeleton.Slot:
    return skeleton.Slot(name, name, lambda ctx: None, sizing=skeleton.SHARE, share_key=name)


def test_a_cross_column_drop_is_refused_rather_than_silently_undone():
    """Dropping "left-b" onto the right column used to be accepted and
    persisted -- ``reconcile`` would have dropped it from "right" (never one
    of its built-in ids) and reinserted it into "left" at its original
    position on the next load, while "left"'s saved order was already
    rewritten to remove it. Refusing outright means neither column's saved
    arrangement is touched at all.
    """
    left_a, left_b = _slot("left-a"), _slot("left-b")
    right_a, right_b = _slot("A"), _slot("B")
    left = skeleton.Column("left", (left_a, left_b))
    right = skeleton.Column("right", (right_a, right_b))
    columns = {"left": left, "right": right}

    settings = _Settings()
    library = layouts.Library(settings)

    ctx = SimpleNamespace(state=SimpleNamespace(mode="probe"))
    app = SimpleNamespace(layouts=library)
    edit = layout_edit.ensure(ctx.state)
    edit.open = True
    edit.hidden = set()
    edit.dragging = "left-b"
    edit.dragging_column = "left"

    layout_mod.FRAME_PANES.clear()
    layout_mod.FRAME_PANES["left-a"] = (0.0, 0.0, 300.0, 50.0)
    layout_mod.FRAME_PANES["left-b"] = (0.0, 50.0, 300.0, 50.0)
    layout_mod.FRAME_PANES["A"] = (300.0, 0.0, 300.0, 50.0)
    layout_mod.FRAME_PANES["B"] = (300.0, 50.0, 300.0, 50.0)

    # x=310 lands inside the "right" column's rect -- a genuine cross-column
    # drop, not a reorder within "left".
    mouse = SimpleNamespace(x=310.0, y=5.0)
    layout_edit._commit(app, ctx, columns, edit, mouse)

    assert library.arrangement("probe").columns == {}


def test_a_reorder_within_the_dragged_from_column_still_commits():
    """The refusal must be specific to a *cross*-column drop -- an ordinary
    reorder within the source column still has to work."""
    left_a, left_b = _slot("left-a"), _slot("left-b")
    left = skeleton.Column("left", (left_a, left_b))
    columns = {"left": left}

    settings = _Settings()
    library = layouts.Library(settings)

    ctx = SimpleNamespace(state=SimpleNamespace(mode="probe"))
    app = SimpleNamespace(layouts=library)
    edit = layout_edit.ensure(ctx.state)
    edit.open = True
    edit.hidden = set()
    edit.dragging = "left-b"
    edit.dragging_column = "left"

    layout_mod.FRAME_PANES.clear()
    layout_mod.FRAME_PANES["left-a"] = (0.0, 0.0, 300.0, 50.0)
    layout_mod.FRAME_PANES["left-b"] = (0.0, 50.0, 300.0, 50.0)

    mouse = SimpleNamespace(x=10.0, y=5.0)
    layout_edit._commit(app, ctx, columns, edit, mouse)

    assert library.arrangement("probe").columns.get("left") == ["left-b", "left-a"]


# --- shell-documents-05 -------------------------------------------------------


def test_jobs_cache_adopt_rejects_a_read_stamped_before_reset_window(svc):
    """``adopt``'s own docstring claims a read started before the last
    ``reset_window`` does not land -- this pins the enforcement the docstring
    described but nothing implemented."""

    svc_jobs.create_job(svc, kind="text", prompt="a")
    cache = JobsCache(svc)
    cache.tick()
    # Widen the window first -- ``reset_window`` is a documented no-op (and
    # bumps nothing) when the window is already at ``LIST_LIMIT``.
    cache.load_more()

    # A read started while the window was still widened -- captured now,
    # adopted after the reset below.
    stale = cache.read(dict(cache._files))
    assert stale.get("window_generation") == cache._window_generation

    cache.reset_window()
    # The reset moved the window's generation on; the in-flight read above is
    # now stamped with a generation that is no longer current.
    assert stale.get("window_generation") != cache._window_generation

    assert cache.adopt(stale) is False


def test_jobs_cache_reset_window_bumps_the_generation_only_when_it_changes_something():
    """``reset_window`` is a no-op (and must stay one) when the window is
    already at ``LIST_LIMIT`` -- so the generation should not move either."""

    cache = JobsCache(object())
    before = cache._window_generation
    cache.reset_window()  # already LIST_LIMIT: an explicit no-op
    assert cache._window_generation == before
    cache.limit = LIST_LIMIT * 2
    cache.reset_window()
    assert cache._window_generation == before + 1
