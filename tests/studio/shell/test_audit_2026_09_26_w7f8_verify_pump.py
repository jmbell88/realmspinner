"""Regression tests for the 2026-09-26 audit's finding shell-shell-pkg-03
(wave w7f8): ``pack:``/``download:``/``remove:`` landings ignored the return
value of their own ``ctx.submit(VERIFY_KEY, ..., force=True)``, so a second
package install performed while an earlier probe was still in flight
(``submit`` refuses a key already in flight) was never actually re-probed.

``TasksMixin._request_verify`` now marks ``ctx.state.preview["verify_dirty"]``
on a refusal, and ``TasksMixin._pump_verify`` -- called every frame from
``FrameMixin._refresh`` -- retries it, the same "a refused submit is retried
on some later frame" shape ``review_mode.pump_findings``/``pump_judge`` use.
"""

from __future__ import annotations

from typing import Any

import pytest

from realmspinner.studio import main as main_mod
from realmspinner.studio.state import AppState


class _Done:
    def __init__(self, key: str, result: Any = None) -> None:
        self.key = key
        self.result = result


class _Ctx:
    """Just enough of ``ctx`` for ``_request_verify``/``_pump_verify`` and the
    ``pack:``/``download:``/``remove:`` landing branches around them."""

    def __init__(self, *, accept_verify: bool = True) -> None:
        self.accept_verify = accept_verify
        self.submitted: list[str] = []
        self.state = AppState()
        self.toasts: list[tuple] = []
        from types import SimpleNamespace

        self.tasks = SimpleNamespace(set_progress=lambda *a, **k: None)
        self.svc = object()

    def submit(self, key: str, _fn: Any = None, *_a: Any, **_k: Any) -> bool:
        self.submitted.append(key)
        if key == "verify-install":
            return self.accept_verify
        return True

    def toast(self, *a: Any, **k: Any) -> None:
        self.toasts.append((a, k))


class _App:
    """Just enough of ``App`` for ``_on_task_done``'s ``pack:``/``download:``/
    ``remove:`` branches to run -- ``test_audit_2026_09_26_w2f2_studio_shell.py``'s
    own shape. ``_request_verify``/``_pump_verify`` are the real methods,
    not stand-ins: this file's whole point is what they do."""

    _request_verify = main_mod.App._request_verify
    _pump_verify = main_mod.App._pump_verify

    def __init__(self, ctx: _Ctx) -> None:
        self.app_ctx = ctx
        self.svc = object()

    def _resume_deferred_quit(self) -> None:
        pass


@pytest.fixture(autouse=True)
def _clean_model_storage_flag():
    """``download:``/``remove:`` also calls ``app_settings._stale_model_storage``
    -- harmless here, but leave the module flag as this test found it."""
    from realmspinner.studio.modes.settings.ui.panes import app_settings

    app_settings._reset_measure()
    yield
    app_settings._reset_measure()


# --- _request_verify itself ---------------------------------------------------


def test_a_refused_verify_submit_marks_it_dirty_instead_of_dropping_it():
    ctx = _Ctx(accept_verify=False)
    app = _App(ctx)

    main_mod.App._request_verify(app)

    assert ctx.submitted == ["verify-install"]
    assert ctx.state.preview.get("verify_dirty") is True


def test_an_accepted_verify_submit_leaves_nothing_dirty():
    ctx = _Ctx(accept_verify=True)
    app = _App(ctx)
    ctx.state.preview["verify_dirty"] = True  # a stale mark from an earlier refusal

    main_mod.App._request_verify(app)

    assert ctx.submitted == ["verify-install"]
    assert "verify_dirty" not in ctx.state.preview


# --- _pump_verify: the retry ---------------------------------------------------


def test_pump_verify_does_nothing_when_nothing_is_dirty():
    ctx = _Ctx(accept_verify=True)
    app = _App(ctx)

    main_mod.App._pump_verify(app)

    assert ctx.submitted == []


def test_pump_verify_retries_and_clears_the_flag_once_it_is_accepted():
    ctx = _Ctx(accept_verify=True)
    app = _App(ctx)
    ctx.state.preview["verify_dirty"] = True

    main_mod.App._pump_verify(app)

    assert ctx.submitted == ["verify-install"]
    assert "verify_dirty" not in ctx.state.preview


def test_pump_verify_stays_dirty_while_still_refused():
    ctx = _Ctx(accept_verify=False)
    app = _App(ctx)
    ctx.state.preview["verify_dirty"] = True

    main_mod.App._pump_verify(app)

    assert ctx.state.preview.get("verify_dirty") is True


# --- end to end: a pack/download/remove landing during an in-flight probe -----


def test_a_second_install_landing_during_an_in_flight_probe_is_retried_not_dropped():
    """The exact shape the finding describes: a ``download:``/``remove:``
    landing whose own ``VERIFY_KEY`` submit is refused (an earlier probe is
    still running) used to silently never re-probe. It must instead be
    retried once ``_pump_verify`` runs on a later frame and that earlier
    probe has landed."""
    ctx = _Ctx(accept_verify=False)
    app = _App(ctx)

    main_mod.App._on_task_done(app, _Done("download:sdxl-1.0"))

    # The refusal did not stop the rest of the landing's own bookkeeping --
    # only the re-probe itself was silently dropped before this fix.
    assert ctx.toasts, "the landing's own toast must still fire"
    assert ctx.state.preview.get("verify_dirty") is True

    # A later frame, after the earlier probe has landed and freed the key.
    ctx.accept_verify = True
    main_mod.App._pump_verify(app)

    assert ctx.submitted.count("verify-install") == 2, (
        "the retry must actually resubmit the probe, not just clear the flag"
    )
    assert "verify_dirty" not in ctx.state.preview


def test_a_pack_landing_during_an_in_flight_probe_is_also_retried():
    ctx = _Ctx(accept_verify=False)
    app = _App(ctx)

    from realmspinner.service import packs as svc_packs

    def _no_unresolved(_keys):
        return []

    orig = svc_packs.unresolved
    svc_packs.unresolved = _no_unresolved
    try:
        main_mod.App._on_task_done(app, _Done("pack:text2image"))
    finally:
        svc_packs.unresolved = orig

    assert ctx.state.preview.get("verify_dirty") is True
    assert ctx.toasts, "the pack landing's own toast must still fire"
