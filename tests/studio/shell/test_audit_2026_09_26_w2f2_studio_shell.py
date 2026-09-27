"""Two 2026-09-26 audit findings against the Settings pane / shell landing:

shell-review-settings-01: the Storage pane's model-store and evidence-archive
figures (``app_settings._model_storage``/``_evidence_storage``) are each
measured once per session, behind a module flag
(``_MEASURED``/``_EVIDENCE_MEASURED``) nothing ever cleared again -- so a
download, a removal or a delete that actually changed what is on disk left
the pane showing whatever it read the first time Settings opened, for the
rest of the session.

shell-review-settings-02: ``app_settings._staged`` verifies a staged
installer by hashing it -- "hundreds of megabytes" per its own module
comment -- and used to do that inline, on whatever thread called it, the
first time a new stat slot was seen; that thread is the frame thread, since
this pane draws every frame the Updates category is open.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest


class _Done:
    def __init__(self, key: str, result: Any = None, tag: Any = None) -> None:
        self.key = key
        self.result = result
        self.tag = tag


class _Cache:
    def __init__(self) -> None:
        self.invalidated = 0

    def invalidate(self) -> None:
        self.invalidated += 1

    def measure(self) -> dict:
        return {}


class _Ctx:
    """Just enough of ``ctx`` for the ``download:``/``remove:``/``delete:``
    landing branches to run: ``submit`` records rather than runs, since
    driving a real service is not what this test is about.
    """

    def __init__(self) -> None:
        self.submitted: list[str] = []
        self.calls: list[tuple[str, Any, tuple, dict]] = []
        self.toasts: list[tuple] = []
        self.cache = _Cache()
        self.tasks = SimpleNamespace(set_progress=lambda *a, **k: None)
        self.svc = object()

    def submit(self, key: str, fn: Any = None, *args: Any, **kwargs: Any) -> bool:
        self.submitted.append(key)
        self.calls.append((key, fn, args, kwargs))
        return True

    def toast(self, *a: Any, **k: Any) -> None:
        self.toasts.append((a, k))


class _App:
    """Just enough of ``App`` for ``_on_task_done``'s routing to run --
    ``test_audit_2026_09_26_w1f3_studio_shell.py``'s own shape."""

    def __init__(self, ctx: Any) -> None:
        self.app_ctx = ctx
        self.svc = object()  # never touched: ``ctx.submit`` above does not run its fn

    def _report_failed_checks(self) -> None:  # pragma: no cover - not exercised here
        pass

    def _request_storage(self, job_id: Any = None) -> None:
        self.app_ctx.submit("storage" if job_id is None else f"storage:{job_id}")


@pytest.fixture(autouse=True)
def _clean_flags():
    """Belt and braces over ``tests/conftest.py``'s own autouse reset: this
    file sets the flags to ``True`` directly (rather than through
    ``_model_storage``/``_evidence_storage``) to start each test from "already
    measured", so it needs its own guaranteed-clean start and end."""
    from realmspinner.studio.modes.settings.ui.panes import app_settings

    app_settings._reset_measure()
    yield
    app_settings._reset_measure()


def test_model_storage_is_remeasured_after_a_download_or_removal_lands():
    """Against the unfixed ``_on_task_done``, ``_MEASURED`` stays ``True``
    after a ``"download:"``/``"remove:"`` task lands, so
    ``app_settings._model_storage`` keeps answering from its first
    measurement no matter how many downloads or removals happened since.
    """
    from realmspinner.studio import main as main_mod
    from realmspinner.studio.modes.settings.ui.panes import app_settings

    app_settings._MEASURED = True
    ctx = _Ctx()
    app = _App(ctx)

    main_mod.App._on_task_done(app, _Done("download:sdxl-1.0"))
    assert app_settings._MEASURED is False, "a download landing must force a re-measure"

    app_settings._MEASURED = True
    main_mod.App._on_task_done(app, _Done("remove:sdxl-1.0"))
    assert app_settings._MEASURED is False, "a removal landing must force a re-measure"


def test_evidence_storage_is_remeasured_after_a_delete_prune_purge_or_empty_trash_lands():
    """The evidence archive's own half: ``_evidence_storage``'s own docstring
    says it fills up *on* a delete, but nothing cleared
    ``_EVIDENCE_MEASURED`` when one landed -- so the "Kept as evidence" line
    stayed at whatever it read on the first open of Settings, through
    however many deletes happened after.
    """
    from realmspinner.studio import main as main_mod
    from realmspinner.studio.modes.settings.ui.panes import app_settings

    for key in ("delete:job1", "prune", "purge:job2", "empty-trash"):
        app_settings._EVIDENCE_MEASURED = True
        ctx = _Ctx()
        app = _App(ctx)

        main_mod.App._on_task_done(app, _Done(key))

        assert app_settings._EVIDENCE_MEASURED is False, (
            f"{key!r} landing must force the evidence archive to re-measure"
        )


# --- shell-review-settings-02: the installer digest ran on the frame thread -


def test_the_staged_installer_digest_is_not_computed_on_the_frame_thread(tmp_path, monkeypatch):
    """Against the unfixed ``_staged``, calling it directly -- the way the
    Updates category's own frame-thread draw does -- runs the whole SHA-256
    right there and returns a verified path on the very first call, with
    nothing submitted for a task thread to do instead.
    """
    import hashlib

    from realmspinner.service import updates as svc_updates
    from realmspinner.studio.modes.settings.ui.panes import app_settings

    app_settings._stale_staged_installer()

    installer = tmp_path / "Setup.exe"
    installer.write_bytes(b"installer bytes")
    digest = hashlib.sha256(installer.read_bytes()).hexdigest()
    info = {"installer_name": "Setup.exe", "sha256": digest}

    monkeypatch.setattr(svc_updates, "staging_dir", lambda svc: tmp_path)
    hashed: list[Any] = []
    real_sha256 = svc_updates._sha256

    def _spy_sha256(path):
        hashed.append(path)
        return real_sha256(path)

    monkeypatch.setattr(svc_updates, "_sha256", _spy_sha256)

    ctx = _Ctx()
    result = app_settings._staged(ctx, info)

    assert result is None, "not verified yet -- the frame thread must not have hashed anything"
    assert hashed == [], "the digest must not run on the thread that called _staged"
    assert ctx.submitted == [app_settings.STAGED_TASK_KEY], (
        "a verification task must be submitted instead of hashing inline"
    )

    # Running the deferred task -- as a task thread eventually would -- and
    # landing its result through the same branch ``_on_task_done`` uses is
    # what actually verifies the file and fills the cache.
    key, fn, args, kwargs = ctx.calls[0]
    result_tuple = fn(*args, **kwargs)
    assert hashed == [installer], "the digest runs once the deferred task itself is invoked"

    from realmspinner.studio import main as main_mod

    app = _App(ctx)
    main_mod.App._on_task_done(app, _Done(key, result=result_tuple))

    assert app_settings._staged(ctx, info) == installer
    assert hashed == [installer], "a cache hit must not hash the file a second time"
