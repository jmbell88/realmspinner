"""Step 5 of the Create redesign: an upload becomes a source, not a job.

Choosing a file on the Mesh Source chip (or dropping one) used to call
``create_job(kind="image")``, which queued a mesh at once -- past the cutout
check, and ignoring Count. It now imports a finished reference row and points
``source_job`` at it, exactly as picking a library card does.
"""

from __future__ import annotations

from types import SimpleNamespace

from realmspinner.service import jobs as svc_jobs
from realmspinner.studio import main as main_mod
from realmspinner.studio import matte_preview
from realmspinner.studio.modes.create.ui.panes import settings_3d
from realmspinner.studio.state import DEFAULT_FORM_3D, AppState


class _Ctx:
    def __init__(self, jobs: dict | None = None) -> None:
        self.state = AppState()
        self.state.form_3d = dict(DEFAULT_FORM_3D)
        self._jobs = dict(jobs or {})
        self.svc = SimpleNamespace()
        self.submitted: list = []
        self.toasts: list = []
        self.invalidated = 0
        self.cache = SimpleNamespace(
            get=lambda job_id: self._jobs.get(job_id),
            jobs=list(self._jobs.values()),
            invalidate=self._invalidate,
        )
        self.textures = None
        self.settings = SimpleNamespace(get=lambda k, d=None: d, set=lambda k, v: None)

    def _invalidate(self) -> None:
        self.invalidated += 1

    def submit(self, key, fn, *args, **kwargs):
        self.submitted.append((key, fn, args, kwargs))
        return True

    def busy(self, key):
        return any(k == key for k, *_ in self.submitted)

    def toast(self, message, level="info", **extra):
        self.toasts.append((message, level))


def test_an_upload_sets_the_source_and_does_not_submit(tmp_path, monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("an upload must not queue a mesh job")

    monkeypatch.setattr(settings_3d.svc_jobs, "create_job", refuse)
    monkeypatch.setattr(
        settings_3d.svc_jobs, "import_reference", lambda svc, image, **kw: {"id": "ref9"}
    )
    ctx = _Ctx()
    path = tmp_path / "hero.png"
    path.write_bytes(b"png")

    settings_3d.upload(ctx, path)
    (key, fn, args, kwargs), = ctx.submitted
    assert key == settings_3d.IMPORT_KEY
    assert key != "submit"
    result = fn(*args, **kwargs)

    main_mod.App._on_task_done(
        SimpleNamespace(app_ctx=ctx), SimpleNamespace(key=key, result=result, error=None)
    )
    assert ctx.state.source_job == "ref9"
    assert [k for k, *_ in ctx.submitted] == [settings_3d.IMPORT_KEY]


def test_an_imported_source_goes_through_the_cutout_check_and_submits_count(monkeypatch):
    ref = {"id": "ref9", "stage": "reference", "status": "done"}
    ref |= {"files": ["input.png"], "params": {}}
    ctx = _Ctx({"ref9": ref})
    ctx.state.source_job = "ref9"
    ctx.state.form_3d["count"] = 3

    settings_3d.promote(ctx, ctx.cache.get("ref9"), ctx.state.form_3d)
    assert matte_preview.is_open(ctx), "Make 3D on an imported source opens the cutout check"
    assert ctx.submitted == [], "nothing is queued until the check is accepted"

    job_id = ctx.state.matte.job_id
    matte_preview.accept(
        ctx, lambda kwargs, force: settings_3d.submit_promotion(ctx, job_id, kwargs, force)
    )
    (key, fn, args, kwargs), = ctx.submitted
    assert key == "submit"
    assert fn is svc_jobs.promote_candidates
    assert args[1] == "ref9"
    assert kwargs["count"] == 3
