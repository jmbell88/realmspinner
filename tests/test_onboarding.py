"""The F-section of NEXT_ROADMAP Phase 2: onboarding, doctor and remedies.

The theme is that a failure the app already knows about should be answerable
without leaving the app. Every test here is about a *remedy* being present and
reachable, not about a check being right -- ``tests/test_doctor.py`` owns that.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from warlock import config as config_module
from warlock import doctor, fetch
from warlock.config import Config
from warlock.service.errors import Invalid

# --- F54: every failing check still names a way forward ---------------------


def _config(tmp_path) -> Config:
    return Config(
        data_dir=tmp_path / "assets",
        db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe",
        trellis_models_dir=tmp_path / "no-models",
        t2i_model_root=tmp_path / "t2i",
        bench_dir=tmp_path / "bench",
    )


def test_every_failing_check_names_a_remedy(tmp_path):
    """The rule the non-fatal model rows have always followed (F54).

    **Neither prerequisite this fixture is missing is fatal any more.** The
    GGUF weights stopped being fatal 2026-09-04, and ``trellis-server.exe``
    stopped today: it is a ``Settings -> Models`` download
    (``models.ENGINE_MODELS["trellis_runtime"]``) now, so a machine that has
    not fetched it is an ordinary fresh install, not a broken one --
    ``tests/test_doctor.py`` asserts "no fatal rows on a correct fresh
    install" as its own headline claim, and this fixture is exactly that
    install. Both rows report ``pending_install`` instead, and re-aiming the
    claim at *those* rows is what keeps it worth asserting.

    **``pending_install`` rather than every failing row**, which is a narrower
    net than it looks and deliberately so. ``pending_install`` means "you have
    not downloaded this yet", so a row carrying it is by definition asking the
    user to do something, and one that asks without saying how is the F54
    defect exactly. A plain warning is a different animal: ``trellis:
    birefnet.gguf`` fails by saying background matting falls back to a
    threshold cutout, which is a stated consequence and not a request. Widening
    this to every failing row would either fail on rows like that or force the
    word list to accept any prose at all, and a test that accepts any prose is
    not testing anything.
    """
    config = _config(tmp_path)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    checks = doctor.static_checks(config, probe_slow=False)
    failing = [c for c in checks if not c.ok and c.pending_install]
    assert failing, "the fixture is meant to have downloaded neither prerequisite"
    for check in failing:
        assert any(
            word in check.detail
            for word in ("download", "unpack", "install", "WARLOCK_")
        ), f"{check.name} names no remedy: {check.detail!r}"


def test_the_gguf_remedy_is_the_command_from_the_install_instructions(tmp_path):
    """Spelled once. The README and the manual carry the same line, and a row
    that invented its own would send a user to a different repo."""
    config = _config(tmp_path)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    row = next(
        c for c in doctor.static_checks(config, probe_slow=False)
        if c.name == "TRELLIS GGUF weights"
    )
    hint = doctor.trellis_gguf_hint(config)
    assert hint in row.detail
    assert "ilintar/trellis2-gguf" in hint
    # Resolved, never relative: the whole reason the constant became a
    # function (see its docstring).
    assert f"--local-dir {fetch.quote_for_shell(config.trellis_models_dir)}" in hint


# --- F55: missing weights are refused at the door ---------------------------


def test_a_text_job_whose_checkpoint_is_absent_is_refused_with_its_command(svc):
    """Refused before the queue, in the shape ``check_vram`` set: name the
    problem and a thing the user can do about it."""
    from warlock import fetch, models
    from warlock.service import jobs as svc_jobs

    spec = models.BASE_MODELS[config_module.DEFAULT_BASE_MODEL]
    (fetch.base_model_dir(svc.config, spec) / "model_index.json").unlink()

    with pytest.raises(Invalid) as caught:
        svc_jobs.create_job(svc, kind="text", prompt="a rock", output="reference")
    message = caught.value.message
    assert caught.value.field == "base_model"
    # The command resolved against *this* service's config, not the registry's
    # default-home rendering: a refusal that names one directory and a remedy
    # that names another is DST-02, and this door is one of the places a user
    # copies the line straight out of.
    assert fetch.download_text(svc.config, "base", spec) in message
    assert str(fetch.base_model_dir(svc.config, spec)) in message
    # And it leads with the route that does not need a terminal.
    assert "Settings" in message


def test_a_missing_style_lora_is_refused_rather_than_silently_skipped(svc):
    """The one that would otherwise fail *quietly*: ``_load_loras`` skips a
    missing style adapter, so the job would finish looking wrong while its
    params claimed a style that never ran -- and that row would then join the
    findings corpus as evidence about it."""
    from warlock import fetch, models
    from warlock.service import jobs as svc_jobs

    key, lora = next(iter(models.STYLE_LORAS.items()))
    (svc.config.t2i_model_root / "loras" / lora.filename).unlink()

    with pytest.raises(Invalid) as caught:
        svc_jobs.create_job(
            svc, kind="text", prompt="a rock", output="reference",
            guidance_fields={"style_lora": key},
        )
    assert caught.value.field == "style_lora"
    assert fetch.download_text(svc.config, "lora", lora) in caught.value.message
    assert str(svc.config.t2i_model_root / "loras") in caught.value.message


def test_an_image_job_is_not_asked_about_image_models(svc):
    """An image job never touches SDXL, so its checkpoint is irrelevant --
    and refusing one for a missing image model would be a refusal the user
    could not act on, since no control of theirs chose it."""
    from warlock import fetch, models
    from warlock.service import jobs as svc_jobs

    for spec in models.BASE_MODELS.values():
        index = fetch.base_model_dir(svc.config, spec) / "model_index.json"
        if index.exists():
            index.unlink()

    png = _tiny_png()
    job = svc_jobs.create_job(svc, kind="image", image=png)
    assert job["id"]


def test_a_refused_job_leaves_nothing_behind(svc):
    """The ordering rule ``create_job`` states: the guard sits with the other
    door checks, before ``input.png`` is written."""
    from warlock import fetch, models
    from warlock.service import jobs as svc_jobs

    spec = models.BASE_MODELS[config_module.DEFAULT_BASE_MODEL]
    (fetch.base_model_dir(svc.config, spec) / "model_index.json").unlink()
    before = set(svc.config.data_dir.glob("*"))

    with pytest.raises(Invalid):
        svc_jobs.create_job(svc, kind="text", prompt="a rock", output="reference")

    assert set(svc.config.data_dir.glob("*")) == before
    assert svc.store.list(10) == []


def _tiny_png() -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), "grey").save(buf, "PNG")
    return buf.getvalue()


# --- F57: the three surfaces that lead to troubleshooting -------------------


def test_troubleshooting_is_named_once_and_resolves():
    from warlock.kernels.manual import loader, targets

    chapter, anchor = targets.TROUBLESHOOTING
    assert anchor is None
    assert chapter in {c.key for c in loader.chapters()}


def test_troubleshooting_is_not_a_help_target():
    """It is the same shape and deliberately not in that dict: HELP_TARGETS is
    asserted against the pane (?) call sites in both directions, and none of
    the three surfaces that lead here is a pane with a (?)."""
    from warlock.kernels.manual.targets import HELP_TARGETS

    assert "diagnostics" not in HELP_TARGETS


# --- F59: Dismiss puts it away, it does not forget it -----------------------


def test_dismissing_the_banner_keeps_the_text_reachable():
    from warlock.studio.state import AppState

    state = AppState()
    state.note_error("trellis-server.exe: not found at C:/x")
    state.note_error("The GPU worker stopped: boom. Restart Warlock.")
    state.dismiss_errors()

    assert state.errors == []
    assert len(state.dismissed_errors) == 2
    assert "trellis-server.exe" in state.dismissed_errors[0]


def test_dismissing_twice_does_not_duplicate():
    from warlock.studio.state import AppState

    state = AppState()
    state.note_error("boom")
    state.dismiss_errors()
    state.note_error("boom")
    state.dismiss_errors()
    assert state.dismissed_errors == ["boom"]


# --- 2026-09-10: the engine-runtime gate reaches every door in front of it --
#
# ``engine:trellis_runtime`` (the reconstruction engine's own binaries, no
# longer shipped in the installer) joined ``modes.NEEDS_ROWS["create"]``
# beside the GGUF weights. The rail and the palette already read that table
# through ``model_gate.mode_gate``; these two groups are the doors that did
# not -- Home's own "New..." menu, which called ``create_stages.go`` (which
# does end in ``state.set_mode`` and so refuses, but silently, after the form
# was already reset), and the tour offer, which could pick a tour whose first
# step waits on a mode ``state.set_mode`` will never grant.


def _gated_ctx(*, model_rows=(), pack_rows=(), total=0):
    """A ctx with just enough on it for ``model_gate.mode_gate`` and a real
    ``state.set_mode`` to answer -- the same shape ``test_mode_gate.py``'s own
    ``_ctx`` builds, plus the bits Home's actions touch."""
    from warlock.studio.state import AppState

    return SimpleNamespace(
        state=AppState(),
        model_rows=list(model_rows),
        pack_rows=list(pack_rows),
        model_picks=set(),
        cache=SimpleNamespace(total=total),
        settings=SimpleNamespace(get=lambda *_a, **_k: None, set=lambda *_a, **_k: None),
        toast=lambda *_a, **_k: None,
    )


_MISSING_CREATE_ROWS = [
    {"row_key": "engine:trellis_gguf", "present": False, "size_gib": 16.1},
    {"row_key": "engine:trellis_runtime", "present": False, "size_gib": 0.68},
    {"row_key": "base:sdxl_cfg", "present": False, "size_gib": 6.5},
]


def test_homes_new_3d_model_does_not_open_create_through_a_shut_gate(monkeypatch):
    """Before this, ``landing.start_3d`` called ``create_stages.go`` directly
    rather than checking the gate first -- so a fresh install saw its Mesh
    form reset and its selection cleared, and then nothing: ``go`` ends in
    ``state.set_mode``, which refused silently, leaving the reader back on
    Home with no explanation. The rail and the palette both turn the same
    refusal into a trip to Settings; Home's menu must now do the same."""
    from warlock.studio.modes.create.ui import stages as create_stages
    from warlock.studio.panes import app_settings, landing

    ctx = _gated_ctx(model_rows=_MISSING_CREATE_ROWS)
    called = []
    monkeypatch.setattr(create_stages, "go", lambda *a, **k: called.append(a))

    landing.start_3d(ctx)

    assert called == [], "a shut gate must not open Create's Mesh stage"
    assert ctx.state.mode == "settings"
    assert ctx.state.preview[app_settings.CATEGORY_SLOT] == "models"


def test_homes_new_2d_image_routes_to_packs_before_models(monkeypatch):
    """Packs first, same ordering as ``model_gate.mode_gate``: Home's menu
    must agree with the rail about which door a gated Create actually points
    at, or a user sent to Models here buys nothing without the pack too."""
    from warlock.studio.modes.create.ui import stages as create_stages
    from warlock.studio.panes import app_settings, landing

    pack_row = {
        "key": "text2image",
        "label": "Image generation",
        "modes": ["create"],
        "present": False,
        "download_gib": 3.3,
    }
    ctx = _gated_ctx(pack_rows=[pack_row])
    called = []
    monkeypatch.setattr(create_stages, "go", lambda *a, **k: called.append(a))

    landing.start_2d(ctx)

    assert called == []
    assert ctx.state.mode == "settings"
    assert ctx.state.preview[app_settings.CATEGORY_SLOT] == "packs"


def test_homes_new_menu_still_opens_create_once_the_door_is_open(monkeypatch):
    """The gate check must not itself become a new way to refuse a healthy
    install: with everything present, the New... menu still opens Create."""
    from warlock.studio.modes.create.ui import stages as create_stages
    from warlock.studio.panes import landing

    present_rows = [dict(row, present=True) for row in _MISSING_CREATE_ROWS]
    ctx = _gated_ctx(model_rows=present_rows)
    called = []
    monkeypatch.setattr(create_stages, "go", lambda *a, **k: called.append(a))

    landing.start_3d(ctx)

    assert called == [(ctx, "mesh")]
    assert ctx.state.mode != "settings"


def test_a_tour_whose_mode_is_gated_is_not_offered():
    """``muse-basics`` is every step naming ``mode="muse"``, and its first
    step waits on ``Condition("mode_is", "muse")`` -- which ``state.set_mode``
    refuses outright while the ACE-Step weights are missing (H14). Offered
    anyway, the card would sit on step 1 forever with no way forward."""
    from warlock.studio.panes import landing
    from warlock.studio.tour import scripts as tour_scripts

    row = {"row_key": "music:ace_step_v1", "present": False, "size_gib": 8.3}
    ctx = _gated_ctx(model_rows=[row])
    # Every other tour already finished, so muse-basics would otherwise be
    # exactly the one ``_offerable_tour`` reaches for next.
    ctx.state.tour.finished = tuple(
        one.key for one in tour_scripts.TOURS if one.key != "muse-basics"
    )
    assert landing._offerable_tour(ctx) is None


def test_an_ungated_tour_is_offered_once_its_door_is_open():
    """The other direction: a gate check that never lifts is as wrong as one
    that never falls."""
    from warlock.studio.panes import landing
    from warlock.studio.tour import scripts as tour_scripts

    row = {"row_key": "music:ace_step_v1", "present": True, "size_gib": 8.3}
    ctx = _gated_ctx(model_rows=[row])
    ctx.state.tour.finished = tuple(
        one.key for one in tour_scripts.TOURS if one.key != "muse-basics"
    )
    offer = landing._offerable_tour(ctx)
    assert offer is not None and offer.key == "muse-basics"


def test_a_tour_spanning_more_than_one_mode_derives_no_single_mode():
    """``first-hour`` starts on Home and walks the reader into Create -- it
    must not be gated on either mode alone, which is what a hand-picked "the
    tour's mode is its first step's mode" would have done."""
    from warlock.studio.tour import scripts as tour_scripts

    assert tour_scripts.MUSE_BASICS.mode == "muse"
    assert tour_scripts.INKER_BASICS.mode == "inker"
    assert tour_scripts.FIRST_HOUR.mode is None


def test_starting_a_gated_tour_toasts_the_reason_instead_of_hanging_on_step_one():
    """The other route in: a tour can also be started directly (a settings
    link, a test, the first-run panel's own "Show me around first" for a
    different tour). ``panes.tour.start`` must refuse the same way
    ``landing._offerable_tour`` does, or a caller that skips the offer still
    lands the reader on a card that will never advance."""
    from warlock.studio.panes import tour as tour_pane

    toasts = []
    row = {"row_key": "music:ace_step_v1", "present": False, "size_gib": 8.3}
    ctx = _gated_ctx(model_rows=[row])
    ctx.toast = lambda *args: toasts.append(args)

    tour_pane.start(ctx, "muse-basics")

    assert not ctx.state.tour.running
    assert toasts, "a refused tour must say why rather than doing nothing"
    assert toasts[0][1] == "warn"
