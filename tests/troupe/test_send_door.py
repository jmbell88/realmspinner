"""The two silent Troupe doors, and the question they now ask.

``send_to_troupe`` has always taken a sprite size and a rig template. The
library's right-click item and the inspector's button passed neither, so a
64 px sprite meant knowing to enter Troupe first and open a collapsed
sub-header, and a quadruped was rigged as a humanoid and animated with human
walk cycles -- discovered after a rig and up to 512 EEVEE frames.
"""

from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import pytest

from warlock.service import troupe as svc_troupe
from warlock.service.errors import Invalid
from warlock.studio import asset_exits, troupe_mode
from warlock.studio.panes import troupe_send


class _Ctx:
    """The slice of the app context the door's logic touches. No GL."""

    def __init__(self, svc):
        self.svc = svc
        self.cache = svc.store
        self.state = SimpleNamespace(
            troupe=None, preview={}, mode="library", troupe_send=None
        )
        self.submitted: list[tuple[str, object]] = []

    def job_dir(self, job_id):
        return self.svc.job_dir(job_id)

    def toast(self, *a, **k):
        pass

    def busy(self, key):
        return False

    def submit(self, key, run, *a, **kw):
        self.submitted.append((key, run))
        return True


@pytest.fixture
def ctx(svc):
    return _Ctx(svc)


def _mesh(svc, *, rigged=False):
    job_id = svc.store.create("image", "a hooded ranger", {}, stage="model")
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"fake-glb")
    files = ["model.glb"]
    if rigged:
        (job_dir / "rig.glb").write_bytes(b"fake-rig")
        (job_dir / "rig.json").write_text(
            json.dumps({"template": "humanoid"}), "utf-8"
        )
        files.append("rig.glb")
    svc.store.set_status(job_id, "done")
    return {"id": job_id, "prompt": "a hooded ranger", "files": files}


def test_the_library_door_asks_before_it_spends_a_rig(ctx, svc):
    """The press used to submit; now it opens the question and submits nothing
    until Send."""
    job = _mesh(svc)
    assert troupe_send.ask(ctx, job)
    assert troupe_send.is_open(ctx)
    assert ctx.submitted == []
    # ``asset_exits._troupe_in`` rather than the library's own item: the two
    # doors are one builder now (2026-09-07), which is what the next test
    # asserts. The claim is unchanged -- the press opens the question.
    source = inspect.getsource(asset_exits._troupe_in)
    assert "troupe_send.ask" in source
    assert "troupe_mode.send_to_troupe" not in source


def test_both_doors_ask_the_same_question():
    """They cannot disagree, because there is one of them.

    This used to read two sources and assert they said the same thing. The
    library's menu and the inspector's button are both drawn from
    ``asset_exits.exits_for`` now, so the guarantee is structural rather than
    a pair of greps that could both be edited -- and the grep that remains is
    on the single builder they share.
    """
    source = inspect.getsource(asset_exits._troupe_in)
    # ``ctx, _mesh`` since B1 (2026-09-09): a follow-up row's own exits close
    # over the *resolved* mesh (``asset_exits._mesh_for``), not the selected
    # row -- both call sites invoke ``exit_.open(ctx, job)`` with whatever the
    # user actually picked, which for a rig row is not a mesh at all.
    assert "troupe_send.ask(ctx, _mesh)" in source
    assert asset_exits._troupe_in in asset_exits._BUILDERS


def test_a_size_chosen_at_the_door_reaches_the_job_row(ctx, svc):
    """Through the real service, not a captured kwarg: 32 was the fallback the
    user could not see."""
    job = _mesh(svc, rigged=True)
    troupe_send.ask(ctx, job)
    state = ctx.state.troupe_send
    state.logical_size = 64
    troupe_send._send(ctx, state, troupe_mode.form(ctx))
    (_key, run), = ctx.submitted
    made = run()
    row = svc.store.get(made["id"])
    assert row["params"]["logical_size"] == 64


def test_the_door_remembers_the_size_the_last_send_chose(ctx, svc):
    """One form behind both doors, so the inspector opens on what the library
    chose. Two doors remembering separately would be two defaults."""
    job = _mesh(svc, rigged=True)
    troupe_send.ask(ctx, job)
    state = ctx.state.troupe_send
    state.logical_size = 64
    troupe_send._send(ctx, state, troupe_mode.form(ctx))

    troupe_send.ask(ctx, job)
    assert ctx.state.troupe_send.logical_size == 64
    assert troupe_mode.form(ctx)["logical_size"] == 64


def test_a_cancelled_question_leaves_the_form_alone(ctx, svc):
    job = _mesh(svc, rigged=True)
    troupe_send.ask(ctx, job)
    before = int(troupe_mode.form(ctx)["logical_size"])
    ctx.state.troupe_send.logical_size = 128
    troupe_send.close(ctx)
    assert troupe_mode.form(ctx)["logical_size"] == before
    assert not troupe_send.is_open(ctx)


def test_an_unrigged_mesh_may_be_rigged_on_any_skeleton_that_has_clips(ctx, svc, monkeypatch):
    """The second silent default. ``humanoid`` was pinned, and the manual's own
    workaround for a quadruped was to go and re-rig it from Create."""
    offered = {row["key"] for row in svc_troupe.clip_templates()}
    assert {"humanoid", "quadruped", "bird", "blob"} <= offered
    assert "fish" not in offered  # nothing is authored for it

    monkeypatch.setattr(
        "warlock.doctor.blender_check", lambda: SimpleNamespace(ok=True, detail="")
    )
    job = _mesh(svc)
    troupe_send.ask(ctx, job)
    state = ctx.state.troupe_send
    state.template = "quadruped"
    troupe_send._send(ctx, state, troupe_mode.form(ctx))
    (_key, run), = ctx.submitted
    made = run()
    row = svc.store.get(made["id"])
    assert row["kind"] == "rig"
    assert row["params"]["template"] == "quadruped"
    assert row["params"]["troupe_sheet"]["template"] == "quadruped"


def test_a_skeleton_with_no_clips_is_refused_with_a_field(svc):
    """The refusal already happened, as "the fish clip library is missing
    'walk'" -- a message about a dictionary, with no field for the control that
    now asks the question."""
    job = _mesh(svc)
    with pytest.raises(Invalid) as excinfo:
        svc_troupe.send_to_troupe(svc, job["id"], template="fish")
    assert excinfo.value.field == "template"
    assert "fish" in str(excinfo.value)

    with pytest.raises(Invalid) as unknown:
        svc_troupe.send_to_troupe(svc, job["id"], template="dragon")
    assert unknown.value.field == "template"


def test_a_missing_specific_clip_on_an_otherwise_clip_bearing_template_is_refused_with_a_field(
    svc, monkeypatch
):
    """The 2026-09-11 audit, finding service-06.

    ``has_clips`` -- the guarded refusal three lines above the one under test
    -- only proves a template's library is *non-empty*; it says nothing about
    whether the library holds the *specific* clip a layout names. A template
    with some clips but not all five is an ordinary shape for a user-edited
    library: ``service.clips.save``'s own ``_check_renders`` only holds
    ``TROUPE_TEMPLATE`` ("humanoid") to Troupe's frame table -- every other
    template's library is accepted with any subset of clips, by design, "since
    another template's clips are not laid into Troupe's frame table at all".
    So a quadruped whose walk cycle was authored but whose jump never was still
    answers ``has_clips`` True, and the request reaches ``expand_clips``'s
    ``KeyError`` branch in ``_charsheet_spec`` -- unguarded, and until now with
    no ``field=`` for the Skeleton control (``troupe_send._skeleton``) that
    asked the question.
    """
    from warlock import rigging

    library = rigging.clip_library("quadruped")
    trimmed = {
        "poses": library["poses"],
        "clips": [c for c in library["clips"] if c["name"] != "jump"],
    }
    monkeypatch.setattr(
        rigging, "clip_library", lambda key: trimmed if key == "quadruped" else library
    )
    # Still non-empty -- the guarded ``has_clips`` refusal does not catch this.
    assert svc_troupe.has_clips("quadruped")

    monkeypatch.setattr(
        "warlock.doctor.blender_check", lambda: SimpleNamespace(ok=True, detail="")
    )
    job = _mesh(svc)
    with pytest.raises(Invalid) as excinfo:
        svc_troupe.send_to_troupe(svc, job["id"], template="quadruped")
    assert excinfo.value.field == "template"
    assert "jump" in str(excinfo.value)


def test_a_rigged_mesh_is_not_asked_which_skeleton_to_use(ctx, svc):
    """The skeleton is on disk and ``create_charsheet`` reads it off
    ``rig.json``, so a picker there would be a control whose value is
    discarded."""
    troupe_send.ask(ctx, _mesh(svc, rigged=True))
    assert ctx.state.troupe_send.rigged is True
    source = inspect.getsource(troupe_send._skeleton)
    assert "if state.rigged:" in source
    assert "return" in source


def test_colours_is_hidden_when_a_palette_is_named(ctx, svc):
    """``troupe_settings._palette``'s rule: the budget is what a *derived*
    palette gets, so offering it beside a named one is a control that is
    silently ignored."""
    source = inspect.getsource(troupe_send._body)
    assert "if state.palette:" in source
    assert source.index("if state.palette:") < source.index('"Colours"')

    job = _mesh(svc, rigged=True)
    troupe_mode.form(ctx)["palette"] = "nes"
    troupe_send.ask(ctx, job)
    state = ctx.state.troupe_send
    state.colors = 8
    troupe_send._send(ctx, state, troupe_mode.form(ctx))
    # A named palette means the budget was never askable, so it is not written.
    assert troupe_mode.form(ctx)["colors"] != 8


def test_the_send_dialog_owns_the_keyboard():
    """Every global shortcut leaking through a modal is UX-08."""
    from warlock.studio import dialogs

    source = inspect.getsource(dialogs.modal_open)
    assert "troupe_send.is_open(ctx)" in source


# --- the bug this work sits on top of ----------------------------------------


def test_building_another_sheet_with_no_form_yet_uses_the_modes_defaults(ctx, svc):
    """``troupe_sheets._rebuild``'s fallback called
    ``troupe_settings._form(state, troupe_settings._options(ctx))``. Both moved
    to ``troupe_mode`` on 2026-09-05 and neither exists in that module any
    more, so the guard against submitting the door's defaults raised
    AttributeError in exactly the case it exists for -- ``state.form`` empty,
    which is a fresh session's state."""
    from warlock.studio.panes import troupe_settings, troupe_sheets

    assert not hasattr(troupe_settings, "_form")
    state = troupe_mode.ensure(ctx)
    state.form = {}
    form = troupe_sheets._form(ctx, state)
    assert form["logical_size"] == 32
    assert form is troupe_mode.form(ctx)


def test_a_door_that_asks_first_says_so():
    """The ellipsis convention, on the two labels that gained a dialog."""
    from warlock.studio import verbs

    source = inspect.getsource(asset_exits._troupe_in)
    assert "verbs.send_to('troupe')}..." in source
    assert verbs.send_to("troupe") == "Send to Troupe"
