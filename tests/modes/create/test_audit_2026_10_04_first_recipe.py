"""The 2026-10-04 audit, Create's first recipe pass (settings_2d.py, brief.py).

create-05  "Decide the pending candidates first" lived only in the Generate
           button's disabled state, so Ctrl+Enter and the palette (which call
           ``settings_2d.generate`` directly) queued a second candidate group
           over an undecided one, orphaning it.
create-06  choosing "Custom" for the cell target while it was blank drew 256 in
           the box and stored nothing, so Generate submitted a blank target.
"""

from __future__ import annotations

from types import SimpleNamespace

from realmspinner.studio.modes.create.engine import assets as create_assets
from realmspinner.studio.modes.create.ui import brief as create_brief
from realmspinner.studio.modes.create.ui.panes import settings_2d
from realmspinner.studio.state import AppState, default_form_2d


def _form(**over):
    form = default_form_2d()
    create_assets.sync_legacy_fields(form)
    form.update({"prompt": "a wooden crate", **over})
    return form


def _pending_jobs():
    return [
        {"id": "a1", "candidate_group": "g", "candidate_index": 0,
         "status": "done", "created_at": 1.0},
        {"id": "a2", "candidate_group": "g", "candidate_index": 1,
         "status": "done", "created_at": 1.0},
    ]


def _ctx(jobs):
    submitted: list[tuple] = []
    toasts: list[tuple] = []
    ctx = SimpleNamespace(
        svc=SimpleNamespace(config=None),
        state=AppState(),
        cache=SimpleNamespace(jobs=jobs),
        toast=lambda *a, **k: toasts.append(a),
        submit=lambda *a, **k: submitted.append(a) or True,
        busy=lambda key: False,
    )
    return ctx, submitted, toasts


# --- create-05 -----------------------------------------------------------------


def test_ctrl_enter_generate_refuses_while_a_candidate_group_is_pending():
    """Ctrl+Enter (``shell.events``) calls ``settings_2d.generate`` itself; the
    refusal the button wears has to be said there too, and nothing queued."""
    form = _form(seed=1234, seed_locked=False)
    ctx, submitted, toasts = _ctx(_pending_jobs())

    settings_2d.generate(ctx, form)

    assert submitted == [], "a keyboard submit queued over an undecided candidate group"
    assert toasts and create_brief.PENDING_CANDIDATES_REFUSAL in str(toasts[0][0])
    assert form["seed"] == 1234, "a refused press must not spend the seed"


def test_palette_generate_refuses_while_a_candidate_group_is_pending(monkeypatch):
    from realmspinner.studio import palette
    from realmspinner.studio.modes.create.ui import stages as create_stages

    monkeypatch.setattr(create_stages, "at", lambda state, key: key == "reference")
    ctx, submitted, toasts = _ctx(_pending_jobs())
    ctx.state.form_2d = _form()

    palette._generate(ctx)

    assert submitted == []
    assert toasts and create_brief.PENDING_CANDIDATES_REFUSAL in str(toasts[0][0])


def test_generate_without_a_pending_group_still_submits():
    """The guard is the refusal, not a broken door: no pending group, it queues."""
    form = _form()
    ctx, submitted, toasts = _ctx([])

    settings_2d.generate(ctx, form)

    assert not toasts
    assert submitted, "an ordinary Generate must still reach the queue"


# --- create-06 -----------------------------------------------------------------


def test_choosing_custom_target_cell_from_blank_stores_the_number_the_box_shows(monkeypatch):
    """The box drew 256 over a blank form value; Generate sent the blank."""
    form: dict = {"target_cell_px": ""}
    shown: list[int] = []

    class _FormUI:
        def number(self, field, label, value, **kwargs):
            shown.append(value)
            return False, value  # nothing typed into the box

    monkeypatch.setattr(settings_2d.widgets, "combo", lambda name, before, values: "custom")
    monkeypatch.setattr(settings_2d.widgets, "muted_wrapped", lambda *a, **k: None)

    settings_2d._target_cell(SimpleNamespace(), form, _FormUI())

    assert shown, "the Custom box was never drawn"
    assert form["target_cell_px"] == str(shown[0]), (
        f"the box shows {shown[0]} but the form holds {form['target_cell_px']!r}"
    )


def test_choosing_custom_target_cell_keeps_a_stored_number(monkeypatch):
    form: dict = {"target_cell_px": "48", "_target_cell_custom": True}

    class _FormUI:
        def number(self, field, label, value, **kwargs):
            return False, value

    monkeypatch.setattr(settings_2d.widgets, "combo", lambda name, before, values: "custom")
    monkeypatch.setattr(settings_2d.widgets, "muted_wrapped", lambda *a, **k: None)

    settings_2d._target_cell(SimpleNamespace(), form, _FormUI())

    assert form["target_cell_px"] == "48"
