"""The 2026-10-04 audit's panes findings create-34, create-37 and create-43.

Retarget's default budget, the sprite draft's delete confirm and the Inker exit
for a reference the viewport toolbar will not draw. Each test's name is the
claim; a fake ``ctx`` and no GL, as ``tests/test_asset_exits.py`` does.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from realmspinner.studio import asset_exits
from realmspinner.studio.panes import retarget_panel, sprite_panel
from realmspinner.studio.state import AppState

# --- create-34: the Triangle budget panel's default is the manual's ----------


class _Path:
    def __init__(self, present: bool) -> None:
        self.present = present

    def is_file(self) -> bool:
        return self.present

    def __str__(self) -> str:
        return f"gltfpack-{'present' if self.present else 'absent'}"


def _retarget_ctx(present: bool) -> SimpleNamespace:
    return SimpleNamespace(
        svc=SimpleNamespace(config=SimpleNamespace(gltfpack_exe=_Path(present))),
        state=SimpleNamespace(preview={}),
    )


def test_retarget_panel_default_profile_is_the_one_the_manual_names(monkeypatch):
    """Manual 23 says "Standard is this panel's default", but ``_form`` started
    every job at "raw", so an untouched Rebuild mesh rebuilt at Raw."""
    monkeypatch.setattr(retarget_panel, "_gltfpack_seen", {})
    form = retarget_panel._form(_retarget_ctx(True), "job-1")
    assert form["profile"] == "standard"


def test_retarget_panel_default_profile_is_raw_when_gltfpack_is_absent(monkeypatch):
    """Every named tier needs the binary, so without it Raw is the only offer and
    the only possible default."""
    monkeypatch.setattr(retarget_panel, "_gltfpack_seen", {})
    form = retarget_panel._form(_retarget_ctx(False), "job-1")
    assert form["profile"] == "raw"


# --- create-37: Delete draft asks first --------------------------------------


class _Confirms:
    def __init__(self) -> None:
        self.asked: list = []

    def ask(self, confirm) -> None:
        self.asked.append(confirm)


class _DraftCtx:
    def __init__(self) -> None:
        self.confirms = _Confirms()
        self.submitted: list[tuple] = []
        self.svc = object()
        self.state = SimpleNamespace(preview={})
        self.textures = None

    def submit(self, key, fn, *args, **kwargs) -> None:
        self.submitted.append((key, fn, args))


def test_delete_draft_asks_before_removing_the_pair(monkeypatch):
    """Sprite "Delete draft" removed up to 16 generations with no confirm and no
    undo, where the sibling sheet Delete confirms."""
    monkeypatch.setattr(
        sprite_panel,
        "imgui",
        SimpleNamespace(push_id=lambda *_a: None, pop_id=lambda *_a: None),
    )
    monkeypatch.setattr(sprite_panel.widgets, "muted", lambda *a, **k: None)
    monkeypatch.setattr(sprite_panel, "_candidate", lambda *a, **k: None)
    monkeypatch.setattr(
        sprite_panel.controls, "small_button", lambda label, **k: label == "Delete draft"
    )
    ctx = _DraftCtx()

    sprite_panel._draft(ctx, "job-1", {"id": "draft-1", "candidates": []})

    assert ctx.submitted == [], "the press deleted before anyone confirmed"
    assert len(ctx.confirms.asked) == 1
    confirm = ctx.confirms.asked[0]
    assert confirm.confirm_label == "Delete" and confirm.cancel_label == "Keep"
    confirm.on_confirm()
    assert [(key, args[1:]) for key, _fn, args in ctx.submitted] == [
        ("sprite-del:job-1:draft-1", ("job-1", "draft-1"))
    ]


# --- create-43: an in-flight reference in Create keeps a dimmed Inker exit ---


class _CreateCtx:
    def __init__(self) -> None:
        self.state = AppState()
        self.state.mode = "create"
        self.state.create.stage = "reference"
        self.state.selected = None
        self.cache = None

    def job(self):
        return None


def _reference(status: str, files: list[str]) -> dict:
    return {"id": "j", "kind": "text", "stage": "reference", "status": status, "files": files}


@pytest.mark.parametrize("status,words", [("running", "generated"), ("error", "failed")])
def test_a_running_reference_in_create_offers_a_dimmed_inker_exit_with_the_status_reason(
    status, words
):
    """At the Reference stage ``_inker`` returned None ("the toolbar owns it"),
    but the toolbar draws Open in Inker only for a finished row, so a queued,
    running or failed reference had no Inker affordance anywhere -- the 2026-10-04
    audit, finding create-43."""
    exits = asset_exits.exits_for(_CreateCtx(), _reference(status, []))
    inker = [e for e in exits if e.mode == "inker"]
    assert len(inker) == 1, [e.mode for e in exits]
    assert inker[0].reason and words in inker[0].reason.lower()


def test_a_finished_reference_in_create_still_leaves_inker_to_the_toolbar():
    """The complement rule: where the toolbar draws the lit button, an exit would
    be the second control for one action."""
    exits = asset_exits.exits_for(_CreateCtx(), _reference("done", ["input.png"]))
    assert not [e for e in exits if e.mode == "inker"]
