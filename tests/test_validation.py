"""The pack door beside the weights door (F4's job-submit half).

The mode-level gate (``model_gate.mode_gate``, ``tests/test_pack_gate.py``)
only greys a mode's rail item on a machine with no finished job at all --
deliberately, since Create's later stages act on jobs that already exist.
Once the library holds one job, Create, Muse and Poser/Troupe open with fully
live forms whose *submit* used to reach the worker and die there on a missing
import (``_q_generate.py``'s and ``_q_music.py``'s ``RuntimeError``). This file
is about the door that now stands in front of that: ``validation.check_pack``,
wired into ``validation.check_weights`` ahead of the weights checks it already
made, in the same order ``model_gate.mode_gate`` uses for the pane -- pack
first, because a pack is the code and the weights are what the code reads.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from warlock import packs as packs_mod
from warlock.service import jobs as svc_jobs
from warlock.service import validation
from warlock.service.errors import Invalid


@pytest.fixture
def _svc_stub() -> SimpleNamespace:
    """The bare shape ``check_pack``/``check_weights`` read: a ``config``
    attribute that is never actually touched by the pack half, since presence
    is ``packs.installed`` probing the running interpreter, not the config."""
    return SimpleNamespace(config=SimpleNamespace())


def _pack_missing(monkeypatch, *present: str) -> None:
    """Every pack absent except the keys named, regardless of what is
    actually importable in whatever interpreter is running the suite."""
    monkeypatch.setattr(packs_mod, "installed", lambda pack: pack.key in present)


# --- check_pack, directly ----------------------------------------------------


def test_check_pack_refuses_a_missing_pack_by_kind(monkeypatch, _svc_stub):
    _pack_missing(monkeypatch)
    with pytest.raises(Invalid) as caught:
        validation.check_pack(_svc_stub, "text", {}, field="base_model")
    assert caught.value.field == "base_model"
    assert caught.value.packs == ("text2image",)


def test_check_pack_names_the_pack_label_and_settings_not_uv(monkeypatch, _svc_stub):
    """The job-door sentence's reader is a packaged-install toast: it has
    never opened a terminal and cannot act on a ``uv sync`` line. That line is
    the worker ``RuntimeError``'s to print, for the source-checkout reader who
    is at one -- see ``_q_generate.py``/``_q_music.py``'s own comments."""
    _pack_missing(monkeypatch)
    with pytest.raises(Invalid) as caught:
        validation.check_pack(_svc_stub, "text", {}, field="base_model")
    message = str(caught.value)
    assert "Image generation" in message
    assert "Settings" in message
    assert "uv" not in message.lower()


def test_check_pack_is_a_noop_once_the_pack_is_present(monkeypatch, _svc_stub):
    _pack_missing(monkeypatch, "text2image")
    validation.check_pack(_svc_stub, "text", {}, field="base_model")  # must not raise


def test_check_pack_has_nothing_to_say_about_a_kind_with_no_pack(monkeypatch, _svc_stub):
    """``rig``, ``mesh``, ``remesh``... never touch SDXL or torch, so there is
    no pack for them to be short of -- the same rule ``check_weights`` already
    applies to every kind but ``text``/``music``/``separate``."""
    _pack_missing(monkeypatch)
    validation.check_pack(_svc_stub, "mesh", {}, field="whatever")  # must not raise


@pytest.mark.parametrize(
    ("kind", "field_name", "pack_key"),
    [
        ("text", "base_model", "text2image"),
        ("music", "music_model", "music"),
        ("separate", "separation_model", "music"),
    ],
)
def test_every_kind_check_weights_inspects_has_a_pack(
    monkeypatch, _svc_stub, kind, field_name, pack_key
):
    """``_pack_for_kind`` derives from ``Pack.modes`` rather than a hand-kept
    kind -> pack table -- this pins the answer that derivation must keep
    giving for the three kinds ``check_weights`` actually gates."""
    _pack_missing(monkeypatch)
    with pytest.raises(Invalid) as caught:
        validation.check_weights(_svc_stub, kind, {})
    assert caught.value.field == field_name
    assert caught.value.packs == (pack_key,)


# --- ordering: packs before weights -----------------------------------------


def test_packs_are_refused_before_weights_when_both_are_missing(monkeypatch, _svc_stub):
    """F4's ordering, at the job door rather than the pane: weights installed
    without their pack buy the user nothing, so the pack refusal must fire
    without ever reaching the weights check. Fails against the unfixed code,
    which called ``check_base_model_weights`` unconditionally and would trip
    the stub below before ``check_pack`` gets a chance to run at all."""
    _pack_missing(monkeypatch)

    def _boom(*_args, **_kwargs):
        raise AssertionError("weights were checked before the pack door")

    monkeypatch.setattr(validation, "check_base_model_weights", _boom)
    with pytest.raises(Invalid) as caught:
        validation.check_weights(_svc_stub, "text", {"base_model": "sdxl_cfg"})
    assert caught.value.packs == ("text2image",)


def test_once_the_pack_is_in_the_weights_door_is_reached(monkeypatch, _svc_stub):
    """The mirror of the ordering test: with the pack present, ``check_weights``
    must fall through to its ordinary weights check rather than stopping at
    the pack door forever."""
    _pack_missing(monkeypatch, "text2image")
    reached: list[bool] = []

    def _mark(*_args, **_kwargs):
        reached.append(True)

    monkeypatch.setattr(validation, "check_base_model_weights", _mark)
    validation.check_weights(_svc_stub, "text", {"base_model": "sdxl_cfg"})
    assert reached == [True]


# --- the real door: create_job ------------------------------------------------


def test_a_text_submit_is_refused_at_the_door_when_the_pack_is_missing(monkeypatch, svc):
    """The end-to-end claim: a real ``create_job`` call, through the real
    door, on a library that already has one finished job (the case
    ``model_gate.mode_gate`` deliberately lets through) -- and it must still
    be refused, because that gate only covers the pane, not the submit."""
    _pack_missing(monkeypatch)
    with pytest.raises(Invalid) as caught:
        svc_jobs.create_job(svc, kind="text", prompt="a barrel")
    assert caught.value.field == "base_model"
    assert caught.value.packs == ("text2image",)
    assert "Image generation" in str(caught.value)
    assert "Settings" in str(caught.value)
