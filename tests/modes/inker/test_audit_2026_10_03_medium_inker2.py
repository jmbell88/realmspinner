"""The 2026-10-03 audit's Medium Flourish findings, inker-35 .. inker-39 and
inker-61 (inker-40 needs a measurement document first and has no test here).

Each test's name is the claim, and each fails against the code as it stood
before the fix.
"""

from __future__ import annotations

import dataclasses
import json
import zipfile
from types import SimpleNamespace

import pytest

from realmspinner.kernels import pixel as inker
from realmspinner.kernels.pixel import ora
from realmspinner.kernels.pixel.flourish import bake as B
from realmspinner.kernels.pixel.flourish import engines, keywords, presets, prims
from realmspinner.kernels.pixel.flourish.curves import Curve
from realmspinner.studio.modes.inker import flourish as inker_flourish
from realmspinner.studio.modes.inker import mode as inker_mode
from realmspinner.studio.modes.inker import state as inker_state
from realmspinner.studio.tasks import Done


class _Ctx:
    """Tasks run inline; with ``auto_land`` off the result waits for ``land_all``."""

    def __init__(self) -> None:
        self.state = SimpleNamespace(inker=inker_state.InkerState())
        self.toasts: list[tuple[str, str]] = []
        self.tasks = SimpleNamespace(set_progress=lambda *a, **k: None)
        self.pending: list[Done] = []
        self.auto_land = True
        self._busy: set[str] = set()
        self.results: list[dict] = []

    def toast(self, text: str, level: str = "info", **_: object) -> None:
        self.toasts.append((text, level))

    def busy(self, key: str) -> bool:
        return key in self._busy

    def progress(self, key: str):
        return None

    def submit(self, key: str, fn, *args, **kwargs) -> bool:
        if key in self._busy:
            return False
        try:
            done = Done(key=key, result=fn(*args, **kwargs))
        except Exception as exc:  # noqa: BLE001
            done = Done(key=key, error=exc)
        if isinstance(done.result, dict):
            self.results.append(done.result)
        if self.auto_land:
            inker_mode.on_task_done(self, done)
        else:
            self._busy.add(key)
            self.pending.append(done)
        return True

    def land_all(self) -> None:
        while self.pending:
            done = self.pending.pop(0)
            self._busy.discard(done.key)
            inker_mode.on_task_done(self, done)


def _open(ctx: _Ctx, size=(32, 32)) -> inker_state.InkerDoc:
    tab = inker_state.InkerDoc(doc=inker.Document.blank(*size), title="spell.ora")
    ctx.state.inker.docs.append(tab)
    ctx.state.inker.active_uid = tab.uid
    return tab


def _small(name: str = "smoke_puff"):
    return dataclasses.replace(presets.load(name), width=32, height=32, supersample=2)


# -- inker-35 ---------------------------------------------------------------------


@pytest.mark.parametrize("edited", [False, True])
def test_a_regenerate_that_lands_on_a_busy_tab_is_retried_with_its_force_flag_or_reported(edited):
    ctx = _Ctx()
    state = ctx.state.inker
    tab = _open(ctx)
    rec = _small()
    group = tab.doc.insert_flourish(B.bake(rec))
    if edited:
        inker_flourish.set_pending(
            state, group, dataclasses.replace(rec, seed=99), now=inker_flourish.clock() - 10
        )
        state.flourish_due.clear()  # the debounce already went out with the press below
    ctx.auto_land = False
    assert inker_mode.flourish_regenerate(ctx, tab, force=True)
    assert len(ctx.pending) == 1
    # The bake finishes after playback started.
    tab.playing = True
    before = len(ctx.toasts)
    ctx.land_all()
    tab.playing = False

    ctx.auto_land = True
    sent = inker_flourish.tick(ctx, state, tab, now=inker_flourish.clock() + 5.0)
    reported = len(ctx.toasts) > before
    assert (sent == 1 and ctx.results[-1]["force"] is True) or reported, (
        "the press was dropped with no toast, or retried without its force flag"
    )
    assert sent == 1 and ctx.results[-1]["force"] is True


# -- inker-36 ---------------------------------------------------------------------


def test_a_pending_edit_in_one_tab_is_not_discarded_when_another_tab_ticks():
    ctx = _Ctx()
    state = ctx.state.inker
    tab_a = _open(ctx)
    rec = _small()
    group = tab_a.doc.insert_flourish(B.bake(rec))
    tab_b = _open(ctx)  # now in front
    edited = dataclasses.replace(rec, seed=123)
    inker_flourish.set_pending(state, group, edited, now=0.0)

    sent = inker_flourish.tick(ctx, state, tab_b, now=1.0)

    assert sent == 1, "the edit staged in tab A was thrown away when tab B ticked"
    assert tab_a.doc.flourish_state(group).recipe == edited
    assert group not in state.flourish_pending


def test_a_pending_edit_waits_for_its_own_tab_when_that_tab_is_busy():
    ctx = _Ctx()
    state = ctx.state.inker
    tab_a = _open(ctx)
    rec = _small()
    group = tab_a.doc.insert_flourish(B.bake(rec))
    tab_b = _open(ctx)
    inker_flourish.set_pending(state, group, dataclasses.replace(rec, seed=5), now=0.0)
    tab_a.playing = True

    assert inker_flourish.tick(ctx, state, tab_b, now=1.0) == 0
    assert group in state.flourish_pending and group in state.flourish_due


def test_a_group_no_open_tab_owns_is_still_dropped_by_tick():
    ctx = _Ctx()
    state = ctx.state.inker
    tab = _open(ctx)
    inker_flourish.set_pending(state, 424242, _small(), now=0.0)
    assert inker_flourish.tick(ctx, state, tab, now=1.0) == 0
    assert 424242 not in state.flourish_pending and 424242 not in state.flourish_due


# -- inker-37 ---------------------------------------------------------------------

HUGE = 10**400


def test_param_clamp_survives_a_non_finite_or_huge_number():
    count = prims.params_of("particles")["count"]
    assert count.clamp(float("inf")) == count.default
    assert count.clamp(HUGE) == count.hi  # an integer too large for float still clamps
    size = prims.params_of("particles")["size"]
    assert size.clamp(HUGE) == size.default
    life = prims.params_of("particles")["size_over_life"]
    assert life.clamp({"keys": [[0.0, HUGE], [1.0, 1.0]]}) == life.clamp(life.default)
    assert life.clamp(HUGE) == life.clamp(life.default)


def test_a_curve_key_time_that_is_not_finite_is_clamped_to_zero_to_one_on_load():
    curve = Curve.from_json({"keys": [[float("nan"), 1.0], [float("inf"), 2.0], [-5, 3.0]]})
    assert all(0.0 <= t <= 1.0 for t, _ in curve.keys)
    curve.sample([0.0, 0.5, 1.0])  # the renderer's call must not raise


def test_apply_diff_names_a_number_it_cannot_hold_instead_of_raising():
    rec = presets.load("fireball")
    layer = next(each for each in rec.layers if each.kind == "particles")
    inf = float("inf")
    diff = {
        "layers": {
            str(layer.uid): {"count": inf, "size": HUGE, "opacity": HUGE},
        },
        "phases": {rec.phases[0].name: {"frames": inf}},
        "seed": inf,
        "fps": HUGE,
    }
    out, notes = keywords.apply_diff(rec, diff)
    assert out.layers[rec.layers.index(layer)].params.get("count") == layer.params.get("count")
    assert notes


def test_an_ora_with_an_infinite_flourish_parameter_opens_without_its_recipe(tmp_path):
    doc = inker.Document.blank(32, 32)
    rec = _small()
    group = doc.insert_flourish(B.bake(rec))
    path = tmp_path / "puff.ora"
    ora.write_ora(doc, path)
    good = inker.Document.load(path)
    assert len(good.flourish) == 1 and group is not None

    # Rewrite animation.json with an int parameter that is JSON Infinity.
    member = "animation.json"
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        assert member in names
        payload = json.loads(zf.read(member).decode("utf-8"))
        others = {n: zf.read(n) for n in names if n != member}
        infos = {n: zf.getinfo(n).compress_type for n in names}
    layers = payload["flourish"][0]["recipe"]["layers"]
    target = next(
        each for each in layers if "count" in prims.params_of(each["kind"])
    )
    target.setdefault("params", {})["count"] = float("inf")
    bad = tmp_path / "bad.ora"
    with zipfile.ZipFile(bad, "w") as out:
        for name in names:
            data = json.dumps(payload).encode("utf-8") if name == member else others[name]
            out.writestr(name, data, compress_type=infos[name])

    opened = inker.Document.load(bad)  # must not raise
    assert opened.size == (32, 32)
    assert len(opened.anim.frames) == len(good.anim.frames)


# -- inker-38 ---------------------------------------------------------------------


def _tagged_scene():
    ctx = _Ctx()
    tab = _open(ctx)
    rec = dataclasses.replace(presets.load("sword_impact"), width=32, height=32, supersample=2)
    tab.doc.insert_flourish(B.bake(rec))
    return ctx, tab


def test_the_snippet_popup_names_a_tag_it_cannot_make_a_filename_from_instead_of_raising():
    ctx, tab = _tagged_scene()
    tab.doc.anim.tags[0].name = "攻撃"  # the Japanese word for "attack"
    name = tab.doc.anim.tags[0].name
    for engine in engines.ENGINES:
        assert inker_flourish.snippet_text(tab, name, engine) == ""
    problem = inker_flourish.snippet_problem(tab, name)
    assert name in problem
    # A tag that does have a file name has no problem to name.
    other = tab.doc.anim.tags[1].name
    assert inker_flourish.snippet_problem(tab, other) == ""


# -- inker-39 ---------------------------------------------------------------------


def test_apply_diff_leaves_a_tuned_value_alone_when_the_models_value_is_unparseable():
    rec = presets.load("fireball")
    layer = next(each for each in rec.layers if each.kind == "particles")
    tuned = rec.replace_layer(layer.with_param("count", 300).with_param("color_start", "#112233"))
    key = str(layer.uid)
    out, notes = keywords.apply_diff(
        tuned,
        {"layers": {key: {"count": "lots", "color_start": "#FFF", "emission": "sideways"}}},
    )
    after = out.layer(layer.uid)
    assert after.params["count"] == 300
    assert after.params["color_start"] == "#112233"
    assert after.params["emission"] == tuned.layer(layer.uid).params.get(
        "emission", prims.params_of("particles")["emission"].default
    )
    joined = " ".join(notes)
    assert "count" in joined and "color_start" in joined and "emission" in joined
    # Not reported as an ordinary change.
    assert f"{layer.name}: count" not in notes


# -- inker-61 ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("arrange", "wrap", "expected"),
    [
        (None, 4, None),  # the default row-wrap
        ("horizontal", 4, "frames"),
        ("vertical", 4, 1),
        ("columns", 2, 2),
        ("rows", 2, "half"),
    ],
)
def test_the_engine_snippet_columns_follow_the_chosen_export_arrange(arrange, wrap, expected):
    ctx = _Ctx()
    tab = _open(ctx)
    tab.doc.insert_flourish(B.bake(_small("fireball")))
    anim = tab.doc.anim
    name = max(anim.tags, key=lambda t: t.end - t.start).name
    default = inker_flourish.snippet_info(tab, name)
    frames = default["frames"]
    assert frames >= 4  # the fixture, not the claim
    info = inker_flourish.snippet_info(tab, name, arrange=arrange, wrap=wrap)
    want = {None: default["columns"], "frames": frames, "half": -(-frames // 2)}.get(
        expected, expected
    )
    assert info["columns"] == want
    assert info["rows"] == -(-frames // want)
    text = inker_flourish.snippet_text(tab, name, "pygame-ce", arrange=arrange, wrap=wrap)
    assert f"columns={want}," in text
