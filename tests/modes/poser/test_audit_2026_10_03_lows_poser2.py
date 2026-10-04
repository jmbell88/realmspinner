"""The 2026-10-03 audit's Low findings owned by fixer ``poser2``.

One file for the whole slice (characters, rig, service.clips, Poser's mode and
panes) because the brief asks for a single new test file per fixer; each test's
name is the claim it pins.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from realmspinner.characters import family as family_mod
from realmspinner.characters.errors import CharacterError
from realmspinner.characters.recipe import Recipe
from realmspinner.characters.resolve import THEME_WORDS, resolve
from realmspinner.kernels.rig import cliplib, skeleton, store, templates

# --- poser-23 -----------------------------------------------------------------


def test_every_look_a_species_declares_can_be_asked_for_in_words():
    """Ten shipped looks (albino, azure, black, dapple, pale, panther, storm,
    tar, venom, winter) had no word in THEME_WORDS, so a brief could never
    name them: the Create column fills the look from the words."""
    unreachable = []
    for fam_key, fam in family_mod.families().items():
        species_word = fam.aliases[0]
        for theme in fam.themes:
            if theme.key == "natural":
                continue
            words = THEME_WORDS.get(theme.key, ())
            reached = any(
                resolve(f"{word} {species_word}").theme == theme.key
                and resolve(f"{word} {species_word}").family == fam_key
                for word in words
            )
            if not reached:
                unreachable.append(f"{fam_key}/{theme.key}")
    assert not unreachable, unreachable


def test_black_wolf_asks_for_the_wolfs_own_black_look_not_blackened():
    """``black wolf`` resolved the unrelated ``blackened`` look and was dropped
    as "Wolf has no 'blackened' look"; the same word on a knight still means
    blackened steel."""
    assert resolve("black wolf").theme == "black"
    assert resolve("black knight").theme == "blackened"


# --- poser-24 -----------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        {"elevation": True},
        {"appearance": {"bulk": True}},
        {"pixel_art": "false"},
        {"pixel_art": 0},
        {"family_version": 0},
        {"family_version": -3},
    ],
)
def test_a_boolean_or_a_string_is_never_read_as_a_number_or_a_switch_by_the_recipe(raw):
    """``Recipe.from_dict`` refuses and never clamps: ``True`` was one degree of
    elevation, ``"false"`` was a pixel-art switch left on and version 0 or -3
    named no shipped version."""
    if "appearance" in raw:
        channel = family_mod.get_family("ogre").channels[0].key
        raw = {"appearance": {channel: True}}
    with pytest.raises(CharacterError):
        Recipe.from_dict(raw)


def test_a_real_boolean_still_sets_pixel_art_and_a_real_number_still_sets_elevation():
    assert Recipe.from_dict({"pixel_art": False}).pixel_art is False
    assert Recipe.from_dict({"pixel_art": True}).pixel_art is True
    assert Recipe.from_dict({"elevation": 30}).elevation == 30.0
    assert Recipe.from_dict({}).family_version == family_mod.get_family("ogre").version


# --- poser-37 -----------------------------------------------------------------


def test_check_registry_refuses_a_theme_that_misses_a_region():
    """``_theme`` takes ``**hexes`` so nothing checked regions at import, and
    ``_check_registry`` only looked at channel keys: a theme with a hole was
    caught by a test and by no import."""
    fam = family_mod.get_family("ogre")
    theme = fam.themes[0]
    region = fam.regions[0]
    holed = dataclasses.replace(
        theme, materials={k: v for k, v in theme.materials.items() if k != region}
    )
    broken = dataclasses.replace(fam, themes=(holed, *fam.themes[1:]))
    with pytest.raises(CharacterError, match=region):
        family_mod._check_registry({"ogre": broken})
    misspelt = dataclasses.replace(theme, materials={**theme.materials, "skni": "#000000"})
    broken = dataclasses.replace(fam, themes=(misspelt, *fam.themes[1:]))
    with pytest.raises(CharacterError, match="skni"):
        family_mod._check_registry({"ogre": broken})
    family_mod._check_registry({"ogre": fam})  # the shipped row stays fine


# --- poser-32 / 33 ------------------------------------------------------------


def test_delete_pose_does_not_choke_on_a_directory_standing_in_for_the_record(tmp_path):
    """A directory squatting at ``poses/<id>.json`` read as found and
    ``unlink`` raised PermissionError on Windows."""
    pose_id = "abc123456789"
    squatter = store.pose_path(tmp_path, pose_id)
    squatter.mkdir(parents=True)
    assert store.delete_pose(tmp_path, pose_id) is False
    assert squatter.is_dir()


def _stage_rig_temps(job_dir: Path) -> None:
    (job_dir / store.RIG_GLB_TMP).write_bytes(b"new-rig")
    (job_dir / store.RIG_JSON_TMP).write_text("{}", encoding="utf-8")


def test_finalize_rig_survives_a_reader_holding_a_stale_bake_open(tmp_path, monkeypatch):
    """The stale-bake unlinks sat outside the PermissionError retry that wraps
    the renames, so a reader holding ``animated.glb`` or a pose bake open threw
    away a minutes-long solve before any rename."""
    poses = tmp_path / store.POSE_DIR_NAME
    poses.mkdir()
    (poses / "abc123456789.glb").write_bytes(b"stale bake")
    (tmp_path / "animated.glb").write_bytes(b"stale clip bake")
    _stage_rig_temps(tmp_path)
    monkeypatch.setattr(store.time, "sleep", lambda _s: None)

    real_unlink = Path.unlink
    held = {"animated.glb", "abc123456789.glb"}
    attempts: dict[str, int] = {}

    def unlink(self, missing_ok=False):
        if self.name in held:
            attempts[self.name] = attempts.get(self.name, 0) + 1
            raise PermissionError(13, "in use", str(self))
        return real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", unlink)
    store.finalize_rig(tmp_path)  # must not raise

    assert (tmp_path / "rig.glb").read_bytes() == b"new-rig"
    assert (tmp_path / "rig.json").exists()
    assert attempts["animated.glb"] > 1  # it retried before giving way


def test_finalize_rig_retries_a_stale_bake_until_the_reader_lets_go(tmp_path, monkeypatch):
    (tmp_path / "animated.glb").write_bytes(b"stale clip bake")
    _stage_rig_temps(tmp_path)
    monkeypatch.setattr(store.time, "sleep", lambda _s: None)
    real_unlink = Path.unlink
    calls = {"n": 0}

    def unlink(self, missing_ok=False):
        if self.name == "animated.glb":
            calls["n"] += 1
            if calls["n"] < 3:
                raise PermissionError(13, "in use", str(self))
        return real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", unlink)
    store.finalize_rig(tmp_path)
    assert not (tmp_path / "animated.glb").exists()


# --- poser-34 -----------------------------------------------------------------


def _library(*, segments, version=3, closed=False):
    pose = {"name": "a", "bones": {}}
    return {
        "version": version,
        "poses": [pose, {"name": "b", "bones": {}}],
        "clips": [
            {
                "name": "c",
                "keys": ["a", "b"],
                "segments": segments,
                "closed": closed,
                "duration_ms": 100,
            }
        ],
    }


@pytest.mark.parametrize("bad", [2.9, "8", True])
def test_parse_clip_library_refuses_a_fractional_or_string_segment_length(bad):
    """``int(n)`` read 2.9 as 2, "8" as 8 and true as 1 where the write door
    (``service.clips._segment_length``) refuses all three."""
    with pytest.raises(ValueError, match="segment must be a whole number"):
        cliplib.parse_clip_library(_library(segments=[bad]))


@pytest.mark.parametrize("bad", [3.9, "3", True])
def test_parse_clip_library_refuses_a_fractional_or_string_version(bad):
    with pytest.raises(ValueError, match="version must be a whole number"):
        cliplib.parse_clip_library(_library(segments=[4], version=bad))


def test_parse_clip_library_still_reads_whole_numbers():
    out = cliplib.parse_clip_library(_library(segments=[4]))
    assert out["clips"][0]["segments"] == [4]


# --- poser-35 -----------------------------------------------------------------


def _fitted_payload():
    template = templates.get_template("humanoid")
    fitted = skeleton.fit_template(template, [-1, -1, 0], [1, 1, 2])
    return template, [dict(name=b["name"], head=b["head"], tail=b["tail"]) for b in fitted]


def test_validate_skeleton_refuses_a_tail_far_outside_the_mesh():
    """Only each bone's head was box-checked, so a tail at 1e9 m built a
    skeleton kilometres across."""
    base = templates.get_template("humanoid")
    fitted = skeleton.fit_template(base, [-1, -1, 0], [1, 1, 2])
    bones = [dict(b) for b in fitted]
    bones[1]["tail"] = [0.0, 0.0, 1e9]
    with pytest.raises(store.RigError, match="tail is far outside") as exc:
        skeleton.validate_skeleton(
            {"bones": bones}, base=base, bounds={"min": [-1, -1, 0], "max": [1, 1, 2]}
        )
    assert exc.value.field == "bones"


def test_validate_joints_names_the_far_joint_not_a_zero_length_bone():
    """One head at 1e9 m inflated the head span and was refused as "bone 'hips'
    would be zero-length"; with the rig's bounds in hand the far joint is named."""
    template, bones = _fitted_payload()
    bones[3]["head"] = [0.0, 0.0, 1e9]
    bounds = {"min": [-1, -1, 0], "max": [1, 1, 2]}
    with pytest.raises(ValueError, match=rf"{bones[3]['name']}.*far outside"):
        skeleton.validate_joints({"bones": bones}, template, bounds=bounds)


def test_validate_joints_refuses_a_far_tail_when_the_rigs_bounds_are_given():
    template, bones = _fitted_payload()
    bones[3]["tail"] = [0.0, 0.0, 1e9]
    bounds = {"min": [-1, -1, 0], "max": [1, 1, 2]}
    with pytest.raises(ValueError, match="tail is far outside"):
        skeleton.validate_joints({"bones": bones}, template, bounds=bounds)


def test_validate_joints_without_bounds_behaves_as_it_always_did():
    template, bones = _fitted_payload()
    assert skeleton.validate_joints({"bones": bones}, template)


# --- poser-49 / poser-54 ------------------------------------------------------


def _clip_payload(svc):
    from realmspinner.service import clips as svc_clips

    view = svc_clips.library(svc, "humanoid")
    return svc_clips, {"space": view["space"], "poses": view["poses"], "clips": view["clips"]}


@pytest.mark.parametrize("bad", ["false", "true", 0, 1, None])
def test_clip_save_refuses_a_string_typed_closed(svc, bad):
    """``bool("false")`` is True, so a clip meant open was saved looping and the
    parser's own refusal never fired (it was handed a real bool)."""
    from realmspinner.service import Invalid

    svc_clips, payload = _clip_payload(svc)
    payload["clips"][0]["closed"] = bad
    with pytest.raises(Invalid) as caught:
        svc_clips.save(svc, "humanoid", payload)
    assert caught.value.field == "closed"


def test_save_refuses_a_non_boolean_closed_flag(svc):
    from realmspinner.service import Invalid

    svc_clips, payload = _clip_payload(svc)
    payload["clips"][0]["closed"] = "false"
    with pytest.raises(Invalid, match="closed"):
        svc_clips.save(svc, "humanoid", payload)


@pytest.mark.parametrize("bad", [0, "", {}, "0,1,2"])
def test_save_refuses_a_present_non_list_root_translation(svc, bad):
    """A falsy-but-present ``root_translation`` was skipped as if absent."""
    from realmspinner.service import Invalid

    svc_clips, payload = _clip_payload(svc)
    payload["poses"][0]["root_translation"] = bad
    with pytest.raises(Invalid) as caught:
        svc_clips.save(svc, "humanoid", payload)
    assert caught.value.field == "poses"


def test_save_still_accepts_a_real_boolean_closed_and_a_null_root_translation(svc):
    svc_clips, payload = _clip_payload(svc)
    payload["poses"][0]["root_translation"] = None
    svc_clips._check_shape(payload)


# --- poser-43 -----------------------------------------------------------------


def _save_buttons_drawn(monkeypatch, *, current, library_ids):
    from types import SimpleNamespace

    from realmspinner.studio import widgets
    from realmspinner.studio.modes.poser import mode as poser_mode
    from realmspinner.studio.modes.poser.ui.panes import controls

    state = poser_mode.PoserState()
    state.template = "humanoid"
    state.poses = [{"id": i, "name": i} for i in library_ids]
    ctx = SimpleNamespace(
        state=SimpleNamespace(poser=state), busy=lambda _key: False, rig_default="humanoid"
    )
    viewer = SimpleNamespace(editor=SimpleNamespace(current=current))
    drawn: list[str] = []

    def fake_button(label, *_a, **_k):
        drawn.append(label)
        return False

    monkeypatch.setattr(widgets, "disabled_button", fake_button)
    controls._save_library(ctx, viewer)
    return drawn


def test_save_button_is_offered_only_when_the_current_pose_is_a_library_record(monkeypatch):
    """In an asset session ``editor.current`` can name one of the asset's own
    poses; "Save" ("Overwrite the pose named above, in the shared library")
    then opened a name prompt and made a new library pose."""
    assert "Save" not in _save_buttons_drawn(
        monkeypatch, current="asset-pose", library_ids=["lib-1"]
    )
    assert "Save" in _save_buttons_drawn(monkeypatch, current="lib-1", library_ids=["lib-1"])
    assert "Save as reusable pose..." in _save_buttons_drawn(
        monkeypatch, current="asset-pose", library_ids=[]
    )


# --- poser-44 -----------------------------------------------------------------


def test_clips_empty_state_does_not_claim_only_humanoid_has_clips(monkeypatch):
    """The empty state named the humanoid as the one template with clips while
    four ship them; it now reads the sentence off the shipped libraries."""
    from types import SimpleNamespace

    from realmspinner.studio import widgets
    from realmspinner.studio.modes.poser import mode as poser_mode
    from realmspinner.studio.modes.poser.ui.panes import clips as clips_pane

    state = poser_mode.PoserState()
    state.template = "fish"
    state.clips = {"clips": []}
    state.clips_loading = False
    ctx = SimpleNamespace(state=SimpleNamespace(poser=state), rigging_available=True)
    said: list[str] = []
    monkeypatch.setattr(widgets, "section", lambda *_a, **_k: None)
    monkeypatch.setattr(widgets, "muted", lambda text, *_a, **_k: said.append(text))
    monkeypatch.setattr(widgets, "muted_wrapped", lambda text, *_a, **_k: said.append(text))
    monkeypatch.setattr(clips_pane.manual_render, "help_button", lambda *_a, **_k: None)
    monkeypatch.setattr(clips_pane, "_import_button", lambda *_a, **_k: None)
    monkeypatch.setattr(poser_mode, "clips_pump", lambda *_a, **_k: None)

    clips_pane.draw(ctx)

    text = " ".join(said)
    assert "only the humanoid template" not in text
    for key in cliplib.shipped_clip_templates():
        assert key in text, key


# --- poser-45 / poser-47 ------------------------------------------------------


def _opened(svc, monkeypatch):
    from modes.poser.test_poser_mode import FakeCtx, FakeViewer, _fake_blender, _rigged_job
    from realmspinner.studio.modes.poser import mode as poser_mode

    _fake_blender(monkeypatch)
    job_id = _rigged_job(svc)
    ctx = FakeCtx(svc)
    ctx.poser_viewer = viewer = FakeViewer()
    poser_mode.open_asset(ctx, {"id": job_id, "name": "Prop"})
    poser_mode.sync_asset(ctx, viewer)
    return ctx, viewer, job_id


def test_a_template_switch_clears_onion_ghosts(svc, monkeypatch):
    """``state.onion`` and ``viewer.onion`` were never reset, so the previous
    clip's neighbouring keys drew as ghosts over the next skeleton and, with no
    clips there, the checkbox that turns them off was not drawn."""
    from realmspinner.studio.modes.poser import mode as poser_mode

    ctx, viewer, _job_id = _opened(svc, monkeypatch)
    state = poser_mode.ensure(ctx)
    state.onion = True
    viewer.onion = [{"hips": [0, 0, 0, 1]}, {}]

    poser_mode.set_template(ctx, "quadruped")

    assert state.onion is False
    assert viewer.onion == []


def test_closing_or_opening_an_asset_clears_onion_ghosts(svc, monkeypatch):
    from realmspinner.studio.modes.poser import mode as poser_mode

    ctx, viewer, job_id = _opened(svc, monkeypatch)
    state = poser_mode.ensure(ctx)
    state.onion = True
    viewer.onion = [{"hips": [0, 0, 0, 1]}, {}]
    poser_mode.close_asset(ctx)
    assert state.onion is False
    assert viewer.onion == []

    state.onion = True
    viewer.onion = [{}, {}]
    poser_mode.open_asset(ctx, {"id": job_id, "name": "Prop"})
    assert state.onion is False
    assert viewer.onion == []


def test_viewer_exit_pose_mode_empties_the_onion_ghosts():
    """The real viewer's own ``exit_pose_mode`` is the one door every session
    ending passes (``clear``, ``adopt_model``): the ghosts go with it."""
    from realmspinner.studio import _viewer_pose

    class Stub:
        pass

    stub = Stub()
    stub.onion = [{"a": [0, 0, 0, 1]}]
    stub._render_dirty = False
    stub.rotate_gizmo = type("G", (), {"end_drag": lambda self: None})()
    stub.translate_gizmo = type("G", (), {"end_drag": lambda self: None})()
    stub._close_pose_step = lambda: None
    stub.editor = type("E", (), {"clear": lambda self: None})()
    stub._notify_pose_dirty = lambda: None
    stub._grab = None
    _viewer_pose.PoseOps.exit_pose_mode(stub)
    assert stub.onion == []


def test_declining_the_land_confirm_leaves_a_way_to_bind_the_new_rig(svc, monkeypatch):
    """Declining the "land this re-rig" confirm dropped the job from
    ``rerig_jobs`` before asking, so nothing ever rebound the session to the new
    rig; "Load new rig" now offers the same landing again."""
    from types import SimpleNamespace

    from realmspinner.studio.modes.poser import mode as poser_mode

    ctx, viewer, job_id = _opened(svc, monkeypatch)
    poser_mode.rerig(ctx, "humanoid")
    key = f"{poser_mode.ASSET_RERIG_KEY_PREFIX}{job_id}"
    result = ctx.results[key]
    poser_mode.on_task_done(ctx, SimpleNamespace(key=key, result=result))
    state = poser_mode.ensure(ctx)
    viewer.editor.dirty = True  # an unsaved pose edit: the landing will ask

    ctx.jobs[result["id"]] = {"id": result["id"], "status": "done"}
    poser_mode.pump_rerig(ctx)
    assert len(ctx.confirms.asked) == 1  # ... and the user declines: no on_confirm
    assert viewer.pose_mode is True
    assert job_id in state.rerig_ready, "something must still say a new rig is waiting"

    poser_mode.pump_rerig(ctx)
    assert len(ctx.confirms.asked) == 1, "the landed job is not asked about every frame"

    poser_mode.load_new_rig(ctx)
    assert len(ctx.confirms.asked) == 2
    ctx.confirms.asked[1].on_confirm()
    assert viewer.pose_mode is False, "confirming lands the new rig"
    assert job_id not in state.rerig_ready


# --- poser-46 -----------------------------------------------------------------


def _quit_app(*, shared_dirty, poser_dirty):
    from types import SimpleNamespace

    from realmspinner.studio.shell.quit import QuitMixin

    def viewer(dirty):
        return SimpleNamespace(
            pose_mode=True, editor=SimpleNamespace(has_unsaved_edits=lambda: dirty)
        )

    app = object.__new__(QuitMixin)
    app._title_marked = None
    app._sync_title = lambda: None
    app.viewer = viewer(shared_dirty)
    app.poser_viewer = viewer(poser_dirty)
    app.app_ctx = SimpleNamespace(state=SimpleNamespace(pose_dirty=False))
    return app


def test_shared_viewer_adopt_does_not_clear_poser_dirty_mark():
    """Both viewers report into one ``AppState.pose_dirty``; the shared viewer
    adopting a model reports "clean" and used to clear the mark over a Poser
    pose that is still unsaved."""
    app = _quit_app(shared_dirty=False, poser_dirty=True)
    app._on_pose_dirty(True)  # Poser's edit
    assert app.app_ctx.state.pose_dirty is True
    app._on_pose_dirty(False)  # the shared viewer's adopt_model: "clean"
    assert app.app_ctx.state.pose_dirty is True, "Poser's pose is still unsaved"


def test_the_mark_clears_only_when_neither_editor_holds_unsaved_edits():
    app = _quit_app(shared_dirty=False, poser_dirty=False)
    app.app_ctx.state.pose_dirty = True
    app._on_pose_dirty(False)
    assert app.app_ctx.state.pose_dirty is False


def test_poser_status_label_ignores_the_inspectors_unsaved_edit(svc, monkeypatch):
    """An inspector edit marked Poser's status label dirty through the shared
    mirror; Poser's label asks its own editor."""
    from realmspinner.studio.modes.poser import mode as poser_mode

    ctx, viewer, _job_id = _opened(svc, monkeypatch)
    ctx.state.pose_dirty = True  # the inspector's viewer is the one that is dirty
    viewer.editor.dirty = False
    viewer.editor.moved.clear()
    label = poser_mode.document_label(ctx)
    assert label is not None and label[1] is False
    viewer.editor.dirty = True
    assert poser_mode.document_label(ctx)[1] is True


# --- poser-50 -----------------------------------------------------------------


def test_the_send_dialog_does_not_forward_the_poser_forms_sheet_name(svc, monkeypatch):
    """``name``, ``dither`` and ``reduce_mode`` belong to the "Build a new
    sheet" section for the bound character; the Send dialog neither shows nor
    asks them, yet submitted them with an unrelated mesh."""
    from modes.poser.test_send_door import _capture_sends, _Ctx, _mesh
    from realmspinner.studio.modes.poser import mode as poser_mode
    from realmspinner.studio.modes.poser.ui.panes import send as poser_send

    ctx = _Ctx(svc)
    sent = _capture_sends(monkeypatch)
    bound = _mesh(svc, rigged=True)
    state = poser_mode.ensure(ctx)
    state.job_id = bound["id"]
    state.template = "humanoid"
    form = poser_mode.sheet_form(ctx)
    form["name"] = "Hero's next sheet"
    form["dither"] = True
    form["reduce_mode"] = "median"

    other = _mesh(svc, rigged=True)
    poser_send.ask(ctx, other)
    poser_send._send(ctx, ctx.state.poser_send, form)

    assert sent[-1]["name"] == ""
    assert sent[-1]["dither"] is False
    assert sent[-1]["reduce_mode"] != "median"
    # The standing form is the bound character's, and is left exactly as typed.
    assert form["name"] == "Hero's next sheet"
    assert form["dither"] is True


def test_the_send_dialog_still_forwards_the_bound_characters_own_answers(svc, monkeypatch):
    from modes.poser.test_send_door import _capture_sends, _Ctx, _mesh
    from realmspinner.studio.modes.poser import mode as poser_mode
    from realmspinner.studio.modes.poser.ui.panes import send as poser_send

    ctx = _Ctx(svc)
    sent = _capture_sends(monkeypatch)
    bound = _mesh(svc, rigged=True)
    state = poser_mode.ensure(ctx)
    state.job_id = bound["id"]
    state.template = "humanoid"
    form = poser_mode.sheet_form(ctx)
    form["name"] = "Hero's next sheet"
    poser_send.ask(ctx, bound)
    poser_send._send(ctx, ctx.state.poser_send, form)
    assert sent[-1]["name"] == "Hero's next sheet"
