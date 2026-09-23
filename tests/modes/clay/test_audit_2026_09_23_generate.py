"""Regressions for the 2026-09-23 audit's Clay-Generate findings.

clay-03 (a document past the reopen ceiling can still be saved, and the
ceiling ``land()`` checks against is never refreshed after a landing),
clay-04 (Cancel during the "landing" stage does not stop the decode already
in flight from landing anyway), clay-13 (``merge_into`` wraps a single-root
hierarchy in a group it should not), clay-14/clay-15 (the popup's two submit
buttons do not name "Saving..." as their disabled reason), clay-16 (a texture
pick can land on a material that shifted into its slot) and docs-02 (the
composition-doubt reasons never reach the popup, only a one-shot toast).

The harness is ``tests/modes/clay/test_clay_generate.py``'s own ``_Ctx``
(read, not imported: this fixer owns only a new test file, and duplicating a
dozen lines here is cheaper than coupling to a module owned by another
fixer's return).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import merge, serialize
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.service import jobs as svc_jobs
from realmspinner.service.errors import TooLarge
from realmspinner.service.files import MAX_CLAY_SOURCE_BYTES
from realmspinner.studio.modes.clay import generate as clay_generate
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay.ui.panes import bridge as clay_bridge
from realmspinner.studio.modes.clay.ui.panes import props as clay_props
from realmspinner.studio.state import DEFAULT_FORM_3D, default_form_2d
from realmspinner.studio.tasks import Done

# --- shared harness (test_clay_generate.py's own _Ctx shape) -----------------


class _Store:
    def __init__(self) -> None:
        self.jobs: dict[str, dict] = {}

    def get(self, job_id: str) -> dict | None:
        return self.jobs.get(job_id)


class _Svc:
    def __init__(self, root: Path) -> None:
        self.store = _Store()
        self.root = root
        self.config = None

    def job_dir(self, job_id: str) -> Path:
        return self.root / job_id


class _Confirms:
    def __init__(self) -> None:
        self.pending: Any = None

    def ask(self, confirm: Any) -> None:
        self.pending = confirm


class _AppState:
    def __init__(self) -> None:
        self.clay = None
        self.mode = "home"
        self.form_2d = default_form_2d()
        self.form_3d = dict(DEFAULT_FORM_3D)


class _Settings:
    def __init__(self) -> None:
        self.store: dict[str, Any] = {}

    def get(self, key: str) -> Any:
        return self.store.get(key)

    def set(self, key: str, value: Any) -> None:
        self.store[key] = value


class _Ctx:
    def __init__(self, tmp_path: Path) -> None:
        self.svc = _Svc(tmp_path)
        self.state = _AppState()
        self.settings = _Settings()
        self.toasts: list[tuple[str, str]] = []
        self.confirms = _Confirms()
        self.submitted: list[str] = []
        self._busy: set[str] = set()
        self._queue: list[Done] = []
        self.clay_view = None

    def toast(self, message: str, kind: str = "info", action: Any = None) -> None:
        self.toasts.append((message, kind))

    def busy(self, key: str) -> bool:
        return key in self._busy

    def submit(self, key: str, fn: Any, *args: Any, tag: Any = None, **kwargs: Any) -> bool:
        if key in self._busy:
            return False
        self.submitted.append(key)
        try:
            done = Done(key=key, result=fn(*args, **kwargs), tag=tag)
        except Exception as exc:  # noqa: BLE001 - the same failure a real pool reports
            done = Done(key=key, error=exc, message=str(exc), tag=tag)
        self._busy.add(key)
        self._queue.append(done)
        return True

    def land_all(self) -> None:
        while self._queue:
            done = self._queue.pop(0)
            self._busy.discard(done.key)
            if done.ok:
                clay_mode.on_task_done(self, done)
            else:
                clay_mode.on_task_failed(self, done)


def _tab(ctx: Any) -> Any:
    doc = bd.ClayDoc()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box()))
    return clay_mode.adopt(ctx, doc, title="Scene")


def _small_incoming_doc() -> bd.ClayDoc:
    doc = bd.ClayDoc()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box()))
    return doc


def _patch_landing(monkeypatch: pytest.MonkeyPatch, make_doc: Any = _small_incoming_doc) -> None:
    from realmspinner.kernels.mesh import glbimport

    monkeypatch.setattr(clay_mode, "_within_mesh_ceiling", lambda path: b"")
    monkeypatch.setattr(glbimport, "glb_to_claydoc", lambda data, name="Imported": make_doc())


def _through_preview(ctx: Any, tab: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(svc_jobs, "create_job", lambda svc, **kw: {"id": "ref1"})
    clay_generate.submit_text(ctx, tab, "a wooden barrel")
    ctx.land_all()
    ctx.svc.store.jobs["ref1"] = {"status": "done"}
    clay_generate.poll(ctx)


def _through_mesh_queued(
    ctx: Any, tab: Any, monkeypatch: pytest.MonkeyPatch, promote: Any
) -> None:
    _through_preview(ctx, tab, monkeypatch)
    monkeypatch.setattr(svc_jobs, "promote_to_model", promote)
    clay_generate.accept_reference(ctx, tab)
    ctx.land_all()


# --- clay-03: the save doors have no ceiling ----------------------------------


def test_a_clay_document_past_the_reopen_ceiling_is_refused_at_save(tmp_path, monkeypatch):
    """``_load`` refuses anything over ``MAX_CLAY_SOURCE_BYTES``; before this
    fix ``save_to`` wrote past it with no check at all, producing a file the
    app would then refuse to reopen."""
    ctx = _Ctx(tmp_path)
    tab = _tab(ctx)
    oversized = b"0" * (MAX_CLAY_SOURCE_BYTES + 1)
    monkeypatch.setattr(serialize, "snapshot_bytes", lambda snap: oversized)

    path = tmp_path / "too-big.rblk"
    clay_mode.save_to(ctx, tab, path)
    ctx.land_all()

    assert not path.exists(), "an oversized document must not be written at all"
    assert tab.saving is False, "a refused save must not leave the tab locked"


def test_a_clay_document_past_the_reopen_ceiling_is_refused_at_save_as(tmp_path, monkeypatch):
    ctx = _Ctx(tmp_path)
    tab = _tab(ctx)
    oversized = b"0" * (MAX_CLAY_SOURCE_BYTES + 1)
    monkeypatch.setattr(serialize, "snapshot_bytes", lambda snap: oversized)
    path = tmp_path / "too-big-as.rblk"
    monkeypatch.setattr(
        "realmspinner.studio.dialogs.save_file", lambda *a, **kw: path
    )

    clay_mode.save_as(ctx, tab)
    ctx.land_all()

    assert not path.exists()
    assert tab.saving is False


def test_the_save_ceiling_check_raises_too_large_with_a_field(tmp_path):
    """Refusals raise ``service.errors`` exceptions carrying a ``field`` --
    the app-wide contract every other refusal in this codebase follows."""
    with pytest.raises(TooLarge) as excinfo:
        clay_mode._refuse_oversized_save(b"0" * (MAX_CLAY_SOURCE_BYTES + 1))
    assert excinfo.value.field == "save"


def test_a_generate_landing_grows_rblk_bytes_so_the_next_landing_is_bounded(
    tmp_path, monkeypatch
):
    """clay-03's second half: nothing refreshed ``tab.rblk_bytes`` after a
    landing, so a tab that had genuinely been measured (a non-zero
    ``rblk_bytes``) stayed at its pre-generate size forever, no matter how
    many meshes were landed into it afterwards."""
    _patch_landing(monkeypatch)
    ctx = _Ctx(tmp_path)
    tab = _tab(ctx)
    tab.rblk_bytes = 1_000

    def fake_decode(svc, job_id, tab_uid, group_name):
        return {
            "tab_uid": tab_uid,
            "doc": _small_incoming_doc(),
            "triangles": 10,
            "incoming_bytes": 5_000,
            "group_name": group_name,
        }

    monkeypatch.setattr(clay_generate, "_decode_landing", fake_decode)
    _through_mesh_queued(ctx, tab, monkeypatch, lambda svc, job_id, **kw: {"id": "mesh1"})
    ctx.svc.store.jobs["mesh1"] = {"status": "done"}
    clay_generate.poll(ctx)
    ctx.land_all()

    assert tab.rblk_bytes == 6_000, "the landed bytes were never added before this fix"


# --- clay-04: Cancel during "landing" must stop the merge ---------------------


def test_cancelling_during_the_landing_stage_prevents_the_merge(tmp_path, monkeypatch):
    """Reproduces the exact race the audit's probe (clay-mode-01.py) found:
    the decode task is already submitted (stage == "landing") when Cancel is
    pressed. Before this fix, ``_landed`` called ``land()`` unconditionally
    once that task's result arrived, merging the mesh in anyway -- exactly
    what ``cancel()``'s own docstring says does not happen."""
    _patch_landing(monkeypatch)
    ctx = _Ctx(tmp_path)
    tab = _tab(ctx)
    _through_mesh_queued(ctx, tab, monkeypatch, lambda svc, job_id, **kw: {"id": "mesh1"})
    ctx.svc.store.jobs["mesh1"] = {"status": "done"}

    clay_generate.poll(ctx)  # submits the decode-and-land task
    pending = ctx.state.clay.generate_pending
    assert pending is not None and pending["stage"] == "landing"

    clay_generate.cancel(ctx, tab)
    assert ctx.state.clay.generate_pending is None

    before = len(tab.doc.objects)
    ctx.land_all()  # the decode task that was already in flight lands now

    assert len(tab.doc.objects) == before, "cancel must stop the merge, not just the pending flag"


# --- clay-13: merge_into over-wraps a single-root hierarchy -------------------


def test_merge_into_does_not_wrap_a_single_root_hierarchy_in_a_group():
    """``merge_into``'s own docstring: "A single-root import is not wrapped
    in a group at all." The old code branched on the *count of objects*
    added, not the count of roots, so a one-root, one-child hierarchy (two
    objects, one root) was wrapped anyway."""
    incoming = bd.ClayDoc()
    root = incoming.add_object(bd.Obj(uid=bd.new_uid(), name="Root", mesh=bp.box()))
    child = incoming.add_object(
        bd.Obj(uid=bd.new_uid(), name="Child", mesh=bp.box((0.3, 0.3, 0.3)))
    )
    incoming.set_parent(child.uid, root.uid, keep_world=True)

    doc = bd.ClayDoc()
    merge.merge_into(doc, incoming, offset=np.zeros(3, dtype="f8"))

    root_after = doc.by_uid(root.uid)
    assert root_after.parent is None, "a single-root hierarchy must not gain a synthesizing group"
    assert len(doc.objects) == 2, "exactly Root and Child, no group empty"


def test_merge_into_still_groups_two_real_roots():
    """The other half of the same rule: two independent roots *do* still get
    a group, so the fix above does not just delete the feature."""
    incoming = bd.ClayDoc()
    incoming.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=bp.box()))
    incoming.add_object(bd.Obj(uid=bd.new_uid(), name="B", mesh=bp.box()))

    doc = bd.ClayDoc()
    added = merge.merge_into(doc, incoming, offset=np.zeros(3, dtype="f8"), group_name="Generated")

    assert len(doc.objects) == 3, "A, B, and the synthesizing group"
    assert all(obj.parent is not None for obj in added), "both roots landed inside the group"


# --- clay-14 / clay-15: the popup's Saving reason ------------------------------


def test_generate_text_button_greys_while_the_tab_is_saving():
    assert clay_bridge._generate_text_reason(prompt_ok=True, saving=True) == "Saving..."
    assert clay_bridge._generate_text_reason(prompt_ok=False, saving=True) == "Saving..."
    assert clay_bridge._generate_text_reason(prompt_ok=False, saving=False) != ""
    assert clay_bridge._generate_text_reason(prompt_ok=True, saving=False) == ""


def test_generate_from_image_button_names_saving_as_its_disabled_reason():
    assert clay_bridge._generate_image_reason(saving=True) == "Saving..."
    assert clay_bridge._generate_image_reason(saving=False) == ""


# --- clay-16: a texture pick must not land on a shifted slot ------------------


def _material_doc() -> bd.ClayDoc:
    """A palette of four (index 0 is ``ClayDoc``'s own default), with the
    object's faces all pointed at slot 2 -- so slot 1 is unused and
    ``remove_material(1)`` actually succeeds, the way
    ``test_clay_material_shift.py``'s own ``_obj``/``_palette`` helpers build
    one. Removing slot 1 shifts slot 2 (the pick's own target) down to slot
    1, and slot 3 down to slot 2 -- so slot 2 now holds a *different*
    material by the time the pick lands, the exact shift clay-16 is about.
    """
    doc = bd.ClayDoc()
    doc.add_material()  # slot 1 -- unused, the one that gets removed
    doc.add_material()  # slot 2 -- what the pick below is opened for
    doc.add_material()  # slot 3 -- shifts into slot 2's place
    mesh = bp.box()
    count = len(mesh.starts) - 1
    mesh = replace(mesh, material=np.full(count, 2, dtype=mesh.material.dtype))
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=mesh, material=2))
    return doc


def test_texture_assign_refuses_to_land_on_a_material_that_shifted_into_its_slot(tmp_path):
    # ``clay-mattex:`` results are dispatched straight to ``clay_props.
    # on_task_done`` by ``shell/tasks.py``, never through ``clay_mode.
    # on_task_done`` -- see that function's own docstring -- so this drives
    # it directly rather than through ``_Ctx.land_all``.
    ctx = _Ctx(tmp_path)
    tab = clay_mode.adopt(ctx, _material_doc(), title="Scene")
    doc = tab.doc
    target = doc.materials[2]

    # The picker opened on slot 2, naming *target*'s identity via ``tag`` --
    # exactly what ``_assign_texture`` does.
    key = clay_props._texture_task_key(tab.uid, 2, "base_color")

    # Before the queued result lands, the unused slot below it is removed --
    # every index above it, including 2, shifts down by one. The object now
    # at slot 2 is no longer *target*.
    assert doc.remove_material(1) is True
    assert doc.materials[2] is not target

    done = Done(key=key, result={"width": 1, "height": 1, "rgba": b"\xff\xff\xff\xff"}, tag=target)
    clay_props.on_task_done(ctx, done)

    assert doc.materials[2].base_color is None, "the shifted-in material must be untouched"
    assert not any(m.base_color is not None for m in doc.materials), (
        "the texture must not have landed on any material at all"
    )


def test_texture_assign_lands_normally_when_nothing_shifted(tmp_path):
    """The fix must not break the ordinary case: no removal happened, so the
    material at the index is still the one the picker was opened for."""
    ctx = _Ctx(tmp_path)
    tab = clay_mode.adopt(ctx, _material_doc(), title="Scene")
    doc = tab.doc
    target = doc.materials[2]

    key = clay_props._texture_task_key(tab.uid, 2, "base_color")
    done = Done(key=key, result={"width": 1, "height": 1, "rgba": b"\xff\xff\xff\xff"}, tag=target)
    clay_props.on_task_done(ctx, done)

    assert doc.materials[2].base_color is not None


# --- docs-02: the composition-doubt reasons reach the popup -------------------


def test_generate_popup_shows_the_composition_doubt_reason(tmp_path, monkeypatch):
    from realmspinner.service import errors as svc_errors

    def fake_promote(svc, job_id, **kw):
        if not kw.get("force"):
            raise svc_errors.Invalid("this reference cannot reconstruct")
        return {"id": "mesh1"}

    ctx = _Ctx(tmp_path)
    tab = _tab(ctx)
    _through_preview(ctx, tab, monkeypatch)
    ctx.svc.store.jobs["ref1"]["params"] = {
        "reference_report": {
            "ok": False,
            "reasons": ["There is more than one object in the reference."],
        }
    }
    monkeypatch.setattr(svc_jobs, "promote_to_model", fake_promote)
    clay_generate.accept_reference(ctx, tab)
    ctx.land_all()

    pending = ctx.state.clay.generate_pending
    assert pending is not None and pending["force_offer"] is True
    assert pending.get("doubt_reasons") == (
        "There is more than one object in the reference.",
    ), "the popup has nothing to render the doubt from before this fix"
