"""Mason's controller: opening, saving, exporting, guarding and keys.

Stage E gives Mason a document, so this module is what ``studio/modes/clay/mode.py`` is to
Clay -- everything here is *about* scenes rather than geometry. The engine
under ``mason/`` has no idea a job or a task thread exists; this is the layer
that knows about both, and the panes draw against it.

The rule that shapes this file is Clay's own: **no file dialog and no encode
ever runs on the frame thread.** A native picker is modal to the OS and blocks
until dismissed, and a ``.wscn`` of any size is a zip to build. Both go through
``ctx.submit`` and come back through :func:`on_task_done`, which is why saving
is a *state* (``MasonTab.saving``) rather than a function call that returns.

``mason_io.py`` already states the split for this format's own reasons:
``serialize.snapshot`` is the cheap frame-thread half (it reads ``doc.view``
itself -- there is no ``view=`` kwarg, unlike Clay's ``.wblk`` snapshot, since
a ``.wscn``'s camera lives inside ``scene.json`` rather than a second small
member); ``snapshot_bytes`` is the expensive half and runs inside ``run()``.
:func:`camera_of` therefore writes the live camera onto ``doc.view`` *before*
either save path takes its snapshot -- there is nowhere else for the camera to
land, since ``snapshot`` never takes one as an argument.

**A failed save must clear that state**, Clay's own rule restated: ``saving``
gates every control that changes the document, so without
:func:`on_task_failed` one failed write leaves the tab permanently read-only.

Every task key carries the ``mason-`` prefix, because the app claims results
by prefix: a key without one is a result delivered nowhere. ``mason-asset:``
keys are :mod:`.mason_assets`'s alone -- :func:`on_task_done` gives that module
first refusal before doing anything of its own with a ``mason-`` key, since a
library asset's background parse is not a document task at all.
"""

from __future__ import annotations

import hashlib
import logging
import weakref
from pathlib import Path
from typing import Any

from ....core.safeio import sizeguard
from ... import dialogs, docmodes, journal
from ..._view_frame import AXIS_VIEW_KEYS, axis_view_key
from . import assets as mason_assets
from . import fileio as mason_io
from . import state as mason_state
from .state import MasonState, MasonTab

log = logging.getLogger(__name__)


def _path_key(path: Path) -> str:
    """A short, stable id for a path -- ``clay_mode``'s own function,
    restated: ``hash(str(path))`` is salted per process and can collide within
    one session; sha1 never does."""
    return hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]


def ensure(ctx: Any) -> MasonState:
    """The mode's state, built on first use -- ``clay_mode.ensure``'s reason:
    a session that never opens Mason should not pay for it."""
    state = ctx.state.mason
    if state is None:
        state = MasonState()
        ctx.state.mason = state
    return state


# The three recents wrappers every document mode carries.
remember_path, forget_path, recent_paths = docmodes.recents_for("mason")


def persist(ctx: Any) -> None:
    """A no-op -- ``clay_mode.persist``'s reason: the recent list lives in
    :mod:`.recents`, which persists itself on every write. Kept because it is
    called from a dozen sites after every open and save, and turning each of
    those into "call this only if the mode still has settings" is how one of
    them comes to skip a write that mattered later."""


def active(ctx: Any) -> MasonTab | None:
    state = ctx.state.mason
    return state.active if state is not None else None


def _enter_mason(ctx: Any) -> None:
    """Adoption switches modes through ``state.set_mode``, never by
    assignment -- ``clay_mode._enter_clay``'s reason, restated here."""
    from ...state import set_mode

    set_mode(ctx.state, "mason")


# --- opening ------------------------------------------------------------------


def adopt(
    ctx: Any,
    doc: Any,
    *,
    path: Path | None = None,
    title: str | None = None,
    view: dict[str, Any] | None = None,
) -> MasonTab:
    state = ensure(ctx)
    tab = MasonTab(
        doc=doc,
        title=title or mason_state.title_for(path),
        path=path,
        saved_head=doc.history.head,
    )
    if view:
        tab.view.yaw = view["yaw"]
        tab.view.pitch = view["pitch"]
        tab.view.distance = view["distance"]
        tab.view.target = view["target"]
        # Already framed: an auto-fit over the top would throw away the
        # answer just read off disk, ``clay_mode.adopt``'s own reason.
        tab.view.fitted = True
    state.add(tab)
    remember_path(ctx, path)
    persist(ctx)
    return tab


def new_document(ctx: Any) -> MasonTab:
    from .engine import document as md

    return adopt(ctx, md.MasonDoc(), title="Untitled")


def _within_ceiling(path: Path) -> Path:
    return sizeguard.within_ceiling(Path(path), mason_io.MAX_WSCN_BYTES)


def ask_open(ctx: Any) -> None:
    """The picker, on a task thread, then the decode on the same one."""
    ensure(ctx)

    def run() -> dict[str, Any] | None:
        path = dialogs.open_file("Open Mason scene", mason_io.WSCN_FILTER)
        return None if path is None else mason_io.load(Path(path))

    ctx.submit("mason-open", run)


def open_path(ctx: Any, path: Path) -> None:
    state = ensure(ctx)
    path = Path(path)
    existing = state.find_path(path)
    if existing is not None:
        # Focus rather than fork: two tabs over one path would race on save.
        state.activate(existing.uid)
        return
    ctx.submit(f"mason-open:{_path_key(path)}", mason_io.load, path)


# --- placing ------------------------------------------------------------------


def _over_max_placed(doc: Any, adding: int) -> int | None:
    """The document's node count after attaching ``adding`` more nodes, or
    ``None`` when that stays within :data:`scene.MAX_PLACED`.

    The 2026-09-18 audit's mason-01: every node-adding controller in this
    module (:func:`place_ref`, :func:`place_light`, :func:`place_camera`,
    :func:`place_prefab`, :func:`group_selected`, :func:`add_terrain`,
    :func:`duplicate_selected`) called ``doc.add_node``/``add_nodes`` with no
    pre-check of its own, so the ``ValueError``
    ``MasonDoc._check_max_placed`` raises at the ceiling escaped uncaught --
    nothing between a Mason key handler and ``App.run()``'s whole-loop
    ``except`` catches it (shell-04), so the session ended with every open
    document's unsaved work. ``ui/panes/tools.py``'s own ``_over_max_placed``
    already guards the Array buttons this exact way (count first, toast,
    build nothing); this is the same shape, generalised to a plain "how many
    nodes is this attach about to add" count, since each door here adds a
    different shape of thing (one leaf, one empty group, a batch of whole
    subtrees).
    """
    from .engine import scene as msc

    total = len(doc.all_nodes()) + adding
    return total if total > msc.MAX_PLACED else None


def _toast_over_max_placed(ctx: Any, over: int) -> None:
    from .engine import scene as msc

    ctx.toast(
        f"That would bring this scene to {over} nodes, past the "
        f"{msc.MAX_PLACED} limit -- refusing rather than building it.",
        "error",
    )


def place_ref(ctx: Any, ref: Any, *, name: str = "") -> int | None:
    """Place a resolved reference (a primitive or a library asset) as a new
    :class:`~.mason.nodes.MeshNode`, selected."""
    tab = active(ctx)
    if tab is None or tab.saving:
        return None
    from .engine import nodes as nd

    node = nd.MeshNode(uid=nd.new_uid(), name=name or "Mesh", ref=ref)
    over = _over_max_placed(tab.doc, len(list(nd.walk([node]))))
    if over is not None:
        _toast_over_max_placed(ctx, over)
        return None
    tab.doc.add_node(node)
    tab.doc.select([node.uid])
    return node.uid


def place_primitive(ctx: Any, generator: str) -> int | None:
    from .engine import refs as mason_refs

    ref = mason_refs.primitive_ref(generator, {})
    return place_ref(ctx, ref, name=generator.replace("_", " ").title())


def place_light(ctx: Any, kind: str) -> int | None:
    tab = active(ctx)
    if tab is None or tab.saving:
        return None
    from .engine import nodes as nd

    node = nd.LightNode(uid=nd.new_uid(), name=kind.title(), kind=kind)
    over = _over_max_placed(tab.doc, len(list(nd.walk([node]))))
    if over is not None:
        _toast_over_max_placed(ctx, over)
        return None
    tab.doc.add_node(node)
    tab.doc.select([node.uid])
    return node.uid


def place_camera(ctx: Any) -> int | None:
    tab = active(ctx)
    if tab is None or tab.saving:
        return None
    from .engine import nodes as nd

    node = nd.CameraNode(uid=nd.new_uid(), name="Camera")
    over = _over_max_placed(tab.doc, len(list(nd.walk([node]))))
    if over is not None:
        _toast_over_max_placed(ctx, over)
        return None
    tab.doc.add_node(node)
    tab.doc.select([node.uid])
    return node.uid


def place_prefab(ctx: Any, name: str) -> int | None:
    """Place one instance of the template called ``name``.

    Refused -- ``None``, nothing added -- if the document has no such template.
    A :class:`~.mason.nodes.PrefabNode` naming a template that does not exist
    is a legal thing for a *loaded* document to contain (``remove_prefab``'s
    own docstring says so, and ``scene.py`` resolves one as dangling), but
    minting one on purpose would be authoring the dangling case.
    """
    tab = active(ctx)
    if tab is None or tab.saving or name not in tab.doc.prefabs:
        return None
    from .engine import nodes as nd

    node = nd.PrefabNode(uid=nd.new_uid(), name=name, template=name)
    over = _over_max_placed(tab.doc, len(list(nd.walk([node]))))
    if over is not None:
        _toast_over_max_placed(ctx, over)
        return None
    tab.doc.add_node(node)
    tab.doc.select([node.uid])
    return node.uid


#: What :func:`place_armed` does with each ``MasonState.place_kind`` prefix.
#: A table rather than a chain of ``startswith`` tests, so the Assets pane's
#: arming keys and this dispatch cannot drift into a kind the pane can arm and
#: nothing can place -- which is exactly what Stage E shipped: ``place_kind``
#: was written by the pane and read by nothing but the hint line, so arming a
#: primitive, a light or a camera and clicking in the viewport placed nothing
#: at all.
_PLACERS = {
    "light:": lambda ctx, key: place_light(ctx, key.split(":", 1)[1]),
    "camera": lambda ctx, _key: place_camera(ctx),
}


def place_armed(ctx: Any, point: Any = None) -> int | None:
    """Place whatever the Assets pane has armed, at ``point`` if given.

    The viewport's click handler is the only caller: a primitive, a light and a
    camera have no position of their own until the user says where, which is
    why the pane arms rather than places (``mason_palette``'s own docstring
    draws that line) and why this takes a world point.

    The arming is **not** cleared afterwards. Placing a row of fence posts is
    one arming and six clicks, and a palette that disarmed itself after the
    first would make the other five a trip back to the sidebar each; Esc is
    what clears it, and the hint line says so.
    """
    tab = active(ctx)
    if tab is None or tab.saving:
        return None
    state = ensure(ctx)
    # Before the add, not between the add and the move: ``collapse_since``
    # folds what was pushed *after* its mark, so a mark taken later would leave
    # the add as a step of its own and a first Ctrl+Z would undo only the move,
    # leaving the new node at the origin.
    mark = tab.doc.mark()
    if state.place_prefab:
        uid = place_prefab(ctx, state.place_prefab)
    else:
        key = state.place_kind
        if not key:
            return None
        placer = next((fn for prefix, fn in _PLACERS.items() if key.startswith(prefix)), None)
        uid = placer(ctx, key) if placer is not None else place_primitive(ctx, key)
    if uid is not None and point is not None:
        _move_to(tab.doc, uid, point)
        tab.doc.collapse_since(mark)
    return uid


def _move_to(doc: Any, uid: int, point: Any) -> None:
    """Put a just-placed node's *local* translation where a world click was.

    A world point is a local one here only because :func:`place_armed` adds at
    the root, where the two coincide. Said rather than assumed: the day a
    placement lands under a selected group, this needs the inverse of that
    group's world matrix and the node will otherwise be off by it.
    """
    import numpy as np

    node = doc.node(uid)
    if node is None:
        return
    doc.set_transform(uid, translation=np.asarray(point, dtype="f8"), was=node.trs())


def import_glb_path(ctx: Any, path: Path) -> None:
    """A bare ``.glb`` dropped on Mason: import it into the library, then
    place the resulting job as a mesh node.

    A :class:`~.mason.refs.LibraryRef` names a job id, never a path -- see
    that module's own docstring -- so a file with no job behind it has to get
    one first. The import (a task-thread read plus ``svc_jobs.import_mesh``)
    and the placement both go through :func:`on_task_done`'s
    ``mason-import`` arm, mirroring ``clay_mode.import_glb_path``'s split.
    """
    path = Path(path)

    def run() -> dict[str, Any]:
        from ....service import jobs as svc_jobs
        from ....service.validation import MAX_MESH_BYTES

        data = sizeguard.within_ceiling(path, MAX_MESH_BYTES).read_bytes()
        result = svc_jobs.import_mesh(ctx.svc, data, name=path.stem, prompt=path.stem)
        return {"job_id": result["id"], "name": path.stem}

    ctx.submit(f"mason-import:{_path_key(path)}", run)


def place_job(ctx: Any, job: Any) -> int | None:
    """A library mesh row -> a placed :class:`~.mason.nodes.MeshNode`."""
    from .engine import refs as mason_refs

    job_id = job["id"] if isinstance(job, dict) else str(job)
    name = (job.get("name") if isinstance(job, dict) else "") or ""
    ref = mason_refs.LibraryRef(job_id=job_id, name=name)
    return place_ref(ctx, ref, name=name or "Asset")


def add_asset_to_scene(ctx: Any, job: Any) -> int | None:
    """A library mesh -> a node in the open scene, from anywhere in the app.

    :func:`place_job` with the two things the Assets pane already has and the
    library's overflow menu does not: a document to place into, and Mason on
    screen to see it happen. Both are the same defect in two halves -- an exit
    that quietly placed a node into a scene the user is not looking at, or into
    no scene at all, would report success and show nothing. So a scene is minted
    if none is open (``new_document``, exactly what the empty state's own "New
    scene" button does) and the mode is switched to afterwards.

    -> the new node's uid, or None if the placement itself refused.
    """
    if active(ctx) is None:
        new_document(ctx)
    uid = place_job(ctx, job)
    if uid is not None:
        _enter_mason(ctx)
    return uid


# --- editing --------------------------------------------------------------------


def group_selected(ctx: Any) -> None:
    """Wrap the current selection in one new :class:`~.mason.nodes.GroupNode`,
    as a single undo step.

    Every selected node keeps its own parent's *transform space* by being
    reparented as a sibling of itself first -- ``move_node`` only reorders and
    reparents, it never re-expresses a local transform relative to its new
    parent -- so a selection under different parents ends up moved to wherever
    the group node landed, exactly as a modelling package's "group" gesture
    reads on screen: the arrangement is preserved by construction because the
    new group starts at the identity, not by any transform math here.
    """
    tab = active(ctx)
    if tab is None or tab.saving:
        return
    doc = tab.doc
    uids = sorted(doc.selection)
    if not uids:
        return
    from .engine import nodes as nd

    # The 2026-09-18 audit's mason-01: this used to call ``doc.add_node``
    # with no ceiling pre-check, so a scene already sitting at MAX_PLACED
    # raised ``add_node``'s ``ValueError`` uncaught -- see
    # :func:`_over_max_placed`. The new group is one leaf node (children are
    # moved into it below, not added), so the count is 1.
    over = _over_max_placed(doc, 1)
    if over is not None:
        _toast_over_max_placed(ctx, over)
        return
    mark = doc.mark()
    group = nd.GroupNode(uid=nd.new_uid(), name="Group")
    # The first selected node's own parent, so the group lands beside what it
    # is about to hold rather than always at the root.
    parent_uid = doc.parent_uid_of(uids[0])
    doc.add_node(group, parent_uid=parent_uid)
    for uid in uids:
        doc.move_node(uid, len(group.children), parent_uid=group.uid)
    doc.collapse_since(mark)
    doc.select([group.uid])


def ungroup_selected(ctx: Any) -> None:
    """Lift every selected group's children up to the group's own parent and
    remove the group, as one step.

    The companion to :func:`group_selected` and the reason that function's
    "preserved by construction" argument holds in both directions: the group
    node starts at the identity and is never given a transform by grouping, so
    dissolving one cannot move what it held either. A group the *user* has since
    moved is a different matter -- its children do move, because their local
    transforms were always relative to it -- and that is the honest answer
    rather than a silent re-expression of six transforms that would then
    disagree with the numbers in Properties.
    """
    tab = active(ctx)
    if tab is None or tab.saving:
        return
    doc = tab.doc
    from .engine import nodes as nd

    groups = [
        uid
        for uid in sorted(doc.selection)
        if isinstance(doc.node(uid), nd.GroupNode) and doc.node(uid).children
    ]
    if not groups:
        return
    mark = doc.mark()
    freed: list[int] = []
    for uid in groups:
        group = doc.node(uid)
        parent_uid = doc.parent_uid_of(uid)
        index = doc.index_of(uid)
        # Children in order, each inserted where the group sat, so sibling
        # order -- which is export order -- reads the way the outliner did.
        for offset, child in enumerate(list(group.children)):
            doc.move_node(child.uid, index + offset, parent_uid=parent_uid)
            freed.append(child.uid)
        doc.remove_node(uid)
    doc.collapse_since(mark)
    doc.select(freed)


def define_prefab_from_selection(ctx: Any, name: str = "") -> str:
    """Turn the one selected node into a template, and the selection itself
    into an instance of it. -> the template's name, or "" if refused.

    **Two mechanisms, not conflated** -- the plan's own instruction. This is the
    *authoring* half: ``document.define_prefab`` stores the subtree and the
    scene node is then replaced by a :class:`~.mason.nodes.PrefabNode`, so what
    the user selected becomes the first instance rather than staying a
    one-off copy beside the template. Without that replacement, "make prefab"
    would leave the thing the user was looking at untracked by the template it
    just defined, and editing the template would visibly change every instance
    *except* the one it was made from.

    One node only, for ``mason_props._selected``'s reason: a prefab of "these
    four things" is a prefab of a group, and asking the user to group them
    first is one gesture they can see rather than a group this silently mints
    under a name they did not choose.
    """
    tab = active(ctx)
    if tab is None or tab.saving or len(tab.doc.selection) != 1:
        return ""
    doc = tab.doc
    uid = next(iter(doc.selection))
    node = doc.node(uid)
    if node is None:
        return ""
    from .engine import nodes as nd

    if isinstance(node, (nd.TerrainNode, nd.PrefabNode)):
        # The terrain is a document singleton whose large array must not be
        # copied into a template (``mason/nodes.py``'s own reason for the
        # node/array split), and an instance of an instance is the recursion
        # ``define_prefab`` refuses at the door anyway -- refused here instead
        # so it reads as a disabled button rather than an exception.
        return ""
    name = name or node.name or "Prefab"
    parent_uid = doc.parent_uid_of(uid)
    index = doc.index_of(uid)
    mark = doc.mark()
    try:
        doc.define_prefab(name, node)
    except ValueError as exc:
        # The 2026-09-12 audit's docs-03: this refusal used to be swallowed
        # here with no toast at either context-menu call site, so a name that
        # collided with a template the selection already places (the
        # recursion ``document.define_prefab`` itself refuses) did nothing
        # that the reader could see -- Chapter 17's "give it a name" gesture
        # looked like it simply failed to register.
        ctx.toast(f"Could not make a prefab: {exc}", "error")
        return ""
    instance = nd.PrefabNode(uid=nd.new_uid(), name=node.name or name, template=name)
    instance.translation = node.translation
    instance.rotation = node.rotation
    instance.scale = node.scale
    doc.remove_node(uid)
    doc.add_node(instance, parent_uid=parent_uid, index=index)
    doc.collapse_since(mark)
    doc.select([instance.uid])
    return name


def prompt_define_prefab_from_selection(ctx: Any) -> None:
    """Ask for the new template's name, then define it -- the gesture Chapter
    17 describes as "right-click it. Choose Make prefab and give it a name."

    The other half of the 2026-09-12 audit's docs-03: both context-menu call
    sites (``modes/mason/ui/panes/menu.py``, ``modes/mason/ui/panes/outliner.py``) used to call
    :func:`define_prefab_from_selection` with no name at all, so there was
    nowhere in the whole gesture the chapter's naming step could happen --
    the template was silently named after the node it was made from. This
    follows the house pattern (``poser_mode.name_pose``'s own ``ctx.prompts``
    call) rather than inventing a second way to ask a one-line question.
    Seeded with the selected node's own name so confirming with no edit keeps
    today's behaviour exactly, and refused up front (no popup) for the same
    selection shapes :func:`define_prefab_from_selection` itself refuses, so a
    disabled menu row never opens a dialog with nothing it can do.
    """
    tab = active(ctx)
    if tab is None or tab.saving or len(tab.doc.selection) != 1:
        return
    doc = tab.doc
    uid = next(iter(doc.selection))
    node = doc.node(uid)
    if node is None:
        return
    from .engine import nodes as nd

    if isinstance(node, (nd.TerrainNode, nd.PrefabNode)):
        return
    default = node.name or "Prefab"

    def accept(name: str) -> None:
        define_prefab_from_selection(ctx, name)

    ctx.prompts.ask(
        dialogs.Prompt(title="Make prefab", label="Name", value=default, on_accept=accept)
    )


def unpack_selected(ctx: Any) -> None:
    """Replace every selected prefab instance with an independent copy of its
    template -- ``document.unpack_instance``, which is the one escape hatch
    Mason offers instead of per-child overrides."""
    tab = active(ctx)
    if tab is None or tab.saving:
        return
    doc = tab.doc
    from .engine import nodes as nd

    uids = [uid for uid in sorted(doc.selection) if isinstance(doc.node(uid), nd.PrefabNode)]
    if not uids:
        return
    mark = doc.mark()
    fresh: list[int] = []
    refused = 0
    for uid in uids:
        try:
            fresh.append(doc.unpack_instance(uid).uid)
        except (KeyError, TypeError):
            # A dangling instance -- its template was removed -- has nothing to
            # unpack into. Skipped rather than raised: the Prefabs pane can
            # offer the button over a mixed selection without having to resolve
            # every instance itself first.
            continue
        except ValueError:
            # The 2026-09-18 audit's mason-02: ``unpack_instance`` gained its
            # own MAX_PLACED refusal in the 2026-09-15 audit's mason-01 for
            # exactly this attach point, but this call site only caught
            # ``(KeyError, TypeError)`` -- the ``ValueError`` escaped
            # uncaught, past App.run()'s whole-loop catch-all (shell-04),
            # ending the session with every open document's unsaved work.
            # Skipped like a dangling instance, but counted, so a mixed
            # selection can still unpack whatever it can and the toast below
            # says how many of the rest it could not.
            refused += 1
            continue
    doc.collapse_since(mark)
    if fresh:
        doc.select(fresh)
    if refused:
        ctx.toast(
            f"Skipped {refused} instance(s): unpacking would have passed "
            "the scene's node limit.",
            "error",
        )


def remove_prefab(ctx: Any, name: str) -> bool:
    """Drop a template. Instances are left naming it and resolve as dangling --
    ``document.remove_prefab``'s own rule, restated here so a caller does not
    have to read that one to know this does not delete anything from the
    scene."""
    tab = active(ctx)
    if tab is None or tab.saving:
        return False
    return tab.doc.remove_prefab(name)


# --- terrain --------------------------------------------------------------------


def add_terrain(ctx: Any, side: int = 0, size: float = 0.0) -> int | None:
    """Give the document its one ground: a flat height field, and the
    :class:`~.mason.nodes.TerrainNode` that refers to it, as one undo step.

    Both halves together, because either alone is a state nothing in the app
    can act on: ``doc.terrain`` with no node has no outliner row, no transform
    and no export (``scene.py`` yields geometry for the *node*), and a node with
    no ``doc.terrain`` draws and picks nothing. ``set_terrain`` is already one
    undoable step and the add is another, so the two are folded --
    ``unpack_instance``'s pattern -- and one Ctrl+Z takes the ground away whole.
    """
    tab = active(ctx)
    if tab is None or tab.saving or tab.doc.terrain is not None:
        return None
    import numpy as np

    from ....kernels.geom3d import gltf
    from .engine import nodes as nd
    from .engine.terrain import Terrain

    doc = tab.doc
    # The 2026-09-18 audit's mason-01: this used to call ``doc.add_node``
    # with no ceiling pre-check, so a scene already sitting at MAX_PLACED
    # raised ``add_node``'s ``ValueError`` uncaught -- see
    # :func:`_over_max_placed`. Checked before the (not cheap) height-field
    # array is even built, since the placement would be refused anyway.
    over = _over_max_placed(doc, 1)
    if over is not None:
        _toast_over_max_placed(ctx, over)
        return None
    side = int(side or mason_state.DEFAULT_TERRAIN_SIDE)
    size = float(size or mason_state.DEFAULT_TERRAIN_SIZE)
    terrain = Terrain(
        heights=np.zeros((side + 1, side + 1), dtype=np.float32),
        size_x=size,
        size_z=size,
        material=gltf.Material(name="terrain"),
    )
    node = nd.TerrainNode(uid=nd.new_uid(), name="Terrain")
    mark = doc.mark()
    doc.set_terrain(terrain)
    # At the front, so the ground is the first row in the outliner and the
    # first node in every export -- which is the order a scene reads in.
    doc.add_node(node, index=0)
    doc.collapse_since(mark)
    doc.select([node.uid])
    return node.uid


def remove_terrain(ctx: Any) -> bool:
    """Take the ground back out, node and height field together, as one step.

    Recoverable by Ctrl+Z: ``set_terrain`` pushes a ``TerrainSwapEdit`` holding
    the array itself, which is exactly why that edit type exists -- see its own
    docstring for the incident where clearing a terrain silently discarded
    however long someone had spent sculpting it.
    """
    tab = active(ctx)
    if tab is None or tab.saving or tab.doc.terrain is None:
        return False
    doc = tab.doc
    from .engine import nodes as nd

    mark = doc.mark()
    for node in [n for n in doc.all_nodes() if isinstance(n, nd.TerrainNode)]:
        doc.remove_node(node.uid)
    doc.set_terrain(None)
    doc.collapse_since(mark)
    return True


def duplicate_selected(ctx: Any) -> None:
    """Deep-copy every selected node, as siblings of the originals, as one
    step."""
    tab = active(ctx)
    if tab is None or tab.saving:
        return
    doc = tab.doc
    uids = sorted(doc.selection)
    if not uids:
        return
    from .engine import nodes as nd

    copies: list[Any] = []
    parents: dict[int, int | None] = {}
    for uid in uids:
        node = doc.node(uid)
        if node is None:
            continue
        parents[uid] = doc.parent_uid_of(uid)
        copies.append((parents[uid], nd.copy_subtree(node, fresh_uids=True)))
    if not copies:
        return
    # The 2026-09-18 audit's mason-01: this used to call ``doc.add_node`` in
    # a loop with no ceiling pre-check, so a duplication that crossed
    # MAX_PLACED mid-loop raised ``add_node``'s ``ValueError`` uncaught --
    # ending the session -- *and* left whatever copies had already been
    # added attached with the undo mark still open, uncollapsed, since the
    # exception unwound before ``collapse_since`` ever ran. Counted and
    # refused here, before ``mark()`` is even taken, so neither half-attached
    # state is reachable.
    adding = sum(len(list(nd.walk([copy]))) for _parent_uid, copy in copies)
    over = _over_max_placed(doc, adding)
    if over is not None:
        _toast_over_max_placed(ctx, over)
        return
    mark = doc.mark()
    new_uids: list[int] = []
    for parent_uid, copy in copies:
        doc.add_node(copy, parent_uid=parent_uid)
        new_uids.append(copy.uid)
    doc.collapse_since(mark)
    doc.select(new_uids)


def delete_selected(ctx: Any) -> None:
    """Remove every selected node -- and its subtree -- as one step."""
    tab = active(ctx)
    if tab is None or tab.saving:
        return
    doc = tab.doc
    uids = sorted(doc.selection)
    if not uids:
        return
    mark = doc.mark()
    for uid in uids:
        if doc.node(uid) is not None:
            doc.remove_node(uid)
    doc.collapse_since(mark)
    doc.select([])


# --- camera -------------------------------------------------------------------


def camera_of(ctx: Any, tab: MasonTab) -> Any:
    """The tab's stored camera, refreshed from the live viewport first, and
    written onto ``doc.view`` -- the one place a ``.wscn``'s camera lives.

    ``serialize.snapshot`` takes no ``view=`` argument (see the module
    docstring): it reads ``doc.view`` itself, so this is the write every save
    path must make before taking a snapshot.
    """
    view = getattr(ctx, "mason_view", None)
    if view is not None and getattr(view, "camera", None) is not None:
        tab.view.read_from(view.camera)
    from .engine import serialize

    tab.doc.view = {
        **{name: getattr(tab.view, name) for name in serialize.VIEW_FIELDS},
        "target": tuple(tab.view.target),
    }
    return tab.view


def remember_camera(ctx: Any, tab: MasonTab | None) -> None:
    """Snapshot the live camera onto a tab that is being switched away from."""
    if tab is not None:
        camera_of(ctx, tab)


def apply_camera(ctx: Any, tab: MasonTab) -> None:
    """Put a tab's camera back on the viewport, or frame it if it has none."""
    view = getattr(ctx, "mason_view", None)
    if view is None or getattr(view, "camera", None) is None:
        return
    if tab.view.fitted:
        tab.view.write_to(view.camera)
        return
    source = mason_assets.ensure(ctx)
    view.frame_selection(tab.doc, source)
    tab.view.read_from(view.camera)
    tab.view.fitted = True


# --- saving ---------------------------------------------------------------------

_start = docmodes.start_save


def save_to(ctx: Any, tab: MasonTab, path: Path) -> None:
    """Write the document to a known path. See :func:`camera_of` and the
    module docstring for the read/encode ordering this follows."""
    from .engine import serialize

    path = Path(path)
    doc = tab.doc
    rev = doc.history.head
    camera_of(ctx, tab)
    snap = serialize.snapshot(doc)

    def run() -> dict[str, Any]:
        mason_io.write_wscn(path, serialize.snapshot_bytes(snap))
        return {"rev": rev, "path": str(path), "retitle": True}

    _start(ctx, tab, f"mason-save:{tab.uid}", run)


def save(ctx: Any, tab: MasonTab | None = None) -> None:
    tab = tab or active(ctx)
    docmodes.save(
        tab, save_as=lambda: save_as(ctx, tab), save_to=lambda: save_to(ctx, tab, tab.path)
    )


def save_as(ctx: Any, tab: MasonTab | None = None) -> None:
    """The picker and the encode on one task thread -- ``clay_mode.save_as``'s
    reason: the *snapshot* is cheap enough to take before an unbounded modal
    dialog, and the encode must not run before or during it."""
    from .engine import serialize

    tab = tab or active(ctx)
    if tab is None or tab.saving:
        return
    doc, title = tab.doc, tab.title
    rev = doc.history.head
    camera_of(ctx, tab)
    snap = serialize.snapshot(doc)

    def run() -> dict[str, Any] | None:
        path = dialogs.save_file(
            "Save Mason scene",
            f"{title}{mason_state.WSCN_SUFFIX}",
            mason_io.WSCN_FILTER,
        )
        if path is None:
            return None
        path = Path(path).with_suffix(mason_state.WSCN_SUFFIX)
        mason_io.write_wscn(path, serialize.snapshot_bytes(snap))
        return {"rev": rev, "path": str(path), "retitle": True}

    _start(ctx, tab, f"mason-saveas:{tab.uid}", run)


# --- export -----------------------------------------------------------------------


def export_glb(ctx: Any, tab: MasonTab | None = None) -> None:
    """Write ``scene.glb`` plus its ``scene.json`` sidecar, an engine can read
    directly."""
    tab = tab or active(ctx)
    if tab is None or tab.saving:
        return
    doc, title = tab.doc, tab.title

    def run() -> dict[str, Any] | None:
        path = dialogs.save_file(
            "Export Mason scene as GLB", f"{title}.glb", mason_io.GLB_FILTER
        )
        if path is None:
            return None
        path = Path(path).with_suffix(".glb")
        source = mason_assets.ensure(ctx)
        files = mason_io.glb_bundle(doc, source)
        mason_io.write_files(files, path, primary="scene.glb")
        return {"exported": True, "path": str(path)}

    _start(ctx, tab, f"mason-exportglb:{tab.uid}", run)


def export_obj(ctx: Any, tab: MasonTab | None = None) -> None:
    """Write a merged ``scene.obj``, surfacing ``skipped`` -- a loss that is
    stated is not the same as a loss that is invisible."""
    from .engine import objout

    tab = tab or active(ctx)
    if tab is None or tab.saving:
        return
    doc, title = tab.doc, tab.title

    def run() -> dict[str, Any] | None:
        path = dialogs.save_file("Export Mason scene as OBJ", f"{title}.obj", mason_io.OBJ_FILTER)
        if path is None:
            return None
        path = Path(path).with_suffix(".obj")
        source = mason_assets.ensure(ctx)
        files, skipped, vertices, triangles = mason_io.obj_bundle(doc, source)
        mason_io.write_files(files, path, primary=objout.OBJ)
        return {
            "exported": True,
            "path": str(path),
            "skipped": skipped,
            "vertices": vertices,
            "triangles": triangles,
        }

    _start(ctx, tab, f"mason-exportobj:{tab.uid}", run)


# --- the library ------------------------------------------------------------------


def export_library(ctx: Any, tab: MasonTab | None = None) -> None:
    """Mint an ordinary asset from the scene: the round trip's outward half.

    ``clay_mode.export_asset``'s shape and the same payoff -- what comes out is
    a ``done`` **model** row, so the library, the inspector, every mesh export
    and the whole 3D pipeline work on it without any of them learning that
    Mason exists. The merged GLB goes first (``import_mesh`` is what creates
    the row at all) and the ``scene.wscn`` sidecar second, so a crash between
    the two leaves the sidecar absent rather than describing an arrangement the
    mesh on disk is not.

    **The GLB written here is the scene's own, not the OBJ's merge.** A
    ``scene.glb`` keeps the node graph -- the groups, the instances, the lights
    and the cameras -- and ``import_mesh`` stores it as ``source.glb`` with
    ``model.glb`` derived from it, so the hierarchy survives into the library
    row rather than being flattened on the way in. The *manifest* half of
    :func:`export_glb`'s bundle is deliberately not written beside it: it is
    the provenance sidecar for a **file** export an engine import script reads,
    and the library row's provenance is the ``.wscn`` itself, which says
    strictly more and is the thing that reopens.

    ``authored="mason"`` is the marker :func:`edit_asset_in_mason` is offered
    from -- see ``service._jobs_create.import_mesh`` for why the mesh side
    needed a field the reference side already had.
    """
    from .engine import serialize

    tab = tab or active(ctx)
    if tab is None or tab.saving:
        return
    doc, title = tab.doc, tab.title
    if not doc.roots:
        # Refused here rather than at the service door, so no job directory is
        # ever created for it -- ``clay_mode.export_asset``'s own reason:
        # ``check_glb`` would refuse the same empty bytes, but only after this
        # had told the user an export was under way.
        docmodes.refuse(ctx, "There is nothing in this scene to export.")
        return
    camera_of(ctx, tab)
    snap = serialize.snapshot(doc)

    def run() -> dict[str, Any]:
        from ....kernels.geom3d import glbwrite
        from ....service import files as svc_files
        from ....service import jobs as svc_jobs
        from .engine import gltfout

        source = mason_assets.ensure(ctx)
        export = gltfout.scene_model(doc, source)
        result = svc_jobs.import_mesh(
            ctx.svc,
            glbwrite.write_glb(export.model),
            name=title,
            prompt=title,
            authored="mason",
        )
        job_id = result["id"]
        svc_files.save_mason_source(ctx.svc, job_id, serialize.snapshot_bytes(snap))
        return {"job_id": job_id, "exported_asset": True}

    _start(ctx, tab, f"mason-library:{tab.uid}", run)


def edit_asset_in_mason(ctx: Any, job: Any) -> None:
    """Reopen the ``scene.wscn`` beside a library asset: the round trip's
    inward half.

    **No fallback, unlike ``clay_mode.edit_asset_in_clay``**, and the asymmetry
    is the point rather than an omission. Clay falls back to importing
    ``model.glb`` because a Clay document *is* geometry, so the mesh is a
    lesser but honest version of the document. A Mason document is an
    arrangement, and its merged mesh is not a lesser scene -- it is one mesh
    node where there were sixty, with the groups, the instances, the lights and
    the links gone. Opening that and calling it the scene would show the user
    finished work that is not there and let them save over it, which is exactly
    what ``.wblk``'s refuse-rather-than-substitute rule is about. So the door is
    offered only for a row that carries ``params["authored"] == "mason"``, and
    if the sidecar has gone anyway the task raises and the failure is reported.
    """
    job_id = job["id"] if isinstance(job, dict) else str(job)
    name = (job.get("name") if isinstance(job, dict) else "") or "Scene"
    ensure(ctx)

    def run() -> dict[str, Any]:
        from ....service import files as svc_files
        from ....service.errors import invalid_from
        from .engine import serialize

        path = svc_files.mason_source_path(ctx.svc, job_id)
        data = _within_ceiling(Path(path)).read_bytes()
        try:
            doc = serialize.read_wscn(data)
        except ValueError as exc:
            raise invalid_from(exc, "This scene could not be reopened", field="file") from exc
        return {"doc": doc, "title": name, "job_id": job_id, "view": doc.view or None}

    ctx.submit(f"mason-reopen:{job_id}", run)


# --- scene stats ------------------------------------------------------------------


#: A memo for :func:`scene_stats`, keyed on the document itself rather than
#: ``id(doc)``: the 2026-09-14 audit's mason-05 found ``mason_bridge._facts``
#: and ``mason_hud.stats_overlay`` each calling this every frame with no memo
#: of its own, each running its own ``scene.resolve(doc, include_hidden=True)``
#: -- about triple the resolve cost at ``PLACED_WARN_THRESHOLD`` once
#: ``MasonView.resolved()``'s own per-(id(doc), doc.rev) memo is counted in.
#: This function cannot simply share that cache -- it needs
#: ``include_hidden=True`` and the view's memo never keeps a hidden node --
#: so it keeps its own, small, keyed the same shape: the document's own
#: ``rev``. A ``WeakKeyDictionary`` rather than a plain dict keyed by
#: ``id(doc)``: scenes open and close all session, and an id-keyed dict would
#: either leak one entry per closed document forever or, worse, let a fresh
#: document that happened to reuse a freed id read a stale answer.
_STATS_CACHE: weakref.WeakKeyDictionary[Any, tuple[int, dict[str, Any]]] = (
    weakref.WeakKeyDictionary()
)


def scene_stats(ctx: Any, tab: Any) -> dict[str, Any]:
    """How big a scene is, against the engine's own constant -- never a number
    typed in a pane."""
    from .engine import scene as msc

    if tab is None:
        return {"placed": 0, "warn": False, "threshold": msc.PLACED_WARN_THRESHOLD, "missing": 0}
    doc = tab.doc
    cached = _STATS_CACHE.get(doc)
    if cached is not None and cached[0] == doc.rev:
        return cached[1]
    try:
        placed = len(msc.resolve(doc, include_hidden=True))
    except ValueError:
        placed = 0
    stats = {
        "placed": placed,
        "warn": placed >= msc.PLACED_WARN_THRESHOLD,
        "threshold": msc.PLACED_WARN_THRESHOLD,
        "missing": len(doc.missing_refs()),
    }
    _STATS_CACHE[doc] = (doc.rev, stats)
    return stats


# --- task results -----------------------------------------------------------------


def on_task_done(ctx: Any, done: Any) -> None:
    """Called from the app for every ``mason-`` key.

    ``mason_assets`` gets first refusal: a ``mason-asset:`` key is its
    background parse, never a document task, and it claims that prefix
    itself (see its own ``on_task_done``).
    """
    if mason_assets.on_task_done(ctx, done):
        return

    state = ensure(ctx)
    key, result = done.key, done.result
    name = key.split(":", 1)[0]

    if name == "mason-open":
        if isinstance(result, dict):
            adopt(
                ctx,
                result["doc"],
                path=Path(result["path"]),
                title=result.get("title"),
                view=result.get("view"),
            )
            _enter_mason(ctx)
        return

    if name == "mason-import":
        if isinstance(result, dict):
            place_job(ctx, {"id": result["job_id"], "name": result.get("name") or ""})
            ctx.cache.invalidate()
        return

    if name == "mason-reopen":
        if isinstance(result, dict):
            # ``path=None``: the row is not a file on disk the user chose, so
            # there is nothing to put in the recent list and nothing a plain
            # Save could write over -- ``plotter_mode``'s reopen takes the same
            # floor. ``job_id`` is carried onto the tab so a second export from
            # the reopened scene can say which row it last became.
            tab = adopt(
                ctx,
                result["doc"],
                path=None,
                title=result.get("title"),
                view=result.get("view"),
            )
            tab.job_id = str(result.get("job_id") or "")
            _enter_mason(ctx)
        return

    if name == "mason-recover":
        if result is None:
            journal.adopt_failed(ctx, "scene")
            return
        if isinstance(result, dict):
            tab = adopt(ctx, result["doc"], path=None, title=result.get("title"))
            docmodes.mark_recovered(tab, result["autosave"])
            _enter_mason(ctx)
        return

    tab = state.get(key.split(":", 1)[1]) if ":" in key else None
    if tab is None:
        return
    tab.saving = False
    if not isinstance(result, dict):
        return  # a cancelled dialog

    if result.get("exported_asset"):
        # The thumbnail is the App's: it is an offscreen GL draw and belongs on
        # the frame thread, so ``main`` takes it from the same ``done`` this
        # arm is reading. What is left here is the tab's own memory of which
        # row it became, which the Document pane reports.
        tab.job_id = str(result.get("job_id") or "")
        # The library draws from the cache, so a row minted behind its back is
        # invisible until something invalidates it -- ``plotter_mode``'s own
        # arm does exactly this, and without it the user is told the export
        # worked and finds nothing in the workshop.
        ctx.cache.invalidate()
        ctx.toast("Exported to the library.")
        return

    if result.get("exported"):
        skipped = result.get("skipped") or []
        if skipped:
            ctx.toast("Exported, with " + "; ".join(skipped), "warn")
        else:
            ctx.toast("Exported.")
        return

    tab.mark_saved(result.get("rev"))
    journal.drop(ctx, tab)
    if result.get("retitle") and result.get("path"):
        tab.path = Path(result["path"])
        tab.title = mason_state.title_for(tab.path)
        remember_path(ctx, tab.path)
        persist(ctx)
    ctx.toast("Saved.")


def on_task_failed(ctx: Any, done: Any) -> None:
    """A failed save must not leave the document locked -- ``clay_mode``'s
    own reason."""
    state = ctx.state.mason
    if state is None or ":" not in done.key:
        return
    tab = state.get(done.key.split(":", 1)[1])
    if tab is not None:
        tab.saving = False


# --- the guard ------------------------------------------------------------------


def guard(ctx: Any, verb: str, proceed: Any) -> bool:
    return docmodes.guard(ctx, "mason", "scene", "scenes", verb, proceed)


def close_tab(ctx: Any, uid: str) -> None:
    """Close one scene, asking first if it has unsaved work.

    What a Mason document owns in the single GL context: the active tab's
    cache in ``ctx.mason_view``, cleared after cancelling a live drag.
    """
    state = ensure(ctx)

    def release(tab: MasonTab) -> None:
        view = getattr(ctx, "mason_view", None)
        if view is not None and tab.uid == state.active_uid:
            if getattr(view, "dragging", False):
                view.cancel_drag(tab.doc)
            view.clear()

    docmodes.close_tab(ctx, state, uid, release)


# --- keys -------------------------------------------------------------------------

# Built from ``mason_state.TOOLS`` so the keyboard binding and the tool list
# can never name two different tools for the same letter.
TOOL_KEYS = {shortcut.lower(): key for key, _label, shortcut in mason_state.TOOLS}

# The 2026-09-14 audit's mason-07: this used to also list "i" and "m", which
# _ctrl_key has no arm for and shortcuts.py's Mason table never advertises --
# dead entries that blocked nothing a user could actually trigger while a
# save was in flight.
_MUTATING_CTRL = docmodes.WRITE_CHORDS | frozenset({"a", "j"})
_DRAG_BLOCKED_CTRL = frozenset({"z", "y", "n", "o", "tab", "w", "s"})


def undo(ctx: Any, tab: Any) -> None:
    tab.doc.undo()


def redo(ctx: Any, tab: Any) -> None:
    tab.doc.redo()


def step_history(ctx: Any, tab: Any, index: int) -> bool:
    del ctx
    return tab.doc.step_history(index)


def handle_key(ctx: Any, event: Any) -> bool:
    """Mason's shortcuts. -> whether the key was consumed.

    **False with nothing open**, which the caller's fall-through depends on:
    Mason owns a viewport, and with no document the viewport's own shortcuts
    must still work.
    """
    import pygame

    if event.type == pygame.KEYDOWN and pygame.key.name(event.key) == "f" and not (
        event.mod & (pygame.KMOD_CTRL | pygame.KMOD_ALT)
    ):
        ensure(ctx).frame_pending = True
        return True

    state = ctx.state.mason
    if state is None or not state.docs:
        return False
    tab = state.active
    if tab is None:
        return False
    doc = tab.doc

    if event.type != pygame.KEYDOWN:
        return True

    mods = event.mod
    ctrl = bool(mods & pygame.KMOD_CTRL)
    shift = bool(mods & pygame.KMOD_SHIFT)
    name = pygame.key.name(event.key)

    if ctrl:
        return _ctrl_key(ctx, state, tab, doc, name, shift=shift)

    if name in TOOL_KEYS and not shift:
        state.tool = TOOL_KEYS[name]
    elif name == "g" and not tab.saving:
        ungroup_selected(ctx) if shift else group_selected(ctx)
    elif event.key == pygame.K_DELETE:
        if not tab.saving:
            delete_selected(ctx)
    elif event.key == pygame.K_ESCAPE:
        # Disarm first, and clear the selection only if nothing was armed. One
        # key, two jobs, in the order the user means them: Esc after arming a
        # light is "not that after all", and it must not also throw away the
        # selection they are about to place it beside. The hint line promises
        # exactly this ("Esc to cancel") while a placement is armed.
        if state.place_kind or state.place_prefab:
            state.place_kind = ""
            state.place_prefab = ""
        else:
            doc.select([])
    return True


def _ctrl_key(
    ctx: Any, state: MasonState, tab: MasonTab, doc: Any, name: str, *, shift: bool
) -> bool:
    if docmodes.blocked_while_writing(tab, name, _MUTATING_CTRL):
        return True
    view = getattr(ctx, "mason_view", None)
    if getattr(view, "dragging", False) and name in _DRAG_BLOCKED_CTRL:
        return True
    if name in AXIS_VIEW_KEYS or name == "5":
        if view is not None:
            axis_view_key(view.camera, name, shift)
    elif name == "s":
        save_as(ctx, tab) if shift else save(ctx, tab)
    elif name == "o":
        ask_open(ctx)
    elif name == "n":
        new_document(ctx)
    elif name == "w":
        close_tab(ctx, tab.uid)
    elif name == "e" and not shift:
        export_glb(ctx, tab)
    elif name == "z":
        redo(ctx, tab) if shift else undo(ctx, tab)
    elif name == "y":
        redo(ctx, tab)
    elif name == "a":
        doc.select([n.uid for n in doc.all_nodes()])
    elif name == "j" and not tab.saving:
        duplicate_selected(ctx)
    elif name == "d" and not tab.saving:
        doc.select([])
    elif name == "tab":
        state.cycle(-1 if shift else 1)
    return True


# --- crash recovery (UX-05) ---------------------------------------------------


def _journal_encode(tab: Any) -> bytes:
    from .engine import serialize

    tab.doc.view = {
        **{name: getattr(tab.view, name) for name in serialize.VIEW_FIELDS},
        "target": tuple(tab.view.target),
    }
    return serialize.wscn_bytes(tab.doc)


def _journal_adopt(ctx: Any, path: Path, meta: dict[str, Any]) -> bool:
    """Reopen one recovered ``.wscn`` as an *untitled, dirty* document --
    ``clay_mode._journal_adopt``'s reason: the file it was copied from may
    still be on disk with its own contents, and adopting the path would arm
    Ctrl+S to overwrite something the user has not looked at."""
    ensure(ctx)
    ctx.submit(f"mason-recover:{_path_key(path)}", _load_recovery, Path(path), dict(meta))
    return True


def _load_recovery(path: Path, meta: dict[str, Any]) -> dict[str, Any] | None:
    from .engine import serialize

    try:
        doc = serialize.read_wscn(_within_ceiling(path).read_bytes())
    except Exception:
        log.exception("could not reopen the recovered scene at %s", path)
        return None
    return {
        "doc": doc,
        "title": f"{meta.get('title') or path.stem} (recovered)",
        "autosave": str(path),
    }


JOURNAL = journal.tab_provider(
    "mason", ".wscn", "scene", encode=_journal_encode, adopt=_journal_adopt
)


def release_all(ctx: Any) -> None:
    """Teardown door: release the viewport's GL state. Every mode's own
    tabs stay in memory -- this is the GL half, the part that must not
    outlive the context."""
    view = getattr(ctx, "mason_view", None)
    if view is not None:
        view.release()
    source = getattr(ctx, "mason_assets", None)
    if source is not None:
        source.release()
