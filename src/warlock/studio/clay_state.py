"""Multi-document state for Clay, without imgui.

The same split the raster editor makes, for the same reason: which document is
open, which is dirty, where the camera is and what the snap settings are would
all still make sense if the app were driven by a script, so none of it needs a
window to be tested.

The two conventions travel across from the raster editor unchanged, because
they are what a user arrives expecting. **Tool and snap settings belong to the
app, not to the document** -- switching tabs must not silently change your grid
size. And **the view belongs to the document** -- a tab remembers where its
camera was, because orbiting back to where you were working is not something
the user should have to redo on every tab switch.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import docmodes

# Name, label, and the key that selects it. The primitive tools mirror the
# generator registry; the transform tools mirror the three gizmos.
TOOLS = (
    ("select", "Select", "Q"),
    ("move", "Move", "W"),
    ("rotate", "Rotate", "E"),
    ("scale", "Scale", "R"),
)

# The snap increments a modelling package opens on: an eighth of a unit, and
# fifteen degrees. Both are *app* settings, and both are switched off by
# setting them to zero -- which is why every snap function treats zero as the
# identity rather than as a degenerate case to guard.
DEFAULT_SNAP_TRANSLATE = 0.125
DEFAULT_SNAP_ROTATE = 15.0

WBLK_SUFFIX = ".wblk"


# ``LastOp`` and ``ClayState.last_op`` lived here until the 2026-09-07 audit's
# clay-10: written by every ``clay_ops.run`` call and read by no pane -- the
# "adjust last operation" card and Repeat it was recorded for were never
# built. Removed rather than kept on the chance a future card wants it; a
# card that does can rebuild the record from the undo stack it would need
# anyway.

_uids = itertools.count(1)


#: Where this document's camera sits. **Moved to :mod:`.docmodes`** when
#: Mason's viewport wanted the identical class: the four numbers and the
#: goal-field rule in ``read_from``/``write_to`` are genuinely one rule for
#: every 3-D document mode, and a second copy is a second place that rule can
#: quietly stop being true. Re-exported under this name because it is the name
#: Clay's tabs, its tests and ``clay_mode`` all reach for -- the class moved,
#: the spelling did not.
CameraView = docmodes.CameraView


@dataclass
class ClayTab:
    """One open document.

    ``uid`` is stable and never reused, because imgui identifies a tab by its
    label: a title alone would make two documents called "Untitled" the same
    tab, and would move a tab's identity every time a Save As renamed it.
    """

    doc: Any
    title: str = "Untitled"
    path: Path | None = None
    uid: str = field(default_factory=lambda: f"bd{next(_uids)}")
    view: CameraView = field(default_factory=CameraView)
    # The history position the file on disk was written from. Dirty is a
    # *comparison*, not a flag, so undoing back to the saved state correctly
    # stops being dirty -- which the document's revision cannot express,
    # because it counts changes and an undo is one.
    saved_head: int = 0
    saving: bool = False

    # Crash-safety, owned by :mod:`studio.journal` (UX-05). Inker's three
    # fields, verbatim, because they are the same three questions: which file
    # this tab owns under the autosave directory (minted on the first copy, so
    # an untouched tab litters nothing), the history position that copy
    # captured (an undo back to it is not a new edit), and the debounce.
    journal_name: str = ""
    journal_head: int | None = None
    journal_at: float = 0.0
    # The asset this document was last exported to, if any. Not a link in the
    # raster editor's sense: a built asset is a *snapshot*, and editing the
    # document afterwards does not change the mesh already on disk.
    job_id: str = ""

    @property
    def dirty(self) -> bool:
        return self.doc.history.head != self.saved_head

    @property
    def label(self) -> str:
        return docmodes.tab_label(self)

    def mark_saved(self, head: int | None = None) -> None:
        """Record which history position is now on disk.

        Captured when the *encode* starts, not when it finishes: an edit made
        while the file was being written is genuinely not in it, and clearing a
        flag here would call it saved.
        """
        self.saved_head = self.doc.history.head if head is None else head
        self.saving = False


def title_for(path: Path | None) -> str:
    """The stem, not the name -- the one mode that differs from
    ``docmodes.title_for`` on purpose. A Clay tab is named for the *document*
    (``barrel``) rather than the file (``barrel.wblk``): the export beside it
    is ``barrel.glb`` and the library row is ``barrel``, and the suffix would
    be the only one of the three saying ``.wblk``."""
    return path.stem if path is not None else "Untitled"


@dataclass
class ClayState:
    """Everything Clay remembers across frames."""

    docs: list[ClayTab] = field(default_factory=list)
    active_uid: str = ""
    #: ``F`` has been pressed and the viewport has not framed yet.
    #:
    #: A flag rather than a call, the house pattern (``plotter_state``'s
    #: ``setup_pending``, Inker's ``pending_dialog``): framing needs the
    #: viewport, which is a thing ``main`` owns and ``clay_mode`` may not
    #: import, so the mode records the *intent* and the pane consumes it. What
    #: it replaces is a branch in ``App._shortcut`` that reached past
    #: ``clay_mode.handle_key`` to do the same thing -- the one Clay binding
    #: that did not live with the others.
    frame_pending: bool = False

    # Tool and snap settings: shared across documents on purpose.
    tool: str = "select"
    snap: bool = False
    snap_translate: float = DEFAULT_SNAP_TRANSLATE
    snap_rotate: float = DEFAULT_SNAP_ROTATE
    # Snap a move onto the vertex under the cursor, in preference to the grid.
    # A *separate* switch rather than a mode of ``snap``, because the two answer
    # different questions -- "put it on round numbers" and "put it exactly
    # there" -- and a user aligning two parts wants the second without giving up
    # the first everywhere else. Off by default: it changes what a plain drag
    # does, and a viewport that silently jumps is worse than one that does not.
    snap_vertex: bool = False

    # Proportional editing: an element drag carries the geometry around the
    # selection with it, fading out over ``proportional_radius`` metres of world
    # space. Off by default and radius-driven rather than count-driven, because
    # a radius is the thing the user can see -- a "how many rings" control means
    # nothing on an imported mesh whose density varies across it.
    proportional: bool = False
    proportional_radius: float = 0.5
    grid: bool = True
    # The grid's own size, in metres -- a user setting rather than something
    # derived from the document, unlike Mason's, Poser's and the asset
    # viewer's grids (``viewer/grid.py``'s ``span_for``). 100 m at 1 m cells
    # by default: room enough for most props with no resize needed, and a
    # round unit a user reads at a glance. Persisted with ``grid`` (see
    # ``clay_mode.persist``) rather than reset every launch, and **not**
    # touched by ``F`` -- ``_view_bounds.BoundsOps.frame_selection`` frames the
    # camera on the selection and deliberately leaves this alone, which is the
    # bug this field exists to fix (Task A, 2026-09-12).
    grid_size: float = 100.0
    # A directional light straight down onto a ground plane under the grid,
    # replacing the ordinary key light rather than adding to it -- see
    # ``viewer.env.Environment.god_light`` and ``viewer.render.Renderer.
    # light_override``. Off by default: it is a deliberately flat, shadowless
    # look for checking silhouette and proportions, not the render most
    # editing happens under. Persisted alongside ``grid``/``grid_size``.
    god_light: bool = False

    # How the surface itself is drawn: "solid", "material" or "wireframe".
    #
    # Three modes rather than the one wireframe toggle this replaced, and the
    # addition that matters is **Solid** -- the albedo with no lighting, which
    # is what a modeller works in: it shows silhouette and topology without a
    # specular highlight sitting on the vertex being dragged. Material is the
    # lit render and is what the object will look like; wireframe replaces the
    # fill entirely.
    shading: str = "material"

    # The see-through pass. Off by default because it changes what a click
    # picks as well as what is drawn -- an element behind the surface becomes
    # reachable, which is the whole point and is also a surprise if it happens
    # without being asked for.
    xray: bool = False

    # What the viewport draws *over* the model, by name. A dict rather than a
    # field apiece because the header's popover is a loop over
    # ``clay_header.OVERLAY_ROWS`` and a sixth overlay should be one line there
    # rather than one line in four files.
    #
    # ``grid`` is deliberately **not** in it: it is wired straight to
    # ``ClayView.show_grid`` and has been since the viewport existed, and a
    # second home for one switch is two places that can disagree about it.
    overlays: dict[str, bool] = field(default_factory=lambda: {"wire": False})

    # What the properties panel offers when the user adds something -- the
    # key of whichever add-tool (a primitive or a figure) was last pressed.
    #
    # ``""`` means *nothing yet*, not Box. clay-12 (2026-09-08 audit, second run): this
    # used to default to ``"box"``, so a session that had never touched an
    # add-tool showed the grid's Box icon lit and its defaults printed in the
    # options block below it -- state that reads as a click nobody made. No
    # key in ``primitives.GENERATORS`` or ``presets.ASSEMBLIES`` is ever the
    # empty string, so this sentinel can never collide with a real tool's name
    # and every reader that compares against a key (the icon grid's
    # "selected" highlight, ``clay_tools._options_for``) already treats it as
    # "select nothing" without needing to know it is special.
    generator: str = ""
    # The object whose name is being edited in the outliner, or 0. A uid rather
    # than an index, for the reason every address in this package is one: the
    # outliner reorders and a rename in flight must not follow the position.
    renaming: int = 0

    # Drag state. ``ref`` is what the handle was grabbed at, so a drag is
    # measured against the press rather than against the previous frame --
    # which is also what feeds ``set_transform``'s ``was`` argument, and the
    # reason the gizmo itself has to remember nothing about the object.
    # No ``drag_kind`` here: a live drag is the *view*'s (``ClayView._grab`` and
    # ``_key_kind``), and the field that used to sit here was written by nothing
    # but ``clear_drag`` -- read once, by the hint line, which therefore never
    # showed a drag.
    drag_axis: str = ""
    ref: dict[str, Any] = field(default_factory=dict)

    # A hook the app wires once a live ``ClayView`` exists (``clay_mode.
    # ensure``), so a tab switch can settle a live G/R/S/gizmo drag against
    # the document being *left* before ``active_uid`` moves out from under
    # it. Called with the outgoing tab's uid, from :meth:`activate` only.
    #
    # The 2026-09-15 audit's clay-01: a live drag has already written TRS
    # onto the object in place, and nothing records that until the drag's
    # own commit or cancel runs -- which, with no hook, happened later
    # (mouse-up, Esc) against whichever document *became* active rather than
    # the one the drag began on. ``clay_mode.close_tab``'s ``release``
    # already cancelled the drag first for the tab-*closing* case; this hook
    # is the tab-*switching* half, needed because :meth:`activate` (unlike
    # ``close_tab``) is reached from ``docmodes.tab_bar``'s click handler and
    # ``cycle``'s Ctrl+Tab with no view in scope to cancel or commit against.
    #
    # A callable rather than a direct import of ``ClayView``: this class
    # stays driveable with no window at all, which is the whole point of the
    # split this module's own docstring describes -- ``None`` here is a
    # ClayState under script or test control with nothing to settle.
    settle_drag: Any = field(default=None, repr=False, compare=False)

    # The parameterised op whose popup is open, by name, and the values every
    # such op was last run with. Remembered per op rather than per invocation:
    # a user beveling six edges in turn wants the same width each time, and
    # retyping it is the whole reason a modeller keeps the last value.
    pending_op: str = ""
    op_params: dict[str, dict[str, float]] = field(default_factory=dict)
    # Set when something outside the pane wants that popup *opened*, because
    # ``imgui.open_popup`` only takes effect inside the window whose id stack
    # is current -- the keyboard path runs in the event layer, where there is
    # no window at all. Without it a bare-letter key bound to a parameterised
    # op set ``pending_op`` and nothing ever opened the popup, leaving the mode
    # holding a request it could not act on. Cleared by the pane that opens it.
    open_op_popup: bool = False

    # Where a Shift+click range in the outliner is measured from. A uid, for the
    # reason every address in this package is one: the list reorders, and an
    # anchor that was an index would silently point at a different row.
    outliner_anchor: int = 0

    # The last manifold check, per object: the ``Mesh`` it measured and the rows
    # it produced. Held here rather than recomputed because ``check_manifold``
    # builds a whole adjacency -- O(corners), and not something to run sixty
    # times a second to redraw a line that has not changed.
    #
    # **Keyed on the mesh object, not on a revision or an id.** A ``Mesh`` is
    # immutable and every op replaces it, so ``obj.mesh is measured`` is exactly
    # "this result is still about what is on screen"; an ``id()`` would be
    # recycled by the allocator onto a different mesh and silently report last
    # edit's holes. Keeping the mesh alive is the price, and it is one mesh per
    # object the user has actually asked about.
    manifold: dict[int, tuple[Any, list[Any]]] = field(default_factory=dict)

    # -- documents ---------------------------------------------------------

    @property
    def active(self) -> ClayTab | None:
        for doc in self.docs:
            if doc.uid == self.active_uid:
                return doc
        return self.docs[-1] if self.docs else None

    @property
    def any_dirty(self) -> bool:
        return any(doc.dirty for doc in self.docs)

    def add(self, tab: ClayTab) -> ClayTab:
        # ``activate`` was fixed to settle a live drag on the tab it leaves
        # before moving ``active_uid`` (2026-09-15 audit, clay-01); ``add``
        # moves ``active_uid`` exactly the same way and was left out, so
        # creating, opening, importing or auto-recovering a document mid-drag
        # (New/Open have no drag gate in the UI at all, and the async adopt
        # paths are inherently decoupled from whatever drag is live when
        # their result lands) left the old tab's TRS mutated in place with no
        # history step behind it -- unrevertable (2026-09-16 audit). Guarded
        # the same way ``activate`` is: a fresh ``ClayState`` has no
        # ``settle_drag`` yet, and the very first ``add()`` has no
        # ``active_uid`` to settle.
        if self.settle_drag is not None and self.active_uid:
            self.settle_drag(self.active_uid)
        self.docs.append(tab)
        self.active_uid = tab.uid
        self.clear_drag()
        return tab

    def get(self, uid: str) -> ClayTab | None:
        for doc in self.docs:
            if doc.uid == uid:
                return doc
        return None

    def close(self, uid: str) -> bool:
        tab = self.get(uid)
        if tab is None:
            return False
        index = self.docs.index(tab)
        self.docs.remove(tab)
        if self.active_uid == uid:
            # The neighbour, not the first: closing a tab should leave you next
            # to where you were rather than at the far end of the bar.
            self.active_uid = self.docs[min(index, len(self.docs) - 1)].uid if self.docs else ""
        self.clear_drag()
        return True

    def activate(self, uid: str) -> None:
        if uid != self.active_uid:
            if self.settle_drag is not None:
                self.settle_drag(self.active_uid)
            self.active_uid = uid
            self.clear_drag()

    def cycle(self, step: int = 1) -> None:
        if len(self.docs) < 2:
            return
        current = self.active
        index = self.docs.index(current) if current in self.docs else 0
        self.activate(self.docs[(index + step) % len(self.docs)].uid)

    def find_path(self, path: Path) -> ClayTab | None:
        """``docmodes.find_path``: the one case-folding body every mode shares."""
        return docmodes.find_path(self.docs, path)

    # -- drag ---------------------------------------------------------------

    def clear_drag(self) -> None:
        self.drag_axis = ""
        self.ref = {}

