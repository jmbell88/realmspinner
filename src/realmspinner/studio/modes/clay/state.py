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
from typing import TYPE_CHECKING, Any

from ... import docmodes

if TYPE_CHECKING:
    # Deferred: ``ui.panes.uv`` imports ``mode``, which imports *this* module
    # for ``ClayState``/``ClayTab`` -- a module-scope import here would be a
    # cycle. ``from __future__ import annotations`` already makes the
    # ``ClayTab.uv_view`` annotation below a string, so this import only ever
    # runs for a type checker, never at class-definition time.
    from .ui.panes.uv import UvPaneState

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

RBLK_SUFFIX = ".rblk"


# ``LastOp`` and ``ClayState.last_op`` lived here until the 2026-09-07 audit's
# clay-10: written by every ``clay_ops.run`` call and read by no pane. Removed
# rather than kept on the chance a card wanted it. The card and Repeat Last now
# exist, and so does a record -- ``recent_op.RecentOp`` on the *document*
# (``ClayDoc.recent_op``), not here: ``Op.enabled`` only ever sees the document,
# and an agent's own tab must repeat only its own ops.

_uids = itertools.count(1)


#: Where this document's camera sits. **Moved to :mod:`.docmodes`** when
#: Mason's viewport wanted the identical class: the four numbers and the
#: goal-field rule in ``read_from``/``write_to`` are genuinely one rule for
#: every 3-D document mode, and a second copy is a second place that rule can
#: quietly stop being true. Re-exported under this name because it is the name
#: Clay's tabs, its tests and ``clay_mode`` all reach for -- the class moved,
#: the spelling did not.
CameraView = docmodes.CameraView


def _new_uv_view() -> UvPaneState:
    """``ClayTab.uv_view``'s default factory.

    A local import rather than a module-scope one: ``UvPaneState`` lives in
    ``ui/panes/uv.py`` (it is pane state, not document state -- moved there
    2026-09-19 so this module keeps its one-tab-state-class-per-mode shape,
    see ``test_docmodes.py::test_every_document_mode_inherits_the_one_tab_
    list``), and that module imports ``mode``, which imports *this* module
    for ``ClayState``/``ClayTab``. Importing it at module scope here would be
    a cycle; importing it lazily, on first ``ClayTab()`` construction -- long
    after both modules have finished loading -- is not.
    """
    from .ui.panes.uv import UvPaneState

    return UvPaneState()


@dataclass
class ClayTab(docmodes.HistoryTab):
    """One open document (``docmodes.DocTab`` holds the shared fields)."""

    uid: str = field(default_factory=lambda: f"bd{next(_uids)}")
    view: CameraView = field(default_factory=CameraView)
    # The asset this document was last exported to, if any. Not a link in the
    # raster editor's sense: a built asset is a *snapshot*, and editing the
    # document afterwards does not change the mesh already on disk.
    job_id: str = ""

    # Tranche 6: the UV pane's own pan/zoom and island selection. See
    # ``UvPaneState``'s own docstring (``ui/panes/uv.py``) for why it is a
    # nested dataclass beside ``view`` rather than loose fields here, and
    # ``_new_uv_view``'s above for why the default factory imports it lazily.
    uv_view: UvPaneState = field(default_factory=_new_uv_view)

    # The Inker documents whose pixels flow into this document's palette entries
    # (``texture_link.InkerLink``): one per "Edit texture in Inker" or "Take
    # texture back". Typed ``Any`` because ``texture_link`` imports the Clay
    # document types and this module is imported first; session state only, a
    # saved ``.rblk`` carries the texture, never the link.
    inker_links: list[Any] = field(default_factory=list)


def title_for(path: Path | None) -> str:
    """The stem, not the name -- the one mode that differs from
    ``docmodes.title_for`` on purpose. A Clay tab is named for the *document*
    (``barrel``) rather than the file (``barrel.rblk``): the export beside it
    is ``barrel.glb`` and the library row is ``barrel``, and the suffix would
    be the only one of the three saying ``.rblk``."""
    return path.stem if path is not None else "Untitled"


@dataclass
class ClayState(docmodes.DocTabs[ClayTab]):
    """Everything Clay remembers across frames."""

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

    #: The tab whose camera the shared viewport is showing right now -- written
    #: by ``ui/viewport.py`` after its handoff. The 2026-10-03 audit's clay-12:
    #: the per-frame camera sync keyed on ``active_uid``, which a keyboard tab
    #: switch or a close moves before the draw, so it wrote the outgoing tab's
    #: live camera onto the incoming tab. ``None`` until something is drawn.
    camera_tab: Any = None

    # Tool and snap settings: shared across documents on purpose.
    tool: str = "select"
    snap: bool = False
    snap_translate: float = DEFAULT_SNAP_TRANSLATE
    snap_rotate: float = DEFAULT_SNAP_ROTATE
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
    # key of whichever add-tool (a primitive) was last pressed.
    #
    # ``""`` means *nothing yet*, not Box. clay-12 (2026-09-08 audit, second run): this
    # used to default to ``"box"``, so a session that had never touched an
    # add-tool showed the grid's Box icon lit and its defaults printed in the
    # options block below it -- state that reads as a click nobody made. No
    # key in ``primitives.GENERATORS`` is ever the
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

    # Which tab of the Properties pane is showing (``props.TABS`` keys). App
    # level rather than per document, like ``tool``: it is a place you are
    # *looking*, and switching documents should not move you back to Object.
    # An unknown key (a tab since removed) is read as ``"object"`` by the pane.
    props_tab: str = "object"

    # Where a Shift+click range in the outliner is measured from. A uid, for the
    # reason every address in this package is one: the list reorders, and an
    # anchor that was an index would silently point at a different row.
    outliner_anchor: int = 0
    # Tranche 3: scene structure. Which rows' subtrees the outliner's own
    # expander has hidden, by uid -- the same address rule as the anchor
    # above, and for the same reason: a set of *indices* would point at
    # whatever now sits at that row once the tree reorders. Not persisted:
    # a document reopened expanded is the same "nothing remembered" default
    # every other transient view setting in this class gets.
    outliner_collapsed: set[int] = field(default_factory=set)

    # Units and up-axis for the next mesh import (the "Import Mesh..." button
    # and a file dropped onto the viewport both read these), remembered across
    # imports the way a modelling package's own import dialog does -- a
    # session importing a batch of centimetre-scale, Z-up assets should not
    # retype both on every file. Not persisted to settings: unlike the view
    # block, a scale/axis choice is about the *files* a session happens to be
    # importing today, not a preference to carry into a different one.
    import_scale: float = 1.0
    import_up: str = "y"

    # The Properties panel's transform block (``ui/panes/props._transform``).
    # All three are display state, never document state -- storage stays in
    # metres and quaternions -- and all three are app level rather than per
    # tab, ``tool``'s reason: they are how the user is *looking*.
    #
    # ``euler_cache`` is what makes a typed rotation survive its own frame:
    # see ``transform_edit``'s module docstring. Keyed by uid, like every
    # address here, and tiny (one entry per object whose rotation was shown).
    length_unit: str = "m"
    # Why the adjust card's last re-run was refused ("" when it was not), kept
    # here so the card can print it on the frames after the refusal. Cleared
    # by the card itself the moment the recent op stops being live.
    adjust_message: str = ""
    size_lock_aspect: bool = False
    euler_cache: dict[int, tuple[tuple[float, ...], tuple[float, float, float]]] = field(
        default_factory=dict
    )

    # -- documents ---------------------------------------------------------

    def _switched(self, previous: str) -> None:
        # ``activate`` was fixed to settle a live drag on the tab it leaves
        # (2026-09-15 audit, clay-01); ``add`` moves ``active_uid`` exactly the
        # same way and was left out, so creating, opening, importing or
        # auto-recovering a document mid-drag (New/Open have no drag gate in
        # the UI at all, and the async adopt paths are inherently decoupled
        # from whatever drag is live when their result lands) left the old
        # tab's TRS mutated in place with no history step behind it --
        # unrevertable (2026-09-16 audit). One hook for both now. A fresh
        # ``ClayState`` has no ``settle_drag`` yet, and the very first
        # ``add()`` has no previous tab to settle.
        if self.settle_drag is not None and previous:
            self.settle_drag(previous)
        self.clear_drag()

    def _closed(self, was_active: bool) -> None:
        self.clear_drag()

    # -- drag ---------------------------------------------------------------

    def clear_drag(self) -> None:
        self.drag_axis = ""
        self.ref = {}

