"""Multi-document state for Mason, without imgui.

The same split Clay makes, for the same reason: which scene is open, which is
dirty, where the camera is and what the transform tool and its snap settings
are would all still make sense if the app were driven by a script, so none of
it needs a window to be tested.

The two conventions travel across from the raster editor and from Clay
unchanged, because they are what a user arrives expecting. **Tool and snap
settings belong to the app, not to the document** -- switching tabs must not
silently change your grid size. And **the view belongs to the document** -- a
tab remembers where its camera was, because orbiting back to where you were
working is not something the user should have to redo on every tab switch.

Mason needs a *scene* state rather than an object one because its document is
an arrangement of many placed things -- primitives, library props, lights, a
terrain, a camera -- and not one mesh being sculpted. The tool set is the same
four letters as Clay's transform gizmos, but what they act on is a selection
of nodes rather than a selection of mesh elements, and the settings this state
carries reflect that: a pivot for a multi-node drag, a place-tool that can arm
a light or a camera as easily as a primitive, and no per-element proportional
falloff because there is no mesh here to fall off across.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ... import docmodes

WSCN_SUFFIX = ".wscn"

# Name, label, and the key that selects it. Clay's four tools, verbatim: the
# transform gizmos are one idea, not two, and a scene editor needs the same
# three plus Select.
#
# **Sculpt is a fifth tool rather than an armed placement**, and the difference
# is what the mouse does: placing a light is one click that ends, where a brush
# owns the left button for a sustained drag the way Move does -- and a mode the
# left button behaves differently in is exactly what a *tool* is. It draws no
# gizmo (``mason_view.GIZMO_FOR_TOOL`` has no entry), which is what leaves the
# button free for the stroke.
TOOLS = (
    ("select", "Select", "Q"),
    ("move", "Move", "W"),
    ("rotate", "Rotate", "E"),
    ("scale", "Scale", "R"),
    ("sculpt", "Sculpt", "T"),
)

#: The terrain brushes, by the ``mason.terrain`` function each one calls. Order
#: is the order they are offered in, and the labels are the whole of what the
#: Assets pane needs -- the arithmetic is that module's.
BRUSHES = (
    ("raise", "Raise", "Pull the ground up under the brush"),
    ("lower", "Lower", "Push it down"),
    ("smooth", "Smooth", "Blend each cell toward its neighbours"),
    ("flatten", "Flatten", "Pull the ground toward one height"),
    ("noise", "Noise", "Add falloff-shaped, seeded roughness"),
)

#: The side, in cells, a new terrain is created at -- a quarter of
#: ``terrain.MAX_TERRAIN_SIDE``, which is where that measurement put a full
#: mesh rebuild at a third of a frame. A sculpt drag rebuilds the mesh on every
#: frame it is open (the memo is keyed on the ``heights`` array a brush
#: rebinds), so the default is the resolution a drag is comfortable at and the
#: ceiling is what a user may raise it to knowing that.
DEFAULT_TERRAIN_SIDE = 64

#: How wide a new terrain is, in metres. Sixty-four metres over
#: ``DEFAULT_TERRAIN_SIDE`` cells is a metre a cell, which is the scale props
#: are placed at.
DEFAULT_TERRAIN_SIZE = 64.0

# The snap increments a scene editor opens on. A scene is placed in metres --
# a wall, a barrel, a lamp post -- and a quarter-metre grid is the useful
# increment for props at that scale, where Clay's eighth-of-a-unit is a
# *modelling* increment for the mesh sitting inside one of them. Fifteen
# degrees carries over unchanged: rotation snap is not a function of scale.
DEFAULT_SNAP_TRANSLATE = 0.25
DEFAULT_SNAP_ROTATE = 15.0

# The gizmo pivot for a multi-node drag: the selection's own centre, the last
# node clicked, or the world origin. Clay has no equivalent because an
# element drag inside one mesh has no second object to pivot around instead
# of the selection.
PIVOTS = ("median", "active", "origin")

_uids = itertools.count(1)


def title_for(path: Path | None) -> str:
    """The stem, not the name -- ``clay_state.title_for``'s reasoning applies
    unchanged: a Mason tab is named for the *scene* (``arena``) rather than
    for the file (``arena.wscn``), because the export beside it is
    ``arena.glb`` and the library row is ``arena``, and the suffix would be
    the only one of the three saying ``.wscn``."""
    return path.stem if path is not None else "Untitled"


@dataclass
class MasonTab:
    """One open scene.

    ``uid`` is stable and never reused, because imgui identifies a tab by its
    label: a title alone would make two scenes called "Untitled" the same
    tab, and would move a tab's identity every time a Save As renamed it.
    """

    doc: Any
    title: str = "Untitled"
    path: Path | None = None
    uid: str = field(default_factory=lambda: f"ms{next(_uids)}")
    view: docmodes.CameraView = field(default_factory=docmodes.CameraView)
    # The history position the file on disk was written from. Dirty is a
    # *comparison*, not a flag, so undoing back to the saved state correctly
    # stops being dirty -- which the document's revision cannot express,
    # because it counts changes and an undo is one.
    saved_head: int = 0
    saving: bool = False

    # Crash-safety, owned by :mod:`studio.journal`. Clay's three fields,
    # verbatim, because they are the same three questions: which file this tab
    # owns under the autosave directory (minted on the first copy, so an
    # untouched tab litters nothing), the history position that copy
    # captured (an undo back to it is not a new edit), and the debounce.
    journal_name: str = ""
    journal_head: int | None = None
    journal_at: float = 0.0
    # The asset this scene was last exported to, if any. Not a link in the
    # raster editor's sense: a built export is a *snapshot*, and editing the
    # document afterwards does not change what is already on disk.
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
        while the file was being written is genuinely not in it, and clearing
        a flag here would call it saved.
        """
        self.saved_head = self.doc.history.head if head is None else head
        self.saving = False


@dataclass
class MasonState:
    """Everything Mason remembers across frames."""

    docs: list[MasonTab] = field(default_factory=list)
    active_uid: str = ""
    #: ``F`` has been pressed and the viewport has not framed yet.
    #:
    #: A flag rather than a call, ``ClayState.frame_pending``'s pattern
    #: verbatim: framing needs the viewport, which is a thing ``main`` owns
    #: and ``mason_mode`` may not import, so the mode records the *intent*
    #: and the pane consumes it.
    frame_pending: bool = False

    # Tool and snap settings: shared across scenes on purpose.
    tool: str = "select"
    pivot: str = "median"
    snap: bool = False
    snap_translate: float = DEFAULT_SNAP_TRANSLATE
    snap_rotate: float = DEFAULT_SNAP_ROTATE
    # Drop a placed object onto the terrain (or the ground plane) rather than
    # onto the plane it was armed from. A separate switch rather than a mode
    # of ``snap``, for the reason Clay's ``snap_vertex`` gives: it answers a
    # different question -- "put it on the surface below" rather than "put it
    # on round numbers" -- and a user placing a row of props wants both
    # together as often as either alone.
    snap_ground: bool = False
    grid: bool = True

    # How the surface itself is drawn: "solid", "material" or "wireframe".
    # Clay's three-mode argument carries over unchanged: Material is the lit
    # render a scene will actually look like, and Solid strips the specular
    # highlight that sits on whatever is being dragged.
    shading: str = "material"

    # The see-through pass. Off by default because it changes what a click
    # picks as well as what is drawn -- a node behind a wall becomes
    # reachable, which is the whole point and is also a surprise if it
    # happens without being asked for.
    xray: bool = False

    # What the viewport draws *over* the scene, by name. A dict rather than a
    # field apiece for Clay's reason: a sixth overlay should be one line here
    # rather than one line in four files. ``grid`` is deliberately not in it,
    # for the same reason it is not in Clay's: it is wired straight to the
    # view's own grid switch, and a second home for one toggle is two places
    # that can disagree about it.
    overlays: dict[str, bool] = field(default_factory=lambda: {"wire": False})

    # What the Assets pane has armed to place on the next click: a
    # ``primitives.GENERATORS`` key, ``"light:point"``/``"light:spot"``/
    # ``"light:directional"``, ``"camera"``, or ``""`` for nothing armed.
    #
    # ``""`` means *nothing yet*, not the first entry in any of those lists --
    # ``ClayState.generator``'s clay-12 argument, restated: a session that has
    # never touched a place-tool must not show one lit and its options
    # printed for a click nobody made. No real key is ever the empty string,
    # so this sentinel can never collide with one.
    place_kind: str = ""

    # The node whose name is being edited in the outliner, or 0. A uid rather
    # than an index, for the reason every address in this package is one: the
    # outliner reorders and a rename in flight must not follow the position.
    renaming: int = 0

    # Where a Shift+click range in the outliner is measured from. A uid, for
    # the same reason: the list reorders, and an anchor that was an index
    # would silently point at a different row.
    outliner_anchor: int = 0

    # -- the terrain brush -------------------------------------------------
    #
    # App settings like the snap ones above and for the same reason: switching
    # scenes must not silently change the brush in your hand. A ``BRUSHES``
    # key, and the four numbers the five brushes between them read.
    brush: str = "raise"
    #: The brush's radius in *cells*, which is what ``terrain``'s brushes take
    #: -- they work in height-field index space, and the conversion from a
    #: world click to a cell centre is the viewport's.
    brush_radius: float = 6.0
    #: Metres per second of drag for raise/lower. Per *frame* would make the
    #: same stroke a different hill on a faster machine, which is the bug the
    #: viewport's own sculpt step multiplies this by ``dt`` to avoid.
    brush_amount: float = 4.0
    #: 0..1, how far toward the target one second of smoothing or flattening
    #: gets. Both brushes scale it by their own falloff again.
    brush_strength: float = 0.5
    #: The height Flatten pulls toward. Set by the pane, or picked up from the
    #: ground under the first click of a stroke when ``brush_level_from_pick``
    #: is on -- which is how a user levels a plateau to the height it already
    #: is somewhere.
    brush_level: float = 0.0
    brush_level_from_pick: bool = True
    #: Noise's seed. Bumped per stroke rather than per frame, so one drag lays
    #: down one field instead of re-rolling it forty times.
    brush_seed: int = 1

    #: The prefab name the next viewport click places an instance of, or "".
    #: Separate from ``place_kind`` because a prefab's "kind" is a user-chosen
    #: name and the two namespaces must not be able to collide -- a prefab
    #: called ``camera`` is a perfectly reasonable thing to author.
    place_prefab: str = ""

    # No drag state here. A live drag is the *view*'s to own -- Clay's own
    # comment on this point is the whole argument: ``ClayState`` used to carry
    # a ``drag_kind`` field written only by ``clear_drag`` and read by nothing
    # but a hint line, which therefore never showed a drag in progress. Mason
    # does not repeat that field; a gizmo drag in progress belongs beside the
    # camera and the picking state the viewport already owns.

    # -- documents ---------------------------------------------------------

    @property
    def active(self) -> MasonTab | None:
        for doc in self.docs:
            if doc.uid == self.active_uid:
                return doc
        return self.docs[-1] if self.docs else None

    @property
    def any_dirty(self) -> bool:
        return any(doc.dirty for doc in self.docs)

    def add(self, tab: MasonTab) -> MasonTab:
        self.docs.append(tab)
        self.active_uid = tab.uid
        return tab

    def get(self, uid: str) -> MasonTab | None:
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
            # The neighbour, not the first: closing a tab should leave you
            # next to where you were rather than at the far end of the bar.
            self.active_uid = self.docs[min(index, len(self.docs) - 1)].uid if self.docs else ""
        return True

    def activate(self, uid: str) -> None:
        self.active_uid = uid

    def cycle(self, step: int = 1) -> None:
        if len(self.docs) < 2:
            return
        current = self.active
        index = self.docs.index(current) if current in self.docs else 0
        self.activate(self.docs[(index + step) % len(self.docs)].uid)

    def find_path(self, path: Path) -> MasonTab | None:
        """``docmodes.find_path``: the one case-folding body every mode shares."""
        return docmodes.find_path(self.docs, path)
