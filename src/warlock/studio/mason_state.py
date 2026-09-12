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

from . import docmodes

WSCN_SUFFIX = ".wscn"

# Name, label, and the key that selects it. Clay's four tools, verbatim: the
# transform gizmos are one idea, not two, and a scene editor needs the same
# three plus Select.
TOOLS = (
    ("select", "Select", "Q"),
    ("move", "Move", "W"),
    ("rotate", "Rotate", "E"),
    ("scale", "Scale", "R"),
)

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
