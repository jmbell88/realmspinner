"""The shell: the window, the frame loop, and everything wired together.

The P4 split of the former ``src/realmspinner/studio/main.py`` (5,971 lines) into
one module per concern, following the mixin idiom ``studio/modes/clay/ui/viewport.py`` and
``review_panes.py`` already established for a body of drawing that belongs to
the shell -- ``self`` in every method here is the assembled
:class:`~.app.App`, and every method's body is unchanged from the line it was
lifted off of. **No symbol is re-exported here**: a caller imports the
submodule it needs and addresses the symbol through it
(``from realmspinner.studio.shell import frame`` / ``frame._takes_pointer``), the
same rule ``realmspinner.kernels.rig`` already follows.

* :mod:`.app` -- the :class:`App` class itself: construction, the three setup
  phases (window, runtime, context), the splash-guarded startup and the
  pygame loop that drives ``frame()``. Assembles every mixin below alongside
  the four pane mixins (``ClayViewport``, ``MasonViewport``, ``PoserViewport``,
  ``ReviewPanes``) that already lived beside ``main.py``. ``StartupRefused``,
  the named startup failure ``run()`` shows in a native dialog, lives here too.
* :mod:`.frame` -- the frame loop proper: the idle-skip gate, the per-frame
  cache refresh, ``_build_ui``'s dispatch over the rail/menu/workspace
  skeleton, and the viewport pane Create draws its canvas through. The
  module-level layout helpers several workspaces (and the four pane mixins)
  share -- ``_split_column``, ``_right_column``, ``_column_boundary``,
  ``_stage_pane``, ``_takes_pointer``, ``_ui_scale`` -- live here too, since
  this is where most of their callers are.
* :mod:`.tasks` -- the task pump: landing a finished ``ctx.tasks`` result onto
  the document it belongs to, the viewer's parse/adopt split for a selection
  change, and the storage/health/library re-probes a landed task can trigger.
* :mod:`.events` -- input: the pygame event pump, the one router that decides
  whether a press reaches a workspace's own viewport or the global shortcut
  table, the mode switch and its persistence, and a file dropped on the
  window.
* :mod:`.quit` -- the guard chain a quit runs through (one unsaved-document
  question per mode, never all of them at once), teardown, and the window
  caption's dirty marker.
* :mod:`.paintview` -- not part of the P4 main.py split above but the same
  idiom: ``PaintView`` and the zoomable, pannable 2D viewport it drives,
  promoted out of Inker in P5 once Plotter and Packwright turned out to have
  been importing it, not reimplementing it, from the day each was written.

``studio/main.py`` stays the process entry -- ``run()``, the single-instance
lock, session-marker bookkeeping -- and imports :class:`App` from here to
build one; every constant and cross-cutting helper (``_step``, ``_background``,
the task-key constants, the startup-geometry functions) that more than one
shell module needs stays defined there, the same way ``TARGET_FPS`` already
crossed a module boundary into ``studio/modes/clay/ui/viewport.py`` before this split existed.
"""

from __future__ import annotations
