"""Create's own corner of ``AppState``.

P5 of the restructure (``dev/RESTRUCTURE.md``) split Create into
``engine``/``ui`` packages but left its state as fifteen loose fields on
``studio/state.py``'s ``AppState`` -- this is the fold that gives it one
object instead. Only the fields nothing *outside* Create reads move here:
``form_2d``, ``form_3d`` and ``source_job`` stay on ``AppState`` because
``review_mode.capture_base`` reads the live 2D/3D forms to seed a sweep
without ever leaving Review, and ``panes/library.py``'s ``select``/
``copy_settings`` write ``source_job``/``form_2d`` as a side effect of
selecting or copying *any* card, library-wide, not only on the way into
Create -- moving either would put a non-Create reader through a mode's
private state, which is the one thing this fold exists to prevent.

No ``dataclasses``/``typing`` import is declared in
``tests/test_create_engine_imports.py``'s ``OUTWARD_IMPORTS`` because both
are stdlib and already filtered by that test's own ``_UNINTERESTING`` set --
this module reaches for nothing else, so it needs no entry there at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CreateState:
    """Create's mutable, per-session state. Built once per ``AppState``."""

    # Where the user is standing, one of ``create_stages.STAGES`` (the UI
    # redesign, wave 5). **Volatile and never persisted**, which is the same
    # rule ``AppState.mode`` follows and for a stronger reason: a stage is a
    # *derived* position over the selected asset, so a remembered one would be
    # restored against whatever happens to be selected next launch and would
    # routinely name a stage that asset has not reached. Written only by
    # ``create_stages.go`` -- the one switch.
    stage: str = "reference"
    # Whether this session has already checked the 2D reference path restored
    # from settings (the 2026-09-07 Create review, item 5.3a): ``ref_path``
    # persists like ``ip_adapter``/``control`` now (``settings.VOLATILE``), so
    # a restart reopens with it still selected -- unless the file moved or was
    # deleted while Warlock was shut, in which case a live value would fail
    # silently at submit instead of failing where it broke. One shot rather
    # than every frame: the file will not appear or vanish while the pane sits
    # open, so re-``stat``-ing it on every keystroke elsewhere in the form
    # would be pure cost with nothing new to find. Never persisted itself --
    # it describes this process's own check, not a preference.
    reference_path_checked: bool = False
    # ``(frame, id(form)) -> problems`` for the Reference stage's plan footer
    # (``recipe.problems_for``). Per-ctx rather than a module global: a
    # module-level cache keyed on ``id(form)`` alone would let a second ctx's
    # form -- reusing a GC'd id -- read the first ctx's stale verdict.
    problems_cache: tuple[tuple[int, int], list[Any]] | None = None
    #: Why the last Generate press was refused, when the refusal names no
    #: control. The VRAM door is the one such refusal in ``service.validation``
    #: and deliberately so -- ``vram.remedies`` offers several answers and
    #: sometimes an environment variable, which is not a widget -- so it had
    #: nowhere to go but a fading toast, while the plan block a few pixels away
    #: went on saying "Ready to generate." A multi-remedy paragraph is not
    #: something a toast can hold. Cleared by the next accepted submit.
    submit_refusal: str = ""
    # Draw a tile repeated in the 2D viewport rather than once. Off by default,
    # and that is a decision rather than an oversight: every other view of an
    # asset in this app -- the thumbnail, the exports, the Inker -- shows one
    # cell, so a viewport that silently showed four would make the texture look
    # a quarter of its size. What repetition answers that nothing else does is
    # whether the pattern *reads* as repeating, which is a question the user
    # asks deliberately; the seam question already has the wrapped view.
    tile_preview: bool = False
