"""What a reconstruction backend is, so a second one can be added without a rewrite.

Until now "the 3D engine" was one class, ``trellis.TrellisServer``, and every
caller reached into it by name. That is fine for one backend and wrong for the
experiments this project keeps deferring -- a multi-view reconstructor, a
different TRELLIS runtime, Pixal3D -- each of which would otherwise be a
fork of the queue. This module is the smallest seam that lets one of them be
*added* rather than *swapped in*: a protocol the queue holds its backend as, a
request type that can carry more than one image, and a capability record the
admission door can refuse against.

**No behaviour changes.** TRELLIS.2 is the only backend, is what
:func:`active_capabilities` answers, and takes exactly one image; ``reconstruct``
forwards to the ``generate`` it always had, byte for byte. A new backend lands
non-default, and must clear a bench run on frozen references (``bench``'s
``create-3d-v1``) before anything about the default moves -- ``dev/INVARIANTS.md``
says so.

Stdlib only, importable from ``queue``, ``service`` and ``bench`` alike: it is
below all three.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

#: The name a manifest records for a run, and what an older manifest -- written
#: before the key existed, when TRELLIS.2 was the only choice -- reads as.
DEFAULT_BACKEND = "trellis.2"


@dataclass(frozen=True, slots=True)
class ReconCapabilities:
    """What a backend can be asked for. Admission refuses against this."""

    #: How many reference images one reconstruction may be given. TRELLIS.2 is
    #: single-image; a multi-view backend declares more and nothing else about
    #: the queue has to know.
    max_views: int = 1


#: TRELLIS.2 as the native server runs it: one image in, one GLB out.
TRELLIS_CAPS = ReconCapabilities(max_views=1)


@dataclass(frozen=True, slots=True)
class ReconRequest:
    """One reconstruction: the images, where the GLB goes, and the knobs the
    queue has always passed. ``images`` is a tuple even for one image, which is
    the whole point of the type."""

    images: tuple[Path, ...]
    output: Path
    seed: int = 42
    resolution: int = 1024
    bg_removal: str | None = None
    #: The engine's own QEM face target, applied before its UV and PBR bake
    #: (``trellis-server``'s per-request ``faces`` field). ``None`` is "do not
    #: send it": the engine's built-in ~300k target applies, which is also the only
    #: thing an engine without ``--faces`` can do (``trellis.engine_caps``).
    faces: int | None = None


@runtime_checkable
class ReconBackend(Protocol):
    """What ``queue.Worker`` needs of whatever turns a reference into a mesh.

    Deliberately only what the queue's *scheduling* touches: whether the child is
    up, when it last worked (the idle timer and the "did this job prime it"
    comparison), how to stop it (the VRAM handoff), and the one call. Launch
    configuration (``TrellisServer.ensure_config``) is **not** here -- it is a
    fact about one backend's command line, and a backend without one is simply
    not asked.
    """

    name: str
    capabilities: ReconCapabilities
    last_used: float

    @property
    def running(self) -> bool: ...

    def stop(self) -> None: ...

    async def reconstruct(self, request: ReconRequest) -> Path: ...


def active_backend_name(config: Any) -> str:
    """The backend this configuration reconstructs with. One answer today; the
    function exists so the manifest and the admission door ask one place."""
    del config
    return DEFAULT_BACKEND


def active_capabilities(config: Any) -> ReconCapabilities:
    """The capabilities of :func:`active_backend_name`'s backend."""
    del config
    return TRELLIS_CAPS
