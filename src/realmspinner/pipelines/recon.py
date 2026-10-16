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
    #: The reconstruction resolutions it can run. Admission refuses against this
    #: with ``field="resolution"``.
    resolutions: tuple[int, ...] = (512, 1024, 1536)


#: TRELLIS.2 as the native server runs it: one image in, one GLB out.
TRELLIS_CAPS = ReconCapabilities(max_views=1)

#: The name a manifest records for a Pixal3D run, and what ``params["recon_model"]``
#: says on the job. Pixal3D is TRELLIS.2 fine-tuned: the same engine, selected per request.
PIXAL3D_BACKEND = "pixal3d"
#: Pixal3D publishes one texture flow, at 1024: there is no 512 model to run, so a 512
#: request has no texture and the door refuses it rather than ship geometry only.
PIXAL3D_CAPS = ReconCapabilities(max_views=1, resolutions=(1024, 1536))
#: Pixal3D's own default horizontal field of view in degrees, from its reference
#: implementation. The image's real camera is unknowable here (MoGe-2, which the reference
#: uses to estimate it, is not ported), so it is a per-job parameter.
PIXAL3D_DEFAULT_FOV = 49.13


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
    #: ``"pixal3d"`` to ask for the Pixal3D family on the same engine; ``None`` leaves
    #: the server's own default (TRELLIS.2), so the field is simply not sent.
    model: str | None = None
    #: Pixal3D's camera: the image's horizontal field of view, in degrees. Sent only
    #: when set, and meaningless to TRELLIS.2.
    fov: float | None = None
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


def active_backend_name(config: Any, params: Any = None) -> str:
    """The backend a job reconstructs with: Pixal3D when its ``recon_model`` says so,
    otherwise TRELLIS.2.

    Per *job*, not per configuration: one resident server serves either family and
    the request names which. ``config`` is kept for the callers (the bench manifest)
    that have only that, and who therefore get the default.
    """
    del config
    if isinstance(params, dict) and params.get("recon_model") == PIXAL3D_BACKEND:
        return PIXAL3D_BACKEND
    return DEFAULT_BACKEND


def active_capabilities(config: Any, params: Any = None) -> ReconCapabilities:
    """The capabilities of :func:`active_backend_name`'s backend."""
    if active_backend_name(config, params) == PIXAL3D_BACKEND:
        return PIXAL3D_CAPS
    return TRELLIS_CAPS
