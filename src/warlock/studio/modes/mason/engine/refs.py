"""How a mesh node names its source, and the one door both kinds resolve through.

A :class:`MeshNode` (or a :class:`~.nodes.TerrainNode`'s material override, later)
never holds geometry -- it holds a *reference* to where the geometry comes from,
because the document links rather than embeds (see the plan's "the document
links, the export embeds"). There are exactly two ways to name a source, and
each is its own frozen type rather than one shape with an optional field, for
the reason ``clay.document.Obj`` already gives for ``generator``/``params``:
a mesh built from a primitive and a mesh borrowed from the asset library answer
different questions about themselves (what parameters made this shape, versus
what job on disk does this alias), and a single type would make each question
optional on every reference regardless of which kind it is.

**Neither reference touches the filesystem, and this package never imports
``clay``.** A :class:`PrimitiveRef` names a generator in ``clay.primitives`` by
its registry key -- an opaque string as far as this module is concerned -- and
a :class:`LibraryRef` names a job id, never a path. Resolving either into actual
triangles is the host's job, through the one callback both kinds share:
:class:`GeometrySource`. This mirrors ``plotter/tmx.py``'s
``tsx_loader``/``image_loader`` split, collapsed to one door instead of two,
because the reason a library asset must be a callback (a job id is not a path
this package is allowed to open) applies just as well to a primitive (its
builder lives in ``clay.primitives``, a sibling engine this package may not
import) -- and one resolution mechanism beats two of them living side by side
for no reason but which reference happens to be older.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .....kernels.geom3d import gltf


def _normalize(param: str, value: Any) -> Any:
    """One params value, turned into the hashable form ``ref_key`` can compare.

    **The rule: every number becomes a plain ``float``, and every sequence
    becomes a nested tuple.** Clay's own generator defaults mix the two kinds
    of number freely -- ``box``'s ``size`` default is a tuple of floats,
    ``lathe``'s ``profile`` and ``sweep``'s ``outline`` are lists of ``[x, y]``
    pairs that a caller may have built from ints, floats, or a numpy array's
    ``.tolist()`` -- and a properties panel round-trips whichever of those a
    widget happens to hand back. Two placements of the same shape with the same
    numbers must be **one** ``ref_key``, or they silently double the GPU upload
    and the glTF mesh index every time a spinbox happens to emit an int one
    frame and a float the next. Python's own numeric tower already guarantees
    ``hash(1) == hash(1.0)`` for a bare value, but that guarantee does not
    extend to every numpy scalar width without a promotion first (an ``int64``
    and a ``float32`` holding the same value are not guaranteed to hash alike),
    so this coerces explicitly rather than leaning on a guarantee that holds
    only for the two built-in number types. Every generator that takes a count
    (``segments``, ``sides``, ``rings``...) already tolerates a float there --
    ``clay.primitives._clamp_segments`` calls ``int()`` on its way in -- so
    losing the int/float distinction here costs a builder nothing.

    A bare ``bool`` is left alone rather than folded into the float branch
    (``isinstance(True, int)`` is true in Python, so the check for numbers
    comes second): no generator parameter is a flag today, but a future one
    that is should not have its ``True`` silently become a ``1.0`` that still
    prints as a number in a missing-reference message.

    Sequences recurse so a profile's list of ``[x, y]`` pairs normalizes the
    same way a flat tuple does; strings are left as strings even though they
    are technically sequences, because a generator name buried one level down
    (there is none today, but nothing rules one out) is not the array of
    numbers this exists to canonicalize.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(
                f"parameter {param!r} is not a finite number: {value!r} -- a NaN or an "
                "infinity here would make two otherwise-identical refs compare unequal, "
                "so this is refused where the params are built rather than debugged "
                "wherever the ref later fails to match"
            )
        return number
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, np.ndarray)):
        return tuple(_normalize(param, item) for item in value)
    raise ValueError(
        f"parameter {param!r} has a value {value!r} of type {type(value).__name__}, "
        "which a primitive ref cannot normalize into something hashable"
    )


def primitive_ref(generator: str, params: Mapping[str, Any]) -> PrimitiveRef:
    """Build a :class:`PrimitiveRef`, normalizing ``params`` on the way in.

    Every key must be a plain ``str`` -- a non-string key would still round-trip
    through equality and hashing today, but it is not a shape any generator's
    keyword arguments can ever actually be, so it is refused here rather than
    carried as a ref that can never resolve.
    """
    normalized: list[tuple[str, Any]] = []
    for key, value in params.items():
        if not isinstance(key, str):
            raise ValueError(
                f"primitive parameter key {key!r} is not a string -- a generator's "
                "keyword arguments are always strings, so a ref carrying anything else "
                "could never be splatted back into one"
            )
        normalized.append((key, _normalize(key, value)))
    normalized.sort(key=lambda pair: pair[0])
    return PrimitiveRef(generator=str(generator), params=tuple(normalized))


@dataclass(frozen=True)
class PrimitiveRef:
    """A generator name plus the normalized parameters it was built with.

    ``generator`` is a key into ``clay.primitives.GENERATORS``, held as an
    opaque string: this package never imports that registry (see the module
    docstring), so it cannot and does not validate the name against it -- that
    is the host resolver's job, the moment it tries to look the key up.

    ``params`` is a sorted tuple of ``(key, value)`` pairs rather than a
    ``dict`` for the one reason that matters: a ``dict`` is unhashable, and an
    unhashable field makes ``@dataclass(frozen=True)`` raise the moment
    anything tries to hash a ``PrimitiveRef`` -- which every ``ref_key`` lookup
    does. Build one through :func:`primitive_ref` rather than the constructor
    directly, so the normalization in :func:`_normalize` always runs.
    """

    generator: str
    params: tuple[tuple[str, Any], ...]

    def as_params(self) -> dict[str, Any]:
        """The params as a plain ``dict``, ready to splat into a builder.

        A nested list (``profile``, ``outline``, ``path``) comes back as a
        tuple of tuples rather than the list of lists it may have started as.
        That is safe rather than lossy: every generator that takes one of
        these indexes it positionally (``profile[i][0]``, ``outline[i][1]``),
        and a tuple supports exactly the same positional indexing a list does
        -- nothing downstream mutates a profile in place, so there is no
        capability a list had that a tuple is missing here.
        """
        return dict(self.params)


@dataclass(frozen=True)
class LibraryRef:
    """A link to one artifact of one library job -- never a path.

    ``job_id`` is opaque the same way ``generator`` is: resolving it into
    ``svc.config.job_dir(job_id) / artifact`` is service-layer business this
    package is not allowed to do (see ``tests/mason/test_mason_imports.py``'s
    ``test_the_engine_never_imports_the_service_layer``), so it goes through
    :class:`GeometrySource` exactly like a primitive does.

    ``name`` and ``sha256`` are recorded for the human, not for identity --
    see :func:`ref_key` for why they are not part of it.
    """

    job_id: str
    artifact: str = "model.glb"
    #: The job's name *at save time*, carried only so a missing reference can
    #: be reported in words ("missing: Barrel (job 3f2a...)") rather than as a
    #: bare id nobody recognizes.
    name: str = ""
    #: For relink: offering the library job whose current export hashes the
    #: same as what this scene last saw, when the id itself no longer resolves.
    sha256: str = ""


#: Either kind of reference, or the caller has not decided yet.
Ref = PrimitiveRef | LibraryRef

#: A stable, distinct answer for ``ref_key(None)`` -- a tag no real
#: :class:`PrimitiveRef` or :class:`LibraryRef` key can ever produce, since
#: both of theirs start with a different literal tag. A :class:`~.nodes.GroupNode`
#: has no ref at all, and a caller walking placed items (grouping by shared
#: geometry, say) needs to ask for its key without special-casing ``None``
#: as a separate branch every time.
_NONE_KEY: tuple[Any, ...] = ("mason.ref", "none")


def ref_key(ref: Ref | None) -> tuple[Any, ...]:
    """The caching identity of a reference: one GPU upload, one glTF mesh index.

    **Deliberately drops ``name`` and ``sha256``.** Two :class:`LibraryRef`\\ s
    at the same job and artifact are the same geometry no matter what the job
    was called when either was saved -- ``name`` exists only so a *missing*
    reference can be described in words, and keying on it would silently
    upload the same asset twice the moment a user renamed the job in the
    library between one placement and the next. ``sha256`` is carried only for
    relink's benefit and is absent entirely until a relink needs it, so keying
    on it would make an old scene's references never match a freshly-placed
    one from the same job.
    """
    if ref is None:
        return _NONE_KEY
    if isinstance(ref, PrimitiveRef):
        return ("primitive", ref.generator, ref.params)
    if isinstance(ref, LibraryRef):
        return ("library", ref.job_id, ref.artifact)
    raise TypeError(f"not a Ref: {ref!r}")  # pragma: no cover - the Ref union is closed


class GeometrySource(Protocol):
    """The one door both kinds of reference resolve through.

    This package never touches the filesystem and never imports ``clay`` (see
    the module docstring), so nothing in it can turn a :class:`PrimitiveRef`
    into a mesh by calling a generator, or a :class:`LibraryRef` into one by
    opening a GLB. Instead every consumer that needs actual triangles -- the
    viewport, the exporters, the thumbnail render -- is handed one of these by
    the host (``studio/mason_assets.py``), which is free to import both halves
    and does the resolving on its behalf. This is ``plotter/tmx.py``'s
    ``tsx_loader``/``image_loader`` precedent collapsed to one callback instead
    of two, since here both reference kinds need the same shape of answer.

    ``rev`` is a revision counter the host bumps once each time a background
    parse finishes -- a library asset arriving, a relink resolving. Nothing
    about the document changed when that happens (no node moved, no undo step
    was pushed), but the *picture* did, so the viewport's redraw key reads
    ``rev`` directly rather than the document's own change counter, which
    would never move for an edit the document never saw.
    """

    def primitives(self, ref: Ref) -> list[gltf.Primitive]:
        """The resolved geometry for ``ref``, or an empty list while it is
        still loading or if it could not be resolved at all."""
        ...

    def box(self, ref: Ref) -> tuple[np.ndarray, np.ndarray] | None:
        """``(min, max)`` of ``ref``'s geometry in its own local space, or
        ``None`` before it has resolved -- cheap enough to call every frame
        for framing and picking without walking every primitive's vertices."""
        ...

    @property
    def rev(self) -> int:
        ...
