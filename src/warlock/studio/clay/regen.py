"""What a generator rebuild carries over, and what it cannot.

Every generator in ``primitives.py`` builds flat and grey: ``_mesh`` stamps
``material=np.zeros(len(faces), dtype="i4")`` and ``smooth=np.zeros(...)`` on
every call, because a generator is a pure function of its own parameters and
has no memory of what the object it is rebuilding used to look like. That is
correct for the generator -- it is exactly what makes ``clamp_params``'
rebuild a pure function of a shape's own numbers, with nothing about the
object's *history* smuggled into the shape -- but it means every door that
calls a generator a second time, to rebuild an existing object rather than to
place a new one, is the door responsible for not throwing the object's own
paint job and shading away on the very next keystroke.

Until this module existed only one of those doors did that job, and only for
one of the two attributes: ``panes/clay_props._carry_shading`` (now deleted;
its reasoning moved here) carried ``smooth`` back onto a rebuilt mesh but
never ``material``, and ``agent_clay._h_set_params`` carried neither -- it
called ``shading.auto_smooth(bp.GENERATORS[obj.generator][1](**merged))``
directly, which re-derives shading from scratch every time and repaints every
face to slot 0 regardless of what the object was wearing. So a box painted
with palette slot 3 came back grey after a resize typed into the properties
panel, and after the identical resize sent by an agent the shading came back
wrong too -- two doors quietly building two different meshes for the same
edit, because the carry rule lived beside one of them instead of belonging to
neither.

:func:`carry_over` is the one rule now, reused by both doors rather than
reinvented by either -- the same shape ``clay.shading.auto_smooth`` itself
was extracted for (that extraction's docstring is the 2026-09-06 audit's
organic-shapes decision, and this module leans on the identical function for
its own re-derive case below).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from . import mesh as bm
from . import shading
from .mesh import Mesh


def carry_over(old: Mesh, rebuilt: Mesh, *, material: int = 0) -> Mesh:
    """*rebuilt*, with the per-face attributes a generator rebuild must not silently lose.

    Two cases, because "the same faces" is only sometimes true of a rebuild:

    * **Face count unchanged** -- a radius or a position moved, not a segment
      count -- means the rebuilt mesh has the same faces in the same order
      every generator in ``primitives.py`` always emits them in, so *both* the
      old ``smooth`` array and the old ``material`` array are carried over
      verbatim. This is what makes a hand-picked per-face Shade Smooth *and* a
      hand-picked per-face palette assignment, made in face mode, survive a
      numeric tweak exactly rather than approximately.
    * **Face count changed** -- a segment slider moved, so the faces are not
      the same faces any more and there is no old flag or slot to carry to a
      face that did not exist a moment ago. ``smooth`` is re-derived with
      ``clay.shading.auto_smooth``, the identical rule that gave the object
      its shading the moment it was placed, and ``material`` is stamped with
      *material* -- the object's own default slot, ``Obj.material``, which
      ``document.py`` documents as "the default for *new* faces only". Every
      face of a rebuild whose count changed is, for this purpose, a new face:
      there is no principled way to say which of the old faces' colours a
      newly-appeared one should inherit, so all of them take one slot. That it
      is the object's own default rather than slot 0 is the whole point --
      slot 0 is simply whichever palette entry happens to be first, with no
      relationship to what this object was wearing, and stamping it is how a
      painted box came back grey the moment its segment count moved.
    """
    if bm.face_count(rebuilt) == bm.face_count(old):
        return replace(rebuilt, smooth=old.smooth, material=old.material)
    smoothed = shading.auto_smooth(rebuilt)
    return replace(smoothed, material=np.full(bm.face_count(rebuilt), material, dtype="i4"))
