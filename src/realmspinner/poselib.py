"""Pose constants and the user's clip folder: what the pose editors share.

This was the global pose library -- reusable poses authored in Poser, stored
under ``data_dir/poser/`` -- until Poser left (the 2026-10-13 Clay rig plan,
Tranche 5). The library half went with it: nothing authors, lists or applies a
shared pose any more, and the records already on disk are left alone.

What stays is the half a single mesh's pose and the clip library still need: the
canonical armature box, the root-offset range every pose record is checked
against, and where an *edited* clip library lives (``cliplib.user_clip_dir``'s
other half).

Pure in the ``rigging.py`` sense: no bpy, no ``service``, no ``studio``.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

# The canonical armature's bounding box: the unit box fit_template normalizes
# against, so place() is the identity on the template's own landmarks and the
# armature is exactly one character-height tall, and it is what makes a stored
# root_translation mean character-height units literally.
UNIT_LO = (-0.5, -0.5, 0.0)
UNIT_HI = (0.5, 0.5, 1.0)

# A root offset past two character heights is a fat-fingered gizmo drag, not a
# pose. Blender axes, character-height units, per component.
MAX_ROOT_TRANSLATION = 2.0


class RecordError(ValueError):
    """A refused pose record, naming the field it came from.

    ``guidance.GuidanceError``'s shape for the same reason: a ``ValueError``
    subclass so nothing that already catches ``ValueError`` changes, with
    ``field`` carried so ``service.errors.invalid_from`` can pass the address
    through instead of the message arriving pointed at nothing.
    """

    def __init__(self, message: str, *, field: str | None = None) -> None:
        super().__init__(message)
        self.field = field


def validate_root_translation(raw: Any) -> list[float]:
    """Normalize and range-check a root offset. Raises :class:`RecordError`.

    Finite-checked *before* range-checked, because ``abs(nan) <= 2.0`` is
    False and the range message would then blame a value that is not a
    number. Shared rather than restated: the 2026-09-07 audit (poser-05)
    found this same field handled by two other doors with a bare
    ``float(v)`` and neither check -- ``cliplib.parse_clip_library`` and the
    clip editor's own shape check -- so ``[nan, 1e30, 0.0]`` round-tripped
    through a clip's key poses when the identical field on a library pose
    already refused it here.
    """
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise RecordError("root_translation must be a 3-vector", field="root_translation")
    try:
        root = [float(v) for v in raw]
    except (TypeError, ValueError):
        raise RecordError("root_translation is not numeric", field="root_translation") from None
    if not all(math.isfinite(v) for v in root):
        raise RecordError("root_translation is not numeric", field="root_translation")
    if any(abs(v) > MAX_ROOT_TRANSLATION for v in root):
        raise RecordError(
            f"root_translation components must be within "
            f"+/-{MAX_ROOT_TRANSLATION} character heights",
            field="root_translation",
        )
    return root


# --- paths ------------------------------------------------------------------
#
# data_dir/poser/ sits beside the job directories. A collision with a job id is
# impossible -- ids are 12 lowercase hex -- but service.files.storage_sizes
# counts it as one more "job dir", exactly as it already does autosave/; the
# miscount is a byte total under a different heading, not a correctness issue.


def poser_dir(config: Any) -> Path:
    return Path(config.data_dir) / "poser"


def clip_dir(config: Any) -> Path:
    """Where an edited clip library lives. See ``cliplib.user_clip_dir``.

    Spelled here as well as there because the two callers are different: this
    is the app asking where to write, that is the loader asking where to read,
    and the loader may not import ``config``. They must agree, and
    ``tests/service/test_clip_editing.py`` is what asserts they do.
    """
    return poser_dir(config) / "clips"


def clip_path(config: Any, template_key: str) -> Path:
    """The editable clip library for one template.

    Named by template key, which ``cliplib._load_clip_library`` reads back as
    the file *stem* -- so the two halves of that round trip are one expression
    here rather than a convention two modules each half-remember.
    """
    if not template_key or not re.fullmatch(r"[A-Za-z0-9_-]+", template_key):
        raise RecordError(f"{template_key!r} is not a template key", field="template")
    return clip_dir(config) / f"{template_key}.json"


def mirror_root_translation(v: Any) -> list[float]:
    """Reflect a root offset across the subject's YZ plane: (x, y, z) -> (-x, y, z).

    The positional half of ``poses.mirror_quaternion``'s reflection --
    Blender axes, X the mirror normal -- so mirroring a pose that steps left
    steps right.
    """
    x, y, z = (float(c) for c in v)
    return [-x, y, z]
