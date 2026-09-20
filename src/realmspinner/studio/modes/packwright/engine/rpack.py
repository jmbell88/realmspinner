"""``.rpack`` -- the Packwright document on disk, as a zip.

``pack.json`` plus one ``sources/<n>.png``, epoch-stamped throughout, so two
saves of an unchanged document are byte-identical -- the ``.rblk``/``.rmap``
rule, applied a third time and for the third time because a file that changes
every time it is written is undiffable and its content hash useless.

**The atlas is not in here, and neither is the layout.** Both are derived from
what is: storing an atlas would mean the file could disagree with the settings
beside it, and the packer is deterministic so there is nothing to gain by it.
Exporting is a separate act with its own destination.

**Source pixels are embedded rather than referenced.** A source is routinely a
*frame of an Inker document* that has since been edited, or a layer of one, and
neither has a path at all -- so a reference would be unresolvable for the
majority of them. What the document records is what was packed; re-adding the
source is how you pick up a change. This is the ``.rmap`` tileset argument with
a sharper case.
"""

from __future__ import annotations

import io
import json
import zipfile
from typing import Any

import numpy as np

from .....core.safeio import zipguard
from ...plotter.engine.pngio import png_bytes
from .document import PackDoc, Source, new_uid
from .layout import PackSettings
from .sources import EMPTY_META, SliceSpec, Sprite, SpriteMeta

VERSION = 1
MANIFEST = "pack.json"
SOURCE_DIR = "sources"

_EPOCH = (1980, 1, 1, 0, 0, 0)

# A zip's directory declares what each member unpacks to and nothing makes that
# number honest -- a few kilobytes of archive can claim terabytes, and the read
# that discovers this is the one that has already exhausted memory. One gigabyte
# is far past any atlas document this editor produces (a few hundred sprites of
# a megapixel each is tens of megabytes) and far short of a machine's RAM, so
# the ceiling is only ever hit by a file that was not written by us. The
# ``clay/serialize.py`` constant verbatim, read from module globals at call time
# for its reason too: a test lowers it rather than building a gigabyte.
MAX_DECOMPRESSED_BYTES = 1 << 30

# And a ceiling per source image, because the byte ceiling above is about the
# archive rather than about any one member: a single 60000-square PNG deflates
# to very little and decodes to fourteen gigabytes of RGBA. The number mirrors
# ``service.validation.MAX_IMAGE_PIXELS`` rather than importing it -- this
# package may not reach for the service layer, which its import pin enforces --
# so the two are one answer written twice on purpose.
MAX_SOURCE_PIXELS = 16_000_000

# The 2026-09-13 audit's packwright-01, the aggregate half of the same finding
# that already gives :data:`MAX_SOURCE_PIXELS` its per-image ceiling.
# ``MAX_SOURCE_PIXELS`` bounds one image, ``MAX_DECOMPRESSED_BYTES`` bounds the
# archive's *stored* (still-PNG) bytes -- neither bounds what ``read_rpack``
# decodes in total across every source it reads, so a ``.rpack`` of a few
# hundred near-ceiling sources, each individually legal, decodes to hundreds of
# gigabytes of RGBA before the loop that reads them ever finishes. No atlas
# this app can pack ever holds more content than the largest atlas it can
# produce -- ``kernels.sheet.MAX_ATLAS_PX`` (8192) squared -- which is the
# same figure ``studio.pixelguard.MAX_DECODE_PIXELS`` is already set to and
# for the identical reason stated there. The number is written here rather
# than imported, the same call :data:`MAX_SOURCE_PIXELS` above already makes
# about ``service.validation.MAX_IMAGE_PIXELS``: this package's own import
# pin (``tests/modes/packwright/test_packwright_imports.py``) enumerates its eight
# outward reaches exactly, and a ninth for one shared constant is a worse
# trade than one number written twice on purpose. Module-level so a test
# lowers it rather than decoding anywhere near a document this size.
MAX_DOCUMENT_PIXELS = 8192 * 8192


def _rect(rect: Any) -> dict[str, int]:
    return {"x": int(rect[0]), "y": int(rect[1]), "w": int(rect[2]), "h": int(rect[3])}


def _meta_json(meta: SpriteMeta) -> dict[str, Any]:
    """A sprite's pivot and slices, written **only when it has them**.

    The version stays 1 for the reason ``animation.json``'s does: every read of
    this section is ``.get``-based, so a build that has never heard of these
    keys opens the file and gets the document it always got -- and a document
    whose sprites carry none produces the same bytes it did before they existed,
    which is what the byte-identity test pins.
    """
    out: dict[str, Any] = {}
    if meta.pivot is not None:
        out["pivot"] = {"x": float(meta.pivot[0]), "y": float(meta.pivot[1])}
    if meta.slices:
        out["slices"] = [
            {
                "name": one.name,
                "bounds": _rect((one.x, one.y, one.w, one.h)),
                **(
                    {"pivot": {"x": float(one.pivot[0]), "y": float(one.pivot[1])}}
                    if one.pivot is not None
                    else {}
                ),
                **({"center": _rect(one.center)} if one.center is not None else {}),
            }
            for one in meta.slices
        ]
    return out


def manifest_json(doc: PackDoc) -> str:
    settings = doc.settings
    payload = {
        "version": VERSION,
        "settings": {
            "mode": settings.mode,
            "padding": settings.padding,
            "extrude": settings.extrude,
            "trim": settings.trim,
            "max_size": settings.max_size,
            "power_of_two": settings.power_of_two,
            # Written only away from their defaults, the ``_meta_json`` rule:
            # a document that never touched either keeps producing the exact
            # bytes it always did, VERSION unchanged, and an older build's
            # ``.get``-based read of a *newer* file falls back to the default
            # it already knows.
            **({"columns": settings.columns} if settings.columns is not None else {}),
            **(
                {"json_schema": settings.json_schema}
                if settings.json_schema != PackSettings().json_schema
                else {}
            ),
        },
        "sources": [
            {
                "key": source.key,
                "name": source.sprite.name,
                "name_override": source.name_override,
                "image": f"{SOURCE_DIR}/{index}.png",
                **_meta_json(source.sprite.meta),
            }
            for index, source in enumerate(doc.sources)
        ],
    }
    return json.dumps(payload, sort_keys=True, indent=2)


class Snapshot:
    """What a ``.rpack`` is written from, taken while the document is still.

    The manifest is a string and the sprite arrays are read-only (``Sprite``
    freezes them), so holding references is enough: the document can go on
    being edited on the frame thread -- sources added, renamed, removed --
    while a task encodes this. The PNG encode is the expensive half by two
    orders of magnitude, and it used to run on the frame thread on every
    save.
    """

    __slots__ = ("manifest", "sprites")

    def __init__(self, manifest: str, sprites: tuple[np.ndarray, ...]) -> None:
        self.manifest = manifest
        self.sprites = sprites


def snapshot(doc: PackDoc) -> Snapshot:
    """The frame-thread half of a save: cheap, and reads the document once."""
    return Snapshot(manifest_json(doc), tuple(source.sprite.pixels for source in doc.sources))


def snapshot_bytes(snap: Snapshot) -> bytes:
    """The task-thread half: encode a :func:`snapshot`."""
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(zipfile.ZipInfo(MANIFEST, _EPOCH), snap.manifest)
        for index, pixels in enumerate(snap.sprites):
            # Stored, not deflated: a PNG is already compressed.
            zf.writestr(
                zipfile.ZipInfo(f"{SOURCE_DIR}/{index}.png", _EPOCH),
                png_bytes(pixels),
                zipfile.ZIP_STORED,
            )
    return out.getvalue()


def rpack_bytes(doc: PackDoc) -> bytes:
    """Both halves at once, for the callers that are already off-thread."""
    return snapshot_bytes(snapshot(doc))


def read_rpack(data: bytes) -> PackDoc:
    """A ``.rpack``'s bytes back into a :class:`~.document.PackDoc`.

    Restored by construction, so the document reads clean: a file that has just
    been opened is not unsaved.
    """
    from . import layout as laylib

    try:
        zf = zipguard.BoundedZip(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ValueError("this is not a Realmspinner atlas document") from exc

    # Everything that can raise is inside the ``with``, the version check and
    # the manifest parse included. They used to sit in the gap between the open
    # and the ``with``, where a refusal -- the one thing they exist to do --
    # left the archive open with nothing to close it but the collector. The
    # ``clay/serialize.read_rblk`` and ``rmap.read_rmap`` shape, third instance.
    with zf:
        # Before anything is read: the directory says what every member unpacks
        # to, and the cheapest place to refuse an archive that claims more than
        # this build will hold is before the first ``read``.
        claimed = sum(int(info.file_size) for info in zf.infolist())
        ceiling = MAX_DECOMPRESSED_BYTES
        if claimed > ceiling:
            raise ValueError(
                f"this atlas document claims {claimed} bytes unpacked, "
                f"past the {ceiling} this build will read"
            )

        try:
            manifest = json.loads(zf.read(MANIFEST))
        except (
            zipfile.BadZipFile,
            KeyError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            raise ValueError("this is not a Realmspinner atlas document") from exc
        if not isinstance(manifest, dict):
            raise ValueError("this atlas document's manifest is malformed")

        try:
            version = int(manifest.get("version", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError("this atlas document's manifest is malformed") from exc
        if version > VERSION:
            raise ValueError(
                f"this atlas document was written by a newer version of Realmspinner "
                f"(format {version}, this build reads {VERSION})"
            )

        entries = manifest.get("sources", [])
        if not isinstance(entries, list):
            raise ValueError("this atlas document's manifest is malformed")
        # The same ceiling the packer refuses at, asked before any of them is
        # decoded: a manifest listing a million sources is a million PNG decodes
        # ahead of the refusal that was always going to come.
        if len(entries) > laylib.MAX_SPRITES:
            raise ValueError(
                f"this atlas document lists {len(entries)} sources; "
                f"{laylib.MAX_SPRITES} is the most this build will open"
            )

        settings = _settings_from(manifest.get("settings"))
        sources: list[Source] = []
        seen: set[str] = set()
        # The 2026-09-13 audit's packwright-01: a running total of what this
        # loop has decoded so far, checked against :data:`MAX_DOCUMENT_PIXELS`
        # *before* the next source is decoded rather than after -- the same
        # "asked before the allocating call" rule ``_pixels_from``'s own
        # per-image check already follows, applied across the whole document
        # instead of to one member of it. A one-element list rather than a
        # nonlocal ``int``: ``_pixels_from`` mutates it in place so the check
        # and the decode it guards stay in one function, the way the per-image
        # ceiling already does.
        budget = [MAX_DOCUMENT_PIXELS]
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("this atlas document holds a malformed source")
            key = str(entry.get("key", ""))
            if not key:
                raise ValueError("this atlas document holds a source with no key")
            if key in seen:
                # Refused rather than deduplicated: the two would pack into one
                # slot and the loser would be missing with nothing to say so.
                raise ValueError(f"this atlas document holds {key!r} twice")
            seen.add(key)
            name = str(entry.get("image", ""))
            try:
                raw = zf.read(name)
            except KeyError as exc:
                raise ValueError(
                    f"this atlas document names a source image the file does not "
                    f"carry ({name})"
                ) from exc
            sources.append(
                Source(
                    uid=new_uid(),
                    sprite=Sprite(
                        key=key,
                        name=str(entry.get("name", key)),
                        pixels=_pixels_from(raw, name, budget),
                        meta=_meta_from(entry, key),
                    ),
                    name_override=str(entry.get("name_override", "")),
                )
            )

    doc = PackDoc(sources=sources, settings=settings)
    doc.mark_saved()
    return doc


def _read_point(raw: Any) -> tuple[float, float] | None:
    return None if raw is None else (float(raw["x"]), float(raw["y"]))


def _read_rect(raw: Any) -> tuple[int, int, int, int] | None:
    return (
        None
        if raw is None
        else (int(raw["x"]), int(raw["y"]), int(raw["w"]), int(raw["h"]))
    )


def _meta_from(entry: dict, key: str) -> SpriteMeta:
    """One source's pivot and slices, or a refusal naming the source.

    ``.get``-based, so a manifest written before these keys existed reads clean
    -- but **refused** when a key is present and malformed, which is the rule
    ``.rpack`` is written under throughout: this format is ours and versioned,
    and a file that is wrong about itself is a file to say so about rather than
    one to silently open with a pivot missing. That is the opposite of
    ``sources.sprite_meta``'s tolerance, which reads *somebody else's* data.

    The refusal names the source, because a manifest with forty sprites in it
    and a message that says only "malformed" is a message that cannot be acted
    on.
    """
    if "pivot" not in entry and "slices" not in entry:
        return EMPTY_META
    try:
        slices = []
        for one in entry.get("slices") or ():
            bounds = _read_rect(one["bounds"])
            if bounds is None:
                raise ValueError("a slice with no bounds")
            slices.append(
                SliceSpec(
                    str(one.get("name", "")),
                    *bounds,
                    pivot=_read_point(one.get("pivot")),
                    center=_read_rect(one.get("center")),
                )
            )
        return SpriteMeta(pivot=_read_point(entry.get("pivot")), slices=tuple(slices))
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"this atlas document's metadata for {key!r} is malformed"
        ) from exc


def _pixels_from(raw: bytes, name: str, budget: list[int]) -> np.ndarray:
    """One embedded member as RGBA, refused at the door if it is not one.

    ``UnidentifiedImageError`` is an ``OSError``, so a member that is not an
    image at all and one that is a truncated PNG come out of the same clause.
    The pixel ceiling is asked *before* ``convert``, because that is the call
    that allocates: a header saying 60000 squared is four bytes on disk and
    fourteen gigabytes decoded.

    ``budget`` is ``read_rpack``'s running document-wide total, in a
    one-element list so this call can deduct from it -- the 2026-09-13 audit's
    packwright-01. ``MAX_SOURCE_PIXELS`` alone bounds one member; a document of
    many members individually under that ceiling was never checked in
    aggregate, so a few hundred near-ceiling sources decoded to hundreds of
    gigabytes before this function ever refused one. Checked here, against the
    header's declared size, in the same breath as the per-image ceiling and
    before the same ``convert`` call that would pay for either.
    """
    from PIL import Image

    try:
        image = Image.open(io.BytesIO(raw))
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError(
            f"this atlas document's {name} is not an image this build can read"
        ) from exc
    with image:
        declared = image.width * image.height
        if declared > MAX_SOURCE_PIXELS:
            raise ValueError(
                f"this atlas document's {name} is {image.width}x{image.height}; "
                f"the limit is {MAX_SOURCE_PIXELS} pixels"
            )
        if declared > budget[0]:
            raise ValueError(
                f"this atlas document's sources decode past "
                f"{MAX_DOCUMENT_PIXELS} pixels in total once {name} is added; "
                "that is the most this build will unpack from one document"
            )
        budget[0] -= declared
        return np.asarray(image.convert("RGBA"), dtype=np.uint8)


def _json_bool(value: Any, default: bool) -> bool:
    """A JSON boolean, refusing to read the *string* ``"false"`` as True.

    ``bool("false")`` is True, and this file is hand-editable -- so a manifest
    somebody had typed a quoted boolean into came back with trim on when it
    said off, and the atlas was silently packed the other way. Only the two
    spellings JSON itself has are honoured plus the two obvious strings;
    anything else keeps the default rather than guessing, which is the same
    doctrine every other reader in this package follows.
    """

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes"):
            return True
        if lowered in ("false", "0", "no", ""):
            return False
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _settings_from(entry: Any) -> PackSettings:
    values = entry if isinstance(entry, dict) else {}
    default = PackSettings()
    try:
        kwargs = {
            "mode": str(values.get("mode", default.mode)),
            "padding": int(values.get("padding", default.padding)),
            "extrude": int(values.get("extrude", default.extrude)),
            "trim": _json_bool(values.get("trim"), default.trim),
            "max_size": int(values.get("max_size", default.max_size)),
            # The ``None`` sentinel survives: a manifest that leaves this
            # unsaid gets the *file's* mode's default via ``__post_init__``,
            # not the default constructed above -- which resolved for the
            # default mode, so a maxrects file with the key missing was handed
            # the grid answer and the documented per-mode default never fired.
            "power_of_two": (
                None
                if values.get("power_of_two") is None
                else _json_bool(values.get("power_of_two"), False)
            ),
            "columns": (
                None if values.get("columns") is None else int(values.get("columns"))
            ),
            "json_schema": str(values.get("json_schema", default.json_schema)),
        }
    except (TypeError, ValueError) as exc:
        raise ValueError("this atlas document's settings are malformed") from exc
    # Constructed *outside* the coercion guard on purpose. ``PackSettings``
    # validates on construction -- including the padding-against-extrude rule --
    # so a hand-edited manifest is refused with the reason rather than producing
    # an atlas that bleeds at some zoom levels, and catching that here would
    # replace "padding must be at least twice extrude" with a generic sentence.
    return PackSettings(**kwargs)
