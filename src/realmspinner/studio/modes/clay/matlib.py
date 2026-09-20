"""A cross-document library of named materials, under ``REALMSPINNER_HOME``.

Tranche 6 ("UV and materials", ``dev/CLAY-PLAN.md``). A document's own
palette (``ClayDoc.materials``) is scoped to the one ``.rblk`` it lives in;
this is the shelf beside it -- name a look once ("Rusty Metal"), reuse it on
the next barrel in a different document. The Inker's swatch row
(``studio/modes/inker/ui/panes/colors.py``) is the nearest precedent for a
per-user, cross-document store, but it answers a smaller question: a swatch
is four floats that already fit inside the app's one shared, hand-editable
``settings.json`` (``studio/settings.py``). A material can carry up to five
baked textures, each potentially megabytes, and folding that into the one
settings file would make it neither small nor hand-editable -- so this is a
directory of its own instead, the same reasoning ``service/packs.py`` and
``service/updates.py`` give the *other* per-feature folders already living
under ``config.home``.

**One JSON file plus PNG side cars per material, not one JSON for the whole
shelf.** Three reasons, all a single shared file would cost:

* a hand-edited or half-written entry must not take the rest of the shelf
  down with it (see :func:`list_materials`'s own docstring) -- one corrupt
  file among many is a skipped row, not a library that fails to open;
* saving or deleting one material never needs a read-modify-write of
  everything else, so two Clay windows (or a save racing a delete) cannot
  clobber each other's entries;
* deleting is exactly the files that one entry owns, unlinked, rather than a
  splice out of a shared array.

This mirrors ``kernels/mesh/serialize.py``'s own ``.rblk`` shape (a JSON
manifest naming ``textures/<n>.png`` members) one level up: a directory
instead of a zip, because a zip's benefit -- one file to move around -- is
not wanted here; the individual PNGs are meant to be deleted and replaced
independently of each other.

No imgui here -- ``ui/panes/props.py`` is the only caller, and it calls this
the way every other pure-data module in this package is called: never
mutating a stored ``gltf.Material`` in place, always handed a whole one to
write or getting a whole one back.

**Every read tolerates a half-written or hand-edited file.** A corrupt entry
is skipped (:func:`list_materials`) or degrades the one texture slot it names
(:func:`load_material`), logged, never raised into a frame that is only
trying to draw a list of buttons -- the "a corrupt entry must not break the
pane" rule the brief states in so many words.

Synchronous, deliberately, unlike ``props.py``'s "assign a texture from an
arbitrary file" door: every byte a save here needs is already decoded in
memory (an object's current, in-app ``gltf.Material``) and every byte a load
here reads back was written by this same module under the same
``pixelguard`` ceiling, so there is no unknown-sized file on disk the way an
arbitrary user pick is. A save or an apply is an occasional button press, not
a per-frame cost.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ....core.safeio import atomic, pixelguard
from ....kernels.geom3d import gltf

log = logging.getLogger(__name__)

#: The five texture slots ``gltf.Material`` carries, in the order they are
#: checked and written -- the same tuple ``kernels/mesh/serialize.py``'s
#: ``TEXTURE_FIELDS`` and ``ui/panes/props.py``'s ``TEXTURE_SLOTS`` name, so
#: all three read as one convention rather than three that happen to agree
#: today.
TEXTURE_SLOTS = ("base_color", "metallic_roughness", "normal", "emissive", "occlusion")

_LIBRARY_SUBDIR = ("clay", "materials")


@dataclass(frozen=True)
class MaterialEntry:
    """One shelf row -- enough to list and preview without decoding a texture.

    ``id`` is the file stem, minted once at :func:`save_material` and never
    reused (see :func:`delete_material`): it is the address every "Apply" and
    "Delete" button holds, and a name is not an address -- two materials may
    share a name exactly as two palette slots in one document may
    (``ClayDoc.add_material`` places no uniqueness rule on ``Material.name``
    either).
    """

    id: str
    name: str
    base_color_factor: tuple[float, float, float, float]
    #: Which of :data:`TEXTURE_SLOTS` this entry carries -- the list row's own
    #: chip, answerable with no PNG ever opened.
    texture_slots: tuple[str, ...] = ()


def library_dir(home: Path | str) -> Path:
    """Where the shelf lives under *home* (``REALMSPINNER_HOME``, or a sandboxed
    stand-in a test points here instead)."""
    path = Path(home)
    for part in _LIBRARY_SUBDIR:
        path = path / part
    return path


def _slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(name).strip().lower()).strip("-")
    return slug or "material"


def _entry_path(home: Path | str, entry_id: str) -> Path:
    return library_dir(home) / f"{entry_id}.json"


def _texture_path(home: Path | str, entry_id: str, slot: str) -> Path:
    return library_dir(home) / f"{entry_id}_{slot}.png"


def _write_png(path: Path, image: tuple[int, int, bytes]) -> None:
    from PIL import Image

    width, height, data = image
    picture = Image.frombytes("RGBA", (int(width), int(height)), bytes(data))
    atomic.save_image(path, picture, "PNG")


def save_material(home: Path | str, name: str, material: gltf.Material) -> MaterialEntry:
    """Save *material* as a **new** shelf entry named *name*. -> the new row.

    Always a fresh id, never an overwrite of an existing entry: "Save current
    as..." is an *add*, the same verb the properties panel's own palette
    "Add" button already uses for a document's local materials. Replacing a
    saved look is delete-then-save, which is what the pane's own Delete
    button is for.
    """
    home = Path(home)
    folder = library_dir(home)
    folder.mkdir(parents=True, exist_ok=True)
    entry_id = f"{_slug(name)}-{uuid.uuid4().hex[:8]}"

    # Which slots exist, and their deterministic filenames, decided *before*
    # any PNG is written -- a slot's filename never depends on its bytes.
    images: dict[str, tuple[int, int, bytes]] = {}
    for slot in TEXTURE_SLOTS:
        image = getattr(material, slot, None)
        if image is not None:
            images[slot] = image
    textures = {slot: f"{entry_id}_{slot}.png" for slot in images}

    payload: dict[str, Any] = {
        "name": str(name),
        "base_color_factor": [float(c) for c in material.base_color_factor],
        "metallic_factor": float(material.metallic_factor),
        "roughness_factor": float(material.roughness_factor),
        "emissive_factor": [float(c) for c in material.emissive_factor],
        "double_sided": bool(material.double_sided),
        "alpha_mode": str(material.alpha_mode),
        "alpha_cutoff": float(material.alpha_cutoff),
        "textures": textures,
    }
    # The 2026-09-19 audit, finding clay-32: the manifest used to be written
    # *after* every texture PNG, so a crash partway through the texture loop
    # left PNGs on disk that no manifest named -- orphans ``list_materials``
    # (it only ever walks ``*.json``) and ``delete_material`` (it only knows
    # an id's side cars by its own naming convention) can never find. Writing
    # the manifest first instead means an interrupted save leaves, at worst,
    # a listed entry with a texture slot ``load_material`` already tolerates
    # missing -- never an unlisted, unsweepable file.
    atomic.write_text(_entry_path(home, entry_id), json.dumps(payload, indent=2))
    for slot, image in images.items():
        _write_png(folder / textures[slot], image)
    return MaterialEntry(
        id=entry_id,
        name=str(name),
        base_color_factor=tuple(payload["base_color_factor"]),  # type: ignore[arg-type]
        texture_slots=tuple(textures),
    )


def _read_entry(path: Path) -> MaterialEntry | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        color = tuple(float(c) for c in payload.get("base_color_factor", (0.8, 0.8, 0.8, 1.0)))
        if len(color) != 4:
            raise ValueError(f"base_color_factor has {len(color)} components, not 4")
        textures = payload.get("textures", {})
        if not isinstance(textures, dict):
            raise ValueError("'textures' is not a mapping")
    except Exception:
        # A corrupt or half-written entry (a crash mid-save, a hand edit) is
        # skipped rather than raised: one bad file must not take the rest of
        # the shelf down with it, which is this module's own reason for
        # being one file per material rather than one file for all of them.
        log.warning("clay material library: skipped an unreadable entry (%s)", path, exc_info=True)
        return None
    return MaterialEntry(
        id=path.stem,
        name=str(payload.get("name") or path.stem),
        base_color_factor=color,  # type: ignore[arg-type]
        texture_slots=tuple(slot for slot in TEXTURE_SLOTS if slot in textures),
    )


def list_materials(home: Path | str) -> list[MaterialEntry]:
    """Every entry the shelf holds, by name. Never raises.

    No PNG is opened here -- only the small JSON manifest each entry writes,
    so listing costs nothing proportional to how much texture the shelf
    holds. An absent library folder (nothing saved yet) is an empty list, not
    a refusal.
    """
    folder = library_dir(home)
    if not folder.is_dir():
        return []
    out: list[MaterialEntry] = []
    for path in sorted(folder.glob("*.json")):
        entry = _read_entry(path)
        if entry is not None:
            out.append(entry)
    out.sort(key=lambda e: e.name.lower())
    return out


def load_material(home: Path | str, entry_id: str) -> gltf.Material | None:
    """The full material an entry describes, textures decoded. -> ``None``
    for an id the shelf does not have, or whose own JSON cannot be read.

    A texture side car that is missing or fails :mod:`pixelguard`'s ceiling
    degrades that **one slot** to ``None`` rather than refusing the whole
    material -- the scalar fields (base colour, roughness...) are still
    exactly what was saved, and a material short one map is still usable,
    the same tolerance :meth:`~.document.ClayDoc.set_mesh` gives a seam that
    no longer fits rather than refusing the whole edit.
    """
    home = Path(home)
    path = _entry_path(home, entry_id)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        log.warning("clay material library: could not read %s", path, exc_info=True)
        return None

    try:
        kwargs: dict[str, Any] = {
            "name": str(payload.get("name") or entry_id),
            "base_color_factor": tuple(
                float(c) for c in payload.get("base_color_factor", (0.8, 0.8, 0.8, 1.0))
            ),
            "metallic_factor": float(payload.get("metallic_factor", 0.0)),
            "roughness_factor": float(payload.get("roughness_factor", 0.6)),
            "emissive_factor": tuple(
                float(c) for c in payload.get("emissive_factor", (0.0, 0.0, 0.0))
            ),
            "double_sided": bool(payload.get("double_sided", False)),
            "alpha_mode": str(payload.get("alpha_mode", "OPAQUE")),
            "alpha_cutoff": float(payload.get("alpha_cutoff", 0.5)),
        }
    except (TypeError, ValueError):
        log.warning("clay material library: malformed fields in %s", path, exc_info=True)
        return None

    textures = payload.get("textures", {})
    if isinstance(textures, dict):
        for slot, filename in textures.items():
            if slot not in TEXTURE_SLOTS or not isinstance(filename, str):
                continue
            tex_path = library_dir(home) / filename
            try:
                pixels = pixelguard.decode_rgba(tex_path, f"a saved material's {slot} texture")
            except Exception:
                log.warning(
                    "clay material library: dropped %s's %s texture (%s)",
                    entry_id, slot, tex_path, exc_info=True,
                )
                continue
            height, width = pixels.shape[:2]
            kwargs[slot] = (int(width), int(height), pixels.tobytes())

    return gltf.Material(**kwargs)


def delete_material(home: Path | str, entry_id: str) -> bool:
    """Remove an entry and its texture side cars. -> whether anything was there.

    Every one of :data:`TEXTURE_SLOTS`' possible side car names is unlinked
    unconditionally (``missing_ok``), since a texture slot the manifest never
    listed leaves no other trace that would need cleaning up, and a manifest
    that failed to parse (:func:`_read_entry` returned ``None`` for it, so it
    never appears in :func:`list_materials`) still has its side cars swept by
    name here.

    The side cars go **first** and the JSON manifest **last** -- the mirror
    of :func:`save_material`'s own fix, and the 2026-09-19 audit's same
    finding, clay-32: unlinking the manifest first left an interruption's
    PNGs on disk with no manifest left to name them, an orphan
    :func:`list_materials` can never see and this function can then never be
    asked to sweep by id. Deleting the textures before the manifest means an
    interrupted delete leaves, at worst, a manifest whose texture slots
    :func:`load_material` already tolerates missing -- never an orphaned PNG.
    """
    home = Path(home)
    path = _entry_path(home, entry_id)
    existed = path.is_file()
    for slot in TEXTURE_SLOTS:
        _texture_path(home, entry_id, slot).unlink(missing_ok=True)
    path.unlink(missing_ok=True)
    return existed
