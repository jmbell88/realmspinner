"""Move a ``~/.warlock`` home to ``~/.realmspinner``, once.

The product was renamed from Warlock Studio to Realmspinner on 2026-09-19.
:mod:`realmspinner.config` points every root at ``~/.realmspinner`` now, and this
module is what makes that a change of address rather than a change of subject --
the same bargain :mod:`realmspinner.migrate` struck when the roots moved out of
the source checkout, and deliberately a *second* module rather than a fifth
entry in that one's ``_ROOTS``: the two answer different questions (``migrate``
moves four named trees out of a checkout, this moves one home to another) and
chaining them in one pass would make each one's preconditions depend on the
other's outcome.

**Pure, in the ``vram.py`` sense.** Stdlib only. ``config.get_config()`` calls
:func:`run` and everything in the app calls ``get_config``, so an import here is
an import in every process the project has.

**A rename, not a copy, in the case this was written for.** Both paths are
children of ``Path.home()``, so they are on one volume and ``os.rename`` is
atomic, instantaneous and needs no free space -- which matters, because the tree
being moved routinely holds 95 GB of model weights. The copy-verify-delete path
below is the fallback for what rename cannot serve (a junction, a mount, a home
whose two children somehow straddle volumes), and it keeps ``migrate``'s
ordering exactly: nothing is deleted until both sides have been recounted and
agree.

Three preconditions, all checked before anything moves:

* **The user has not already chosen.** ``REALMSPINNER_HOME`` set means the user
  named their own home, and this leaves it alone. So does any per-root variable
  pointing *into* the legacy home: moving the tree out from under a path the
  user configured by hand would break it silently.
* **The destination is not already a library.** A populated ``~/.realmspinner``
  is somebody's data; merging two homes is not a thing this can get right
  unattended.
* **Nothing else is live.** Tested by taking ``BEGIN EXCLUSIVE`` on the legacy
  ``jobs.sqlite``, the same real lock ``migrate`` uses.

``REALMSPINNER_NO_MIGRATE`` turns it off. ``REALMSPINNER_MIGRATE_KEEP=1`` keeps
the legacy tree after a successful copy, for a user who would rather delete it
by hand.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover -- import cycle; config imports this module
    from .config import Config

#: The home the app owned under its old name.
LEGACY_HOME_NAME = ".warlock"

#: Appended to whatever :mod:`realmspinner.migrate` may already have written there.
BREADCRUMB = "MIGRATED.txt"

SPACE_MARGIN = 1.1

#: Internal sidecars and locks whose names carried the product, renamed inside
#: the moved tree. The constants that look for them (``publish.JOURNAL_NAME``,
#: ``instance.MODEL_LOCK_NAME``, ...) moved with the rename, so a publish
#: transaction interrupted before the upgrade would otherwise never be
#: recovered -- nothing would be looking for its journal.
SIDECARS: tuple[tuple[str, str], ...] = (
    (".warlock-resume.json", ".realmspinner-resume.json"),
    (".warlock-publish.json", ".realmspinner-publish.json"),
    (".warlock-txn.json", ".realmspinner-txn.json"),
    (".warlock-fetch.json", ".realmspinner-fetch.json"),
    (".warlock-db.lock", ".realmspinner-db.lock"),
    (".warlock-models.lock", ".realmspinner-models.lock"),
    ("warlock.log", "realmspinner.log"),
)

#: The document extensions whose leading ``w`` stood for Warlock, renamed inside
#: the moved tree so a document saved in the library survives the rename. The
#: app refuses the old suffix outright, so one left behind is unopenable.
#: Documents saved *outside* the home are the user's to rename -- the CHANGELOG
#: entry for this release says so.
SUFFIXES: tuple[tuple[str, str], ...] = (
    (".wscn", ".rscn"),
    (".wblk", ".rblk"),
    (".wpack", ".rpack"),
    (".wmap", ".rmap"),
    (".wsng", ".rsng"),
)

#: Every path variable that can point at something inside the legacy home.
#: Checked rather than assumed: a user who pointed one at ``~/.warlock/models``
#: by hand gets their home left alone rather than moved out from under them.
#:
#: Not derived from ``config.SETTINGS``: that table also carries non-path
#: settings (ports, timeouts, booleans), so it is hand-curated here the same
#: way ``config.SETTINGS`` itself is (see that table's own comment on why it
#: is a table and not a derivation). ``REALMSPINNER_EXPORT_DIR`` and
#: ``REALMSPINNER_TRELLIS_EXE`` were missing -- the 2026-09-23 audit, finding
#: service-06 -- so a user who pointed either inside ``~/.warlock`` had it
#: moved out from under them the moment this rename ran, exactly the failure
#: every other entry here already exists to prevent.
_ROOT_VARS: tuple[str, ...] = (
    "REALMSPINNER_DATA_DIR",
    "REALMSPINNER_DB",
    "REALMSPINNER_BENCH_DIR",
    "REALMSPINNER_EVIDENCE_DIR",
    "REALMSPINNER_PALETTE_DIR",
    "REALMSPINNER_EXPORT_DIR",
    "REALMSPINNER_TRELLIS_EXE",
    "REALMSPINNER_T2I_ROOT",
    "REALMSPINNER_T2I_DIR",
    "REALMSPINNER_TRELLIS_MODELS",
    "REALMSPINNER_TRELLIS_RUNTIME",
    "REALMSPINNER_FAMILIAR_MODELS",
    "REALMSPINNER_FAMILIAR_RUNTIME",
)


class RenameError(RuntimeError):
    """The move could not be made safely. Nothing has been deleted."""


def _is_empty(path: Path) -> bool:
    try:
        return not any(os.scandir(path))
    except OSError:
        return True


def _tree_size(root: Path) -> tuple[int, int]:
    """``(files, bytes)`` under ``root``, following no symlinks."""
    files = 0
    total = 0
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                stack.append(Path(entry.path))
            else:
                files += 1
                with contextlib.suppress(OSError):
                    total += entry.stat(follow_symlinks=False).st_size
    return files, total


def _human(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TB"  # pragma: no cover -- unreachable, the loop ends at TB


def legacy_home() -> Path:
    """Where the app kept its data under the old name."""
    return Path.home() / LEGACY_HOME_NAME


def _blocked_by(legacy: Path) -> str | None:
    """Why this migration must not run, or ``None`` if it may.

    Separate from :func:`run` so the reason can be stated once, on stderr, and
    tested without moving anything.
    """
    if os.environ.get("REALMSPINNER_HOME"):
        return "REALMSPINNER_HOME is set"
    for var in _ROOT_VARS:
        raw = os.environ.get(var)
        if not raw:
            continue
        try:
            chosen = Path(raw).resolve()
        except OSError:  # pragma: no cover -- an unresolvable user-supplied path
            continue
        if chosen == legacy or legacy in chosen.parents:
            return f"{var} points inside {legacy}"
    return None


def _refuse_if_live(legacy: Path) -> None:
    """Raise unless nothing else is using the legacy job store.

    ``BEGIN EXCLUSIVE`` on the legacy ``jobs.sqlite``: a real lock rather than
    ``session.marker``, which survives a crash and would block the migration
    forever.

    **Taken and released, where ``migrate._no_live_writer`` holds its
    transaction open across the whole copy.** That difference is not a
    weakening of RUN-02's fix, it is what makes the move possible at all: an
    open sqlite connection keeps ``jobs.sqlite-shm`` open, and on Windows a
    directory containing an open handle can neither be renamed nor copied --
    so holding the lock across the move made the migration fail against its own
    guard ("[WinError 33] another process has locked a portion of the file",
    every start, forever). The guarantee here is therefore "nothing was live a
    moment ago", and the two callers earn the rest differently: the rename path
    is a single atomic syscall, so the window is microseconds, and the copy
    path re-takes the lock for the one file that can actually change under it
    (see :func:`_copy_verify_delete`, which snapshots the database through
    ``sqlite3.Connection.backup`` rather than copying its bytes).
    """
    db = legacy / "assets" / "jobs.sqlite"
    if not db.exists():
        return
    try:
        conn = sqlite3.connect(str(db), timeout=0)
    except sqlite3.Error as exc:  # pragma: no cover -- a corrupt or unreadable file
        raise RenameError(f"cannot open {db}: {exc}") from exc
    try:
        try:
            conn.execute("BEGIN EXCLUSIVE")
        except sqlite3.OperationalError as exc:
            raise RenameError(
                f"another Realmspinner (or Warlock Studio) process is using {db} -- "
                f"close it and start again (moving a live library would lose "
                f"whatever it is writing)"
            ) from exc
        except sqlite3.DatabaseError:
            # Not a database at all -- truncated, foreign, or a leftover of an
            # older layout. Nothing can be holding it as one, which is the only
            # question this asks, so the move goes ahead and carries the file
            # verbatim like every other byte in the tree.
            return
        with contextlib.suppress(sqlite3.Error):
            conn.execute("ROLLBACK")
    finally:
        conn.close()


def _require_space(dest: Path, needed: int) -> None:
    required = int(needed * SPACE_MARGIN)
    try:
        free = shutil.disk_usage(dest.anchor).free
    except OSError as exc:  # pragma: no cover -- an unreadable volume root
        raise RenameError(f"cannot measure free space on {dest.anchor}: {exc}") from exc
    if free < required:
        raise RenameError(
            f"not enough room on {dest.anchor} to move {legacy_home()} to {dest}: "
            f"{_human(required)} required, {_human(free)} free. Set "
            f"REALMSPINNER_HOME={legacy_home()} to keep using the old location."
        )


def retitle(root: Path) -> int:
    """Rename the sidecars and documents under ``root``. -> how many moved.

    Bottom-up, so a rename never invalidates a directory this walk has yet to
    descend into. Each rename is individually best-effort: a file that cannot be
    renamed (open, read-only) costs that one file's continuity, and failing the
    whole migration over it would be worse -- the tree has already been moved
    and published by the time this runs.
    """
    renamed = 0
    for parent, _dirs, files in os.walk(root, topdown=False):
        for name in files:
            new: str | None = None
            for old_name, new_name in SIDECARS:
                # Matched at the end, not only whole: two of these are applied
                # as *suffixes* to a file they guard rather than used as
                # filenames -- ``instance.DB_LOCK_SUFFIX`` makes
                # ``jobs.sqlite.realmspinner-db.lock`` -- so a whole-name test
                # walked straight past the real library's own db lock.
                if name == old_name or name.endswith(old_name):
                    new = name[: len(name) - len(old_name)] + new_name
                    break
            else:
                for old_ext, new_ext in SUFFIXES:
                    if name.endswith(old_ext):
                        new = name[: -len(old_ext)] + new_ext
                        break
            if new is None:
                continue
            target = Path(parent) / new
            if target.exists():
                continue
            with contextlib.suppress(OSError):
                os.rename(Path(parent) / name, target)
                renamed += 1
    return renamed


#: The job store and its write-ahead sidecars. Never byte-copied: the committed
#: database is the ``.sqlite`` file *plus* whatever is still in ``-wal``, so a
#: plain copy of the three can capture them mid-checkpoint, and the ``-shm`` is
#: a memory-mapped lock file that is not meaningful anywhere but the machine and
#: moment it was made. Snapshotted through ``sqlite3.Connection.backup``
#: instead, which is the same thing ``JobStore.backup_to`` does.
_STORE_FILES = ("jobs.sqlite", "jobs.sqlite-wal", "jobs.sqlite-shm")


def _ignore_store(_dir: str, names: list[str]) -> set[str]:
    return {n for n in names if n in _STORE_FILES}


def _tree_size_excluding_store(root: Path) -> tuple[int, int]:
    files, total = _tree_size(root)
    for name in _STORE_FILES:
        candidate = root / "assets" / name
        if candidate.is_file():
            files -= 1
            total -= candidate.stat().st_size
    return files, total


def _try_rename(legacy: Path, dest: Path) -> bool:
    """Move the home with one atomic syscall. -> whether it worked.

    Its own function so the copy fallback has a seam a test can force. Patching
    ``os.rename`` instead would also disable :func:`retitle`, which uses it for
    every document it renames inside the moved tree -- and would then quietly
    assert nothing about the fallback at all.
    """
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            dest.rmdir()
        os.rename(legacy, dest)
    except OSError:
        # Cross-volume, a junction, or a handle held somewhere in the tree.
        return False
    return True


def _copy_verify_delete(legacy: Path, dest: Path) -> None:
    """``migrate._move``'s order, for the case ``os.rename`` cannot serve.

    Copy, verify, delete, and it is not negotiable: the copy lands in a staging
    directory beside the destination, so a crash halfway through leaves an
    ``.incoming`` to delete rather than a half-populated home that the next
    start would mistake for a finished move.

    The job store is the one thing not copied as bytes. It is held exclusively
    and snapshotted afterwards, which is both safer (a byte copy of a WAL
    database can catch it mid-checkpoint) and the only way this can work at
    all: the exclusive hold is an open handle, and on Windows an open handle
    inside a directory blocks copying that directory.
    """
    files, total = _tree_size_excluding_store(legacy)
    _require_space(dest, total)
    staging = dest.parent / f"{dest.name}.incoming"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(staging, ignore_errors=True)
    try:
        shutil.copytree(legacy, staging, symlinks=True, ignore=_ignore_store)
    except OSError as exc:
        shutil.rmtree(staging, ignore_errors=True)
        raise RenameError(f"could not copy {legacy} to {dest}: {exc}") from exc

    copied_files, copied_bytes = _tree_size(staging)
    if (copied_files, copied_bytes) != (files, total):
        shutil.rmtree(staging, ignore_errors=True)
        raise RenameError(
            f"the copy of {legacy} did not match the original "
            f"({copied_files} files / {copied_bytes} bytes copied, "
            f"{files} / {total} expected). Nothing has been deleted."
        )

    try:
        _snapshot_store(legacy, staging)
    except (OSError, sqlite3.Error) as exc:
        shutil.rmtree(staging, ignore_errors=True)
        raise RenameError(
            f"the library copied, but its job history could not be taken from "
            f"{legacy}: {exc}. Nothing has been deleted."
        ) from exc

    if dest.exists():
        # Empty -- run() declined otherwise -- and os.replace will not rename
        # onto an existing directory even so.
        try:
            dest.rmdir()
        except OSError as exc:
            shutil.rmtree(staging, ignore_errors=True)
            raise RenameError(f"could not clear {dest}: {exc}") from exc
    os.replace(staging, dest)


def _snapshot_store(legacy: Path, staging: Path) -> None:
    """Copy the legacy job store into ``staging`` as a consistent database.

    Through ``sqlite3.Connection.backup``, the same as ``JobStore.backup_to``
    and ``migrate._carry_the_database`` -- not ``shutil.copy2``. The store runs
    in WAL mode, so the committed database is the ``.sqlite`` file *plus*
    whatever is still in its ``-wal`` sidecar; a plain byte copy of the
    ``.sqlite`` alone silently drops every transaction since the last
    checkpoint, and copying all three can catch them mid-checkpoint.

    **No exclusive transaction is taken here, and that is the point of using
    ``backup`` at all.** The API takes its own read lock and restarts itself if
    a writer commits underneath it, which is exactly the guarantee wanted; an
    exclusive hold on this same connection instead *deadlocks* it -- ``backup``
    never returns, and the first cut of this module hung the whole app at
    startup. :func:`_refuse_if_live` has already established that nothing else
    was live a moment ago; this is what makes the snapshot itself consistent.
    """
    source_path = legacy / "assets" / "jobs.sqlite"
    if not source_path.is_file():
        return
    target_path = staging / "assets" / "jobs.sqlite"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        source = sqlite3.connect(str(source_path))
    except sqlite3.Error:  # pragma: no cover -- an unreadable file
        shutil.copy2(source_path, target_path)
        return
    try:
        target = sqlite3.connect(str(target_path))
        try:
            source.backup(target)
        finally:
            target.close()
    except sqlite3.DatabaseError:
        # Not a database at all -- truncated, foreign, or a leftover of an
        # older layout. Carried verbatim, like any other file in the tree.
        target_path.unlink(missing_ok=True)
        shutil.copy2(source_path, target_path)
    finally:
        source.close()


def _breadcrumb(dest: Path, legacy: Path, renamed: int) -> None:
    lines = [
        f"Realmspinner moved its data here from {legacy} on {datetime.now():%Y-%m-%d %H:%M}.",
        "This app was called Warlock Studio before that release.",
    ]
    if renamed:
        lines.append(f"  {renamed} file(s) renamed off the old .w* names.")
    lines.append("")
    try:
        dest.mkdir(parents=True, exist_ok=True)
        with (dest / BREADCRUMB).open("a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    except OSError:
        # Advisory. A note that could not be written is not a reason to fail a
        # move that has already succeeded.
        pass


def run(config: Config) -> str | None:
    """Move ``~/.warlock`` to ``config.home``. -> the destination, or ``None``.

    Idempotent by construction: a successful run leaves nothing at the legacy
    path, so every later call is one ``is_dir()`` check.
    """
    if os.environ.get("REALMSPINNER_NO_MIGRATE"):
        return None
    legacy = legacy_home()
    dest = config.home
    try:
        if not legacy.is_dir() or _is_empty(legacy):
            return None
        if legacy.resolve() == dest.resolve():
            return None
        if dest.exists() and not _is_empty(dest):
            # A populated destination is a library in its own right.
            return None
    except OSError:  # pragma: no cover -- an unreadable home
        return None

    blocked = _blocked_by(legacy)
    if blocked:
        print(
            f"realmspinner: {legacy} was not moved to {dest} because {blocked}.",
            file=sys.stderr,
            flush=True,
        )
        return None

    # stderr rather than the log: this runs inside get_config(), long before
    # studio.main installs a file handler, and a move with no output at all is
    # indistinguishable from a hang.
    print(
        f"realmspinner: moving {legacy} to {dest} -- this happens once.",
        file=sys.stderr,
        flush=True,
    )

    keep = os.environ.get("REALMSPINNER_MIGRATE_KEEP") == "1"
    renamed_in_place = False

    # Proved and released before anything moves -- see _refuse_if_live for why
    # holding it across the move cannot work on Windows.
    _refuse_if_live(legacy)

    if not keep:
        # Falls through to the copy, which raises an error of its own if it
        # cannot manage either.
        renamed_in_place = _try_rename(legacy, dest)
    if not renamed_in_place:
        _copy_verify_delete(legacy, dest)

    if not renamed_in_place and not keep:
        # Outside the exclusive hold, deliberately: that hold is an open handle
        # on the legacy ``jobs.sqlite``, and Windows will not unlink a tree that
        # contains one. Safe here because ``dest`` is already published, so a
        # legacy tree that survives costs disk space and nothing else.
        shutil.rmtree(legacy, ignore_errors=True)

    renamed = retitle(dest)
    _breadcrumb(dest, legacy, renamed)
    print(f"realmspinner: moved to {dest}.", file=sys.stderr, flush=True)
    return str(dest)
