"""The 2026-09-19 rename of Warlock Studio to Realmspinner.

Four claims, and each fails against the tree as it stood at d416cb42: the old
name is gone from the source, the home directory moves itself once and takes
the names inside it with it, the installer registers as a new product that
removes the old one, and a document written under the old suffix is answered
with something the user can act on rather than a refusal.
"""

from __future__ import annotations

import os
import re
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from realmspinner import rename
from realmspinner.studio import dialogs

ROOT = Path(__file__).resolve().parents[1]

# Where the old name survives on purpose, and why. Every entry is a record of
# something that already happened or a door held open for it -- never a
# description of what the app is now. Checked in both directions below, so an
# entry that stops being needed fails just as loudly as a file that starts
# spelling the old name: a hand list that only fails open is how ``PUBLISHERS``
# went stale twice.
_ALLOWED: dict[str, str] = {
    # The 2026-09-23 audit (familiar-03): this used to exempt the whole
    # cards/ directory, but router-1.txt is never sha-pinned against a
    # trained weights pin (contract.py's own docstring: the router "never
    # gates on a trained weights pin ... prompt-engineered against whatever
    # instruct model is running, never trained on") -- so unlike the Clay
    # card below, rewriting it does not hand a fine-tuned model a prompt it
    # has never seen. Narrowed to the one tracked file that actually needs
    # the exemption; router-1.txt no longer names the old product. (A second
    # Clay card, clay-2.txt, is not yet tracked -- it is part of the user's
    # own uncommitted Q2 work -- so it is not named here; whoever commits it
    # adds its own entry then, the same way this one was added.)
    "src/realmspinner/familiar/cards/clay-1.txt": (
        "frozen fine-tune prompt, identified to the model by sha256 -- the "
        "shipped Familiar trained on text naming Warlock, and rewriting a card "
        "hands the model a prompt it has never seen"
    ),
    "CHANGELOG.md": "entries below 0.0.52 describe releases published under that name",
    "src/realmspinner/kernels/pixel/ora.py": (
        "LEGACY_MEMBER: the metadata member inside every .ora this app has ever "
        "written, read and never written, so those documents keep their slices"
    ),
    "src/realmspinner/rename.py": "the migration off the old home is named for what it moves",
    "src/realmspinner/config.py": "one comment, saying which home rename.run moves",
    "src/realmspinner/studio/dialogs.py": "LEGACY_SUFFIXES: the .w* documents and their sentence",
    "installer/realmspinner.iss": "names the old AppId, in order to uninstall it",
    "tests/kernels/geom3d/test_glbwrite.py": "records why the pinned glTF bytes moved",
    "tests/modes/inker/test_cel_z.py": "records why the pinned .ora bytes moved",
    "tests/test_rename.py": "this file",
}

_NAME = re.compile(rb"[Ww]arlock|WARLOCK")


def _tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [line for line in out.splitlines() if line.strip()]


def _names_the_old_product(rel: str) -> bool:
    path = ROOT / rel
    if not path.is_file():
        return False
    return bool(_NAME.search(path.read_bytes()))


def test_the_old_product_name_is_gone_from_every_tracked_file():
    """The guard that stops the old name creeping back in.

    A rename this size lands as one scripted sweep, and the way a sweep like
    that fails is not loudly -- it is one file that was open in an editor, or a
    path the file list did not reach, sitting there spelling the old name at a
    user months later. Checked over ``git ls-files`` rather than a directory
    walk, so ``dist/``, ``.venv/`` and the gitignored maintainer-only trees
    cannot make it pass or fail by accident. Binary files are checked too: the
    shipped character meshes carried the product in their glTF ``generator``.
    """
    offenders = [
        rel
        for rel in _tracked_files()
        if not any(rel.replace("\\", "/").startswith(a) for a in _ALLOWED)
        and _names_the_old_product(rel)
    ]
    assert not offenders, (
        "the old product name survives in:\n  "
        + "\n  ".join(offenders)
        + "\n\nRename it, or -- if it is a record of something that already "
        "happened -- add it to _ALLOWED with the reason."
    )


def test_every_allowance_for_the_old_name_is_still_needed():
    """The other direction, so the list above cannot quietly go stale.

    An exemption that no longer matches anything is worse than no exemption:
    it is a hole the next sweep falls through silently.
    """
    tracked = [rel.replace("\\", "/") for rel in _tracked_files()]
    for prefix, reason in _ALLOWED.items():
        matched = [rel for rel in tracked if rel.startswith(prefix)]
        assert matched, f"_ALLOWED names {prefix!r}, which matches no tracked file"
        assert any(_names_the_old_product(rel) for rel in matched), (
            f"_ALLOWED still exempts {prefix!r} ({reason}), but nothing under it "
            "names the old product any more -- drop the entry"
        )


def test_every_document_suffix_dropped_its_warlock_w():
    """The five document extensions, and the sentence for an old one.

    The leading ``w`` stood for Warlock, so all five moved. The formats did
    not: a ``.wblk`` is byte-for-byte a ``.rblk``, which is why the answer to
    being handed one names the rename and the new filename rather than
    reporting an unsupported file.
    """
    assert dialogs.LEGACY_SUFFIXES == {
        ".wscn": ".rscn",
        ".wblk": ".rblk",
        ".wpack": ".rpack",
        ".wmap": ".rmap",
        ".wsng": ".rsng",
    }
    for old, new in dialogs.LEGACY_SUFFIXES.items():
        hint = dialogs.legacy_suffix_hint(Path("castle" + old))
        assert hint is not None
        assert "castle" + new in hint, hint
        assert "Realmspinner" in hint
    # Upper case reaches it too: Windows hands back whatever the user typed.
    assert dialogs.legacy_suffix_hint(Path("Castle.WBLK")) is not None
    # And nothing else is caught by it.
    for other in (".rblk", ".glb", ".ora", ".png", ".tmx"):
        assert dialogs.legacy_suffix_hint(Path("castle" + other)) is None


def test_the_installer_is_a_new_product_that_removes_the_old_one():
    """A new AppId, and a step that uninstalls the one it replaces.

    Inno keys an in-place upgrade off ``AppId``, so reusing the old GUID would
    have left the product installed in a directory named for a name it no
    longer has. A new GUID alone would leave two Add/Remove Programs entries,
    which is the other half of this.
    """
    source = (ROOT / "installer" / "realmspinner.iss").read_text(encoding="utf-8")
    legacy = "C64355D5-8A1F-4A10-8DBB-7E72BCE2C297"
    found = re.search(r"^AppId=\{\{([0-9A-Fa-f-]+)\}", source, re.MULTILINE)
    assert found, "the installer must declare an AppId"
    assert found.group(1).upper() != legacy, "the renamed product needs its own AppId"
    assert legacy in source, "the old AppId must still be named, to uninstall it"
    assert "PrepareToInstall" in source
    assert r"Programs\Realmspinner" in source


class _Cfg(SimpleNamespace):
    home: Path


def _populate(legacy: Path) -> None:
    """A legacy home with one of everything the move has to carry."""
    (legacy / "assets").mkdir(parents=True)
    (legacy / "models" / "familiar").mkdir(parents=True)
    (legacy / "assets" / "jobs.sqlite").write_bytes(b"not really a database")
    (legacy / "assets" / "warlock.log").write_text("old log", encoding="utf-8")
    (legacy / "assets" / ".warlock-txn.json").write_text("{}", encoding="utf-8")
    (legacy / "assets" / ".warlock-publish.json").write_text("{}", encoding="utf-8")
    # Applied as a *suffix* to the file it guards, not used as a filename --
    # instance.DB_LOCK_SUFFIX. The real library's only leftover after the first
    # cut of retitle(), which matched whole names alone.
    (legacy / "assets" / ".jobs.sqlite.warlock-db.lock").write_text("", encoding="utf-8")
    (legacy / "models" / "familiar" / "weights.gguf").write_bytes(b"\x00" * 64)
    (legacy / "castle.wscn").write_bytes(b"PK\x03\x04scene")
    (legacy / "assets" / "hero.wblk").write_bytes(b"PK\x03\x04clay")


@pytest.fixture
def _migrating(monkeypatch, tmp_path):
    """A legacy home, an unused destination, and migration switched back on."""
    monkeypatch.delenv("REALMSPINNER_NO_MIGRATE", raising=False)
    monkeypatch.delenv("REALMSPINNER_MIGRATE_KEEP", raising=False)
    monkeypatch.delenv("REALMSPINNER_HOME", raising=False)
    for var in rename._ROOT_VARS:
        monkeypatch.delenv(var, raising=False)
    legacy = tmp_path / ".warlock"
    dest = tmp_path / ".realmspinner"
    _populate(legacy)
    monkeypatch.setattr(rename, "legacy_home", lambda: legacy)
    return SimpleNamespace(legacy=legacy, dest=dest, config=_Cfg(home=dest))


def test_the_home_moves_once_and_renames_what_is_inside_it(_migrating):
    """The whole claim of ``rename.run`` in one test.

    The library arrives, the legacy path is gone, the internal journals and the
    log come with it under their new names -- a publish transaction interrupted
    before the upgrade is otherwise never recovered, because nothing is looking
    for ``.warlock-txn.json`` any more -- and documents saved inside the home
    lose the ``w`` that stood for Warlock.
    """
    moved = rename.run(_migrating.config)
    assert moved == str(_migrating.dest)
    dest = _migrating.dest

    assert not _migrating.legacy.exists()
    assert (dest / "models" / "familiar" / "weights.gguf").read_bytes() == b"\x00" * 64
    assert (dest / "assets" / "jobs.sqlite").is_file()

    assert (dest / "assets" / "realmspinner.log").read_text(encoding="utf-8") == "old log"
    assert not (dest / "assets" / "warlock.log").exists()
    assert (dest / "assets" / ".realmspinner-txn.json").is_file()
    assert (dest / "assets" / ".realmspinner-publish.json").is_file()
    assert not (dest / "assets" / ".warlock-txn.json").exists()
    assert (dest / "assets" / ".jobs.sqlite.realmspinner-db.lock").is_file()
    assert not (dest / "assets" / ".jobs.sqlite.warlock-db.lock").exists()

    # Nothing anywhere under the new home still spells the old name.
    assert not [p.name for p in dest.rglob("*") if "warlock" in p.name.lower()]

    assert (dest / "castle.rscn").read_bytes() == b"PK\x03\x04scene"
    assert (dest / "assets" / "hero.rblk").read_bytes() == b"PK\x03\x04clay"
    assert not (dest / "castle.wscn").exists()

    assert "Warlock Studio" in (dest / rename.BREADCRUMB).read_text(encoding="utf-8")


def _real_store(legacy: Path) -> None:
    """Replace the placeholder with a genuine WAL database carrying a row."""
    db = legacy / "assets" / "jobs.sqlite"
    db.unlink()
    conn = sqlite3.connect(str(db))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE jobs (id TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO jobs VALUES ('the-one-job')")
        conn.commit()
    finally:
        conn.close()


def test_the_liveness_probe_does_not_block_the_move_it_guards(_migrating):
    """The probe must be closed before anything moves.

    The first cut held ``BEGIN EXCLUSIVE`` open across the move, the way
    ``migrate._no_live_writer`` does. On Windows that is self-defeating: an
    open sqlite connection keeps ``jobs.sqlite-shm`` open, a directory holding
    an open handle can be neither renamed nor copied, and the migration failed
    against its own guard every single start --
    ``[WinError 33] another process has locked a portion of the file`` -- with
    the library left stranded at the old name forever.

    Driven with a real WAL database rather than the placeholder the other
    tests use, because that is the whole point: a file that is not a database
    takes the ``sqlite3.DatabaseError`` branch and never opens a lock at all,
    which is exactly the case that hid the bug.
    """
    _real_store(_migrating.legacy)
    assert rename.run(_migrating.config) == str(_migrating.dest)

    moved = _migrating.dest / "assets" / "jobs.sqlite"
    assert moved.is_file()
    conn = sqlite3.connect(str(moved))
    try:
        assert conn.execute("SELECT id FROM jobs").fetchall() == [("the-one-job",)]
    finally:
        conn.close()


def test_the_copy_path_takes_the_job_history_as_a_snapshot(_migrating, monkeypatch):
    """The fallback carries the database as a database, not as bytes.

    A committed WAL database is the ``.sqlite`` file *plus* whatever is still
    in its ``-wal`` sidecar, so copying the three files can catch it
    mid-checkpoint. ``sqlite3.Connection.backup`` is what ``JobStore.backup_to``
    already uses, and it is also the only way the copy can hold its exclusive
    lock at all without blocking itself.
    """
    _real_store(_migrating.legacy)
    monkeypatch.setattr(rename, "_try_rename", lambda *_a: False)
    assert rename.run(_migrating.config) == str(_migrating.dest)

    moved = _migrating.dest / "assets" / "jobs.sqlite"
    conn = sqlite3.connect(str(moved))
    try:
        assert conn.execute("SELECT id FROM jobs").fetchall() == [("the-one-job",)]
    finally:
        conn.close()
    # The -shm is a machine-and-moment lock file; it has no meaning here.
    assert not (_migrating.dest / "assets" / "jobs.sqlite-shm").exists()
    assert (_migrating.dest / "castle.rscn").is_file()


def test_a_second_start_does_nothing(_migrating):
    """Idempotent by construction: the legacy path is gone, so it is one check."""
    assert rename.run(_migrating.config) == str(_migrating.dest)
    before = sorted(p.name for p in _migrating.dest.rglob("*"))
    assert rename.run(_migrating.config) is None
    assert sorted(p.name for p in _migrating.dest.rglob("*")) == before


def test_a_home_the_user_chose_is_left_exactly_where_they_put_it(_migrating, monkeypatch):
    """``REALMSPINNER_HOME`` set means the user has already answered this.

    Said out loud rather than passed over in silence: a user who relocated
    their library and then finds the app starting empty needs to be told why.
    """
    monkeypatch.setenv("REALMSPINNER_HOME", str(_migrating.dest))
    assert rename.run(_migrating.config) is None
    assert (_migrating.legacy / "assets" / "jobs.sqlite").is_file()


def test_a_root_pointed_inside_the_legacy_home_stops_the_move(_migrating, monkeypatch):
    """Moving the tree out from under a path the user configured by hand would
    break it silently, so the whole migration declines instead."""
    monkeypatch.setenv("REALMSPINNER_T2I_ROOT", str(_migrating.legacy / "models"))
    assert rename.run(_migrating.config) is None
    assert (_migrating.legacy / "models" / "familiar" / "weights.gguf").is_file()


def test_a_populated_destination_is_never_merged_into(_migrating):
    """Two homes have no join -- two ``jobs.sqlite`` files least of all."""
    _migrating.dest.mkdir(parents=True)
    (_migrating.dest / "assets").mkdir()
    (_migrating.dest / "assets" / "jobs.sqlite").write_bytes(b"the other library")
    assert rename.run(_migrating.config) is None
    assert (_migrating.legacy / "assets" / "jobs.sqlite").is_file()
    assert (_migrating.dest / "assets" / "jobs.sqlite").read_bytes() == b"the other library"


def test_nothing_is_deleted_when_the_copy_cannot_be_verified(_migrating, monkeypatch):
    """The order is copy, verify, delete, and it is not negotiable.

    Forced down the copy path (the rename is what runs on one volume) and then
    given a verify that disagrees: the legacy tree must still be there, and the
    staging directory must not be left behind pretending to be a finished move.
    """
    monkeypatch.setattr(rename, "_try_rename", lambda *_a: False)
    monkeypatch.setattr(rename, "_tree_size", _lying_tree_size())
    with pytest.raises(rename.RenameError):
        rename.run(_migrating.config)
    assert (_migrating.legacy / "assets" / "jobs.sqlite").is_file()
    assert (_migrating.legacy / "castle.wscn").is_file()
    assert not _migrating.dest.exists()
    assert not (_migrating.dest.parent / f"{_migrating.dest.name}.incoming").exists()


def _lying_tree_size():
    """A ``_tree_size`` that reports a different count the second time.

    Which is what a real interrupted copy looks like to the verify: the sizes
    were measured before, the copy landed short, and the two disagree.
    """
    calls = {"n": 0}

    def sized(root: Path) -> tuple[int, int]:
        calls["n"] += 1
        return (99, 99) if calls["n"] == 1 else (1, 1)

    return sized


@pytest.mark.skipif(os.name != "nt", reason="the pipe name is the Windows address")
def test_the_agent_pipe_no_longer_answers_to_the_old_name():
    """An external surface: an agent dials this name, so it had to move too."""
    from realmspinner.mcp import pipe, rpc

    address = pipe.address_for(Path(sys.prefix))
    assert "realmspinner-mcp-" in address
    assert "warlock" not in address.lower()
    assert rpc.SERVER_NAME == "realmspinner"
