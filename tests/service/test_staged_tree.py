"""``service.export.staged_tree``: swap a whole directory in atomically-ish.

``staged_copy_all`` narrows the window between "nothing" and "the new file"
to two renames for a *pair* of served files; ``staged_tree`` is the same
trick generalised to however many files a caller wants under one export
folder -- a character's frame tree plus manifest, or a renamed GLB beside a
Godot ``.tscn``, which must land together or not at all.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from realmspinner.service import export as svc_export
from realmspinner.service.errors import Invalid


def _leftover_dirs(dest_root: Path) -> list[Path]:
    """Any ``.name.hex.tmp`` / ``.name.hex.old`` sibling ``staged_tree`` made."""
    return [p for p in dest_root.iterdir() if p.name.startswith(".")]


def test_a_tree_lands_whole_under_its_name(tmp_path):
    def write(tmp_dir: Path) -> None:
        (tmp_dir / "manifest.json").write_text("{}", "utf-8")
        sub = tmp_dir / "walk" / "n"
        sub.mkdir(parents=True)
        (sub / "000.png").write_bytes(b"png")

    dest = svc_export.staged_tree(tmp_path, "Hero", write)

    assert dest == tmp_path / "Hero"
    assert (dest / "manifest.json").read_text("utf-8") == "{}"
    assert (dest / "walk" / "n" / "000.png").read_bytes() == b"png"


def test_a_write_that_fails_part_way_leaves_nothing_at_the_destination(tmp_path):
    def write(tmp_dir: Path) -> None:
        (tmp_dir / "partial.png").write_bytes(b"only-this-much")
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        svc_export.staged_tree(tmp_path, "Hero", write)

    assert not (tmp_path / "Hero").exists()
    assert list(tmp_path.iterdir()) == []


def test_re_exporting_replaces_the_folder_whole(tmp_path):
    def write_v1(tmp_dir: Path) -> None:
        (tmp_dir / "stale.png").write_bytes(b"v1")
        (tmp_dir / "manifest.json").write_text("v1", "utf-8")

    def write_v2(tmp_dir: Path) -> None:
        (tmp_dir / "manifest.json").write_text("v2", "utf-8")

    svc_export.staged_tree(tmp_path, "Hero", write_v1)
    dest = svc_export.staged_tree(tmp_path, "Hero", write_v2)

    assert (dest / "manifest.json").read_text("utf-8") == "v2"
    # v1's file is not merged forward into the new tree -- the whole folder
    # was replaced, not updated in place.
    assert not (dest / "stale.png").exists()


def test_a_failed_rewrite_leaves_the_previous_export_intact(tmp_path):
    def write_v1(tmp_dir: Path) -> None:
        (tmp_dir / "manifest.json").write_text("good", "utf-8")

    def write_v2(tmp_dir: Path) -> None:
        (tmp_dir / "manifest.json").write_text("garbage", "utf-8")
        raise RuntimeError("boom")

    dest = svc_export.staged_tree(tmp_path, "Hero", write_v1)
    with pytest.raises(RuntimeError, match="boom"):
        svc_export.staged_tree(tmp_path, "Hero", write_v2)

    assert (dest / "manifest.json").read_text("utf-8") == "good"


@pytest.mark.parametrize(
    "bad_name",
    [
        "../escape",
        "sub/dir",
        "sub\\dir",
        "",
        "   ",
        ".",
        "..",
        "CON",
        "con.txt",
        "NUL",
    ],
)
def test_a_name_with_a_path_in_it_is_refused(tmp_path, bad_name):
    with pytest.raises(Invalid) as excinfo:
        svc_export.staged_tree(tmp_path, bad_name, lambda tmp_dir: None)
    assert excinfo.value.field == "name"
    # Refusing must not create anything -- not even the safe half of a bad name.
    assert list(tmp_path.iterdir()) == []


def test_no_temp_or_old_directories_survive_a_successful_export(tmp_path):
    def write_v1(tmp_dir: Path) -> None:
        (tmp_dir / "manifest.json").write_text("v1", "utf-8")

    def write_v2(tmp_dir: Path) -> None:
        (tmp_dir / "manifest.json").write_text("v2", "utf-8")

    svc_export.staged_tree(tmp_path, "Hero", write_v1)
    svc_export.staged_tree(tmp_path, "Hero", write_v2)

    assert [p.name for p in tmp_path.iterdir()] == ["Hero"]
    assert _leftover_dirs(tmp_path) == []


def test_a_failed_move_aside_leaves_no_temp_folder(tmp_path, monkeypatch):
    """Defect, fixed 2026-09-13: ``staged_tree`` removed its temp directory
    only when ``write`` itself raised. If the *move-aside* of the previous
    export failed instead -- a file inside it locked by another program --
    the freshly written temp directory was left behind forever under a
    hidden ``.<stem>.<hex>.tmp`` name nothing ever sweeps."""

    def write_v1(tmp_dir: Path) -> None:
        (tmp_dir / "manifest.json").write_text("v1", "utf-8")

    def write_v2(tmp_dir: Path) -> None:
        (tmp_dir / "manifest.json").write_text("v2", "utf-8")

    svc_export.staged_tree(tmp_path, "Hero", write_v1)

    real_replace = os.replace
    calls: list[tuple[Path, Path]] = []

    def failing_replace(src, dst, *a, **k):
        calls.append((Path(src), Path(dst)))
        if len(calls) == 1:
            # The move-aside of the previous export: refuse it, as a locked
            # file inside it would.
            raise PermissionError("locked by another program")
        return real_replace(src, dst, *a, **k)

    monkeypatch.setattr(svc_export.os, "replace", failing_replace)

    with pytest.raises(PermissionError):
        svc_export.staged_tree(tmp_path, "Hero", write_v2)

    # dest is untouched -- the move-aside never got that far.
    assert (tmp_path / "Hero" / "manifest.json").read_text("utf-8") == "v1"
    # And nothing hidden is left beside it.
    assert _leftover_dirs(tmp_path) == []


def test_a_failed_restore_keeps_the_original_error(tmp_path, monkeypatch):
    """Defect, fixed 2026-09-13: when the swap onto ``dest`` failed *and* the
    attempt to restore the previous export also failed, the restore's own
    ``OSError`` propagated in place of the swap failure that actually
    explains what went wrong -- masking it instead of chaining onto it."""

    def write_v1(tmp_dir: Path) -> None:
        (tmp_dir / "manifest.json").write_text("v1", "utf-8")

    def write_v2(tmp_dir: Path) -> None:
        (tmp_dir / "manifest.json").write_text("v2", "utf-8")

    svc_export.staged_tree(tmp_path, "Hero", write_v1)

    real_replace = os.replace
    calls: list[tuple[Path, Path]] = []

    def failing_replace(src, dst, *a, **k):
        calls.append((Path(src), Path(dst)))
        if len(calls) == 1:
            # The move-aside: let it succeed, so a restore is attempted below.
            return real_replace(src, dst, *a, **k)
        # Both the swap (call 2) and the restore (call 3) fail.
        raise OSError(f"boom-{len(calls)}")

    monkeypatch.setattr(svc_export.os, "replace", failing_replace)

    with pytest.raises(OSError) as excinfo:
        svc_export.staged_tree(tmp_path, "Hero", write_v2)

    # The swap's own failure (call 2) is what propagates...
    assert "boom-2" in str(excinfo.value)
    # ...with the restore's failure (call 3) chained onto it, not lost.
    assert excinfo.value.__cause__ is not None
    assert "boom-3" in str(excinfo.value.__cause__)
    # And the temp directory that never made it to `dest` is still cleaned up.
    assert not any(p.name.endswith(".tmp") for p in _leftover_dirs(tmp_path))
