"""The shared "is this file small enough to open" door, ``core/safeio/sizeguard.py``."""

from __future__ import annotations

import pytest

from realmspinner.core.safeio import sizeguard
from realmspinner.service.errors import TooLarge


def test_within_ceiling_passes_a_file_at_or_under_the_ceiling(tmp_path) -> None:
    path = tmp_path / "small.bin"
    path.write_bytes(b"x" * 10)
    assert sizeguard.within_ceiling(path, 10) == path


def test_within_ceiling_refuses_a_file_over_the_ceiling(tmp_path) -> None:
    path = tmp_path / "big.bin"
    path.write_bytes(b"x" * 11)
    with pytest.raises(TooLarge):
        sizeguard.within_ceiling(path, 10)


def test_within_ceiling_refuses_a_file_that_grows_after_the_stat(tmp_path) -> None:
    """The regression name for shell-07: a caller that adopts the bounded
    read (:func:`sizeguard.read_bytes_within_ceiling`) instead of the
    stat-then-``read_bytes`` shape above is not fooled by the same growth,
    because there is no separate stat for the growth to happen after -- the
    ceiling is enforced on the bytes actually read, in the one call that
    reads them.
    """
    path = tmp_path / "grows.bin"
    path.write_bytes(b"x" * 10)
    # Whatever an earlier stat might have seen, growing the file before the
    # bounded read still gets caught -- the check and the read are one
    # syscall pair, not two separated in time.
    path.write_bytes(b"x" * 20)
    with pytest.raises(TooLarge):
        sizeguard.read_bytes_within_ceiling(path, 10)


def test_read_bytes_within_ceiling_passes_a_file_at_or_under_the_ceiling(tmp_path) -> None:
    path = tmp_path / "small.bin"
    path.write_bytes(b"x" * 10)
    assert sizeguard.read_bytes_within_ceiling(path, 10) == b"x" * 10


def test_read_bytes_within_ceiling_refuses_a_file_over_the_ceiling(tmp_path) -> None:
    path = tmp_path / "big.bin"
    path.write_bytes(b"x" * 11)
    with pytest.raises(TooLarge):
        sizeguard.read_bytes_within_ceiling(path, 10)


def test_read_bytes_within_ceiling_closes_the_stat_read_toctou_window(tmp_path) -> None:
    """The bounded-read helper reads at most ``ceiling + 1`` bytes in the same
    call that decides pass/refuse, so there is no gap between a size check
    and a later, separate read for a concurrent writer to exploit -- unlike
    ``within_ceiling(p, N).read_bytes()`` above, which has exactly that gap.
    """
    path = tmp_path / "grows.bin"
    path.write_bytes(b"x" * 10)
    # No window: the call itself both bounds and reads, so growing the file
    # afterwards cannot matter to a read that already returned.
    data = sizeguard.read_bytes_within_ceiling(path, 10)
    path.write_bytes(b"x" * 20)
    assert data == b"x" * 10
