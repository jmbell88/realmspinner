"""service-gates-01 (2026-09-26 audit): ``sweep_staging`` takes no lease.

``_download``'s phase one renames each finished child's staging tree to a
``held`` sibling (``.<dest name>.<token>.fetch.part``) before phase two ever
calls ``publish.begin`` -- so for the whole gap between "the bytes landed"
and "the journal exists", that tree carries no resume marker (the worker
unlinks it on success, before the ``no-publish`` early return) and appears in
no journal's ``staged_dirs``. ``sweep_staging`` is the pane's own opportunistic
cleanup, called from another ``TaskRunner`` thread whenever Settings opens,
and it took no lease of its own -- so a sweep landing in exactly that gap saw
an ordinary ``.fetch.part`` directory, matched neither exemption, and deleted
a download ``_download`` was one line away from publishing.
"""

from __future__ import annotations

import threading

from realmspinner import leases
from realmspinner.service import downloads as svc_downloads


def test_sweep_staging_does_not_remove_a_held_tree_while_a_download_holds_the_lease(svc):
    root = svc.config.t2i_model_root
    held = root / ".sdxl-base-1.0.deadbeef.fetch.part"
    held.mkdir(parents=True)
    (held / "keepme.bin").write_bytes(b"x")

    entered = threading.Event()
    release = threading.Event()

    def _hold() -> None:
        with leases.MODELS.maintain():
            entered.set()
            release.wait(5)

    holder = threading.Thread(target=_hold)
    holder.start()
    try:
        assert entered.wait(5), "the holder thread never took the maintenance lease"
        removed = svc_downloads.sweep_staging(svc)
    finally:
        release.set()
        holder.join(5)

    assert removed == []
    assert held.is_dir()
    assert (held / "keepme.bin").exists()
