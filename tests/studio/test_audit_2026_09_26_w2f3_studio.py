"""Regression test for the 2026-09-26 audit's shell-documents-04 finding:
``JobsCache.request`` clearing ``_dirty`` before knowing whether the submit
it was clearing it for was actually accepted.
"""

from __future__ import annotations

from typing import Any

from realmspinner.studio.jobs_cache import JobsCache


class _StubRunner:
    """Just enough of ``TaskRunner`` for :meth:`JobsCache.request`: a
    ``submit`` whose acceptance the test controls, the same refusal shape
    ``TaskRunner.submit`` gives when the key is already in flight."""

    def __init__(self, accept: bool) -> None:
        self.accept = accept
        self.calls = 0

    def submit(self, key: str, fn: Any, *args: Any, tag: Any = None, **kwargs: Any) -> bool:
        self.calls += 1
        return self.accept


def test_a_refused_jobs_list_submit_keeps_the_cache_dirty(svc):
    """shell-documents-04 (2026-09-26 audit).

    ``request`` used to clear ``_dirty`` unconditionally, before
    ``runner.submit`` had said whether the read was actually taken --
    ``TaskRunner.submit`` refuses a key already in flight, so a read still
    landing from the frame before swallowed the very ``invalidate()`` that
    made this frame due: nothing was submitted for it, and rows stayed stale
    for up to the 3 s idle tick.

    Fails against the unfixed code: after a refused submit, ``cache._dirty``
    is already ``False``, so a later, un-refused runner is never even asked
    again by ``_due()``.
    """
    cache = JobsCache(svc)
    assert cache._dirty is True, "a fresh cache starts dirty"

    refusing = _StubRunner(accept=False)
    assert cache.request(refusing) is False
    assert refusing.calls == 1
    assert cache._dirty is True, "a refused submit must not have consumed the dirty flag"

    accepting = _StubRunner(accept=True)
    assert cache.request(accepting) is True
    assert cache._dirty is False, "an accepted submit clears it, same as before"
