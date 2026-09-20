"""W1.12: the Cancel button beside an in-progress download explains itself.

The reviewer's original wording ("Cancel discards what has downloaded")
claimed the opposite of what the fetch path actually does: ``_cancel`` kills
the tracked child (``winjob.terminate_tracked("fetch")``), which never runs
``fetch_worker.fetch_one``'s own unwind -- the staging tree, and the resume
marker in it, are left exactly as a dropped connection would leave them
(F1, 2026-09-05: ``fetch_worker.py`` keeps the staging tree across a failed
attempt on purpose, and ``service/downloads.py``'s sweep spares any tree
carrying that marker). So Cancel and a lost connection are the same case for
the partial file, and the copy beside the button has to say that instead.
"""

from __future__ import annotations

import inspect


def _cancel_source() -> str:
    from realmspinner.studio.modes.settings.ui.panes import app_settings

    return inspect.getsource(app_settings._cancel)


def test_cancel_says_what_has_downloaded_is_kept():
    source = _cancel_source()
    # Either muted spelling. The claim is that the sentence is a *muted note*
    # beside the button, which ``muted_wrapped`` is -- it wraps, and in a 300 px
    # sidebar this sentence needs to. Pinning the exact call was pinning a
    # formatting choice as though it were the contract.
    assert "widgets.muted(" in source or "widgets.muted_wrapped(" in source
    assert "keeps what has downloaded" in source
    assert "resumes" in source
    # It must not claim the opposite of the verified behaviour.
    assert "discard" not in source.lower()
