"""Regression tests for the 2026-09-26 audit's w1f3 slice.

Five findings land here because their fixes are all in ``studio``-root
modules: the journal's payload naming (shell-documents-01), the Inker menu's
duplicate rows (shell-chrome-04), ``dialogs.modal_open``'s missing Plotter/
Inker predicates (shell-chrome-05), and the boot sequence's exception
handling and lock ordering (shell-boot-01, shell-boot-02).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from realmspinner.studio import journal

# --- shell-documents-01: the journal's payload name, checked against disk -----


class _Ctx:
    """A Ctx with a task runner that runs inline, so a submit is a write.

    Copied from ``tests/studio/test_journal.py``'s own fixture rather than
    imported: that module's ``kind`` fixture mutates the shared provider
    registry and is scoped to its own file.
    """

    def __init__(self, root: Path, *, accept: bool = True) -> None:
        self.svc = SimpleNamespace(config=SimpleNamespace(autosave_dir=root))
        self.state = SimpleNamespace()
        self.submitted: list[str] = []
        self.accept = accept

    def submit(self, key: str, run: Any, *args: Any, **kwargs: Any) -> bool:
        self.submitted.append(key)
        if not self.accept:
            return False
        result = run(*args, **kwargs)
        journal.on_task_done(self, SimpleNamespace(key=key, result=result))
        return True

    def toast(self, text: str, level: str = "info", **_kw: Any) -> None:
        pass


class _Slot:
    def __init__(self, uid: str = "s1", title: str = "thing", body: bytes = b"a") -> None:
        self.uid = uid
        self.title = title
        self.body = body
        self.head = 1
        self.journal_name = ""
        self.journal_head = None
        self.journal_at = 0.0


@pytest.fixture
def probe_kind(monkeypatch):
    slots: list[_Slot] = []
    provider = journal.Provider(
        kind="probe",
        ext=".probe",
        label="probe",
        slots=lambda ctx: list(slots),
        uid_of=lambda s: s.uid,
        title_of=lambda s: s.title,
        head_of=lambda s: s.head,
        encode=lambda s: s.body,
        adopt=lambda ctx, path, meta: True,
    )
    before = dict(journal._PROVIDERS)
    journal.register(provider)
    yield SimpleNamespace(provider=provider, slots=slots)
    journal._PROVIDERS.clear()
    journal._PROVIDERS.update(before)


def test_a_new_sessions_first_copy_never_overwrites_an_unrecovered_crash_copy(
    tmp_path, probe_kind
):
    """shell-documents-01, the 2026-09-26 audit: every mode's uid counter is a
    fresh ``itertools.count(1)`` per process (Clay's ``bd1``, and every other
    kind shaped the same way), so a new session's first untitled document
    computes exactly the same payload name as the *previous* session's first
    untitled document -- and ``write()`` never checked disk before claiming
    it, so it overwrote a crash copy nobody had recovered yet.
    """
    ctx = _Ctx(tmp_path)
    # A previous session's still-unrecovered crash copy, sitting under the
    # exact name this session's first slot will independently compute.
    stale_name = journal.payload_name(probe_kind.provider, _Slot(uid="s1", title="thing"))
    stale_payload = tmp_path / stale_name
    stale_payload.write_bytes(b"stale-unrecovered-work")
    journal.meta_path(stale_payload).write_text(
        json.dumps(
            {"version": journal.VERSION, "kind": "probe", "title": "thing", "uid": "s1", "at": 1.0}
        ),
        encoding="utf-8",
    )

    slot = _Slot(uid="s1", title="thing", body=b"new-session-work")
    probe_kind.slots.append(slot)
    journal.pump(ctx, now=1.0)  # first sight arms the debounce
    journal.pump(ctx, now=journal.JOURNAL_SECONDS + 1.0)  # past the debounce: writes

    assert stale_payload.read_bytes() == b"stale-unrecovered-work", (
        "the new session's write clobbered the previous session's unrecovered crash copy"
    )
    # The new session's own copy still landed, just under a different name.
    assert slot.journal_name and slot.journal_name != stale_name
    assert (tmp_path / slot.journal_name).read_bytes() == b"new-session-work"


# --- shell-chrome-04: the Inker menu's duplicate rows --------------------------


def test_the_inker_menu_never_lists_two_rows_with_the_same_label_under_one_root():
    """shell-chrome-04, the 2026-09-26 audit: the Inker menu is built from
    both ``palette.commands`` and ``inker_ops.OPS``, and only ``export_sheet``/
    ``export_gif`` were shadowed against the palette's own generic document
    commands -- so Save, Undo and Redo each drew twice under the same root
    with the identical label.
    """
    from collections import Counter

    from realmspinner.studio import menus
    from realmspinner.studio.modes.inker import state as inker_state

    ctx = SimpleNamespace(
        state=SimpleNamespace(
            mode="inker",
            selected=None,
            wireframe=False,
            turntable=False,
            show_fps=False,
            filters=SimpleNamespace(trash=False),
            errors=[],
            inker=inker_state.InkerState(),
        ),
        cache=SimpleNamespace(get=lambda _key: None, jobs=[]),
        runtime=SimpleNamespace(checks=[]),
        viewer=None,
    )
    rows = menus.specs(ctx, evaluate=False)
    counts = Counter((row.path[0], row.label) for row in rows if row.path)
    dupes = {key: n for key, n in counts.items() if n > 1}
    assert dupes == {}, f"duplicate (root, label) pairs in the Inker menu: {dupes}"


# --- shell-chrome-05: modal_open's missing Plotter/Inker predicates -----------


def test_modal_open_is_true_while_plotters_new_map_or_inkers_size_dialog_is_up():
    """shell-chrome-05, the 2026-09-26 audit: ``dialogs.modal_open`` did not
    know about Plotter's New-map modal or Inker's Image size / Canvas size
    dialogs, so Delete/Esc/Ctrl+Z/Ctrl+K reached the document behind them.
    Both predicates are plain flags -- ``modal_open`` also runs from a bare
    ``ctx`` with no imgui context at all, in headless tests, and asking imgui
    directly there (``imgui.is_popup_open``) segfaults rather than raising.
    """
    from realmspinner.studio import dialogs
    from realmspinner.studio.modes.inker.ui.panes import canvas as inker_canvas
    from realmspinner.studio.modes.plotter.ui.panes import canvas as plotter_canvas

    ctx = SimpleNamespace(
        confirms=dialogs.ConfirmQueue(),
        prompts=dialogs.PromptQueue(),
        state=SimpleNamespace(_library_export=None, _library_convert=None),
    )
    assert dialogs.modal_open(ctx) is False

    plotter_canvas._setup_open = True
    try:
        assert dialogs.modal_open(ctx) is True
    finally:
        plotter_canvas._setup_open = False
    assert dialogs.modal_open(ctx) is False

    inker_canvas._size_dialogs_open = True
    try:
        assert dialogs.modal_open(ctx) is True
    finally:
        inker_canvas._size_dialogs_open = False
    assert dialogs.modal_open(ctx) is False


# --- shell-boot-01: a non-OSError inside _setup_logging ------------------------


def test_run_alerts_when_get_config_raises_a_non_oserror_inside_logging_setup(monkeypatch):
    """shell-boot-01, the 2026-09-26 audit: ``_setup_logging``'s own
    ``get_config`` call caught only ``OSError``, so a ``migrate.MigrationError``
    (a ``RuntimeError``, not an ``OSError``) escaped ``_setup_logging`` as a
    bare traceback -- before ``run``'s own ``try/except Exception`` around its
    second ``get_config`` call ever got a chance to show the "cannot use its
    home directory" alert. ``get_config`` is patched to always raise, so both
    calls -- the one inside ``_setup_logging`` and ``run``'s own, later one --
    hit it the same way.
    """
    import logging as logging_mod

    import realmspinner.config as config_mod
    from realmspinner import instance, migrate
    from realmspinner.studio import main

    root = logging_mod.getLogger()
    before_handlers = list(root.handlers)
    before_level = root.level
    try:
        said: list[tuple[str, str]] = []
        monkeypatch.setattr(instance, "alert", lambda title, body: said.append((title, body)))
        monkeypatch.setattr(instance, "ask", lambda title, body: False)
        monkeypatch.setattr(main, "_install_excepthooks", lambda: None)

        def boom() -> Any:
            raise migrate.MigrationError("cannot open jobs.sqlite: [Errno 13] Access is denied")

        monkeypatch.setattr(config_mod, "get_config", boom)

        assert main.run() == 1
        assert said and "home directory" in said[0][0]
    finally:
        for handler in list(root.handlers):
            if handler not in before_handlers:
                root.removeHandler(handler)
        root.handlers[:] = before_handlers
        root.setLevel(before_level)
        if main._crash_log is not None:
            main._crash_log.close()
            main._crash_log = None


# --- shell-boot-02: the lock before get_config and file logging ---------------


def test_run_takes_the_instance_lock_before_get_config_and_file_logging(monkeypatch):
    """shell-boot-02, the 2026-09-26 audit: the instance lock used to be taken
    *after* ``_setup_logging`` had already called ``get_config`` (migrate +
    mkdirs) and opened ``realmspinner.log``/``crash.log`` -- so a refused
    second instance still migrated and held a rotating handler on the first
    instance's log. Locked out here by making the lock fail, and proving
    neither ``get_config`` nor ``_setup_logging`` ever ran.
    """
    from realmspinner import instance
    from realmspinner.studio import main

    said: list[tuple[str, str]] = []
    monkeypatch.setattr(instance, "alert", lambda title, body: said.append((title, body)))
    monkeypatch.setattr(
        instance.InstanceLocks, "acquire", lambda self, *, allow_unsafe=False: False
    )

    calls: list[str] = []

    def get_config_spy() -> Any:
        calls.append("get_config")
        raise AssertionError("get_config must not run before the lock is held")

    def setup_logging_spy() -> None:
        calls.append("_setup_logging")
        raise AssertionError("_setup_logging must not run before the lock is held")

    import realmspinner.config as config_mod

    monkeypatch.setattr(config_mod, "get_config", get_config_spy)
    monkeypatch.setattr(main, "_setup_logging", setup_logging_spy)

    assert main.run() == 1
    assert calls == [], f"ran before the lock check: {calls}"
    assert said, "the lock-refused dialog never fired"
