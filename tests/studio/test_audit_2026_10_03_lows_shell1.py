"""Regressions for the 2026-10-03 audit's Low findings, fixer ``shell1``.

One file for the whole batch (shell-44 .. shell-82): the findings share the
shell/settings/chrome owners rather than a theme. Each test's name is the claim
it proves; the docstring says which finding it closes.
"""

from __future__ import annotations

import ast
import inspect
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from realmspinner.studio import main as main_mod
from realmspinner.studio import menus, modes, palette
from realmspinner.studio.modes.settings.ui.panes import app_settings

REPO = Path(__file__).resolve().parents[2]
MANUAL = REPO / "docs" / "manual"


# --- shell-44: a single-asset Convert is deliberately silent -------------------


def test_a_single_asset_convert_is_not_reported_as_an_unclaimed_task(caplog):
    """``library._start_convert`` runs a one-card Convert under
    ``convert:<id>`` and returns ``None`` -- the inner ``save:`` task toasts --
    but the key was neither claimed nor in ``SILENT_TASK_KEYS``, so every
    convert logged the line reserved for routing bugs."""
    app = main_mod.App.__new__(main_mod.App)
    app._unclaimed = set()
    app.app_ctx = SimpleNamespace(toast=lambda *a, **k: None, state=SimpleNamespace(preview={}))
    done = SimpleNamespace(key="convert:abc123", result=None, ok=True, message="", action=None)
    with caplog.at_level("INFO"):
        app._on_task_done(done)
    assert not [r for r in caplog.records if "nowhere to deliver" in r.message]


# --- shell-45: every switch the Settings pane draws can be searched for --------


def _switch_labels(function: Any) -> list[str]:
    """The literal label of every ``.switch(...)`` call in ``function``.

    ``form_ui.switch(id, label, ...)`` carries the label second; the bare
    ``controls.switch(label, ...)`` carries it first. The label is whichever of
    the first two string constants has a space or a capital -- the id is
    snake_case.
    """
    tree = ast.parse(inspect.getsource(function).lstrip())
    labels: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "switch"
        ):
            for arg in node.args[:2]:
                if (
                    isinstance(arg, ast.Constant)
                    and isinstance(arg.value, str)
                    and (" " in arg.value or arg.value[:1].isupper())
                ):
                    labels.append(arg.value)
                    break
    return labels


def test_every_switch_label_in_settings_is_found_by_searching_it():
    """``SEARCH_INDEX`` omitted "Don't ask for clean cutouts" and "Check for
    updates on startup", so the Settings search answered "Nothing matches."
    for a control that exists."""
    labels = [
        *_switch_labels(app_settings._interface),
        *_switch_labels(app_settings._updates),
    ]
    assert "Don't ask for clean cutouts" in labels, "the AST scan lost the cutout switch"
    assert "Check for updates on startup" in labels, "the AST scan lost the update switch"
    for label in labels:
        # A parenthetical hint ("(F10)") is the chord, not part of the name.
        query = re.sub(r"\s*\(.*?\)", "", label)
        assert app_settings.search_rows(query), f"searching {label!r} finds nothing"
    for guess in ("cutout", "matte", "updates on startup"):
        assert app_settings.search_rows(guess), f"searching {guess!r} finds nothing"


# --- shell-52: chapter 20 does not contradict Settings -> Startup --------------


def test_the_overview_does_not_say_no_mode_is_ever_remembered():
    """Chapter 20 said the app "opens on Home, every launch: no mode is
    remembered between runs"; Settings -> Startup's Last workspace (chapters
    21 and 42) does reopen the last mode."""
    text = (MANUAL / "20-overview.md").read_text(encoding="utf-8")
    sentence = text.split("## The window", 1)[1].strip().split("\n\n", 1)[0]
    assert sentence.startswith("The app opens on Home"), "the section's opening moved"
    assert "Last workspace" in sentence, "the sentence must name the option that remembers"
    assert "every launch: no mode is remembered" not in text


# --- shell-54 / shell-55: palette wording --------------------------------------


def _palette_ctx(mode: str = "create") -> Any:
    from realmspinner.studio.state import ManualState

    return SimpleNamespace(
        state=SimpleNamespace(
            mode=mode,
            previous_mode=mode,
            mode_observed=mode,
            create=SimpleNamespace(stage="mesh"),
            selected=None,
            source_job=None,
            wireframe=False,
            turntable=False,
            show_fps=False,
            manual=ManualState(),
        ),
        cache=SimpleNamespace(jobs=[], get={}.get),
        viewer=None,
        svc=SimpleNamespace(store=SimpleNamespace(trashed=lambda: [])),
    )


def test_the_generate_greyed_reason_names_a_real_place():
    """The generate command's reason named "the 2D or 3D generate pane", the
    stale vocabulary shell-chrome-08 removed from ``_VIEWPORT_WHY``. Already
    fixed when this batch started; pinned so it stays so."""
    command = next(c for c in palette.commands(_palette_ctx("inker")) if c.key == "generate")
    assert "2D" not in command.why and "3D" not in command.why
    assert "Create" in command.why


#: The palette commands whose run opens a dialog, popup or switcher: the only
#: rows allowed an ellipsis ("A label ending in an ellipsis opens a dialog; one
#: without does not", chapter 20).
_DIALOG_COMMANDS = {"save-as", "workspace-layout", "empty-trash", "export"}


def test_a_palette_label_ends_in_an_ellipsis_only_when_its_command_opens_a_dialog():
    """"Delete the selected asset..." carried an ellipsis though the command
    trashes immediately (shell-chrome-01 removed its confirm)."""
    for mode in modes.KEYS:
        for command in palette.commands(_palette_ctx(mode)):
            if command.label.endswith("..."):
                assert command.key in _DIALOG_COMMANDS, (
                    f"{command.key!r} is labelled {command.label!r} but opens no dialog"
                )
    delete = next(c for c in palette.commands(_palette_ctx()) if c.key == "delete")
    assert not delete.label.endswith("...")


# --- shell-57: the manual names every mode that keeps the arrows ---------------


def test_the_manual_names_every_nav_key_mode():
    """Chapter 38 listed six surfaces ("In those six") while
    ``NAV_KEY_MODES`` holds eight: Muse and Sirens were never named."""
    text = (MANUAL / "38-shortcuts.md").read_text(encoding="utf-8")
    paragraph = text.split("**The arrow keys belong to whatever is on screen.**", 1)[1]
    paragraph = paragraph.strip().split("\n\n", 1)[0]
    words = {6: "six", 7: "seven", 8: "eight", 9: "nine"}
    assert f"In those {words[len(modes.NAV_KEY_MODES)]}" in paragraph
    for name in ("Home", "library", "Review", "Poser", "Inker", "Plotter", "Muse", "Sirens"):
        assert name in paragraph, f"{name} is not named in the arrow-key paragraph"


# --- shell-59: no ordinal counts a mode that the registry does not hold --------

_ORDINALS = [
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth",
    "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth",
]


def test_no_comment_in_modes_names_a_mode_ordinal_beyond_the_registry():
    """Mason "the fourteenth mode", Muse "the thirteenth", Sirens "the
    twelfth" and the palette's "a thirteenth mode would gain a switch segment"
    were landing-order ordinals from before Troupe folded into Poser; counting
    ``MODES`` found a fourteenth that does not exist."""
    pattern = re.compile(r"\b(" + "|".join(_ORDINALS) + r")\s+mode\b", re.IGNORECASE)
    registry = Path(modes.__file__).read_text(encoding="utf-8")
    assert not pattern.findall(registry), (
        "modes/__init__.py names a mode by landing order, which goes stale the next time "
        "one is added or folded"
    )
    palette_text = Path(palette.__file__).read_text(encoding="utf-8")
    for found in pattern.findall(palette_text):
        assert _ORDINALS.index(found.lower()) + 1 == len(modes.MODES) + 1, (
            f"palette.py's hypothetical {found!r} mode is not the one after the {len(modes.MODES)}"
        )


# --- shell-67: Download selected names the real cause of its grey --------------


def test_download_selected_names_a_removal_when_a_removal_is_what_is_running():
    """``busy`` is a download *or* a removal, but the bulk button's reason said
    "A download is already running." for both."""
    source = inspect.getsource(app_settings._models)
    call = source[source.index('f"Download selected'):]
    window = call[: call.index("_selection_progress")]
    assert "A download is already running." not in window
    assert "_MODEL_BUSY_REASON" in window


# --- shell-68: a refused measurement is asked for again ------------------------


@pytest.fixture
def _fresh_latches():
    app_settings._reset_measure()
    app_settings._reset_sweep()
    yield
    app_settings._reset_measure()
    app_settings._reset_sweep()


def _refusing_ctx() -> tuple[Any, list[str], dict[str, bool]]:
    mode = {"accept": False}
    submitted: list[str] = []

    def submit(key: str, fn: Any, *args: Any, **kwargs: Any) -> bool:
        submitted.append(key)
        return mode["accept"]

    ctx = SimpleNamespace(
        submit=submit,
        svc=SimpleNamespace(config=None),
        tasks=SimpleNamespace(any_busy=lambda prefix: False),
        model_storage=None,
        evidence_storage=None,
    )
    return ctx, submitted, mode


def test_a_refused_measure_submit_is_asked_again_on_the_next_frame(
    monkeypatch, tmp_path, _fresh_latches
):
    """``_MEASURED``, ``_EVIDENCE_MEASURED``, ``_SWEPT`` and ``_STAGED_PENDING``
    were latched before ``ctx.submit`` and its refusal (a task of the same key
    still in flight) was ignored, so a refresh requested during a running
    measurement was lost."""
    monkeypatch.setattr(app_settings.widgets, "muted", lambda *a, **k: None)

    ctx, submitted, mode = _refusing_ctx()
    app_settings._model_storage(ctx)
    app_settings._evidence_storage(ctx)
    app_settings._sweep_staging(ctx)
    assert submitted == ["model-storage", "evidence-storage", "sweep-staging"]
    mode["accept"] = True
    app_settings._model_storage(ctx)
    app_settings._evidence_storage(ctx)
    app_settings._sweep_staging(ctx)
    assert submitted[3:] == ["model-storage", "evidence-storage", "sweep-staging"], (
        "a refused submit latched the measurement as done"
    )
    # And once accepted, it is not asked a third time.
    app_settings._model_storage(ctx)
    app_settings._evidence_storage(ctx)
    app_settings._sweep_staging(ctx)
    assert len(submitted) == 6

    from realmspinner.service import updates as svc_updates

    monkeypatch.setattr(svc_updates, "staging_dir", lambda svc: tmp_path)
    (tmp_path / "setup.exe").write_bytes(b"x")
    info = {"installer_name": "setup.exe", "sha256": "ab" * 32}
    ctx2, submitted2, mode2 = _refusing_ctx()
    assert app_settings._staged(ctx2, info) is None
    mode2["accept"] = True
    assert app_settings._staged(ctx2, info) is None
    assert submitted2 == [app_settings.STAGED_TASK_KEY] * 2, (
        "a refused verification was recorded as pending and never re-submitted"
    )
    assert app_settings._staged(ctx2, info) is None
    assert len(submitted2) == 2, "an accepted verification must not be re-submitted each frame"


# --- shell-69: the training-folder scan has a ceiling ---------------------------


def test_training_folder_scan_stops_at_the_trainers_ceiling(tmp_path):
    """``training_images`` listed and stat'ed every entry of a folder on the
    frame thread, though the trainer takes at most ``MAX_IMAGES``."""
    from realmspinner.pipelines import lora_train

    big = tmp_path / "photos"
    big.mkdir()
    for index in range(lora_train.MAX_IMAGES + 40):
        (big / f"{index:04d}.png").write_bytes(b"")
    found = app_settings.training_images(big)
    assert len(found) == lora_train.MAX_IMAGES + 1, "the scan did not stop one past the ceiling"
    assert found == sorted(found)

    small = tmp_path / "few"
    small.mkdir()
    for name in ("b.png", "a.png", "notes.txt"):
        (small / name).write_bytes(b"")
    assert [p.name for p in app_settings.training_images(small)] == ["a.png", "b.png"]


# --- shell-70: a release feed cannot smuggle a scheme or a path ----------------


def test_a_release_feed_cannot_name_a_non_http_release_url_or_a_path_as_installer_name(
    monkeypatch,
):
    """``release_url`` reached ``os.startfile`` and ``installer_name`` a
    filesystem join with only GitHub's own sanitising in between."""
    from realmspinner.pipelines import download, update_worker

    installer = "RealmspinnerSetup.exe"
    manifest_url = "https://example.invalid/m.json"

    def serve(html_url: str, filename: str):
        routes = {
            update_worker.RELEASES_URL: json.dumps(
                {
                    "tag_name": "v1",
                    "html_url": html_url,
                    "assets": [
                        {
                            "name": update_worker.MANIFEST_ASSET,
                            "browser_download_url": manifest_url,
                        },
                        {"name": filename, "browser_download_url": "https://example.invalid/i"},
                    ],
                }
            ).encode(),
            manifest_url: json.dumps(
                {"version": "1", "installer": {"filename": filename, "sha256": "a" * 64}}
            ).encode(),
        }

        class _Response:
            def __init__(self, body: bytes) -> None:
                self.body = body

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self, n=-1):
                out, self.body = self.body[:n], self.body[n:]
                return out

        monkeypatch.setattr(
            download, "open_url", lambda url, *, timeout=None: _Response(routes[url])
        )

    serve("file:///C:/Windows/System32/calc.exe", installer)
    assert update_worker.check({})["release_url"] == ""

    serve("javascript:alert(1)", installer)
    assert update_worker.check({})["release_url"] == ""

    serve("https://github.com/o/r/releases/tag/v1", installer)
    assert update_worker.check({})["release_url"] == "https://github.com/o/r/releases/tag/v1"

    for bad in ("..\\evil.exe", "C:\\evil.exe", "sub/evil.exe", "/abs.exe", ".."):
        serve("https://github.com/o/r", bad)
        with pytest.raises(ValueError, match="filename"):
            update_worker.check({})


# --- shell-71: first-run and the rail gate wait for the same rows ---------------


def test_first_run_generation_rows_are_the_creates_gate_rows():
    """``first_run.GENERATION_ROWS`` and ``modes.NEEDS_ROWS["create"]`` were the
    same three keys written twice; chapter 42 described a two-row gate. The
    equality half passes on the unfixed tree (the two lists agreed that day);
    the derivation and the chapter wording are what failed."""
    from realmspinner.studio.panes import first_run

    assert modes.NEEDS_ROWS["create"] == first_run.GENERATION_ROWS
    source = Path(first_run.__file__).read_text(encoding="utf-8")
    assert '"base:sdxl_cfg"' not in source, "first_run spells the gate rows out a second time"
    chapter = (MANUAL / "42-app-settings.md").read_text(encoding="utf-8")
    assert "until both rows are present" not in chapter
    assert len(modes.NEEDS_ROWS["create"]) == 3
    assert "until all three" in chapter


# --- shell-72: queued-row keys drop the jobs cache ------------------------------


class _FakeCache:
    def __init__(self) -> None:
        self.invalidated = 0

    def invalidate(self) -> None:
        self.invalidated += 1


def _dispatch(key: str, result: Any = None) -> _FakeCache:
    cache = _FakeCache()
    app = SimpleNamespace(
        app_ctx=SimpleNamespace(
            state=SimpleNamespace(preview={}),
            cache=cache,
            svc=None,
            toast=lambda *a, **k: None,
        ),
        runtime=SimpleNamespace(checks=[]),
        _unclaimed=set(),
    )
    main_mod.App._on_task_done(app, SimpleNamespace(key=key, result=result, ok=True, tag=None))
    return cache


def test_a_landed_lora_train_or_sprite_submit_drops_the_jobs_cache():
    """"lora:train" and "sprite:<id>" each queue a job row but never dropped the
    cache, so the row waited for the 3 s idle read after the toast said
    "Training queued"."""
    assert _dispatch("lora:train", {"id": "job1"}).invalidated == 1
    assert _dispatch("sprite:ref1", {"id": "job2"}).invalidated == 1


# --- shell-81: every palette command has a menu path ----------------------------


def test_every_palette_command_has_a_menu_path():
    """A command with no ``_COMMAND_PATHS`` entry is silently dropped from the
    menu bar, so it is reachable only through Ctrl+K while chapter 38 promises
    the menu has it."""
    for mode in modes.KEYS:
        ctx = _palette_ctx(mode)
        commands = palette.commands(ctx)
        drawn = {spec.identity for spec in menus._command_specs(ctx, commands, evaluate=False)}
        for command in commands:
            assert f"command:{command.key}" in drawn, (
                f"{command.key!r} ({mode}) is in the palette but not in the menu bar"
            )
    text = (MANUAL / "38-shortcuts.md").read_text(encoding="utf-8")
    assert "Everything in the palette is also in the menu bar, and the reverse" not in text


# --- shell-82: the pure helpers say what the pane shows -------------------------


def test_the_settings_pure_helpers_say_what_the_pane_shows():
    assert app_settings.update_size_note({}) == ""
    assert app_settings.update_size_note({"size_bytes": 0}) == ""
    assert app_settings.update_size_note({"size_bytes": -5}) == ""
    assert app_settings.update_size_note({"size_bytes": 418 * 1024**2}) == "418 MB"
    # Rounded, not truncated: 417.6 MB reads as 418.
    assert app_settings.update_size_note({"size_bytes": int(417.6 * 1024**2)}) == "418 MB"

    assert app_settings._vram_note({}) == ""
    assert app_settings._vram_note({"vram_gib": None}) == ""
    assert app_settings._vram_note({"vram_gib": "7"}) == ""
    assert app_settings._vram_note({"vram_gib": 7.4}) == "About 7 GB of VRAM while it is loaded."
    assert app_settings._vram_note({"vram_gib": 12}) == "About 12 GB of VRAM while it is loaded."

    assert app_settings.downloadable_note({"present": True}) == ""
    assert app_settings.downloadable_note({"downloadable": True}) == ""
    note = app_settings.downloadable_note({})
    assert "no download recipe" in note and "by hand" in note
