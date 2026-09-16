"""The driver's pure half.

Import-level only, deliberately: pressing a control needs a real window and a
GL context, which is the whole reason ``exercise_mode.py`` is a script rather
than a test. What *can* be pinned here is everything that decides what the
driver does -- the refusal list, the digest, and the verdict classifier -- and
those are the parts whose being wrong would make a whole run's findings wrong
without anything looking broken.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(name: str):
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        # Registered before execution: a dataclass in the module resolves its
        # string annotations through ``sys.modules``, so a module that is not
        # there yet fails at its own decorator.
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SCRIPTS))


@pytest.fixture(scope="module")
def driver():
    return _load("exercise_mode")


def test_the_refusal_list_is_not_empty_and_names_the_two_hazards(driver):
    assert driver.REFUSED
    assert driver.refused("Quit")
    assert driver.refused("Save As...")
    assert driver.refused("open…")
    # And nothing broad: a pattern that swallowed ordinary controls would skip
    # exactly what the pass exists to test.
    assert not driver.refused("Save")
    assert not driver.refused("Bucket")
    assert not driver.refused("Export sheet")


def test_exercise_mode_refuses_every_control_that_opens_a_native_file_dialog(driver):
    """The 2026-09-14 audit, finding pipelines-04, closed by shell-09.

    Every native-picker call in the tree now runs inside a ``def run():``
    closure that ``ctx.submit``/``self.submit``/``docmodes.start_save`` hands
    to a task thread -- ``install_stubs`` already neutralises every one of
    those by faking ``TaskRunner.submit``, so pressing the button never
    reaches ``dialogs.open_file`` at all during a run. Settings' "Import a
    LoRA file..." and "Train from a folder..." used to be the only two calls
    in the tree that called ``dialogs.open_file``/``select_folder`` straight
    from the button handler, on the frame thread, matching no REFUSED stem --
    so an unattended exercise pass parked on a real OS file dialog forever.
    shell-09 (the 2026-09-14 audit) moved both into ``def run(): ...``
    closures submitted under the "preview" key, the same shape
    "Train from my library..." beside them already used.

    Swept rather than asserted by name alone, and AST-depth-counted rather
    than text-counted so a properly wrapped call (nested two ``def``s deep,
    inside a closure) does not re-trip this the way a plain substring count
    would: counting every *inline* (frame-thread) ``dialogs.open_file``/
    ``select_folder``/``save_file`` call left in ``app_settings.py`` catches
    a new one added later and not wrapped or added to REFUSED, the same way
    this pair was missed the first time.
    """
    from test_app_settings_pickers import _inline_picker_calls

    from warlock.studio.panes import app_settings

    source = Path(app_settings.__file__).read_text(encoding="utf-8")
    calls = _inline_picker_calls(source)
    assert calls == [], (
        f"app_settings.py now calls a native picker inline (on the frame "
        f"thread, outside any ``def run(): ...`` closure) {len(calls)} "
        f"time(s): {calls} -- wrap each new one in a closure submitted "
        "through ctx.submit, the way its neighbours in this file already are"
    )
    # REFUSED still names both buttons. Harmless now that neither opens a
    # picker inline -- the driver would press them safely either way, since
    # ``install_stubs`` fakes ``TaskRunner.submit`` -- so left in place
    # rather than removed for a cosmetic tidy-up.
    assert driver.refused("Import a LoRA file...")
    assert driver.refused("Train from a folder...")
    # Not a rubber stamp: a label with similar wording that does *not* touch a
    # picker (a library scan, not a dialog -- app_settings.py says so at its
    # own call site) must stay pressable, or the driver would silently skip
    # coverage of a control this pass exists to test.
    assert not driver.refused("Train from my library...")


def test_isolate_home_clears_every_env_var_config_resolves_under_home():
    """The 2026-09-14 audit, finding pipelines-03.

    ``_appharness._ROOTS_UNDER_HOME`` used to be a hand-written six-entry
    tuple. ``config.py`` grew four more ``_env_path(name, _home() / ...)``
    roots after it was written -- ``WARLOCK_TRELLIS_MODELS``,
    ``WARLOCK_TRELLIS_RUNTIME``, ``WARLOCK_FAMILIAR_RUNTIME`` and
    ``WARLOCK_FAMILIAR_MODELS`` -- and none of the four was ever added, so a
    harness run left them pointed at the developer's real ``~/.warlock`` and
    leaked those paths into screenshot/exercise captures.

    The expected set is scanned here with a regex over ``config.py``'s own
    source, independent of ``_appharness``'s own ``ast``-based walk -- a bug
    shared between the two implementations would otherwise cancel out and
    this test would pass for the wrong reason.
    """
    import re

    from warlock import config as config_mod

    harness = _load("_appharness")
    source = Path(config_mod.__file__).read_text(encoding="utf-8")
    expected = set(
        re.findall(r'_env_path\(\s*"(WARLOCK_[A-Z0-9_]+)"\s*,\s*_home\(\)', source)
    )
    assert expected, "the regex found nothing -- it has gone stale against config.py"
    assert set(harness._ROOTS_UNDER_HOME) == expected


def _isolate_with(monkeypatch, tmp_path, env: dict[str, str]) -> list[tuple]:
    """Run ``isolate_home`` against a private copy of the environment, a fake
    temp dir and a recording ``atexit`` -> every cleanup it registered.

    ``os.environ`` is swapped for a plain dict rather than edited in place,
    because ``isolate_home`` pops every root under home and repoints
    ``WARLOCK_HOME`` -- in this process that is ``tests/conftest.py``'s own
    pinned throwaway home, and a test that moved it would move it for every
    test after this one in the same worker.
    """
    import os

    harness = _load("_appharness")
    fake_env = {k: v for k, v in os.environ.items() if k != "WARLOCK_HOME"}
    fake_env.pop(harness.REAL_HOME_ENV, None)
    fake_env.pop(harness.KEEP_HOME_ENV, None)
    fake_env.update(env)
    monkeypatch.setattr(os, "environ", fake_env)
    home = tmp_path / "throwaway-home"
    monkeypatch.setattr(harness.tempfile, "mkdtemp", lambda **_: str(home))
    registered: list[tuple] = []
    monkeypatch.setattr(harness.atexit, "register", lambda *a, **k: registered.append(a))

    assert harness.isolate_home() == home
    return registered


def test_isolate_home_keeps_the_throwaway_home_when_told_to(monkeypatch, tmp_path):
    """The 2026-09-15 Clay agent benchmark sitting.

    ``agent_bench.py --serve`` ran a graded session whose two ``clay_export``
    calls minted Library rows inside the throwaway home, and the atexit
    cleanup deleted the whole directory the moment the window closed --
    exactly the assets the benchmark's pre-registration says to keep until
    its results are written up.
    """
    harness = _load("_appharness")
    registered = _isolate_with(monkeypatch, tmp_path, {harness.KEEP_HOME_ENV: "1"})

    assert registered == []


def test_isolate_home_still_deletes_the_throwaway_home_by_default(monkeypatch, tmp_path):
    """The other half: ``screenshot_modes.py`` and ``exercise_mode.py`` want
    the cleanup, and keeping homes is opt-in for the one caller that needs
    it, never the new default."""
    registered = _isolate_with(monkeypatch, tmp_path, {})

    assert [args[1] for args in registered] == [tmp_path / "throwaway-home"]


def test_delta_names_only_the_components_that_moved(driver):
    before = ("inker", "", None, "brush", 4, (), 0, False, ())
    after = ("inker", "", None, "bucket", 4, (), 0, False, ())
    assert driver.describe_delta(before, after) == ["tool: 'brush' -> 'bucket'"]
    assert driver.describe_delta(before, before) == []
    assert len(driver.DIGEST_NAMES) == len(before)


def _verdict(driver, **kwargs):
    base = dict(
        raised=None,
        enabled=True,
        reason="",
        toast_levels=(),
        submitted=(),
        state_delta=[],
        pixel_delta=0.0,
    )
    return driver.verdict(**(base | kwargs))


def test_verdict_maps_each_signal_to_its_label(driver):
    assert _verdict(driver) == "inert"
    assert _verdict(driver, pixel_delta=0.5) == "pixels-changed"
    assert _verdict(driver, state_delta=["tool: a -> b"]) == "state-changed"
    assert _verdict(driver, submitted=("export:1",)) == "submitted"
    assert _verdict(driver, enabled=False, reason="Open a drawing first.") == "disabled"
    assert _verdict(driver, enabled=False) == "disabled-no-reason"
    assert _verdict(driver, toast_levels=("error",)) == "toast-error"
    assert _verdict(driver, raised="Traceback...") == "raised"


def test_verdict_orders_by_severity_not_by_convenience(driver):
    # A control that crashed is a crash whatever else it also did.
    assert (
        _verdict(
            driver,
            raised="boom",
            submitted=("x",),
            state_delta=["mode: a -> b"],
            toast_levels=("error",),
        )
        == "raised"
    )
    # And an error toast outranks the work that produced it.
    assert _verdict(driver, submitted=("x",), toast_levels=("error",)) == "toast-error"


def test_a_pixel_flicker_below_the_threshold_is_still_inert(driver):
    assert driver.PIXEL_EPSILON > 0
    assert _verdict(driver, pixel_delta=driver.PIXEL_EPSILON) == "inert"


def test_always_look_covers_every_verdict_a_reader_must_judge(driver):
    for name in ("raised", "inert", "toast-error", "disabled-no-reason", "hard-reset"):
        assert name in driver.ALWAYS_LOOK


def test_keys_are_stable_and_disambiguate_a_repeated_label(driver):
    from warlock.studio.probe import Control

    def _one(name, pane="inker_tools"):
        return Control(label=name, kind="button", rect=(0, 0, 1, 1), pane=pane)

    seen: dict[str, int] = {}
    first = driver.key_for(_one("Add"), seen)
    second = driver.key_for(_one("Add"), seen)
    assert first == "inker_tools/button/Add#0"
    assert second == "inker_tools/button/Add#1"
    assert driver.key_for(_one("Add", pane=""), {}) == "floating/button/Add#0"
    assert "/" not in driver.safe_name(first)


def test_the_harness_is_shared_rather_than_reimplemented(driver):
    """Two scripts booting the app two ways is the drift the extraction ends."""

    harness = _load("_appharness")
    for name in (
        "boot",
        "capture",
        "close_popups",
        "seed",
        "seed_asset",
        "seed_review",
        "seed_tile",
        "seed_matte",
        "WARMUP",
        "SETTLE_FRAMES",
    ):
        assert hasattr(harness, name), name
    shots = (SCRIPTS / "screenshot_modes.py").read_text(encoding="utf-8")
    assert "from _appharness import" in shots
    # The boot sequence must live in exactly one place.
    assert "app.setup_window()" not in shots
    assert "app.setup_window(size_override=size)" in (SCRIPTS / "_appharness.py").read_text(
        encoding="utf-8"
    )


def test_the_health_page_is_never_photographed():
    """A screenshot that cannot be taken without leaking is not taken.

    Settings -> Health prints the absolute paths it probed -- the user's home,
    the vendor directory, the model roots. That is its entire content, so no
    ``WARLOCK_HOME`` isolation makes a capture of it safe: it only swaps one
    real machine's paths for another's. Three such images were written into
    this public repository once, carrying a developer's username about eight
    times each and the capture's own temp path down to its session GUID.
    Deleting them was not the fix; this is, because the harness would have
    rewritten them on the next run.
    """
    shots = (SCRIPTS / "screenshot_modes.py").read_text(encoding="utf-8")
    body = "\n".join(line for line in shots.splitlines() if not line.lstrip().startswith("#"))
    assert "settings-health" not in body
    corpus = SCRIPTS.parent / "screenshots"
    if corpus.is_dir():
        assert not list(corpus.glob("*settings-health*")), "a health capture is in the corpus"


def test_the_two_audio_modes_are_photographed_with_work_in_them():
    """Sirens and Muse were the last modes whose every picture was empty.

    Ten panes between them draw an empty state until something is open: the
    tracker's grid, instruments, envelopes and order, and Muse's take tray with
    its Play, Open in Sirens, Make more and Stems controls. That is the gap
    ``--seed`` closes for Inker, Clay, Plotter and Packwright and ``--troupe``
    closes for Troupe, and the harness's own rule says it plainly -- a mode
    with no seed is a picture of nothing.

    **Neither seeder may open an audio device.** A capture has nothing to hear,
    and ``sirens_audio`` is the one module that touches ``pygame.mixer``; a
    screenshot pass that started sound would be a side effect on the machine
    that ran it. Muse's player is therefore built the way ``on_task_done``
    builds it and stops short of ``sirens_audio.play``.
    """
    harness = _load("_appharness")
    for name in ("seed_sirens", "seed_muse"):
        assert hasattr(harness, name), name
    # Asked of the compiled function rather than of the file's text, so that
    # the paragraph above -- which names the module it is forbidding -- cannot
    # fail its own test.
    for name in ("seed_sirens", "seed_muse", "_write_demo_figure"):
        used = getattr(harness, name).__code__.co_names
        assert "sirens_audio" not in used, f"{name} is starting the mixer"
    shots = (SCRIPTS / "screenshot_modes.py").read_text(encoding="utf-8")
    assert "args.music" in shots


def test_every_settings_page_is_photographed_but_the_two_that_leak():
    """Seven pages, one picture: the mode walk only ever opens Appearance.

    Settings is one mode and seven screens, and the pass derived from
    ``modes.KEYS`` draws whichever page the pane opens on -- always the first.
    So Models, Packs, Updates and Storage were the largest surfaces in the app
    with no picture of them anywhere, and a regression on any of the four
    would have been invisible to the corpus that exists to answer "did anybody
    look at this".

    The refusal is asserted in the same breath because it is the same subject.
    ``health`` prints the absolute paths it probed; ``advanced`` draws
    ``app_settings.config_table``, which prints every effective setting's
    *value*, and those values are the home, the model roots and the sqlite
    store. Neither is fixed by an isolated home -- that only swaps one real
    machine's paths for another's, and on a harness run it substitutes the
    capture's own temp directory down to its session GUID.
    """
    from warlock.studio.panes import app_settings

    shots = _load("screenshot_modes")
    assert set(shots.SETTINGS_REFUSED) == {"advanced", "health"}
    keys = [key for key, _label in app_settings.CATEGORIES]
    # Derived from the pane's own table, so a page added there is enrolled
    # here without this test being edited -- the rule the mode list follows.
    wanted = [k for k in keys[1:] if k not in shots.SETTINGS_REFUSED]
    assert wanted, "the refusal set has swallowed every page"
    corpus = SCRIPTS.parent / "screenshots"
    if not corpus.is_dir():
        return
    for theme in ("dark", "light", "pixel"):
        for key in wanted:
            assert (corpus / f"{theme}-settings-{key}.png").is_file(), f"{theme}/{key}"
        # Appearance keeps the bare name it has always had rather than gaining
        # a second file under a page-suffixed one.
        assert (corpus / f"{theme}-settings.png").is_file(), theme
        assert not (corpus / f"{theme}-settings-appearance.png").exists(), theme


def test_the_sheet_arms_of_the_create_form_are_photographed():
    """``--asset`` seeds a mesh, so the form's sheet branches need their own.

    Create's 2D form draws the Tile-layout picker, the materials list, the
    terrain fields, the sprite Action/Directions pair, Palette, Dither and
    Outline only on an ``asset_type`` no capture ever landed on -- the corpus
    looked like five pictures of the form and pictured none of those controls.
    """
    harness = _load("_appharness")
    for name in ("seed_palette", "seed_sheet_form"):
        assert hasattr(harness, name), name
    shots = _load("screenshot_modes")
    arms = {arm for _name, arm, _extra, _adv in shots.SHEET_ARMS}
    assert arms == {"tileset", "sprite_sheet"}
    modes = {extra.get("tile_mode") for _name, _arm, extra, _adv in shots.SHEET_ARMS}
    assert {"materials", "terrain"} <= modes
    # Opening the disclosure is not enough to photograph what is inside it:
    # the form is taller than the window, so at least one arm per branch has
    # to be captured scrolled to the end as well.
    assert sum(1 for _n, _a, _e, adv in shots.SHEET_ARMS if adv) >= 2


def test_the_driver_reports_its_own_blind_spot(driver):
    from test_probe import RAW_IMGUI_CONTROLS

    assert driver.raw_imgui_controls() == RAW_IMGUI_CONTROLS


# --- the throwaway home -------------------------------------------------------
#
# Driven through a subprocess rather than by importing the harness here. The
# thing under test is what happens when ``WARLOCK_HOME`` is *unset*, and this
# suite's own conftest pins it at a throwaway directory for every test in the
# run -- so an in-process check would be asking the question with the answer
# already supplied, which is exactly the shape of the defect it exists to catch.


def _harness_env(env: dict[str, str]) -> dict[str, str]:
    """Import ``_appharness`` in a clean interpreter and report the environment.

    Returns the child's own view of the four things that decide where a harness
    run writes, so the assertions below are about the process the scripts
    actually get rather than about this one.
    """
    import json
    import os
    import subprocess

    code = (
        "import json, os, sys;"
        f"sys.path.insert(0, {str(SCRIPTS)!r});"
        "import _appharness;"
        "from warlock.config import get_config;"
        "c = get_config();"
        "print(json.dumps({"
        "'home': os.environ.get('WARLOCK_HOME'),"
        "'no_migrate': os.environ.get('WARLOCK_NO_MIGRATE'),"
        "'data_dir': str(c.data_dir),"
        "'db': str(c.db_path),"
        "'models': str(c.t2i_model_root),"
        "'harness_home': str(_appharness.HARNESS_HOME) if _appharness.HARNESS_HOME else None,"
        "}))"
    )
    base = {k: v for k, v in os.environ.items() if not k.startswith("WARLOCK_")}
    base.pop("WARLOCK_HARNESS_REAL_HOME", None)
    out = subprocess.run(
        [sys.executable, "-c", code],
        env={**base, **env},
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_a_harness_run_never_lands_on_the_real_warlock_home():
    """The 2026-09-07 screenshot refresh, which nothing prevented.

    ``screenshot_modes.py`` was run with ``WARLOCK_HOME`` unset, so every root
    resolved under the developer's real ``~/.warlock`` and the captures came
    back carrying that machine's GPU and free VRAM, a dozen real job cards with
    their prompts, and two crash-recovery entries. The harness docstring had
    said to use a throwaway home since the day it was extracted; saying it is
    what this replaces.
    """
    from pathlib import Path

    seen = _harness_env({})
    assert seen["home"], "the harness left WARLOCK_HOME unset"
    home = Path(seen["home"]).resolve()
    assert home != (Path.home() / ".warlock").resolve()
    assert seen["harness_home"] == str(Path(seen["home"]))


def test_the_throwaway_home_takes_every_root_with_it():
    """``WARLOCK_HOME`` alone is not enough, which is the harness docstring's
    own warning: a per-root variable already in the environment carries that
    one root back out of the throwaway home, and ``WARLOCK_DATA_DIR`` in
    particular does not move the sqlite store with it.
    """
    from pathlib import Path

    escaped = str(Path.home() / ".warlock" / "assets")
    seen = _harness_env({"WARLOCK_DATA_DIR": escaped, "WARLOCK_T2I_ROOT": escaped})
    home = Path(seen["home"]).resolve()
    for key in ("data_dir", "db", "models"):
        assert Path(seen[key]).resolve().is_relative_to(home), f"{key} escaped to {seen[key]}"


def test_a_harness_run_never_migrates_a_checkouts_library():
    """``migrate.run`` moves ``PROJECT_ROOT/assets``, ``bench``, ``palettes``
    and ``models`` into ``config.home`` whenever the destination is empty --
    which a fresh throwaway home always is. Unguarded, a checkout still holding
    those directories would have them moved into a temp dir and then deleted by
    the harness's own cleanup.
    """
    assert _harness_env({})["no_migrate"]


def test_an_explicit_home_is_left_alone_and_so_is_an_opt_out(tmp_path):
    """Pointing the harness at a prepared library stays a one-variable job, and
    photographing your own work on purpose stays possible -- only the unset
    default, the one nobody notices, is redirected.
    """
    from pathlib import Path

    chosen = tmp_path / "prepared"
    chosen.mkdir()
    seen = _harness_env({"WARLOCK_HOME": str(chosen)})
    assert Path(seen["home"]).resolve() == chosen.resolve()
    assert seen["harness_home"] is None
    assert seen["no_migrate"], "an explicit home still must not trigger a migration"

    real = _harness_env({"WARLOCK_HARNESS_REAL_HOME": "1"})
    assert real["home"] is None
    assert real["harness_home"] is None
