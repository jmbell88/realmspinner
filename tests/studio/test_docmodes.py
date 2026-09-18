"""The rules the four document modes share, once.

Inker, Clay, Plotter and Packwright grew four copies of each of these, and four
copies of a rule are three places for it to change without the fourth. The pins
at the bottom are the point of the file: the modes must reach for *this* object
rather than keeping a fourth copy that happens to agree today.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from warlock.studio import docmodes


class _Confirms:
    def __init__(self) -> None:
        self.asked: list[Any] = []

    def ask(self, confirm: Any) -> None:
        self.asked.append(confirm)


class _State:
    def __init__(self) -> None:
        self.inker = None
        self.preview: dict[str, Any] = {}


class FakeCtx:
    def __init__(self, *, accept: bool = True) -> None:
        self.accept = accept
        self.state = _State()
        self.confirms = _Confirms()
        self.submitted: list[str] = []
        self.toasts: list[tuple[str, str]] = []

    def submit(self, key: str, run: Any, *args: Any) -> bool:
        self.submitted.append(key)
        return self.accept

    def toast(self, text: str, level: str = "info", **_kwargs: Any) -> None:
        self.toasts.append((text, level))


class _Tab:
    def __init__(self, dirty: bool = False) -> None:
        self.saving = False
        self.dirty = dirty


class _Docs:
    def __init__(self, *dirty: bool) -> None:
        self.docs = [_Tab(flag) for flag in dirty]

    @property
    def any_dirty(self) -> bool:
        return any(doc.dirty for doc in self.docs)


# --- purity -------------------------------------------------------------------


def test_it_imports_no_window_and_no_service_at_module_scope():
    """The ``recents`` pin, second instance. ``docmodes`` is imported from
    ``inker_state`` and ``plotter_state``, which every headless test of a
    document loads -- so a window import here would put imgui behind them.

    Parsed rather than grepped, because this module's own docstring names the
    three things it imports *lazily* and a substring scan cannot tell the two
    apart. numpy is the one third-party name allowed: :func:`decode_rgba`
    returns one.

    ``dataclasses`` joined the list when ``CameraView`` moved here out of
    ``clay_state`` (Mason's viewport wanted the identical class, and a second
    copy would be a second place its goal-field rule could quietly stop being
    true). It is stdlib and imports nothing, so it costs this pin's actual
    purpose -- no window, no service -- exactly nothing.
    """
    import ast

    source = (
        Path(__file__).resolve().parents[2] / "src" / "warlock" / "studio" / "docmodes.py"
    ).read_text(encoding="utf-8")
    roots: set[str] = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            roots.add("." * node.level + (node.module or "").split(".")[0])
    assert roots <= {
        "__future__",
        "collections",
        "dataclasses",
        "logging",
        "os",
        "pathlib",
        "typing",
        "numpy",
    }, roots


# --- start_save ---------------------------------------------------------------


def test_an_accepted_submit_leaves_the_tab_locked():
    ctx, tab = FakeCtx(), _Tab()
    docmodes.start_save(ctx, tab, "mode-save:1", lambda: None)
    assert tab.saving is True
    assert ctx.submitted == ["mode-save:1"]


def test_a_refused_submit_unlocks_the_tab():
    """The runner refuses a key already in flight; leaving the flag set is what
    made a tab read-only forever after a double press."""
    ctx, tab = FakeCtx(accept=False), _Tab()
    docmodes.start_save(ctx, tab, "mode-save:1", lambda: None)
    assert tab.saving is False


# --- title_for ----------------------------------------------------------------


def test_a_title_is_the_file_name_and_nothing_is_untitled():
    assert docmodes.title_for(Path("D:/atlases/hero.wpack")) == "hero.wpack"
    assert docmodes.title_for(None) == "Untitled"


# --- decode_rgba --------------------------------------------------------------


def test_an_image_decodes_to_rgba_uint8(tmp_path):
    from PIL import Image

    path = tmp_path / "one.png"
    Image.new("RGB", (5, 3), (10, 20, 30)).save(path)
    pixels = docmodes.decode_rgba(path)
    assert pixels.shape == (3, 5, 4)
    assert pixels.dtype == np.uint8
    assert tuple(pixels[0, 0]) == (10, 20, 30, 255)


# --- textures -----------------------------------------------------------------


class _Texture:
    def __init__(self, log: list[str], name: str) -> None:
        self.log = log
        self.name = name

    def release(self) -> None:
        self.log.append(f"release:{self.name}")


class _Renderer:
    def __init__(self, log: list[str]) -> None:
        self.log = log

    def forget_texture(self, texture: Any) -> None:
        self.log.append(f"forget:{texture.name}")


@pytest.fixture()
def _renderer(monkeypatch):
    log: list[str] = []
    from warlock.studio import imgui_backend

    monkeypatch.setattr(imgui_backend, "current", lambda: _Renderer(log))
    return log


def test_a_texture_is_forgotten_before_it_is_released(_renderer):
    """The other order leaves the backend holding a dead object under a GL name
    the driver will reuse."""
    docmodes.forget_texture(_Texture(_renderer, "a"))
    assert _renderer == ["forget:a", "release:a"]


def test_a_release_sweep_pops_only_the_matching_prefix(_renderer):
    ctx = FakeCtx()
    ctx.state.preview.update(
        {
            "mode_tex:t1:atlas": _Texture(_renderer, "a"),
            "mode_tex:t1:atlas:gen": 4,
            "mode_tex:t2:atlas": _Texture(_renderer, "b"),
            "other:t1": _Texture(_renderer, "c"),
        }
    )
    docmodes.release_prefix(ctx, "mode_tex:t1:")
    assert set(ctx.state.preview) == {"mode_tex:t2:atlas", "other:t1"}
    assert _renderer == ["forget:a", "release:a"]


def test_a_release_sweep_takes_the_whole_prefix_when_asked(_renderer):
    ctx = FakeCtx()
    ctx.state.preview.update(
        {
            "mode_tex:t1:atlas": _Texture(_renderer, "a"),
            "mode_tex:t2:atlas": _Texture(_renderer, "b"),
        }
    )
    docmodes.release_prefix(ctx, "mode_tex:")
    assert ctx.state.preview == {}
    assert _renderer == ["forget:a", "release:a", "forget:b", "release:b"]


# --- guard --------------------------------------------------------------------


def test_the_guard_is_silent_before_the_mode_has_ever_been_opened():
    """``getattr`` rather than ``ensure``: asking which documents are unsaved
    must not create the state that says none is."""
    ctx = FakeCtx()
    calls: list[str] = []
    assert docmodes.guard(ctx, "inker", "drawing", "drawings", "quit", lambda: calls.append("go"))
    assert calls == ["go"] and ctx.state.inker is None and ctx.confirms.asked == []


def test_a_clean_state_proceeds_without_asking():
    ctx = FakeCtx()
    ctx.state.inker = _Docs(False, False)
    calls: list[str] = []
    assert docmodes.guard(ctx, "inker", "drawing", "drawings", "quit", lambda: calls.append("go"))
    assert calls == ["go"] and ctx.confirms.asked == []


def test_one_question_covers_however_many_are_dirty():
    ctx = FakeCtx()
    ctx.state.inker = _Docs(True)
    calls: list[str] = []
    assert (
        docmodes.guard(ctx, "inker", "drawing", "drawings", "quit", lambda: calls.append("go"))
        is False
    )
    assert calls == []
    assert len(ctx.confirms.asked) == 1
    assert "One drawing has unsaved changes" in ctx.confirms.asked[0].message
    assert "if you quit" in ctx.confirms.asked[0].message

    ctx.state.inker = _Docs(True, True, False)
    docmodes.guard(ctx, "inker", "drawing", "drawings", "close", lambda: calls.append("go"))
    assert "2 drawings have" in ctx.confirms.asked[-1].message
    ctx.confirms.asked[-1].on_confirm()
    assert calls == ["go"]


# --- the pins -----------------------------------------------------------------


def test_every_mode_reaches_for_the_shared_helpers():
    """The point of the module. A fourth copy that happens to agree today is
    the drift this replaced."""
    from warlock.studio.modes.clay import mode as clay_mode
    from warlock.studio.modes.inker import mode as inker_mode
    from warlock.studio.modes.inker import state as inker_state
    from warlock.studio.modes.inker.ui.panes import textures as inker_textures
    from warlock.studio.modes.packwright.ui.panes import textures as packwright_textures
    from warlock.studio.modes.plotter import fileio as plotter_io
    from warlock.studio.modes.plotter import state as plotter_state
    from warlock.studio.modes.plotter.ui.panes import textures as plotter_textures

    assert clay_mode._start is docmodes.start_save
    assert inker_mode._start is docmodes.start_save
    assert plotter_io._start is docmodes.start_save
    assert plotter_io._decode is docmodes.decode_rgba
    assert inker_state.title_for is docmodes.title_for
    assert plotter_state.title_for is docmodes.title_for
    for module in (inker_textures, plotter_textures, packwright_textures):
        assert not hasattr(module, "_forget"), f"{module.__name__} kept its own copy"


# --- TAB_CLOSED listeners ------------------------------------------------


class _ListenerTab:
    def __init__(self, uid: str, *, dirty: bool = False, saving: bool = False) -> None:
        self.uid = uid
        self.dirty = dirty
        self.saving = saving
        self.title = uid


class ClayState:
    """Named ``ClayState`` on purpose: :func:`docmodes._mode_for` (T3) derives
    the mode key from the state class's own name, and this fixture proves it
    reads "clay" back out of that name rather than off some other guess."""

    def __init__(self, *tabs: _ListenerTab) -> None:
        self._tabs = {tab.uid: tab for tab in tabs}

    def get(self, uid: str) -> Any:
        return self._tabs.get(uid)

    def close(self, uid: str) -> None:
        self._tabs.pop(uid, None)


@pytest.fixture()
def _listener_log():
    log: list[tuple[str, str]] = []

    def listener(mode: str, uid: str) -> None:
        log.append((mode, uid))

    docmodes.TAB_CLOSED.append(listener)
    yield log
    docmodes.TAB_CLOSED.remove(listener)


def test_a_clean_close_tells_tab_closed_listeners(_listener_log):
    ctx = FakeCtx()
    state = ClayState(_ListenerTab("bd1"))
    docmodes.close_tab(ctx, state, "bd1", lambda tab: None)
    assert _listener_log == [("clay", "bd1")]


def test_a_save_in_progress_refusal_fires_no_listener(_listener_log):
    """The tab is still open (``state`` unchanged) -- a listener firing here
    would end a conversation about a document that never actually closed."""
    ctx = FakeCtx()
    state = ClayState(_ListenerTab("bd1", saving=True))
    docmodes.close_tab(ctx, state, "bd1", lambda tab: None)
    assert _listener_log == []
    assert state.get("bd1") is not None


def test_a_cancelled_dirty_close_fires_no_listener(_listener_log):
    """Asking is not closing: the listener must wait for ``on_confirm``."""
    ctx = FakeCtx()
    state = ClayState(_ListenerTab("bd1", dirty=True))
    docmodes.close_tab(ctx, state, "bd1", lambda tab: None)
    assert _listener_log == []
    assert len(ctx.confirms.asked) == 1
    assert state.get("bd1") is not None


def test_a_confirmed_dirty_close_fires_the_listener(_listener_log):
    ctx = FakeCtx()
    state = ClayState(_ListenerTab("bd1", dirty=True))
    docmodes.close_tab(ctx, state, "bd1", lambda tab: None)
    ctx.confirms.asked[0].on_confirm()
    assert _listener_log == [("clay", "bd1")]


def test_a_bad_listener_does_not_break_closing_the_tab():
    """One misbehaving listener (a future ``familiar`` hook, say) must not
    stop the tab from actually closing -- it is cleanup, and cleanup that can
    be broken by an observer is not cleanup you can rely on."""

    def boom(mode: str, uid: str) -> None:
        raise RuntimeError("boom")

    docmodes.TAB_CLOSED.append(boom)
    try:
        ctx = FakeCtx()
        state = ClayState(_ListenerTab("bd1"))
        docmodes.close_tab(ctx, state, "bd1", lambda tab: None)
        assert state.get("bd1") is None
    finally:
        docmodes.TAB_CLOSED.remove(boom)


def _modules_calling_docmodes_close_tab() -> set[str]:
    """Every ``studio/`` module with a top-level call to ``docmodes.close_tab``
    (the ``docmodes.close_tab(...)`` shape, not a same-named local wrapper
    such as ``clay_mode.close_tab`` itself) -- found by walking the AST rather
    than hand-listed, so a future mode that grows a document tab enrols
    itself here the same way it enrols in ``_pure_packages``.

    Returned as the full dotted path *relative to* ``warlock.studio``
    (``"modes.mason.mode"``, but ``"modes.clay.mode"`` for Clay since P5 folded it
    into a mode package) rather than the bare file stem: a bare stem worked
    while every caller was a flat ``studio/<mode>_mode.py``, but Clay's is now
    ``studio/modes/clay/mode.py`` -- stem ``"mode"`` -- which collided with
    nothing usable and made the caller drop out of every check below it.

    2026-09-17 (dev/RESTRUCTURE.md P3 sweep-coverage pass): stays scoped to
    ``studio/`` on purpose -- ``docmodes.close_tab`` is L4 shell API, and the
    ``*_mode.py`` callers it exists to find are the mode-UI half that P3 left
    in place (only the engines moved to ``kernels/``). The caller's own
    ``expected_at_least`` floor already guards the failure this pass is
    about: a broken root would return an empty set and fail that assertion,
    not pass it.
    """
    import ast

    studio_dir = Path(__file__).resolve().parents[2] / "src" / "warlock" / "studio"
    files = sorted(studio_dir.rglob("*.py"))
    assert len(files) > 200, (
        f"only {len(files)} files under {studio_dir} -- did the sweep root break?"
    )
    modules: set[str] = set()
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "close_tab"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "docmodes"
            ):
                rel = path.relative_to(studio_dir).with_suffix("")
                modules.add(".".join(rel.parts))
                break
    return modules


def test_every_tabbed_state_class_name_maps_to_a_real_mode_key():
    """``_mode_for`` derives a mode key from a state class's own name
    (``ClayState`` -> ``"clay"``, and so on) -- every ``*State`` class behind
    a module that actually calls ``docmodes.close_tab`` must derive a key
    that :mod:`.modes` recognises, or a close would silently tell
    ``TAB_CLOSED`` listeners about a mode that does not exist.

    The state classes are looked up by name in their own module (rather than
    a hand list) once the *modules* are found by walking the AST -- only the
    convention "the caller of close_tab lives beside its own ``*State``
    class" is assumed, and that convention already has to hold for
    ``close_tab``'s own ``release`` callback to reach the right document.
    """
    from warlock.studio import modes

    caller_modules = _modules_calling_docmodes_close_tab()
    # Sanity floor: at least the six modes known to have document tabs today
    # (Clay, Mason, Plotter, Packwright, Sirens, Inker) must have been found --
    # a walk that silently found nothing would pass the loop below for free.
    # Clay's and Inker's are the dotted mode-package path (P5); the rest are
    # still the flat `studio/<mode>_mode.py` shape.
    expected_at_least = {
        "modes.clay.mode",
        "modes.mason.mode",
        "modes.plotter.mode",
        "modes.packwright.mode",
        "modes.sirens.mode",
        "modes.inker.mode",
    }
    assert expected_at_least <= caller_modules

    checked = 0
    for module_name in caller_modules:
        module = importlib.import_module(f"warlock.studio.{module_name}")
        # The mode key a caller's own *State class should derive to: the mode
        # package's own name for a nested caller (`modes.clay.mode` -> "clay"),
        # or the first underscore-joined word for the flat `<mode>_mode.py`
        # shape every other caller still has (`mason_mode` -> "mason").
        prefix = (
            module_name.split(".")[1]
            if module_name.startswith("modes.")
            else module_name.split("_")[0]
        )
        state_name = f"{prefix.capitalize()}State"
        state_cls = getattr(module, state_name, None)
        if state_cls is None:
            continue
        key = docmodes._mode_for(state_cls.__new__(state_cls))
        assert key in modes.KEYS, (
            f"{module_name}.{state_name} derives mode key {key!r}, not one of {modes.KEYS}"
        )
        checked += 1
    assert checked > 0, "no *State class was found beside a docmodes.close_tab caller"


def test_clay_titles_a_tab_by_stem_on_purpose():
    """Deliberately *not* shared: a Clay tab is named for the document rather
    than for the file it came from."""
    from warlock.studio.modes.clay import state as clay_state

    assert clay_state.title_for is not docmodes.title_for
    assert clay_state.title_for(Path("D:/x/hero.wblk")) == "hero"


# -- DocTabs: one tab list for every document mode (restructure P7) ----------


def test_doc_tabs_hooks_fire_on_arrival_and_close_with_the_right_arguments():
    from dataclasses import dataclass, field

    @dataclass
    class _Tabs(docmodes.DocTabs):
        log: list = field(default_factory=list)

        def _switched(self, previous):
            self.log.append(("switched", previous, self.active_uid))

        def _closed(self, was_active):
            self.log.append(("closed", was_active, self.active_uid))

    tabs = _Tabs()
    a = SimpleNamespace(uid="a", dirty=False, path=None)
    b = SimpleNamespace(uid="b", dirty=True, path=None)
    tabs.add(a)
    tabs.add(b)
    tabs.activate("b")  # already in front: no arrival
    tabs.activate("a")
    tabs.close("b")  # a background tab: not arriving anywhere
    tabs.close("a")
    assert tabs.log == [
        ("switched", "", "a"),
        ("switched", "a", "b"),
        ("switched", "b", "a"),
        ("closed", False, "a"),
        ("closed", True, ""),
    ]


def test_closing_the_front_tab_lands_on_its_neighbour_not_the_first():
    tabs = docmodes.DocTabs()
    for uid in "abc":
        tabs.add(SimpleNamespace(uid=uid, dirty=False, path=None))
    tabs.activate("b")
    tabs.close("b")
    assert tabs.active_uid == "c"
    tabs.cycle(1)
    assert tabs.active_uid == "a"


def test_every_document_mode_inherits_the_one_tab_list():
    """Six modes carried the list methods byte for byte and drifted in what
    each did on arrival. Each state is a ``DocTabs`` now; a mode that shadows
    a list method rather than a hook fails here by name. Sirens' ``activate``
    is the one recorded override: it stops the other song first (S6).
    Poser journals a pose viewer rather than a tab list, so it is not here."""
    from warlock.studio.mode_manifest import DOC_MODES, module_of

    allowed = {("sirens", "activate")}
    names = ("active", "any_dirty", "add", "get", "close", "activate", "cycle", "find_path")
    seen = set()
    tabbed = [entry for entry in DOC_MODES if entry.key in docmodes.DOC_MODES]
    for entry in tabbed:
        package = module_of(entry).__name__.rsplit(".", 1)[0]
        state_module = importlib.import_module(f"{package}.state")
        classes = [
            c
            for c in vars(state_module).values()
            if isinstance(c, type) and c.__module__ == state_module.__name__
            and c.__name__.endswith("State")
        ]
        assert len(classes) == 1, (entry.key, classes)
        state_cls = classes[0]
        assert issubclass(state_cls, docmodes.DocTabs), entry.key
        for name in names:
            if (entry.key, name) in allowed:
                continue
            assert name not in vars(state_cls), f"{entry.key} overrides DocTabs.{name}"
        seen.add(entry.key)
    assert seen == set(docmodes.DOC_MODES)
