"""Press controls the way a person does: through imgui's own input queue.

Extracted from the pattern ``test_clay_tools_panels`` wrote for the flyout's Box
entry -- ``io.add_mouse_pos_event`` / ``io.add_mouse_button_event`` with targets
read off ``probe.FRAME_CONTROLS`` -- and kept here so a second test does not
write it a fifth time (``_ui_context``'s docstring tells that story for the
context itself).

**Why real input and not a call to the handler.** A control wired to nothing, a
control clipped past its pane, a control disabled with no reason, a click that a
text field still holding focus swallows: each passes every test that calls the
setter directly. A press that has to find its target in the frame's census,
hover it for a frame and release it on the next is the one thing that fails on
all of them.

**It counts.** :attr:`UiDriver.presses` (mouse buttons pressed) and
:attr:`UiDriver.keys` (characters and keys typed) are the *gestures* a task cost,
which is what ``test_first_try_path``'s ratchet measures. Counting here, at the
one place input enters, means a test cannot spend a gesture without it showing up
in the total.

**It does not fake.** The target is found by label in the census of the frame
that was just drawn; when it is not there, :meth:`UiDriver.find` raises with the
nearest labels rather than falling back to a direct call.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any

from realmspinner.studio import probe

#: Where the pointer rests when nothing is being pointed at: off every window.
OFFSCREEN = (-100.0, -100.0)


class UiDriver:
    """Drives one imgui context, a frame at a time, with real input events.

    *frame* draws one complete frame (``new_frame`` through ``end_frame`` or
    ``render``) -- the caller owns that because a pane test wants a host window
    and a GL test wants a renderer. The driver brackets it with the census.
    """

    def __init__(self, imgui: Any, frame: Callable[[], None]) -> None:
        self.imgui = imgui
        self._frame = frame
        self.mouse: tuple[float, float] = OFFSCREEN
        #: Mouse buttons pressed, and characters / keys typed.
        self.presses = 0
        self.keys = 0
        #: Every control pressed, in order: ``(label, enabled)``.
        self.log: list[tuple[str, bool]] = []
        #: The census of the frame most recently drawn.
        self.controls: list[probe.Control] = []

    # -- frames ---------------------------------------------------------------------

    @property
    def gestures(self) -> int:
        """Presses plus keys: what the task cost the person doing it."""
        return self.presses + self.keys

    @property
    def disabled_presses(self) -> list[str]:
        """The labels of every disabled control that was pressed."""
        return [label for label, enabled in self.log if not enabled]

    def frame(self, pos: tuple[float, float] | None = None, down: bool | None = None) -> list:
        """Draw one frame, after queueing a pointer move and/or a button change."""
        io = self.imgui.get_io()
        if pos is not None:
            self.mouse = (float(pos[0]), float(pos[1]))
        io.add_mouse_pos_event(*self.mouse)
        if down is not None:
            io.add_mouse_button_event(0, bool(down))
        probe.begin_frame()
        self._frame()
        self.controls = list(probe.FRAME_CONTROLS)
        return self.controls

    def settle(self, frames: int = 2) -> list:
        """Frames with no input, for a layout or a pushed page to catch up."""
        controls = self.controls
        for _ in range(frames):
            controls = self.frame()
        return controls

    # -- finding ----------------------------------------------------------------------

    def find(
        self,
        *,
        label: str | None = None,
        text: str | None = None,
        contains: str | None = None,
        kind: str | None = None,
        window: str | None = None,
    ) -> probe.Control:
        """The one visible control in the last frame that matches every filter.

        Raises ``LookupError`` naming the closest labels when there is none, and
        when there are several (a test that does not say which is guessing).
        """
        hits = [
            c
            for c in self.controls
            if c.visible
            and (label is None or c.label == label)
            and (text is None or c.text == text)
            and (contains is None or contains in c.label)
            and (kind is None or c.kind == kind)
            and (window is None or c.window == window)
        ]
        wanted = label or text or contains or kind or window
        if not hits:
            near = sorted({c.label for c in self.controls if c.visible})[:40]
            raise LookupError(f"no visible control matches {wanted!r}; drawn: {near}")
        if len(hits) > 1:
            raise LookupError(f"{len(hits)} controls match {wanted!r}: {[c.label for c in hits]}")
        return hits[0]

    # -- input ------------------------------------------------------------------------

    def click(self, target: probe.Control) -> None:
        """Hover, press and release *target*, over three frames.

        imgui's default button fires on release-while-hovered, and hover is only
        current once a frame has already put the pointer over the item -- so the
        down/up pair needs a frame of hover in front of it, the same three-frame
        shape every pane test presses a real button with.

        A disabled target is pressed anyway (a person would) and *recorded*, so
        the test that cares asserts :attr:`disabled_presses` is empty rather than
        the driver refusing and hiding the dead end.
        """
        cx, cy = target.centre
        self.frame((cx, cy), down=False)
        self.frame((cx, cy), down=True)
        self.frame((cx, cy), down=False)
        self.presses += 1
        self.log.append((target.label, bool(target.enabled)))

    def type_text(self, text: str) -> None:
        """Type *text* into whatever has keyboard focus, a frame per character."""
        io = self.imgui.get_io()
        for char in text:
            io.add_input_character(ord(char))
            self.frame()
            self.keys += 1

    def press_key(self, key: Any) -> None:
        """Press and release one imgui key (``imgui.Key.enter``)."""
        io = self.imgui.get_io()
        io.add_key_event(key, True)
        self.frame()
        io.add_key_event(key, False)
        self.frame()
        self.keys += 1


@contextlib.contextmanager
def gl_imgui(gl: Any):
    """An imgui context with a real renderer over the standalone GL context.

    ``tests/studio/test_studio_smoke.py``'s ``imgui_ctx`` fixture, as a context
    manager so a test file that is not the smoke suite can have one: the
    renderer is needed even though nothing is presented (imgui 1.92 hands its
    font atlas to the backend, and ``widgets.texture_ref`` must register the
    viewport's texture with it), and **one imgui context at a time** is the rule
    -- two over one GL context is a Windows access violation, so this one is torn
    down on the way out rather than left for the next file in the worker.
    """
    from imgui_bundle import imgui

    from realmspinner.studio import imgui_backend, theme

    previous_screen = type(gl).__dict__.get("screen")
    fbo = gl.simple_framebuffer((1600, 950))
    fbo.use()
    type(gl).screen = property(lambda _self: fbo)

    imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (1600, 950)
    io.delta_time = 1 / 60
    io.fonts.add_font_default()
    theme.apply(imgui)
    renderer = imgui_backend.ImguiRenderer(gl)
    try:
        yield imgui, renderer
    finally:
        renderer.shutdown()
        imgui.destroy_context()
        if previous_screen is not None:
            type(gl).screen = previous_screen


@contextlib.contextmanager
def app_context(gl: Any, svc: Any, data_dir: Any):
    """A real ``Ctx`` over the throwaway service: what ``App`` draws against.

    The smoke suite's ``app_ctx`` fixture, as a context manager (see
    :func:`gl_imgui`).
    """
    from realmspinner.service import system as svc_system
    from realmspinner.studio import textures
    from realmspinner.studio.app_ctx import Ctx
    from realmspinner.studio.jobs_cache import JobsCache
    from realmspinner.studio.runtime import Runtime
    from realmspinner.studio.settings import Settings
    from realmspinner.studio.state import AppState
    from realmspinner.studio.tasks import TaskRunner
    from realmspinner.studio.viewer_embed import Viewer

    runtime = Runtime(svc.config)
    runtime.store = svc.store
    runtime.tasks = TaskRunner(workers=1)
    viewer = Viewer(gl)
    ctx = Ctx(
        svc=svc,
        runtime=runtime,
        state=AppState(),
        cache=JobsCache(svc),
        tasks=runtime.tasks,
        settings=Settings.load(data_dir),
        viewer=viewer,
        textures=textures.ThumbnailCache(gl),
    )
    ctx.guidance = svc_system.guidance_catalog(svc)
    ctx.base_models = [("turbo", "SDXL-Turbo")]
    ctx.style_loras = [("", "none")]
    ctx.rig_templates = [{"key": "biped", "label": "Biped"}]
    ctx.rig_default = "biped"
    try:
        yield ctx
    finally:
        viewer.release()
        ctx.textures.release()
        runtime.tasks.shutdown(wait=False)
