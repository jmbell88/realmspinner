"""Frame plumbing shared by every document viewport, and owned by none of them.

``clay_view`` grew the ``_view_*.py`` mixins as code motion out of one class,
and each of those is still *Clay's*: the GPU cache keys on Clay's objects, the
overlay draws Clay's elements, and the whole of ``_view_drag`` is Clay's mouse
map. This one is the leaf underneath all of them -- the part that is true of
any viewport drawing a document into an imgui image, and that a second such
viewport would otherwise copy.

It exists because copying it is exactly what goes wrong. Two of the four things
here are invisible-failure rules rather than conveniences:

* **Release before forget.** ``Viewport.resize`` releases its texture and makes
  a new one, and ``studio/imgui_backend.py`` maps GL names to moderngl objects.
  Resizing without forgetting first leaves the backend holding a dead object
  under a name the driver is free to reissue, which is how an unrelated image
  starts rendering as this one. A viewport that reimplemented its own resize
  and missed this would look correct until something else claimed the name.
* **Modifiers are read at the press, from ``pygame.key.get_mods()``.** Not from
  the event: a ``MOUSEBUTTONDOWN`` carries no modifier state, and tracking
  KEYDOWN/KEYUP to shadow it is a second copy of something the platform already
  knows, and gets wrong the first time the window loses focus with Shift held.

The other two are the right-button rule (a menu on a release *within four
pixels* of the press, so a right-drag does nothing and grabbing the wrong
button mid-orbit loses nothing) and the redraw bookkeeping, which decides that
a frame can be skipped entirely and the last texture returned as-is.

**Nothing here knows what a document is.** No Clay import, no element modes, no
selection: a mixin that needed any of those would belong in one of the
``_view_*`` siblings instead. What it does assume is the field set every
viewport already has -- ``viewport``, ``camera``, ``_rect``, ``_last_mouse``,
``_rmb_at``, ``menu_request``, ``_render_dirty`` and ``_last_render_key`` --
which is why this is a mixin rather than a helper object: the state is the
host's, and threading it through arguments would be a second copy of it.
"""

from __future__ import annotations

from typing import Any

#: The three axis-view digits, and the view each snaps to. Blender's numpad
#: layout, which is what anyone arriving from a modelling tool already has in
#: their hands.
AXIS_VIEW_KEYS = {"1": "front", "3": "right", "7": "top"}

#: How far a right-button release may sit from its press and still open the
#: menu, in pixels. Four rather than zero: a right-click on a trackpad
#: routinely moves one or two, and a menu that refuses to open because the
#: finger shifted reads as the app ignoring the click.
RMB_MENU_SLOP = 4.0


class Composite:
    """The renderer's view of many cached objects at once.

    Not a ``GpuModel``: it owns nothing and releases nothing, and the two
    methods here are the entire surface ``Renderer._draw_model`` uses. Skinning
    is not part of it -- Clay has no skins, which is also why ``glbwrite``
    refuses one.
    """

    __slots__ = ("draws",)

    def __init__(self, draws: list[Any]) -> None:
        self.draws = draws

    def palette(self, node: Any) -> None:
        return None


def axis_view_key(camera: Any, name: str, shift: bool) -> bool:
    """One Ctrl+digit view key, on any camera. -> whether ``name`` was one.

    Shift is the opposite view, as Blender's numpad does it -- Ctrl+1 is the
    front and Ctrl+Shift+1 the back, so six views cost three keys; Ctrl+5
    toggles orthographic. **Shared rather than restated per viewport**, so two
    3-D viewports cannot come to disagree about which number is the front --
    and so that both require Ctrl. Poser's copy tested the bare digit, so a 1
    typed into nothing snapped its camera while Clay's did not.
    """
    if name in AXIS_VIEW_KEYS:
        wanted = AXIS_VIEW_KEYS[name]
        if shift:
            wanted = {"front": "back", "right": "left", "top": "bottom"}[wanted]
        camera.look_along(wanted)
        return True
    if name == "5":
        camera.orthographic = not camera.orthographic
        return True
    return False


class FrameOps:
    """The frame plumbing. See the module docstring."""

    # -- the imgui texture -------------------------------------------------

    def _resize(self: Any, width: int, height: int) -> None:
        """Resize, forgetting the outgoing texture first.

        ``Viewport.resize`` releases its texture and makes a new one, and the
        imgui backend maps GL names to moderngl objects: releasing without
        forgetting leaves it holding a dead object under a name the driver is
        free to reissue, which is how an unrelated image starts rendering as
        this one.
        """
        if (width, height) == self.viewport.size:
            return
        self._forget(self.viewport.texture)
        self.viewport.resize((width, height))

    def _forget(self: Any, texture: Any) -> None:
        if texture is None:
            return
        from . import imgui_backend

        renderer = imgui_backend.current()
        if renderer is not None:
            renderer.forget_texture(texture)

    # -- the redraw decision -----------------------------------------------

    def _frame_unchanged(self: Any, key: Any) -> bool:
        """Whether this frame can be skipped and the last texture reused (B13).

        Four questions, and all four have to answer: the caller's key covers
        everything about the document and the view settings that decides the
        picture, ``_render_dirty`` covers the input that does not show up in a
        key (hover, marquee, a live drag), the camera answers for itself
        because an eased move is still moving after the edit that started it,
        and a viewport with no texture yet has nothing to return.
        """
        return (
            not self._render_dirty
            and key == self._last_render_key
            and self.camera.settled()
            and self.viewport.texture is not None
        )

    # -- input -------------------------------------------------------------

    def _local(self: Any, event: Any) -> tuple[float, float]:
        pos = getattr(event, "pos", None)
        if pos is None:
            return self._last_mouse
        return (pos[0] - self._rect[0], pos[1] - self._rect[1])

    def _mods(self: Any) -> tuple[bool, bool, bool]:
        """``(shift, ctrl, alt)`` at this instant. See the module docstring."""
        try:
            import pygame

            mods = pygame.key.get_mods()
            return (
                bool(mods & pygame.KMOD_SHIFT),
                bool(mods & pygame.KMOD_CTRL),
                bool(mods & pygame.KMOD_ALT),
            )
        except Exception:  # pragma: no cover - headless pygame without a display
            return (False, False, False)

    def _rmb_release(self: Any, local: tuple[float, float]) -> bool:
        """The context menu, on a release that did not travel.

        Four pixels rather than zero: a right-click on a trackpad routinely
        moves one or two, and a menu that refuses to open because the finger
        shifted reads as the app ignoring the click.
        """
        at, self._rmb_at = self._rmb_at, None
        if at is None:
            return False
        if abs(local[0] - at[0]) < RMB_MENU_SLOP and abs(local[1] - at[1]) < RMB_MENU_SLOP:
            self.menu_request = local
        return True
