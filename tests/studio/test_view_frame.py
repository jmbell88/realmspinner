"""``_view_frame`` is a leaf, and code motion must not have changed a surface.

Two claims, and they pull in opposite directions on purpose.

The first is that the extraction took nothing away: ``ClayView`` was one class
with one surface before ``_view_frame`` existed, and every name that moved out
of ``clay_view`` and ``_view_drag`` has to still resolve on it. A move that
quietly dropped a method would not fail Clay's own suite for as long as no test
happened to call it -- ``_mods`` is reached through three call sites and
``_frame_unchanged`` through one, so a typo in a base-class list is exactly the
kind of thing that shows up later as a viewport that stops honouring Shift.

The second is that the leaf stays a leaf. The whole reason this module exists
is that a second viewport can inherit it, so an import of anything Clay-shaped
would make it Clay's again with nothing to say so -- the module would still
work, and the next viewport would simply drag Clay's document model in behind
it. Pinning the *exact* outward set (rather than denying a list of bad names)
is what makes widening it a visible edit to this file.
"""

from __future__ import annotations

import ast
from pathlib import Path

from realmspinner.studio import _view_frame
from realmspinner.studio.modes.clay.ui.view import ClayView

#: Every import ``_view_frame`` is allowed to make, as ``(module, name)`` --
#: ``name`` is ``None`` for a plain ``import x``. Both package-relative entries
#: are deliberately *inside* a method: ``imgui_backend`` because importing the
#: GL backend at module scope would make this leaf unimportable headless, and
#: ``pygame`` because ``_mods`` is called on a frame thread that may have no
#: display at all.
OUTWARD_IMPORTS = {
    ("__future__", "annotations"),
    ("typing", "Any"),
    (".", "imgui_backend"),
    ("pygame", None),
}


def _imports(module) -> set[tuple[str, str | None]]:
    tree = ast.parse(Path(module.__file__).read_text("utf-8"))
    found: set[tuple[str, str | None]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update((alias.name, None) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            where = "." * (node.level or 0) + (node.module or "")
            found.update((where, alias.name) for alias in node.names)
    return found


def test_clay_view_still_exposes_every_name_that_moved_into_the_leaf():
    """Derived from ``FrameOps`` itself, so a seventh method enrols unasked."""
    moved = sorted(n for n in vars(_view_frame.FrameOps) if not n.startswith("__"))
    missing = [name for name in moved if not hasattr(ClayView, name)]
    assert not missing, f"{missing} moved into FrameOps and no longer resolve on ClayView"
    assert moved, "the scan found no methods on FrameOps at all"


def test_the_moved_methods_are_the_leafs_own_and_not_a_second_copy():
    """The point of the move was one implementation, not two that agree today."""
    for name in ("_resize", "_forget", "_frame_unchanged", "_local", "_mods", "_rmb_release"):
        assert getattr(ClayView, name) is getattr(_view_frame.FrameOps, name), (
            f"ClayView.{name} is not FrameOps' own function -- a copy was left behind"
        )


def test_the_leaf_reaches_nothing_that_would_make_it_clays_again():
    found = _imports(_view_frame)
    assert found == OUTWARD_IMPORTS, (
        "_view_frame's imports changed. This module is the part of a viewport "
        "that is true of any document viewport; anything it needs from a mode "
        "belongs in that mode's own _view_* mixin, where it costs no second "
        "viewport a dependency it cannot use."
    )


def test_no_mode_module_is_named_anywhere_in_the_leaf():
    """The blunt half of the check above, and it catches what imports miss:
    a deferred ``from .clay_state import ...`` inside a method is still Clay."""
    source = Path(_view_frame.__file__).read_text("utf-8")
    code = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )
    body = code.split('"""', 2)[-1]  # past the module docstring
    # "modes.clay"/"modes/clay" joined the list when P5 folded Clay into a mode
    # package: after that move, a stray ``from .modes.clay import state`` names
    # Clay exactly as much as the old flat ``clay_state`` import did, and the
    # bare-name bans above would not catch it -- this dotted/path form is now
    # the only way a within-package reference into Clay's home can be spelled.
    for mode in (
        "clay_state",
        "clay_view",
        "clay_mode",
        "modes.clay",
        "modes/clay",
        "inker",
        "plotter",
        "mason_",
    ):
        assert mode not in body, f"_view_frame names {mode}, which makes it that mode's"


def test_the_import_scan_actually_matches_this_module():
    """The guard on the guard: an ``ast`` walk that stopped finding imports
    would make the pin above pass by comparing two empty sets."""
    found = _imports(_view_frame)
    assert (".", "imgui_backend") in found, "the import scan no longer finds a known import"
    assert len(found) >= 3, f"the import scan found only {sorted(found)}"


def test_every_3d_host_resolves_the_frame_plumbing_to_the_leaf():
    """The embedded viewer (Create, Poser, the inspector) carried its own
    ``_forget``, ``_local`` and ``_resize``, byte for byte but for the
    viewport argument, plus an inlined ``_frame_unchanged`` and an
    ``_alt_held`` that was ``_mods``' third answer. Restructure P7 folded
    them in; a host that grows a local copy again fails here by name."""
    from realmspinner.studio.modes.mason.ui.view import MasonView
    from realmspinner.studio.viewer_embed import Viewer

    names = sorted(n for n in vars(_view_frame.FrameOps) if not n.startswith("__"))
    assert names, "the scan found no methods on FrameOps at all"
    for host in (ClayView, MasonView, Viewer):
        for name in names:
            assert getattr(host, name) is getattr(_view_frame.FrameOps, name), (
                f"{host.__name__}.{name} is not FrameOps' own function"
            )
