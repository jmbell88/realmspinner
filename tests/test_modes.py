"""Comments in ``modes.py`` that state a count, checked against the data.

``RAIL_GROUPS`` and ``RAIL_GROUP_LABELS`` are the source of truth for how the
rail is grouped; a comment nearby that restates their shape in words is prose,
not data, and drifts the moment a mode moves groups without the comment being
touched.
"""

from __future__ import annotations

from pathlib import Path

from warlock.studio import modes

MODES_PATH = Path(modes.__file__)

ALL_MODE_KEYS = (
    "home", "library", "create", "inker", "clay", "mason", "poser", "troupe",
    "plotter", "packwright", "muse", "sirens", "review", "settings",
)


def test_every_mode_carries_a_purpose_and_the_rail_shows_it():
    """Every entry of ``MODES`` is a (key, label, icon, purpose) 4-tuple, and
    ``rail._item`` uses ``purpose`` in the tooltip and as a second line.

    Before this test, ``MODES`` was a 3-tuple and the purpose sentence lived
    only in a separate ``PURPOSE`` dict that ``rail.draw`` consulted, so a
    fresh mode could be added to ``MODES`` without ever getting a purpose --
    nothing tied the two together. This asserts the sentence is part of the
    tuple itself, that it exists (non-empty, plain, no trailing period-less
    fragment) for all fourteen modes, and that the rail module actually wires
    it into both the tooltip and the labelled-rail drawing path.
    """
    assert {key for key, *_rest in modes.MODES} == set(ALL_MODE_KEYS)

    for entry in modes.MODES:
        assert len(entry) == 4, f"{entry!r} is not a (key, label, icon, purpose) 4-tuple"
        key, label, icon, purpose = entry
        assert isinstance(purpose, str) and purpose.strip(), (
            f"mode {key!r} ({label!r}) has no purpose string"
        )
        # Plain and in the app's voice: no second person ("you"/"your"), and
        # not a fragment ending mid-sentence.
        lowered = purpose.lower()
        assert " you " not in f" {lowered} " and "your" not in lowered, (
            f"mode {key!r}'s purpose reads second-person: {purpose!r}"
        )
        assert purpose[-1] in ".!", f"mode {key!r}'s purpose has no closing punctuation"

    # ``PURPOSE`` is derived from the tuple, not a second hand-kept table.
    assert {key: purpose for key, _label, _icon, purpose in modes.MODES} == modes.PURPOSE

    # ``rail._item`` accepts and uses ``purpose``: the icon-mode tooltip joins
    # label and purpose with "·", and the labels-mode drawing path measures a
    # second, muted line for it rather than ignoring the argument.
    import inspect

    from warlock.studio import rail

    source = inspect.getsource(rail._item)
    assert "purpose" in inspect.signature(rail._item).parameters
    assert "·" in source, "rail._item's icon-mode tooltip should join label and purpose with ·"
    assert "purpose_size" in source, (
        "rail._item should measure a second line for the purpose in labels mode"
    )

    draw_source = inspect.getsource(rail.draw)
    assert "purpose=purpose" in draw_source, (
        "rail.draw must pass the mode's purpose through to _item"
    )


def _comment_before(marker: str) -> str:
    """The comment block ending at the line containing ``marker``."""
    lines = MODES_PATH.read_text(encoding="utf-8").splitlines()
    index = next(i for i, line in enumerate(lines) if marker in line)
    start = index
    while start > 0 and lines[start - 1].lstrip().startswith("#:"):
        start -= 1
    return "\n".join(lines[start : index + 1])


def test_rail_groups_comment_states_the_real_pipeline_and_workspace_counts():
    """The 2026-09-07 audit, finding tour-04: ``modes.py``'s comment above
    ``RAIL_GROUP_LABELS`` called the grouping's claim "these four are one
    pipeline; these seven are workspaces", but ``RAIL_GROUPS`` holds three
    pipeline modes (home, library, create) and eight workspace modes.
    """
    pipeline_count = len(modes.RAIL_GROUPS[0])
    workspace_count = len(modes.RAIL_GROUPS[1])
    assert (pipeline_count, workspace_count) == (3, 9), (
        f"sanity: expected RAIL_GROUPS to hold 3 and 9 modes, found "
        f"{pipeline_count} and {workspace_count} -- update the numbers below "
        "as well as this test"
    )

    comment = _comment_before('RAIL_GROUP_LABELS: tuple[str, ...] = ("Pipeline"')
    assert "these four are one pipeline" not in comment, (
        "modes.py's RAIL_GROUP_LABELS comment still says 'these four are one "
        f"pipeline', but RAIL_GROUPS[0] holds {pipeline_count}"
    )
    assert "these seven are workspaces" not in comment, (
        "modes.py's RAIL_GROUP_LABELS comment still says 'these seven are "
        f"workspaces', but RAIL_GROUPS[1] holds {workspace_count}"
    )
    assert "these three are one pipeline" in comment, (
        f"modes.py's RAIL_GROUP_LABELS comment should say 'these three are "
        f"one pipeline' to match RAIL_GROUPS[0]'s {pipeline_count} entries"
    )
    assert "these nine are workspaces" in comment, (
        f"modes.py's RAIL_GROUP_LABELS comment should say 'these nine are "
        f"workspaces' to match RAIL_GROUPS[1]'s {workspace_count} entries"
    )


# --- the dispatch arms -------------------------------------------------------
#
# Registering a mode in ``WORKSPACE_MODES`` without giving it an arm in
# ``_build_ui`` is silent: the dispatch ends in a bare ``else``, so the new mode
# draws whatever that else draws rather than failing. Measured on 2026-09-11,
# adding Mason: it was in ``KEYS``, ``WORK_MODES`` and ``WORKSPACE_MODES``, the
# rail drew its rung, ``test_the_three_categories_still_partition_the_modes``
# passed -- and clicking it drew *Inker's* workspace. Every membership test in
# the tree passed for the whole time that was true, because membership is not
# the claim that matters here; having an arm is.
#
# Two directions, per the tour's rule that a one-way check grows a hole from the
# other side, plus a guard proving the scan can still fail -- without it, a
# regex that stopped matching would make both directions pass by finding
# nothing at all.

#: The one mode ``_build_ui``'s bare ``else`` is allowed to draw.
_ELSE_ARM = "inker"


def _dispatch_arms() -> set[str]:
    """Every mode key ``_build_ui`` dispatches on by name."""
    import inspect
    import re

    from warlock.studio import main

    return set(re.findall(r'mode == "([a-z_]+)"', inspect.getsource(main.App._build_ui)))


def test_every_mode_that_fills_the_window_has_its_own_arm_in_the_dispatch():
    from warlock.studio import main

    owed = (set(main._SINGLE_PANE_MODES) | set(modes.WORKSPACE_MODES)) - {_ELSE_ARM}
    missing = sorted(owed - _dispatch_arms())
    assert not missing, (
        f"{missing} reach _build_ui's dispatch with no arm of their own, so "
        f"each one silently draws the bare else's workspace ({_ELSE_ARM}'s)"
    )


def test_the_dispatch_names_no_mode_that_does_not_exist():
    """The other direction: an arm left behind by a mode that was renamed or
    removed is dead code that reads as live wiring."""
    unknown = sorted(_dispatch_arms() - set(modes.KEYS))
    assert not unknown, f"_build_ui dispatches on {unknown}, which are not mode keys"


def test_the_arm_scan_actually_matches_the_dispatch():
    """The guard on the guard. Both checks above are built from one regex, and
    a regex that stopped matching -- a rename, a refactor to a dict lookup --
    would make them pass by finding nothing on both sides at once."""
    found = _dispatch_arms()
    assert "clay" in found, "the arm scan no longer matches _build_ui's dispatch"
    assert len(found) >= 10, f"the arm scan found only {sorted(found)}"
