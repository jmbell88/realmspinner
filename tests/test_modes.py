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
    "home", "library", "create", "inker", "clay", "poser", "troupe", "plotter",
    "packwright", "muse", "sirens", "review", "settings",
)


def test_every_mode_carries_a_purpose_and_the_rail_shows_it():
    """Every entry of ``MODES`` is a (key, label, icon, purpose) 4-tuple, and
    ``rail._item`` uses ``purpose`` in the tooltip and as a second line.

    Before this test, ``MODES`` was a 3-tuple and the purpose sentence lived
    only in a separate ``PURPOSE`` dict that ``rail.draw`` consulted, so a
    fresh mode could be added to ``MODES`` without ever getting a purpose --
    nothing tied the two together. This asserts the sentence is part of the
    tuple itself, that it exists (non-empty, plain, no trailing period-less
    fragment) for all thirteen modes, and that the rail module actually wires
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
    assert (pipeline_count, workspace_count) == (3, 8), (
        f"sanity: expected RAIL_GROUPS to hold 3 and 8 modes, found "
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
    assert "these eight are workspaces" in comment, (
        f"modes.py's RAIL_GROUP_LABELS comment should say 'these eight are "
        f"workspaces' to match RAIL_GROUPS[1]'s {workspace_count} entries"
    )
