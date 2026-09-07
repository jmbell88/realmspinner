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
