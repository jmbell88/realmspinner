"""The human-readable shape of one Create press, shared by both stages' footers.

Lives in ``engine/`` rather than beside the footer that draws it because the
Mesh stage's ``engine/mesh.py`` builds one too, and an engine module may not
import ``ui/`` (``tests/modes/create/test_create_engine_imports.py``).
``ui/workspace.py`` re-exports :class:`Plan`, so the old name still resolves.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Plan:
    """The human-readable work implied by one Create press."""

    candidates: int
    generations: int
    duration: str
    stages: str
    recipe: str
    #: What one unit of ``generations`` is called. Empty for a stage whose
    #: candidates *are* its units (a mesh attempt is a candidate), so the count
    #: line does not say the same number twice.
    unit: str = "image generation"

    @property
    def count_line(self) -> str:
        # The 2026-09-26 audit, finding create-workspace-07: ``generations``
        # was never pluralised, so a press generating four images read "4
        # image generation" -- the same singular/plural agreement
        # ``candidates``/``candidate`` just above already gets right.
        candidate_noun = "candidate" if self.candidates == 1 else "candidates"
        line = f"{self.candidates} {candidate_noun}"
        if not self.unit:
            return line
        unit = self.unit if self.generations == 1 else f"{self.unit}s"
        return f"{line} · {self.generations} {unit}"
