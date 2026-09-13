"""Dataset generator and verifier for the Clay-assistant training programme.

Phase 1 scaffold only -- see the approved plan
(``we-are-going-to-cosmic-dream.md``, sections "Phase 1", "Verifier design"
and "Phase 0 result") for the full programme this package is one piece of.
No ``src/`` integration lives here; this package only turns hand-authored
``drafts/*.jsonl`` rows into a verified ``dataset/`` through the real
``agent_clay.call`` door, exactly the way a live agent would be driven.
"""

from __future__ import annotations
