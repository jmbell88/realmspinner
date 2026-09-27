"""Closes the 2026-09-26 audit's docs-F2-01: four manual chapters said model-weight
download (via ``fetch_worker``) was Realmspinner's *only* network use -- "Realmspinner
never touches the network", "That download is the only network use there is" -- when
``pack_worker`` (Settings -> Packs) and ``update_worker`` (Settings -> Updates) are two
more separate, user-initiated subprocesses that also go online. This test reads the
prose directly rather than the CLAUDE.md summary, so it fails on the original chapters
(none of which name the update check at all) and passes once each names all three.
"""

from pathlib import Path

MANUAL = Path(__file__).resolve().parents[2] / "docs" / "manual"

# The five segments the audit's finding named for this claim; four of them are ours to
# fix (chapter 42 -- Settings -- belongs to another fixer in this pass).
CHAPTERS = [
    "01-before-you-begin.md",
    "20-overview.md",
    "40-installation.md",
    "44-architecture.md",
]


def _text(name: str) -> str:
    return (MANUAL / name).read_text(encoding="utf-8")


def test_manual_states_all_three_network_workers():
    for name in CHAPTERS:
        text = _text(name)
        assert "Settings → Packs" in text or "pack worker" in text, (
            f"{name}: does not name the dependency-pack install as a network exception"
        )
        assert "Settings → Updates" in text or "update worker" in text, (
            f"{name}: does not name the release/update check as a network exception"
        )


def test_manual_does_not_call_model_fetch_the_only_network_use():
    """The literal overclaims the audit quoted, so a future edit cannot reintroduce one
    without tripping this test even if the two-worker check above is satisfied some
    other way. Deliberately the exact quoted phrases rather than a loose substring like
    "never touches the network" -- chapter 01's line 6 ("after the initial downloads it
    never touches the network again") is a *different*, still-accurate claim the audit's
    finding did not cite (it names only line 102 there), and widening this test to catch
    it would be widening the finding, which the brief for this pass forbids.
    """
    overclaims = [
        "Realmspinner never touches the network",
        "the only network use there is",
    ]
    for name in CHAPTERS:
        text = _text(name)
        for phrase in overclaims:
            assert phrase not in text, f"{name}: still contains the overclaim {phrase!r}"
