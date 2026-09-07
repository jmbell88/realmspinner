"""THIRD-PARTY-NOTICES.md's licence-surfacing claim, checked against the code.

The 2026-09-06 audit, finding docs-03: the notices file's closing paragraph
said ``warlock.models`` carries a ``license`` field "on every entry" and that
the model picker and download confirmation show it for all of them. Only
three of the ten registry dataclasses (``BaseModel``, ``MusicModel``,
``SeparationModel``) declare the field -- ``EngineModel``, ``MattingModel``,
``MetricModel``, ``PoseModel``, ``StyleLora``, ``IPAdapter`` and
``ControlNet`` do not -- so ``service/downloads.py``'s
``getattr(entry.spec, "license", "") or ""`` silently returns "" for those
seven and no licence line is shown for them, including TRELLIS.2-4B and
BiRefNet, both of which the table above the paragraph lists as MIT by hand.

This test does not hard-code which classes carry the field: it walks
``warlock.models`` itself (every module-level dict built by that module's own
``_table()`` is one registry, and the type of its values is one "registry
dataclass"), and derives the license-bearing subset from ``dataclasses.fields``.
It then reads the same three facts back out of the notices paragraph and
requires they match exactly. If a future registry class gains a ``license``
field, ``_license_bearing_classes()`` grows and this test starts failing
until the paragraph is widened to say so -- that is the point of deriving
instead of hard-coding.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

from warlock import models as models_mod

ROOT = Path(__file__).resolve().parents[1]
NOTICES = ROOT / "THIRD-PARTY-NOTICES.md"


def _registry_dataclasses() -> dict[str, type]:
    """One dataclass type per ``_table()``-built registry in ``warlock.models``.

    A "registry dataclass" is defined structurally, not by a hard-coded list:
    any module-level attribute that is a non-empty ``dict`` whose values are
    all instances of the same dataclass defined in ``warlock.models`` itself
    (excluding ``Fetch``, which is a field *of* those dataclasses, not a
    registry entry type on its own -- it has no ``_table()`` dict of its own).
    """
    found: dict[str, type] = {}
    for value in vars(models_mod).values():
        if not isinstance(value, dict) or not value:
            continue
        sample = next(iter(value.values()))
        cls = type(sample)
        if (
            dataclasses.is_dataclass(cls)
            and cls.__module__ == models_mod.__name__
            and all(type(v) is cls for v in value.values())
        ):
            found[cls.__name__] = cls
    return found


def _license_bearing_classes() -> set[str]:
    registries = _registry_dataclasses()
    return {
        name
        for name, cls in registries.items()
        if any(f.name == "license" for f in dataclasses.fields(cls))
    }


def test_registry_walk_finds_the_ten_classes_the_finding_named():
    # Pins the shape docs-03 was filed against, so a silent change in
    # models.py's registry count is visible here rather than only as a
    # confusing failure in the paragraph-matching test below.
    registries = _registry_dataclasses()
    assert set(registries) == {
        "BaseModel",
        "StyleLora",
        "IPAdapter",
        "ControlNet",
        "EngineModel",
        "MetricModel",
        "PoseModel",
        "MusicModel",
        "SeparationModel",
        "MattingModel",
    }


def test_registry_walk_finds_the_three_license_bearing_classes_the_finding_named():
    assert _license_bearing_classes() == {"BaseModel", "MusicModel", "SeparationModel"}


def _paragraph() -> str:
    text = NOTICES.read_text(encoding="utf-8")
    marker = "The application surfaces this per model"
    idx = text.index(marker)
    return text[idx:]


def _backtick_class_names(segment: str) -> set[str]:
    return set(re.findall(r"`(\w+)`", segment))


def _normalize_ws(text: str) -> str:
    # Markdown hand-wraps prose at ~80 columns, so a phrase spanning a line
    # break (e.g. "carry no\n`license` field") is still one phrase to a
    # reader; collapse whitespace before searching so the test isn't coupled
    # to exactly where the paragraph happens to wrap.
    return re.sub(r"\s+", " ", text)


def _between(text: str, start_marker: str, end_marker: str) -> str:
    text = _normalize_ws(text)
    start_marker = _normalize_ws(start_marker)
    end_marker = _normalize_ws(end_marker)
    start = text.index(start_marker) + len(start_marker)
    end = text.index(end_marker, start)
    return text[start:end]


def test_notices_paragraph_names_exactly_the_ten_registry_classes():
    """The paragraph must enumerate all ten classes it is talking about.

    Catches the "on every entry" phrasing docs-03 flagged: a paragraph that
    never names which classes it means can claim anything about "every
    entry" without a test being able to check it against the code. Requiring
    an explicit, exhaustive class list is what makes the next two tests
    possible.
    """
    paragraph = _paragraph()
    all_named = _backtick_class_names(paragraph)
    registries = set(_registry_dataclasses())
    # Every registry class the code has must be named somewhere in the
    # paragraph (as either license-bearing or not) -- nothing left silently
    # unaccounted for.
    assert registries <= all_named, (
        f"registry classes missing from the notices paragraph: "
        f"{registries - all_named}"
    )


def test_notices_paragraph_license_bearing_list_matches_the_code():
    """The classes named as carrying `license` must be exactly the real set.

    This is the direction docs-03 was actually filed over: the old sentence
    over-claimed (said "every entry" when 7 of 10 classes have no field).
    """
    paragraph = _paragraph()
    segment = _between(
        paragraph,
        "declare a `license` field",
        "and only for those",
    )
    claimed = _backtick_class_names(segment)
    assert claimed == _license_bearing_classes(), (
        f"notices paragraph claims license-bearing classes {claimed}, "
        f"but warlock.models says {_license_bearing_classes()}"
    )


def test_notices_paragraph_no_license_list_matches_the_code():
    """The classes named as carrying no `license` field must be exactly right.

    The other direction: if the paragraph under-claims (omits a class that
    truly has no field, or wrongly lists one that does), this catches it --
    the honesty is required both ways, per the finding's fix instruction.
    """
    paragraph = _paragraph()
    segment = _between(
        paragraph,
        "download confirmation.",
        "carry no `license` field",
    )
    claimed = _backtick_class_names(segment)
    registries = set(_registry_dataclasses())
    expected = registries - _license_bearing_classes()
    assert claimed == expected, (
        f"notices paragraph claims no-license classes {claimed}, "
        f"but the complement of the license-bearing set is {expected}"
    )


def test_notices_no_longer_claims_license_on_every_entry():
    """The specific over-claim docs-03 quoted must be gone.

    Guards against a future edit reverting to the old wording without
    breaking the (harder to notice at a glance) set-comparison tests above.
    """
    text = _normalize_ws(NOTICES.read_text(encoding="utf-8"))
    assert "license` field on every entry" not in text
