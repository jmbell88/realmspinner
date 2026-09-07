"""The 2026-09-06 docs audit, findings docs-02 and docs-21.

Both are the same shape: a manual chapter and a piece of code disagree about a
plain fact, and the chapter is the one that is wrong. Both tests are driven
off the code rather than a second hard-coded copy of the list, so a future
change to ``config.SWITCHES`` or to the ``studio`` extra fails here instead of
drifting silently again.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from warlock import config

ROOT = Path(__file__).resolve().parents[2]


def _chapter(name: str) -> str:
    return (ROOT / "docs" / "manual" / name).read_text(encoding="utf-8")


def test_manual_config_lists_native_and_migrate_vars_as_reported_by_effective_config():
    """docs-02: config.SWITCHES exists precisely so effective() prints these four
    vars (marked from_env) after the SETTINGS table -- to `warlock doctor`, the
    Health/Advanced popup and its Copy-as-text. The chapter's "Seeing which of
    these are actually set" section told a reader to distrust all five of
    WARLOCK_LOG_LEVEL/NATIVE/NATIVE_DLL/NO_MIGRATE/MIGRATE_KEEP as absent from
    that readout, when only WARLOCK_LOG_LEVEL genuinely is -- the other four are
    config.SWITCHES's whole reason to exist.
    """
    chapter = _chapter("40-configuration.md")
    section = chapter.split("### Seeing which of these are actually set", 1)[1]
    section = section.split("\n## ", 1)[0]

    switch_vars = {env for _name, env in config.SWITCHES}
    settings_vars = {env for _name, env in config.SETTINGS}

    # Isolate whichever sentence(s) in the section still claim a variable is
    # "absent" from the Effective configuration readout.
    absent_sentences = [s for s in re.split(r"(?<=[.])\s+", section) if "absent" in s.lower()]
    assert absent_sentences, (
        "expected the section to still say which variable(s) are absent from "
        "effective() -- WARLOCK_LOG_LEVEL genuinely is"
    )

    # Every var effective() actually prints (config.SWITCHES) must not be
    # named in an "absent" sentence -- that is exactly the false claim docs-02
    # found.
    for var in switch_vars:
        for sentence in absent_sentences:
            assert var not in sentence, (
                f"{var} is appended by config.effective() (it is in "
                f"config.SWITCHES) but the chapter still lists it as absent: "
                f"{sentence!r}"
            )

    # But it must still be mentioned somewhere in the section as reported --
    # silently dropping it would just trade one failure for another.
    for var in switch_vars:
        assert var in section, (
            f"{var} is reported by config.effective() but is not mentioned "
            f"anywhere in the chapter's 'Seeing which of these are actually "
            f"set' section"
        )

    # WARLOCK_LOG_LEVEL is in neither table -- effective() genuinely never
    # prints it -- so the chapter should keep saying so.
    assert "WARLOCK_LOG_LEVEL" not in settings_vars
    assert "WARLOCK_LOG_LEVEL" not in switch_vars
    assert any("WARLOCK_LOG_LEVEL" in s for s in absent_sentences), (
        "WARLOCK_LOG_LEVEL is not in config.SETTINGS or config.SWITCHES, so "
        "effective() never prints it -- the chapter should still say it is "
        "absent"
    )


def test_installation_md_studio_extra_row_matches_pyproject():
    """docs-21: the installation chapter's `studio` extra row named moderngl,
    pygame-ce and imgui-bundle but not zstandard, which pyproject.toml's
    ``studio`` extra also carries (for zstd-compressed Tiled layer data).
    Parses the extra out of pyproject.toml rather than hard-coding the
    distribution list a second time.
    """
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    studio_extra = pyproject["project"]["optional-dependencies"]["studio"]
    # A requirement string is "name<specifier>"; keep just the distribution name.
    names = [re.split(r"[<>=!~;\[\s]", dep, maxsplit=1)[0].strip() for dep in studio_extra]
    assert names, "pyproject.toml's [project.optional-dependencies].studio is empty"

    chapter = _chapter("39-installation.md")
    match = re.search(r"\|\s*`studio`\s*\|([^|]*)\|", chapter)
    assert match, "no `studio` row found in the installation chapter's extras table"
    row_text = match.group(1)

    for name in names:
        assert name in row_text, (
            f"{name} is in pyproject.toml's studio extra but missing from the "
            f"installation chapter's studio row: {row_text!r}"
        )
