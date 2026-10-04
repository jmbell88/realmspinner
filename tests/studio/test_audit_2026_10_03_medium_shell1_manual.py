"""The 2026-10-03 audit's shell-32 (Medium): the manual names what the boot path reads.

Boot is the one stretch of the app where a person has no window to read a
setting from, so an environment variable it honours is only discoverable from
the manual. ``REALMSPINNER_ALLOW_UNSAFE_LOCK`` was read by ``studio/main.py``
and the refusal dialog told the reader to set it, and neither chapter 41 nor 43
mentioned it.

Scoped on purpose to the two modules the boot path is made of -- ``studio/main.py``
and ``instance.py``, the single-instance lock it calls -- rather than the whole
package: ``config.py`` reads dozens more, and the configuration chapter has its
own table for those.
"""

from __future__ import annotations

import re
from pathlib import Path

import realmspinner

_NAME = re.compile(r"REALMSPINNER_[A-Z][A-Z0-9_]*[A-Z0-9]")

_PACKAGE = Path(realmspinner.__file__).resolve().parent
_BOOT_MODULES = (_PACKAGE / "studio" / "main.py", _PACKAGE / "instance.py")
_MANUAL = Path(__file__).resolve().parents[2] / "docs" / "manual"
_CHAPTERS = ("41-configuration.md", "43-troubleshooting.md")


def _boot_variables() -> set[str]:
    found: set[str] = set()
    for path in _BOOT_MODULES:
        found |= set(_NAME.findall(path.read_text(encoding="utf-8")))
    return found


def test_the_manual_names_every_environment_variable_the_boot_path_reads():
    names = _boot_variables()
    # The scan must be reading something: these are the ones main.py names today,
    # and an empty or shrunken set would pass the assertion below for free.
    assert {
        "REALMSPINNER_HOME",
        "REALMSPINNER_DB",
        "REALMSPINNER_T2I_ROOT",
        "REALMSPINNER_LOG_LEVEL",
        "REALMSPINNER_ALLOW_UNSAFE_LOCK",
    } <= names
    text = "\n".join((_MANUAL / chapter).read_text(encoding="utf-8") for chapter in _CHAPTERS)
    missing = sorted(name for name in names if name not in text)
    assert not missing, (
        f"{missing} are read at boot (studio/main.py, instance.py) but named in neither "
        f"{' nor '.join(_CHAPTERS)}"
    )
