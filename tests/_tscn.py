"""A minimal, honest parser for Godot text scenes (``.tscn``), used only by
``tests/test_godotscene.py`` to check the structure ``godotscene.py`` writes.

This is not a Godot implementation. It understands exactly the subset of the
format this repo emits: section headers (``[kind key=value ...]``), simple
``key = value`` body properties, and the handful of value shapes
``godotscene.py`` writes (quoted strings, ints, floats, ``ExtResource(...)``,
``SubResource(...)``, ``&"..."`` StringNames, and the ``transitions`` array).
It does not evaluate expressions, does not know about every Godot resource
type, and would not survive a scene written by the real editor -- it exists
to let the test file ask "does every reference resolve" and "is every rule
in the brief honoured" without a Godot binary in CI.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = [
    "Section",
    "parse",
    "sections",
    "by_id",
    "parse_subresource",
    "parse_extresource",
    "parse_stringname",
    "parse_transitions",
]


@dataclass
class Section:
    kind: str
    header: dict[str, object]
    body: dict[str, str] = field(default_factory=dict)


_HEADER_ATTR_RE = re.compile(
    r'(\w+)=('
    r'"(?:[^"\\]|\\.)*"'
    r'|ExtResource\("[^"]*"\)'
    r'|SubResource\("[^"]*"\)'
    r'|[^\s\]]+'
    r")"
)
_INT_RE = re.compile(r"^-?\d+$")
_FLOAT_RE = re.compile(r"^-?\d+\.\d+$")
_EXT_REF_RE = re.compile(r'^ExtResource\("([^"]*)"\)$')
_SUB_REF_RE = re.compile(r'^SubResource\("([^"]*)"\)$')


def _parse_header_value(raw: str) -> object:
    if raw.startswith('"') and raw.endswith('"'):
        return raw[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    m = _EXT_REF_RE.match(raw)
    if m:
        return ("ExtResource", m.group(1))
    m = _SUB_REF_RE.match(raw)
    if m:
        return ("SubResource", m.group(1))
    if _INT_RE.match(raw):
        return int(raw)
    if _FLOAT_RE.match(raw):
        return float(raw)
    return raw


def parse(text: str) -> list[Section]:
    """Split a ``.tscn`` document into its sections, header attrs parsed."""
    result: list[Section] = []
    current: Section | None = None
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            inner = stripped[1:-1]
            kind_match = re.match(r"^(\w+)", inner)
            if not kind_match:
                raise ValueError(f"unparsable section header: {line!r}")
            kind = kind_match.group(1)
            attrs_text = inner[kind_match.end() :]
            header: dict[str, object] = {}
            for attr_match in _HEADER_ATTR_RE.finditer(attrs_text):
                key, raw = attr_match.group(1), attr_match.group(2)
                header[key] = _parse_header_value(raw)
            current = Section(kind=kind, header=header)
            result.append(current)
            continue
        if current is None:
            raise ValueError(f"body line before any section header: {line!r}")
        key, sep, value = stripped.partition(" = ")
        if not sep:
            raise ValueError(f"unparsable body line: {line!r}")
        current.body[key.strip()] = value.strip()
    return result


def sections(doc: list[Section], kind: str) -> list[Section]:
    return [s for s in doc if s.kind == kind]


def by_id(doc: list[Section], kind: str, resource_id: str) -> Section | None:
    for s in sections(doc, kind):
        if s.header.get("id") == resource_id:
            return s
    return None


def parse_subresource(raw: str) -> str:
    m = _SUB_REF_RE.match(raw.strip())
    if not m:
        raise ValueError(f"not a SubResource reference: {raw!r}")
    return m.group(1)


def parse_extresource(raw: str) -> str:
    m = _EXT_REF_RE.match(raw.strip())
    if not m:
        raise ValueError(f"not an ExtResource reference: {raw!r}")
    return m.group(1)


def parse_stringname(raw: str) -> str:
    m = re.fullmatch(r'&"([^"]*)"', raw.strip())
    if not m:
        raise ValueError(f"not a StringName literal: {raw!r}")
    return m.group(1)


_TRANSITIONS_TOKEN_RE = re.compile(r'"(?:[^"\\]|\\.)*"' r'|SubResource\("[^"]*"\)')


def parse_transitions(raw: str) -> list[tuple[str, str, str]]:
    """``transitions = ["from", "to", SubResource("id"), ...]`` -> triples."""
    stripped = raw.strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        raise ValueError(f"not an array literal: {raw!r}")
    inner = stripped[1:-1]
    tokens = _TRANSITIONS_TOKEN_RE.findall(inner)
    if len(tokens) % 3 != 0:
        raise ValueError(f"transitions array is not a multiple of 3: {raw!r}")
    triples: list[tuple[str, str, str]] = []
    for i in range(0, len(tokens), 3):
        frm_raw, to_raw, sub_raw = tokens[i], tokens[i + 1], tokens[i + 2]
        triples.append((frm_raw[1:-1], to_raw[1:-1], parse_subresource(sub_raw)))
    return triples
