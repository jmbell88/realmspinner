"""``agent_character``'s tool surface: schemas built from live registries,
the call contract (unknown tool/argument, field aliasing, the blast-radius
limit on ``character_cancel``), and each handler's own refusals.

S1's own doors (``service.characters``, ``service.troupe``, ``service.rig``,
``cliplib.shipped_clip_*``, ``clips.shipped_clip_timing``) are real on this
branch now -- ``cliplib.shipped_clip_library``/``shipped_clip_templates``/
``shipped_clip_names``, ``clips.shipped_clip_timing`` and
``service.characters.ASSET_FILTERS`` need no stand-in any more and this file
no longer supplies one. What is still monkeypatched, test by test, is
narrower: doors that would spawn Blender (``create_character``,
``send_to_troupe``, ``rig.create_rig`` -- rigging genuinely shells out) or
need a real job store row this file has no reason to build twice
(``_jobs_lifecycle.cancel_job``). ``service.characters.agent_export_stem``
and ``service.export.run_character_export``'s/``export_frames``'s/
``export_godot``'s/``export_package``'s own ``stem=`` keyword have since
landed too, and every export test below now runs them for real (see
``test_an_agent_export_lands_in_a_folder_named_for_its_ids``'s and
``test_a_hostile_job_name_cannot_escape_the_export_dir``'s own docstrings) --
``run_character_export`` itself is still faked in the first of those two,
only to capture what this module passed it, never because the real door is
unavailable.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import re
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import pytest

from warlock.mcp import rpc
from warlock.studio import agent_character as ac

pytestmark = pytest.mark.filterwarnings("ignore")


# --- test scaffolding ---------------------------------------------------------


def _new_job_id() -> str:
    return uuid.uuid4().hex[:12]


def _mint_model_job(svc: Any, *, job_id: str | None = None) -> str:
    """A minimal 'done' model row -- enough for ``svc.require_job``/
    ``svc.job_dir`` to answer, which is all ``_h_character_rig`` reads
    before consulting ``store.read_rig``/``rig_in_flight`` (both
    monkeypatched directly by the tests that need them, rather than by
    planting real files -- see each test)."""
    job_id = job_id or _new_job_id()
    svc.store.create("image", "a knight", {}, job_id, stage="model", status="done")
    return job_id


# --- schema construction --------------------------------------------------


def _walk_enum_properties(tool_name: str, node: Any, path: tuple, found: dict) -> None:
    """Populates *found* with ``{(tool_name, path): set(enum values)}`` for
    *node* and everything reachable from it -- every top-level property,
    recursively into ``items`` (an array's element schema) and nested
    ``properties`` (an object property's own sub-schema). ``path`` marks an
    array step as the literal string ``"[]"`` so ``movements[].name`` and a
    bare ``movements`` property can never collide as dict keys."""
    if not isinstance(node, dict):
        return
    if "enum" in node:
        found[(tool_name, path)] = set(node["enum"])
    if node.get("type") == "object" and isinstance(node.get("properties"), dict):
        for key, sub in node["properties"].items():
            _walk_enum_properties(tool_name, sub, path + (key,), found)
    if node.get("type") == "array":
        items = node.get("items")
        if isinstance(items, dict):
            _walk_enum_properties(tool_name, items, path + ("[]",), found)


def test_every_enum_is_its_registry_and_every_registry_value_is_in_an_enum() -> None:
    """Every enum *any* tool schema declares -- walked recursively through
    every top-level property, every array's own item schema and every
    nested object property, not a hand-picked sample of them -- is exactly
    what the *real* registry it is supposed to mirror reports.

    A hand-picked list used to check only ``character_create``'s own
    top-level enums plus a handful of others: ``character_sheet_create``'s
    own ``directions``/``fps``/``camera``/``colors``/``outline``/
    ``reduce_mode`` (the same pixel-settings properties ``character_create``
    already carries, declared a second time on a second tool) and *every*
    tool's own ``movements[]`` item enums (``name`` on both minting tools,
    ``directions`` on ``character_sheet_create``'s own movement items) were
    never read by this test at all -- so pointing any one of those at the
    wrong registry (the rig catalog instead of the shipped clip templates,
    say) would still pass. This walk finds every enum-bearing property by
    construction (never a name typed twice, once in the schema and once
    here) and checks it against an expectation table keyed by
    ``(tool, path)`` rather than by property name alone, because the same
    name means a different registry on different tools -- ``template``
    names a rig on ``character_rig`` but a *skeleton with shipped clips* on
    ``character_clips``/``character_sheet_create``, two sets that do not
    even have to agree (a template can rig without shipping a single clip).
    Never checked against ``agent_character._enums()`` itself, which would
    only prove :func:`tools` agrees with :func:`_enums` and miss a bug where
    both read the wrong door the same wrong way (e.g. the user-first
    ``cliplib.clip_library`` instead of the shipped-only
    ``cliplib.shipped_clip_names`` -- see the next test for that one, since
    a fresh ``WARLOCK_HOME`` with no user file makes the two agree here).

    ``size`` is deliberately absent from this walk's own expectation table:
    since master's 8b091e98 Send to Troupe, it is a bounded integer, not an
    enum (see ``agent_character``'s own "Registries, not a hand-kept menu"
    paragraph), so this walk -- which only ever records an ``enum`` key --
    must never find one at ``("character_create"|"character_sheet_create",
    ("size",))``. The checks below the sorted-order block, at the end of
    this same test, are this walk's replacement for that property: both
    tools' own ``size`` schema is asserted to carry no ``enum`` and to
    bound the same ``TROUPE_CUSTOM_SIZE_RANGE`` the door underneath enforces.
    """
    from warlock.characters import family as family_mod
    from warlock.kernels.rig import cliplib, templates
    from warlock.pipelines import charsheet, pixelize
    from warlock.service import characters as svc_characters
    from warlock.service import export as svc_export
    from warlock.service import troupe as svc_troupe

    families = family_mod.families()
    themes = {t.key for fam in families.values() for t in fam.themes}
    sheet_templates = set(cliplib.shipped_clip_templates())
    movements = {
        name for t in sheet_templates for name in cliplib.shipped_clip_names(t)
    }
    directions = set(charsheet.DIRECTION_PRESETS)
    facings = set(charsheet.COMPASS_16)
    cameras = {key for key, _label, _elev in charsheet.CAMERA_PRESETS}
    fps = set(charsheet.FPS_CHOICES)
    colors = set(svc_troupe.TROUPE_COLOR_CHOICES)
    outlines = set(pixelize.OUTLINE_MODES)
    reduce_modes = set(pixelize.REDUCE_MODES)
    rig_templates = {r["key"] for r in templates.catalog()}
    filters = set(svc_characters.ASSET_FILTERS)
    formats = set(svc_export.CHARACTER_EXPORTS)

    # Every enum-bearing (tool, path) this surface's schemas are supposed to
    # declare, and the real registry each one must equal. A property with no
    # entry here fails below (an enum the table forgot); an entry here with
    # no matching property also fails (a removed enum, or a stale path from
    # a renamed one) -- so the table cannot silently go stale in either
    # direction.
    expected: dict[tuple[str, tuple], set[Any]] = {
        ("character_assets", ("filter",)): filters,
        ("character_clips", ("template",)): sheet_templates,
        ("character_create", ("family",)): set(families),
        ("character_create", ("theme",)): themes,
        ("character_create", ("movements", "[]", "name")): movements,
        ("character_create", ("directions",)): directions,
        ("character_create", ("fps",)): fps,
        ("character_create", ("camera",)): cameras,
        ("character_create", ("colors",)): colors,
        ("character_create", ("outline",)): outlines,
        ("character_create", ("reduce_mode",)): reduce_modes,
        ("character_rig", ("template",)): rig_templates,
        ("character_sheet_create", ("movements", "[]", "name")): movements,
        ("character_sheet_create", ("movements", "[]", "directions")): directions,
        ("character_sheet_create", ("directions",)): directions,
        ("character_sheet_create", ("template",)): sheet_templates,
        ("character_sheet_create", ("fps",)): fps,
        ("character_sheet_create", ("camera",)): cameras,
        ("character_sheet_create", ("colors",)): colors,
        ("character_sheet_create", ("outline",)): outlines,
        ("character_sheet_create", ("reduce_mode",)): reduce_modes,
        ("character_sheet_preview", ("movement",)): movements,
        ("character_sheet_preview", ("direction",)): facings,
        ("character_export", ("format",)): formats,
    }

    found: dict[tuple[str, tuple], set[Any]] = {}
    for tool in ac.tools():
        _walk_enum_properties(tool.name, tool.schema, (), found)

    undocumented = sorted(map(str, set(found) - set(expected)))
    assert not undocumented, (
        f"these enum properties have no entry in this test's own expectation "
        f"table: {undocumented}. Add one naming the real registry it must "
        f"mirror."
    )
    vanished = sorted(map(str, set(expected) - set(found)))
    assert not vanished, (
        f"these expectation-table entries no longer correspond to a real "
        f"enum property -- a tool or a property was renamed or removed: "
        f"{vanished}. Update or drop the stale entry."
    )

    for key, expected_values in expected.items():
        assert found[key] == expected_values, (key, found[key], expected_values)

    # Order is defined for the four registries _enums() itself builds by
    # sorting -- checked as sorted sequences too, on top of the set checks
    # above, so a schema that kept the right *values* but stopped sorting
    # them (or sorted the wrong axis) still fails here.
    tools_by_name = {t.name: t for t in ac.tools()}
    assert tools_by_name["character_create"].schema["properties"]["family"]["enum"] == sorted(
        families
    )
    assert tools_by_name["character_create"].schema["properties"]["theme"]["enum"] == sorted(
        themes
    )
    assert tools_by_name["character_create"].schema["properties"]["directions"][
        "enum"
    ] == sorted(directions)
    assert tools_by_name["character_create"].schema["properties"]["movements"]["items"][
        "properties"
    ]["name"]["enum"] == sorted(movements)

    # "size" is the one property this walk deliberately never finds an enum
    # at any more (see this test's own docstring): master's 8b091e98 Send to
    # Troupe widened it to any whole number in
    # ``service.troupe.TROUPE_CUSTOM_SIZE_RANGE``, so the schema declares
    # that range instead of ``charsheet.SIZES``'s preset ladder. Checked on
    # both tools that carry it -- ``pix_properties()`` is shared between
    # them -- against the same process-stable constant, not a restated pair
    # of numbers.
    size_lo, size_hi = svc_troupe.TROUPE_CUSTOM_SIZE_RANGE
    for tool_name in ("character_create", "character_sheet_create"):
        size_schema = tools_by_name[tool_name].schema["properties"]["size"]
        assert "enum" not in size_schema, (tool_name, size_schema)
        assert size_schema["type"] == "integer", (tool_name, size_schema)
        assert size_schema["minimum"] == size_lo, (tool_name, size_schema)
        assert size_schema["maximum"] == size_hi, (tool_name, size_schema)


def test_a_camera_preset_added_at_runtime_appears_in_the_next_catalogue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from warlock.pipelines import charsheet

    before = ac.tools()
    camera_before = set(
        {t.name: t for t in before}["character_create"].schema["properties"]["camera"]["enum"]
    )
    assert "extra_angle" not in camera_before

    monkeypatch.setattr(
        charsheet,
        "CAMERA_PRESETS",
        (*charsheet.CAMERA_PRESETS, ("extra_angle", "Extra angle", 12.0)),
    )
    after = {t.name: t for t in ac.tools()}
    camera_after = set(after["character_create"].schema["properties"]["camera"]["enum"])
    assert "extra_angle" in camera_after


def test_editing_a_user_clip_library_does_not_move_the_character_catalogue(
    tmp_path: Path,
) -> None:
    """Writes a *real* user clip library for humanoid -- a v3 file with an
    extra clip Poser's editor would save -- through ``cliplib.
    set_user_clip_dir``, invalidating the caches the way ``service.clips.
    save`` does after a real write. ``cliplib.shipped_clip_*`` reads only
    the shipped ``templates/clips`` tree and must never move for this
    (see ``agent_character``'s own "Movements are the shipped vocabulary"
    paragraph); a version of this module that fell back to the user-first
    ``cliplib.clip_library`` for its movements enum would grow the extra
    clip into both the tool schema and the vocabulary resource, and this
    test would catch it there.
    """
    from warlock.kernels.rig import cliplib
    from warlock.studio import agent_character_resources as acr

    before_tools = json.dumps([rpc.tool_dict(t) for t in ac.tools()], sort_keys=True)
    before_vocab = acr.read_static(acr.VOCABULARY_URI)[1]

    raw = json.loads((cliplib.CLIP_DIR / "humanoid.json").read_text(encoding="utf-8"))
    assert raw.get("version") == 3  # the shipped file is already v3; leave it untouched
    # A structurally valid new clip -- "keys" must name real poses this same
    # file already carries, and "segments" one count per key, or
    # parse_clip_library raises and the whole user file is dropped (logged,
    # not refused) rather than read -- which would make this test pass for
    # the wrong reason (no user edit ever took effect). Copied from the
    # shipped "idle" clip's own keys/segments shape, under a new name and a
    # duration_ms no shipped clip carries.
    idle = next(c for c in raw["clips"] if c["name"] == "idle")
    raw["clips"].append(
        {
            "name": "a_users_own_extra_clip",
            "closed": idle["closed"],
            "easing": idle["easing"],
            "keys": list(idle["keys"]),
            "segments": list(idle["segments"]),
            "duration_ms": 250,
        }
    )
    (tmp_path / "humanoid.json").write_text(json.dumps(raw), encoding="utf-8")

    cliplib.set_user_clip_dir(tmp_path)
    try:
        cliplib.invalidate_clips()  # what service.clips.save does after writing
        # Sanity: the user file really is being read by *something* -- the
        # user-first door sees the new clip -- so a false pass here (both
        # sides equal only because the write never took effect) is ruled out.
        assert "a_users_own_extra_clip" in {
            c["name"] for c in cliplib.clip_library("humanoid")["clips"]
        }

        after_tools = json.dumps([rpc.tool_dict(t) for t in ac.tools()], sort_keys=True)
        after_vocab = acr.read_static(acr.VOCABULARY_URI)[1]
        assert after_tools == before_tools
        assert after_vocab == before_vocab
    finally:
        cliplib.set_user_clip_dir(None)
        cliplib.invalidate_clips()


# --- handler/tool bookkeeping -----------------------------------------------


def test_every_handler_has_a_tool_and_every_tool_has_a_handler() -> None:
    assert set(ac.HANDLERS) == {t.name for t in ac.tools()}


def test_no_handler_takes_a_ctx() -> None:
    """Every handler's first parameter is named ``svc`` -- Clay's handlers
    take ``ctx``; this surface never does (see the module docstring: it
    calls ``service`` doors directly, which take a ``WarlockService``, not
    the frame-thread ``ctx`` Clay's ``ClayState``-backed handlers need)."""
    for name, handler in ac.HANDLERS.items():
        params = list(inspect.signature(handler).parameters)
        assert params[0] == "svc", f"{name} takes {params[0]!r} as its first argument"
        assert "ctx" not in params, f"{name} takes a ctx argument"


def test_no_character_tool_shares_a_name_with_a_clay_or_transport_tool() -> None:
    from warlock.studio import agent_clay, agent_host

    clay_names = {t.name for t in agent_clay.tools()} | {agent_host.STATUS_TOOL}
    character_names = {t.name for t in ac.tools()}
    assert not (clay_names & character_names)


def test_every_recovery_a_character_refusal_names_is_in_agent_clays_vocabulary() -> None:
    """``agent_clay.RECOVERY`` is read by parsing ``agent_clay.py``'s own
    source rather than importing the module, per this tranche's own rule --
    ``agent_character`` must never import ``agent_clay`` (it pulls GL), and
    a test for this module should not need to either."""
    agent_clay_path = (
        Path(ac.__file__).resolve().parent / "agent_clay.py"
    )
    source = agent_clay_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    recovery_words: set[str] | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "RECOVERY" for t in node.targets
        ):
            recovery_words = set(ast.literal_eval(node.value.args[0]))
            break
    assert recovery_words, "could not find RECOVERY in agent_clay.py"

    used: set[str] = set()
    for match in re.finditer(r'recovery\s*=\s*"([a-z_]+)"', Path(ac.__file__).read_text("utf-8")):
        used.add(match.group(1))
    # fail()'s own default (a refusal that names a field) always resolves to
    # this word -- see agent_character.fail's docstring.
    used.add("fix_arguments")
    assert used <= recovery_words, used - recovery_words


def test_the_character_catalogue_fits_its_own_budget() -> None:
    """Pins the character catalogue's own size budget, mirroring
    ``tests/test_agent_schemas.py::test_the_tools_list_catalogue_and_
    instructions_fit_an_agents_first_call``'s reasoning for Clay: growing
    this is a decision, not a side effect of an unrelated registry change.

    Measured on 2026-09-13 (this worktree's shipped clip registries: ten
    clips per template, four sheet templates): the ten tools' catalogue
    JSON plus ``instructions()`` totalled about 10.5k chars. CHARACTER_CEILING
    here is 12,000 -- roughly 14% of headroom above that measurement.

    Raised to 12,500 the same day: the custom-size fix (``size`` widened
    from an enum to a bounded integer, see ``agent_character``'s "Registries,
    not a hand-kept menu" paragraph) gave ``character_create``'s and
    ``character_sheet_create``'s own ``size`` property a ``description``
    naming the range and the preset ladder, on both tools -- deliberate
    growth (~12.1k measured), not a side effect.
    """
    CHARACTER_CEILING = 12_500

    tools = ac.tools()
    tool_jsons = [rpc.tool_dict(t) for t in tools]
    catalogue = json.dumps({"tools": tool_jsons})
    instructions = ac.instructions()
    total = len(catalogue) + len(instructions)

    sizes = sorted(
        ((len(json.dumps(tj)), tj.get("name", "?")) for tj in tool_jsons), reverse=True
    )
    biggest = ", ".join(f"{name}={size}" for size, name in sizes[:5])

    assert total <= CHARACTER_CEILING, (
        f"the character catalogue ({len(catalogue)} chars) plus its "
        f"instructions ({len(instructions)} chars) now total {total} chars, "
        f"over the {CHARACTER_CEILING}-char budget. Largest schemas: {biggest}. "
        f"If this growth is deliberate, raise CHARACTER_CEILING here and say "
        f"why in the same commit."
    )


def test_a_refusal_names_a_property_the_tool_declares(tmp_path: Path) -> None:
    """A sample of this surface's own pre-door validations (see the module
    docstring's "Validated here, not left to the door"): every one refuses
    on one of its own tool's declared properties."""

    class _Config:
        export_dir = tmp_path

    class _Svc:
        config = _Config()

    session = ac.Session()
    cases: list[tuple[str, dict[str, Any], str, Any]] = [
        ("character_create", {"family": "not-a-family"}, "family", object()),
        ("character_create", {"theme": "not-a-theme"}, "theme", object()),
        ("character_create", {"directions": 3}, "directions", object()),
        ("character_create", {"size": 7}, "size", object()),
        ("character_create", {"colors": 7}, "colors", object()),
        ("character_create", {"outline": "bogus"}, "outline", object()),
        ("character_create", {"reduce_mode": "bogus"}, "reduce_mode", object()),
        ("character_create", {"fps": 3}, "fps", object()),
        ("character_create", {"camera": "bogus"}, "camera", object()),
        ("character_rig", {"job_id": "0" * 12, "template": "bogus"}, "template", object()),
        (
            "character_sheet_create",
            {"job_id": "0" * 12, "movements": [{"name": "bogus"}]},
            "movements",
            object(),
        ),
        (
            "character_export",
            {"job_id": "0" * 12, "format": "bogus"},
            "format",
            _Svc(),
        ),
    ]
    for name, args, expected_field, fake_svc in cases:
        result = ac.call(fake_svc, session, name, args)
        assert result["isError"], (name, args)
        declared = ac._allowed_argument_names()[name]
        field = result["structuredContent"]["field"]
        assert field == expected_field
        assert field in declared


def test_a_bad_job_id_refusal_still_names_the_job_id_field(svc: Any) -> None:
    """The 2026-09-14 audit (agents-06): ``_mapped_field``'s documented
    "job_id, if the tool declares one" fallback never ran when the
    ``ServiceError``'s own ``field`` was empty, because an early
    ``if not raw_field: return None`` returned before the fallback was ever
    tried -- and every ``check_job_id``/``check_sheet_id`` refusal (the
    commonest character-tool mistake) raises exactly that shape
    (``NotFound("no such job")``, no ``field``). A well-formed but
    non-existent job id must still come back pointing at ``job_id``, not at
    no control at all."""
    result = ac.call(svc, ac.Session(), "character_rig", {"job_id": "0" * 12})
    assert result["isError"]
    assert result["structuredContent"]["field"] == "job_id"


def test_an_agent_may_ask_for_a_custom_sprite_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """40px is off ``charsheet.SIZES``' own preset ladder but inside
    ``service.troupe.TROUPE_CUSTOM_SIZE_RANGE`` -- master's 8b091e98 Send to
    Troupe accepts exactly this off-ladder size, and this surface's own
    ``size`` property, now a bounded integer rather than an enum of
    presets, must reach the sheet door with it unchanged rather than refuse
    it the way the old enum-only schema did."""
    from warlock.service import rig as svc_rig
    from warlock.service import troupe as svc_troupe

    e = ac._enums()
    assert 40 not in e.sizes  # sanity: genuinely off the preset ladder
    assert e.size_range[0] <= 40 <= e.size_range[1]

    monkeypatch.setattr(svc_rig, "rig_in_flight", lambda svc, jid: False)
    captured: dict[str, Any] = {}

    def fake_send_to_troupe(svc, jid, **kw):
        captured.update(kw)
        return {"id": "dddddddddddd", "source_job": jid, "sheet_id": "eeeeeeeeeeee"}

    monkeypatch.setattr(svc_troupe, "send_to_troupe", fake_send_to_troupe)

    result = ac.call(
        object(),
        ac.Session(),
        "character_sheet_create",
        {"job_id": "a" * 12, "movements": [{"name": e.movements[0]}], "size": 40},
    )
    assert result["isError"] is False, result
    assert captured["logical_size"] == 40


def test_a_size_outside_the_custom_range_is_refused_on_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Below the floor (7) or above the ceiling (257) of
    ``service.troupe.TROUPE_CUSTOM_SIZE_RANGE``, on both tools that declare
    ``size`` -- refused by the handler itself, on ``field="size"``, before
    either door (``create_character``/``send_to_troupe``) ever runs."""
    from warlock.service import rig as svc_rig

    monkeypatch.setattr(svc_rig, "rig_in_flight", lambda svc, jid: False)
    e = ac._enums()
    lo, hi = e.size_range
    assert lo == 8 and hi == 256  # sanity: this test's own claimed bounds

    for size in (7, 257):
        create_result = ac.call(object(), ac.Session(), "character_create", {"size": size})
        assert create_result["isError"], size
        assert create_result["structuredContent"]["field"] == "size"

        sheet_result = ac.call(
            object(),
            ac.Session(),
            "character_sheet_create",
            {"job_id": "a" * 12, "movements": [{"name": e.movements[0]}], "size": size},
        )
        assert sheet_result["isError"], size
        assert sheet_result["structuredContent"]["field"] == "size"


def test_no_character_tool_accepts_a_path() -> None:
    for tool in ac.tools():
        for prop in tool.schema.get("properties", {}):
            assert "path" not in prop.lower() and "file" not in prop.lower(), (
                tool.name,
                prop,
            )


#: The real modules that talk to an inference model -- SDXL's own
#: worker-side pipeline, the in-app handle that spawns/drives it, the
#: TRELLIS.2 native-server client, and Muse's two ACE-Step subprocesses.
#: Checked against ``importlib.util.find_spec`` below so a rename of any of
#: these cannot silently empty this test's own denylist and leave it
#: passing for the wrong reason.
_FORBIDDEN_MODEL_MODULES = (
    "warlock.pipelines.text2image",
    "warlock.pipelines.t2i_client",
    "warlock.pipelines.trellis",
    "warlock.pipelines.music_worker",
    "warlock.pipelines.separation_worker",
)
_FORBIDDEN_MODEL_SUBSTRINGS = ("torch", "diffusers", "transformers")


def _base_package_for_level(package: str, level: int) -> str:
    """Python's own relative-import resolution rule, applied to a dotted
    package name instead of a live module -- both files this test walks
    live directly in *package* (``warlock.studio``), never in a
    sub-package, so this is exactly what the interpreter would compute for
    a ``from .``/``from ..`` in either of them."""
    parts = package.split(".")
    if level <= 1:
        return package
    trimmed = parts[: -(level - 1)] if level - 1 <= len(parts) else []
    return ".".join(trimmed)


def _dotted_names(node: ast.AST, package: str) -> list[str]:
    """Every real dotted module name one import statement could resolve
    to -- including, for ``from X import Y``, ``X.Y`` itself (an attribute
    import can also be a submodule import in disguise: ``from ..pipelines
    import trellis`` imports ``warlock.pipelines.trellis`` whether
    ``trellis`` is read as a module or an attribute of ``pipelines``). A
    check that only looked at ``node.module`` -- as this test used to --
    would see ``"pipelines"`` for exactly that import and never notice
    ``trellis`` naming the forbidden module at all.
    """
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        if node.level:
            base = _base_package_for_level(package, node.level)
            module_path = f"{base}.{node.module}" if node.module else base
        else:
            module_path = node.module or ""
        names = [module_path] if module_path else []
        for alias in node.names:
            if alias.name == "*":
                continue
            names.append(f"{module_path}.{alias.name}" if module_path else alias.name)
        return names
    return []


def test_the_character_surface_cannot_reach_a_model_that_runs_inference() -> None:
    """Neither module imports a real inference stack -- checked three ways,
    each catching a gap the other two cannot:

    1. An AST walk of *every* import in both files (``ast.walk`` already
       reaches one buried inside a function body, not just top-of-file
       ones), each resolved to the real dotted module it names -- see
       :func:`_dotted_names` for why a raw substring match on
       ``node.module`` alone is not enough.
    2. A fresh subprocess that actually imports both modules and inspects
       ``sys.modules`` afterward, so a lazy import three hops down a door
       this surface calls (a service function importing a pipeline module
       inside its own body) cannot hide from the static check alone.
    3. The denylist names real, importable modules (``importlib.util.
       find_spec``), so a file rename could not silently empty it and
       leave this test passing having checked nothing.
    """
    for dotted in _FORBIDDEN_MODEL_MODULES:
        assert importlib.util.find_spec(dotted) is not None, dotted

    for name in ("agent_character.py", "agent_character_resources.py"):
        path = Path(ac.__file__).resolve().parent / name
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            for dotted in _dotted_names(node, "warlock.studio"):
                lowered = dotted.lower()
                assert not any(bad in lowered for bad in _FORBIDDEN_MODEL_SUBSTRINGS), (
                    name,
                    dotted,
                )
                assert not any(
                    dotted == bad or dotted.startswith(bad + ".")
                    for bad in _FORBIDDEN_MODEL_MODULES
                ), (name, dotted)

    script = (
        "import sys\n"
        "import warlock.studio.agent_character\n"
        "import warlock.studio.agent_character_resources\n"
        f"substrings = {_FORBIDDEN_MODEL_SUBSTRINGS!r}\n"
        f"modules = {_FORBIDDEN_MODEL_MODULES!r}\n"
        "hits = sorted(\n"
        "    m for m in sys.modules\n"
        "    if any(b in m.lower() for b in substrings)\n"
        "    or any(m == b or m.startswith(b + '.') for b in modules)\n"
        ")\n"
        "print(hits)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "[]", completed.stdout


# --- call contract ----------------------------------------------------------


def test_an_agent_may_cancel_only_the_jobs_it_minted(monkeypatch: pytest.MonkeyPatch) -> None:
    from warlock.service import _jobs_lifecycle

    job_id = _new_job_id()
    session = ac.Session()

    refused = ac.call(object(), session, "character_cancel", {"job_id": job_id})
    assert refused["isError"]
    assert refused["structuredContent"]["field"] == "job_id"

    monkeypatch.setattr(_jobs_lifecycle, "cancel_job", lambda svc, jid: {"ok": True})
    session.minted[job_id] = "model"
    accepted = ac.call(object(), session, "character_cancel", {"job_id": job_id})
    assert accepted["isError"] is False


def test_an_agent_may_cancel_the_sheet_its_rig_queued(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fix 5: ``character_cancel`` also reaches the follow-up sheet job of
    a rig this session minted -- resolved fresh through
    ``troupe.follow_up_sheet_job`` at call time, never by widening
    ``session.minted`` itself -- but nothing else this session did not
    start, even a sheet job that names a *different* rig."""
    from warlock.service import _jobs_lifecycle
    from warlock.service import troupe as svc_troupe

    session = ac.Session()
    rig_id = "a" * 12
    sheet_id = "b" * 12
    session.minted[rig_id] = "rig"

    def fake_follow_up_sheet_job(svc, rig_job_id):
        assert rig_job_id == rig_id
        return sheet_id

    monkeypatch.setattr(svc_troupe, "follow_up_sheet_job", fake_follow_up_sheet_job)
    monkeypatch.setattr(
        _jobs_lifecycle, "cancel_job", lambda svc, jid: {"ok": True, "job_id": jid}
    )

    result = ac.call(object(), session, "character_cancel", {"job_id": sheet_id})
    assert result["isError"] is False, result
    assert sheet_id not in session.minted  # never widened -- resolved fresh each time

    # A sheet job naming a *different* rig's follow-up is still refused.
    monkeypatch.setattr(svc_troupe, "follow_up_sheet_job", lambda svc, rid: "c" * 12)
    refused = ac.call(object(), session, "character_cancel", {"job_id": "d" * 12})
    assert refused["isError"]
    assert refused["structuredContent"]["field"] == "job_id"


def test_character_sheet_create_refuses_while_a_rig_is_already_running_and_instructions_agree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 2026-09-16 audit, agents-03: ``instructions()`` told a client that
    ``character_sheet_create`` accepts "a rigged mesh -- or one whose rig is
    still running, which queues the sheet to follow it", but
    ``_h_character_sheet_create`` refuses outright ("a rig is already
    running for this mesh", ``field="job_id"``, ``recovery="wait"``)
    whenever ``svc_rig.rig_in_flight`` is true, before ``send_to_troupe`` is
    ever called -- the same refusal ``character_rig``'s own tool
    description (not ``instructions()``) already documents. The prose is
    the one that moved: it must no longer promise queueing behind an
    in-flight rig."""
    from warlock.service import rig as svc_rig

    e = ac._enums()
    monkeypatch.setattr(svc_rig, "rig_in_flight", lambda svc, jid: "some-rig-id")

    result = ac.call(
        object(),
        ac.Session(),
        "character_sheet_create",
        {"job_id": "a" * 12, "movements": [{"name": e.movements[0]}]},
    )
    assert result["isError"]
    assert result["structuredContent"]["field"] == "job_id"
    assert result["structuredContent"]["recovery"] == "wait"

    instructions = ac.instructions()
    assert "which queues the sheet to follow it" not in instructions
    assert "still running" in instructions
    assert "wait" in instructions


def test_manual_46_cancel_paragraph_names_the_follow_up_sheet_exception() -> None:
    """The 2026-09-16 audit, agents-08: the manual's "What gets refused, and
    why" paragraph says flatly that "an agent may cancel only the jobs it
    started on its own connection -- a job a human began, or an earlier
    session minted, is not reachable by character_cancel at all", but
    ``_h_character_cancel`` also reaches one job it never minted: the
    follow-up sheet job that a rig it minted has since queued
    (``_rig_queued_this_sheet``), which ``instructions()`` itself documents
    ("or the sheet job a rig it minted has since queued"). The manual is
    stricter than the code; this is a doc fix, not a code fix, so the
    regression reads the chapter text itself."""
    manual = Path(__file__).resolve().parents[1] / "docs" / "manual" / "46-extending.md"
    text = manual.read_text(encoding="utf-8")
    paragraph_start = text.index("**What gets refused, and why.**")
    paragraph = text[paragraph_start : paragraph_start + 1200]
    assert "cancel" in paragraph
    assert "sheet job a rig it" in paragraph or "follow-up sheet" in paragraph


def test_a_missing_required_argument_is_refused_on_that_field() -> None:
    """Fix 4: a call missing a REQUIRED top-level argument used to reach a
    handler's own ``args["job_id"]`` indexing directly and surface as the
    generic "failed unexpectedly" refusal (a bare ``KeyError``, caught only
    by ``call()``'s blanket ``except Exception``).
    ``agent_character._structural_refusal`` now catches it before any
    handler runs, naming the missing argument -- and, for a ``movements``
    array whose item is not even an object, refuses on the array property
    itself rather than ``AttributeError``-ing on a bare string's own
    ``.get(...)``.
    """
    result = ac.call(object(), ac.Session(), "character_rig", {})
    assert result["isError"]
    assert result["structuredContent"]["field"] == "job_id"
    assert "failed unexpectedly" not in result["content"][0]["text"]

    result2 = ac.call(
        object(),
        ac.Session(),
        "character_sheet_create",
        {"job_id": "a" * 12, "movements": ["not-an-object"]},
    )
    assert result2["isError"]
    assert result2["structuredContent"]["field"] == "movements"
    assert "failed unexpectedly" not in result2["content"][0]["text"]


#: What the shipped prose actually tells an agent to poll ``character_job``
#: on -- "poll character_job on [the] [returned] WORD". Matches both
#: "poll character_job on the returned rig_job_id" (the prompt's first
#: instruction) and "poll character_job on that job" (either prose's own
#: second instruction, for the follow-up sheet) -- the second capture is
#: never asserted against, only the *first* match matters here (see the
#: test below), but the pattern still has to tolerate it to find that first
#: match at all.
_POLL_TARGET_RE = re.compile(
    r"poll\s+character_job\s+on\s+(?:the\s+)?(?:returned\s+)?(\w+)", re.IGNORECASE
)


def test_the_prompt_polls_the_rig_job_not_the_mesh() -> None:
    """Fix 1's own exact-prose claim: the shipped
    ``character_sheets_from_description`` prompt (and ``instructions()``)
    must tell the agent to poll ``character_job`` on the returned
    ``rig_job_id`` -- not merely *mention* that word somewhere -- until
    ``follow_up_sheet_job`` names a job, then poll that job, and to read
    ``follow_up_failure`` and stop rather than poll forever when the rig
    itself ends in error. A bare substring check (``"rig_job_id" in text``)
    would still pass if the poll instruction were rewritten to name
    ``mesh_job_id`` as what to poll, so long as ``rig_job_id`` was mentioned
    *anywhere else* in the prose (e.g. "mesh_job_id reports the same rig_job_id
    too"); :data:`_POLL_TARGET_RE` instead pulls out exactly the word each
    prose names as the poll target and asserts it is ``rig_job_id``, never
    ``mesh_job_id``. The pre-fix prose told the agent to poll the *mesh* id
    waiting for ``follow_up_sheet_job`` to appear there, with no mention of a
    rig ending in error at all."""
    from warlock.studio import agent_prompts

    rendered = agent_prompts.render(
        "character_sheets_from_description", {"description": "a swamp knight"}
    )
    assert rendered is not None
    _description, messages = rendered
    text = messages[0]["content"]["text"]
    prompt_targets = _POLL_TARGET_RE.findall(text)
    assert prompt_targets, text  # sanity: the prose really does say "poll character_job on ..."
    assert prompt_targets[0] == "rig_job_id", prompt_targets
    assert "mesh_job_id" not in prompt_targets
    assert "follow_up_sheet_job" in text
    assert "follow_up_failure" in text

    instructions = ac.instructions()
    instruction_targets = _POLL_TARGET_RE.findall(instructions)
    assert instruction_targets, instructions
    assert instruction_targets[0] == "rig_job_id", instruction_targets
    assert "mesh_job_id" not in instruction_targets
    assert "follow_up_sheet_job" in instructions
    assert "follow_up_failure" in instructions


def test_character_rig_refuses_a_mesh_that_is_already_rigged(
    monkeypatch: pytest.MonkeyPatch, svc: Any
) -> None:
    from warlock.kernels.rig import store

    job_id = _mint_model_job(svc)
    monkeypatch.setattr(store, "read_rig", lambda job_dir: {"template": "humanoid"})

    result = ac.call(svc, ac.Session(), "character_rig", {"job_id": job_id})
    assert result["isError"]
    assert result["structuredContent"]["field"] == "job_id"
    assert "never replaces" in result["content"][0]["text"]


def test_character_rig_refuses_while_a_rig_is_running(
    monkeypatch: pytest.MonkeyPatch, svc: Any
) -> None:
    from warlock.kernels.rig import store
    from warlock.service import rig as svc_rig

    job_id = _mint_model_job(svc)
    monkeypatch.setattr(store, "read_rig", lambda job_dir: None)
    monkeypatch.setattr(svc_rig, "rig_in_flight", lambda svc, jid: "some-rig-id", raising=False)

    result = ac.call(svc, ac.Session(), "character_rig", {"job_id": job_id})
    assert result["isError"]
    assert result["structuredContent"]["field"] == "job_id"
    assert result["structuredContent"]["recovery"] == "wait"


def test_character_rig_mints_and_records_a_rig(
    monkeypatch: pytest.MonkeyPatch, svc: Any
) -> None:
    from warlock.kernels.rig import store
    from warlock.service import rig as svc_rig

    job_id = _mint_model_job(svc)
    monkeypatch.setattr(store, "read_rig", lambda job_dir: None)
    monkeypatch.setattr(svc_rig, "rig_in_flight", lambda svc, jid: None, raising=False)
    monkeypatch.setattr(
        svc_rig,
        "create_rig",
        lambda svc, jid, *, template=None: {
            "id": "abc123abc123",
            "source_job": jid,
            "template": "humanoid",
        },
    )

    session = ac.Session()
    result = ac.call(svc, session, "character_rig", {"job_id": job_id})
    assert result["isError"] is False
    payload = result["structuredContent"]
    assert payload["minted"] == [{"job_id": "abc123abc123", "kind": "rig", "role": "rig"}]
    assert session.minted["abc123abc123"] == "rig"


def test_an_unknown_species_is_refused_in_offer_sentences_words() -> None:
    """The real ``recipe_from_prompt`` (it mints nothing) for a prompt that
    names no species at all -- "a kraken" is a real word ``resolve`` tries
    to match and fails, never a stand-in for "empty string". The refusal's
    exact wording is computed the same way ``recipe_from_prompt`` computes
    it -- ``resolve.offer_sentence`` off the real resolution, falling back
    to the documented generic sentence when there is nothing to offer --
    rather than hard-coded, so a change to either sentence's wording moves
    this test's expectation along with it instead of going stale.
    """
    from warlock.characters import family as family_mod
    from warlock.characters import resolve

    prompt = "a kraken"
    resolution = resolve.resolve(prompt)
    assert resolution.family is None  # sanity: this prompt really names no species
    offered = resolve.offer_sentence(resolution)
    expected = offered or (
        f"Warlock builds {len(family_mod.families())} species across four "
        "body plans, and this prompt names none of them."
    )

    result = ac.call(object(), ac.Session(), "character_create", {"prompt": prompt})
    assert result["isError"]
    assert result["structuredContent"]["field"] == "prompt"
    assert result["content"][0]["text"] == expected


def test_swamp_knight_creates_a_knight(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real ``recipe_from_prompt`` resolves "a swamp knight" to the
    Knight family with the swamp theme dropped (Knight offers natural and
    blackened only -- see the Facts section); only ``create_character``,
    which would spawn a Blender rig, is faked, and only to capture the
    recipe it was actually handed."""
    from warlock.service import characters as svc_characters

    captured: dict[str, Any] = {}

    def fake_create_character(svc, recipe, *, name=None, prompt="", resolution=None):
        captured["recipe"] = recipe
        assert prompt == "a swamp knight"
        return {"id": "aaaaaaaaaaaa", "rig": "bbbbbbbbbbbb", "kind": "character"}

    monkeypatch.setattr(svc_characters, "create_character", fake_create_character)

    session = ac.Session()
    result = ac.call(object(), session, "character_create", {"prompt": "a swamp knight"})
    assert result["isError"] is False, result
    payload = result["structuredContent"]
    assert payload["mesh_job_id"] == "aaaaaaaaaaaa"
    assert payload["rig_job_id"] == "bbbbbbbbbbbb"
    assert session.minted["aaaaaaaaaaaa"] == "model"
    assert session.minted["bbbbbbbbbbbb"] == "rig"

    recipe = captured["recipe"]
    assert recipe["family"] == "knight"
    assert recipe.get("theme") != "swamp"
    ignored_text = " ".join(
        f"{item.get('text', '')} {item.get('reason', '')}" for item in payload["ignored"]
    ).lower()
    assert "swamp" in ignored_text


def test_character_create_without_blender_is_refused_before_a_row_exists(
    monkeypatch: pytest.MonkeyPatch, svc: Any
) -> None:
    """A real ``svc``/job-store fixture and the real ``create_character``
    door -- only ``doctor.blender_check`` is faked, to report the one
    machine state ``create_character`` itself checks for before minting
    anything (see its own "Blender, before the mesh" comment). Proves the
    refusal really does happen *before* a row exists, on this connection's
    real store, rather than trusting a mocked door's promise to have
    checked first."""
    from warlock import doctor

    monkeypatch.setattr(
        doctor,
        "blender_check",
        lambda *a, **kw: doctor.Check("Blender (rigging)", False, "not installed", fatal=True),
    )

    session = ac.Session()
    result = ac.call(svc, session, "character_create", {"family": "knight"})
    assert result["isError"]
    assert "Blender" in result["content"][0]["text"]
    assert session.minted == {}
    assert svc.store.list(limit=100) == []


def _real_sheet(
    svc: Any,
    *,
    frame_size: int = 32,
    movement_count: int = 1,
    frames_per_movement: int = 2,
    direction_count: int = 4,
    job_name: str | None = None,
    noise: bool = False,
) -> tuple[str, str, int, int]:
    """A real, on-disk character sheet -- built the way ``_q_troupe`` builds
    one (see ``tests/service/test_character_exports.py``'s own
    ``_build_sheet``, which this mirrors in miniature rather than
    importing: that module carries no ``__init__.py`` and is not a stable
    import path from here). Defaults are small and fast; a caller after a
    genuinely large atlas (a preview that must not fit without downscaling)
    passes bigger counts.

    ``noise=True`` fills the atlas with seeded random RGBA pixels instead of
    one solid colour -- see ``test_a_preview_fits_one_rpc_frame``'s own
    docstring for why a solid atlas (PNG-compressible to near nothing
    regardless of its raw dimensions) cannot exercise the preview door's own
    byte-size guard, and a noisy one can."""
    import time as time_mod

    import numpy as np
    from PIL import Image

    from warlock.kernels.rig import store
    from warlock.pipelines import charsheet
    from warlock.pipelines import sheet as sheetlib

    job_id = svc.store.create("image", job_name or "a knight", {}, stage="model")
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"fake-glb")
    svc.store.set_status(job_id, "done")
    if job_name is not None:
        svc.store.set_meta(job_id, name=job_name)

    movement_specs = [
        {
            "key": f"move_{i}",
            "loop": True,
            "duration_ms": 100,
            "frames": frames_per_movement,
            "directions": direction_count,
        }
        for i in range(movement_count)
    ]
    layout = charsheet.resolve_layout({"version": 3, "movements": movement_specs})
    table = charsheet.frame_table(layout)
    columns = layout.columns
    rows = (len(table) + columns - 1) // columns
    cells = [
        sheetlib.Cell(
            index=tc.index,
            row=tc.index // columns,
            column=tc.index % columns,
            x=(tc.index % columns) * frame_size,
            y=(tc.index // columns) * frame_size,
            pose=None,
            pose_name=tc.animation,
            yaw=tc.yaw,
            frame=tc.frame,
        )
        for tc in table
    ]
    plan = sheetlib.Plan(
        frame_size=frame_size,
        columns=columns,
        rows=rows,
        yaws=tuple(y for _n, y in layout.directions),
        elevation=30.0,
        lighting="flat",
        poses=(),
        cells=tuple(cells),
    )

    if noise:
        pixels = np.random.default_rng(0).integers(
            0, 256, size=(plan.height, plan.width, 4), dtype=np.uint8
        )
        atlas = Image.fromarray(pixels, "RGBA")
    else:
        atlas = Image.new("RGBA", (plan.width, plan.height), (10, 20, 30, 255))
    store.sheet_dir(job_dir).mkdir(parents=True, exist_ok=True)
    sheet_id = store.new_id()
    png_path = store.sheet_png_path(job_dir, sheet_id)
    atlas.save(png_path, "PNG")

    meta = sheetlib.sidecar(
        plan,
        sheet_id=sheet_id,
        source_job=job_id,
        image=png_path.name,
        created=time_mod.time(),
        name=job_name or "Test Sheet",
        animation=charsheet.animation_block(layout),
    )
    meta["troupe"] = layout.as_dict()
    store.sheet_path(job_dir, sheet_id).write_text(json.dumps(meta), "utf-8")
    return job_id, sheet_id, plan.width, plan.height


def test_a_preview_fits_one_rpc_frame(svc: Any) -> None:
    """A real sheet, previewed through the real ``character_sheet_preview``
    handler and the real ``sheet_preview_png`` door -- then the reply is
    framed exactly the way ``agent_host._serve_rpc_frame`` frames a ``call``
    reply (``{"hash": ...}`` header, the result JSON as the body, through
    ``rpc.encode_reply``) and checked against ``rpc.MAX_FRAME`` itself, not
    against this handler's own internal ``max_bytes`` budget -- proving the
    number this surface computes really does keep the framed reply under the
    one limit that matters on the wire.

    The atlas here is seeded random noise, not one solid colour: a solid
    atlas compresses to well under a hundred KB regardless of how many raw
    pixels it has, so the old version of this test proved only that a *tiny*
    PNG fits one frame -- ``sheet_preview_png``'s own ``max_bytes`` guard
    (the ``if len(data) > max_bytes: raise Invalid`` in
    ``service/characters.py``) was never actually exercised. PNG's own zlib
    step cannot shrink noise, so the atlas this test writes to disk is
    itself tens of megabytes once encoded -- genuinely too big for one frame
    before anything downscales it.
    """
    from warlock.kernels.rig import store
    from warlock.mcp import rpc as rpc_mod

    # One movement, MAX_CLIP_FRAMES (32) frames, 8 directions, 256 px cells:
    # 2048x8192 raw pixels -- 64 MiB, and (being noise) almost exactly that
    # once PNG-encoded too.
    job_id, sheet_id, width, height = _real_sheet(
        svc, frame_size=256, movement_count=1, frames_per_movement=32, direction_count=8,
        noise=True,
    )
    assert width * height * 4 > rpc_mod.MAX_FRAME  # sanity: the raw atlas alone would not fit

    # The precondition fix 2 asks for: encode the *whole* atlas -- the shape
    # an unbounded preview (no resize-to-max_side at all) would have handed
    # back -- exactly the way a real reply is framed, and prove that alone
    # already blows the one RPC frame ceiling that matters on the wire. Read
    # off disk rather than re-encoded, since ``_real_sheet`` already paid
    # that PNG-encoding cost once building the fixture.
    full_png = store.sheet_png_path(svc.job_dir(job_id), sheet_id).read_bytes()
    unbounded_meta = ac.text(json.dumps({"width": width, "height": height}))
    unbounded_reply = ac.ok(ac.image_png(full_png), unbounded_meta)
    unbounded_body = json.dumps(unbounded_reply, separators=(",", ":")).encode("utf-8")
    unbounded_framed = rpc_mod.encode_reply({"hash": "0" * 64}, unbounded_body)
    assert len(unbounded_framed) > rpc_mod.MAX_FRAME

    result = ac.call(
        svc,
        ac.Session(),
        "character_sheet_preview",
        {"job_id": job_id, "sheet_id": sheet_id, "max_side": 2048},
    )
    assert result["isError"] is False, result
    assert "structuredContent" not in result
    kinds = {c["type"] for c in result["content"]}
    assert kinds == {"image", "text"}

    body = json.dumps(result, separators=(",", ":")).encode("utf-8")
    framed = rpc_mod.encode_reply({"hash": "0" * 64}, body)
    assert len(framed) <= rpc_mod.MAX_FRAME

    # And the bound actually acted -- this is not merely a small preview by
    # coincidence: the returned image is smaller than the full atlas it was
    # cropped from in both dimensions, proving the max_side resize (or, had
    # this atlas been chosen to still miss the max_bytes budget afterward,
    # the door's own refusal naming ``max_side`` -- see
    # ``sheet_preview_png``'s own ``if len(data) > max_bytes`` branch) ran
    # rather than the raw atlas passing straight through.
    meta = json.loads(next(c["text"] for c in result["content"] if c["type"] == "text"))
    assert meta["width"] < width
    assert meta["height"] < height


def test_character_export_refuses_without_an_export_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Config:
        export_dir = None

    class _Svc:
        config = _Config()

    result = ac.call(
        _Svc(), ac.Session(), "character_export", {"job_id": "0" * 12, "format": "animated_glb"}
    )
    assert result["isError"]
    assert "WARLOCK_EXPORT_DIR" in result["content"][0]["text"]


def test_an_agent_export_lands_in_a_folder_named_for_its_ids(
    monkeypatch: pytest.MonkeyPatch, svc: Any, tmp_path: Path
) -> None:
    """Fix 2's own scope rule, proved against the real, now-landed
    ``characters.agent_export_stem``: ``character_export`` must compute its
    stem from the asset's own ids and pass it through explicitly, rather
    than let a door fall back to formatting one from a job's own (agent- or
    user-chosen) name -- the one thing that would let a second, unrelated
    export collide with (or be mistaken for) one this agent already made.

    Only ``export.run_character_export`` itself is faked here, to capture
    exactly what this module passed it without needing a Blender-baked
    ``animated.glb`` for the mesh-only format's own sake (the other three
    tests in this file exercise the real writing doors end to end).
    ``agent_export_stem`` runs for real -- it is cheap (one job read) and is
    the exact function this fix is about.
    """
    from warlock.service import export as svc_export

    svc.config.export_dir = tmp_path
    job_id, sheet_id, _w, _h = _real_sheet(svc, job_name="Ranger")

    captured: dict[str, Any] = {}

    def fake_run_character_export(svc_arg, key, jid, sid=None, *, stem=None):
        captured["args"] = (key, jid, sid, stem)
        return {"copied": 1, "dir": str(tmp_path), "degraded": []}

    monkeypatch.setattr(svc_export, "run_character_export", fake_run_character_export)

    # A mesh-only format (needs_sheet is False): a caller-supplied sheet_id
    # must never reach agent_export_stem, and so never fold into the stem,
    # even though the schema still accepts sheet_id for the two
    # sheet-shaped formats' sake.
    result = ac.call(
        svc,
        ac.Session(),
        "character_export",
        {"job_id": job_id, "format": "animated_glb", "sheet_id": sheet_id},
    )
    assert result["isError"] is False, result
    key, jid, sid, stem = captured["args"]
    assert (key, jid, sid) == ("animated_glb", job_id, sheet_id)
    assert stem == f"Ranger-{job_id}"
    assert sheet_id not in stem

    # A sheet-shaped format (needs_sheet is True): the same sheet_id now
    # belongs in the stem too -- the folder-collision guard fix 2 is named
    # for.
    result2 = ac.call(
        svc,
        ac.Session(),
        "character_export",
        {"job_id": job_id, "format": "sheet_package", "sheet_id": sheet_id},
    )
    assert result2["isError"] is False, result2
    key2, jid2, sid2, stem2 = captured["args"]
    assert (key2, jid2, sid2) == ("sheet_package", job_id, sheet_id)
    assert stem2 == f"Ranger-{job_id}-{sheet_id}"


def test_a_hostile_job_name_cannot_escape_the_export_dir(svc: Any, tmp_path: Path) -> None:
    """A real job named ``../../evil``, ``C:\\x`` or ``CON`` -- and a real
    rendered sheet, exported through the real, now-landed
    ``agent_export_stem``/``run_character_export``/``export_package``
    chain end to end (``sheet_package`` needs no Blender, so nothing here
    is mocked). ``characters._package_stem`` already turns a hostile name
    into a safe fragment (or drops it to the sheet id outright for a
    reserved DOS device name); ``agent_export_stem`` appending the job and
    sheet ids on top is what proves two different, identically-hostilely-
    named characters could never collide even if ``_package_stem`` alone
    let them.
    """
    for hostile in ("../../evil", "C:\\x", "CON"):
        svc.config.export_dir = tmp_path
        job_id, sheet_id, _w, _h = _real_sheet(svc, job_name=hostile)

        result = ac.call(
            svc,
            ac.Session(),
            "character_export",
            {"job_id": job_id, "format": "sheet_package", "sheet_id": sheet_id},
        )
        assert result["isError"] is False, (hostile, result)
        payload = result["structuredContent"]
        assert payload["paths"], (hostile, payload)
        for path_str in payload["paths"]:
            path = Path(path_str)
            resolved = path.resolve()
            assert resolved.parent == tmp_path.resolve(), (hostile, path)
            assert ".." not in path.name
            assert path.name.split(".", 1)[0].upper() not in {"CON", "NUL", "PRN", "AUX"}
            assert job_id in path.name
            assert path.exists()
