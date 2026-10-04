"""The 2026-10-04 audit's Mesh and history findings: create-11, 12, 17, 18.

Every test name is the finding's claim, and each was run against the unfixed
code first.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from realmspinner import studio
from realmspinner.bench import findings as findings_lib
from realmspinner.service.validation import not_done_message
from realmspinner.studio.modes.create.engine import mesh as create_mesh
from realmspinner.studio.modes.create.ui import workspace as generation_workspace
from realmspinner.studio.modes.create.ui.panes import settings_3d
from realmspinner.studio.state import DEFAULT_FORM_3D, AppState

# --- create-11: one fact, one sentence --------------------------------------


@pytest.mark.parametrize("status", ("queued", "running", "error", "cancelled"))
def test_mesh_validate_uses_the_shared_not_done_sentence(status):
    """``mesh.validate`` said "That reference is error." where ``stages.available``
    says what the status *means* for the same row at the same door."""
    source = {"id": "r", "status": status, "files": ["input.png"]}
    found = create_mesh.validate(source)
    assert [str(p) for p in found] == [not_done_message("That reference", status)]
    assert str(found[0]) != f"That reference is {status}."


# --- create-12: the findings file is read once a frame ----------------------

_SOURCE = {
    "id": "ref-1",
    "name": "mossy well",
    "status": "done",
    "stage": "reference",
    "files": ["input.png"],
    "prompt": "a mossy well",
}


def _mesh_ctx(bench_dir):
    state = AppState(mode="create")
    state.create.stage = "mesh"
    state.source_job = "ref-1"
    state.form_3d = dict(DEFAULT_FORM_3D)
    return SimpleNamespace(
        state=state,
        cache=SimpleNamespace(get={"ref-1": _SOURCE}.get, jobs=[_SOURCE], active=None),
        job=lambda: None,
        model_rows=[],
        submit=lambda *a, **k: True,
        busy=lambda _k: False,
        svc=SimpleNamespace(config=SimpleNamespace(bench_dir=bench_dir)),
        guidance={"fields": {"platform": []}, "bg_removal": []},
        rigging_available=False,
        toast=lambda *a, **k: None,
    )


def test_mesh_column_reads_findings_json_once_per_frame(tmp_path, monkeypatch):
    """Every hinted control called ``findings_lib.load`` twice (``findings_hint``
    and ``_best_value_offer``) -- about 25-30 ``stat()`` calls a frame. The
    column loads once and hands the document down, as the 2D column does."""
    from _ui_context import imgui_context

    from realmspinner.studio import forms, probe

    (tmp_path / "findings.json").write_text(
        json.dumps({"version": 5, "generated": "x", "params": {}}), encoding="utf-8"
    )
    findings_lib._CACHE.clear()
    loads: list[int] = []
    real = findings_lib.load
    monkeypatch.setattr(findings_lib, "load", lambda *a, **k: (loads.append(1), real(*a, **k))[1])
    # Open the Engine disclosure so its seven hinted controls draw as well.
    monkeypatch.setattr(settings_3d.controls, "collapsing_header", lambda *a, **k: True)
    ctx = _mesh_ctx(tmp_path)

    with imgui_context(monkeypatch) as imgui:
        probe.begin_frame()
        imgui.new_frame()
        imgui.set_next_window_size((400.0, 900.0))
        imgui.begin("host")
        try:
            with forms.Form("create-3d", errors={}) as form_ui:
                settings_3d._draw_form(ctx, form_ui, "Mesh resolution", _SOURCE)
        finally:
            imgui.end()
            imgui.end_frame()
            imgui.render()

    assert len(loads) == 1, f"the Mesh column loaded findings.json {len(loads)} times in one frame"


# --- create-17: stale references, and a private reach -----------------------

_STUDIO = Path(studio.__file__).parent
_SRC = _STUDIO.parent.parent


def _module_of(path: Path) -> list[str]:
    parts = list(path.relative_to(_SRC).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return parts


def _imported_modules(tree: ast.AST, here: list[str], is_package: bool) -> dict[str, Path]:
    """``{local alias: the studio module file it names}`` for every import, at any scope."""
    found: dict[str, Path] = {}
    package = here if is_package else here[:-1]

    def module_file(parts: list[str]) -> Path | None:
        base = _SRC.joinpath(*parts)
        if base.with_suffix(".py").is_file():
            return base.with_suffix(".py")
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                anchor = package[: len(package) - (node.level - 1)]
                base = anchor + (node.module.split(".") if node.module else [])
            else:
                base = (node.module or "").split(".")
            for alias in node.names:
                target = module_file([*base, alias.name])
                if target is not None:
                    found[alias.asname or alias.name] = target
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    target = module_file(alias.name.split("."))
                    if target is not None:
                        found[alias.asname] = target
    return found


def _private_reaches(target_root: Path) -> list[str]:
    """Every ``alias._private`` in the studio tree where ``alias`` is a module
    under ``target_root``."""
    out: list[str] = []
    for path in sorted(_STUDIO.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        aliases = _imported_modules(tree, _module_of(path), path.name == "__init__.py")
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in aliases
                and node.attr.startswith("_")
                and not node.attr.startswith("__")
                and aliases[node.value.id] != path
                and target_root in aliases[node.value.id].parents
            ):
                rel = path.relative_to(_SRC).as_posix()
                out.append(f"{rel}:{node.lineno} {node.value.id}.{node.attr}")
    return out


def test_no_pane_reaches_another_panes_private_helpers():
    """``tests/test_layering.py`` sees module imports, not attribute reach, so
    ``workspace._candidate_grid`` called ``candidates_panel._grades`` and
    ``_nudge_text`` unseen. Scoped to the shared panes (``studio/panes``) as the
    *target*, reached from anywhere in the studio tree: that is the whole of the
    finding and it needs no allowlist. Mode-internal reaches between a mode's own
    modules (``inker_mode._settle``, ``brief._TYPE_HINTS``...) are a larger,
    separate question and are deliberately not swept here."""
    assert _private_reaches(_STUDIO / "panes") == []


def test_the_private_reach_sweep_can_fail(tmp_path):
    """The sweep above is only worth having if it finds what it is for."""
    probe = _STUDIO / "modes" / "create" / "ui" / "workspace.py"
    tree = ast.parse("from ....panes import candidates_panel\ncandidates_panel._grades()\n")
    aliases = _imported_modules(tree, _module_of(probe), False)
    assert aliases == {"candidates_panel": _STUDIO / "panes" / "candidates_panel.py"}


def test_candidate_docstrings_name_code_that_exists():
    """Four comments named ``candidates_panel.draw`` / ``._member`` (neither
    exists) and a ``panes/settings_3d`` that is ``modes/create/ui/panes/``."""
    from realmspinner.studio import candidates, matte_preview
    from realmspinner.studio.panes import candidates_panel

    for name in ("draw", "_member"):
        assert not hasattr(candidates_panel, name)
        for path in (
            Path(candidates.__file__),
            Path(generation_workspace.__file__),
            Path(matte_preview.__file__),
        ):
            assert f"candidates_panel.{name}" not in path.read_text(encoding="utf-8"), (
                path.name,
                name,
            )
    assert "``panes/settings_3d``" not in Path(matte_preview.__file__).read_text(encoding="utf-8")
    assert not (_STUDIO / "panes" / "settings_3d.py").exists()


# --- create-18: the creations history draws only what is on screen ----------


def test_creations_history_measures_only_visible_rows(monkeypatch):
    """The history drew one button per creation per frame and measured each label
    with ``fit_text``; "Load older creations" can widen it to 5000 rows."""
    from _ui_context import imgui_context

    from realmspinner.studio import probe

    count = 2000
    ordered = [
        (
            f"key{i:04d}",
            [{"id": f"j{i:04d}", "name": f"creation number {i} " + "x" * 80, "status": "done"}],
        )
        for i in range(count)
    ]
    monkeypatch.setattr(generation_workspace.session, "index", lambda ctx: object())
    monkeypatch.setattr(generation_workspace.families, "ordered_creations", lambda idx: ordered)
    measured: list[str] = []
    real_fit = generation_workspace.widgets.fit_text
    monkeypatch.setattr(
        generation_workspace.widgets,
        "fit_text",
        lambda text, width: (measured.append(text), real_fit(text, width))[1],
    )
    ctx = SimpleNamespace(cache=SimpleNamespace(can_load_more=lambda: False))
    monkeypatch.setattr(generation_workspace, "_FIT_MEMO", {})
    frames: list[list[str]] = []

    with imgui_context(monkeypatch) as imgui:
        for _ in range(3):  # the clipper measures its first row, then settles
            measured.clear()
            probe.begin_frame()
            imgui.new_frame()
            imgui.set_next_window_size((300.0, 600.0))
            imgui.begin("history")
            try:
                generation_workspace.history(ctx)
            finally:
                imgui.end()
                imgui.end_frame()
                imgui.render()
            frames.append(list(measured))

    first = frames[0]
    assert 0 < len(first) < 80, f"measured {len(first)} of {count} labels in one frame"
    assert first[0].startswith("creation number 0 ")
    # The fitted labels are remembered: a frame that shows the same rows measures none.
    assert frames[-1] == []
