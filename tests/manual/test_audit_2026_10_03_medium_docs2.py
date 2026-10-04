"""The 2026-10-03 audit's Medium documentation findings docs-20 and docs-23 through docs-29.

Each test's name is the claim. They read the shipped chapters as data and, where the
chapter describes a thing the code can answer for itself, ask the code too, so a chapter
and its subject cannot drift apart silently the way these seven did.
"""

import inspect
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANUAL = ROOT / "docs" / "manual"


def _chapter(name: str) -> str:
    return (MANUAL / name).read_text(encoding="utf-8")


def _flat(text: str) -> str:
    """Whitespace-normalised, so a sentence that wraps across lines still matches."""
    return " ".join(text.split())


def _section(text: str, heading: str) -> str:
    """The body under ``## heading``, up to the next ``## `` heading."""
    match = re.search(rf"^## {re.escape(heading)}\s*$(.*?)(?=^## |\Z)", text, re.M | re.S)
    assert match, f"no '## {heading}' section"
    return match.group(1)


def test_the_manual_cites_no_path_under_dev():
    """docs-20: ``dev/`` is gitignored and its measurements are lost, so a chapter that
    cites ``dev/measurements/...`` or ``dev/TODO.md`` sends a reader to a dead end."""
    pattern = re.compile(r"(?<![\w.])dev[/\\]\w")
    hits = [
        f"{path.name}:{number}: {line.strip()[:80]}"
        for path in sorted(MANUAL.glob("*.md"))
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if pattern.search(line)
    ]
    assert not hits, "the shipped manual cites a path under dev/:\n" + "\n".join(hits)
    # And the sentence that used to cite it now says the document is gone.
    assert "document that is now lost" in _flat(_chapter("23-generating-meshes.md"))


def test_manual_42_and_43_do_not_describe_a_home_health_row():
    """docs-23: Home draws setup, queue and review only; chapters 03 and 21 were fixed
    for this and 42 and 43 were not."""
    from realmspinner.studio.modes.home.ui.panes import landing

    assert "health" not in landing.HOME_STATUS, "sanity: Home grew a health row"
    for name in ("42-app-settings.md", "43-troubleshooting.md"):
        text = _flat(_chapter(name)).lower()
        assert "health row" not in text.replace("draws no health row", ""), (
            f"{name} describes a Home health row; landing.HOME_STATUS = {landing.HOME_STATUS}"
        )
        assert "things need attention" not in text, f"{name} quotes a Home health row's text"
    assert "Generation is not set up yet" in _chapter("43-troubleshooting.md")


def test_manual_46_document_mode_paragraph_names_mode_manifest():
    """docs-24: four of the five tables the paragraph told a contributor to edit are
    derived from one ``mode_manifest.DOC_MODES`` row; only ``recents.KINDS`` is not."""
    from realmspinner.studio import mode_manifest, palette, recents

    text = _flat(_chapter("46-extending.md"))
    start = text.index("**A document mode costs more.**")
    paragraph = text[start : text.index("**The workspace.**", start)]
    assert "mode_manifest" in paragraph and "DOC_MODES" in paragraph
    assert "recents.KINDS" in paragraph
    assert "joins `docmodes.DOC_MODES`" not in paragraph

    # The claims the paragraph makes about the code.
    assert mode_manifest.export_table() == palette._DOC_MODES
    assert set(recents.KINDS) == set(mode_manifest.opener_table()), (
        "recents.KINDS is the one hand-written tuple the paragraph names; it must agree "
        "with the manifest's kinds that have an opener"
    )


def test_manual_44_lists_every_worker_module_and_the_agent_host_thread():
    """docs-25: the subprocess table omitted the music, separation, LoRA and recipe
    workers and the chapter had no agent-host thread. Both directions: every
    ``pipelines/*_worker.py`` has a row, and no row names a worker that is gone."""
    rows = {
        "blender_worker": "| Blender |",
        "fetch_worker": "| The fetch worker |",
        "lora_train_worker": "| The LoRA trainer |",
        "matting_worker": "| BiRefNet matting |",
        "music_worker": "| The music worker |",
        "pack_worker": "| The pack worker |",
        "recipe_worker": "| The recipe worker |",
        "separation_worker": "| The separation worker |",
        "text2image_worker": "| The image model |",
        "update_worker": "| The update worker |",
    }
    on_disk = {p.stem for p in (ROOT / "src" / "realmspinner" / "pipelines").glob("*_worker.py")}
    assert on_disk == set(rows), (
        f"a worker module was added or removed: on disk {sorted(on_disk)}; add or drop its "
        "row in docs/manual/44-architecture.md and in this test"
    )
    text = _chapter("44-architecture.md")
    missing = [mod for mod, label in rows.items() if label not in text]
    assert not missing, f"chapter 44's subprocess table has no row for {missing}"

    from realmspinner.studio import agent_host

    assert "realmspinner-agent-host" in inspect.getsource(agent_host)
    assert "realmspinner-agent-host" in text, "chapter 44 never names the agent-host thread"
    assert "two workers" in _flat(text), "chapter 44 omits the agent host's second TaskRunner pool"
    assert agent_host.SERVICE_WORKERS == 2
    assert "A vendored native binary; it was never Python" not in text


def test_manual_29_does_not_place_the_exports_on_the_timeline_strip():
    """docs-26: the exports and their settings moved to the Export block and the File
    menu (2026-09-05); chapter 28 says so and chapter 29 still put them on the strip."""
    ch29 = _flat(_chapter("29-inker-animation.md"))
    for stale in (
        "the three exports",
        "export magnification",
        "The export row also offers",
    ):
        assert stale not in ch29, f"chapter 29 still says {stale!r}"
    assert "Export** block" in ch29 and "File** menu" in ch29
    ch28 = _flat(_chapter("28-inker.md"))
    assert "keeps only the two switches" in ch28  # the sentence chapter 29 now agrees with

    from realmspinner.studio.modes.inker.ui.panes import timeline

    source = inspect.getsource(timeline._view_toggles)
    assert '"Onion"' in source and '"Thumbs"' in source


def test_manual_25_agrees_with_manual_23_on_how_rig_controls_look_without_blender():
    """docs-27: chapter 25 said the rig controls are hidden, never greyed; the Mesh
    stage's checkbox and the stage rail's segments are greyed with a reason."""
    ch25 = _flat(_chapter("25-rigging-and-posing.md"))
    assert "hides the rig controls entirely" not in ch25
    assert "rather than greying them out" not in ch25
    assert "greyed" in ch25 and "Rig when the mesh lands" in ch25
    assert "greyed" in _flat(_chapter("23-generating-meshes.md"))

    from realmspinner.studio.modes.create.ui import stages
    from realmspinner.studio.modes.create.ui.panes import settings_3d

    # The code the paragraph describes: the checkbox is drawn, disabled, with the
    # rail's own sentence; the rail's reason names Blender.
    assert "enabled=blocked is None" in inspect.getsource(settings_3d._rig)
    ctx = type("Ctx", (), {"rigging_available": False})()
    assert "Blender" in stages.blender_reason("rig", ctx)
    assert "Blender" in stages.blender_reason("pose", ctx)


def test_manual_20_scopes_the_file_panel_promise_to_document_modes():
    """docs-28: Poser and Muse draw no file panel, four file verbs or document tabs."""
    from realmspinner.studio import mode_manifest

    section = _flat(_section(_chapter("20-overview.md"), "What is the same in every workspace"))
    assert "Whichever one is open:" not in section
    for named in ("Inker", "Clay", "Mason", "Plotter", "Packwright", "Sirens", "Poser", "Muse"):
        assert named in section, f"the scoped paragraph never names {named}"
    assert "pose library" in section and "job row" in section

    keys = {m.key for m in mode_manifest.DOC_MODES}
    assert "muse" not in keys, "Muse became a document mode: the overview's scoping is stale"
    # Poser is in the manifest for its journal and palette rows but owns no persist and no
    # opener, which is why it has no file panel to promise.
    poser = mode_manifest.by_key("poser")
    assert poser is not None and poser.opener is None


def test_manual_23_documents_the_finishing_choice_beside_budget():
    """docs-29: the Finishing combo and the Mesh quality finishing lines shipped with no
    chapter text, and Game-ready was still described as always a voxel pass and unwrap."""
    from realmspinner.pipelines import remesh
    from realmspinner.studio.modes.create.ui.panes import settings_3d

    text = _chapter("23-generating-meshes.md")
    budget = _flat(_section(text, "Mesh parameters"))
    source = inspect.getsource(settings_3d._budget)
    for label in ("Finishing", "Preserve shape", "Repair and close holes"):
        assert f'"{label}"' in source, f"the pane no longer draws {label!r}"
        assert label in budget, f"chapter 23's Budget text never names {label!r}"
    assert "Game-ready remeshes the surface itself inside the model job: a voxel" not in budget
    assert "does not close holes" in budget

    flat = _flat(text)
    lines = " ".join(
        remesh.finishing_lines(
            {
                "triangles": 5000,
                "requested": 5000,
                "geometry": {
                    "measured": True,
                    "worst_lost_coverage": 0.02,
                    "worst_added_coverage": 0.0,
                    "views": [{"closed_openings": 0.0}],
                },
                "tiercheck": {"ok": True, "failures": []},
            }
        )
    )
    for phrase in ("Finished mesh:", "Silhouette coverage lost", "opening pixels filled"):
        assert phrase in lines, f"finishing_lines no longer prints {phrase!r}"
        assert phrase in flat, f"chapter 23 never mentions the line {phrase!r}"
