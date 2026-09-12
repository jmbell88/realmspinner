# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Warlock Studio: a fully offline, single-process desktop app for making game assets. Text or image prompt → SDXL 1.0 reference PNG (`pipelines/text2image_worker.py`, a child process) → `trellis-server.exe` (the TRELLIS.2 native server, not the Python TRELLIS package) → textured GLB. **The engine is a download, not a vendored binary**: since 2026-09-10 it is two Settings → Models rows, `engine:trellis_runtime` (the exe and its CUDA libraries) and `engine:trellis_gguf` (the weights), and `vendor/trellis/` is only the source-checkout fallback — a downloaded copy wins over it and `WARLOCK_TRELLIS_EXE` beats both. One pygame window, one ModernGL context, imgui panels drawn through it (`studio/imgui_backend.py`). No HTTP, no browser. Windows-first; CI is Windows-only.

Fourteen modes, and `src/warlock/studio/modes.py` is the authoritative list: Home, Library, Create, then the workspaces Inker (pixel art), Clay (mesh primitives/booleans), Mason (3D scene editor: place assets and primitives, light and export to engines), Poser (rig + clips), Troupe (3D → sprite sheet, the one Experimental mode), Plotter (tile maps → Tiled), Packwright (atlas packing), Muse (ACE-Step music generation, output is a job row not a document), Sirens (chiptune tracker), then Review and Settings. The Manual, the guided tour (`studio/tour/`) and Flourish (procedural VFX inside Inker, `studio/inker/flourish/`) are deliberately *not* modes.

## Commands

```powershell
uv sync --extra studio --extra text2image --extra rig --extra music   # dev group installs by default; a bare `uv sync` prunes the extras and breaks ~10 test files at collection
uv run warlock            # the app
uv run warlock doctor     # deps, weights, config, which native kernels are live
uv run warlock mcp        # the MCP bridge an agent's client spawns; talks to a running app over a named pipe
uv run warlock sweep --image <ref.png> --bands auto,4,8   # mesh quality across trellis --band values
uv run pytest             # default lane: -n 8 --dist loadfile, gpu+perf excluded, ~2 min for 20k+ tests
uv run pytest tests/test_x.py::test_name -n 0     # single test: skip worker startup
uv run pytest -m gpu -n 0   # real card + weights; xdist is refused for this lane (N workers = N 7 GB loads)
uv run pytest -m perf -n 0  # wall-clock budgets; meaningless under contention
uv run ruff check .
uv run python scripts/preflight.py   # release gate; run before a PR (`--fast` is what CI runs)
pwsh scripts\rebuild.ps1  # all of Windows CI locally, then the installer, then prunes dist\; -Native also rebuilds vendor\warlockc\warlockc.dll (default leaves it alone)
pwsh native\build.ps1     # optional C kernels → vendor/warlockc/warlockc.dll (gitignored); WARLOCK_NATIVE=0 forces numpy
```

- Python 3.13 is required (`bpy` ships 3.13 wheels only); the `rig` extra silently installs nothing elsewhere.
- **Never edit `src/` while the suite runs** — several tests read module source.
- `--dist loadfile` is load-bearing, not tuning: it keeps at most one imgui context alive per process and preserves module-level cache couplings. Don't switch to `--dist load` or `-n auto` (measured slower and OOM-prone; the reasoning is in `pyproject.toml`).
- `tests/conftest.py` pins `WARLOCK_HOME` to a throwaway dir and pins `memlog.system_memory`, so no test sees the real `~/.warlock` model library or depends on machine load. The gpu lane is exempt on purpose.

## Where the truth lives

- **`docs/INVARIANTS.md`** is the authoritative record of hard constraints *and the measured incident behind each*. Read the section for a subsystem before touching it; update it when an invariant changes. The bullets below are only the catastrophic-mistake preventers.
- **`docs/measurements/`** — a constant the stored corpus is keyed on (`trellis_band`, `SEAM_MAX`, the grade scale…) gets a dated document there *before* it changes.
- **`docs/COMPAT.md`** — the two interop ledgers (Plotter↔Tiled, Inker↔Aseprite) in one file. Only the Tiled rows are executable (`tests/plotter/test_compat_matrix.py` parses them as data).
- **`docs/MODELS.md`** — optional weights, their licences, and the SDXL recipe registry.
- **`TODO.md`** is the one plan file and it is not a roadmap: only what a human must do (hardware validation, art direction, design decisions) plus fully-specified, deliberately unstarted work. If an item there could be built, build it and strike it out. Never cite it from `src/` or `scripts/`; finished plan files are deleted, not ticked (`tests/test_ux_todo_fixes.py` sweeps dead filenames out).
- **`docs/audit-*.md`** — a dated, disposable ledger from one slice audit. A finding is struck the day it is built (`~~claim~~ **Built YYYY-MM-DD:** what was done, and the test`), a human-only one moves to `TODO.md`, and the file is deleted rather than ticked when nothing is left. Same rule as `TODO.md`: never cite it from `src/` or `scripts/`.
- Style: comments explain *why*, naming the incident that motivated a guard. A regression test's name is the claim, and it must fail against the unfixed code. Manual changes ship in the same commit as the behaviour change.

## Architecture in one pass

- **Layers.** `service/` is the only business-logic layer; panes (`studio/panes/`) and tests both call it, never each other's internals. Refusals raise `service.errors` exceptions carrying a `field` so the UI can point at a control. `queue.py` schedules jobs; `db.py`'s `JobStore` is one sqlite connection behind an RLock, and partial `params` writes go through `merge_params`. Job kinds live in `_q_*.py` modules; adding one is a sweep of every stage-keyed table (an invariant, test-gated).
- **Three threads.** The pygame frame loop never blocks; the asyncio `Worker` lives on the `warlock-loop` thread; everything blocking goes through `TaskRunner` (`studio/tasks.py`). A fourth, `warlock-agent-host`, exists only once the MCP bridge is switched on. One GL context — textures must be registered/forgotten with the imgui backend.
- **Heavy work is always a child process, inside the `winjob` kill-on-close job**: text2image (`t2i_client` is the in-app handle; its stdin reader must never leave a read pending or the child deadlocks on its next native import), matting (`matting_worker.py`), Blender (`blender_worker.py` is the only module that imports `bpy`), music (`music_worker.py`, `separation_worker.py`), LoRA training, doctor's load probe, Flourish's text model (`recipe_worker.py`) and `gltfpack` (`optimize.py`). `tests/test_vram.py::test_every_subprocess_spawn_is_in_the_kill_on_close_job` walks the package by AST: any `subprocess.Popen`/`run` whose own function does not also call `winjob.assign` fails it.
- **Offline.** `HF_HUB_OFFLINE=1` is set in `warlock/__init__.py` before anything imports. There are three user-initiated exceptions, each its own subprocess: `fetch_worker` (model weights), `pack_worker` (dependency packs, Settings → Packs) and `update_worker` (release-feed check and installer download, Settings → Advanced/Updates).
- **VRAM.** Admission at the door (`service.validation.check_vram`/`check_weights`), re-checked at dispatch; coexist-vs-exclusive handoff is `queue._needs_handoff`; teardown is `unload()` never `trim()`; drop every reference before `_reclaim`.
- **Artifacts.** `source.glb` is the reconstruction; `model.glb` is derived (optimize, then normalize; grounding always runs); every other export is a pure function of `model.glb`. Every write onto a served name is staged to a temp and `os.replace`d, never in place. `DERIVED_PARAMS` strips worker-recorded values on rerun/promotion; `VECTOR_PARAMS` is an allowlist in `vectors.py` (queue.py may not import `service`). A stage that publishes onto a served name must also join `PUBLISHERS` in `tests/test_job_durability.py` — a hand list that fails open, and has gone stale twice.
- **Headless editor packages** under `studio/` — `inker/` (and `inker/flourish/`, `inker/walk/`), `clay/`, `mason/`, `plotter/`, `packwright/`, `sirens/`, `troupe/`, `muse/` — import no imgui, moderngl, pygame or `service` (`sirens/` and `muse/` also ban scipy so renders are byte-identical across a `uv sync`). Import-pinning tests enforce the exact outward set; adding an import means updating the pin deliberately. `mason/`'s pin derives its sibling ban from `tests/_pure_packages.py` rather than listing it, so the next pure package enrols itself; the other six still carry hand lists that disagree with each other. `studio/sirens_audio.py` is the only module that touches `pygame.mixer`. Undo is addressed by uid, never index.
- **Agents** drive Clay over MCP: `warlock/mcp/` is the pure protocol/pipe/bridge (stdlib only, no studio), `studio/agent_host.py` is the listener thread plus the frame-thread call queue drained in `App.frame`, and `studio/agent_clay.py` is the tool surface — **derived** from `primitives.GENERATORS`, `presets.ASSEMBLIES` and `clay_ops.OPS`, never hand-listed. Inbound only: no model, no inference, no socket, `HF_HUB_OFFLINE` untouched. Off until switched on in Settings; an agent gets its own Clay tab and can address no other.
- **Crash recovery** is `studio/journal.py`, one mechanism for every document kind: payload plus a `.meta.json` sidecar written *last* as the completion gate.
- **Native kernels** (`native/*.c`) are optimisations with a reference: the numpy fallback is never deleted and the bar is bit-identical parity (`/fp:precise`, no FMA), except `contours.c` whose bar is the unit-edge set.
- **Manual** (`studio/manual/`, `docs/manual/`): a chapter's number decides its order and part, so adding one is a renumbering, gated by `tests/manual/` in both directions. Chapters 01–19 are reserved for the tutorial series.
- **Tour** (`studio/tour/`): pure data with no outward imports, drawn from `_overlays`; steps wait on *named* conditions and never act for the reader.
- **Create** is one staged mode: the Reference stage is a command bar (`studio/create_brief.py`, *what* to make) over a recipe column (`panes/settings_2d.py`, *how*), and no control appears in both.

## Tooling notes

- `scripts/exercise_mode.py` (also the `/exercise-mode` skill) drives every control in a mode through the real input path and screenshots each press; `scripts/screenshot_modes.py` captures all modes.
- **Five skills** under `.claude/skills/` (local to this machine — `/.claude/` is gitignored): `/exercise-mode` (above), `/warlock-audit` (fan explorers over a slice, merge into a dated `docs/audit-*.md`), `/warlock-land` (land a finished change: prove the regression tests fail first, pay what it owes beyond `src/`, run the gate once, commit), `/warlock-sweep` (walk a repeated multi-file addition — a job kind, a mode, a chapter — across every site it must touch), `/warlock-sitting` (produce the artefacts a `TODO.md` entry needs a human to judge).
- **A release is a five-file version bump** — `pyproject.toml`, `src/warlock/__init__.py`, `CHANGELOG.md`, `INSTALL.md`, `uv.lock` — and `/warlock-land --release` is what walks them.
- Fable and Opus plan, orchestrate and review; Sonnet subagents (`model: "sonnet"`) do the implementing, unless the user says otherwise.
- Subagents working in parallel must never `git stash`, `checkout` or `reset` — a stash from one fixer reverts the shared tree for the others.

### Searching this codebase

**Reach for the Synergy MCP before Grep/Read for any question that spans files.** It indexes the repo (~1,400 files, ~36k symbols); the server is in the tracked `.mcp.json` and the index sits in `.synergy/` (gitignored). `repo_search` (literal/regex) finds a name anywhere in the tree, docs included; `kg_file_summary` gives a symbol table for a big module instead of reading it; `repo_context`/`repo_references` walk callers and dependency chains; `kg_find`/`kg_definition` resolve one symbol. Say so in subagent briefs too, or they default to ripgrep.

Its limits here, so you do not fight it. **No semantic model is installed**, so `semantic`/`hybrid` modes are unavailable — use `literal` and `regex`. `repo_read` truncates hard — it answered a 31-line range with 6 lines and an `expand` pointer — so once you know the file, plain `Read` is better. The transitive-reach tools fan out uselessly at this repo's size: `kg_tests_covering("warlock.winjob.assign")` never reaches `tests/test_winjob.py` and returns 60 distance-3 hits from unrelated suites instead; `kg_impact_of` ran past four minutes on a five-file diff and had to be killed. Neither is worth a second try here. Every response carries a coverage preamble, so for a single lookup in a file you can already name, Grep is cheaper. **Judge the graph's freshness by `repo_status`'s `git_head_at_scan`, never by its `state`.** A filesystem watcher keeps the graph on HEAD and reparses a file seconds after you save it (`kg_watch_status` says whether it is alive), so `kg_*` answers are normally current — but `repo_*` tools report `state: "indexing", progress 0/N` even when the graph is fully refreshed, while `kg_*` tools report `ready, N/N` off the same refresh. That contradiction is cosmetic: `status: incomplete` is not a reason to abandon the tool, which is what it cost one explorer on 2026-09-12. If `git_head_at_scan` really is behind HEAD, run `kg_refresh`. The `.mcp.json` command is an absolute machine-local path, so the server will not connect on another machine: fall back to Grep/Read and say so once, rather than retrying.

Provisional, while the habit settles: at the end of a session that explored the code, say whether Synergy actually saved anything.
