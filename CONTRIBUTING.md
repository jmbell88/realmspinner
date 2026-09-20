# Contributing

Thanks for looking. A few things about this codebase will save you time.

## Getting set up

```powershell
uv sync --extra studio --extra text2image --extra rig --extra music
uv run pytest
uv run ruff check .
```

**Pass all four extras** (`studio`, `text2image`, `rig`, `music`). A bare `uv sync` *prunes* them
and breaks about ten test files at collection. `rig` needs a Python 3.13 environment or it silently
installs nothing.

The suite is ~2 min for 20k+ tests, parallel by default -- a range rather than an
exact count, because the 2026-09-06 audit's finding docs-11 found the count already
13% stale (a hand-kept exact number drifts the way the rest of this section warns
about). Three lanes are excluded from the default run and each is opt-in:

| Lane | Command | When |
|---|---|---|
| GPU | `uv run pytest -m gpu -n 0` | Before changing model loading, VRAM accounting or conditioning. Serial is enforced -- N workers means N simultaneous 7 GB loads onto one card. |
| Performance | `uv run pytest -m perf -n 0` | Wall-clock budgets; meaningless under contention. |
| One test | `uv run pytest tests/x.py::y -n 0` | Quicker than paying for worker startup. |

**Never edit `src/` while the suite is running.** Several tests read module
source, and you will get failures that have nothing to do with your change.

## Before you write anything

This codebase runs on a set of hard constraints, most of which exist because
something specific went wrong. The maintainer keeps the authoritative record
of each one, with the measured reasoning behind it, in a local development
ledger that isn't part of this public repo -- ask in your PR or issue if
you're unsure whether a change touches one.

The ones that most often surprise people:

- **The app is fully offline.** `HF_HUB_OFFLINE=1` is set before anything
  imports. Three user-initiated subprocesses are the only code that reaches the
  network, each in its own environment: `fetch_worker` (model weights),
  `pack_worker` (dependency packs) and `update_worker` (the release-feed check
  and installer download). `tests/test_docs_inventories.py` derives that set
  from the tree, so a fourth fails a test rather than quietly making this
  sentence wrong -- which is what it was until 2026-09-12.
- **Three threads.** The pygame frame loop never blocks; the asyncio worker
  lives on `realmspinner-loop`; everything blocking goes through `TaskRunner`. One GL
  context.
- **`service/` is the only business-logic layer.** Panes and tests both call it.
  Refusals raise `service.errors` exceptions carrying a `field`.
- **`bpy` never runs in the app process**, and every subprocess goes in the
  `winjob` kill-on-close job. A scan test enforces each: `tests/modes/poser/test_poser_imports.py`
  for the import, `tests/test_vram.py` for the job.
- **The headless engine packages** (`kernels/mesh/`, the `clay` engine;
  `kernels/pixel/`, the `inker` engine, incl. `flourish/` and `walk/`;
  `kernels/grid2d/`, `kernels/geom3d/`, `kernels/manual/`, `kernels/audio/`;
  plus each mode's own `studio/modes/<name>/engine/` for
  `mason`, `plotter`, `packwright`, `sirens`, `poser` and `muse`) import no
  imgui, moderngl, pygame or `service`. Import-pinning tests enforce the exact
  outward set, so adding an import means updating the pin -- deliberately.
  The list is `tests/_pure_packages.py`'s `pure_packages()`, worked out from
  the tree rather than hand-kept, because the 2026-09-12 audit found this
  list four packages short of it.
- **Document writers stage to a temp and `os.replace`.** Never write over a
  user's file in place.
- **Untrusted parsers bound their allocations** before making them, not after.

## Style

Match the surrounding code. The distinctive thing about this codebase is that
comments explain *why*, usually by naming the incident that motivated the guard
-- if you fix a bug, say in a comment what the bug was, and add a test whose
name is the claim. A regression test that passes against the unfixed code is not
a regression test; check that it fails first.

## Commits and pull requests

- Run `uv run python scripts/preflight.py` before opening a PR. To run the whole
  of Windows CI locally first -- and the installer after it -- use
  `pwsh scripts\rebuild.ps1`.
- One logical change per commit.
- If you change something a stored measurement depends on (`trellis_band`,
  `SEAM_MAX`, the grade scale), say so in the PR description -- the maintainer
  keeps a dated record of the measurement behind each such constant outside
  this repo, and it needs updating before the change lands.
- If you change behaviour the manual describes, update the manual in the same
  commit. Chapter numbering is test-gated in both directions.

## Licence

Contributions are accepted under **GPL-3.0-or-later**, matching the project. By
opening a pull request you agree your contribution ships under those terms.
