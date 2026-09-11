# Why the DC remesh band stays on the exe's own heuristic — 2026-08-01

Why `Config.trellis_band` defaults to **`None`** (the flag omitted, so
`trellis-server.exe` picks `res/512` itself), and what that default is measured
against.

**Backfilled 2026-09-11.** The run below happened on 2026-08-01 and its table
has lived in `config.py`'s own comment ever since; it never got a document
here. The 2026-09-11 audit (finding docs-01) found that gap while checking this
directory both ways, and it is the sharpest possible instance: `CLAUDE.md` names
`trellis_band` as *the* example of a constant the stored corpus is keyed on,
and it was the one such constant with nothing behind it. Nothing about the
default changed — this is the transcription, not a new measurement.

## The question

The DC remesh runs over a narrow band around the surface. `--band` sets its
width; omitting the flag lets the exe use `res/512`. The earlier guess was that
the default band was too narrow and that widening it would close the
see-through holes the props corpus kept showing.

## The run

`warlock sweep`, one reference image, seed 42, `res 1024`, `hole_fraction`
measured at 1024. The figure is the worst-view see-through fraction.

| `--band` | hole fraction | faces | time |
|---|---|---|---|
| auto (flag omitted) | 0.0077 | 267,360 | 123 s |
| 2 | 0.0077 | 266,632 | 143 s |
| 4 | 0.0167 | 290,774 | 124 s |
| 8 | 0.0110 | 297,898 | 136 s |
| 16 | 0.0125 | 289,586 | 193 s |

At `res 1024`, `res/512` **is** band 2, so the first two rows are the same
setting reached two ways.

## What it decided

Two conclusions, both against the guess that prompted the sweep:

1. **The heuristic is already the best rung of the ladder.** No widened band
   beat it on holes.
2. **Widening makes the surface *more* perforated**, not less, while adding
   faces and time — 4 is twice the hole fraction of auto for 23,000 more faces.

So the flag stays off, and `DEFAULT_TRELLIS_BAND` is `None`.

## The noise floor this also establishes

Auto and 2 are the same setting and still disagreed by **728 faces**. That puts
a floor under what counts as a real difference in this pipeline: anything under
roughly **0.3%** is run-to-run noise, not a result. Later sweeps over the
reconstruction flags are read against that floor.

## What it supersedes

An earlier note claimed the default left "~1300 disconnected plates, 7–31% of
the silhouette". Nothing in this sweep measured worse than 1.7%, so whatever
produced those numbers was not this exe at these settings. That claim is
retired; re-measure before acting on it.

## Keyed on this

`Config.trellis_band` / `DEFAULT_TRELLIS_BAND` (`src/warlock/config.py`). The
stored corpus records the band each reconstruction ran with, so this default
does not move on an opinion — a change needs a dated document here first.
