# Re-examining the `trellis_tex_res = 512` pin, 2026-09-12

## Why now

`docs/measurements/2026-09-02-trellis-060-props.md`'s decision rule one fired
on v0.6.0 and left one pin still owed a run: `trellis_tex_res = 512`, pinned
in `src/warlock/config.py` against a noise defect in the exe's own auto
texture-resolution heuristic. The rule this document answers, quoted from
that file: *"the rule asks for the auto-tex-res noise to be reproduced with
`trellis-cli.exe --tex-res 1024` on one reference from this corpus, and a
measurement document lifts the pin if the texture is clean."*

## What is under test

One reference from `docs/measurements/corpora/props-v1.txt` (`easy | a
ceramic jug glazed in deep blue`), generated fresh at `text -> sdxl_cfg`,
seed 42 (`job 7448b5b2ae0b`). Two `trellis-cli.exe` runs against that
byte-identical `input.png`, `-m` pointed at the real `trellis2-gguf`
directory, `-s 42`, `--webp off` (for viewing; texture-resolution behaviour is
independent of the container format), otherwise no other flag changed from
its default:

| arm | flag | internal PBR decode |
|---|---|---|
| shipped (control) | `--tex-res 512` | `PBR voxels=1861730 @res512` |
| test | `--tex-res 1024` | `PBR voxels=7434446 @res1024` |

Confirmed from the full run log that the two arms share identical upstream
geometry (`mesh V=7434446 F=14867218` before decimation, on both) and both
used real BiRefNet background removal — the only thing that differs between
them is the texture-guide decode resolution, exactly the axis under test.

One correction made mid-run and worth recording: omitting `--tex-res`
entirely (the CLI's own "auto") decoded at `@res1024` on this binary, not
`@res512` as the flag's help text describes ("default: auto — drops a dense
res-1024 decode to a clean res-512 PBR volume"). Warlock's shipped default is
not "auto" — `config.py` pins the value explicitly — so this is not itself a
finding about the shipped path, but it means the correct control for this
comparison is the explicit `--tex-res 512` run, not an unflagged one.

## What will be run / decision rule

Fixed by the citing document: one reference, both arms, one visual judgement.
If the 1024 texture is clean, a measurement document lifts the pin; if the
noise reproduces, this document records the reproduction and the pin stays.

## Results

Both GLBs opened in Mason (`--webp off` was required — the exe's default
WebP-encoded textures hit `EXT_texture_webp`, which Mason's viewer does not
implement; this is a viewer gap, not a defect in the pin question, and is
outside this document's scope).

The 1024-texture arm shows visible per-texel noise in the baseColor atlas
that the 512-texture arm does not.

## Verdict

**The noise reproduces on v0.6.0; the pin stays.** `trellis_tex_res = 512`'s
comment in `src/warlock/config.py` is confirmed current rather than stale —
no code change follows from this sitting. `docs/measurements/2026-09-02-trellis-060-props.md`'s
outstanding item is closed.

## What this changed in the tree

`TODO.md`'s P32 entry is struck, closed record added. No `src/` change: the
pin was already correctly set, and this sitting confirms rather than revises
it.
