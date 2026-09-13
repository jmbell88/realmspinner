# Troupe's open clip vocabulary — 2026-09-12

Two things the stored corpus is keyed on are about to move: the Troupe layout
snapshot's `version` (`charsheet.LAYOUT_VERSION`, 2 today, written into every
charsheet row's `params["layout"]` and every character sidecar's `troupe`
block) and the clip library's `version` (2 today, in
`templates/clips/*.json` and every user copy under `data_dir/poser/clips/`).
This repo's rule is that such a constant gets a dated document here *before*
it changes, so this is that document. It records what the old files mean, the
exact rule that keeps them meaning it, and the one derivation (compass names)
that a docstring gets backwards.

Nothing here is a timing or a quality measurement; every figure is read from
the code at `67a54124`.

## Why the vocabulary is closed today

One table owns both the *names* a character may perform and their *timing*:

```python
# src/warlock/pipelines/charsheet.py
ANIMATIONS = (
    ("idle", 4, True, 150),
    ("walk", 8, True, 100),
    ("run", 8, True, 60),
    ("attack", 6, False, 80),
    ("jump", 6, False, 100),
)
```

`resolve_layout` refuses any movement not in it, and `clips.animation_tracks`
takes `animated.glb`'s per-frame duration and loop flag from it. A clip a user
authors under any other name renders at a guessed 100 ms in the GLB and cannot
be put on a sheet at all.

## What the legacy table means, restated as clip data

The clip library's authored expansion already reproduces every frame count, so
the table's only information the libraries lack is the millisecond figure:

| clip | closed | segments | frames = sum(segments) + (0 if closed else 1) | table frames | table ms |
|---|---|---|---|---|---|
| idle | yes | [2,1,1] | 4 | 4 | 150 |
| walk | yes | [2,2,2,2] | 8 | 8 | 100 |
| run | yes | [2,2,2,2] | 8 | 8 | 60 |
| attack | no | [2,1,1,1] | 6 | 6 | 80 |
| jump | no | [1,1,1,1,1] | 6 | 6 | 100 |

The table's loop flag equals `closed` for all five, which
`test_the_cyclic_clips_are_the_looping_ones_and_the_one_shots_are_not` already
pins. So loop gets no field of its own in v3: **a closed clip loops**.

## Clip library v3

- Every clip carries an integer `duration_ms`, 10–1000, a multiple of
  `CLIP_DURATION_STEP_MS = 10`. The step is 1000 / `clips.ANIMATION_FPS` (100),
  so every legal duration is a whole number of scene frames in `animated.glb`
  and no tempo is rounded on export.
- Optional `provisional: true` and an optional `source` object survive parse
  and save.
- **Migration.** A v2 or unversioned file is read with `duration_ms` from the
  legacy map `{idle: 150, walk: 100, run: 60, attack: 80, jump: 100}` and 100
  for any other name — exactly what `animation_tracks` does for an unlisted
  name today, so no v2 file changes behaviour on read. Save always writes v3.
  The direction-suffix refusal below is a v3-only rule for the same reason: it
  applies to a v3 read and to the save door (which always writes v3), never to
  a v2 file, so a clip already legally named `turn_left` keeps loading.
- A clip name ending in `_<one of the sixteen direction keys>` is refused:
  Inker's tag parser splits `fall_back` into clip `fall` facing `back`.
- The library pose cap rises from 256 to 1024 (`MAX_LIBRARY_KEYS` and
  `rigging.MAX_CLIP_LIBRARY_POSES` together) so imported clips fit beside the
  shipped ones.

## Layout snapshot v3, and why most rows stay v2

`resolve_layout(payload, *, timing=None)` decides each movement's timing by
precedence:

1. `timing` passed (the service door, which has the rig's clip library): the
   name must be in it, or it is refused by name.
2. A v3 movement carrying `loop` and `duration_ms`: used as written. This is
   the worker's and every stored snapshot's path — no library needed, so a later
   clip edit cannot change a sheet already queued or rendered.
3. The name is in the legacy table: the table.
4. Otherwise refused.

`LayoutSpec.as_dict` writes `"version": 2` **whenever the layout is expressible
in v2** — every movement a legacy name at legacy timing, and no global `fps`.
Every default row and sidecar written after this change is therefore
byte-identical to one written before it, the existing v2 fixtures keep
passing, and an older build still opens every sheet that uses only the five.
An older build shown a v3 snapshot refuses it cleanly at its re-render door
("Troupe layout version 3 is not supported").

A global `fps` (from `FPS_CHOICES = (6, 8, 10, 12, 15, 24, 30)`) is a layout
property: it sets every movement's `duration_ms` to `round(1000 / fps)` and
derives an omitted frame count as
`clamp(round(frames × ms × fps / 1000), 1, MAX_CLIP_FRAMES)`, so a walk keeps
its real length at a new rate. It does not touch `animated.glb`, whose time is
continuous.

## `SIZES` is not corpus-keyed

`charsheet.SIZES` gains 256. A search of `docs/measurements/` for `SIZES`,
`logical_size` and `LAYOUT_VERSION` returns nothing, and no stored artefact is
keyed on the ladder's *members*: a sidecar records the size it was rendered at,
not its index in a ladder. 512 / 256 is an exact stride of two, so
`RENDER_SIZE` and the framing margin are untouched. The ceiling that does bite
is the atlas: at 256 px, `sheet.MAX_ATLAS_PX = 8192` over 8 columns allows 32
rows = 256 cells — the default five movements × eight directions is exactly
256, and any sixth movement at that size is refused by the existing atlas check.

## Compass names follow the camera arithmetic, not the docstring

Frame-folder exports name directions N/NE/E/…; the rule is
**bearing = (180 + yaw) mod 360**. Derived from
`blender_worker._view_forward`, the testable restatement of `_aim_camera`:

- Yaw 0: forward = (0, 1, 0). The camera sits at −Y looking along +Y; the
  templates face −Y, so the subject faces the camera: **S**.
- Yaw 90: forward = (−1, 0, 0). The camera sits at +X — the subject's left
  (templates put the subject's left at +X). Screen-right is
  forward × up = (0, 1, 0) = +Y, so the subject's facing, −Y, points
  **screen-left: W**.
- Yaw 180: the camera sits at +Y behind the subject: **N**. Yaw 270: **E**.

| key | yaw | compass |
|---|---|---|
| front | 0 | S |
| front_left | 45 | SW |
| left | 90 | W |
| back_left | 135 | NW |
| back | 180 | N |
| back_right | 225 | NE |
| right | 270 | E |
| front_right | 315 | SE |

Sixteen directions take the three-letter points (front_front_left, 22.5 → SSW;
left_front_left, 67.5 → WSW; and so on). The mapping uses the cell's canonical
yaw, never yaw + `front_yaw`: `front_yaw` only turns the mesh so its chosen
front is at yaw 0.

`_aim_camera`'s docstring says "yaw increases clockwise seen from above". The
camera's *position* orbits counter-clockwise (+X at 90 is anticlockwise from −Y
viewed from +Z); what turns clockwise is the subject as the camera sees it. The
test therefore derives the table from `_view_forward` rather than from that
sentence.
