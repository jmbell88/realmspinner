# Troupe — the palette ramp on an authored humanoid, judged at sprite scale

2026-09-16, on top of `9cc3b1d2`, RTX 5090 dev machine.

## Why now

`TODO.md`'s P4 asked for the first Troupe sheet made from human-authored art
rather than a generated species, and a verdict on whether the palette ramp
survives at sprite scale on it. The mesh, its licence and the
import → rig → sheet chain were proven on 2026-09-12, and looking at that
first sheet found two clip defects (F7, the jump's backward knee, and F8, the
walk's swapped passing poses). Both were built on 2026-09-14 together with the
wider backward-knee sweep (F10, `72a4107a`), so the only thing P4 still owed
was the judgement itself on a fresh sheet.

There is no antecedent retry condition, so no retry section.

## What is under test

- **Mesh:** `Superhero_Male_FullBody` from Quaternius's *Universal Base
  Characters* (CC0 1.0), kept locally under the gitignored `docs/examples/`.
  14,318 faces, T-pose, textured, shipped with its own rig, which
  `_strip_incoming_rig` discards.
- **Conversion:** the `.gltf` and its PNG textures were packed straight into a
  `.glb`, with no Blender round trip (2026-09-12 used Blender's exporter). The
  `.gltf` names two normal maps (`T_Hair_1_Normal_png.png`,
  `T_Eye_Normal_png.png`) that the download does not contain, so
  `T_Hair_1_Normal.png` and `T_Eye_Normal.png` stood in. No base-colour
  texture was substituted.
- **Path:** `service.jobs.import_mesh`, then `service.troupe.send_to_troupe`
  with every option at its default except `palette`. That gave a `humanoid`
  rig with `joints="measured"`, then a charsheet: 32 px, 64 colours, outer
  outline, box reduce, no dither, flat lighting, 30° elevation, eight yaws,
  and five movements (idle, walk, run, attack, jump). 256 cells.
- **One departure:** `palette=dawnlight`, not the `cosmos` the entry names,
  because `cosmos` is not installed on this machine. The verdict is about
  `dawnlight`.
- **Home:** a throwaway `WARLOCK_HOME` with `WARLOCK_NO_MIGRATE=1`, so nothing
  reached the real library. The sheet's own validation reported no clipped,
  blank or missing cells.

## The instrument

There is no pre-registered scale. The entry asks one open question: whether
the ramp works at sprite scale on this sheet. Troupe's heatmap
(`studio/troupe/qa.py`, thresholds in
[`2026-09-02-troupe-qa-thresholds.md`](2026-09-02-troupe-qa-thresholds.md))
was offered as the reading order and not as the verdict. It flagged 89 of the
256 cells.

## What was run

One sheet, one subject. The human looked at a static gallery, with no Review
rows and no blinding:
- for each movement, a looping 4x GIF of all eight directions;
- the same movement as an 8-row grid at zoom 1 and at zoom 4;
- the whole atlas;
- the rig's deformation QA sheet.

## Decision rules

None were fixed before the sitting, and this document says so rather than
inventing one afterwards. The entry names no threshold. Its expected outcome
is "a verdict on whether the ramp works at sprite scale", so the human's
answer is recorded as given.

## Retention

The sheet, the gallery and the packed `.glb` lived in the session scratchpad
and are not kept. The recipe above rebuilds them: pack, import,
`send_to_troupe` with `palette=dawnlight`. The mesh stays under
`docs/examples/`.

## Results

In the operator's words:

- **Sheets:** "The sheets themselves look good for the most part, with only
  minimal defects present that would need addressed." The defects are
  "some pixel gaps in the transitions, only noticeable for a frame or two",
  located in the **jump** transitions.
- **Rig deformation QA sheet:** "half of them are in positions most humanoids
  couldn't achieve."

## Verdict

The ramp works at sprite scale on an authored humanoid, with `dawnlight`.
What remains is a small defect in the jump's transitions, not a ramp problem.
It is recorded as its own open finding (F12) rather than holding this entry
open.

## What this changed in the tree

- **The deformation QA battery was wrong, and is fixed.**
  `templates/deform_qa/humanoid.json` was authored in the pose editor's `node`
  frame. `_load_pose_library` had no way to carry a pose space, and
  `_q_rig._deform_qa` never passed one to the renderer. On a `measured` rig
  that rendered three of the four poses impossibly:
  - the squat folded the legs up behind the head, because every leg sign was
    inverted, the F7 mistake again;
  - "arms overhead" pointed the arms straight forward;
  - "elbow and knee 90" bent both knees backward.

  The battery now declares `"space": "delta"`, the frame the clip libraries
  use, and the loader and `_deform_qa` carry it to each cell. The poses were
  re-authored in that frame: the squat is thigh −90°, shin +100°, foot −10°
  with the spine +20°; arms overhead is upper arm +150° and forearm −15°; the
  knee pose's shin is +90°. A re-render on the same mesh showed a deep squat,
  both hands straight up, a kneel with the elbows bent forward, and the
  unchanged torso twist.
- **Regression tests.** Each fails against the unfixed battery or loader.
  - `tests/test_rigging.py::test_the_deformation_battery_declares_delta_space`
  - `tests/test_rig_worker.py::test_the_deformation_battery_renders_in_delta_space`
  - `tests/test_clip_library_poses.py::test_deform_battery_squat_flexes_the_hip_forward_and_the_knee_back`
  - `tests/test_clip_library_poses.py::test_no_deform_battery_pose_bends_a_knee_backward_past_fifteen_degrees`
- **Found and not fixed (F13).** On `tests/fixtures/humanoid/cesium_man.glb`
  the QA sheet renders the figure lying on its side in every cell, although
  `model.glb` stands upright (Y from 0 to 1.51 m). A render with the *old*
  battery shows the same thing, so the pose fix did not cause it.
- **`TODO.md`:** P4 closed into *Closed records*; F12 and F13 opened.
