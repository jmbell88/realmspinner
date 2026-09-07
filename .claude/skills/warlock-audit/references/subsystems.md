# Subsystems — the slice table

One row per scope key. Paste a row whole into the explorer brief. Paths are under
`src/warlock/` unless they start with `tests/`, `docs/` or `scripts/`. *INVARIANTS* gives
the line of each bold lead-in in `docs/INVARIANTS.md` at 2026-09-05; if the line has
moved, grep the quoted words. *Gates* are tests that already refuse a class of defect in
this slice; an explorer checks them before reporting.

Mode keys (`warlock.studio.modes.KEYS`): home library create inker clay poser troupe
plotter packwright muse sirens review settings. `home`, `library`, `review`, `settings`
→ `shell`.

## The segment contract

Every row is split into **segments**, and one explorer is launched per segment. A row of
this table is too large for one agent to read honestly — `shell` is 52 modules and 13
panes, `inker` 55 package files plus its codecs and panes, `pipelines` 45. Both real runs
of this skill split a row by hand rather than obey the old one-explorer-per-slice rule
(`create` four ways on 2026-09-05, which is what found that run's two Highs; `docs` seven
ways on 2026-09-06, recorded in that audit file as a deviation). The split is written down
here so it is a decision, not an improvisation.

- **A row's segments partition its Source list exactly.** No path in two segments, no path
  in none. That is what makes the coverage note checkable.
- **Every segment carries the whole 18-item defect-class checklist.** The split is by file,
  not by defect class: an explorer reads a bounded set of files once and walks every class
  over them.
- **The count is not a per-run judgement.** Do not fold two segments into one call because
  one looks small, and do not split a segment further. `tour` lists one segment and says
  so, so "one explorer" there is a decision rather than the absence of a split.
- **A segment names the INVARIANTS leads, tests and manual chapters it owns.** Where the
  row's list does not divide, the segment inherits the row's whole list.
- **Segments are read-only boundaries, not blinkers.** A defect an explorer notices in a
  neighbouring segment is still reported, with `where` pointing at the real file; the merge
  decides. Segments do not apply to a *theme* scope: a theme is one explorer per row,
  walking its single checklist item over the whole row.

## shell

- **Source:** `studio/main.py`, `runtime.py`, `app_ctx.py`, `state.py`, `tasks.py`,
  `layout.py`, `layouts.py`, `layout_edit.py`, `layout_skeleton.py`, `dialogs.py`,
  `palette.py`, `menus.py`, `rail.py`, `status_bar.py`, `toolbar.py`, `shortcuts.py`,
  `journal.py`, `recents.py`, `docmodes.py`, `focus.py`, `dpi.py`, `fonts.py`, `theme.py`,
  `tokens.py`, `motion.py`, `guard.py`, `probe.py`, `splash.py`, `review_mode.py`,
  `review_panes.py`, `jobs_cache.py`, `asset_open.py`, `artifacts.py`, `imgui_backend.py`,
  `widgets.py`, `controls.py`, `forms.py`, `undo.py`, `verbs.py`, `icons.py`,
  `surfaces.py`, `textures.py`, `shadows.py`, `vibrancy.py`, `fps.py`, `resources.py`,
  `settings.py`, `component_gallery.py`, `filetypes.py`, `npyguard.py`, `sizeguard.py`,
  `xmlguard.py`, `zipguard.py`; panes `landing.py`, `library.py`, `library_full.py`,
  `inspector.py`, `overlay.py`, `app_settings.py`, `first_run.py`, `model_gate.py`,
  `palette.py`, `thumbs.py`, `candidates_panel.py`, `stamps.py`, `tour.py`.
- **INVARIANTS:** 11, 13 (three threads), 21, 23 (absent models), 43 (refusal names its
  control), 151–169 (shortcut arms, ConfirmQueue, toast ladder, filter box, trash, palette,
  motion, thirteen modes, menu/status bar, Home), 203 (layouts), 320–324, 332 (GL, fonts,
  untrusted files, startup dialog), 334–346 (allocation checks, task prefixes, broad
  except, ceilings, caches), 350–352 (screenshots, startup phases), 400–404
  (accessibility, boundaries, imgui ids), 410–422 (file picker, i18n, Tab, TEXTINPUT,
  columns, DPI), 452–458 (rail rungs, drag-drop, probe, exercise driver), 485 (panel
  section), 489–503 (journal, crash recovery).
- **Tests:** `tests/test_studio_smoke.py`, `tests/test_editor_shell.py`,
  `tests/test_journal.py`, `tests/test_layouts.py`, `tests/test_panes_*.py`,
  `tests/test_landing_*.py`, `tests/test_library_*.py`, `tests/test_inspector_*.py`,
  `tests/test_review_mode.py`, `tests/test_settings_*.py`, `tests/test_notifications.py`,
  `tests/test_dialogs_prompt.py`, `tests/test_palette.py`, `tests/ui/`, `tests/app/`.
- **Manual:** 03, 04, 20, 21, 36, 37, 38, 41.
- **Gates:** `tests/test_undo_gesture_doors.py`, `tests/test_frame_thread_doors.py`,
  `tests/test_task_thread_writes.py`, `tests/test_pane_guard.py` (imgui id collisions,
  raising panes), `tests/test_accessibility.py`, `tests/test_ux_todo_fixes.py` (toast
  levels), `tests/test_findings_themes.py`, `tests/test_findings_blind_spots.py`.
- **Segments (6):**
  1. `boot` — `main.py`, `runtime.py`, `app_ctx.py`, `state.py`, `tasks.py`, `splash.py`,
     `guard.py`, `probe.py`, `focus.py`. Leads 11, 13, 332, 334–346, 350–352, 452–458.
     Tests `tests/test_studio_smoke.py`, `tests/test_editor_shell.py`, `tests/app/`, and
     the three thread gates.
  2. `documents` — `layout.py`, `layouts.py`, `layout_edit.py`, `layout_skeleton.py`,
     `docmodes.py`, `journal.py`, `recents.py`, `asset_open.py`, `artifacts.py`,
     `jobs_cache.py`, `filetypes.py`, `npyguard.py`, `sizeguard.py`, `xmlguard.py`,
     `zipguard.py`. Leads 203, 320–324, 489–503. Tests `tests/test_layouts.py`,
     `tests/test_journal.py`.
  3. `chrome` — `rail.py`, `menus.py`, `status_bar.py`, `toolbar.py`, `shortcuts.py`,
     `dialogs.py`, `palette.py`; panes `palette.py`, `overlay.py`. Leads 151–169, 410–422,
     452–458. Tests `tests/test_palette.py`, `tests/test_dialogs_prompt.py`,
     `tests/test_notifications.py`, `tests/ui/`. Manual 37, 38.
  4. `widgets` — `widgets.py`, `controls.py`, `forms.py`, `undo.py`, `verbs.py`,
     `icons.py`, `surfaces.py`, `textures.py`, `shadows.py`, `vibrancy.py`, `fps.py`,
     `resources.py`, `settings.py`, `component_gallery.py`, `motion.py`, `theme.py`,
     `tokens.py`, `dpi.py`, `fonts.py`, `imgui_backend.py`. Leads 43, 332, 400–404,
     410–422, 485. Tests `tests/test_accessibility.py`, `tests/test_pane_guard.py`,
     `tests/test_undo_gesture_doors.py`.
  5. `home-library` — panes `landing.py`, `library.py`, `library_full.py`, `inspector.py`,
     `thumbs.py`, `stamps.py`, `candidates_panel.py`, `tour.py`. Leads 21, 23, 151–169
     (filter box, trash, Home). Tests `tests/test_landing_*.py`, `tests/test_library_*.py`,
     `tests/test_inspector_*.py`, `tests/test_panes_*.py`. Manual 03, 21.
  6. `review-settings` — `review_mode.py`, `review_panes.py`; panes `app_settings.py`,
     `first_run.py`, `model_gate.py`. Leads 23, 43, 356–380. Tests
     `tests/test_review_mode.py`, `tests/test_settings_*.py`. Manual 04, 36, 41.

## create

- **Source:** `studio/create_brief.py`, `create_stages.py`, `create_assets.py`,
  `generation_workspace.py`, `matte_preview.py`, `quality.py`, `candidates.py`,
  `studio/viewer/`, `studio/viewer_embed.py`, `_view_*.py`; panes `settings_2d.py`,
  `settings_3d.py`, `settings_character.py`, `texture_panel.py`, `remesh_panel.py`,
  `retarget_panel.py`, `sprite_panel.py`, `sheet_panel.py`, `stage_rig.py`,
  `pose_panel.py`; `guidance.py`, `judge.py`, `generation.py`, `sweep.py`,
  `provenance.py`.
- **INVARIANTS:** 49 (`hole_worst`), 98–100 (LoRA, conditioning axes), 108 (chunked
  prompt), 136–138 (glTF loader ceilings), 146–149 (taxonomy, quality tier), 318
  (painted reference), 354 (rendering parity), 384–388 (derivation, sweep, candidates),
  398 (GLB loader).
- **Tests:** `tests/test_create_*.py`, `tests/test_generation_*.py`,
  `tests/test_viewer_*.py`, `tests/test_gltf_loader.py`, `tests/test_candidates.py`,
  `tests/test_judge*.py`, `tests/test_matte_*.py`, `tests/test_settings_*.py`,
  `tests/test_prompt_*.py`, `tests/test_conditioning_*.py`.
- **Manual:** 02, 12, 22, 23, 24.
- **Gates:** `tests/test_create_stages.py` (no control in both bar and column),
  `tests/test_resource_ceilings.py`, `tests/test_field_error_wiring.py`.
- **Segments (4)** — this is the 2026-09-05 split, which found both of that run's Highs:
  1. `brief` — `create_brief.py`, `create_stages.py`, `create_assets.py`, `quality.py`,
     `guidance.py`, `judge.py`. Leads 146–149, 318. Gate `tests/test_create_stages.py`.
     Tests `tests/test_create_*.py`, `tests/test_judge*.py`. Manual 02, 22.
  2. `workspace` — `generation_workspace.py`, `candidates.py`, `matte_preview.py`,
     `generation.py`, `sweep.py`, `provenance.py`. Leads 49, 384–388. Tests
     `tests/test_generation_*.py`, `tests/test_candidates.py`, `tests/test_matte_*.py`.
     Manual 12, 24.
  3. `viewer` — `studio/viewer/` (all), `viewer_embed.py`, `_view_bounds.py`,
     `_view_cache.py`, `_view_drag.py`, `_view_overlay.py`, `_view_pick.py`. Leads
     136–138, 354, 398. Tests `tests/test_viewer_*.py`, `tests/test_gltf_loader.py`,
     `tests/test_resource_ceilings.py`. Manual 23.
  4. `panes` — panes `settings_2d.py`, `settings_3d.py`, `settings_character.py`,
     `texture_panel.py`, `remesh_panel.py`, `retarget_panel.py`, `sprite_panel.py`,
     `sheet_panel.py`, `stage_rig.py`, `pose_panel.py`. Leads 98–100, 108, 146–149. Gate
     `tests/test_field_error_wiring.py`. Tests `tests/test_settings_*.py`,
     `tests/test_prompt_*.py`, `tests/test_conditioning_*.py`. Manual 12, 22, 23.

## inker

- **Source:** `studio/inker/` (and `inker/flourish/`), `studio/inker_*.py`,
  `inker_flourish.py`, `pixelguard.py`, `ninepatch.py`, `anchors.py`, `ants.py`; panes
  `inker_*.py`; `pipelines/sheet.py`, `pixel.py`, `pixelize.py`, `pixelsheet.py`.
- **INVARIANTS:** 53 (zoom), 187 (selection is view state), 193 (asein ledger), 209–261
  (Inker: context bar, layout, pan, quarter turns, timeline, undo, animation, colour modes,
  indexed palettes, links, grayscale, GIF, frame cache, playback, clip export, the 24
  divergences), 288–294 (Aseprite parity record), 424 (layer groups), 466–468 (Flourish),
  550.
- **Tests:** `tests/inker/` (149 files), `tests/test_inker_*.py`, `tests/test_pixel*.py`,
  `tests/test_sheet*.py`, `tests/test_undo_budget.py`.
- **Manual:** 05, 06, 15, 28, 29.
- **Gates:** headless import pin (grep `tests/inker/` and `tests/test_inker_*.py` for
  `imgui`), `tests/test_undo_gesture_doors.py`, `tests/test_inker_busy_guards.py`,
  `tests/test_inker_export_refusals.py`, `docs/COMPAT.md` Aseprite ledger (not executable).
- **Segments (7):**
  1. `document` — `inker/document.py`, `_doc_anim.py`, `_doc_flourish.py`,
     `_doc_geometry.py`, `_doc_history.py`, `_doc_indexed.py`, `_doc_layers.py`,
     `_doc_paint.py`, `_doc_ranges.py`, `_doc_selection.py`, `_doc_sheet.py`,
     `_doc_slices.py`, `_doc_tiles.py`, `layers.py`, `groups.py`, `undo.py`,
     `selection.py`, `transform.py`. Leads 187, 209–261 (undo, links, layout), 424. Gates
     `tests/test_undo_gesture_doors.py`, `tests/test_undo_budget.py`.
  2. `paint` — `brush.py`, `composite.py`, `dither.py`, `filters.py`, `gradient.py`,
     `inpaint.py`, `indexed.py`, `index_plane.py`, `textstamp.py`, `mirror.py`,
     `animation.py`, `anim_edits.py`. Leads 209–261 (animation, colour modes, indexed
     palettes, grayscale), 550. Manual 05, 06.
  3. `sheets` — `tiles.py`, `tile_edits.py`, `tiling.py`, `slices.py`, `sheetin.py`,
     `sheetout.py`, `sheetmerge.py`, `sheetscope.py`; `pipelines/sheet.py`, `pixel.py`,
     `pixelize.py`, `pixelsheet.py`. Tests `tests/test_sheet*.py`,
     `tests/test_pixel*.py`. Manual 15, 29.
  4. `codecs` — `asein.py`, `aseout.py`, `ora.py`, `gifin.py`, `gifout.py`, `gpl.py`.
     Leads 193, 288–294, 209–261 (GIF, clip export). Gate
     `tests/test_inker_export_refusals.py` and `docs/COMPAT.md`'s Aseprite ledger, read
     both ways.
  5. `flourish` — `inker/flourish/` (all), `studio/inker_flourish.py`; pane
     `inker_flourish.py`. Leads 466–468.
  6. `mode` — `studio/inker_export.py`, `inker_keys.py`, `inker_mode.py`, `inker_open.py`,
     `inker_ops.py`, `inker_palette_io.py`, `inker_playback.py`, `inker_sheet.py`,
     `inker_state.py`, `inker_walk.py`, `pixelguard.py`, `ninepatch.py`, `anchors.py`,
     `ants.py`. Leads 53, 209–261 (pan, quarter turns, frame cache, playback). Manual 28.
  7. `panes` — panes `inker_bridge.py`, `inker_canvas.py`, `inker_colors.py`,
     `inker_context.py`, `inker_drag.py`, `inker_generate.py`, `inker_gestures.py`,
     `inker_menu.py`, `inker_picker.py`, `inker_preview.py`, `inker_sheet.py`,
     `inker_slices.py`, `inker_textures.py`, `inker_tiles.py`, `inker_timeline.py`,
     `inker_tools.py`, `inker_walk.py`, `inker_walk_canvas.py`. Leads 209–261 (context
     bar, timeline). Gates `tests/test_inker_busy_guards.py`, the headless import pin.
     Manual 28.

## clay

- **Source:** `studio/clay/`, `studio/clay_*.py`, `clay_view.py`, `clay_viewport.py`,
  `clay_hints.py`; panes `clay_*.py`; `glbio.py`, `studio/viewer/gltf.py`.
- **INVARIANTS:** 136–138 (loader ceilings, `MAX_TOTAL_BYTES`), 296–308 (CSR mesh, one
  conversion out, uid undo, freezing, two merges, drag delta, boundary ops), 398.
- **Tests:** `tests/clay/`, `tests/test_clay_*.py`, `tests/test_mesh_import.py`,
  `tests/test_gltf_loader.py`.
- **Manual:** 07, 30.
- **Gates:** headless import pin (`tests/test_clay_ops.py`), `tests/test_clay_view.py`
  (outward imports), `tests/test_resource_ceilings.py`.
- **Segments (3):**
  1. `mesh` — `clay/mesh.py`, `topo.py`, `adjacency.py`, `earclip.py`, `ops.py`,
     `ops_bevel.py`, `ops_boolean.py`, `ops_dissolve.py`, `ops_subdiv.py`, `ops_topo.py`,
     `uv.py`, `shading.py`, `diagnose.py`. Leads 296–308 (CSR mesh, freezing, two merges,
     boundary ops). Tests `tests/clay/`.
  2. `document` — `clay/document.py`, `elements.py`, `edits.py`, `select.py`,
     `selection.py`, `pick.py`, `drag.py`, `serialize.py`, `primitives.py`, `presets.py`,
     `glbimport.py`; `glbio.py`, `studio/viewer/gltf.py`. Leads 136–138, 300 (uid undo),
     306 (drag delta), 398. Tests `tests/test_mesh_import.py`,
     `tests/test_gltf_loader.py`, `tests/test_resource_ceilings.py`.
  3. `mode` — `studio/clay_mode.py`, `clay_ops.py`, `clay_state.py`, `clay_view.py`,
     `clay_viewport.py`, `clay_hints.py`; panes `clay_bridge.py`, `clay_header.py`,
     `clay_hud.py`, `clay_menu.py`, `clay_outliner.py`, `clay_props.py`, `clay_tools.py`.
     Gates `tests/test_clay_view.py`, the headless import pin. Tests
     `tests/test_clay_*.py`. Manual 07, 30.

## poser

- **Source:** `studio/poser_mode.py`, `poser_viewport.py`, `skeletons.py`,
  `_viewer_pose.py`; panes `poser_*.py`, `pose_panel.py`, `retarget_panel.py`;
  `rigging.py`, `poselib.py`, `clips.py`, `service/rig.py`, `service/poses.py`,
  `service/clips.py`, `pipelines/blender_worker.py`, `jointfit.py`, `pose2d.py`,
  `templates/*.json`.
- **INVARIANTS:** 116 (Blender out of process), 124–134 (weld before heat, deformation
  battery, joint sources, pose contract, poses are files, validated at read), 140–142
  (sheet grid on host, importer invents objects), 436–438 (rotation frames, measured
  joints), 515 (skeleton is JSON).
- **Tests:** `tests/test_poser_*.py`, `tests/test_rig*.py`, `tests/test_rigging.py`,
  `tests/test_inspector_rig.py`, `tests/test_panes_rig_bone_count.py`.
- **Manual:** 08, 25, 26.
- **Gates:** `tests/test_poser_imports.py` (only `blender_worker` imports `bpy`),
  `tests/test_poser_panes_smoke.py`.
- **Segments (3):**
  1. `rig` — `rigging.py`, `skeletons.py`, `templates/*.json`,
     `pipelines/blender_worker.py`, `pipelines/jointfit.py`. Leads 116, 124–134 (weld
     before heat, deformation battery, joint sources), 436–438, 515. Gate
     `tests/test_poser_imports.py`. Tests `tests/test_rig*.py`, `tests/test_rigging.py`.
     Manual 25.
  2. `poses` — `poselib.py`, `clips.py`, `pipelines/pose2d.py`, `service/rig.py`,
     `service/poses.py`, `service/clips.py`. Leads 128–134 (pose contract, poses are
     files, validated at read), 140–142. Manual 26.
  3. `mode` — `studio/poser_mode.py`, `poser_viewport.py`, `_viewer_pose.py`; panes
     `poser_clips.py`, `poser_controls.py`, `poser_library.py`, `pose_panel.py`,
     `retarget_panel.py`. Gate `tests/test_poser_panes_smoke.py`. Tests
     `tests/test_poser_*.py`, `tests/test_inspector_rig.py`,
     `tests/test_panes_rig_bone_count.py`. Manual 08.

## troupe

- **Source:** `studio/troupe/`, `studio/troupe_mode.py`, `troupe_state.py`; panes
  `troupe_*.py`, `settings_character.py`; `_q_troupe.py`, `_q_sprite.py`,
  `service/troupe.py`, `service/sprites.py`, `service/sheets.py`,
  `service/characters.py`, `pipelines/charsheet.py`, `spritesynth.py`, `sheet.py`.
- **INVARIANTS:** 390–394 (sprite draft, cell order, layout), 426–448 (Troupe programme,
  frame tables, clips, sidecar, pixeliser, joints, atlas size, four jobs and a gate, T-pose
  guide, character sheet, no document), 460–464 (tag names, corrections, three-way
  re-render merge), 470 (scores rank, never gate).
- **Tests:** `tests/troupe/`, `tests/test_sprite_*.py`, `tests/test_effect_sprites.py`.
- **Manual:** 11, 27, 33.
- **Gates:** headless import pin (`tests/troupe/`), `tests/test_effect_sprites.py`
  (outward imports).
- **Segments (3):**
  1. `engine` — `studio/troupe/qa.py`, `spec.py`, `ulpc.py`; `studio/troupe_state.py`.
     Leads 426–448 (frame tables, joints, atlas size, T-pose guide), 470. Gates the
     headless import pin, `tests/test_effect_sprites.py`. Tests `tests/troupe/`.
  2. `jobs` — `_q_troupe.py`, `_q_sprite.py`, `service/troupe.py`, `service/sprites.py`,
     `service/sheets.py`, `service/characters.py`. Leads 390–394, 440–448 (four jobs and a
     gate, no document), 460–464.
  3. `render` — `pipelines/charsheet.py`, `spritesynth.py`, `sheet.py`;
     `studio/troupe_mode.py`; panes `troupe_bridge.py`, `troupe_characters.py`,
     `troupe_preview.py`, `troupe_send.py`, `troupe_settings.py`, `troupe_sheets.py`,
     `settings_character.py`. Leads 432–436 (sidecar, pixeliser), 444 (character sheet).
     Tests `tests/test_sprite_*.py`. Manual 11, 27, 33.

## plotter

- **Source:** `studio/plotter/`, `studio/tilegrid/`, `studio/plotter_*.py`; panes
  `plotter_*.py`; `_q_tileset.py`, `_q_tilesheet.py`, `service/tilesheets.py`,
  `pipelines/tileatlas.py`, `tilemask.py`, `tilesheet.py`.
- **INVARIANTS:** 78–80 (grid pack never trims, one tileset per syntax), 171–173
  (Plotter is a mode with a pure engine, AI tile sheet is one job), 191 (`tilegrid` shared
  leaf), 195–201 (tilemap cel materialisation, `tiles.json`, view-state toggles,
  reopenable exports), 406–408 (new map asked for, tile size vs projection).
- **Tests:** `tests/plotter/` (53), `tests/tilegrid/`, `tests/test_plotter_*.py`,
  `tests/test_tileset_*.py`, `tests/test_tilesheet_*.py`.
- **Manual:** 09, 13, 31.
- **Gates:** `tests/plotter/test_compat_matrix.py` (parses `docs/COMPAT.md` Tiled rows
  as data), headless import pins (`tests/plotter/`, `tests/tilegrid/`).
- **Segments (3):**
  1. `map` — `studio/plotter/` (all: `_map_geometry.py`, `_map_layers.py`, `_map_model.py`,
     `_map_objects.py`, `_map_paint.py`, `_map_project.py`, `_map_tilesets.py`,
     `edits.py`, `layer_rows.py`, `pngio.py`, `project.py`, `props.py`, `render.py`,
     `scene.py`, `terrain.py`, `tilemap.py`, `tmx.py`, `tools.py`, `tsx.py`, `wmap.py`).
     Leads 171–173, 195–201, 406–408. Gates `tests/plotter/test_compat_matrix.py`, the
     headless import pin. Tests `tests/plotter/`.
  2. `tiles` — `studio/tilegrid/` (all); `_q_tileset.py`, `_q_tilesheet.py`,
     `service/tilesheets.py`, `pipelines/tileatlas.py`, `tilemask.py`, `tilesheet.py`.
     Leads 78–80, 173 (AI tile sheet is one job), 191. Tests `tests/tilegrid/`,
     `tests/test_tileset_*.py`, `tests/test_tilesheet_*.py`.
  3. `mode` — `studio/plotter_io.py`, `plotter_mode.py`, `plotter_setup.py`,
     `plotter_state.py`, `plotter_tilesets.py`; panes `plotter_bridge.py`,
     `plotter_canvas.py`, `plotter_layers.py`, `plotter_menu.py`, `plotter_objects.py`,
     `plotter_stamps.py`, `plotter_textures.py`, `plotter_tileset.py`,
     `plotter_tileset_editor.py`, `plotter_tools.py`. Leads 197–199 (view-state toggles),
     406–408. Tests `tests/test_plotter_*.py`. Manual 09, 13, 31.

## packwright

- **Source:** `studio/packwright/`, `studio/packwright_*.py`; panes `packwright_*.py`;
  `studio/atomic.py`.
- **INVARIANTS:** 102, 122 (staged writes), 189 (deterministic packer), 201 (reopenable
  exports).
- **Tests:** `tests/packwright/`, `tests/test_packwright_*.py`,
  `tests/test_atomic_writes.py`.
- **Manual:** 10, 32.
- **Gates:** headless import pin (`tests/packwright/`), `tests/test_undo_gesture_doors.py`.
- **Segments (2):**
  1. `packer` — `studio/packwright/compose.py`, `document.py`, `layout.py`,
     `maxrects.py`, `sources.py`, `texturepacker.py`, `trim.py`, `tsxout.py`, `wpack.py`;
     `studio/atomic.py`. Leads 102, 122, 189, 201. Gate the headless import pin. Tests
     `tests/packwright/`, `tests/test_atomic_writes.py`.
  2. `mode` — `studio/packwright_io.py`, `packwright_mode.py`, `packwright_state.py`;
     panes `packwright_bridge.py`, `packwright_items.py`, `packwright_preview.py`,
     `packwright_settings.py`, `packwright_sources.py`, `packwright_textures.py`. Gate
     `tests/test_undo_gesture_doors.py`. Tests `tests/test_packwright_*.py`. Manual 10, 32.

## muse

- **Source:** `studio/muse/`, `studio/muse_*.py`; panes `muse_*.py`; `_q_music.py`,
  `service/_jobs_music.py`, `pipelines/music_worker.py`, `music_client.py`,
  `separation_worker.py`, `audioout.py`, `pipelines/acestep/`.
- **INVARIANTS:** 56–76 (Muse is a job-row mode, no document, the reverse Sirens bridge,
  `sirens_audio` gains volume not offset, loop rotation, headless with scipy banned,
  crossfade vs `smpl`, crossfade declines, separation is a one-shot child, derived
  formats, Demucs digest).
- **Tests:** `tests/muse/`, `tests/test_muse_*.py`, `tests/test_music_*.py`,
  `tests/test_separation.py`.
- **Manual:** 14, 16, 35.
- **Gates:** headless import pin and scipy ban (`tests/muse/`),
  `tests/test_muse_panes_smoke.py`.
- **Segments (3):**
  1. `engine` — `studio/muse/loops.py`, `waveform.py`; `studio/muse_io.py`,
     `muse_state.py`. Leads 62–70 (loop rotation, headless with scipy banned, crossfade vs
     `smpl`, crossfade declines). Gates the headless import pin and the scipy ban. Tests
     `tests/muse/`.
  2. `mode` — `studio/muse_mode.py`, `muse_brief.py`; panes `muse_player.py`,
     `muse_recipe.py`, `muse_results.py`. Leads 56–60 (job-row mode, no document, the
     reverse Sirens bridge). Gate `tests/test_muse_panes_smoke.py`. Tests
     `tests/test_muse_*.py`. Manual 14, 35.
  3. `jobs` — `_q_music.py`, `service/_jobs_music.py`, `pipelines/music_worker.py`,
     `music_client.py`, `separation_worker.py`, `audioout.py`, `pipelines/acestep/`.
     Leads 72–76 (separation is a one-shot child, derived formats, Demucs digest). Tests
     `tests/test_music_*.py`, `tests/test_separation.py`. Manual 16.

## sirens

- **Source:** `studio/sirens/`, `studio/sirens_*.py` (`sirens_audio.py` is the only
  `pygame.mixer` toucher); panes `sirens_*.py`.
- **INVARIANTS:** 55 (playhead is a bisect), 60–62 (one bridge door), 521–548 (headless
  engine, per-tick parameters, equal temperament, numpy patterns, uid order list, `.wsng`
  is the composition, column-keyed keys, render-then-play, device isolation,
  `render_dirty`, audition task key, pure export, panes smoke).
- **Tests:** `tests/sirens/`, `tests/test_sirens_*.py`, `tests/test_findings_sirens.py`.
- **Manual:** 34.
- **Gates:** headless import pin and scipy ban (`tests/sirens/`),
  `tests/test_sirens_panes_smoke.py`, `tests/test_findings_sirens.py` (`_MOVED` tables).
- **Segments (3):**
  1. `engine` — `studio/sirens/document.py`, `edits.py`, `envelope.py`, `instruments.py`,
     `notes.py`, `synth.py`, `voices.py`, `wavout.py`, `wsng.py`. Leads 521–540 (headless
     engine, per-tick parameters, equal temperament, numpy patterns, uid order list,
     `.wsng` is the composition, pure export). Gates the headless import pin and the scipy
     ban. Tests `tests/sirens/`.
  2. `playback` — `studio/sirens_audio.py`, `sirens_play.py`, `sirens_state.py`,
     `sirens_edit.py`, `sirens_io.py`, `sirens_keys.py`, `sirens_mode.py`,
     `sirens_hints.py`. Leads 55, 60–62, 542–548 (render-then-play, device isolation,
     `render_dirty`, audition task key). Gate `tests/test_findings_sirens.py`. Tests
     `tests/test_sirens_*.py`.
  3. `panes` — panes `sirens_bridge.py`, `sirens_effects.py`, `sirens_envelopes.py`,
     `sirens_instruments.py`, `sirens_orders.py`, `sirens_patterns.py`,
     `sirens_transport.py`. Leads 534 (column-keyed keys), 548 (panes smoke). Gate
     `tests/test_sirens_panes_smoke.py`. Manual 34.

## service

- **Source:** `service/` (all), `queue.py`, `db.py`, `vectors.py`, `vram.py`, `leases.py`,
  `memlog.py`, `_q_*.py`, `followups.py`, `asset_workflows.py`, `publish.py`,
  `migrate.py`, `hashes.py`, `progress.py`, `instance.py`, `errors.py`, `models.py`,
  `config.py`.
- **INVARIANTS:** 15, 17 (service is the only business logic; one sqlite connection),
  31–35 (VRAM modes, coexist, last reference), 41 (admission at the door), 47
  (persistent matte), 51 (`config.effective`), 92–96 (`source.glb`/`model.glb`, TRELLIS
  only, remesh), 102–106 (staged writes, grounding), 112–114 (two mesh measurements,
  gltfpack tiers), 120–122 (cancel token, served names), 144 (derived values), 177–185
  (job kind sweep, `source_job`, follow-ups, VRAM halves agree, `t2i_sample` window),
  314–316 (`clean_jobs`, built asset), 348 (job store recovery), 356–380 (verdicts,
  observations, terminal status, dispatch loop, backup, findings, judge, labelling,
  blinding, tier qualification), 396 (a new job kind is four edits).
- **Tests:** `tests/service/`, `tests/test_service*.py`, `tests/test_queue.py`,
  `tests/test_db_*.py`, `tests/test_jobs_*.py`, `tests/test_job_durability.py`,
  `tests/test_vram*.py`, `tests/test_api.py`, `tests/test_failure_paths.py`,
  `tests/test_atomic_writes.py`, `tests/test_authored_sources.py`.
- **Manual:** 36, 43, 44.
- **Gates:** `tests/test_field_error_wiring.py` (refusals carry a `field`), the stage-keyed
  sweep (grep tests for `_q_` and `STAGE`), `tests/test_atomic_writes.py`,
  `tests/test_resource_ceilings.py`.
- **Segments (4):**
  1. `queue` — `queue.py`, `db.py`, `leases.py`, `vram.py`, `memlog.py`, `progress.py`,
     `instance.py`, `vectors.py`. Leads 17, 31–35, 41, 120–122, 181 (VRAM halves agree),
     348. Tests `tests/test_queue.py`, `tests/test_db_*.py`, `tests/test_vram*.py`,
     `tests/test_job_durability.py`, `tests/test_failure_paths.py`.
  2. `kinds` — `_q_generate.py`, `_q_jobs.py`, `_q_lora.py`, `_q_mesh.py`, `_q_music.py`,
     `_q_rig.py`, `_q_sprite.py`, `_q_tileset.py`, `_q_tilesheet.py`, `_q_troupe.py`;
     `service/jobs.py`, `_jobs_create.py`, `_jobs_lifecycle.py`, `_jobs_list.py`,
     `_jobs_music.py`, `_jobs_resubmit.py`, `_jobs_rework.py`; `followups.py`,
     `asset_workflows.py`. Leads 177–185, 314–316, 396. Gate the stage-keyed sweep. Tests
     `tests/test_jobs_*.py`, `tests/test_api.py`.
  3. `assets` — `service/core.py`, `derive.py`, `export.py`, `files.py`, `library.py`,
     `matte.py`, `palettes.py`, `pixelopts.py`, `loras.py`, `sweeps.py`, `characters.py`,
     `clips.py`, `poses.py`, `rig.py`, `sheets.py`, `sprites.py`, `tilesheets.py`,
     `troupe.py`; `publish.py`, `hashes.py`. Leads 47, 92–96, 102–106, 112–114, 144. Gate
     `tests/test_atomic_writes.py`. Tests `tests/service/`,
     `tests/test_authored_sources.py`.
  4. `gates` — `service/validation.py`, `errors.py`, `findings.py`, `judge.py`,
     `verdicts.py`, `system.py`, `updates.py`, `downloads.py`, `packs.py`; `config.py`,
     `models.py`, `migrate.py`, and top-level `errors.py`. Leads 15, 41, 51, 356–380.
     Gates `tests/test_field_error_wiring.py`, `tests/test_resource_ceilings.py`. Tests
     `tests/test_service*.py`. Manual 36, 43, 44.

## pipelines

- **Source:** `pipelines/` (all; `_workerio.py`, `t2i_client.py`, `text2image_worker.py`,
  `matting_worker.py`, `blender_worker.py`, `music_worker.py`, `separation_worker.py`,
  `lora_train_worker.py`, `pack_worker.py`, `fetch_worker.py`, `recipe_worker.py`,
  `update_worker.py`, `loadprobe.py`, `trellis.py`), `winjob.py`, `doctor.py`, `packs.py`,
  `fetch.py`, `native.py`, `tiercheck.py`, `meshaudit.py`, `meshreport.py`, `native/*.c`,
  `installer/`, `scripts/`.
- **INVARIANTS:** 19, 25–29 (offline; the two user-initiated exceptions; fetch is planned
  before performed), 45 (doctor probe in a child), 86–90 (image pipeline out of process,
  stdin reader, winjob), 110 (native kernel has a reference), 116 (Blender), 310–312 (one
  home, the move), 505–511 (checkout-shaped runtime, verified downloads, one transaction,
  no downloaded Python), 552–554 (unsigned installer, 3.13 floor).
- **Tests:** `tests/test_winjob*.py`, `tests/test_offline.py`, `tests/test_doctor.py`,
  `tests/test_fetch*.py`, `tests/test_packs.py`, `tests/test_pack_worker.py`,
  `tests/test_t2i_*.py`, `tests/test_matting*.py`, `tests/test_workerio.py`,
  `tests/test_native_*.py`, `tests/test_installer.py`, `tests/test_runtime_dependencies.py`,
  `tests/test_lora*.py`.
- **Manual:** 01, 39, 40, 41, 42, 44.
- **Gates:** `tests/test_offline.py` (`HF_HUB_OFFLINE`), the winjob scan (grep tests for
  `kill-on-close`), `tests/test_changelog.py`, `scripts/preflight.py` (version lockstep),
  native parity tests (`tests/test_native_*.py`, `tests/test_bvh_native.py`).
- **Segments (5):**
  1. `children` — `pipelines/_workerio.py`, `t2i_client.py`, `text2image_worker.py`,
     `matting_worker.py`, `blender_worker.py`, `music_worker.py`, `separation_worker.py`,
     `lora_train.py`, `lora_train_worker.py`, `pack_worker.py`, `fetch_worker.py`,
     `recipe_worker.py`, `update_worker.py`, `loadprobe.py`; `winjob.py`. Leads 19, 25–29,
     86–90, 116. Gates `tests/test_offline.py`, the winjob scan. Tests
     `tests/test_winjob*.py`, `tests/test_workerio.py`, `tests/test_t2i_*.py`,
     `tests/test_lora*.py`.
  2. `image` — `pipelines/text2image.py`, `asset2d.py`, `conditioning.py`, `control.py`,
     `prompt.py`, `reference.py`, `matting.py`, `postprocess.py`, `rank.py`. Leads 86–88,
     108. Tests `tests/test_matting*.py`. Manual 22.
  3. `mesh` — `pipelines/trellis.py`, `optimize.py`, `remesh.py`, `retexture.py`,
     `material.py`, `seam.py`, `sheetcheck.py`; `meshaudit.py`, `meshreport.py`. Leads
     92–96, 112–114. Manual 23. Constants here are corpus-keyed: check each against its
     latest `docs/measurements/` document before reporting a value as wrong.
  4. `install` — `doctor.py`, `packs.py`, `fetch.py`, `pipelines/download.py`,
     `native.py`, `tiercheck.py`, `native/*.c`, `installer/`, `scripts/`. Leads 45, 110,
     310–312, 505–511, 552–554. Tests `tests/test_doctor.py`, `tests/test_fetch*.py`,
     `tests/test_packs.py`, `tests/test_pack_worker.py`, `tests/test_native_*.py`,
     `tests/test_installer.py`, `tests/test_runtime_dependencies.py`. Manual 01, 39, 40,
     41, 42.
  5. `workspace` — the rest of `pipelines/`, each file owned by a workspace row rather than
     by this one: `pixel.py`, `pixelize.py`, `pixelsheet.py`, `sheet.py` (inker);
     `charsheet.py`, `spritesynth.py` (troupe); `jointfit.py`, `pose2d.py` (poser);
     `tileatlas.py`, `tilemask.py`, `tilesheet.py` (plotter); `audioout.py`,
     `music_client.py` (muse). **Launch this segment only when the scope is `pipelines`
     alone.** Under `all`, or under the workspace's own scope, those rows read these files
     and a second explorer would only duplicate them; say in the coverage note which
     applied.

## docs

- **Source:** `README.md`, `INSTALL.md`, `CONTRIBUTING.md`, `SECURITY.md`,
  `THIRD-PARTY-NOTICES.md`, `CHANGELOG.md`, `CLAUDE.md`, `TODO.md`, `docs/INVARIANTS.md`,
  `docs/COMPAT.md`, `docs/MODELS.md`, `docs/measurements/`, `docs/manual/*.md`,
  `studio/manual/` (the loader), `changelog.py`.
- **INVARIANTS:** 82–84 (chapters 01–19 reserved; number decides order and part), 261
  (the divergence numbering is a citable API), 290 (non-goals are decisions).
- **Tests:** `tests/manual/`, `tests/test_changelog*.py`, `tests/test_external_doc_links.py`,
  `tests/test_ux_todo_fixes.py`, `tests/test_findings_followups.py`.
- **Manual:** all.
- **Gates:** see `docs-checkup.md`; it lists which rows are already gated.
- **Segments (7)** — this row splits by `docs-checkup.md` section rather than by file, and
  it is the 2026-09-06 split, which that run had to record as a deviation. Every segment
  still gets `docs-checkup.md` whole in place of the defect-class checklist, and walks only
  the sections named here:
  1. `A-B` — sections A (structure) and B (inventories) over `README.md`, `INSTALL.md`,
     `CONTRIBUTING.md`, `SECURITY.md`, `THIRD-PARTY-NOTICES.md`, `CHANGELOG.md` and
     `CLAUDE.md`; plus section G over `studio/manual/` and `changelog.py`.
  2. `C1` — section C over `docs/INVARIANTS.md`, lead-ins 1–320.
  3. `C2` — section C over `docs/INVARIANTS.md`, lead-ins 321 to the end.
  4. `D` — section D over every `docs/measurements/*.md` (54 at 2026-09-06) and the
     constants they key; plus `docs/MODELS.md` and `docs/COMPAT.md`'s Aseprite ledger.
  5. `E` — section E over `TODO.md`, plus section G over `service/` and `queue.py`
     docstrings.
  6. `F1` — section F over manual chapters 00–16 and 34–45.
  7. `F2` — section F over manual chapters 20–33.

## tour

- **Source:** `studio/tour/`, `studio/_view_overlay.py` (the drawing side), panes `tour.py`.
- **INVARIANTS:** 450 (points and waits; never acts for the reader).
- **Tests:** `tests/tour/`.
- **Manual:** 01.
- **Gates:** `tests/tour/test_tour_imports.py` (no outward imports),
  `tests/tour/test_tour_conditions.py` (every wait is a named condition).
- **Segments (1)** — this row does not split. `studio/tour/scripts.py`, `steps.py` and the
  two drawing files are one reading; launch a single explorer, keyed `tour-1`.
