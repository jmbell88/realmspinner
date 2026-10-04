"""Output-aware canvas views over the actual exported cells and animation tags."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

from imgui_bundle import imgui

from .....kernels.rig import store
from .... import controls, widgets
from ....tokens import sp
from ..engine import assets

TASK_KEY = "create-preview"


def load(directory: Path, params: dict[str, Any]) -> list[dict[str, Any]]:
    """Filesystem half. Executed in TaskRunner, returning no GPU resources."""
    output: list[dict[str, Any]] = []
    sheet = params.get("sheet") or {}
    if isinstance(sheet, dict) and sheet.get("tile_w"):
        cols, rows = int(sheet.get("columns", 1)), int(sheet.get("rows", 1))
        w, h = int(sheet["tile_w"]), int(sheet.get("tile_h", sheet["tile_w"]))
        output.append(
            {
                "name": "Tileset",
                "path": directory / "input.png",
                "columns": cols,
                "rows": rows,
                "cell_w": w,
                "cell_h": h,
                "cells": [
                    {"x": c * w, "y": r * h, "w": w, "h": h}
                    for r in range(rows)
                    for c in range(cols)
                ],
                "terrain": sheet.get("layout") == "blob47",
            }
        )
    for record in store.list_sprite_drafts(directory):
        for letter, _candidate in zip(
            store.SPRITE_CANDIDATES, record.get("candidates") or [], strict=False
        ):
            output.append(
                {
                    **record,
                    "name": f"{record.get('sheet_type', 'Sheet')} {letter}",
                    "path": store.sprite_draft_png_path(directory, record["id"], letter),
                }
            )
    for record in store.list_sheets(directory):
        sheet_id = record["id"]
        reduced = store.read_sheet_pixel(directory, sheet_id)
        source = reduced if reduced and reduced.get("cells") else record
        path = (
            store.sheet_pixel_png_path(directory, sheet_id)
            if source is reduced
            else (store.sheet_png_path(directory, sheet_id))
        )
        cells = source.get("cells") or []
        if cells:
            output.append(
                {
                    **record,
                    **source,
                    "name": record.get("name") or "Rendered sheet",
                    "path": path,
                    "columns": source.get("columns", record.get("columns", 1)),
                    "rows": source.get("rows", record.get("rows", 1)),
                }
            )
    return output


def request(ctx: Any, job: dict[str, Any]) -> list[dict[str, Any]]:
    key = (str(job["id"]), getattr(ctx.cache, "_generation", None))
    preview = ctx.state.preview
    cached = preview.get("create_preview")
    if preview.get("create_preview_requested") != key and ctx.submit(
        TASK_KEY, load, ctx.job_dir(job["id"]), copy.deepcopy(job.get("params") or {}), tag=key
    ):
        preview["create_preview_requested"] = key
    return cached[1] if cached and cached[0][0] == key[0] else []


def adopt(ctx: Any, done: Any) -> None:
    key = getattr(done, "tag", None)
    if key != ctx.state.preview.get("create_preview_requested"):
        return
    if key and key[0] == ctx.state.selected:
        ctx.state.preview["create_preview"] = (key, done.result or [])
    else:
        # The 2026-10-04 audit, finding create-47: a result for a selection the
        # user has since left was dropped but its key stayed recorded as
        # requested, so ``request`` saw the job as already asked for and never
        # resubmitted when the user came back to it -- the sheet preview stayed
        # empty until the cache generation happened to move.
        ctx.state.preview.pop("create_preview_requested", None)


ANIMATION_FRAME_KEY = "create_animation_drawn"


def note_animation_frame(preview: dict[str, Any], frame: int, *, paused: bool) -> None:
    """Record that an animation was drawn on imgui frame ``frame``, or that it is paused.

    The shell's idle throttle cannot see into this pane: the clip advances by
    ``imgui.get_time()`` with no input, so a throttled window samples it at
    ``IDLE_FPS`` (the 2026-10-03 audit, finding shell-37). A frame stamp rather
    than a flag, because the flag would outlive the pane -- ``animating`` is
    true only for the frame this was last drawn on.
    """
    if paused:
        preview.pop(ANIMATION_FRAME_KEY, None)
    else:
        preview[ANIMATION_FRAME_KEY] = frame


def animating(preview: dict[str, Any], frame: int) -> bool:
    """Whether an unpaused animation was drawn on the most recent frame. Pure."""
    return preview.get(ANIMATION_FRAME_KEY) == frame


def frame_at(record: dict[str, Any], tag: dict[str, Any], seconds: float) -> int:
    """Select an exported cell using its durations and loop policy."""
    cells = record.get("cells") or []
    if not cells:
        return 0
    start = max(0, min(int(tag.get("start", 0)), len(cells) - 1))
    end = max(start, min(int(tag.get("end", start)), len(cells) - 1))
    frames = (record.get("animation") or {}).get("frames") or []
    durations = {
        int(f.get("cell_index", -1)): max(1, int(f.get("duration_ms", 100)))
        for f in frames
        if isinstance(f, dict)
    }
    sequence = list(range(start, end + 1))
    direction = tag.get("direction", "forward")
    if direction in ("reverse", "pingpong_reverse"):
        sequence.reverse()
    if direction in ("pingpong", "pingpong_reverse") and len(sequence) > 2:
        sequence += sequence[-2:0:-1]
    values = [durations.get(i, 100) for i in sequence]
    total = sum(values)
    elapsed = max(0.0, seconds * 1000)
    if tag.get("loop", True):
        elapsed %= total
    else:
        elapsed = min(elapsed, total - 0.001)
    for i, duration in zip(sequence, values, strict=True):
        if elapsed < duration:
            return i
        elapsed -= duration
    return sequence[-1]


def _image(texture: Any, size: tuple[float, float], uv0=(0.0, 0.0), uv1=(1.0, 1.0)) -> None:
    imgui.image(widgets.texture_ref(texture), size, uv0, uv1)


def _cell_uv(record: dict[str, Any], cell: dict[str, Any]) -> tuple[Any, Any]:
    cells = record.get("cells") or []
    w = max(float(c.get("x", 0)) + float(c.get("w", 1)) for c in cells)
    h = max(float(c.get("y", 0)) + float(c.get("h", 1)) for c in cells)
    # Atlas dimensions can include unused slots at the end of the last row.
    w = max(w, int(record.get("columns", 1)) * int(cell.get("w", 1)))
    h = max(h, int(record.get("rows", 1)) * int(cell.get("h", 1)))
    return (
        (cell.get("x", 0) / w, cell.get("y", 0) / h),
        ((cell.get("x", 0) + cell.get("w", 1)) / w, (cell.get("y", 0) + cell.get("h", 1)) / h),
    )


def draw(ctx: Any, width: float, height: float) -> bool:
    job = ctx.job()
    if job is None or job.get("status") != "done":
        return False
    kind = assets.asset_type_from_params(job.get("params") or {}, stage=job.get("stage", ""))
    if kind not in ("tileset", "sprite_sheet", "character") and ctx.state.create.stage != "pose":
        return False
    records = request(ctx, job)
    if not records:
        return False
    preview = ctx.state.preview
    if job.get("stage") == "model" and ctx.state.create.stage != "pose":
        changed, view = controls.segmented_choice(
            "create-model-view",
            (("result", "Model"), ("sheet", "Exported sheets")),
            ctx.state.create.preview_mode,
            compact=True,
        )
        if changed:
            ctx.state.create.preview_mode = view
        if view == "result":
            return False
    names = [(str(i), str(r.get("name") or "Sheet")) for i, r in enumerate(records)]
    selection = str(preview.get("create_atlas_choice", "0"))
    target = preview.get("create_atlas_focus")
    if target and target[0] == str(job["id"]):
        matching = next((i for i, r in enumerate(records) if r.get("id") == target[1]), None)
        if matching is not None:
            selection = str(matching)
            preview.pop("create_atlas_focus", None)
    if selection not in {v for v, _label in names}:
        selection = "0"
    picked = widgets.combo("##create-atlas", selection, names, width=min(width, sp(260)))
    preview["create_atlas_choice"] = picked
    record = records[int(picked)]
    cells = record.get("cells") or []
    if not cells:
        return False
    terrain = bool(record.get("terrain"))
    tags = (record.get("animation") or {}).get("tags") or []
    default = (
        "map" if terrain else "animation" if tags else "cells" if kind == "tileset" else "sheet"
    )
    mode_key = (str(job["id"]), str(record["path"]))
    if preview.get("create_atlas_active") != mode_key:
        preview["create_atlas_active"] = mode_key
        preview["create_atlas_mode"] = default
        preview["create_atlas_started"] = imgui.get_time()
        preview.pop("create_animation_paused", None)
    options = [("sheet", "Sheet"), ("cells", "Cell")]
    if tags:
        options.insert(0, ("animation", "Animation"))
    if terrain:
        options.insert(0, ("map", "Terrain preview"))
    changed, mode = controls.segmented_choice(
        "create-atlas-view", tuple(options), preview.get("create_atlas_mode", default), compact=True
    )
    if changed:
        preview["create_atlas_mode"] = mode
    texture = (
        ctx.textures.get(
            f"create-atlas:{record['path']}",
            record["path"],
            nearest=True,
            max_side=4096,
            background=True,
        )
        if ctx.textures
        else None
    )
    if texture is None:
        widgets.muted("Loading exported sheet...")
        return True
    room_h = max(1, height - sp(128))
    if mode == "sheet":
        factor = min(width / texture.size[0], room_h / texture.size[1])
        if factor >= 1:
            factor = math.floor(factor)
        _image(texture, (texture.size[0] * factor, texture.size[1] * factor))
    elif mode == "map":
        _terrain(texture, record, width, room_h)
    else:
        cell_index = max(0, min(int(preview.get("create_cell", 0)), len(cells) - 1))
        if mode == "animation":
            tag_choices = [
                (str(i), str(t.get("name") or f"Clip {i + 1}")) for i, t in enumerate(tags)
            ]
            tag_choice = str(preview.get("create_animation_tag", "0"))
            if tag_choice not in {v for v, _label in tag_choices}:
                tag_choice = "0"
            picked_tag = widgets.combo("##create-animation", tag_choice, tag_choices, width=sp(210))
            if picked_tag != tag_choice:
                preview["create_atlas_started"] = imgui.get_time()
                preview.pop("create_animation_paused", None)
            preview["create_animation_tag"] = picked_tag
            paused = preview.get("create_animation_paused")
            if controls.button("Resume playback" if paused is not None else "Pause playback"):
                if paused is None:
                    preview["create_animation_paused"] = imgui.get_time()
                else:
                    preview["create_atlas_started"] += imgui.get_time() - paused
                    preview.pop("create_animation_paused", None)
            imgui.same_line()
            if controls.button("Restart playback"):
                preview["create_atlas_started"] = imgui.get_time()
                preview.pop("create_animation_paused", None)
            note_animation_frame(
                preview, imgui.get_frame_count(), paused="create_animation_paused" in preview
            )
            cell_index = frame_at(
                record,
                tags[int(picked_tag)],
                preview.get("create_animation_paused", imgui.get_time())
                - preview["create_atlas_started"],
            )
        else:
            changed_cell, cell_index = controls.slider_int(
                "Cell##create-cell", cell_index, 0, len(cells) - 1
            )
            if changed_cell:
                preview["create_cell"] = cell_index
        cell = cells[cell_index]
        uv0, uv1 = _cell_uv(record, cell)
        factor = max(1, math.floor(min(width / cell["w"], room_h / cell["h"])))
        factor = min(factor, 12)
        _image(texture, (cell["w"] * factor, cell["h"] * factor), uv0, uv1)
        widgets.muted(
            f"{cell['w']} x {cell['h']} px · cell {cell_index + 1}/{len(cells)} · {factor}x"
        )
    return True


def _terrain(texture: Any, record: dict[str, Any], width: float, height: float) -> None:
    import numpy as np

    from .....kernels.grid2d import blob

    member = np.ones((5, 7), dtype=bool)
    member[1:4, 2:5] = False
    indices = blob.indices_from(member, outside=False)
    side = max(1, min(width / 7, height / 5, sp(96)))
    pos = imgui.get_cursor_screen_pos()
    draw_list = imgui.get_window_draw_list()
    for row in range(5):
        for col in range(7):
            cell = record["cells"][int(indices[row, col])]
            uv0, uv1 = _cell_uv(record, cell)
            low = (pos.x + col * side, pos.y + row * side)
            high = (low[0] + side, low[1] + side)
            draw_list.add_image(widgets.texture_ref(texture), low, high, uv0, uv1)
    imgui.dummy((side * 7, side * 5))
    widgets.muted("Example terrain assembled from the exported blob tiles.")
