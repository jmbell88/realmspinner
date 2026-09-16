"""Booting the real app, seeding it, and photographing it --- once.

This is an extraction, not new behaviour. ``screenshot_modes.py`` grew the
whole of it while it was the only script that drove the app, and
``exercise_mode.py`` needs every part: the same boot (so the dispatch under
test is ``_build_ui``'s own), the same warmup and settle (so a capture is not a
picture of a half-cleared crossfade), the same seeding (so a mode has controls
in it rather than an empty state), and the same popup teardown.

Two scripts each booting the app their own way is exactly the drift
``screenshot_modes.py``'s own docstring warns about one level up, where it
derives its mode list rather than writing it out.

Every seeder writes into whatever data directory the process was pointed at, so
a harness run needs a throwaway home --- and :func:`isolate_home` below now
establishes one at import rather than leaving it as advice. That advice is what
this paragraph used to be, and it was followed by nobody: the 2026-09-07
screenshot refresh was captured against a real ``~/.warlock`` and put the
machine into the pictures.
"""

from __future__ import annotations

import ast
import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

_CONFIG_PY = Path(__file__).resolve().parent.parent / "src" / "warlock" / "config.py"


def _roots_under_home() -> tuple[str, ...]:
    """Every ``WARLOCK_*`` var whose default in ``config.py`` resolves under
    ``_home()``. -> the env var names, in source order.

    Read from ``config.py``'s own source with ``ast`` rather than hand-listed,
    because a hand list is exactly what the 2026-09-14 audit (pipelines-03)
    found stale: this module cleared six of the ten roots ``config.py``
    resolved under home, and ``WARLOCK_TRELLIS_MODELS``, ``WARLOCK_TRELLIS_
    RUNTIME``, ``WARLOCK_FAMILIAR_RUNTIME`` and ``WARLOCK_FAMILIAR_MODELS`` --
    all four added to ``config.py`` after this list was written by hand --
    leaked the real machine's paths into a throwaway home's captures. Reading
    the source instead means a fifth one enrols itself the moment it is
    written, the way ``config.SETTINGS`` already keeps its own pairing honest
    (see that table's docstring).

    A call counts when it has the shape ``_env_path("WARLOCK_X", <expr
    containing a call to _home()>)`` -- found by walking each argument's own
    subtree for a call to a function named ``_home``, not by matching text, so
    a reformatted multi-line call (``trellis_models_dir``'s, for one) is still
    found. Matching the *name* ``_home`` rather than any ``.home()`` call is
    what leaves ``WARLOCK_HOME`` itself out: its default is ``Path.home()``,
    the real interpreter, not this module's helper -- clearing the one
    variable that *makes* the throwaway home would defeat it. It is also what
    leaves ``WARLOCK_EXPORT_DIR``, ``WARLOCK_T2I_DIR`` and ``WARLOCK_GLTFPACK``
    out: each defaults to ``PROJECT_ROOT``, so a throwaway home has no opinion
    about them, same as this module has always said.
    """
    tree = ast.parse(_CONFIG_PY.read_text(encoding="utf-8"))
    roots: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        if node.func.id != "_env_path" or len(node.args) < 2:
            continue
        name_arg, default_arg = node.args[0], node.args[1]
        if not (isinstance(name_arg, ast.Constant) and isinstance(name_arg.value, str)):
            continue
        calls_home = any(
            isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Name)
            and inner.func.id == "_home"
            for inner in ast.walk(default_arg)
        )
        if calls_home:
            roots.append(name_arg.value)
    return tuple(roots)


#: Cleared rather than repointed when this module makes a throwaway home:
#: clearing lets each one derive under the new home the way ``config`` already
#: knows how, where repointing would be this module restating its defaults.
_ROOTS_UNDER_HOME = _roots_under_home()

#: Set this to run against the real library on purpose -- photographing your own
#: work, or reproducing a report that only happens with certain weights present.
#: Named for what it does rather than as ``WARLOCK_*`` so it cannot be mistaken
#: for one of the app's own settings; the app never reads it.
REAL_HOME_ENV = "WARLOCK_HARNESS_REAL_HOME"

#: Set this to keep the throwaway home instead of deleting it on the way out.
#: One caller: ``agent_bench.py --serve``, whose whole output is inside that
#: home. The 2026-09-15 Clay agent benchmark sitting is why it exists -- the pre-registration
#: it serves says in as many words to keep the exported GLB of every graded
#: session until the results document is written, and the cleanup below threw
#: both exports away the moment the app window closed. Same naming rule as
#: :data:`REAL_HOME_ENV`: not ``WARLOCK_*``, because the app never reads it.
KEEP_HOME_ENV = "WARLOCK_HARNESS_KEEP_HOME"


def isolate_home() -> Path | None:
    """Point this process at a throwaway ``WARLOCK_HOME``. -> the dir, or None.

    **Called at import, because the alternative is advice.** This module's
    docstring told callers to run against a throwaway home for as long as it
    has existed, and the 2026-09-07 screenshot refresh is what that was worth:
    ``screenshot_modes.py`` was run with the variable unset, so ``_home()``
    resolved to the developer's real ``~/.warlock`` and the captures came back
    carrying a GPU model with its free VRAM, a dozen real job cards with the
    prompts that made them, and two crash-recovery entries -- none of which
    belongs in a repository. The window size came from the same place, and with
    it the modals: at the inherited size the New map dialog was photographed
    mid-open, showing a title and an explainer where the whole form should be.

    Import time rather than inside :func:`boot`, because ``get_config()`` is
    not the only reader --- ``screenshot_modes.py`` imports ``warlock.studio``
    at module scope, and a root resolved once is resolved for the process.
    Both scripts import this module before any ``warlock`` import, which is
    what makes that ordering hold.

    **An explicit ``WARLOCK_HOME`` is left alone**, so pointing the harness at
    a prepared library stays a one-variable job; only the unset case --- the
    dangerous default, and the one nobody notices --- is redirected. Setting
    :data:`REAL_HOME_ENV` opts out of even that, and setting
    :data:`KEEP_HOME_ENV` keeps the throwaway home rather than deleting it ---
    for the one caller whose output lives inside it.

    ``WARLOCK_NO_MIGRATE`` goes on either way, and it is not belt-and-braces.
    ``migrate.run`` treats ``PROJECT_ROOT/assets``, ``bench``, ``palettes`` and
    ``models`` as legacy roots to be **moved** into ``config.home`` whenever the
    destination is empty --- which a fresh throwaway home always is. Without the
    guard, a checkout still carrying those directories would have its library
    moved into a temp dir and then deleted by the cleanup below. It is the same
    variable, for the same reason, that ``tests/conftest.py`` sets.
    """
    # Set before the first return so that every path out of here has it, and
    # unconditionally so that an explicit WARLOCK_HOME is protected too.
    os.environ.setdefault("WARLOCK_NO_MIGRATE", "1")
    if os.environ.get(REAL_HOME_ENV):
        return None
    if os.environ.get("WARLOCK_HOME"):
        return None
    home = Path(tempfile.mkdtemp(prefix="warlock-harness-"))
    os.environ["WARLOCK_HOME"] = str(home)
    for name in _ROOTS_UNDER_HOME:
        # Cleared, not left: a variable already in the environment would carry
        # that one root back out of the throwaway home, which is the half of
        # this that "set WARLOCK_HOME" alone has never covered.
        os.environ.pop(name, None)
    if os.environ.get(KEEP_HOME_ENV):
        # No cleanup at all, and deliberately no warning either: the one
        # caller that sets this prints the path itself, on the way in and on
        # the way out, because a kept home nobody is told about is a leak
        # rather than a retention.
        return home
    # ``ignore_errors`` because the app holds warlock.log and jobs.sqlite open
    # for the life of the process on Windows, and a harness that raised on the
    # way out would turn a successful capture run into a failed one.
    atexit.register(shutil.rmtree, home, True)
    return home


#: The throwaway home this process is using, or ``None`` when it was told to use
#: a real one. Read by the scripts so a run can say where it wrote.
HARNESS_HOME = isolate_home()


def boot(scale: float | None = None, size: tuple[int, int] | None = None):
    """The real App, windowed, warmed and told two harness truths. -> ``App``.

    ``recovery_offered`` and ``first_run`` are set for the reason
    ``screenshot_modes.py`` recorded when it learned them the hard way: both
    are surfaces that own the screen ahead of everything else, both are pending
    on exactly the throwaway home a harness run has, and both are raised on the
    first frame that has a Ctx --- which is inside the first capture, so
    answering them afterwards is a frame too late. Nothing is deleted either
    way: declining recovery keeps the files, and this process has no business
    adopting somebody's documents to take a photograph.
    """
    from imgui_bundle import imgui

    from warlock.config import get_config
    from warlock.studio import theme as theme_mod
    from warlock.studio import tokens
    from warlock.studio.main import App
    from warlock.studio.runtime import Runtime

    app = App(Runtime(get_config()))
    app.setup_window(size_override=size)
    if scale is not None:
        # After the window (which samples the monitor) and before the context
        # (which makes textures): the atlas has to be re-baked at the new scale
        # or every icon sits off-centre by a fraction of the difference, which
        # is ``fonts.reload``'s whole reason for existing. Between frames is
        # satisfied trivially here -- there has not been one yet.
        from warlock.studio import fonts

        tokens.set_scale(scale)
        theme_mod.apply(imgui)
        fonts.reload(imgui)
    app.setup_runtime()
    app.setup_context()
    app.app_ctx.state.recovery_offered = True
    app.app_ctx.first_run = False
    return app


# Frames drawn before the read. Three is not a guess: one to build, one for the
# textures asked for on it to upload, one for anything those made visible.
WARMUP = 3

# How many further frames a capture will wait for the app to stop moving. A
# mode change raises a content crossfade (UX.md Phase 1) and three warmup frames
# is 50 ms of a 200 ms one, so without this every capture is a picture of a
# half-cleared veil -- a harness that made the whole screenshot pass useless in
# exactly the phase it exists to review. Bounded rather than a bare ``while``:
# an animation that never settles is a bug this must report by capturing it,
# not hang on.
SETTLE_FRAMES = 40


def capture(app, path: Path) -> None:
    import pygame
    from PIL import Image

    from warlock.studio import motion

    for _ in range(WARMUP):
        app.frame(1.0 / 60.0)
        pygame.display.flip()
    for _ in range(SETTLE_FRAMES):
        if not motion.animating():
            break
        app.frame(1.0 / 60.0)
        pygame.display.flip()
    width, height = pygame.display.get_window_size()
    data = app.ctx.screen.read(components=3, alignment=1)
    # GL's origin is bottom-left and everybody else's is top-left.
    image = Image.frombytes("RGB", (width, height), data).transpose(Image.FLIP_TOP_BOTTOM)
    image.save(path)
    print(f"  {path.name}", flush=True)


def close_popups(app) -> None:
    """Reset every transient surface after an isolated popup capture."""
    import pygame
    from imgui_bundle import imgui

    from warlock.studio import matte_preview, plotter_mode

    ctx = app.app_ctx
    while ctx.confirms.pending is not None:
        ctx.confirms.dismiss()
    while ctx.prompts.pending is not None:
        ctx.prompts.dismiss()
    matte_preview.close(ctx)
    plotter_mode.ensure(ctx).setup_pending = False
    # Guarded. ``close_popup_to_level`` walks the open-popup stack and trips an
    # IM_ASSERT when there is nothing on it, and the last capture of the run
    # reaches here with the stack already empty -- so the whole ``--popups``
    # pass aborted on its final step, *after* writing its images, which is why
    # it read as a crash with a complete-looking output directory. The public
    # any-popup query is the cheap way to ask before walking.
    any_popup = imgui.PopupFlags_.any_popup_id.value | imgui.PopupFlags_.any_popup_level.value
    if imgui.is_popup_open("", any_popup):
        imgui.internal.close_popup_to_level(0, True)
    # Let popup owners observe the close before another owner opens one under
    # the same host window.
    app.frame(1.0 / 60.0)
    pygame.display.flip()


def seed_matte(app) -> None:
    """Put a deterministic, already-computed matte in the model modal."""
    from PIL import Image, ImageDraw

    from warlock.service.matte import Preview, stamp_for
    from warlock.studio import matte_preview

    ctx = app.app_ctx
    job_id = ctx.svc.store.create(
        "text",
        "a hooded adventurer standing",
        {"seed": 7},
        stage="reference",
        status="done",
    )
    job_dir = ctx.svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (320, 320), (90, 110, 150, 255)).save(job_dir / "input.png")

    preview = Image.new("RGB", (320, 320), (180, 180, 180))
    draw = ImageDraw.Draw(preview)
    cell = 16
    for y in range(0, 320, cell):
        for x in range(0, 320, cell):
            if (x // cell + y // cell) % 2:
                draw.rectangle((x, y, x + cell - 1, y + cell - 1), fill=(140, 140, 140))
    draw.ellipse((72, 24, 248, 306), fill=(82, 103, 148))
    draw.ellipse((118, 42, 202, 132), fill=(198, 171, 145))

    stamp = stamp_for(job_dir / "input.png")
    result = Preview(
        job_id=job_id,
        stamp=stamp,
        width=preview.width,
        height=preview.height,
        rgb=preview.tobytes(),
        source="birefnet",
        approved=False,
        coverage=0.47,
        warnings=("Check the fine edges around the hood before building.",),
    )
    state = matte_preview.open_for(ctx, job_id, {"mesh_seed": 11})
    state.stamp = stamp
    state.preview = result
    state.cache[job_id] = result


def seed(app) -> None:
    """Open a canvas and a model, so the panes that need one are not empty.

    Inker and Clay both draw an empty-state pane with nothing open, which is
    exactly the frame that shows none of the controls Phase 4 added -- the tool
    grid's options, the properties panel's sections, the timeline. Both entry
    points are the ones the buttons call.

    The canvas is **animated**, which this claimed to cover and did not: the
    timeline strip is drawn only for a document with an ``anim``, so a plain
    new canvas left the app's densest row -- the transport, the frame
    operations, the exports and their controls -- out of every capture this
    harness has ever taken. That row was rewritten in the UI redesign, wave 4.2
    precisely because it was clipping at 150 %, which is the defect class the
    scale pass exists to find.

    And a map and an atlas, for the same reason one wave later (the UI redesign,
    wave 6). Plotter and Packwright are four panes each, all four of which
    answer "Open or start a map first" with nothing open -- so the sentence-case
    sweep over their eleven headings, the tool grids, the layer tree and the
    packing controls had never appeared in a capture at any scale. Both
    ``new_document`` calls are synchronous; sprites are not, because they land
    through ``ctx.submit`` on a task thread, and a seeder that races the capture
    is worse than one that stops short of it.

    **The map gets a tileset and a few painted cells** (2026-09-01), and that is
    a third instance of the same defect rather than a nicety. A map with no
    tileset draws the palette's "add one first" branch and nothing else -- so
    the tile picker, the tileset bar, the brush transforms on the toolbar and the
    whole Tile stamps pane had never appeared in a capture at any scale, at
    exactly the moment those four were rewritten. The tileset is built in
    process rather than imported: ``add_tileset`` is synchronous where every
    file door goes through the task thread.

    **And a scene** (Mason, Stage E), which is the same defect a fourth time and
    was measured rather than guessed: the first ``exercise_mode --mode mason``
    run reported 41 of 60 controls as ``hard-reset``, every one carrying the
    identical delta ``documents: () -> ('ms1',)``. With no scene open Mason
    draws its empty state, so the *first* press of any control -- a primitive
    button, a tool, a pivot -- mints the document, and no undo takes a document
    back out of existence. Every verdict in that run was the driver reporting
    its own missing seed, which is exactly what this docstring has had to say
    three times already.
    """
    import numpy as np

    from warlock.studio import (
        clay_mode,
        inker_mode,
        mason_mode,
        packwright_mode,
        plotter_mode,
    )
    from warlock.studio.tilegrid import gid
    from warlock.studio.tilegrid.tileset import Tileset

    inker_mode.new_document(app.app_ctx, 1024, 1024)
    state = inker_mode.ensure(app.app_ctx)
    if state.active is not None:
        inker_mode.animate(app.app_ctx, state.active)
    clay_mode.new_document(app.app_ctx)
    mason_mode.new_document(app.app_ctx)
    _seed_mason(app.app_ctx, mason_mode)
    packwright_mode.new_document(app.app_ctx)

    tab = plotter_mode.new_document(app.app_ctx)
    doc = tab.doc
    # A four-by-four atlas of flat colours: enough tiles for the picker to be a
    # picker, and deterministic, which is what a screenshot corpus needs.
    tile = 32
    pixels = np.zeros((tile * 4, tile * 4, 4), dtype=np.uint8)
    pixels[..., 3] = 255
    for row in range(4):
        for column in range(4):
            shade = 40 + 24 * (row * 4 + column) % 200
            block = pixels[
                row * tile : (row + 1) * tile, column * tile : (column + 1) * tile
            ]
            block[..., 0] = shade
            block[..., 1] = 90 + (row * 30) % 140
            block[..., 2] = 200 - (column * 40) % 160
    ref = doc.add_tileset(Tileset(name="terrain", pixels=pixels, tile_w=tile, tile_h=tile))
    layer = doc.tile_layers()[0]
    cells = np.zeros((doc.height, doc.width), gid.DTYPE)
    for row in range(min(6, doc.height)):
        for column in range(min(10, doc.width)):
            cells[row, column] = gid.compose(ref.firstgid + (row + column) % 4)
    doc.write_region(layer.uid, 0, 0, cells)
    # A brush in hand and a stamp in a slot, so the toolbar's transforms are
    # live and the stamps pane has something to draw rather than nine empties.
    plotter_state = plotter_mode.ensure(app.app_ctx)
    plotter_state.brush = cells[0:2, 0:2].copy()
    doc.set_stamp(1, plotter_state.brush, name="Grass corner")
    doc.mark_saved()


def seed_tile(app, png: Path) -> None:
    """A finished tile job holding ``png``, selected, with 2D showing it.

    Enough of a job for the tile-only controls to be on screen. Writes into
    whatever data directory the process was pointed at, so run this against a
    throwaway ``WARLOCK_DATA_DIR`` rather than a real library.
    """
    import shutil

    ctx = app.app_ctx
    job_id = ctx.svc.store.create("text", "cobblestone", {"seed": 11}, stage="tile", status="done")
    job_dir = ctx.svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(png, job_dir / "input.png")
    from warlock.pipelines import seam

    ctx.svc.store.merge_params(job_id, {"seam_report": seam.report(job_dir / "input.png")})
    ctx.cache.invalidate()
    ctx.cache.tick()
    ctx.state.select(job_id)


def seed_asset(app) -> None:
    """A finished reference and a rigged mesh promoted from it, mesh selected.

    Create's five stages are four columns and an inspector *about an asset*,
    and with nothing selected four of the five draw an empty state -- so the
    mode pass had never photographed the Rig column, the Pose column, the
    export grid, the lineage links, or a rail with any segment ticked. That is
    the gap ``--seed`` closed for Inker and ``--review`` for the verdict panel,
    and this is the same hole one wave later.

    The GLB is a **real** one, written by the app's own exporter: a stub would
    fail to parse on the frame that shows it and put an error toast in every
    capture.

    Writes into whatever data directory the process was pointed at, so run it
    against a throwaway ``WARLOCK_DATA_DIR`` rather than a real library.
    """
    from PIL import Image

    from warlock.studio.clay import document as bd
    from warlock.studio.clay import primitives as bp
    from warlock.studio.viewer import glbwrite

    ctx = app.app_ctx
    ref_id = ctx.svc.store.create(
        "text", "a hooded adventurer standing", {"seed": 7}, stage="reference", status="done"
    )
    ref_dir = ctx.svc.job_dir(ref_id)
    ref_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (256, 256), (90, 110, 150, 255)).save(ref_dir / "input.png")

    doc = bd.ClayDoc()
    doc.objects.append(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box()))
    glb = glbwrite.write_glb(bd.to_model(doc))

    mesh_id = ctx.svc.store.create(
        "image",
        "a hooded adventurer standing",
        {"seed": 7, "mesh_seed": 11},
        stage="model",
        status="done",
        parent_id=ref_id,
    )
    mesh_dir = ctx.svc.job_dir(mesh_id)
    mesh_dir.mkdir(parents=True, exist_ok=True)
    (mesh_dir / "model.glb").write_bytes(glb)
    (mesh_dir / "source.glb").write_bytes(glb)
    # rig.glb is what the Rig and Pose stages gate on. Its own GLB rather than
    # a copy of nothing, for the parse reason above.
    (mesh_dir / "rig.glb").write_bytes(glb)
    Image.new("RGBA", (256, 256), (90, 110, 150, 255)).save(mesh_dir / "input.png")
    ctx.cache.invalidate()
    ctx.cache.tick()
    ctx.state.select(mesh_id)


def seed_review(app) -> None:
    """A finished mesh in the recent-unreviewed bucket, open in Review.

    Review's empty state shows none of the verdict panel: no grade row, no tag
    toggles, no recorded line. That is the same gap ``--seed`` closes for Inker
    and Clay, and it matters more here because the grade row is eleven buttons
    wrapping inside a 300 px sidebar -- ``same_line`` past the content region
    clips rather than wrapping, which is the bug that once hid seven controls,
    and it is invisible to the smoke suite because that asserts only that a pane
    builds.

    Writes into whatever data directory the process was pointed at, so run it
    against a throwaway ``WARLOCK_DATA_DIR`` rather than a real library.
    """
    from warlock.studio import review_mode

    ctx = app.app_ctx
    job_id = ctx.svc.store.create(
        "image",
        "a wooden chest",
        {"lora_weight": 0.9, "seed": 42},
        stage="model",
        status="done",
    )
    ctx.svc.job_dir(job_id).mkdir(parents=True, exist_ok=True)
    ctx.cache.invalidate()
    ctx.cache.tick()
    # Through the scan the Rescan button runs, rather than by building units by
    # hand: a second way to populate this list is a second thing to keep true.
    # It is a *task*, so frames have to be pumped until it lands -- and the
    # staged tag has to be set after that, because the scan's completion opens
    # the bucket and opening disarms. That is the product behaviour (a rescan
    # moves you off the unit, so what you had staged for it goes with it), so
    # the harness waits rather than the rule bending.
    review_mode.scan(ctx)


def seed_troupe(app) -> None:
    """A finished character sheet, selected, so Troupe is not in its empty state.

    Without this the mode exercises nothing it exists for: the Sheet pane, the
    frame table and every handoff in ``sheet_panel`` are drawn only for a
    selected character, so the 2026-08-23 pass covered the *empty* Troupe and
    said so. Everything the preview reads is a file, not a render, which is why
    this can exist at all -- a real character costs an image, a reconstruction,
    a rig and 256 rendered cells behind a GPU.

    **Built through the app's own builders**, not by hand-writing JSON.
    ``clips.expand_clips`` and ``charsheet.plan`` need the shipped clip library
    and nothing else -- no weights, no card, and about a millisecond -- so the
    sidecar this writes is the one the worker would write, tags and runs and
    all. A hand-rolled sidecar would be a second dialect of the format and
    would go stale the first time the real one changed.

    The PNG is drawn here rather than rendered, and deliberately *varies per
    cell*: a flat fill would make every frame of every direction identical, and
    a driver that presses Play, steps a frame and turns the character would
    photograph one unchanging picture and call all three controls dead.
    """
    import json
    import time

    from PIL import Image, ImageDraw

    from warlock import clips, rigging
    from warlock.pipelines import charsheet
    from warlock.pipelines import sheet as sheetlib
    from warlock.service import troupe as svc_troupe
    from warlock.studio import troupe_mode

    ctx = app.app_ctx
    layout = charsheet.resolve_layout()
    # The door's own template, not the string "humanoid": it is pinned there
    # because it is the only one the clip library carries a walk for, and a
    # seeder that names it a second time is a second place for that to be true.
    plan = charsheet.plan(
        clips.expand_clips(svc_troupe.TROUPE_TEMPLATE, layout),
        frame_size=32,
        layout=layout,
    )

    source_id = ctx.svc.store.create(
        "image",
        "a fire guardian",
        {"seed": 7},
        stage="model",
        status="done",
    )
    job_dir = ctx.svc.job_dir(source_id)
    job_dir.mkdir(parents=True, exist_ok=True)

    sheet_id = rigging.new_id()
    png = rigging.sheet_png_path(job_dir, sheet_id)
    image = Image.new("RGBA", (plan.width, plan.height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    for index, cell in enumerate(plan.cells):
        x, y = cell.x, cell.y
        size = plan.frame_size
        # A head and a body, with the head tracking the cell index: adjacent
        # frames differ, so stepping and playing are visible in a screenshot.
        sway = (index % 4) - 1.5
        cx = x + size / 2 + sway
        draw.ellipse(
            (cx - size * 0.14, y + size * 0.12, cx + size * 0.14, y + size * 0.40),
            fill=(226, 232, 240, 255),
        )
        draw.rectangle(
            (cx - size * 0.11, y + size * 0.42, cx + size * 0.11, y + size * 0.86),
            fill=(148, 163, 184, 255),
        )
    png.parent.mkdir(parents=True, exist_ok=True)
    image.save(png)

    meta = sheetlib.sidecar(
        plan,
        sheet_id=sheet_id,
        source_job=source_id,
        image=png.name,
        created=time.time(),
        name="a fire guardian",
        animation=charsheet.animation_block(layout),
    )
    meta["troupe"] = layout.as_dict()
    # The sidecar is the completion marker and is written last, which is
    # ``list_sheets``' own rule: a sidecar with no PNG beside it is skipped.
    rigging.sheet_path(job_dir, sheet_id).write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    ctx.svc.store.create(
        "charsheet",
        "a fire guardian",
        {"source_job": source_id, "sheet_id": sheet_id, "layout": layout.as_dict()},
        stage="sheet",
        status="done",
        parent_id=source_id,
    )
    ctx.cache.invalidate()
    ctx.cache.tick()
    troupe_mode.open_sheet(ctx, source_id, sheet_id)


def _write_demo_figure(doc) -> None:
    """A rising figure on channel one and a pulse under it, in one document.

    Shared by :func:`seed_sirens` and :func:`seed_muse` because they want the
    same thing for opposite reasons: the tracker needs a pattern grid with
    something in it, and the take tray needs a WAV that is not silence -- a
    waveform of an empty song is a flat line, which photographs as a bug.

    Written through ``set_cells``, the one door every pattern write uses.
    Poking ``cells`` directly would be a second way to edit a document, which
    is what that method's docstring exists to prevent.
    """
    import numpy as np

    from warlock.studio.sirens import document as sirens_doc
    from warlock.studio.sirens import notes

    pattern = doc.patterns[0]
    rows = min(16, pattern.rows)
    channels = min(2, pattern.channels)
    block = np.full((rows, channels, sirens_doc.COLUMNS), notes.EMPTY, dtype=np.int16)
    scale = (0, 2, 4, 5, 7, 9, 11, 12)
    for row in range(rows):
        block[row, 0, sirens_doc.NOTE] = notes.A4_NOTE - 9 + scale[row % len(scale)]
        block[row, 0, sirens_doc.INSTRUMENT] = 0
        block[row, 0, sirens_doc.VOLUME] = 48 + (row % 4) * 4
        if channels > 1 and row % 4 == 0:
            block[row, 1, sirens_doc.NOTE] = notes.A4_NOTE - 21
            block[row, 1, sirens_doc.INSTRUMENT] = min(1, len(doc.instruments) - 1)
    doc.set_cells(pattern.uid, 0, 0, 0, block)


def seed_sirens(app) -> None:
    """A song with notes in it, so the tracker is not six empty panes.

    Sirens is six panes and every one of them answers "start a song or open a
    ``.wsng``" with nothing open, so the corpus held three pictures of a mode
    with no pattern grid, no instrument list, no envelope editor and no order
    list -- the same gap ``seed`` closes for Inker, Clay, Plotter and
    Packwright, and the same argument that made ``seed_troupe`` exist.

    ``new_document`` is the entry point the New song button calls.

    **Nothing sounds.** ``sirens_audio`` is never touched: a capture has
    nothing to hear, and opening an audio device inside a screenshot pass is a
    side effect on the machine that ran it.
    """
    from warlock.studio import sirens_mode

    tab = sirens_mode.new_document(app.app_ctx)
    _write_demo_figure(tab.doc)
    # Saved rather than left dirty: an unsaved marker in every Sirens picture
    # is a fact about the harness, not about the mode.
    tab.doc.mark_saved()


def seed_muse(app) -> None:
    """Two finished takes and a decoded player, so Muse has a take tray.

    Muse's results pane draws "No takes yet" until a music row exists, and
    ``muse_player.should_draw`` keeps the transport strip off screen until one
    has actually been auditioned -- so three of the mode's four surfaces were
    absent from every capture: the tray's cards, their Play/Stems/derive
    controls, and the whole full-width player with its waveform, playhead and
    loop markers.

    The audio is **made here**, by the app's own synth and WAV writer, because
    a real take is a GPU and a 23 GB download: ``muse_mode`` already pairs
    ``synth.render`` with ``wavout.wav_bytes`` for its reference tracks, so
    this is that pairing and not a second way to make a WAV.

    The player is built the way ``muse_mode.on_task_done`` builds it, minus
    ``sirens_audio.play``: the strip needs a decoded buffer, and a screenshot
    has nothing to hear. Going through ``muse_mode.play`` instead would open an
    audio device on the machine running the pass and put the read on a task
    thread the capture would race.
    """
    from warlock.studio import muse_io, muse_mode
    from warlock.studio.muse_state import Player
    from warlock.studio.sirens import document as sirens_doc
    from warlock.studio.sirens import synth, wavout

    ctx = app.app_ctx
    song = sirens_doc.new_song()
    _write_demo_figure(song)
    samples, loop = synth.render(song)
    wav = wavout.wav_bytes(samples, synth.SAMPLE_RATE, loop=loop)
    takes = (
        ("dungeon ambience, low strings, slow", 12345, ""),
        ("dungeon ambience, low strings, slow, harp", 12346, "vary"),
    )
    first = ""
    for prompt, seed, task in takes:
        params = {"duration": 60.0, "seed": seed}
        if task:
            params["task"] = task
        job_id = ctx.svc.store.create(
            "music",
            prompt,
            params,
            # ``stage="music"``, which is what ``_jobs_music`` writes and not
            # ``create``'s ``"model"`` default. The difference is not cosmetic:
            # ``JobStore.unverdicted_models`` selects on ``stage = 'model'``
            # with no kind filter, so a take seeded at the default stage walks
            # straight into Review's recent-unreviewed bucket and is offered
            # for mesh grading -- five units, one of which answers "No mesh for
            # this unit". The first Muse capture showed exactly that.
            stage="music",
            status="done",
            parent_id=first or None,
        )
        ctx.svc.job_dir(job_id).mkdir(parents=True, exist_ok=True)
        muse_mode.track_path(ctx, job_id).write_bytes(wav)
        first = first or job_id
    ctx.cache.invalidate()
    ctx.cache.tick()
    state = muse_mode.ensure(ctx)
    state.selected_job = first
    track = muse_io.read_track(muse_mode.track_path(ctx, first))
    state.player = Player(
        job=first,
        pcm=track["pcm"],
        rate=int(track["rate"]),
        env=track.get("env"),
        duration=float(track.get("duration", 0.0)),
    )


def seed_palette(app) -> None:
    """One authored palette on disk, so the Palette combo exists to photograph.

    ``settings_2d._pixel_look`` draws that combo only when the palette folder
    has something in it -- palettes are opt-in and the honest rendering of
    "none installed" is no control at all. A capture home is fresh, so the
    control had never been in a picture: not the combo, and not the two
    branches of the Dither helper below it, which says a different sentence
    depending on whether a palette is chosen.

    Written with the loader's own suffix rather than a hardcoded ``.hex``, for
    the reason the helper line beside that combo records: this repo has already
    shipped a sentence naming suffixes ``palettes.SUFFIXES`` did not carry.
    """
    from warlock.service import palettes

    directory = Path(app.app_ctx.svc.config.palette_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"dawnlight{palettes.SUFFIXES[0]}").write_text(
        "\n".join(("1a1c2c", "5d275d", "b13e53", "ef7d57", "ffcd75", "f4f4f4")) + "\n",
        encoding="utf-8",
    )


def seed_sheet_form(app, *, arm: str) -> None:
    """Park Create's 2D form on a sheet arm, with Advanced open.

    The reason this exists at all: ``--asset`` seeds a *mesh*, and the form
    remembers ``asset_type = Image``, so every Create capture this corpus has
    ever held was taken on the one arm that draws none of the sheet controls.
    The Tile-layout picker, the materials list, the terrain fields, the sprite
    Action and Directions rows, the target cell, Palette, Dither and Outline
    are each on a branch that never executed -- a set of pictures that looks
    like coverage of the form and covers none of it.

    ``arm`` is a generation type from ``generation.GENERATION_TYPES``. Only
    ``asset_type`` is written: ``settings_2d.draw`` runs
    ``create_assets.sync_legacy_fields`` on every frame, so ``output`` and
    ``sheet_type`` are derived rather than set here -- two places setting them
    is how a form ends up in a state the pane cannot reach.
    """
    from warlock.studio import create_stages

    state = app.app_ctx.state
    state.form_2d["asset_type"] = arm
    state.form_2d["generation_type"] = arm
    state.mode = create_stages.MODE
    state.create_stage = "reference"
    # The sheet Dimensions section lives inside the disclosure, and the
    # disclosure is AppState rather than settings, so it starts shut.
    state.create_advanced = True


def _seed_mason(ctx: object, mason_mode: object) -> None:
    """A ground and a prefab in the seeded scene, for the reason the empty
    document was seeded at all (Stage F).

    Half of Mason's controls are behind a condition rather than behind a
    document: the brush row only exists once the scene has a terrain, and the
    whole Prefabs pane is a conditional slot that is not in the column until the
    scene defines a template. Without both, ``exercise_mode --mode mason`` drives
    every control in the mode **except** the ones Stage F added, and reports full
    coverage -- which is the same blind spot, one layer up, that the missing
    document itself was.
    """
    from warlock.studio.mason import nodes as nd

    mason_mode.add_terrain(ctx)
    tab = mason_mode.active(ctx)
    if tab is None:
        return
    doc = tab.doc
    node = doc.add_node(nd.MeshNode(uid=nd.new_uid(), name="Prop"))
    doc.select([node.uid])
    mason_mode.define_prefab_from_selection(ctx, "Prop")
    doc.select([])
