"""Engine snippets: the few lines that load a baked sheet, per engine.

``pipelines.sheet.sidecar`` is engine-neutral on purpose (its docstring says
why), so what an engine needs is *derived from* the sidecar here and never
written into it. Each renderer takes the same small mapping -- what the
export wrote -- and returns text the user pastes. Pure string work; a test
executes the Pygame one against a stub and string-checks the rest.
"""

from __future__ import annotations

from typing import Any

ENGINES = ("pygame-ce", "godot", "unity", "phaser")

# ``pipelines.sheet.MAX_ATLAS_PX``, restated rather than imported: this
# package's import pin (``tests/inker/flourish/test_flourish_imports.py``)
# allows ``warlock.pipelines`` only from ``bake.py``, the way ``curves.py``
# restates the easing table rather than reaching into ``pipelines.sheet`` for
# it. The per-tag export (``sheetout.arrange`` with no ``arrange`` chosen,
# which is the default a Flourish export uses) wraps a strip of frames into
# more rows once one row would cross this width -- see ``_grid`` below.
_MAX_ATLAS_PX = 8192


def _grid(frame_width: int, frames: int) -> tuple[int, int]:
    """``(columns, rows)`` for ``frames`` cells of ``frame_width``, matching
    ``sheetout.arrange``'s default row-wrap exactly -- the layout the per-tag
    export actually writes when nobody picks a different ``arrange``.

    The 2026-09-07 audit (inker-07) found the Pygame and Godot snippets
    hard-coded a single-row strip (``n * frame_width, 0``) while this wrap is
    what the exported PNG really uses past ``_MAX_ATLAS_PX`` -- so a long
    effect's pasted snippet either read the wrong frames or threw slicing off
    the edge of the image.
    """
    columns = max(1, min(frames, _MAX_ATLAS_PX // max(1, frame_width)))
    rows = -(-frames // columns)  # ceil
    return columns, rows


def describe(
    *,
    name: str,
    image: str,
    frame_width: int,
    frame_height: int,
    frames: int,
    fps: int,
    loop: bool,
    origin: tuple[int, int],
) -> dict[str, Any]:
    """The mapping every snippet reads. One shape, so a caller cannot hand
    Godot a different frame count -- or a different grid -- than Pygame."""
    columns, rows = _grid(int(frame_width), int(frames))
    return {
        "name": str(name),
        "image": str(image),
        "frame_width": int(frame_width),
        "frame_height": int(frame_height),
        "frames": int(frames),
        "fps": int(fps),
        "loop": bool(loop),
        "origin": [int(origin[0]), int(origin[1])],
        "columns": columns,
        "rows": rows,
    }


def snippet(engine: str, info: dict[str, Any]) -> str:
    if engine not in ENGINES:
        raise ValueError(f"engine must be one of {list(ENGINES)}")
    return _RENDERERS[engine](info)


def _comment_safe(text: str) -> str:
    """A name safe to splice into a single-line ``#``/``//`` header comment.

    The 2026-09-15 audit (inker-06) found every engine's header comment
    spliced the raw effect name in with no sanitizing at all -- the
    2026-09-14 audit's inker-10 escaped only Godot's quoted string literals,
    so a newline in the name (nothing stops a user naming an effect that)
    still closed the comment early and let the rest of the name splice live
    code into the pasted snippet, in every engine including Godot's own
    header. A comment has no quoting to get right, only one rule: no newline
    may reach the line it is on.
    """
    return text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")


def _pygame(i: dict[str, Any]) -> str:
    ident = _ident(i["name"])
    head = f'{i["frames"]} frames of {i["frame_width"]}x{i["frame_height"]} at {i["fps"]} fps'
    columns = i["columns"]
    return f'''# {_comment_safe(i["name"])}: {head}
import pygame

class Animation:
    def __init__(self, spritesheet, frame_size, frame_count, fps, loop, origin, columns):
        sheet = pygame.image.load(spritesheet).convert_alpha()
        w, h = frame_size
        # The sheet wraps into more rows once one row would cross the atlas
        # width ceiling (the 2026-09-07 audit, inker-07) -- ``columns`` is the
        # per-tag export's own row-wrap, not always every frame in one row.
        self.frames = [
            sheet.subsurface(((n % columns) * w, (n // columns) * h, w, h))
            for n in range(frame_count)
        ]
        self.fps, self.loop, self.origin = fps, loop, origin
        self.time = 0.0

    def update(self, dt):
        self.time += dt

    @property
    def done(self):
        return not self.loop and self.time * self.fps >= len(self.frames)

    def draw(self, surface, pos):
        index = int(self.time * self.fps)
        index = index % len(self.frames) if self.loop else min(index, len(self.frames) - 1)
        ox, oy = self.origin
        surface.blit(self.frames[index], (pos[0] - ox, pos[1] - oy))

{ident} = Animation(
    spritesheet="{i["image"]}",
    frame_size=({i["frame_width"]}, {i["frame_height"]}),
    frame_count={i["frames"]},
    fps={i["fps"]},
    loop={i["loop"]},
    origin=({i["origin"][0]}, {i["origin"][1]}),
    columns={columns},
)
'''


def _gdscript_string(text: str) -> str:
    """Escape ``text`` for a GDScript double-quoted string literal. The
    2026-09-14 audit (inker-10) found the effect name spliced into the
    Godot snippet's ``"..."`` literals with no escaping at all: a quote or
    backslash in the name (nothing stops a user naming an effect ``Bob's
    "big" swing``) closed the literal early and broke the pasted script."""
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _godot(i: dict[str, Any]) -> str:
    ident = _ident(i["name"])
    name = _gdscript_string(i["name"])
    fw, fh = i["frame_width"], i["frame_height"]
    ox, oy = fw / 2 - i["origin"][0], fh / 2 - i["origin"][1]
    columns = i["columns"]
    # ``name``, not ``i["name"]``: the 2026-09-15 audit (inker-06) found this
    # header spliced the *raw* name in while every string literal below it
    # used the escaped one -- the already-GDScript-escaped value has no real
    # newline left in it either, so it is safe here for the same reason.
    return f'''# {name}: build SpriteFrames from the sheet at runtime (Godot 4)
var {ident} := SpriteFrames.new()

func _ready() -> void:
    var sheet: Texture2D = load("res://{i["image"]}")
    {ident}.add_animation("{name}")
    {ident}.set_animation_speed("{name}", {i["fps"]})
    {ident}.set_animation_loop("{name}", {"true" if i["loop"] else "false"})
    # The sheet wraps into more rows once one row would cross the atlas width
    # ceiling (the 2026-09-07 audit, inker-07) -- {columns} is the per-tag
    # export's own row-wrap, not always every frame in one row.
    for n in {i["frames"]}:
        var atlas := AtlasTexture.new()
        atlas.atlas = sheet
        atlas.region = Rect2((n % {columns}) * {fw}, (n / {columns}) * {fh}, {fw}, {fh})
        {ident}.add_frame("{name}", atlas)
    $AnimatedSprite2D.sprite_frames = {ident}
    $AnimatedSprite2D.offset = Vector2({ox}, {oy})
    $AnimatedSprite2D.play("{name}")
'''


def _unity(i: dict[str, Any]) -> str:
    grid = f'{i["frame_width"]}x{i["frame_height"]}'
    pivot = f'({i["origin"][0]}, {i["origin"][1]})'
    name = _comment_safe(i["name"])
    return f'''// {name}: slice {i["image"]} in the Sprite Editor as a {grid} grid
// ({i["frames"]} cells, pivot at {pivot} px), then drive it from a script:
using UnityEngine;

public class {_ident(i["name"], pascal=True)}Player : MonoBehaviour
{{
    public Sprite[] frames;          // the {i["frames"]} sliced sprites, in order
    public float fps = {i["fps"]}f;
    public bool loop = {"true" if i["loop"] else "false"};
    SpriteRenderer sr; float t;

    void Awake() {{ sr = GetComponent<SpriteRenderer>(); }}

    void Update()
    {{
        t += Time.deltaTime;
        int index = (int)(t * fps);
        if (loop) index %= frames.Length;
        else if (index >= frames.Length) {{ Destroy(gameObject); return; }}
        sr.sprite = frames[index];
    }}
}}
'''


def _phaser(i: dict[str, Any]) -> str:
    ident = _ident(i["name"])
    fw, fh = i["frame_width"], i["frame_height"]
    return f'''// {_comment_safe(i["name"])}: Phaser 3
preload() {{
    this.load.spritesheet("{ident}", "{i["image"]}", {{ frameWidth: {fw}, frameHeight: {fh} }});
}}

create() {{
    this.anims.create({{
        key: "{ident}",
        frames: this.anims.generateFrameNumbers("{ident}", {{ start: 0, end: {i["frames"] - 1} }}),
        frameRate: {i["fps"]},
        repeat: {-1 if i["loop"] else 0},
    }});
    const sprite = this.add.sprite(x, y, "{ident}");
    sprite.setOrigin({i["origin"][0] / fw:.4f}, {i["origin"][1] / fh:.4f});
    sprite.play("{ident}");
}}
'''


_RENDERERS = {"pygame-ce": _pygame, "godot": _godot, "unity": _unity, "phaser": _phaser}


def _ident(name: str, *, pascal: bool = False) -> str:
    parts = [p for p in "".join(c if c.isalnum() else " " for c in name).split() if p]
    if not parts:
        parts = ["effect"]
    if pascal:
        return "".join(p[:1].upper() + p[1:] for p in parts)
    ident = "_".join(p.lower() for p in parts)
    return ident if not ident[0].isdigit() else f"fx_{ident}"
