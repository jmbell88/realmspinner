"""The optional gallery: one PNG per accepted record, three-quarter and
front views side by side, so a human can eyeball what a family actually
built without opening every GLB by hand.

Uses the exact path ``tests/test_clay_view.py`` uses with no window --
``moderngl.create_context(standalone=True, require=330)`` (here,
``headless.open_gl()``) feeding ``clay_view.ClayView`` directly -- and
``ClayView.render_png`` itself, the same function ``agent_clay``'s own
``clay_render`` tool calls. Skips cleanly wherever there is no GPU: see
``build.py``'s own handling of a ``None`` from ``headless.open_gl()``.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from PIL import Image  # noqa: E402

from warlock.studio import clay_view  # noqa: E402


def render_gallery(gl: Any, doc: Any, path: Path, *, size: int = 512) -> None:
    """Render *doc* from ``three_quarter`` and ``front``, side by side, to
    *path* as one PNG. ``app_ctx=None`` -- exactly how
    ``agent_clay._view_for`` builds its own private viewport -- since a
    gallery render wants no gizmos and no live app setting, only the doc.
    """
    view = clay_view.ClayView(gl)
    try:
        three_quarter = view.render_png(doc, size=size, view="three_quarter")
        front = view.render_png(doc, size=size, view="front")
    finally:
        view.release()

    img_a = Image.open(io.BytesIO(three_quarter)).convert("RGB")
    img_b = Image.open(io.BytesIO(front)).convert("RGB")
    canvas = Image.new("RGB", (size * 2, size), "white")
    canvas.paste(img_a, (0, 0))
    canvas.paste(img_b, (size, 0))

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
