"""The 2026-10-14 audit, fixer F57: the assistant dock's attach door.

Three findings (familiar-host-01, -05, -06), all on ``studio/assistant/ui.py``'s path from "Send
pressed with a path typed" to the task that carries the image. Each test names the claim it pins.
"""

from __future__ import annotations

import threading
from io import BytesIO

import pytest

from realmspinner.service import familiar as svc_familiar
from realmspinner.service.errors import Invalid
from realmspinner.studio.assistant import ui as familiar_ui
from realmspinner.studio.tasks import Done

from .test_familiar_ui import _FakeCtx, _real_png


def _user_turns(ctx) -> list:
    return [t for turns in ctx.familiar_threads._data.values() for t in turns if t.role == "user"]


# --- familiar-host-01 -----------------------------------------------------------------------


def test_a_refused_attach_leaves_no_user_turn_in_the_thread(tmp_path):
    """Both submit doors appended the user's turn before the attach could be refused, so every
    retry press added "You: ..." again for a message that was never sent."""
    ctx = _FakeCtx(mode="clay")
    ui = familiar_ui.ensure(ctx)
    ui.attach_path = str(tmp_path / "does-not-exist.png")

    for _ in range(3):
        assert familiar_ui.submit_chat(ctx, "what is this?") is False
        assert familiar_ui.submit_build(ctx, "build a box") is False

    assert _user_turns(ctx) == []
    assert ui.attach_error is not None and "not a file" in ui.attach_error


# --- familiar-host-05 -----------------------------------------------------------------------


def test_an_attached_image_over_the_pixel_limit_is_refused_with_a_sentence_not_raised(
    monkeypatch, tmp_path
):
    """``PIL.Image.DecompressionBombError`` is a bare ``Exception``: it escaped the
    ``(UnidentifiedImageError, OSError, ValueError)`` clause and the docstring's "never a raw PIL
    traceback" was false."""
    from PIL import Image

    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)  # 64x64 = 4096 px is past twice that
    data = _real_png((64, 64))

    with pytest.raises(ValueError, match="not a readable image"):
        familiar_ui.normalize_attachment_image(data)

    path = tmp_path / "bomb.png"
    path.write_bytes(data)
    with pytest.raises(ValueError, match="not a readable image"):
        familiar_ui.load_attachment(str(path))


def test_an_attached_file_over_the_byte_cap_is_refused_before_it_is_read(monkeypatch, tmp_path):
    """The unbounded ``read_bytes()`` loaded the whole file before any limit applied."""
    path = tmp_path / "huge.png"
    path.write_bytes(_real_png((8, 8)))
    monkeypatch.setattr(familiar_ui, "MAX_ATTACH_BYTES", 10)

    with pytest.raises(ValueError, match="too large"):
        familiar_ui.load_attachment(str(path))


def test_a_decode_refusal_in_the_task_lands_as_a_sentence_and_restores_the_path(
    monkeypatch, tmp_path
):
    """The decode runs in the task (host-06), so its refusal must come back as a user-facing
    ``Invalid`` and ``on_task_done`` must put it beside the attach control rather than lose it."""
    path = tmp_path / "junk.png"
    path.write_bytes(b"this is plainly not image data")
    monkeypatch.setattr(
        svc_familiar, "ask", lambda *a, **k: pytest.fail("a refused attach must not be asked")
    )
    ctx = _FakeCtx(mode="home")
    ui = familiar_ui.ensure(ctx)
    ui.attach_path = str(path)

    assert familiar_ui.submit_chat(ctx, "what is this?")
    fn, args, kwargs, tag = ctx._pending[familiar_ui.CHAT_KEY]
    with pytest.raises(Invalid) as caught:
        fn(*args, **kwargs)
    assert "not a readable image" in caught.value.message

    familiar_ui.on_task_done(
        ctx,
        Done(key=familiar_ui.CHAT_KEY, error=caught.value, message=caught.value.message, tag=tag),
    )
    assert ui.attach_error is not None and "not a readable image" in ui.attach_error
    assert ui.attach_path == str(path)


# --- familiar-host-06 -----------------------------------------------------------------------


@pytest.mark.parametrize("door", ["chat", "build"])
def test_submit_chat_does_not_decode_the_attached_image_on_the_frame_thread(
    monkeypatch, tmp_path, door
):
    """A 48 MP photo froze the frame loop for 1.2 s: the read, decode, RGB convert and LANCZOS
    resize ran inside ``submit_chat``/``submit_build``. They run in the submitted task now."""
    path = tmp_path / "ref.png"
    path.write_bytes(_real_png((64, 32)))
    frame_thread = threading.get_ident()
    decoded_on: list[int] = []

    real_normalize = familiar_ui.normalize_attachment_image

    def spy(data: bytes) -> bytes:
        decoded_on.append(threading.get_ident())
        return real_normalize(data)

    monkeypatch.setattr(familiar_ui, "normalize_attachment_image", spy)
    seen: dict = {}
    monkeypatch.setattr(
        svc_familiar,
        "ask",
        lambda svc, prompt, **kw: seen.setdefault("image", kw["image"])
        and svc_familiar.Answer(skill="other", text="ok"),
    )
    monkeypatch.setattr(
        svc_familiar,
        "clay_build",
        lambda svc, prompt, scene, image=None: seen.setdefault("image", image) or [],
    )
    ctx = _FakeCtx(mode="clay" if door == "build" else "home")
    familiar_ui.ensure(ctx).attach_path = str(path)

    submit = familiar_ui.submit_chat if door == "chat" else familiar_ui.submit_build
    key = familiar_ui.CHAT_KEY if door == "chat" else familiar_ui.BUILD_KEY
    assert submit(ctx, "what is this?")
    assert decoded_on == [], "the decode ran inside the submit, on the frame thread"

    fn, args, kwargs, _tag = ctx._pending[key]
    worker = threading.Thread(target=lambda: fn(*args, **kwargs))
    worker.start()
    worker.join()
    assert decoded_on and decoded_on[0] != frame_thread
    with BytesIO(seen["image"]) as buf:
        from PIL import Image

        assert Image.open(buf).size == (64, 32)
