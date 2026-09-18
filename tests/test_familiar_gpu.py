"""What only a real card, real weights and a real ``llama-server.exe`` can
answer about Familiar (P54/T9): that the base testing pin actually starts on
loopback and answers a chat turn; that llama.cpp's own constrained decoding
(``response_format``) really honours the router's, ``navigate``'s,
``create``'s and ``character``'s JSON schemas rather than merely being
accepted by the request; that the two ``PARALLEL_SLOTS`` really serve two
requests at once rather than serialising behind one; that
``llama_client.TEMPLATE_MARGIN_TOKENS`` really covers the gap between a raw
``/tokenize`` count and what the server's own chat template actually costs;
that the Clay card gate refuses before any network call when the running
pin's ``card_shas`` is empty; and what the base pin actually costs in VRAM,
which is where ``vram.FAMILIAR_GIB`` stops being a stated guess. None of that
is provable against a fake transport or a canned reply -- it is the one file
in the suite that spawns a real ``llama-server.exe`` child.

Run with: uv run pytest tests/test_familiar_gpu.py -m gpu -n 0

Serial, like every gpu-lane module: ``ensure_started`` refuses to share the
card with a queued GPU job, and this file wants sole occupancy for its own
VRAM measurement regardless. Every test constructs its own
``pipelines.llama.LlamaServer`` (or shares one module-scoped instance) rather
than going through ``queue.Worker`` -- ``key_dir``/``log_path`` point at a
throwaway directory this session owns, never at ``config.data_dir`` under the
real ``~/.warlock``, because this lane is the one exemption from conftest's
``WARLOCK_HOME`` pin and genuinely reads the real model library. The exe and
weights paths, by contrast, *do* come from the real ``get_config()`` -- there
is nothing to download here, and reading them is all this file ever does to
that directory.

Model-quality observations (which skill the router actually picked, whether
a plan parsed, the measured VRAM figures) are printed, each line prefixed
``FAMILIAR-GPU:``, never asserted -- the base testing pin has no eval corpus
of its own (``contract.SAMPLING``'s own docstrings say so for "chat",
"manual", "create" and "character"), so a specific reply shape here would be
a test of today's sampler seed, not of Familiar.
"""

from __future__ import annotations

import asyncio
import ctypes
import json
import socket
import threading
import time
from typing import Any

import httpx
import pytest
import pytest_asyncio

from warlock import fetch, models, vram
from warlock.config import get_config
from warlock.familiar import character_plan, contract, doors, llama_client, router
from warlock.pipelines.llama import LlamaServer
from warlock.service import familiar as familiar_service
from warlock.studio import modes
from warlock.studio.modes.create.engine import assets as create_assets
from warlock.studio.panes import app_settings

pytestmark = [pytest.mark.gpu, pytest.mark.timeout(1800)]


def _free_port() -> int:
    """A port nothing is listening on *right now*, on loopback only.

    Bind-then-close rather than a fixed constant: this file's own server and
    the vram-measurement test's second, independent instance must never
    collide with each other, with a real app's Familiar (``config.
    familiar_port``), or with a leftover orphan from a previous crashed run.
    The race between closing this socket and ``llama-server`` binding the
    same port is the same one ``pipelines/llama.py``'s own port-reclaim dance
    already has to tolerate; nothing here needs to be stronger than that.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _new_server(tmp_dir) -> LlamaServer:
    """One more ``LlamaServer`` against the real pinned exe/weights, mirroring
    ``queue.Worker.__init__``'s own construction except for the three fields
    a test must never point at the real ``~/.warlock``: port, key_dir and
    log_path."""
    config = get_config()
    return LlamaServer(
        lambda: config.familiar_runtime_dir / "llama-server.exe",
        lambda: config.familiar_models_dir / models.FAMILIAR_GGUF_FILE,
        _free_port(),
        key_dir=tmp_dir / "keys",
        log_path=tmp_dir / "familiar.log",
        idle_timeout=3600.0,
        expected_card_shas=lambda: models.FAMILIAR_MODELS["familiar_gguf"].card_shas,
        served_name=lambda: models.FAMILIAR_MODELS["familiar_gguf"].served_name,
    )


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def server(tmp_path_factory):
    """One resident ``llama-server.exe`` for the whole module, on its own
    event loop (``loop_scope="module"``): ``LlamaServer._lock`` is a plain
    ``asyncio.Lock`` that binds to whichever loop first ``acquire()``s it
    (``ensure_started``, called here), and every later test that reuses this
    same server object must run on that same loop or the second call raises
    "bound to a different event loop". Tests that build their own fresh
    ``LlamaServer`` (the vram measurement, the stop-frees-the-card check) are
    exempt from that constraint and run on an ordinary function-scoped loop.

    Skips, rather than fails, when a row is not downloaded -- the same
    ``fetch.present(config, "familiar", spec)`` idiom
    ``doctor._familiar_checks``/``bottom_pane.familiar_state`` already use,
    so this file behaves like every other gpu module when the machine simply
    has not fetched Familiar yet.
    """
    config = get_config()
    for spec in models.FAMILIAR_MODELS.values():
        if not fetch.present(config, "familiar", spec):
            pytest.skip(f"{spec.label} not downloaded")
    tmp_dir = tmp_path_factory.mktemp("familiar-gpu")
    srv = _new_server(tmp_dir)
    await srv.ensure_started()
    yield srv
    # Idempotent (LlamaServer.stop's own docstring) -- a no-op if
    # test_measure_resident_vram already stopped this same instance early to
    # get a clean baseline reading.
    srv.stop()


#: A representative compacted scene, in ``contract.compact_scene``'s own
#: shape (``contract._SCENE_ROW_KEYS`` plus ``materials``/``bounds``) --
#: hand-built rather than pulled from ``dev/training/clay-assistant``'s dataset
#: (as ``tests/familiar/test_contract.py`` does) because this file needs one
#: representative scene, not the corpus's own worst case.
SAMPLE_SCENE: dict[str, Any] = {
    "objects": [
        {
            "uid": "obj-1",
            "name": "Table",
            "generator": "cylinder",
            "params": {"radius": 0.4, "height": 0.75, "segments": 32},
            "translation": [0.0, 0.375, 0.0],
            "rotation": [0.0, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0],
            "size": [0.8, 0.75, 0.8],
            "material": 0,
        },
        {
            "uid": "obj-2",
            "name": "Chair",
            "generator": "box",
            "params": {"width": 0.4, "height": 0.9, "depth": 0.4},
            "translation": [0.8, 0.45, 0.0],
            "rotation": [0.0, 30.0, 0.0],
            "scale": [1.0, 1.0, 1.0],
            "size": [0.4, 0.9, 0.4],
            "material": 1,
        },
        {
            "uid": "obj-3",
            "name": "Lamp",
            "generator": "cylinder",
            "params": {"radius": 0.05, "height": 1.2, "segments": 16},
            "translation": [-0.8, 0.6, 0.5],
            "rotation": [0.0, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0],
            "size": [0.1, 1.2, 0.1],
            "material": 2,
        },
    ],
    "materials": [
        {"index": 0, "base_color": [0.55, 0.35, 0.2, 1.0], "roughness": 0.8},
        {"index": 1, "base_color": [0.3, 0.3, 0.3, 1.0], "roughness": 0.6},
        {"index": 2, "base_color": [0.9, 0.9, 0.8, 1.0], "roughness": 0.4},
    ],
    "bounds": {"min": [-1.0, 0.0, -1.0], "max": [1.2, 1.2, 1.0]},
}


@pytest.mark.asyncio(loop_scope="module")
async def test_the_base_pin_starts_on_loopback_and_answers_a_chat_turn(server):
    """The base testing pin is not just present on disk -- it starts,
    listens on loopback and answers a real chat-completions turn."""
    assert server.base_url.startswith("http://127.0.0.1:")
    before = server.last_used
    messages = contract.build_chat_messages("Say hello in five words.")
    reply = await llama_client.chat(
        server, messages, slot=router.SKILL_SLOT, sampling=contract.SAMPLING["chat"]
    )
    assert reply.strip()
    assert server.last_used >= before
    print(f"FAMILIAR-GPU: chat reply={reply!r}")


#: Six prompts spanning the modes a router call actually has to disambiguate
#: (``build_router_messages``'s own "make a chair means something different
#: sent from Clay than from Home" reasoning) -- the mode label here is
#: provenance for the printed line, not an assertion: what is asserted is
#: only that constrained decoding produced one of :data:`router.SKILLS`.
ROUTER_PROMPTS: tuple[tuple[str, str], ...] = (
    ("how do I export a GLB", "home"),
    ("make a wooden barrel", "clay"),
    ("open Mason", "home"),
    ("make me a goblin", "home"),
    ("thanks", "home"),
    ("generate a sprite of a fox", "create"),
)


@pytest.mark.asyncio(loop_scope="module")
async def test_the_real_server_honours_the_router_schema(server):
    """``response_format``'s constrained decoding, on the installed
    ``b10948`` build, really constrains the router's reply to
    :data:`router.SKILLS` -- proved by decoding the JSON body, not by
    ``router.parse_route``'s own "other" fallback swallowing a malformed one."""
    for prompt, mode in ROUTER_PROMPTS:
        reply = await llama_client.chat(
            server,
            contract.build_router_messages(prompt, mode),
            slot=router.ROUTER_SLOT,
            sampling=contract.SAMPLING["router"],
            response_format={
                "type": "json_schema",
                "json_schema": {"schema": router.ROUTE_SCHEMA},
            },
        )
        decoded = json.loads(reply.strip())
        assert isinstance(decoded, dict)
        assert decoded.get("skill") in router.SKILLS
        print(f"FAMILIAR-GPU: router {prompt!r} (mode={mode}) -> {decoded.get('skill')!r}")


def _character_options_from_service(svc: Any) -> dict[str, Any]:
    """The plan-shaped slice ``familiar_ui._character_options`` builds, off
    the same ``service.characters.character_options(svc)`` read, duplicated
    here rather than called through ``familiar_ui`` -- that function reads
    ``character_engine.options(ctx)``, a frame-thread cache keyed on a
    palette-directory stamp, and there is no ``ctx`` (no window, no App) in a
    headless gpu test to hand it. ``svc`` is a throwaway
    ``WarlockService`` (the ``svc`` fixture from ``tests/conftest.py``, tmp
    ``WARLOCK_HOME``), so this reads only the registries, never the real
    library.
    """
    from warlock.kernels.rig import cliplib
    from warlock.service import characters as svc_characters

    raw = svc_characters.character_options(svc)
    families = [
        {"key": f["key"], "label": f["label"], "themes": [t["key"] for t in f["themes"]]}
        for f in raw["families"]
    ]
    templates = {a["template"] for a in raw["archetypes"]}
    movements = sorted({name for t in templates for name in cliplib.shipped_clip_names(t)})
    return {
        "families": families,
        "movements": movements,
        "directions": list(raw["directions"]),
        "size_range": tuple(raw["troupe"]["logical_size_range"]),
    }


@pytest.mark.asyncio(loop_scope="module")
async def test_constrained_navigate_create_and_character_replies_parse(server, svc):
    """The other three constrained-decoding schemas (T8's ``navigate``/
    ``create``, T7's ``character``), each built from the real runtime lists
    a live app would offer -- every mode plus every Settings category as a
    destination, every ``create_assets`` asset type, and the real character
    registry read off a throwaway service. Only the JSON shape is asserted;
    which destination/asset type/species the model actually named is model
    quality, printed rather than checked (a parser returning ``None`` is
    printed too, per the brief -- it means the model named nothing usable,
    not that this test failed)."""
    destinations = tuple(
        doors.Destination(f"go:{key}", label) for key, label, _icon, _purpose in modes.MODES
    ) + tuple(
        doors.Destination(f"settings:{key}", label) for key, label in app_settings.CATEGORIES
    )
    dest_keys = [d.key for d in destinations]
    asset_types = create_assets.ASSET_TYPE_OPTIONS
    character_options = _character_options_from_service(svc)

    nav_reply = await llama_client.chat(
        server,
        doors.build_navigate_messages("open Mason", destinations),
        slot=router.SKILL_SLOT,
        sampling=contract.SAMPLING["navigate"],
        response_format={
            "type": "json_schema",
            "json_schema": {"schema": doors.navigate_schema(dest_keys)},
        },
    )
    nav_decoded = json.loads(nav_reply.strip())
    assert isinstance(nav_decoded, dict) and "target" in nav_decoded
    print(
        "FAMILIAR-GPU: navigate 'open Mason' -> "
        f"{doors.parse_target(nav_reply, dest_keys)!r}"
    )

    create_reply = await llama_client.chat(
        server,
        doors.build_create_messages("make a reference image of a lantern", asset_types),
        slot=router.SKILL_SLOT,
        sampling=contract.SAMPLING["create"],
        response_format={
            "type": "json_schema",
            "json_schema": {"schema": doors.create_schema(asset_types)},
        },
    )
    create_decoded = json.loads(create_reply.strip())
    assert isinstance(create_decoded, dict)
    assert {"asset_type", "prompt"} <= create_decoded.keys()
    print(
        "FAMILIAR-GPU: create 'make a reference image of a lantern' -> "
        f"{doors.parse_draft(create_reply, asset_types)!r}"
    )

    character_reply = await llama_client.chat(
        server,
        character_plan.build_character_messages("make me a goblin", character_options),
        slot=router.SKILL_SLOT,
        sampling=contract.SAMPLING["character"],
        response_format={
            "type": "json_schema",
            "json_schema": {"schema": character_plan.character_schema(character_options)},
        },
    )
    character_decoded = json.loads(character_reply.strip())
    assert isinstance(character_decoded, dict) and "family" in character_decoded
    print(
        "FAMILIAR-GPU: character 'make me a goblin' -> "
        f"{character_plan.parse_plan(character_reply, character_options)!r}"
    )


@pytest.mark.asyncio(loop_scope="module")
async def test_requests_on_both_slots_are_accepted(server):
    """``PARALLEL_SLOTS = 2`` really serves two ``id_slot``s concurrently --
    both requests are sent together (``asyncio.gather``) rather than one
    awaited after the other, which would pass even if the server secretly
    serialised them."""
    reply0, reply1 = await asyncio.gather(
        llama_client.chat(
            server,
            contract.build_chat_messages("Name a color in one word."),
            slot=0,
            sampling=contract.SAMPLING["chat"],
        ),
        llama_client.chat(
            server,
            contract.build_chat_messages("Name an animal in one word."),
            slot=1,
            sampling=contract.SAMPLING["chat"],
        ),
    )
    assert reply0.strip() and reply1.strip()
    print(f"FAMILIAR-GPU: slot0={reply0!r} slot1={reply1!r}")


@pytest.mark.asyncio(loop_scope="module")
async def test_tokenize_plus_the_template_margin_covers_the_real_prompt_count(server):
    """``llama_client.TEMPLATE_MARGIN_TOKENS`` is a stated ceiling on the
    chat template's own overhead, never measured against a live server
    (its own docstring: "not a measurement"). This is that measurement:
    a raw ``/tokenize`` count of the concatenated message text, exactly the
    way ``llama_client.chat``'s own ``SIZED_SKILLS`` branch counts it, against
    ``usage.prompt_tokens`` from a real ``max_tokens=1`` completion -- the
    server's own count of what the ``--jinja`` template actually built.
    """
    messages = contract.build_messages("clay", "build a small wooden table", SAMPLE_SCENE)
    prompt_text = "\n\n".join(m["content"] for m in messages)
    headers = llama_client._headers(server)

    await server.ensure_started()
    server.touch()
    async with httpx.AsyncClient(
        base_url=server.base_url, timeout=llama_client.CHAT_TIMEOUT
    ) as client:
        tok_resp = await client.post("/tokenize", json={"content": prompt_text}, headers=headers)
        assert tok_resp.status_code == 200
        raw = len(tok_resp.json()["tokens"])

        completion_resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "familiar",
                "messages": messages,
                "id_slot": router.SKILL_SLOT,
                "temperature": 0.0,
                "top_k": 1,
                "top_p": 1.0,
                "max_tokens": 1,
                "stream": False,
            },
            headers=headers,
        )
        assert completion_resp.status_code == 200
        real = completion_resp.json()["usage"]["prompt_tokens"]
    server.touch()

    print(
        f"FAMILIAR-GPU: template overhead real-raw={real - raw} "
        f"margin={llama_client.TEMPLATE_MARGIN_TOKENS}"
    )
    assert raw + llama_client.TEMPLATE_MARGIN_TOKENS >= real


class _RefusesBeforeAnyRequest:
    """A ``svc`` stand-in whose ``call_on_loop`` fails the test outright if
    ever invoked -- ``clay_build`` must refuse on the empty ``card_shas``
    gate *before* ``_call``/``call_on_loop`` is ever reached, so nothing here
    should touch the network, or even the event loop, at all. ``worker`` is
    a plain truthy sentinel so ``_call``'s own "no worker" branch (a
    different refusal, ``reason="missing"``) is not what would make this
    test pass by accident.
    """

    worker = object()

    def call_on_loop(self, fn, timeout=None):
        pytest.fail("clay_build must refuse on the card gate before any request is made")


def test_a_clay_build_on_the_base_pin_refuses_before_any_request():
    """The base pin ships no card_shas (T3/T10's own gate) -- Clay's Build
    must refuse with reason "card" against it, and must do so without ever
    dialling out, so this needs no server at all."""
    assert models.FAMILIAR_MODELS["familiar_gguf"].card_shas == ()
    with pytest.raises(familiar_service.FamiliarRefusal) as exc_info:
        familiar_service.clay_build(_RefusesBeforeAnyRequest(), "build a barrel", SAMPLE_SCENE)
    assert exc_info.value.reason == "card"


# ---------------------------------------------------------------------------
# NVML helpers for the two tests below. Per-process ``usedGpuMemory`` reads
# [N/A] on this machine's driver under WDDM (checked with ``nvidia-smi``:
# RTX 5090, 32607 MiB, driver 610.62 -- every compute process shows
# ``used_gpu_memory [N/A]``), so every VRAM figure below is a device-level
# delta (``vram.live_memory()``, before/after/peak), and pid *presence* in
# NVML's own compute-process list -- which nvidia-smi does still report,
# memory column blank or not -- stands in for "the child is really on the
# card" wherever the brief originally asked for a per-process memory read.
# ---------------------------------------------------------------------------


class _NvmlProcessV2(ctypes.Structure):
    """Mirrors NVML's own ``nvmlProcessInfo_v2_t`` (pid, usedGpuMemory,
    gpuInstanceId, computeInstanceId) -- the shape
    ``nvmlDeviceGetComputeRunningProcesses_v3`` (and its v2 predecessor) fill
    an array of, ctypes-aligned the same way the DLL itself is compiled."""

    _fields_ = [
        ("pid", ctypes.c_uint32),
        ("usedGpuMemory", ctypes.c_uint64),
        ("gpuInstanceId", ctypes.c_uint32),
        ("computeInstanceId", ctypes.c_uint32),
    ]


_NVML_ERROR_INSUFFICIENT_SIZE = 7
#: NVML's own "not available" sentinel for a per-process field the driver
#: cannot report (exactly the WDDM case this module's comment above names).
_NVML_VALUE_NOT_AVAILABLE = 2**64 - 1


def _process_listed_and_memory(pid: int) -> tuple[bool, str]:
    """Whether *pid* is one of NVML's own compute-running-processes right
    now, and its ``usedGpuMemory`` as a printable string ("n/a (WDDM)" when
    the driver reports the sentinel, which is the expected case here).

    Reaches into ``vram._nvml()`` -- private, but the one open NVML session
    this process holds (see that function's own docstring on why it is kept
    open rather than re-initialised per reading), and there is no public
    "list processes" accessor on top of it. Tries the versioned symbol names
    newest-first and falls back silently: a driver too old for ``_v3`` is not
    this test's subject, it is a reason to fall back to ``_v2``/unversioned
    exactly the way ``pipelines/trellis.py``'s own probes degrade.
    """
    session = vram._nvml()
    if session is None:
        return False, "n/a (no NVML)"
    nvml, handle, _name = session
    for fname in (
        "nvmlDeviceGetComputeRunningProcesses_v3",
        "nvmlDeviceGetComputeRunningProcesses_v2",
        "nvmlDeviceGetComputeRunningProcesses",
    ):
        func = getattr(nvml, fname, None)
        if func is None:
            continue
        try:
            count = ctypes.c_uint32(0)
            rc = func(handle, ctypes.byref(count), None)
            if rc not in (0, _NVML_ERROR_INSUFFICIENT_SIZE):
                continue
            n = max(count.value, 1)
            arr = (_NvmlProcessV2 * n)()
            count2 = ctypes.c_uint32(n)
            rc = func(handle, ctypes.byref(count2), arr)
            if rc != 0:
                continue
            for i in range(count2.value):
                if arr[i].pid != pid:
                    continue
                used = arr[i].usedGpuMemory
                if used == _NVML_VALUE_NOT_AVAILABLE:
                    return True, "n/a (WDDM)"
                return True, f"{used / 1024**3:.2f}"
            return False, "n/a (pid not in list)"
        except (OSError, AttributeError, ValueError):
            continue
    return False, "n/a (enumeration unavailable)"


def _driver_version() -> str:
    session = vram._nvml()
    if session is None:
        return "unknown"
    nvml, _handle, _name = session
    func = getattr(nvml, "nvmlSystemGetDriverVersion", None)
    if func is None:
        return "unknown"
    buf = ctypes.create_string_buffer(80)
    try:
        if func(buf, len(buf)) != 0:
            return "unknown"
    except (OSError, AttributeError, ValueError):
        return "unknown"
    return buf.value.decode("utf-8", "replace")


def _sample_used_gib(stop_event: threading.Event, samples: list[float]) -> None:
    """Poll device-used GiB every 0.25s until *stop_event* -- run on its own
    thread so it keeps sampling while the event loop awaits the two
    concurrent generations below."""
    while not stop_event.is_set():
        mem = vram.live_memory()
        if mem is not None:
            samples.append(mem.total_gib - mem.free_gib)
        time.sleep(0.25)


#: Repeated to build a long prompt for the vram measurement -- not a
#: measured token count (the test never asserts on prompt size, only on
#: peak-vs-before), just enough real English text that the tokenizer and the
#: sampler both do genuine work across a large context.
_LONG_FILLER = (
    "The quick brown fox jumps over the lazy dog near the old stone bridge at dusk. "
)


def test_measure_resident_vram(tmp_path_factory, server):
    """What the base pin actually costs on this card, device-level: before
    spawn, after a healthy start, and peak under two concurrent long
    generations -- the measurement ``vram.FAMILIAR_GIB``'s own docstring
    says is owed ("a guess, stated as one, until measured on real
    hardware").

    Stops the shared module ``server`` first so its own ~6.5 GiB is not
    already resident in the "before" reading -- this measurement needs sole
    occupancy of the card for its own spawn, and no other test in this file
    still needs that shared instance running afterward (this is deliberately
    the last test that touches it; ``server``'s own teardown re-stopping an
    already-stopped process is a documented no-op).
    """
    server.stop()

    before = vram.live_memory()
    assert before is not None, "NVML unavailable on this host -- nothing to measure against"
    before_used = before.total_gib - before.free_gib

    tmp_dir = tmp_path_factory.mktemp("familiar-gpu-vram")
    own = _new_server(tmp_dir)

    async def _run() -> tuple[float, float, float, bool, str]:
        await own.ensure_started()

        after = vram.live_memory()
        after_used = (after.total_gib - after.free_gib) if after is not None else before_used
        resident_gib = after_used - before_used

        samples: list[float] = []
        stop_event = threading.Event()
        sampler = threading.Thread(target=_sample_used_gib, args=(stop_event, samples), daemon=True)
        sampler.start()

        long_prompt = _LONG_FILLER * 275  # roughly 5,500 tokens' worth of real prose
        messages = [{"role": "user", "content": long_prompt}]
        sampling = {"temperature": 0.7, "top_k": 64, "top_p": 0.95, "max_tokens": 1500}
        try:
            await asyncio.gather(
                llama_client.chat(own, messages, slot=0, sampling=sampling),
                llama_client.chat(own, messages, slot=1, sampling=sampling),
            )
        finally:
            stop_event.set()
            sampler.join(timeout=5)

        peak_used = max(samples) if samples else after_used
        peak_gib = peak_used - before_used

        listed, process_gib = _process_listed_and_memory(own._proc.pid)  # private: see module note
        own.stop()
        return after_used, resident_gib, peak_gib, listed, process_gib

    after_used, resident_gib, peak_gib, listed, process_gib = asyncio.run(_run())

    card_name = (vram._nvml() or (None, None, "unknown"))[2]
    print(f"FAMILIAR-GPU: card={card_name} driver={_driver_version()}")
    print(f"FAMILIAR-GPU: vram_before_gib={before_used:.2f}")
    print(f"FAMILIAR-GPU: resident_gib={resident_gib:.2f}")
    print(f"FAMILIAR-GPU: peak_gib={peak_gib:.2f}")
    print(f"FAMILIAR-GPU: process_gib={process_gib}")
    print("FAMILIAR-GPU: host_rss_gib=n/a (psutil is not a declared dependency -- memlog.py)")
    print(f"FAMILIAR-GPU: FAMILIAR_GIB={vram.FAMILIAR_GIB}")

    assert listed, "the spawned child never appeared in NVML's own process list"
    assert peak_gib > 0
    # Measured three times on 2026-09-14 (3.20 GiB peak each run,
    # dev/measurements/2026-09-14-familiar-base-vram.md) and FAMILIAR_GIB set
    # from it, so the admission figure is now a claim this lane checks.
    assert peak_gib <= vram.FAMILIAR_GIB, (
        f"Familiar peaked at {peak_gib:.2f} GiB, over vram.FAMILIAR_GIB={vram.FAMILIAR_GIB}"
    )


def test_stop_frees_the_card(tmp_path_factory):
    """``stop()``'s own contract: the child leaves NVML's process list and
    its key file is gone -- checked against a fresh, dedicated instance so
    this in no way depends on the shared module ``server``'s own state."""
    tmp_dir = tmp_path_factory.mktemp("familiar-gpu-stop")
    srv = _new_server(tmp_dir)
    asyncio.run(srv.ensure_started())

    pid = srv._proc.pid  # private: see the NVML helpers' module note above
    key_path = srv.key_path
    assert key_path is not None and key_path.exists()

    listed_before, _mem = _process_listed_and_memory(pid)
    assert listed_before, "the freshly spawned child never appeared in NVML's process list"

    srv.stop()

    listed_after, _mem = _process_listed_and_memory(pid)
    assert not listed_after
    assert srv.key_path is None
    assert not key_path.exists()
