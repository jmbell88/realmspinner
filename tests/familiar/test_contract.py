"""T3: the frozen Clay prompt card, pinned against everything it has to agree
with -- run A's own recorded eval hashes, the live ``agent_clay`` door, the
training dataset's own recorded replies, and the trained model's own context
window.

``warlock.studio.familiar.contract`` is imported directly (no imgui/moderngl/
pygame/httpx/service/queue needed for any of this); a couple of tests reach
into ``training/clay-assistant`` for its own dataset and eval fixtures, added
to ``sys.path`` the same way ``training/clay-assistant/eval/run_val.py``
itself does.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import pytest

from warlock.studio.familiar import contract

ROOT = Path(__file__).resolve().parents[2]
RUN_A_DIR = ROOT / "docs" / "measurements" / "data" / "clay-assistant" / "run-A"
TRAINING_PKG = ROOT / "training" / "clay-assistant"

for p in (TRAINING_PKG, TRAINING_PKG / "gen", TRAINING_PKG / "eval"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def test_the_frozen_clay_card_hashes_to_run_a_s_recorded_card_sha():
    """The card this module ships (``cards/clay-1.txt``) must be the exact
    bytes the user chose to freeze: run A's own training-time card, whose
    hash is independently recorded in run A's own eval JSONs
    (``settings.card_sha256``) -- not just copied once and trusted."""
    expected = "70697ece5bec5781b1f04703745bed2b2f2c56a9a664a04eef831c7aa9838aab"
    assert contract.card_sha("clay") == expected

    card_path = RUN_A_DIR / "card.txt"
    assert hashlib.sha256(card_path.read_bytes()).hexdigest() == expected

    found_any = False
    for eval_path in sorted(RUN_A_DIR.glob("eval-*.json")):
        data = json.loads(eval_path.read_text(encoding="utf-8"))
        recorded = data.get("settings", {}).get("card_sha256")
        if recorded is None:
            # Not every eval-*.json carries `settings` (a couple of run A's
            # earliest files predate that field) -- skipped, not failed.
            continue
        found_any = True
        assert recorded == expected, f"{eval_path.name} recorded a different card sha"
    assert found_any, "no run-A eval-*.json carried settings.card_sha256 to check against"


def test_the_frozen_clay_card_names_only_tools_the_live_door_accepts():
    """``allowed_calls('clay')`` (parsed from the frozen card's own
    ``clay_batch`` schema line) must be a subset of what the live
    ``agent_clay`` door actually names today, and every property the frozen
    card's own tool schemas mention must still exist on the live tool's
    schema -- a live registry is allowed to grow past what an old, frozen
    card describes, never to drop something that card still promises."""
    from warlock.studio import agent_clay

    live_tools = {t.name: t for t in agent_clay.tools()}
    allowed = contract.allowed_calls("clay")
    unknown = allowed - set(live_tools)
    assert not unknown, f"frozen card names tools the live door no longer has: {unknown}"

    card = contract.load_card("clay")
    for line, schema_line in zip(
        (ln for ln in card.splitlines() if ln.strip().startswith("- clay_")),
        (ln for ln in card.splitlines() if ln.strip().startswith("schema:")),
        strict=True,
    ):
        name = line.strip()[2:].split(":", 1)[0]
        assert name in live_tools, f"frozen card describes {name!r}, which the live door dropped"
        frozen_schema = json.loads(schema_line.strip()[len("schema:") :].strip())
        frozen_props = set(frozen_schema.get("properties", {}))
        live_props = set(live_tools[name].schema.get("properties", {}))
        missing = frozen_props - live_props
        assert not missing, (
            f"{name}: frozen card's schema names propert{'y' if len(missing) == 1 else 'ies'} "
            f"{missing} that the live tool's schema no longer has"
        )


def test_derive_clay_card_reproduces_the_dataset_manifest_tools_sha():
    """``derived_card_sha()`` (the live, un-frozen card -- see
    ``derive_clay_card``'s own docstring) must still equal what
    ``training/clay-assistant/dataset/manifest.json`` pinned as ``tools_sha``
    when the dataset now on disk was built -- proving the extraction out of
    ``gen/convert.py`` reproduced ``compact_tools()``'s output byte for
    byte, not just approximately."""
    manifest = json.loads((TRAINING_PKG / "dataset" / "manifest.json").read_text(encoding="utf-8"))
    assert contract.derived_card_sha() == manifest["tools_sha"]


def _run_a_eval_files() -> list[Path]:
    return sorted(RUN_A_DIR.glob("eval-*.json"))


def test_parse_calls_agrees_with_the_eval_scorer_on_every_recorded_reply():
    """Every ``eval-*.json`` under run A's own measurement directory records
    each row's raw ``reply`` text and the outcome ``run_val.py`` scored it
    with. A ``query``-kind row is scored by a different rule (fenced-or-empty
    prose, not ``parse_calls`` at all) and is skipped; every other row's
    outcome tells us unambiguously whether the *original* ``parse_calls``
    accepted or refused the fence -- ``contract.parse_calls`` (moved
    verbatim) must agree on every one."""
    files = _run_a_eval_files()
    assert files, "no run-A eval-*.json files found to check replies against"

    checked = 0
    for eval_path in files:
        data = json.loads(eval_path.read_text(encoding="utf-8"))
        for row in data.get("rows", []):
            if row.get("kind") == "query":
                continue
            for sample in row.get("samples", [row]):
                reply = sample.get("reply")
                outcome = sample.get("outcome")
                if reply is None or outcome is None:
                    continue
                calls, detail = contract.parse_calls(reply)
                checked += 1
                if outcome == "no parse":
                    assert calls is None, (
                        f"{eval_path.name}/{row['id']}: recorded 'no parse' but "
                        f"contract.parse_calls accepted it"
                    )
                    recorded_detail = sample.get("detail")
                    if recorded_detail is not None:
                        assert detail == recorded_detail, (
                            f"{eval_path.name}/{row['id']}: detail {detail!r} != "
                            f"recorded {recorded_detail!r}"
                        )
                else:
                    # accepted / refused / built, failed check -- all three only
                    # happen once parse_calls has already produced a calls list.
                    assert calls is not None, (
                        f"{eval_path.name}/{row['id']}: recorded {outcome!r} but "
                        f"contract.parse_calls found no calls ({detail!r})"
                    )
    assert checked > 0, "no row carried both a reply and an outcome to check"


# 2.73 chars/token: measured directly with run A's own tokenizer against this
# same dataset material (the worst row) during the 2026-09-14 investigation
# that replaced the old, wrong 3.0-chars/token/4,096-reserve version of these
# tests -- that estimate said the largest val prompt fit; the real tokenizer
# put it at ~4,931 tokens, over the 8,192-token window once the full 4,096
# output reserve was added. See ``contract.TRAINED_WINDOW``'s own docstring
# for why the window is a llama-server *slot*, not the server's total context.
CHARS_PER_TOKEN = 2.73


def test_the_server_slot_is_at_least_the_trained_window():
    """``pipelines/llama.py`` runs one llama-server with ``CTX_SIZE`` split
    ``PARALLEL_SLOTS`` ways -- each concurrent chat gets one slot, and that
    slot, not the server's total context, is what a single Clay turn actually
    has to fit inside. If a slot ever shrank below what run A trained at, the
    model would see truncated prompts it was never trained to handle."""
    from warlock.pipelines import llama

    assert llama.CTX_SIZE // llama.PARALLEL_SLOTS >= contract.TRAINED_WINDOW


def test_the_largest_val_prompt_leaves_room_for_the_longest_trained_reply():
    """The largest scene user turn in the dataset's own validation split,
    plus the frozen card as the system turn, estimated at
    :data:`CHARS_PER_TOKEN`, plus :data:`contract.MIN_REPLY_TOKENS` (the
    longest reply actually seen in training), must fit inside
    :data:`contract.TRAINED_WINDOW` -- the real constraint a chat turn runs
    under, not the sampler's own ``max_tokens`` ceiling."""
    from gen import convert, verify

    val_text = (TRAINING_PKG / "dataset" / "val.jsonl").read_text(encoding="utf-8")
    records = [json.loads(line) for line in val_text.splitlines() if line.strip()]
    scene_records = [r for r in records if r["kind"] in ("edit", "query")]
    assert scene_records, "no edit/query rows in val.jsonl to measure a scene turn from"

    card = contract.load_card("clay")
    worst_chars = 0
    worst_id = None
    for record in scene_records:
        replay = verify.replay(record["prior"])
        scene = convert.compact_scene(verify.scene(replay))
        turn = contract.user_turn(record["prompt"], scene)
        total = len(card) + len(turn)
        if total > worst_chars:
            worst_chars, worst_id = total, record["id"]

    est_tokens = math.ceil(worst_chars / CHARS_PER_TOKEN)
    total_needed = est_tokens + contract.MIN_REPLY_TOKENS
    assert total_needed <= contract.TRAINED_WINDOW, (
        f"{worst_id}: card+scene+prompt estimated at {est_tokens} tokens "
        f"({worst_chars} chars / {CHARS_PER_TOKEN}), plus the "
        f"{contract.MIN_REPLY_TOKENS}-token reply floor, is {total_needed} -- over "
        f"the {contract.TRAINED_WINDOW}-token trained window"
    )


def test_output_budget_shrinks_to_fit_the_window_and_refuses_below_the_floor():
    """``output_budget`` hands back the smaller of the skill's own
    ``max_tokens`` and what is left of the window after the prompt, and
    refuses outright once that would go under the reply floor."""
    max_tokens = contract.SAMPLING["clay"]["max_tokens"]

    # Plenty of room left: the configured max_tokens wins outright.
    assert contract.output_budget("clay", prompt_tokens=100) == max_tokens

    # Room enough for a reply, but less than the configured max: shrinks to fit.
    tight_prompt = contract.TRAINED_WINDOW - contract.MIN_REPLY_TOKENS
    budget = contract.output_budget("clay", prompt_tokens=tight_prompt)
    assert budget == contract.MIN_REPLY_TOKENS

    # Not even room for the floor: refuse rather than hand back a reserve no
    # real reply fits in.
    with pytest.raises(ValueError, match="floor"):
        contract.output_budget("clay", prompt_tokens=tight_prompt + 1)


@pytest.mark.asyncio
async def test_the_clay_card_refuses_on_the_testing_pin(tmp_path, monkeypatch):
    """``models.FAMILIAR_MODELS["familiar_gguf"]`` (the base, untuned
    testing pin) carries an empty ``card_shas`` -- it was never validated
    against any prompt card, Clay's included. Wiring ``ensure_started`` to
    require the Clay card's sha against that empty tuple must refuse before
    any subprocess spawn, the same way ``tests/test_familiar.py::
    test_a_card_sha_mismatch_refuses_to_start`` proves for an arbitrary
    fake sha -- this pins the same refusal for the *real* Clay card, so the
    base pin can never be mistaken for one that speaks Clay."""
    from warlock import fetch, models
    from warlock.pipelines import llama as llama_mod
    from warlock.pipelines.llama import LlamaServer

    exe = tmp_path / "llama-server.exe"
    weights = tmp_path / "models" / "familiar" / models.FAMILIAR_GGUF_FILE
    srv = LlamaServer(
        exe,
        weights,
        17972,
        key_dir=tmp_path / "keys",
        log_path=tmp_path / "familiar.log",
        expected_card_shas=lambda: models.FAMILIAR_MODELS["familiar_gguf"].card_shas,
    )
    assert models.FAMILIAR_MODELS["familiar_gguf"].card_shas == ()

    srv._resolve_exe().parent.mkdir(parents=True, exist_ok=True)
    srv._resolve_exe().write_bytes(b"")
    srv._resolve_weights().parent.mkdir(parents=True, exist_ok=True)
    srv._resolve_weights().write_bytes(b"")

    monkeypatch.setattr(
        llama_mod.fetch,
        "verify_manifest",
        lambda dest: fetch.Verification(dest=dest, status=fetch.VERIFY_UNKNOWN),
    )
    spawned = []
    monkeypatch.setattr(llama_mod.subprocess, "Popen", lambda *a, **k: spawned.append(a))

    with pytest.raises(RuntimeError, match="prompt card"):
        await srv.ensure_started(expected_card_sha=contract.card_sha("clay"))
    assert spawned == []


def _unset_text_patterns(gitattributes: str) -> list[str]:
    patterns = []
    for line in gitattributes.splitlines():
        parts = line.split()
        if parts and not parts[0].startswith("#") and "-text" in parts[1:]:
            patterns.append(parts[0])
    return patterns


def test_the_frozen_cards_are_exempt_from_line_ending_conversion():
    """``* text=auto`` plus Git for Windows' default ``core.autocrlf=true``
    writes a checked-out text file with CRLF, which changes a card's sha256
    and the prompt the model is sent. So a fresh clone (CI included) would
    fail the card-hash pin, or ship a card run A never trained on, unless the
    cards are ``-text``."""
    import fnmatch

    patterns = _unset_text_patterns((ROOT / ".gitattributes").read_text(encoding="utf-8"))
    cards = ["src/warlock/studio/familiar/cards/" + name for name in contract.CARDS.values()] + [
        "docs/measurements/data/clay-assistant/run-A/card.txt"
    ]
    for card in cards:
        assert any(fnmatch.fnmatch(card, p) for p in patterns), card
