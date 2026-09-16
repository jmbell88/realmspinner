"""T3: the frozen Clay prompt card, pinned against everything it has to agree
with -- the live ``agent_clay`` door, the trained model's own context window,
and (in ``dev/tests/familiar/test_contract.py``) run A's own recorded eval
hashes and the training dataset's own recorded replies, which moved to
``dev/`` on 2026-09-16.

``warlock.studio.familiar.contract`` is imported directly (no imgui/moderngl/
pygame/httpx/service/queue needed for any of this).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from warlock.studio.familiar import contract

ROOT = Path(__file__).resolve().parents[2]


def test_the_frozen_clay_card_hashes_to_its_recorded_sha():
    """The card this module ships (``cards/clay-1.txt``) must hash to the sha
    run A froze it at. The cross-check against run A's own on-disk card and
    its eval JSONs' recorded ``settings.card_sha256`` -- proof this is not
    just copied once and trusted -- lives in
    ``dev/tests/familiar/test_contract.py``, since it reaches into
    ``dev/measurements/data/clay-assistant/run-A``.
    """
    expected = "70697ece5bec5781b1f04703745bed2b2f2c56a9a664a04eef831c7aa9838aab"
    assert contract.card_sha("clay") == expected


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


# ``test_derive_clay_card_reproduces_the_dataset_manifest_tools_sha`` and
# ``test_parse_calls_agrees_with_the_eval_scorer_on_every_recorded_reply``
# moved to ``dev/tests/familiar/test_contract.py`` on 2026-09-16: both read
# ``dev/training/clay-assistant`` and ``dev/measurements/data/clay-assistant/
# run-A``, neither of which exists on a clean checkout.


def test_the_server_slot_is_at_least_the_trained_window():
    """``pipelines/llama.py`` runs one llama-server with ``CTX_SIZE`` split
    ``PARALLEL_SLOTS`` ways -- each concurrent chat gets one slot, and that
    slot, not the server's total context, is what a single Clay turn actually
    has to fit inside. If a slot ever shrank below what run A trained at, the
    model would see truncated prompts it was never trained to handle."""
    from warlock.pipelines import llama

    assert llama.CTX_SIZE // llama.PARALLEL_SLOTS >= contract.TRAINED_WINDOW


# ``test_the_largest_val_prompt_leaves_room_for_the_longest_trained_reply``
# moved to ``dev/tests/familiar/test_contract.py`` on 2026-09-16: it reads
# ``dev/training/clay-assistant/dataset/val.jsonl`` and imports ``gen``/
# ``verify`` from that package, neither of which exists on a clean checkout.


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
    cards are ``-text``.

    Run A's own card (``dev/measurements/data/clay-assistant/run-A/card.txt``)
    moved to ``dev/`` on 2026-09-16 along with the ``.gitattributes`` pattern
    that exempts it; that half of this check moved to
    ``dev/tests/familiar/test_contract.py`` since the file itself is dev-only,
    even though ``.gitattributes`` stays public.
    """
    import fnmatch

    patterns = _unset_text_patterns((ROOT / ".gitattributes").read_text(encoding="utf-8"))
    cards = ["src/warlock/studio/familiar/cards/" + name for name in contract.CARDS.values()]
    for card in cards:
        assert any(fnmatch.fnmatch(card, p) for p in patterns), card


# ---------------------------------------------------------------------------
# T6: the router card, and the Manual answer's message-building/citation
# helpers. Fails on the pre-T6 tree because ``contract.CARDS`` has no
# "router" key at all (``load_card``/``card_sha`` raise ``KeyError`` on the
# ``CARDS[skill]`` lookup), and ``build_router_messages``/``build_manual_
# messages``/``cited`` do not exist yet.
# ---------------------------------------------------------------------------


def test_the_router_card_is_frozen_and_exempt_from_line_ending_conversion():
    """T6's router card must exist on disk, hash without error, and be
    covered by the same ``-text`` exemption every frozen card needs (see
    ``test_the_frozen_cards_are_exempt_from_line_ending_conversion`` above,
    which already walks every ``CARDS`` entry -- this pins the router's own
    membership explicitly, by name, so a future rename of ``CARDS["router"]``
    still has a test naming exactly what broke)."""
    import fnmatch

    assert "router" in contract.CARDS
    card_path = Path(contract.__file__).resolve().parent / "cards" / contract.CARDS["router"]
    assert card_path.is_file()
    assert contract.card_sha("router") == hashlib.sha256(card_path.read_bytes()).hexdigest()

    patterns = _unset_text_patterns((ROOT / ".gitattributes").read_text(encoding="utf-8"))
    rel = "src/warlock/studio/familiar/cards/" + contract.CARDS["router"]
    assert any(fnmatch.fnmatch(rel, p) for p in patterns), rel


def _citation(n: int, *, chapter: str = "07-clay", anchor: str | None = None):
    from warlock.studio.familiar import retrieval

    return retrieval.Citation(
        n=n, chapter=chapter, anchor=anchor, title_path=f"07 Clay > Section {n}", text=f"text {n}"
    )


def test_cited_keeps_only_citations_the_reply_actually_names():
    """A reply naming ``[1]`` and ``[3]`` but not ``[2]`` must keep exactly
    those two, in the order the markers appear in the reply -- not source
    order -- and never repeat one a reply cites twice."""
    citations = [_citation(1), _citation(2), _citation(3)]
    reply = "Export via glTF [3]. It uses the same pipeline [1] as before [1]."

    result = contract.cited(reply, citations)

    assert [c.n for c in result] == [3, 1]


def test_cited_ignores_a_citation_number_that_was_never_retrieved():
    """``[9]`` with only three citations retrieved must be dropped rather
    than raising or producing a citation object with fabricated content --
    a dead link is worse than no link at all."""
    citations = [_citation(1), _citation(2), _citation(3)]
    reply = "See [9] for details, or just [2]."

    result = contract.cited(reply, citations)

    assert [c.n for c in result] == [2]


def test_manual_messages_number_every_excerpt_with_its_section():
    """Every citation must appear in the user turn as ``[n] <title path>``
    followed by its own text, so the model's own ``[n]`` markers in the
    reply can only ever mean one of these numbered sections."""
    citations = [_citation(1, chapter="07-clay"), _citation(2, chapter="13-troupe")]

    messages = contract.build_manual_messages("how do I export?", citations)

    assert messages[0] == {"role": "system", "content": contract.MANUAL_SYSTEM}
    user = messages[1]["content"]
    assert "[1] 07 Clay > Section 1" in user
    assert "text 1" in user
    assert "[2] 07 Clay > Section 2" in user
    assert "text 2" in user
    assert "how do I export?" in user
    # The prompt itself comes last, after every excerpt.
    assert user.index("how do I export?") > user.index("text 2")


def test_the_manual_prompt_fits_one_slot_at_the_retrieval_budget():
    """``retrieval.Index.search``'s own default ``budget_tokens`` (2500),
    plus the system prompt, plus ``SAMPLING["manual"]["max_tokens"]`` (768),
    inflated by :data:`contract.WORD_TOKEN_SAFETY` to cover a whitespace
    word undercounting real BPE tokens, must still fit inside one
    llama-server slot (``contract.TRAINED_WINDOW``) -- the real constraint a
    Manual turn runs under."""
    import inspect

    from warlock.studio.familiar import retrieval

    budget_tokens = inspect.signature(retrieval.Index.search).parameters["budget_tokens"].default
    system_words = len(contract.MANUAL_SYSTEM.split())
    # A generous stand-in for the excerpts' own title-path overhead and a
    # real user question, on top of the retrieval budget itself.
    overhead_words = 200

    est_input_tokens = (budget_tokens + system_words + overhead_words) * contract.WORD_TOKEN_SAFETY
    total = est_input_tokens + contract.SAMPLING["manual"]["max_tokens"]

    assert total <= contract.TRAINED_WINDOW, (
        f"manual prompt estimated at {est_input_tokens:.0f} tokens plus a "
        f"{contract.SAMPLING['manual']['max_tokens']}-token reply is {total:.0f} -- "
        f"over the {contract.TRAINED_WINDOW}-token slot"
    )
