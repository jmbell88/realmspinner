"""The held-out corpus and the leak check every draft prompt must pass.

``docs/measurements/corpora/clay-agent-v1.txt`` is the pre-registered tier-two
eval (see that file's own header comment and
``docs/measurements/2026-09-10-clay-agent-benchmark-preregistration.md``): a
training row built from one of its five subjects, or a close paraphrase of
one, would let the fine-tune memorise the very benchmark it is meant to be
measured against. ``is_leak`` is the guard ``build.py`` runs every prompt
through before it is ever replayed.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_SRC = _ROOT / "src"
_SCRIPTS = _ROOT / "scripts"
for _p in (_SRC, _SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import campaign_props  # noqa: E402  (scripts/campaign_props.py, see above)

CORPUS_PATH = _ROOT / "docs" / "measurements" / "corpora" / "clay-agent-v1.txt"

STOPWORDS = frozenset(
    {
        "a", "an", "the", "on", "with", "of", "from", "to", "and", "its",
        "whose", "that", "in", "at", "by",
    }
)
"""Dropped before bigrams are formed -- see :func:`is_leak`'s own docstring
for why the bigram check needs this and the Jaccard check does not."""

JACCARD_THRESHOLD = 0.6

_PUNCT_RE = re.compile(r"[^\w\s-]")
_WS_RE = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Lowercased, punctuation collapsed to whitespace, whitespace collapsed
    to single spaces -- the one normalisation both the exact-match and the
    token-set checks in :func:`is_leak` share, so neither can disagree with
    the other about what "the same text" means."""
    text = text.lower().strip()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def _content_tokens(normalised: str) -> list[str]:
    return [t for t in normalised.split() if t not in STOPWORDS]


def _bigrams(tokens: list[str]) -> frozenset[tuple[str, str]]:
    return frozenset(zip(tokens, tokens[1:], strict=False))


@dataclass(frozen=True)
class Holdout:
    """The corpus, pre-processed once at load time rather than per prompt --
    :func:`is_leak` is called once per draft record, and re-tokenising five
    corpus lines on every call would be silly work repeated thousands of
    times over a full ``build.py`` run."""

    lines: tuple[str, ...]
    normalised: tuple[str, ...]
    tokens: tuple[frozenset[str], ...]
    bigrams: tuple[frozenset[tuple[str, str]], ...]


def load_holdout(path: Path = CORPUS_PATH) -> Holdout:
    """Every prompt in the pre-registered corpus, read through
    ``campaign_props.read_corpus`` -- the same ``class | prompt`` parser the
    campaign submitter uses, reused rather than re-written so a malformed
    corpus line is refused the same way in both places."""
    subjects = campaign_props.read_corpus(path)
    lines = tuple(s.prompt for s in subjects)
    normalised = tuple(normalise(line) for line in lines)
    tokens = tuple(frozenset(n.split()) for n in normalised)
    bigrams = tuple(_bigrams(_content_tokens(n)) for n in normalised)
    return Holdout(lines=lines, normalised=normalised, tokens=tokens, bigrams=bigrams)


def is_leak(prompt: str, holdout: Holdout) -> str | None:
    """The offending corpus line, if *prompt* is the held-out corpus wearing
    any of three disguises -- ``None`` if it is none of them.

    1. **Exact match**, once both are :func:`normalise`\\ d -- catches a
       prompt copied verbatim, punctuation and casing aside.
    2. **Token-set Jaccard >= 0.6** -- catches a prompt that reuses most of a
       corpus line's own words in a different order or with a word or two
       swapped.
    3. **A shared "distinctive" bigram** -- a corpus line's own content
       words, adjacent, with :data:`STOPWORDS` dropped first. This is what
       catches a paraphrase Jaccard misses entirely: "a wooden chair with
       four legs and a slatted backrest" shares only 5 of 11 raw words with
       "a four-legged wooden chair with a slatted back" (Jaccard ~0.45), but
       both share the bare bigram ``("wooden", "chair")`` once "a"/"with"
       are out of the way -- which is the phrase actually doing the
       identifying work, and exactly the kind of near-neighbour the plan's
       corpus header warns a training row must never rebuild.
    """
    normalised = normalise(prompt)
    prompt_tokens = frozenset(normalised.split())
    prompt_bigrams = _bigrams(_content_tokens(normalised))

    for line, line_norm, line_tokens, line_bigrams in zip(
        holdout.lines, holdout.normalised, holdout.tokens, holdout.bigrams, strict=True
    ):
        if normalised == line_norm:
            return line
        if line_tokens:
            union = prompt_tokens | line_tokens
            if union and len(prompt_tokens & line_tokens) / len(union) >= JACCARD_THRESHOLD:
                return line
        if prompt_bigrams and (prompt_bigrams & line_bigrams):
            return line
    return None
