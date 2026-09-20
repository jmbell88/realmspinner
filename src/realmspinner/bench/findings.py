"""The studio side of ``findings.json`` -- a pure stdlib reader with no
import of torch or imgui, so a generate pane can call it every frame.

``load`` is the whole cost model: one ``stat()`` per call, and a re-read only
when the file's mtime moves. A missing file is cached as a miss too, keyed on
the path alone (there is no mtime to key on), so a bench dir that has never
been reported costs one failed ``stat()`` per frame forever, not an
exception path. ``hint`` and ``comparison_lines`` are pure lookups against
the loaded doc -- ``service/findings.py`` owns the schema (``params``
marginals keyed by ``str(value)``, wilson-ranked ``vectors``, matched-pair
``comparisons``) and this reader tolerates every older shape it ever wrote.

Strings here render through imgui's Inter atlas, which is baked with the
default Basic-Latin + Latin-1 glyph range: the middle dot (U+00B7) is safe,
but anything past U+00FF (a real ``>=`` sign, a Greek delta) would come out
as the missing-glyph box -- hence "41%+", "worst-hole" and "x4" rather than the
typographically nicer spellings.

Grades (findings v4) are rendered wherever the document carries them and
omitted, to the character, where it does not: a v3 file keeps saying
"accept 6/8 (41%+)" and a v4 one says "usable 6/8 (41%+) · avg +2.6". The
absence of an average is never a zero -- zero is a real grade on this scale
("no opinion either way"), so it cannot also stand for "nobody has said".
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

# {path: (mtime_or_None, doc_or_None)} -- mtime is None for a cached miss
# (file absent or unreadable), which can never collide with a real mtime.
_CACHE: dict[Path, tuple[float | None, dict[str, Any] | None]] = {}


def load(path: Path) -> dict[str, Any] | None:
    """The parsed ``findings.json`` at ``path``, or ``None`` if it is absent
    or unreadable. Cached by mtime: unchanged since the last call returns the
    same object with no re-read."""
    try:
        mtime = path.stat().st_mtime
    except OSError:
        _CACHE[path] = (None, None)
        return None

    cached = _CACHE.get(path)
    if cached is not None and cached[0] == mtime:
        return cached[1]

    try:
        import json

        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # Cached by *this* mtime, not None: stat() succeeds for a
        # corrupt-but-present file, so caching (None, None) here would never
        # match a real mtime and every call would re-read and re-parse the
        # same broken file, forever -- exactly the per-frame cost the mtime
        # gate exists to avoid.
        _CACHE[path] = (mtime, None)
        return None

    _CACHE[path] = (mtime, doc)
    return doc


def _params_section(doc: dict[str, Any], prompt_hash: str | None) -> dict[str, Any]:
    """The marginals to look ``param`` up in: this subject's when it has any,
    the pooled ones otherwise. Every access is defensive -- the file crossed a
    disk and this runs on the frame thread."""
    if prompt_hash:
        prompts = doc.get("prompts")
        scope = prompts.get(prompt_hash) if isinstance(prompts, dict) else None
        scoped = scope.get("params") if isinstance(scope, dict) else None
        if isinstance(scoped, dict):
            return scoped
    return doc.get("params") or {}


def _lookup(bucket: dict[str, Any], value: Any) -> dict[str, Any] | None:
    """``bucket[str(value)]``, with the float32 second pass. See ``hint``."""
    entry = bucket.get(str(value))
    if entry is not None or not isinstance(value, float):
        return entry
    target = round(value, 6)
    for key, candidate in bucket.items():
        try:
            key_value = float(key)
        except ValueError:
            continue
        if round(key_value, 6) == target:
            return candidate
    return None


def _value_matches(key: str, value: Any) -> bool:
    """Whether a bucket key names ``value``, by ``_lookup``'s own rule.

    Factored out so :func:`best_value` can ask "is the leader already what is
    set" with the identical float32 tolerance ``_lookup`` uses for a slider --
    two roundings of the same rule would drift the day one of them didn't.
    """
    if key == str(value):
        return True
    if not isinstance(value, float):
        return False
    try:
        candidate = float(key)
    except ValueError:
        return False
    return round(candidate, 6) == round(value, 6)


def _as_float(value: Any, default: float) -> float:
    """``value`` as a float, or ``default`` when it is not a real number --
    ``bool`` excluded, since it is an ``int`` in Python and would silently
    become 0.0/1.0."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return default


def _best_in(section: dict[str, Any], param: str, min_n: int) -> tuple[str, dict[str, Any]] | None:
    """The ``(value_str, entry)`` clearing ``min_n`` in ``section[param]`` that
    maximises ``(wilson_low, mean_grade, n)`` -- the exact tie-break order
    ``service.findings.aggregate`` sorts ``vectors`` by, so "best" here means
    the same thing it means in the Review ranking, not a second opinion on it.

    A bucket with no grade sorts as if its mean were ``-inf``, ``aggregate``'s
    own rule for the same reason: a bucket nobody graded is not tied with one
    graded exactly zero, which is a real grade on this scale.
    """
    candidates = section.get(param) if isinstance(section, dict) else None
    if not isinstance(candidates, dict):
        return None
    best: tuple[str, dict[str, Any]] | None = None
    best_key: tuple[float, float, int] | None = None
    for value_str, entry in candidates.items():
        if not isinstance(entry, dict):
            continue
        n = entry.get("n")
        if not isinstance(n, int) or isinstance(n, bool) or n < min_n:
            continue
        wilson = _as_float(entry.get("wilson_low"), 0.0)
        mean = _as_float(entry.get("mean_grade"), -math.inf)
        key = (wilson, mean, n)
        if best is None or key > best_key:  # type: ignore[operator]
            best, best_key = (value_str, entry), key
    return best


def best_value(
    doc: dict[str, Any] | None,
    param: str,
    value: Any,
    *,
    min_n: int,
    prompt_hash: str | None = None,
) -> tuple[str, dict[str, Any], str] | None:
    """``(value_str, entry, scope_label)`` for the best-scoring bucket of
    ``param``, or ``None`` when there is nothing worth offering.

    "Best" is the same Wilson-first, mean-grade-second, n-third ordering
    ``aggregate`` ranks whole vectors by (see :func:`_best_in`); "worth
    offering" excludes two cases. Nothing clears ``min_n`` -- a thin bucket is
    noise, exactly the bar ``hint`` already holds a single value to. And the
    leader *is* the current value, under ``_lookup``'s float32 rule -- a
    button that writes back the value already set is not an offer, it is
    theatre, and it is exactly what an unrounded comparison would produce for
    every float32 slider sitting on its own best bucket.

    Scoped the same way ``hint`` is: a subject with ``min_n`` behind it
    answers for itself, labelled ``"this subject"``; short of that the pooled
    corpus answers, labelled ``"all subjects"``; with no ``prompt_hash`` at
    all the pooled corpus answers unlabelled, because a caller that does not
    know its subject is making no claim about one.
    """
    if doc is None:
        return None

    def offer(
        found: tuple[str, dict[str, Any]] | None, scope: str
    ) -> tuple[str, dict[str, Any], str] | None:
        if found is None:
            return None
        value_str, entry = found
        if _value_matches(value_str, value):
            return None
        return (value_str, entry, scope)

    if not prompt_hash:
        return offer(_best_in(doc.get("params") or {}, param, min_n), "")

    scoped = _params_section(doc, prompt_hash)
    pooled = doc.get("params") or {}
    if scoped is not pooled:
        found = _best_in(scoped, param, min_n)
        if found is not None:
            return offer(found, "this subject")
    return offer(_best_in(pooled, param, min_n), "all subjects")


def best_value_line(entry: dict[str, Any], scope: str) -> str:
    """The best-value offer's own line -- ``"7/8 usable (47%+) · avg +2.9 ·
    this subject"`` -- deliberately reordered from :func:`hint`'s "usable 6/8"
    so the two never look like the same claim at a glance: one describes what
    the *current* value scored, the other what the *offered* one did.
    """
    n = entry.get("n", 0) if isinstance(entry, dict) else 0
    n = n if isinstance(n, int) and not isinstance(n, bool) else 0
    accepts = entry.get("accepts", 0) if isinstance(entry, dict) else 0
    accepts = accepts if isinstance(accepts, int) and not isinstance(accepts, bool) else 0
    average = _mean_grade(entry)
    noun = "accept" if average is None else "usable"
    base = f"{accepts}/{n} {noun}"
    bound = entry.get("wilson_low") if isinstance(entry, dict) else None
    if isinstance(bound, (int, float)) and not isinstance(bound, bool):
        base += f" ({round(bound * 100)}%+)"
    if average is not None:
        base += f" · avg {average}"
    if scope:
        base += f" · {scope}"
    return base


def hint(
    doc: dict[str, Any] | None,
    param: str,
    value: Any,
    *,
    min_n: int = 5,
    prompt_hash: str | None = None,
) -> str | None:
    """``"accept 6/8 (41%+)"`` for ``param``/``value`` in ``doc``, or the
    machine-evidence fallback, or ``None`` when neither has ``min_n`` behind
    it -- a thin bucket is noise, not a finding.

    Three tiers. Enough verdicts: the count plus the writer's Wilson lower
    bound as a floor of confidence ("41%+"); a v1 file that never computed
    the bound still hints as the bare count. Too few verdicts but enough
    *observations*: what the worker measured -- ``"holes 3% · watertight 71%
    (21 meshes)"`` -- so a value generated with often but never reviewed still
    says something true. Otherwise nothing.

    ``findings.json`` keys a bucket by ``str()`` of the JSON value the sweep
    recorded (e.g. ``"0.6"``), but an imgui float32 slider hands back
    ``0.6000000238418579`` -- the float32 rounding of ``0.6``, not the
    ``float`` Python would make from the string. A straight ``str(value)``
    lookup misses every such slider, silently, forever. So a float value that
    misses the literal-string lookup gets a second pass: every bucket key
    that itself parses as a float is compared to ``value`` rounded to 6
    decimals, which absorbs float32's error (~1e-7 relative) while still
    telling "0.6" apart from "0.65".

    ``prompt_hash`` scopes the lookup to one subject. A verdict
    crediting every ``param: value`` in its vector is a confound this project
    accepts -- the price of letting daily use feed the hints at all -- but
    pooling across *subjects* is a different bargain: what makes a good wooden
    crate says very little about what makes a good character. So a subject with
    enough behind it answers for itself, a thin one falls back to the pooled
    corpus rather than to silence, and **the line says which**, because an
    unlabelled fallback is exactly the silent mixing this exists to stop.

    Passing nothing keeps the old behaviour to the character, label included:
    a caller that does not know its subject is making no claim about one.
    """
    if doc is None:
        return None

    def rendered(section: dict[str, Any]) -> str | None:
        entry = _lookup(section.get(param) or {}, value)
        if entry is None:
            return None
        if entry.get("n", 0) >= min_n:
            # "usable" once the file carries grades, "accept" when it does not:
            # the count is the same derived cut either way, but on a v4 file
            # there is a scale behind it and the word should say which question
            # was answered. A v3 file renders the v3 string to the character.
            average = _mean_grade(entry)
            noun = "accept" if average is None else "usable"
            base = f"{noun} {entry.get('accepts', 0)}/{entry['n']}"
            bound = entry.get("wilson_low")
            if isinstance(bound, (int, float)) and not isinstance(bound, bool):
                base += f" ({round(bound * 100)}%+)"
            if average is not None:
                base += f" · avg {average}"
            return base
        return _metrics_hint(entry.get("metrics"), min_n)

    if not prompt_hash:
        return rendered(doc.get("params") or {})

    scoped = _params_section(doc, prompt_hash)
    pooled = doc.get("params") or {}
    if scoped is not pooled:
        line = rendered(scoped)
        if line is not None:
            return f"{line} · this subject"
    line = rendered(pooled)
    return None if line is None else f"{line} · all subjects"


def _mean_grade(entry: Any) -> str | None:
    """A bucket's mean grade, rendered signed -- ``"+2.6"``, ``"-1.4"`` -- or
    ``None`` when the bucket carries none.

    ``None`` covers three different situations on purpose and they are the same
    situation to a reader: a v3 file written before grades existed, a v4 bucket
    holding only image labels, and a bucket nobody has graded. In all three
    there is no average to state, and the line simply does not carry one rather
    than carrying a zero -- zero is a real grade on this scale ("no opinion
    either way"), so it cannot also stand for "nobody has said".

    Always signed, including ``"+0.0"``: an unsigned ``0.0`` beside grades that
    are otherwise written ``+3`` reads as a different kind of number.
    """
    if not isinstance(entry, dict):
        return None
    mean = entry.get("mean_grade")
    if not isinstance(mean, (int, float)) or isinstance(mean, bool):
        return None
    return f"{float(mean):+.1f}"


def tag_line(entry: Any) -> str | None:
    """A bucket's tag tallies on one muted line -- ``"good: good-texture x4,
    on-style x2 · bad: holes x3"`` -- or ``None`` when nothing is tagged.

    Top three per polarity, which is what keeps the line one line. The polarity
    split is read off the document rather than derived here: the writer splits by
    membership in its own vocabulary, which is what lets this module stay pure
    stdlib with no ``realmspinner`` import.

    ``x`` rather than a multiplication sign for the reason every other string
    here is ASCII: the atlas is Basic-Latin + Latin-1, and U+00D7 is inside it
    but reads as a dimension separator. The middle dot is the one non-ASCII
    character used, as everywhere else in this module.
    """
    if not isinstance(entry, dict):
        return None
    tags = entry.get("tags")
    if not isinstance(tags, dict):
        return None
    parts: list[str] = []
    for polarity in ("good", "bad"):
        rendered = _tag_group(tags.get(polarity))
        if rendered:
            parts.append(f"{polarity}: {rendered}")
    return " · ".join(parts) if parts else None


def _tag_group(pairs: Any) -> str:
    if not isinstance(pairs, list):
        return ""
    shown: list[str] = []
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        tag, count = pair
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            continue
        shown.append(f"{tag} x{count}")
        if len(shown) == 3:
            break
    return ", ".join(shown)


def _reading(metrics: dict[str, Any], key: str, min_n: int) -> tuple[float, int] | None:
    """``(value, how many observations carried it)``, or None below ``min_n``.

    The count is the metric's own, not the bucket's. The audit can succeed
    while ``meshreport.build`` fails or returns ``status: "invalid"``, so a
    bucket of twenty-one observations can hold exactly one ``watertight``
    reading -- which the bucket count would have advertised as twenty-one.
    Falling back to ``n`` keeps a doc written before ``counts`` existed
    readable, where that assumption was at least the one being made.
    """
    value = metrics.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    counts = metrics.get("counts")
    count = counts.get(key) if isinstance(counts, dict) else None
    if not isinstance(count, int) or isinstance(count, bool):
        count = metrics.get("n")
    if not isinstance(count, int) or isinstance(count, bool) or count < min_n:
        return None
    return float(value), count


def _refusal_hint(metrics: dict[str, Any], min_n: int) -> str | None:
    """The composition gate's rate, when it is worth saying.

    Its own tier rather than a part joined into ``_measured``, and the word
    *references* rather than *meshes* is why: a refused job never reached a
    mesh, so a count of them under the "(21 meshes)" suffix would be a claim
    about geometry that does not exist. It also leads rather than trails --
    "this checkpoint is refused half the time" outranks its hole fraction,
    which is measured only over the half that got through.

    Silent at zero: "refused 0%" under every control is noise, and the absence
    of the line already says it.
    """
    reading = _reading(metrics, "refused_rate", min_n)
    if reading is None or reading[0] <= 0:
        return None
    percent = round(reading[0] * 100)
    return f"refused {percent}% ({reading[1]} references)"


def _measured(parts: list[tuple[str, int]]) -> str:
    """Join measurements, saying how many meshes each rests on.

    One trailing count while they agree -- which is the ordinary case, since
    both measurements usually succeed together -- and a count per measurement
    the moment they do not, because a single number after two metrics of
    different weight is a claim about whichever one the reader assumes.

    *meshes*, not *runs*, and that word is doing work. These lines appear
    under 2D prompt controls as well as mesh ones, because an observation
    credits every param in its vector exactly as a verdict does -- so
    "style_lora pixelxl" carries the hole fraction of the meshes reconstructed
    from pixel-style references. That is a true and useful statement, but only
    if the reader can tell it is about geometry rather than about the picture.
    """
    if len({count for _text, count in parts}) == 1:
        return f"{' · '.join(text for text, _ in parts)} ({parts[0][1]} meshes)"
    return " · ".join(f"{text} ({count} meshes)" for text, count in parts)


def _metrics_hint(metrics: Any, min_n: int) -> str | None:
    """The machine-evidence tier, defensively: the file crossed a disk.

    Each measurement earns its place separately: ``min_n`` is a threshold on
    the readings behind *that number*, so a thin one is dropped rather than
    carried along by a fat one in the same bucket.
    """
    if not isinstance(metrics, dict):
        return None
    refused = _refusal_hint(metrics, min_n)
    if refused is not None:
        return refused
    parts: list[tuple[str, int]] = []
    hole = _reading(metrics, "hole_worst", min_n)
    if hole is not None:
        parts.append((f"holes {round(hole[0] * 100)}%", hole[1]))
    watertight = _reading(metrics, "watertight_rate", min_n)
    if watertight is not None:
        parts.append((f"watertight {round(watertight[0] * 100)}%", watertight[1]))
    if not parts:
        return None
    return _measured(parts)


def metrics_line(metrics: Any) -> str | None:
    """A recipe's machine evidence on one muted line -- ``"holes 3% worst ·
    watertight 71% · 24,120 tri (21 meshes)"`` -- or ``None`` when nothing was
    measured. Unlike the hint tier this has no threshold: it annotates a
    recipe that already earned its place with verdicts."""
    if not isinstance(metrics, dict):
        return None
    parts: list[tuple[str, int]] = []
    hole = _reading(metrics, "hole_worst", 1)
    if hole is not None:
        parts.append((f"holes {round(hole[0] * 100)}% worst", hole[1]))
    watertight = _reading(metrics, "watertight_rate", 1)
    if watertight is not None:
        parts.append((f"watertight {round(watertight[0] * 100)}%", watertight[1]))
    triangles = _reading(metrics, "triangles", 1)
    if triangles is not None:
        parts.append((f"{int(triangles[0]):,} tri", triangles[1]))
    if not parts:
        return None
    return _measured(parts)


def corpus_line(doc: dict[str, Any] | None) -> str:
    """The corpus at a glance -- ``"42 graded meshes · 3 of 19 configurations
    rank · 2 contrasts settled, 4 open"`` -- or ``""`` when the document
    carries no ``corpus`` section (a file written before findings v5's other
    half, or one that crossed a disk and lost the key).

    Reads ``service.findings._corpus`` verbatim; nothing here recomputes a
    count the writer already produced, which is the same discipline every
    other reader in this module follows.
    """
    if not isinstance(doc, dict):
        return ""
    corpus = doc.get("corpus")
    if not isinstance(corpus, dict):
        return ""
    graded_n = _int_count(corpus.get("graded_n"))
    configs_n = _int_count(corpus.get("configs_n"))
    configs_ranked = _int_count(corpus.get("configs_ranked"))
    settled = _int_count(corpus.get("contrasts_settled"))
    open_ = _int_count(corpus.get("contrasts_open"))
    return (
        f"{graded_n} graded mesh{'' if graded_n == 1 else 'es'} · "
        f"{configs_ranked} of {configs_n} configuration{'' if configs_n == 1 else 's'} rank · "
        f"{settled} contrast{'' if settled == 1 else 's'} settled, {open_} open"
    )


def nearest_rank_gap(doc: dict[str, Any] | None, *, min_n: int = 5) -> int | None:
    """How many more verdicts the closest short-of-threshold configuration
    needs before it ranks, or ``None`` when there is no such configuration --
    an empty/missing ``vectors`` section, or one where every configuration
    already ranks.

    Reads ``vectors`` rather than ``corpus``: what a "nothing yet" message
    wants to say is how close the ``top`` list Review already renders is to
    having an entry, and ``n`` on each of those rows already means exactly
    that -- a second computation from the corpus counts could disagree with
    it on a hand-edited file.
    """
    if not isinstance(doc, dict):
        return None
    vectors = doc.get("vectors")
    if not isinstance(vectors, list):
        return None
    best: int | None = None
    for entry in vectors:
        if not isinstance(entry, dict):
            continue
        n = entry.get("n")
        if not isinstance(n, int) or isinstance(n, bool) or n >= min_n:
            continue
        gap = min_n - n
        if best is None or gap < best:
            best = gap
    return best


def sample_jobs(entry: Any) -> list[str]:
    """The job ids a ranked vector's verdicts were drawn from -- what Review's
    "Show examples" opens the Library onto.

    ``service.findings.aggregate`` has written ``jobs`` onto every vector
    entry since the field existed; this is only the defensive read, to the
    same rule every other accessor here follows -- the file crossed a disk, and
    a v3 document written before the key existed, or a hand-edited one with
    junk in the list, must hand back an empty list rather than raise on the
    frame thread. A finding with no known samples disables the button rather
    than opening the Library onto nothing.
    """
    if not isinstance(entry, dict):
        return []
    jobs = entry.get("jobs")
    if not isinstance(jobs, list):
        return []
    return [j for j in jobs if isinstance(j, str) and j]


def vector_line(entry: Any) -> str:
    """A ranked recipe's headline -- ``"usable 80% of 20 (61%+) · avg +2.6"``,
    or ``"80% of 20 (61%+)"`` on a file written before grades existed.

    The bound is *omitted* rather than defaulted to zero when the file does not
    carry one. A v1 findings.json never computed it and ranked by the raw rate,
    so substituting 0.0 labelled every recipe in an existing user's Review pane
    "(0%+)" -- a confidence floor of nothing, under a heading claiming the
    Wilson ranking. ``hint`` already drops the suffix on the same file for the
    same reason; this is that rule applied where the ranking is displayed.

    Pure, so the wording is assertable without a GL context.
    """
    if not isinstance(entry, dict):
        return ""
    n = _int_count(entry.get("n"))
    rate = entry.get("accept_rate")
    ok = isinstance(rate, (int, float)) and not isinstance(rate, bool)
    percent = round(float(rate) * 100) if ok else 0
    # The rate is the derived usable cut on a graded file, and saying so is the
    # point: "80%" alone under a grade scale invites the reader to think it is
    # the scale. On a legacy file there is no scale and the bare figure is what
    # every existing Review pane already shows.
    average = _mean_grade(entry)
    line = f"{percent}% of {n}" if average is None else f"usable {percent}% of {n}"
    bound = entry.get("wilson_low")
    if isinstance(bound, (int, float)) and not isinstance(bound, bool):
        line += f" ({round(bound * 100)}%+)"
    if average is not None:
        line += f" · avg {average}"
    return line


# What a delta line calls each metric, and how its mean is rendered. Holes and
# the two rates are differences of fractions (percent, one decimal, signed);
# triangles is a count.
_DELTA_LABELS = {
    "hole_worst": "worst-hole",
    "hole_mean": "mean-hole",
    "watertight": "watertight",
    "ready": "ready",
}


def suggestion_line(s: dict[str, Any]) -> str:
    """One suggestion from ``review_mode.suggest_sweeps``, said in words --
    ``"trellis_gss: 3.0 vs unset is 3/4 for 3.0 - 1 more matched pair settles
    it"``. Pure, and defensive to the same rule everything else in this module
    follows: the suggestion crossed no disk but is still handed in from a pane
    on the frame thread.

    ASCII-only, this module's own rule: " - " rather than an em dash, for the
    same atlas-coverage reason every other line here spells it out.
    """
    if not isinstance(s, dict):
        return ""
    param = str(s.get("param", "?"))
    a = str(s.get("a", "?"))
    b = str(s.get("b", "?"))
    leader = str(s.get("leader", "?"))
    pairs = _int_count(s.get("pairs"))
    needed = _int_count(s.get("needed"))
    leader_wins = _int_count(s.get("leader_wins"))
    tail = (
        "1 more matched pair settles it"
        if needed == 1
        else f"{needed} more matched pairs settle it"
    )
    return f"{param}: {a} vs {b} is {leader_wins}/{pairs} for {leader} - {tail}"


def comparison_lines(doc: dict[str, Any] | None, *, min_pairs: int = 5) -> list[str]:
    """The axis verdicts, as printable lines, winner first.

    One header line per contrast with at least ``min_pairs`` matched human
    pairs -- ``"lora_weight: 0.6 beat 0.9 in 7 of 8 matched pairs (2 sweeps,
    2 prompts)"`` -- followed by an indented delta line per machine metric
    with ``min_pairs`` paired runs behind it. A contrast with machine pairs
    but too few human ones still shows its deltas under a "machine evidence
    only" header: a sweep says something the moment it finishes."""
    if not isinstance(doc, dict):
        return []
    comparisons = doc.get("comparisons")
    if not isinstance(comparisons, dict):
        return []
    lines: list[str] = []
    for param in sorted(comparisons):
        entries = comparisons[param]
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            lines.extend(_entry_lines(param, entry, min_pairs))
    return lines


def _int_count(value: Any) -> int:
    """A count from a doc that crossed a disk: anything that is not an int
    (or is the bool impostor) is zero, never a raise on the frame thread."""
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _entry_lines(param: str, entry: dict[str, Any], min_pairs: int) -> list[str]:
    pairs = _int_count(entry.get("pairs"))
    a, b = str(entry.get("a", "?")), str(entry.get("b", "?"))
    a_wins = _int_count(entry.get("a_wins"))
    b_wins = _int_count(entry.get("b_wins"))

    raw_deltas: list[tuple[str, float, int]] = []
    deltas_src = entry.get("deltas")
    if isinstance(deltas_src, dict):
        for metric, stat in sorted(deltas_src.items()):
            if not isinstance(stat, dict):
                continue
            delta_pairs = _int_count(stat.get("pairs"))
            mean = stat.get("mean")
            if delta_pairs < min_pairs:
                continue
            if not isinstance(mean, (int, float)) or isinstance(mean, bool):
                continue
            raw_deltas.append((str(metric), float(mean), delta_pairs))

    def delta_lines(flip: bool) -> list[str]:
        # The stored mean is a-minus-b (a = lexicographically low value), but
        # these lines sit unattributed under a header that leads with the
        # *winner* -- so when b won, the sign must be re-oriented to
        # winner-minus-loser or the same outcome reads backwards depending on
        # which value happens to sort first.
        lines = []
        for metric, mean, delta_pairs in raw_deltas:
            shown_mean = -mean if flip else mean
            shown = (
                f"{shown_mean:+,.0f}" if metric == "triangles" else f"{shown_mean * 100:+.1f}%"
            )
            label = _DELTA_LABELS.get(metric, metric)
            lines.append(f"    {label} {shown} over {delta_pairs} paired runs")
        return lines

    def grade_gap(flip: bool) -> str:
        """How much better the winner graded, winner-minus-loser.

        The stored mean is a-minus-b like every other delta here, so it takes
        the same re-orientation ``delta_lines`` does -- a header leading with the
        winner and a signed number oriented the other way is the failure that
        machinery exists to prevent, and it would be less visible here because a
        grade gap is plausible in either direction.

        Omitted entirely on a file with no graded pairs, rather than shown as
        zero: a zero mean is a real reading (the two sides graded the same on
        average) and cannot also mean nobody graded them.
        """
        stat = entry.get("grade_delta")
        if not isinstance(stat, dict):
            return ""
        mean = stat.get("mean")
        if not isinstance(mean, (int, float)) or isinstance(mean, bool):
            return ""
        shown = -float(mean) if flip else float(mean)
        return f", avg {shown:+.1f} grade"

    def plural(count: int, noun: str) -> str:
        return f"{count} {noun}{'' if count == 1 else 's'}"

    breadth = (
        f"({plural(_int_count(entry.get('sweeps')), 'sweep')},"
        f" {plural(_int_count(entry.get('prompts')), 'prompt')})"
    )
    if pairs >= min_pairs:
        if a_wins == b_wins:
            header = f"{param}: {a} vs {b} - tied over {pairs} matched pairs {breadth}"
            return [header, *delta_lines(False)]
        winner, loser = (a, b) if a_wins > b_wins else (b, a)
        flip = b_wins > a_wins
        header = (
            f"{param}: {winner} beat {loser} in {max(a_wins, b_wins)} of {pairs}"
            f" matched pairs{grade_gap(flip)} {breadth}"
        )
        return [header, *delta_lines(flip)]
    if raw_deltas:
        # The "a vs b" header names a first, so the stored orientation stands.
        return [f"{param}: {a} vs {b} - machine evidence only", *delta_lines(False)]
    return []
