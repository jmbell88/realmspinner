"""What a character recipe means: the plan, the validation, the kwargs.

Split out of ``modes/create/ui/settings_character.py`` (2026-09-18
restructure, P5) -- the half of that module with no imgui in it. Familiar
(``assistant/ui.py``, ``assistant/doors.py``), Poser's character-sheet stage
(``modes/poser/mode.py`` -- Troupe's own vocabulary, folded in whole by P9,
2026-09-18) and the shell (``state.py``, ``shell/tasks.py``,
``modes/create/ui/workspace.py``) all read this vocabulary today by reaching
into a *pane*; they import this
module directly now. What stays in ``modes/create/ui/settings_character.py``
is the drawing and the orchestration of a press (``draw_block``, the field
callbacks, ``preflight_fix``'s buttons, and ``submit``, which toasts).

**Nothing here ever substitutes a species.** ``characters.resolve`` refuses
to, by construction (``Resolution.family`` is ``None`` for a creature this
program does not make), and this module keeps that promise: ``character_
family`` stays ``""``, and the one substitution (:func:`apply_offer`) only
runs because a person read the offer sentence and pressed the button that
repeats it.
"""

from __future__ import annotations

import json
from typing import Any

from .....characters import family as family_mod
from .....characters import recipe as recipe_mod
from .....characters import resolve as resolve_mod
from .....service import characters as svc_characters
from .... import problems as problem_types
from . import assets as create_assets

#: ``character_theme``'s "the species' own look" value. Not a theme key: no
#: species declares a ``none`` theme, which is what makes it safe as a sentinel
#: and what makes switching species keep the choice honest -- a fire palette
#: carried onto a creature with no fire would be a look nobody chose.
THEME_UNSET = "none"

#: The movements a character sheet carries, and how many frames each is drawn
#: in. ``recipe.DEFAULT_ANIMATIONS`` rather than the five-row legacy
#: ``charsheet.ANIMATIONS`` table, and the recipe module says why: these three
#: are what a character needs to read as alive in a top-down game. Read from
#: there rather than restated, so the "144 cells" this pane prints and the cells
#: the door plans are one arithmetic.
MOVEMENTS: tuple[tuple[str, int], ...] = tuple(recipe_mod.DEFAULT_ANIMATIONS.items())

#: How many ways each movement is drawn. The recipe's own default, and stated on
#: the camera helper rather than offered as a control: eight is what every
#: preset in ``charsheet.CAMERA_PRESETS`` is laid out for, and a fourth combo
#: for a value with one sensible answer is a control that cannot be operated.
DIRECTIONS = 8

#: Where :func:`options` caches the door's answer for the life of the process.
#: ``poser_mode.OPTIONS_SLOT``'s slot and its reason (Troupe's own, before
#: P9 2026-09-18 folded that mode into Poser): ``character_options`` walks
#: the palette directory, and a directory walk sixty times a second is a cost
#: with no reader.
OPTIONS_SLOT = "character_options"

#: Which recipe field each control answers to. The door refuses in the
#: *recipe's* vocabulary (``logical_size``, ``appearance``, ``animations``)
#: because that is what ``Recipe.from_dict`` validates, and the controls are
#: named for the form keys they persist under -- so without this map a refusal
#: about a colour count would be recorded against a name nothing on the pane
#: draws, and the ring would land nowhere. ``settings_character.mirror_errors``
#: is what walks it; ``tests/test_field_error_wiring.py`` is the standing guard
#: on the drift.
RECIPE_FIELDS: dict[str, tuple[str, ...]] = {
    "character_family": ("family", "family_version"),
    "character_theme": ("theme",),
    "character_camera": ("camera", "elevation"),
    "character_actions": ("animations", "directions"),
    "character_pixel": ("logical_size",),
    "character_colors": ("colors",),
    "character_body": ("appearance",),
    "character_name": ("name",),
}

#: The one task key a preview runs under. Its own rather than ``"submit"``, and
#: that is the whole point: a preview is a look at the form and must never make
#: the Generate button busy -- ``TaskRunner.submit`` refuses a key already in
#: flight, so sharing the key would mean a dragged slider could swallow a press.
PREVIEW_KEY = "character-preview"


# --- what the form is asking for ---------------------------------------------


def options(ctx: Any) -> dict[str, Any]:
    """``service.characters.character_options``, kept fresh against the
    palette directory it nests under ``["troupe"]["palettes"]``.

    Keyed on ``stamps.stamp_ns`` of that directory -- ``panes.inspector.
    palette_names``'s own rule for the identical directory -- rather than read
    once for the life of the process. Before the 2026-09-11 audit (finding
    troupe-04) this cached forever with no invalidation, so a palette file
    dropped in while the app was running never appeared in this form until
    restart, even though ``service.palettes``' module docstring states the
    directory's whole design intent is drop-in-while-running use. The rest of
    the answer -- archetypes, families, ladders -- costs nothing to recompute
    alongside it; splitting the palette list out into its own cache slot would
    be a second cache to keep in step with this one for no measured saving.
    """
    from ....panes import stamps

    key = stamps.stamp_ns(ctx.svc.config.palette_dir)
    cached = ctx.state.preview.get(OPTIONS_SLOT)
    if cached is not None and cached[0] == key:
        return cached[1]
    value = svc_characters.character_options(ctx.svc)
    if stamps.storable(key):
        ctx.state.preview[OPTIONS_SLOT] = (key, value)
    return value


def resolution_of(form: dict[str, Any]) -> resolve_mod.Resolution:
    """The cached resolution of the brief, or an empty one.

    Never re-resolves: :func:`sync_from_prompt` owns that, and a scan run from
    a getter would run it per control per frame.
    """
    raw = str(form.get("character_resolution") or "")
    if not raw:
        return resolve_mod.Resolution()
    try:
        return resolve_mod.Resolution.from_dict(json.loads(raw))
    except (ValueError, TypeError, AttributeError):
        # A hand-edited settings file. An empty resolution is the honest
        # answer -- the pane then says the prompt named no species, which is
        # true of a resolution nobody can read.
        return resolve_mod.Resolution()


def overrides_of(form: dict[str, Any]) -> list[str]:
    raw = form.get("character_overrides")
    return [str(v) for v in raw] if isinstance(raw, list) else []


def touched(form: dict[str, Any], key: str) -> None:
    """Record that the user set this control themselves.

    From here on a prompt edit leaves it alone. Recorded on *every* change
    rather than on a change away from the resolved value, because "I typed the
    species the prompt already said" is still a decision, and a form that
    forgot it would move that field the next time the prompt changed.
    """
    overrides = overrides_of(form)
    if key not in overrides:
        overrides.append(key)
    form["character_overrides"] = overrides


def sync_from_prompt(form: dict[str, Any]) -> bool:
    """Re-resolve the brief and fill what the user has not touched. -> did it.

    Called at the top of ``settings_character.draw_block`` *and* at the top of
    ``settings_2d.generate``'s character arm, which are the two doors: the
    keyboard ones (Ctrl+Enter, the command palette) never draw this pane, so a
    form filled only from the draw would submit the species of the *previous*
    prompt for anyone who typed and pressed Ctrl+Enter in one motion.

    Cheap on the common frame: the scan runs only when the prompt differs from
    the one the cached resolution was made from.
    """
    prompt = str(form.get("prompt") or "")
    if form.get("character_resolution") and prompt == str(
        form.get("character_resolution_prompt") or ""
    ):
        return False
    resolution = resolve_mod.resolve(prompt)
    form["character_resolution"] = json.dumps(resolution.to_dict())
    form["character_resolution_prompt"] = prompt
    _fill(form, resolution)
    return True


def reset_to_prompt(form: dict[str, Any]) -> None:
    """Forget every override and take the brief's answer again."""
    form["character_overrides"] = []
    form["character_resolution"] = ""
    form["character_resolution_prompt"] = ""
    sync_from_prompt(form)


def theme_offered(opts: dict[str, Any], family: str, theme: str) -> bool:
    """Whether species *family* offers the look *theme*, out of the door's own
    options.

    Shared by ``settings_character._family`` (a species change on the combo)
    and :func:`apply_offer` (a substitution) -- the collapsed-underscore name
    is why this promoted out of ``settings_character._theme_offered``, which
    it was before the two callers were on opposite sides of the engine/UI
    split.
    """
    return theme in {key for key, _label in theme_options(opts, family)}


def _theme_offered_by(family_key: str, theme_key: str) -> bool:
    """Whether species *family_key* paints the look *theme_key* -- straight
    off the registry, not :func:`theme_options`: that one needs
    ``character_options``' door-built ``opts``, and :func:`_fill` runs from a
    bare form with no ``ctx`` on hand to build one from.
    """
    if not family_key or not theme_key:
        return False
    try:
        fam = family_mod.get_family(family_key)
    except family_mod.CharacterError:
        return False
    return any(t.key == theme_key for t in fam.themes)


def _fill(form: dict[str, Any], resolution: resolve_mod.Resolution) -> None:
    """Write the resolved brief into the fields the user has not claimed.

    **A field the prompt says nothing about goes back to its default**, not to
    whatever the last prompt left in it. "a wolf" after "an attacking fire ogre"
    has to produce a wolf with the default actions and no fire, or the form
    accumulates a character out of two briefs the user never wrote together.

    **Except a theme the resolved species does not paint.** "a swamp knight"
    resolves ``family="knight"`` and ``theme="swamp"`` -- "swamp" is a real
    theme word, just not one the Knight declares -- and copying it verbatim
    used to fill the form with a combination ``Recipe.from_dict`` refuses by
    construction the moment Generate is pressed (settings_character-01, the
    2026-09-13 audit), with nothing on screen pointing at the species control
    that is actually the fix: the prompt bar showed Knight, no ring anywhere,
    and the refusal landed on ``field="theme"`` for a control this pane may
    not even be drawing (``theme_options`` hides the combo for a
    single-look species). Reset to :data:`THEME_UNSET`, the same as a prompt
    naming no look at all -- the first cut of this fix left the field exactly
    as it was on the theory that "said nothing about a look" and "named a look
    this species cannot paint" need telling apart, but that just moved the
    accumulation bug one prompt later (settings_character-02, the same audit
    day): "a fire ogre" then "a swamp knight" left ``character_theme="fire"``,
    which ``recipe_kwargs`` sends straight to the door and the Knight refuses
    it too, on a theme the second prompt never even named. There is nowhere
    else to park "swamp" that is not this field, so :data:`THEME_UNSET` is the
    only honest value it can hold; ``resolution.theme`` itself still says
    "swamp" for anything that wants to read the brief rather than the form.
    """
    from ....state import default_form_2d

    overrides = set(overrides_of(form))
    defaults = default_form_2d()
    actions = tuple(a for a in resolution.actions if a in dict(MOVEMENTS))
    values = {
        # Never a substitution: ``resolution.family`` is None for a creature
        # this program does not make, and "" is what that means here.
        "character_family": resolution.family or "",
        "character_camera": resolution.camera_preset or "",
        "character_actions": (
            ",".join(actions) if actions else str(defaults["character_actions"])
        ),
    }
    # The species this frame is actually landing on: the form's own value when
    # ``character_family`` is overridden (the prompt cannot move it), the
    # prompt's answer otherwise.
    previous_family = str(form.get("character_family") or "")
    resolved_family = (
        previous_family if "character_family" in overrides else values["character_family"]
    )
    if resolution.theme is None:
        values["character_theme"] = THEME_UNSET
    elif _theme_offered_by(resolved_family, resolution.theme):
        values["character_theme"] = resolution.theme
    else:
        # The prompt named a look this species does not paint. Not left as it
        # was: that let a *previous* prompt's theme survive a species change
        # it was never resolved against -- see the docstring above.
        values["character_theme"] = THEME_UNSET
    for key, value in values.items():
        if key not in overrides:
            form[key] = value
    # The sliders belong to the species, so a change of species drops them --
    # a quadruped has no ``tusk`` channel, and an appearance block carrying one
    # is a request ``Recipe.from_dict`` refuses by name. Unconditional on an
    # actual species change, the same as the manual combo in
    # ``settings_character._family``: the 2026-09-07 audit (troupe-02) found a
    # slider touched under one species (an ogre's ``bulk: 1.0``) survive onto a
    # different one named by a later prompt edit, because being in
    # ``character_overrides`` was enough to skip this reset even though the
    # species underneath it had just changed -- contradicting this function's
    # own claim about accumulating a character out of two briefs.
    if resolved_family != previous_family or "character_body" not in overrides:
        form["character_body"] = "{}"


# --- reading the form ---------------------------------------------------------


def actions_of(form: dict[str, Any]) -> tuple[str, ...]:
    """The movements this sheet carries, in :data:`MOVEMENTS` order.

    The stored order is not trusted: two forms that named the same movements in
    different orders would otherwise plan two different cell layouts, which is
    ``resolve._ACTION_ORDER``'s rule applied at the other end of the same trip.
    """
    stored = {
        part.strip()
        for part in str(form.get("character_actions") or "").split(",")
        if part.strip()
    }
    return tuple(name for name, _frames in MOVEMENTS if name in stored)


def animations_of(form: dict[str, Any]) -> dict[str, int]:
    frames = dict(MOVEMENTS)
    return {name: frames[name] for name in actions_of(form)}


def cell_count(form: dict[str, Any]) -> int:
    return sum(animations_of(form).values()) * DIRECTIONS


def family_of(form: dict[str, Any], opts: dict[str, Any]) -> dict[str, Any] | None:
    """The chosen species' row, or None when the form names none we ship."""
    key = str(form.get("character_family") or "")
    return next((row for row in opts["families"] if row["key"] == key), None)


def body_of(form: dict[str, Any], opts: dict[str, Any]) -> dict[str, float]:
    """The appearance block, filtered to the chosen species' own channels.

    Filtered on the way *out* rather than cleared on the way in, so a form
    restored with a slider from another body plan is simply not sent -- the
    door refuses an unknown channel by name, and a request refused over a
    control that is no longer on screen is the dead end the pane's whole
    override model exists to avoid.
    """
    try:
        stored = json.loads(str(form.get("character_body") or "{}"))
    except (ValueError, TypeError):
        stored = {}
    if not isinstance(stored, dict):
        return {}
    channels = {c["key"] for c in channels_of(form, opts)}
    out: dict[str, float] = {}
    for key, value in stored.items():
        if str(key) in channels:
            try:
                out[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
    return out


def channels_of(form: dict[str, Any], opts: dict[str, Any]) -> list[dict[str, Any]]:
    """The sliders the chosen species declares, with *its* defaults on them."""
    key = str(form.get("character_family") or "")
    return list(opts["channels"].get(key) or ())


def set_channel(form: dict[str, Any], key: str, value: float) -> None:
    body = {}
    try:
        raw = json.loads(str(form.get("character_body") or "{}"))
        if isinstance(raw, dict):
            body = {str(k): v for k, v in raw.items()}
    except (ValueError, TypeError):
        body = {}
    body[key] = float(value)
    form["character_body"] = json.dumps(body, sort_keys=True)


def camera_of(form: dict[str, Any], opts: dict[str, Any]) -> str:
    """The preset this form will be rendered at. Empty means the door's own."""
    presets = opts["troupe"]["camera_presets"]
    chosen = str(form.get("character_camera") or "")
    if chosen in presets:
        return chosen
    return str(opts["troupe"]["defaults"]["camera"])


# --- the options each picker offers -------------------------------------------


def family_options(opts: dict[str, Any], current: str) -> tuple[tuple[str, str], ...]:
    """Every species, grouped by body plan. **A real picker, not a label.**

    ``settings_2d._locked_sheet_recipe``'s rule is that a choice with one
    answer is drawn as a statement rather than as a combo nobody can operate.
    Thirty-one species across four body plans is the opposite situation, so
    this is a combo -- and it is grouped, because a flat alphabetical list of
    thirty-one nouns is a list nobody can find a wolf in. The archetype is
    carried in the label rather than as a header row: ``controls.combo`` draws
    one selectable per entry and a header would be a row that answers a click
    by doing nothing.

    A form naming no species at all keeps a first entry saying so -- the combo
    falls back to entry zero for a value it cannot find, so without it the
    picker would draw "Human" over a form that is refusing to submit.
    """
    labels = {row["key"]: row["label"] for row in opts["archetypes"]}
    out: list[tuple[str, str]] = []
    if not current:
        out.append(("", "Not chosen yet"))
    for archetype in labels:
        for row in opts["families"]:
            if row["archetype"] == archetype:
                out.append((row["key"], f"{labels[archetype]}: {row['label']}"))
    if current and current not in {key for key, _label in out}:
        # ``settings_2d.palette_options``' rule: a stored value the menu does
        # not carry is listed and marked rather than dropped, because dropping
        # it makes the one thing keeping Generate off the one thing not on
        # screen.
        out.append((current, f"{current} - not a species this build ships"))
    return tuple(out)


def theme_options(opts: dict[str, Any], family: str) -> tuple[tuple[str, str], ...]:
    """The looks this species paints, plus "its own"."""
    row = next((f for f in opts["families"] if f["key"] == family), None)
    themes = tuple((t["key"], t["label"]) for t in (row or {}).get("themes", ()))
    return ((THEME_UNSET, "The species' own"), *themes)


def camera_options(opts: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    presets = opts["troupe"]["camera_presets"]
    return tuple((key, str(entry["label"])) for key, entry in presets.items())


def camera_helper(opts: dict[str, Any], camera: str) -> str:
    """The angle and the direction count, in the numbers that transfer.

    ``poser.ui.panes.sheet._camera_helper``'s argument: a user matching these
    sprites to a Plotter map knows what elevation that map is drawn at, and a
    preset's name does not answer that while its angle does.
    """
    entry = opts["troupe"]["camera_presets"].get(camera) or {}
    if "elevation" not in entry:
        return ""
    return f"{float(entry['elevation']):g} degrees elevation, {DIRECTIONS} directions"


# --- what would be refused ----------------------------------------------------


def problems(ctx: Any, form: dict[str, Any]) -> list[problem_types.Problem]:
    """Everything stopping a character press, each naming its own control.

    Appended by ``settings_2d.problems_for``, which is what makes the ring, the
    plan footer, the disabled Generate's tooltip and the Ctrl+Enter toast one
    sentence rather than four -- the property that module's cache exists for.
    """
    # Before anything is judged, because the *bar* asks this question first.
    # ``main._build_ui`` draws the command bar above this column, so a sync that
    # happened only in ``settings_character.draw_block`` would leave the
    # frame's cached verdict -- the ring, the disabled Generate and the footer
    # all read it -- describing the previous prompt while the block below
    # showed the new species. Cheap: the scan runs only when the prompt has
    # actually changed.
    sync_from_prompt(form)
    opts = options(ctx)
    out: list[problem_types.Problem] = []
    resolution = resolution_of(form)
    family = str(form.get("character_family") or "")
    if not family:
        out.append(problem_types.Problem(_no_species(resolution, opts), "prompt"))
    elif family_of(form, opts) is None:
        out.append(
            problem_types.Problem(
                f"{family!r} is not a species this build ships. Pick one from "
                f"Species.",
                "character_family",
            )
        )
    if not actions_of(form):
        out.append(
            problem_types.Problem(
                "A character sheet is at least one movement. Turn on idle, "
                "walk or attack.",
                "character_actions",
            )
        )
    if not getattr(ctx, "rigging_available", False):
        # The Rig segment's own sentence, verbatim -- and ``create_character``
        # raises the identical one at the door. One wording for "this needs
        # Blender" wherever it is met.
        #
        # No ``field``, deliberately: this is a fact about the *install*, not
        # about a control, and ``note_field_error`` refuses an empty address
        # precisely so a machine-shaped refusal reaches the toast and the plan
        # block instead of ringing an arbitrary widget.
        out.append(problem_types.Problem("Rigging needs Blender, which is not installed."))
    # Reachable from a *restored* form rather than from this frame's controls:
    # both values are persisted and the ladders can move between releases, so a
    # segmented control offering three sizes is not on its own a gate.
    sizes = list(opts["troupe"]["logical_sizes"])
    if _int(form.get("character_pixel"), -1) not in sizes:
        out.append(
            problem_types.Problem(f"Sprite size must be one of {sizes}.", "character_pixel")
        )
    colors = list(opts["troupe"]["colors"])
    if _int(form.get("character_colors"), -1) not in colors:
        out.append(
            problem_types.Problem(f"Colours must be one of {colors}.", "character_colors")
        )
    return out


def _no_species(resolution: resolve_mod.Resolution, opts: dict[str, Any]) -> str:
    """Why this brief names nothing we can build. **One home for the wording.**

    The creature case is :func:`resolve.offer_sentence` verbatim and is never
    re-worded here: that function exists precisely because the Create surface, a
    tooltip and a toast all say it, and a second copy is the one that eventually
    gets written as though the substitution had already happened.
    """
    offer = resolve_mod.offer_sentence(resolution)
    if offer is not None:
        return offer
    count = len(opts["families"])
    if resolution.creature_words:
        # A creature word we know and have nothing at all to offer for. Not
        # reachable from today's registry -- every body plan ships species --
        # and still said in the same register rather than left to fall through
        # to the "names none of them" sentence, which would be untrue.
        return (
            f"Warlock has no {resolution.creature_words[0]} yet, and nothing "
            f"close enough to offer. Pick one of its {count} species from "
            f"Species."
        )
    return (
        f"Warlock builds {count} species across four body plans, and this brief "
        f"names none of them. Say what to make, or pick one from Species."
    )


# --- what is submitted --------------------------------------------------------


def recipe_kwargs(form: dict[str, Any], opts: dict[str, Any]) -> dict[str, Any]:
    """The form as ``characters.recipe.Recipe.from_dict`` takes it.

    ``settings_2d.submit_kwargs``' opposite number on this arm, and split out
    for that function's reason: the compilation of the request is the one part
    of a press no test can reach while it lives inside a closure.

    ``elevation`` is sent as the *number* rather than left to the recipe's
    default, which is ``poser.ui.panes.sheet``'s arrangement and its argument
    (Troupe's own ``troupe_settings``, before P9 2026-09-18): a preset is
    only a name for an angle, and a recipe carrying a camera whose elevation
    is somebody else's default would be framed at an angle nobody picked.
    Both come from ``service.troupe.troupe_options``, so the pane holds no
    second copy of the table (``tests/modes/poser/test_camera_presets.py``).
    """
    camera = camera_of(form, opts)
    presets = opts["troupe"]["camera_presets"]
    kwargs: dict[str, Any] = {
        "family": str(form.get("character_family") or ""),
        "camera": camera,
        "elevation": float(presets[camera]["elevation"]),
        "animations": animations_of(form),
        "directions": DIRECTIONS,
        "logical_size": _int(form.get("character_pixel"), 64),
        "colors": _int(form.get("character_colors"), 32),
        "appearance": body_of(form, opts),
        "seed": max(0, _int(form.get("seed"), 0)),
    }
    theme = str(form.get("character_theme") or THEME_UNSET)
    if theme != THEME_UNSET:
        # Omitted rather than sent as the sentinel: absent is what
        # ``Recipe.from_dict`` reads as "this species' own first look", and a
        # literal "none" is a theme key it would refuse by name.
        kwargs["theme"] = theme
    name = str(form.get("character_name") or "").strip()
    if name:
        kwargs["name"] = name
    return kwargs


def toast_for(form: dict[str, Any], opts: dict[str, Any]) -> str:
    """What the app says once the press lands. Named species, counted cells."""
    row = family_of(form, opts)
    label = str((row or {}).get("label") or "character").lower()
    return (
        f"Building the {label}: mesh -> rig -> {cell_count(form)}-cell sheet. "
        f"Watch it here, then in Poser."
    )


def preview(ctx: Any, form: dict[str, Any]) -> bool:
    """Ask the door for a look at the body. Draws nothing -- the busy check and
    the button are ``settings_character._footer``'s.
    """
    opts = options(ctx)
    kwargs = recipe_kwargs(form, opts)
    return bool(ctx.submit(PREVIEW_KEY, svc_characters.preview_character, ctx.svc, kwargs))


# --- the block's non-drawing state ---------------------------------------------


def mirror_errors(ctx: Any) -> None:
    """Re-file a door refusal under the name of the control it is about.

    ``Recipe.from_dict`` refuses in the recipe's vocabulary -- ``logical_size``,
    ``appearance``, ``animations`` -- because that is what it validates, and
    ``service.errors.invalid_from`` passes that address straight through to
    ``main._collect_tasks``. The controls here are named for the form keys they
    persist under, so without this the address is recorded and thrown away: the
    ring lands nowhere and the user gets a red toast in the corner with no idea
    which control was at fault. That is the exact defect
    ``tests/test_field_error_wiring.py`` was written for.

    Run from ``settings_2d.draw`` **before** the ``forms.Form`` is built,
    because that class snapshots the error map at construction -- a mirror
    written afterwards would not be seen until the next frame.
    """
    errors = getattr(ctx.state, "field_errors", None)
    if not errors:
        return
    for control, aliases in RECIPE_FIELDS.items():
        if control in errors:
            continue
        for alias in aliases:
            message = str(errors.get(alias) or "")
            if message:
                errors[control] = message
                break


# --- the repairs offered under a refusal --------------------------------------


def apply_offer(form: dict[str, Any], opts: dict[str, Any]) -> str:
    """Take the offered species. -> the key applied, or ``""``.

    **The only substitution in the program**, and it is here rather than in
    :func:`sync_from_prompt` or in ``resolve`` because it is a thing a person
    does: they read "Warlock has no phoenix yet; the closest it makes is a
    dragon" and pressed the button that repeats it (drawn by
    ``settings_character._offer_fixes``). Recorded as an override for the same
    reason -- it is the user's choice now, and the next prompt edit must not
    quietly take it back.
    """
    resolution = resolution_of(form)
    offer = resolution.offer[0] if resolution.offer else ""
    if not offer or not any(f["key"] == offer for f in opts["families"]):
        return ""
    form["character_family"] = offer
    touched(form, "character_family")
    # The sliders and the look belonged to whatever the form said before.
    form["character_body"] = "{}"
    if not theme_offered(opts, offer, str(form.get("character_theme") or "")):
        form["character_theme"] = THEME_UNSET
    return offer


def switch_to_sprite_sheet(form: dict[str, Any]) -> None:
    """The other deliverable for the same brief. **The prompt is untouched.**

    SDXL draws what the registry does not model, which is the whole reason this
    is an escape route rather than a consolation prize -- and a route that
    rewrote the brief on the way would send a different request than the one
    the user was refused for.
    """
    form["asset_type"] = "sprite_sheet"
    form["generation_type"] = "sprite_sheet"
    create_assets.sync_legacy_fields(form)


def _int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return fallback
