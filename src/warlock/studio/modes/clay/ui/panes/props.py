"""The selected object: its transform, its generator's parameters, its material.

**The parameter widgets are generated from the registry, not written by hand.**
``primitives.GENERATORS`` maps a name to ``(defaults, builder)`` and every
default dictionary is a complete call, so the panel enumerates it and binds one
widget per key. A seventh primitive therefore needs no edit here at all -- which
is the entire reason that registry is data rather than a chain of ``if``s, and
is asserted by a test that registers a fake generator and looks for its
parameters.

A change to a parameter regenerates the mesh as **one** ``MeshEdit`` plus the
props edit that recorded the new parameters, so a Ctrl+Z takes the object back
to the shape it had. That only works while ``generator`` is not None; the first
topology edit clears it and this panel switches to a vertex and face count with
a "frozen" note -- which is the state an *imported* object arrives in too.

Every control here is disabled while a save is in flight, for the reason the
tool panel states.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from imgui_bundle import imgui

from ......kernels.mesh import primitives as bp
from ......kernels.mesh import regen
from ..... import controls, icons, theme, tokens, widgets
from .....manual import render as manual_render
from .....tokens import sp
from ... import matlib as clay_matlib
from ... import mode as clay_mode

log = logging.getLogger(__name__)

# How a parameter's type decides its widget. Read off the *default value*,
# because a registry entry carries no schema and does not need one: a float
# default means a float field, and a tuple means one field per component.
_STEP = {"segments": 1, "rings": 1, "sides": 1}


def draw(ctx: Any) -> None:
    """This pane's headings, on tinted blocks.

    The blocks are opened *here* rather than in :func:`layout.pane`, which is
    flat: a pane on a wide canvas wants no tint, and this is one of the four
    narrow sidebars the grouping was written for (see
    ``tests/test_section_blocks.py`` for the report it came from). Wrapping
    ``_body`` rather than inlining the ``with`` keeps every early return inside
    the scope, and the scope closes its last block on the way out.
    """
    with widgets.section_blocks():
        _body(ctx)


def _body(ctx: Any) -> None:
    state = clay_mode.ensure(ctx)
    tab = state.active
    widgets.section("Properties")
    manual_render.help_button(ctx, "clay-props")
    if tab is None:
        # The heading and nothing else; see ``clay_outliner``.
        return
    doc = tab.doc
    _element_summary(doc)
    obj = _selected(doc)
    if obj is None:
        # Two sentences, because ``_selected`` returns None for two different
        # reasons and one of them used to lie: with sixteen objects lit up the
        # pane said "Nothing selected", which the viewport plainly contradicts
        # -- and a click that drops a whole multi-object figure makes that the
        # ordinary case rather than the odd one. The *refusal* is unchanged
        # (see ``_selected``); only the sentence the user reads is.
        count = len(doc.selection)
        if count > 1:
            widgets.empty_state(
                icons.BOX,
                f"{count} objects selected",
                "Select one to edit it.",
            )
        else:
            widgets.empty_state(icons.BOX, "Nothing selected", "Click an object in the viewport.")
        return

    imgui.begin_disabled(tab.saving)
    _identity(doc, obj)
    imgui.dummy((0, sp(tokens.SP_2)))
    _relations(ctx, doc, obj)
    imgui.dummy((0, sp(tokens.SP_2)))
    _tags(doc, obj)
    imgui.dummy((0, sp(tokens.SP_2)))
    _transform(doc, obj)
    imgui.dummy((0, sp(tokens.SP_2)))
    _generator(doc, obj)
    imgui.dummy((0, sp(tokens.SP_2)))
    _modifiers(ctx, doc, obj)
    imgui.dummy((0, sp(tokens.SP_2)))
    _diagnostics(state, doc, obj)
    imgui.dummy((0, sp(tokens.SP_2)))
    _material(ctx, tab, doc, obj)
    imgui.end_disabled()


def _element_summary(doc: Any) -> None:
    """One line saying what is selected inside the objects, in element modes.

    The object panel below stays exactly as it was -- an element selection is
    still an object selection, by the document's own invariant -- so this adds
    a line rather than replacing the pane. It is also where the *frozen* branch
    of the generator section finally becomes reachable: an op that edits
    topology clears ``generator``, and this is usually the first thing the user
    sees afterwards.
    """
    if doc.element_mode == "object":
        return
    total = sum(sel.count(doc.element_mode) for sel in doc.element_sel.values())
    noun = {"vertex": "vertices", "edge": "edges", "face": "faces"}[doc.element_mode]
    objects = len(doc.element_sel)
    if total == 0:
        widgets.muted(f"{doc.element_mode} mode -- nothing selected")
    else:
        across = "1 object" if objects == 1 else f"{objects} objects"
        widgets.muted(f"{doc.element_mode} mode -- {total} {noun} across {across}")
    imgui.dummy((0, sp(tokens.SP_1)))


def _selected(doc: Any) -> Any:
    """The one selected object, or None.

    One rather than the first of many: a properties panel that silently edited
    whichever object happened to sort first under a multi-selection is worse
    than one that says it cannot.
    """
    if len(doc.selection) != 1:
        return None
    try:
        return doc.by_uid(next(iter(doc.selection)))
    except KeyError:
        return None


def _identity(doc: Any, obj: Any) -> None:
    # commit=True: the 2026-09-06 audit's clay-02 found this field reporting a
    # change on every keystroke, so ``set_props`` -- an unconditional
    # ``history.push`` -- fired once per letter typed and a lone Ctrl+Z after
    # a rename undid one character instead of the whole name.
    name = widgets.input_text("name##buildname", obj.name, max_length=120, commit=True)
    if name != obj.name:
        doc.set_props(obj.uid, name=name)
    changed, value = widgets.toggle(f"{icons.EYE} Visible", obj.visible, tag=str(obj.uid))
    if changed:
        doc.set_props(obj.uid, visible=value)
    # Tranche 3: scene structure. Not a locking door itself
    # (``document.py``'s locking paragraph): toggling the lock is always
    # allowed, the same as visibility, or a mistake made while locked could
    # never be undone by anyone but the lock.
    changed, value = widgets.toggle(f"{icons.LOCK} Locked", obj.locked, tag=f"lock{obj.uid}")
    if changed:
        doc.set_props(obj.uid, locked=value)


def _set_parent(ctx: Any, doc: Any, uid: int, parent: int | None) -> None:
    """``doc.set_parent(..., keep_world=True)``, refused as a toast.

    ``_relations``'s combo already excludes every uid that would make
    ``set_parent`` refuse a cycle, so this is defensive rather than the
    expected path -- the same shape :func:`_set_modifier_stack` gives
    ``set_modifiers``' own refusal, through the same :func:`~.clay.ops.toast`
    door.
    """
    from ......kernels.mesh.elements import OpError
    from ... import ops as clay_ops

    try:
        doc.set_parent(uid, parent, keep_world=True)
    except OpError as error:
        clay_ops.toast(ctx, str(error))


def _relations(ctx: Any, doc: Any, obj: Any) -> None:
    """Parent, and a way out of it (tranche 3: scene structure).

    The combo excludes every one of *obj*'s own descendants
    (``ClayDoc.descendants``) -- offering one would let a click ask
    ``set_parent`` for a cycle it refuses anyway, and a control that visibly
    offers a choice it is about to refuse is worse than one that never shows
    it. ``keep_world=True`` throughout: reparenting through this panel never
    visibly moves the object, only which frame the "Local" transform fields
    below are read in.
    """
    widgets.field_label("relations")
    descendants = set(doc.descendants(obj.uid))
    options = [("0", "(none)")] + [
        (str(other.uid), other.name or f"object {other.uid}")
        for other in doc.objects
        if other.uid != obj.uid and other.uid not in descendants
    ]
    current = "0" if obj.parent is None else str(obj.parent)
    picked = widgets.labeled_combo(
        "parent",
        current,
        options,
        help_text="Reparenting keeps this object's world position -- only "
        "the local numbers below, and which frame they are read in, change.",
    )
    if widgets.disabled_button("Clear parent##clearparent", obj.parent is not None):
        picked = "0"
    if picked != current:
        _set_parent(ctx, doc, obj.uid, None if picked == "0" else int(picked))


def _tags(doc: Any, obj: Any) -> None:
    """Free-form membership tags (tranche 3: scene structure).

    The one membership concept a group or a collection would otherwise have
    been (``document.py``'s module docstring): the outliner's tag filter and
    ``clay_select_by``'s tag query both read ``Obj.tags``. Comma-separated
    entry is deliberately the whole of "add" -- typing "prop, background" and
    leaving the field is one edit, and ``ClayDoc.set_props``'s own
    ``_normalize_tags`` sorts, dedupes and lower-cases whatever comes out of
    it, so this widget does not have to.
    """
    widgets.field_label("tags")
    if not obj.tags:
        widgets.muted("no tags")
    removed: str | None = None
    for tag in obj.tags:
        if controls.small_button(f"{tag}  {icons.X}##tag-{tag}", tooltip=f"Remove {tag!r}"):
            removed = tag
        imgui.same_line()
    if obj.tags:
        imgui.new_line()
    if removed is not None:
        doc.set_props(obj.uid, tags=tuple(t for t in obj.tags if t != removed))
    # ``value=""`` every frame, the same sentinel ``_add_modifier_row``'s
    # combo uses: this is an action ("add these tags"), not a persistent
    # field, so the box reads as empty again the moment the add lands.
    added = widgets.input_text(
        "##tagadd", "", max_length=120, hint="add tags, comma-separated...", commit=True
    )
    if added.strip():
        doc.set_props(obj.uid, tags=tuple(obj.tags) + tuple(added.split(",")))


def _transform(doc: Any, obj: Any) -> None:
    # Tranche 3: scene structure. TRS is local to the parent now, and a root's
    # local TRS *is* its world TRS (``document.py``'s module docstring) --
    # which is what keeps an unparented object's fields, and their labels,
    # exactly as they always were. "Local" on the section heading and on
    # every field beneath it: the numbers say where the object sits relative
    # to its parent, not where it sits in the viewport.
    #
    # Each field below is two literal input_vec calls (one per parent state)
    # rather than one f-string-built label: ``test_ux_consistency_pass4.py``'s
    # allow-list is a static source scan for a literal string argument, and
    # an f-string is invisible to it -- it would not flag anything, but it
    # would also silently drop this row off the label-above-control
    # inventory instead of keeping it an accounted-for, visible exception
    # (see that file's own clay_props.py entries).
    # Same id suffix (``##bt``/``##bs``/``##br``) either way, so reparenting
    # mid-edit does not reset the field's own imgui state.
    parented = obj.parent is not None
    widgets.field_label("local transform" if parented else "transform")
    was = tuple(v.copy() for v in obj.trs())
    changed = False
    if parented:
        edited, translation = controls.input_vec(
            "local position##bt", list(obj.translation), ("X", "Y", "Z")
        )
    else:
        edited, translation = controls.input_vec(
            "position##bt", list(obj.translation), ("X", "Y", "Z")
        )
    # The 2026-09-07 audit's clay-01: these three fields fired ``set_transform``
    # -- an unconditional ``history.push`` -- on every keystroke, same as the
    # material sliders below already fold. ``InputFloat3``/``InputFloat4`` fire
    # per keystroke like any imgui text field, so typing a multi-digit number
    # into Position pushed one undo step per digit and a lone Ctrl+Z only took
    # the last character back rather than the whole edit.
    controls.fold_undo(doc.history)
    changed |= edited
    if parented:
        edited, scale = controls.input_vec(
            "local scale##bs", list(obj.scale), ("X", "Y", "Z")
        )
    else:
        edited, scale = controls.input_vec("scale##bs", list(obj.scale), ("X", "Y", "Z"))
    controls.fold_undo(doc.history)
    changed |= edited
    # Rotation stays a quaternion (the 2026-09-12 consistency pass's call --
    # the viewer gizmo and the pose files are both XYZW, and converting the
    # field to Euler would need the panel to pick a rotation order the rest
    # of the app doesn't have) -- so its fourth letter is W, not a repeated Z.
    if parented:
        edited, rotation = controls.input_vec(
            "local rotation##br", list(obj.rotation), ("X", "Y", "Z", "W")
        )
    else:
        edited, rotation = controls.input_vec(
            "rotation##br", list(obj.rotation), ("X", "Y", "Z", "W")
        )
    controls.fold_undo(doc.history)
    widgets.help_marker(
        "A quaternion, XYZW -- the same order the viewer and the pose files "
        "use. Typing one is for a value you already have; the gizmo is the "
        "way to set one by eye."
    )
    changed |= edited
    _dimensions(doc, obj)
    if changed:
        # ``was`` is the values the fields started from. imgui writes the new
        # ones into the widget's own state as they are typed, so reading
        # "before" off the object here would compare a value against itself and
        # record an empty step -- which is the trap ``set_transform``'s ``was``
        # argument exists for.
        doc.set_transform(
            obj.uid,
            translation=translation,
            rotation=rotation,
            scale=scale,
            was=was,
        )


def _dimensions(doc: Any, obj: Any) -> None:
    """How big the thing actually is, in metres of world space.

    Read-only, and it is the number the panel was missing: a scale of 2 on a
    generator whose radius is 0.35 says nothing about how large the object is,
    in an app whose whole pipeline is denominated in ``size_m``. W x D x H
    rather than X/Y/Z because that is how a physical object is quoted, and
    ``ops.world_box``'s answer rather than a second measurement here, so the
    row and the camera's framing cannot disagree about one object.

    **Measured off the evaluated mesh**, not the base -- a solidify or an
    array modifier changes what is actually on screen, and a size row that
    kept reporting the base's box would disagree with the object the camera
    just framed. :func:`~.ops.world_box`'s own ``mesh`` override is what makes
    this a one-line change rather than a second measurement path: an object
    with no enabled modifiers evaluates to its own base mesh (``is``-identical,
    :mod:`~.modifiers`'s own docstring), so nothing here changes for the
    common case.

    ``world=doc.world_matrix(obj.uid)`` (tranche 3: scene structure): left at
    ``world_box``'s own default, this composed only *obj*'s own TRS, which for
    a parented object is local to its parent and not its world placement --
    reporting the size of a *root sitting where this object's parent happens
    to be*, not the size of the object where it actually sits. A root's world
    matrix is exactly its own local TRS, so this changes nothing for a
    document with no parenting.
    """
    from ......kernels.mesh import ops as bops

    box = bops.world_box(obj, doc.evaluated(obj.uid), world=doc.world_matrix(obj.uid))
    if box is None:
        return
    w, h, d = (float(v) for v in (box[1] - box[0]))
    widgets.muted(f"size  {w:.3f} x {d:.3f} x {h:.3f} m  (W x D x H)")
    widgets.help_marker(
        "The object's world-space bounding box, after its transform. A rotated "
        "object reports the box around its rotated box, which is the same "
        "measurement the camera frames against."
    )


def _generator(doc: Any, obj: Any) -> None:
    if obj.generator is None:
        # A frozen object: edited topology, or imported. The panel says what
        # the object is rather than pretending it still has parameters that
        # would silently discard the edit if changed.
        widgets.field_label("mesh")
        line = (
            f"frozen -- {len(obj.mesh.positions)} vertices, "
            f"{len(obj.mesh.starts) - 1} faces"
        )
        if obj.modifiers:
            # The base counts alone would describe geometry nobody on screen
            # is looking at once a stack is on top of it -- the modifiers
            # section below shows the stack itself, this line only adds the
            # one number it does not: what the base becomes once it runs.
            line += " (base)"
            evaluated = doc.evaluated(obj.uid)
            line += (
                f"; evaluated -- {len(evaluated.positions)} vertices, "
                f"{len(evaluated.starts) - 1} faces"
            )
        widgets.muted(line)
        return
    entry = bp.GENERATORS.get(obj.generator)
    if entry is None:
        widgets.muted(f"unknown generator '{obj.generator}'")
        return

    defaults, build = entry
    widgets.field_label(obj.generator.replace("_", " "))
    params = dict(defaults)
    params.update({k: v for k, v in obj.params.items() if k in defaults})
    edited = dict(params)
    changed = False
    for key, default in defaults.items():
        # A name line per param (2026-09-08 consistency pass): the block
        # label above names the generator, not its individual fields, and
        # the old beside-the-box text was the only place a param's name
        # appeared.
        widgets.field_label(key.replace("_", " "))
        was, changed_here = _widget(key, params.get(key, default), default)
        # The 2026-09-07 audit's clay-01: a generator field fired
        # ``set_generator_params`` -- also an unconditional ``history.push`` --
        # per keystroke, for the same reason the transform fields above needed
        # ``fold_undo``. Folded here, before ``set_generator_params`` runs
        # below, so the invariant every other door in this file follows
        # (draw, fold, act) holds for every field in the loop, not only the
        # one the user happened to stop typing in.
        controls.fold_undo(doc.history)
        if changed_here:
            edited[key] = was
            changed = True
    if not changed:
        return
    # Match what the generator will actually build *before* building it: the
    # 2026-09-06 audit's clay-05 finding was that a segment count of zero (or
    # a torus tube wider than its radius, clay-04) gets clamped inside the
    # generator without being reported back, so this panel used to save the
    # number the user typed rather than the one the mesh was built from.
    edited = bp.clamp_params(obj.generator, edited)
    try:
        mesh = build(**edited)
    except Exception:  # noqa: BLE001
        # A generator raises on a value it cannot build at all -- not the
        # zero segment count or oversized torus tube this comment used to
        # name (both are clamped, by clamp_params above and by the generator
        # itself; see the 2026-09-06 audit's clay-04 and clay-05 findings),
        # but a non-finite number that survives to ``int()``, such as a
        # pasted value large enough to parse as infinity. The old mesh stays;
        # the field keeps the number the user typed, so they can correct it.
        #
        # Logged, not merely swallowed. A refusal about a number and a
        # ``TypeError`` from a renamed keyword are the same silence here, and
        # the second is a defect that would look exactly like a slider that
        # stopped working. Every comparable site in the tree logs; this one is
        # on the frame thread and per-keystroke, so it must not toast.
        log.debug("generator %r refused %r", getattr(build, "__name__", build), edited,
                  exc_info=True)
        return
    # ``regen.carry_over`` is what keeps a rebuild from silently discarding a
    # hand-picked Shade Smooth/Flat or a hand-painted per-face material the
    # moment any generator field is touched -- see that module's own
    # docstring for the two-case rule and why it now lives there rather than
    # here (``agent_clay._h_set_params`` is the other door that rebuilds a
    # generator's mesh, and it used to carry neither attribute at all, so the
    # rule had to move somewhere both could reach rather than staying a
    # method on this pane).
    mesh = regen.carry_over(obj.mesh, mesh, material=obj.material)
    # One step, not two. The numbers and the mesh they build are one act, and
    # a lone Ctrl+Z restoring half the pair showed the old mesh in the viewport
    # while this panel still read the new radius -- and ``InputFloat`` fires
    # per keystroke, so typing a multi-digit number made several of them.
    # ``set_generator_params`` also implies ``keep_generator``: this mesh is
    # precisely what the generator builds from the edited parameters, which is
    # the one case where the object's generator claim is still true.
    doc.set_generator_params(obj.uid, edited, mesh, was={"params": params})


def _widget(key: str, value: Any, default: Any) -> tuple[Any, bool]:
    """One widget for one parameter, chosen from the default's type.

    The visible name moved to the ``field_label`` line the loop above draws
    (2026-09-08 consistency pass); hidden here the same way a previously
    visible control is hidden elsewhere in this pass -- "Foo" becomes
    "##Foo", not a new id -- so the id stays recognisably the old one.
    """
    label = f"##{key.replace('_', ' ')}##gen{key}"
    if isinstance(default, bool):
        return controls.checkbox(label, bool(value))[::-1]
    if isinstance(default, int):
        changed, out = controls.input_int(label, int(value), _STEP.get(key, 1))
        return out, changed
    if isinstance(default, float):
        changed, out = controls.input_float(label, float(value), 0.05)
        return out, changed
    # A *flat* sequence of 2 or 3 real numbers -- box's ``size``, plane's and
    # grid's ``size`` -- is the whole shape this branch knows how to draw.
    # Nothing past that may take it: a length-4-or-more default used to be
    # silently cut to 3 and written back that short (a five-number lathe
    # profile's flat form, say, would lose two numbers with nothing on
    # screen to say so), and a *nested* default such as
    # ``[[0.5, -0.5], [0.5, 0.5]]`` -- what a lathe profile actually is --
    # matched ``len(default) == 2`` and handed ``input_float2`` two Python
    # lists instead of two floats, which raises on the frame thread with no
    # try/except above it. Both, and a saved value that disagrees in shape
    # with the default (a bare scalar, where ``list(value)`` used to raise
    # outright), now fall through to the read-only fallback below, same as
    # any other type nobody has built a widget for yet.
    if (
        isinstance(default, (tuple, list))
        and len(default) in (2, 3)
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in default)
        and isinstance(value, (tuple, list))
    ):
        try:
            values = [float(v) for v in value]
        except (TypeError, ValueError):
            values = None
        if values is not None:
            pad = max(0, len(default) - len(values))
            values = values[: len(default)] + [0.0] * pad
            axes = ("X", "Y") if len(default) == 2 else ("X", "Y", "Z")
            changed, out = controls.input_vec(label, values, axes)
            return tuple(float(v) for v in out), changed
    # A parameter type nobody has added yet: shown, not editable, rather than
    # silently dropped from the panel.
    widgets.secondary(f"{key}: {value!r}")
    return value, False


def modifier_kind_options() -> list[tuple[str, str]]:
    """``(kind, label)`` for every registered modifier, in registration order.

    A small pure function rather than inline in :func:`_add_modifier_row`, so
    the bidirectional registry gate -- every kind in
    ``kernels.mesh.modifiers.MODIFIERS`` reaches this pane's "Add modifier"
    combo, and nothing else does -- is an assertion against this list rather
    than a screenshot (``tests/modes/clay/test_modifier_props.py``), the same
    shape :func:`_palette_remove_reason` already gives its own gated test.
    """
    from ......kernels.mesh import modifiers as mods

    return [(kind, kind_def.label) for kind, kind_def in mods.MODIFIERS.items()]


def _set_modifier_stack(ctx: Any, doc: Any, obj: Any, stack: tuple[Any, ...]) -> None:
    """``doc.set_modifiers``, refused as a toast rather than a crash.

    ``set_modifiers`` raises :class:`~.elements.OpError` -- an unknown kind
    (unreachable from this pane, but a hand-edited ``.wblk`` can still carry
    one) or a boolean stack that would cycle back on itself -- and pushes
    nothing when it does. That is exactly the "refusal, not fatal" contract
    :mod:`.clay.ops` already gives every op in this mode through its own
    :func:`~.ops.toast`, so this reaches for the same function rather than a
    second copy of "catch OpError, show it".
    """
    from ......kernels.mesh.elements import OpError
    from ... import ops as clay_ops

    try:
        doc.set_modifiers(obj.uid, stack)
    except OpError as error:
        clay_ops.toast(ctx, str(error))


def _mod_param_widget(param: Any, value: Any, doc: Any, obj: Any) -> tuple[Any, bool]:
    """One widget for one modifier parameter. -> (new value, changed).

    Unlike the generator loop's :func:`_widget`, which infers a control from
    a default's *Python* type, a kernel-side ``ModParam`` already states its
    own kind -- ``target``/``choices``/``boolean``/``integer`` are read
    straight off it, so there is nothing here to infer and nothing that falls
    through to a read-only fallback the way an unrecognised generator default
    shape does.

    ``target`` draws a combo of every *other* object's name, keyed by uid,
    with ``"0"`` standing for "(none)" -- the one param kind
    :mod:`.modifiers` documents as an object reference (:class:`ModParam`'s
    own docstring), so this is the one branch that reads ``doc.objects``
    rather than only the value it was handed.
    """
    label = f"##{param.name}"
    if param.target:
        options = [("0", "(none)")] + [
            (str(other.uid), other.name) for other in doc.objects if other.uid != obj.uid
        ]
        current = str(int(value))
        picked = widgets.combo(label, current, options)
        return int(picked), picked != current
    if param.choices:
        options = [(str(i), choice) for i, choice in enumerate(param.choices)]
        current = str(int(value))
        picked = widgets.combo(label, current, options)
        return int(picked), picked != current
    if param.boolean:
        changed, out = controls.checkbox(label, bool(value))
        return out, changed
    if param.integer:
        changed, out = controls.input_int(label, int(value), 1)
        return out, changed
    changed, out = controls.input_float(label, float(value), param.step)
    return out, changed


def _modifier_row(
    ctx: Any, doc: Any, obj: Any, mod: Any, index: int, count: int, error: str | None
) -> None:
    """One stack entry: enabled, its label, reorder, apply, remove, its params.

    ``index``/``count`` are the row's own position rather than something read
    back off ``obj.modifiers`` inside this function, because every button
    here can itself change that tuple's length or order -- reading it live
    partway through the row would have Remove and the row below it disagree
    about which modifier is at which index for the rest of the frame.
    """
    from dataclasses import replace

    from ......kernels.mesh import modifiers as mods
    from ......kernels.mesh.elements import OpError
    from ... import ops as clay_ops

    stack = obj.modifiers
    kind_def = mods.MODIFIERS.get(mod.kind)
    label = kind_def.label if kind_def is not None else f"{mod.kind} (unknown kind)"

    imgui.push_id(f"mod{mod.id}")
    changed, enabled = controls.checkbox("##enabled", mod.enabled)
    if changed:
        _set_modifier_stack(
            ctx, doc, obj,
            tuple(replace(m, enabled=enabled) if m.id == mod.id else m for m in stack),
        )
    imgui.same_line()
    imgui.text(label)
    imgui.same_line()
    if controls.small_button(
        f"{icons.ARROW_UP}##up", enabled=index > 0, reason="Already at the top."
    ):
        new_stack = list(stack)
        new_stack[index - 1], new_stack[index] = new_stack[index], new_stack[index - 1]
        _set_modifier_stack(ctx, doc, obj, tuple(new_stack))
    imgui.same_line()
    if controls.small_button(
        f"{icons.ARROW_DOWN}##down", enabled=index < count - 1, reason="Already at the bottom."
    ):
        new_stack = list(stack)
        new_stack[index + 1], new_stack[index] = new_stack[index], new_stack[index + 1]
        _set_modifier_stack(ctx, doc, obj, tuple(new_stack))
    imgui.same_line()
    if controls.small_button(
        "Apply##apply", tooltip="Bake this modifier and everything above it into the base mesh."
    ):
        try:
            doc.apply_modifiers(obj.uid, through_id=mod.id)
        except OpError as apply_error:
            clay_ops.toast(ctx, str(apply_error))
    imgui.same_line()
    if controls.small_button(f"{icons.TRASH}##remove", tooltip="Remove this modifier."):
        _set_modifier_stack(ctx, doc, obj, tuple(m for m in stack if m.id != mod.id))

    if kind_def is not None:
        updates: dict[str, Any] = {}
        for p in kind_def.params:
            widgets.field_label(p.label)
            new_value, changed_here = _mod_param_widget(p, mod.get(p.name, p.default), doc, obj)
            # The same fold ``_generator``'s own loop uses, for the same
            # reason: a drag or a typed number reports a change on every
            # frame it is live, and without this a multi-digit edit would
            # push one ``set_modifiers`` step per digit.
            controls.fold_undo(doc.history)
            if changed_here:
                updates[p.name] = new_value
        if updates:
            new_mod = mods.with_params(mod, updates)
            _set_modifier_stack(
                ctx, doc, obj, tuple(new_mod if m.id == mod.id else m for m in stack)
            )

    if error:
        widgets.text_colored(theme.WARN, error)
    imgui.pop_id()
    widgets.divider()


def _add_modifier_row(ctx: Any, doc: Any, obj: Any) -> None:
    """"Add modifier": a combo that always reads as its own placeholder.

    ``current`` is always the sentinel ``""``, never a value read back off
    the object -- this is an action, not a persistent choice, so the moment a
    kind is picked and the stack is written, the very next frame's call
    passes ``""`` again and the combo shows "Add modifier..." once more with
    no state of its own to reset.
    """
    options = [("", "Add modifier...")] + modifier_kind_options()
    picked = widgets.labeled_combo(
        "add modifier",
        "",
        options,
        help_text="Appended to the bottom of the stack, with its default parameters.",
    )
    if picked:
        from ......kernels.mesh import modifiers as mods

        new_mod = mods.make(picked, id=mods.next_id(obj.modifiers))
        _set_modifier_stack(ctx, doc, obj, obj.modifiers + (new_mod,))


def _modifiers(ctx: Any, doc: Any, obj: Any) -> None:
    """The modifier stack: base mesh run through each enabled entry, in order.

    Reads :meth:`~.document.ClayDoc.evaluation` once per frame for the error
    map alone (empty, and no evaluation at all, on the fast path of no
    modifiers -- see :mod:`.modifiers`'s own docstring), never the mesh: this
    section is about the *recipe*, not the result, and every other reader of
    the result (the dimensions row, the viewport) asks for it on its own.
    """
    stack = obj.modifiers
    widgets.field_label("modifiers")
    errors: dict[int, str] = dict(doc.evaluation(obj.uid).errors) if stack else {}
    if not stack:
        widgets.muted("no modifiers")
    for index, mod in enumerate(stack):
        _modifier_row(ctx, doc, obj, mod, index, len(stack), errors.get(mod.id))
    _add_modifier_row(ctx, doc, obj)


def _diagnostics(state: Any, doc: Any, obj: Any) -> None:
    """What is wrong with this object's mesh, measured on request.

    **On request, never per frame.** ``check_manifold`` builds a whole
    adjacency, which is O(corners) and is exactly the sort of thing that turns
    a properties panel into a stall on an imported mesh -- so the button is the
    interface, and the answer is kept against the ``Mesh`` it was measured from.
    That comparison is by identity and it is sound for the reason the whole
    package rests on: a ``Mesh`` is immutable and every op replaces it, so a
    result about ``obj.mesh`` is a result about what is on screen.

    A row is a button because the useful thing to do with "3 non-manifold
    edges" is to look at them. Clicking sets the element mode *and* the
    selection together, since either one alone leaves the user staring at an
    overlay of the wrong kind.
    """
    from ......kernels.mesh import diagnose

    # Measured against the *base* mesh, never the evaluated one -- a finding
    # names element indices (``diagnose.findings`` selects vertices/edges/
    # faces by position in the array), and those indices only mean anything
    # against the mesh editing actually reads and writes. The label says so
    # once a stack exists, so "3 non-manifold edges" cannot be misread as a
    # statement about the shape on screen when a modifier has since changed
    # how many edges there even are.
    widgets.field_label("mesh check -- base mesh" if obj.modifiers else "mesh check")
    measured, rows = state.manifold.get(obj.uid, (None, []))
    if measured is not obj.mesh:
        if measured is not None:
            widgets.muted("edited since the last check")
        if controls.button(f"{icons.ACTIVITY} Check mesh##claycheck"):
            state.manifold[obj.uid] = (obj.mesh, diagnose.findings(obj.mesh))
        widgets.help_marker(
            "Looks for holes, non-manifold edges, inconsistently wound faces, "
            "duplicate faces and unused vertices. An open sheet is a perfectly "
            "good mesh, so these are measurements rather than a verdict -- but "
            "a game engine will usually want a closed one."
        )
        return

    if not rows:
        widgets.muted(f"{icons.CIRCLE_CHECK} closed, consistent, nothing unused")
        return
    for row in rows:
        if controls.button(f"{icons.TRIANGLE_ALERT} {row.label}##claydiag{row.kind}"):
            _select_finding(doc, obj, row)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Select them")


def _select_finding(doc: Any, obj: Any, row: Any) -> None:
    """Show one finding's elements: the mode, then only those elements.

    The object selection is not set here and must not be: in an element mode it
    is *derived*, and ``set_element_sel`` adds the object itself. Setting it by
    hand in between would be overwritten by the ``clear`` on the next line
    anyway -- the clear is what stops a finding on one object arriving beside a
    stale selection in another.
    """
    doc.set_element_mode(row.mode)
    doc.clear_element_sel()
    doc.set_element_sel(obj.uid, row.sel)


def _material(ctx: Any, tab: Any, doc: Any, obj: Any) -> None:
    widgets.field_label("material")
    if not doc.materials:
        widgets.muted("the palette is empty")
        return
    _palette_row(doc, obj)
    options = [(str(i), m.name or f"slot {i}") for i, m in enumerate(doc.materials)]
    picked = widgets.labeled_combo(
        "slot",
        str(obj.material),
        options,
        help_text=(
            "Which palette entry this object renders and exports with. Editing an "
            "entry writes a replacement, so every object using it follows."
        ),
    )
    if picked != str(obj.material):
        doc.set_props(obj.uid, material=int(picked))

    index = min(max(int(obj.material), 0), len(doc.materials) - 1)
    material = doc.materials[index]
    # Each a sub-field of the "slot" combo above, named on its own line
    # (2026-09-08 consistency pass); ids kept stable, "Foo##bm" -> "##Foo##bm".
    widgets.field_label("base colour")
    changed, colour = controls.color_edit4(
        "##base colour##bm", list(material.base_color_factor)
    )
    # One gesture, one step: a drag reports on every frame the pointer moves,
    # and ``set_material`` pushes a step per report without this.
    controls.fold_undo(doc.history)
    widgets.field_label("metallic")
    metal_changed, metallic = controls.slider_float(
        "##metallic##bm", float(material.metallic_factor), 0.0, 1.0
    )
    controls.fold_undo(doc.history)
    widgets.field_label("roughness")
    rough_changed, roughness = controls.slider_float(
        "##roughness##bm", float(material.roughness_factor), 0.0, 1.0
    )
    controls.fold_undo(doc.history)
    # Tranche 6 ("UV and materials"): the rest of a pbrMetallicRoughness
    # material -- emissive, alpha mode/cutoff, double-sided -- joining the
    # three fields above rather than a second block, so one fold covers
    # every slider in the material the same "one gesture, one step" way.
    widgets.field_label("emissive")
    emissive_changed, emissive = controls.color_edit3(
        "##emissive##bm", list(material.emissive_factor)
    )
    controls.fold_undo(doc.history)
    picked_alpha = widgets.labeled_combo(
        "alpha mode",
        material.alpha_mode,
        [("OPAQUE", "Opaque"), ("MASK", "Mask"), ("BLEND", "Blend")],
        help_text="Mask cuts by the cutoff below; blend composites by alpha.",
    )
    alpha_changed = picked_alpha != material.alpha_mode
    cutoff = material.alpha_cutoff
    cutoff_changed = False
    if picked_alpha == "MASK":
        widgets.field_label("alpha cutoff")
        cutoff_changed, cutoff = controls.slider_float(
            "##alphacutoff##bm", float(material.alpha_cutoff), 0.0, 1.0
        )
        controls.fold_undo(doc.history)
    ds_changed, double_sided = controls.checkbox(
        f"{icons.LAYERS} double-sided##bm", bool(material.double_sided)
    )
    if (
        changed or metal_changed or rough_changed or emissive_changed
        or alpha_changed or cutoff_changed or ds_changed
    ):
        # A *replacement*, never an in-place edit. Identity is what the GPU
        # cache, ``to_model`` and the writer all de-duplicate on, so editing
        # the object in place would leave every one of them showing the old
        # values with nothing in the data to say why.
        #
        # ``replace`` rather than a fresh ``Material``: the five texture slots
        # are fields on it, and building a new one from the fields the panel
        # shows would silently delete a baked map an import carried in.
        fresh = replace(
            material,
            base_color_factor=tuple(float(c) for c in colour),
            metallic_factor=float(metallic),
            roughness_factor=float(roughness),
            emissive_factor=tuple(float(c) for c in emissive),
            alpha_mode=str(picked_alpha),
            alpha_cutoff=float(cutoff),
            double_sided=bool(double_sided),
        )
        doc.set_material(index, fresh)
        material = fresh

    _texture_slots(ctx, tab, doc, index, material)
    _material_library(ctx, doc, obj)


def _palette_remove_reason(material_count: int, users: int) -> str:
    """Why "Remove" is refused for the palette's current slot, or ``""``.

    A pure function beside the draw call -- the shape ``clay_ops.reason_for``
    and ``plotter_menu._layer_reason`` already use for the same rule -- so the
    decision is testable without a live imgui frame. Split out for the
    2026-09-12 audit's finding clay-05: the single-material case fell through
    both the enabling check and the explaining one, because both were gated
    on the same ``len(doc.materials) > 1``, so a document with exactly one
    material showed Remove greyed with nothing on screen saying why.
    """
    if material_count <= 1:
        return "A document keeps at least one material."
    if users:
        # The count spans objects the undo stack still holds, not only the
        # ones in the document -- a slot removed while an undone deletion was
        # the last thing using it came back magenta on redo.
        return f"{users} face(s) use this slot (including undone deletions)"
    return ""


def _palette_row(doc: Any, obj: Any) -> None:
    """Add, rename and remove palette entries.

    Add appends -- never inserts -- because a slot *is* an index that every
    mesh's per-face ``material`` array names, and inserting one in the middle
    would renumber those arrays in every object in the document.

    Remove is offered only for an entry no face uses, and says so rather than
    reassigning those faces somewhere. Reassigning is a silent change to how
    part of the model looks, which is exactly the kind of thing a user
    discovers three edits later with no idea what did it.
    """
    index = min(max(int(obj.material), 0), len(doc.materials) - 1)
    users = doc.material_users(index)
    if controls.small_button(f"{icons.PLUS} Add##matadd"):
        # One step, not two -- the 2026-09-08 audit's clay-02: pushed as
        # ``add_material()`` then ``set_props(...)`` separately, one Ctrl+Z
        # after this click left a stray, unreferenced palette entry behind.
        doc.add_material_and_assign(obj.uid)
    imgui.same_line()
    reason = _palette_remove_reason(len(doc.materials), users)
    if widgets.disabled_button("Remove##matdel", not reason):
        # Same fold as Add, for the same reason.
        doc.remove_material_and_reassign(obj.uid, index)
    if reason:
        widgets.muted(reason)

    # commit=True for the reason ``_identity``'s name field needs it: the
    # 2026-09-06 audit's clay-02 found this one reporting per keystroke too,
    # pushing a ``set_material`` history step for every letter of a slot name.
    name = widgets.input_text(
        "slot name##matname", doc.materials[index].name or "", max_length=60, commit=True
    )
    if name != (doc.materials[index].name or ""):
        # A replacement, never an in-place edit, for the reason the colour
        # fields below state: identity is what every cache de-duplicates on.
        doc.set_material(index, replace(doc.materials[index], name=name))


#: The five texture slots ``gltf.Material`` carries, in the order this panel
#: (and ``matlib.py``'s own ``TEXTURE_SLOTS``, the same tuple by the same
#: name) offers them.
TEXTURE_SLOTS = ("base_color", "metallic_roughness", "normal", "emissive", "occlusion")

#: This pane's own task-key prefix for "assign a texture from a file" --
#: **not** ``clay-bg`` or a bare ``clay-`` key, because landing the result is
#: not a document task in ``clay_mode.on_task_done``'s sense (see
#: ``shell/tasks.py``'s own ``clay-mattex:`` branch, checked before its
#: ``clay-`` one for exactly this reason).
TEXTURE_TASK_PREFIX = "clay-mattex"


def _texture_task_key(tab_uid: str, index: int, slot: str) -> str:
    return f"{TEXTURE_TASK_PREFIX}:{tab_uid}:{index}:{slot}"


def _pick_texture(slot: str) -> dict[str, Any] | None:
    """Blocking; task thread only. -> ``{"width", "height", "rgba"}``, or
    ``None`` for a cancelled picker.

    The picker, then the decode, both off the frame thread -- Inker's
    ``ui/panes/opening.py::ask_import_sheet`` is the precedent this follows:
    a file dialog is the one place an *arbitrary* image reaches this app, so
    it is read through the same ``pixelguard`` pixel ceiling every other
    hand-picked image is, rather than trusted because it came from a picker.
    """
    from ......core.safeio import pixelguard
    from ..... import dialogs

    path = dialogs.open_file(f"Assign a {slot.replace('_', ' ')} texture", dialogs.PNG_FILTER)
    if path is None:
        return None
    pixels = pixelguard.decode_rgba(path, path.name)
    height, width = pixels.shape[:2]
    return {"width": int(width), "height": int(height), "rgba": pixels.tobytes()}


def _assign_texture(ctx: Any, tab: Any, index: int, slot: str) -> None:
    key = _texture_task_key(tab.uid, index, slot)
    if not ctx.submit(key, _pick_texture, slot):
        ctx.toast("A file dialog is already open.", "info")


def on_task_done(ctx: Any, done: Any) -> None:
    """Land a texture picked for a material slot -- ``clay-mattex:<tab uid>:
    <material index>:<slot>``, dispatched here by ``shell/tasks.py`` rather
    than by ``clay_mode.on_task_done`` (see :data:`TEXTURE_TASK_PREFIX`).

    Every way this can be stale is an ordinary no-op, not a refusal: the
    document can have been closed, the palette slot removed or the picker
    simply cancelled (``done.result`` is then ``None``) while the dialog was
    open, and none of those is a failure worth a toast over -- the user asked
    for a file, or didn't pick one, and either way nothing here was promised
    to still exist by the time the answer comes back.
    """
    parts = done.key.split(":", 3)
    if len(parts) != 4:
        return
    _prefix, tab_uid, index_text, slot = parts
    if slot not in TEXTURE_SLOTS:
        return
    result = done.result
    if not isinstance(result, dict):
        return
    state = clay_mode.ensure(ctx)
    tab = state.get(tab_uid)
    if tab is None:
        return
    try:
        index = int(index_text)
    except ValueError:
        return
    doc = tab.doc
    if not 0 <= index < len(doc.materials):
        return
    material = doc.materials[index]
    image = (result["width"], result["height"], result["rgba"])
    doc.set_material(index, replace(material, **{slot: image}))


def _texture_slots(ctx: Any, tab: Any, doc: Any, index: int, material: Any) -> None:
    """Assign-from-file and Clear, one row per slot.

    Clay itself paints no textures (the box/planar/LSCM unwraps only ever
    move *coordinates* around), so every one of these is a baked map an
    import carried in or a look the user is hand-assigning -- never mutated
    in place, always a fresh ``replace(material, ...)``, the same rule the
    scalar PBR fields above already follow.
    """
    widgets.field_label("textures")
    for slot in TEXTURE_SLOTS:
        image = getattr(material, slot, None)
        label = slot.replace("_", " ")
        if image is not None:
            width, height, _data = image
            widgets.muted(f"{label}: {width} x {height}")
        else:
            widgets.muted(f"{label}: empty")
        imgui.same_line()
        if ctx.busy(_texture_task_key(tab.uid, index, slot)):
            widgets.muted("...")
        else:
            assign_tip = "Assign from a file..."
            # FOLDER_OPEN, not UPLOAD (icons.py's rule: UPLOAD reads backwards
            # for both import and export in an app with no remote server) --
            # this button is "in from a file", the same reason bridge.py's
            # "Import Mesh..." already uses it.
            if controls.small_button(f"{icons.FOLDER_OPEN}##texassign{slot}", tooltip=assign_tip):
                _assign_texture(ctx, tab, index, slot)
            imgui.same_line()
            if widgets.disabled_button(f"{icons.X}##texclear{slot}", image is not None):
                doc.set_material(index, replace(material, **{slot: None}))


def _apply_library_material(doc: Any, uids: Any, material: Any) -> bool:
    """Add *material* as a new palette entry and point every one of *uids*'
    default slot at it, as **one** step.

    ``ClayDoc.add_material_and_assign``'s own shape, composed here from its
    two public doors (``add_material``, ``set_props``) rather than a third
    method on ``ClayDoc`` itself, and extended from one object to several:
    applying a saved look to a multi-object selection should cost one
    Ctrl+Z, not one per object, the same "one gesture, one step" rule
    ``add_materials_and_assign``'s own docstring gives ``add_objects``.
    """
    uids = list(uids)
    if not uids:
        return False
    mark = doc.history.mark()
    index = doc.add_material(material)
    for uid in uids:
        doc.set_props(uid, material=index)
    doc.history.collapse_since(mark)
    return True


def _material_library(ctx: Any, doc: Any, obj: Any) -> None:
    """Named materials saved under ``WARLOCK_HOME`` (``matlib.py``): save the
    selected object's current material, list what is saved, apply one back,
    delete one. See ``matlib.py``'s own module docstring for the on-disk
    shape and why it is synchronous unlike :func:`_pick_texture` above.
    """
    widgets.field_label("material library")
    home = ctx.svc.config.home
    index = min(max(int(obj.material), 0), len(doc.materials) - 1)
    # ``value=""`` every frame, the tag-add field's own sentinel
    # (``_tags``, above): this is an action ("save this"), not a persistent
    # field, so pressing Enter both saves and clears the box in one motion.
    typed = widgets.input_text(
        "##matlibsave", "", max_length=60, hint="save the current material as...", commit=True
    )
    if typed.strip():
        clay_matlib.save_material(home, typed.strip(), doc.materials[index])

    entries = clay_matlib.list_materials(home)
    if not entries:
        widgets.muted("nothing saved yet")
        return
    for entry in entries:
        imgui.push_id(f"matlib{entry.id}")
        widgets.muted(entry.name)
        imgui.same_line()
        if controls.small_button(f"{icons.CHECK}##matlibapply", tooltip=f"Apply {entry.name!r}"):
            material = clay_matlib.load_material(home, entry.id)
            if material is None:
                from ... import ops as clay_ops

                clay_ops.toast(ctx, f"{entry.name!r} could not be read.")
            else:
                _apply_library_material(doc, [obj.uid], material)
        imgui.same_line()
        if controls.small_button(f"{icons.TRASH}##matlibdel", tooltip=f"Delete {entry.name!r}"):
            clay_matlib.delete_material(home, entry.id)
        imgui.pop_id()
