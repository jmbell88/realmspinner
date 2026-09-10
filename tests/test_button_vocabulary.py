"""The shared button layer is one vocabulary (the 2026-09-08 audit).

An inventory of the panes found ~40 call sites hand-spelling
``controls.button(..., role=ButtonRole.X)`` where a ``widgets.*_button`` helper
already existed for that exact role. Panes are fixed elsewhere; this file pins
the layer they call: :func:`widgets.destructive_button` grew the same
``reason=``/``tooltip=`` contract its siblings already had (the forcing
function behind most of the drift -- ``plotter_layers.py`` and
``clay_tools.py`` each hand-rolled around its absence), the shared layer's own
call sites (``forms.Form.footer``, ``widgets.pane_header``, dialogs.py's
confirm modal) stopped hand-spelling the role they already had a name for, and
``ButtonRole.ICON`` -- zero call sites anywhere -- is gone rather than kept as
a shape nothing draws.

Every test below is written to fail against the code before that pass; see
each docstring for what specifically it would have caught.
"""

from __future__ import annotations

import ast
import inspect

from _ui_context import imgui_context

from warlock.studio import controls, dialogs, forms, probe, toolbar, widgets


def _role_button_calls(module) -> list[int]:
    """Line numbers of every ``controls.button(role=...)`` call in ``module``.

    AST rather than a text search, deliberately: several of the comments this
    pass left behind *say* ``controls.button(role=...)`` in prose (explaining
    what a call site used to be), and a substring search would count its own
    documentation as a violation. Only an actual ``ast.Call`` counts.
    """
    tree = ast.parse(inspect.getsource(module))
    lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "button"):
            continue
        if not (isinstance(func.value, ast.Name) and func.value.id == "controls"):
            continue
        if any(kw.arg == "role" for kw in node.keywords):
            lines.append(node.lineno)
    return lines


def test_destructive_button_has_grown_its_siblings_reason_and_tooltip():
    """``destructive_button``'s signature now matches ``ghost_button`` and
    ``primary_button`` shape for shape: ``label``, ``size``, then
    ``enabled``/``reason``/``tooltip`` keyword-only.

    Before this pass ``destructive_button`` took only ``label``, ``size`` and
    ``enabled`` -- calling it with ``reason=`` raised ``TypeError``, which is
    exactly what made ``plotter_layers.py`` hand-roll
    ``controls.button(role=ButtonRole.DESTRUCTIVE)`` next to it and
    ``clay_tools.py`` carry a whole second ``_destructive_button`` wrapper.
    """
    ghost_params = inspect.signature(widgets.ghost_button).parameters
    primary_params = inspect.signature(widgets.primary_button).parameters
    destructive_params = inspect.signature(widgets.destructive_button).parameters
    assert destructive_params.keys() == ghost_params.keys() == primary_params.keys()
    for name in ("enabled", "reason", "tooltip"):
        assert destructive_params[name].kind is inspect.Parameter.KEYWORD_ONLY


def test_a_disabled_destructive_button_surfaces_its_reason_to_the_census(monkeypatch):
    """Mirrors ``test_probe.py::test_a_disabled_button_reaches_the_census`` for
    the third role button: a real (rendererless) imgui context, a real click
    attempt, and the reason read back off ``probe.FRAME_CONTROLS``.

    Fails against the unfixed code two ways at once: the old signature has no
    ``reason`` keyword at all (``TypeError`` before the assert is ever
    reached), and even a caller working around that by other means had no
    ``_button_with_note``/``probe.record`` route to report it through.
    """
    with imgui_context(monkeypatch) as imgui:
        imgui.new_frame()
        imgui.set_next_window_size((400.0, 200.0))
        imgui.begin("probe")
        probe.begin_frame()
        try:
            clicked = widgets.destructive_button(
                "Delete forever",
                enabled=False,
                reason="Nothing is selected.",
                tooltip="Removes the selected rows for good.",
            )
        finally:
            imgui.end()
        (one,) = probe.FRAME_CONTROLS
        imgui.end_frame()
        imgui.render()
    assert clicked is False
    assert one.label == "Delete forever"
    assert one.kind == "button"
    assert one.enabled is False
    assert one.reason == "Nothing is selected."


def test_button_role_icon_no_longer_exists():
    """The 2026-09-08 audit found zero call sites for ``ButtonRole.ICON``
    anywhere in ``src/`` or ``tests/`` -- every selected glyph button had
    already moved to ``widgets.icon_button``'s own ``selected`` flag. Deleted
    rather than kept as a documented no-op, so a future ``_button_colours``
    branch cannot be written for a shape nothing draws.
    """
    assert not hasattr(controls.ButtonRole, "ICON")
    assert set(controls.ButtonRole) == {
        controls.ButtonRole.PRIMARY,
        controls.ButtonRole.SECONDARY,
        controls.ButtonRole.GHOST,
        controls.ButtonRole.DESTRUCTIVE,
    }


def test_one_cancel_width_is_published_and_used_by_the_dialogs_module():
    """``widgets.CANCEL_WIDTH`` is the one number a dialog's Cancel is drawn
    at; ``dialogs.py`` -- the confirm-dialog machinery every "Delete this?"
    prompt in the app goes through -- draws its own Cancel buttons at it.

    Before this pass every Cancel in the app (dialogs.py included) carried its
    own literal (``BUTTON_W``, or one of ``sp(90)``/``sp(100)``/``sp(110)``/
    ``sp(120)`` in the panes), so there was no single name to assert on at all.
    """
    assert widgets.CANCEL_WIDTH == 110.0
    source = inspect.getsource(dialogs)
    assert source.count("widgets.CANCEL_WIDTH") >= 2


def test_widgets_and_forms_no_longer_hand_spell_a_role_button():
    """The shared layer draws its own buttons through its own helpers now.

    Fails against the unfixed code: ``widgets.pane_header``'s trailing-action
    row called ``controls.button(role=ButtonRole.GHOST)`` directly, and
    ``forms.Form.footer`` spelled all three of Reset, Cancel and its primary
    action the same hand-rolled way -- in the same module that defines
    ``ghost_button``/``primary_button`` for exactly that job.
    """
    assert _role_button_calls(widgets) == []
    assert _role_button_calls(forms) == []


def test_toolbar_forwards_a_destructive_items_reason_and_tooltip():
    """``toolbar``'s destructive branch used to call
    ``widgets.destructive_button(f"{item.label}{ident}", enabled=item.enabled)``
    and drop ``item.reason``/``item.tooltip`` on the floor -- ``Item`` has
    always carried both fields, and the primary/ghost branches three lines
    either side of this one have always forwarded them; only the destructive
    one could not, because ``destructive_button`` had nowhere to put them
    before this pass. Reads the source for the two call sites that actually
    changed, since driving the tiering machinery this call sits inside needs a
    real toolbar row rather than a unit-level call.

    Also pins ``toolbar``'s one remaining hand-spelled ``controls.button(role=
    ...)`` call at exactly one: the *selected*-item branch, which cannot move
    to a ``widgets.*_button`` at all -- ``item.role`` is chosen at runtime from
    all four roles, and none of ``ghost_button``/``primary_button``/
    ``destructive_button`` accepts the ``selected`` flag that draws the shared
    selection wash and ring. A second hand-spelled call anywhere else in the
    file still fails this.
    """
    source = inspect.getsource(toolbar)
    start = source.index("elif item.role is controls.ButtonRole.DESTRUCTIVE")
    end = source.index("elif item.role is controls.ButtonRole.PRIMARY", start)
    destructive_branch = source[start:end]
    assert "reason=item.reason" in destructive_branch
    assert "tooltip=item.tooltip" in destructive_branch
    assert len(_role_button_calls(toolbar)) == 1


def test_dialogs_role_button_calls_are_the_documented_imgui_injection_pair():
    """``dialogs.py`` converted its confirm modal's Cancel to
    ``widgets.ghost_button``; its Prompt modal's Save and Cancel did not, and
    could not: ``test_dialogs_prompt.py`` drives that exact draw call against
    a fake swapped in as ``dialogs.imgui`` and reads the click back through
    ``controls.button``'s ``_imgui=`` escape hatch, which the ``widgets``
    role helpers do not carry (they always call the real
    ``imgui_bundle.imgui``, see ``widgets._button_with_note``). Converting
    those two would not fail loudly -- it would make every one of
    ``test_dialogs_prompt.py``'s tests reach for a GL context that headless
    pytest does not have. Pinned at exactly two so a third hand-spelled call
    anywhere else in the file still fails here.
    """
    assert len(_role_button_calls(dialogs)) == 2


def test_no_review_tag_chip_is_narrower_than_the_word_in_it(monkeypatch):
    """"good-topology" drew as "good-topolog" in a 300 dp sidebar.

    ``tag_toggles`` asked for ``grid_width(3)`` -- three columns, declared
    rather than derived -- and imgui neither wraps nor shrinks a button's label
    to fit its frame, so the longest tag in the Good vocabulary was simply cut
    off inside its own chip. Review's whole job is choosing between words that
    differ by three characters (``good-texture`` / ``bad-texture``), so a tag
    the reader has to guess at is the one defect this pane cannot carry.

    The column count comes from the widest chip now, measured across *both*
    vocabularies so the two rows share one grid. Asserted against the census's
    real rects at a sidebar width, which is the only place the old arithmetic
    was wrong -- it was correct at any width where three columns happened to
    fit, which is why nothing caught it.
    """
    from warlock.service import verdicts as verdicts_mod

    with imgui_context(monkeypatch) as imgui:
        imgui.new_frame()
        # The Review sidebar's own width, which is where this failed. A wider
        # window fits three columns and hides the bug entirely.
        imgui.set_next_window_size((300.0, 700.0))
        imgui.begin("tags")
        probe.begin_frame()
        try:
            widgets.tag_toggles("review", [], True)
        finally:
            imgui.end()
        rows = list(probe.FRAME_CONTROLS)
        # The census keeps the imgui id, and the id is the label plus a
        # ``##`` suffix; only the part before it is drawn, so only that part
        # is what the frame has to be wide enough for.
        widths = {
            row.label.split("##", 1)[0]: (
                row.rect[2],
                imgui.calc_text_size(row.label.split("##", 1)[0]).x,
            )
            for row in rows
        }
        imgui.end_frame()
        imgui.render()

    every = set(verdicts_mod.GOOD_TAGS) | set(verdicts_mod.BAD_TAGS)
    assert every <= set(widths), sorted(every - set(widths))
    for tag in sorted(every):
        chip, text = widths[tag]
        assert chip >= text, f"{tag} is drawn {chip:.0f} px wide for {text:.0f} px of label"
    # One grid, not two: a Good row sized to "good-topology" and a Bad row
    # sized to "wrong-style" stacked on each other read as ragged even though
    # neither clipped.
    assert len({round(chip) for chip, _text in widths.values()}) == 1
