"""``Problem`` and ``Advisory``: validation messages that know which control
they are about.

Split out of ``widgets.py`` (2026-09-18 restructure, P5) because both are
plain ``str`` subclasses with no imgui in them, while everything else in
``widgets.py`` draws. Create's engine (``modes/create/engine/{recipe,mesh,
character}.py``) builds these -- ``validate``/``problems_for`` are exactly the
"what a recipe means" half the plan calls for -- and an engine module may not
import ``widgets``, which pulls in imgui at module scope. Moving the two
classes to a stdlib-only home is what lets the engine keep using its own
vocabulary for "everything wrong" and "worth knowing" without also importing
a window root.
"""

from __future__ import annotations


class Problem(str):
    """A validation message that knows which control it is about.

    A ``str`` subclass, deliberately: the aggregate block above Generate does
    ``imgui.text_wrapped(problem)`` and the tests compare against plain
    strings, and both keep working unchanged. What it adds is the half that was
    missing when the *keyboard* door refused a submit -- Ctrl+Enter and the
    command palette both call ``generate``/``promote`` directly, where the only
    feedback was the block of red text in a pane the user may not be looking
    at, so a refused Ctrl+Enter did nothing observable at all.

    ``field`` is empty for a problem that names no single control ("Choose a
    reference first" is about the library, not about a widget in the form), and
    :meth:`state.note_field_error` already treats that as "keep going to the
    toast".
    """

    __slots__ = ("field",)

    def __new__(cls, text: str, field: str = "") -> Problem:
        self = super().__new__(cls, text)
        self.field = field
        return self


class Advisory(str):
    """Something worth knowing that is **not** stopping the press.

    ``Problem``'s sibling and deliberately a separate type rather than a
    severity field on it: ``problems_for`` is documented as "everything
    stopping a press" and every one of its members disables Generate, so a
    warning added to that list would refuse a request the app has no grounds to
    refuse. The two are drawn in one block and are never merged into one list.

    The distinction is not decorative. The first thing this carries is the
    open-form lint, and an audit-flagged open form still grades usable two
    times in five (dev/measurements/2026-09-02-fantasy-v1.md) -- a rate that
    is worth telling somebody about and nowhere near a verdict. An advisory
    that blocked would be the app claiming a certainty the corpus does not
    support.

    Same ``field`` contract as ``Problem`` so one repair helper can serve both.
    """

    __slots__ = ("field",)

    def __new__(cls, text: str, field: str = "") -> Advisory:
        self = super().__new__(cls, text)
        self.field = field
        return self
