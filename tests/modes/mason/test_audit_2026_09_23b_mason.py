"""Regression tests for the 2026-09-23 (second run) audit's mason-01,
mason-02, mason-03, mason-04 and docs-01 item 1.

Each test's name is the claim, and each failed against the unfixed code
before the fix in this same change -- see the fixer's return for the pasted
failing output and call counts; that proof is not reproduced here since it
depends on a session-local scratchpad reproduction of the pre-fix methods,
which does not exist on a public checkout or in CI. mason-01 and mason-02 are
timing/complexity claims, so both are pinned by *counting the underlying
work* (``nodes.walk`` calls) against the fixed code rather than by wall
clock, per this change's own standard -- a count is exact and cannot be
flaky the way a clock budget on a shared CI box can.
"""

from __future__ import annotations

from typing import Any

import pytest

from realmspinner.studio.modes.mason import mode as mason_mode
from realmspinner.studio.modes.mason.engine import document as md
from realmspinner.studio.modes.mason.engine import nodes as nd
from realmspinner.studio.modes.mason.engine import scene as msc


def _box() -> nd.MeshNode:
    return nd.MeshNode(uid=nd.new_uid())


class _FakeCtx:
    """The same inline-``submit`` fake ``test_audit_2026_09_23_mason.py``
    already uses for ``mason_mode`` calls, kept local to this file."""

    def __init__(self) -> None:
        self.svc = None
        self.state = _AppState()
        self.settings = _Settings()
        self.toasts: list[tuple[str, str]] = []

    def submit(self, key: str, run: Any, *args: Any) -> bool:
        run(*args)
        return True

    def toast(self, message: str, kind: str = "info", *_a: Any, **_kw: Any) -> None:
        self.toasts.append((message, kind))


class _AppState:
    def __init__(self) -> None:
        self.mason = None
        self.mode = "home"


class _Settings:
    def __init__(self) -> None:
        self.store: dict[str, Any] = {}

    def get(self, key: str) -> Any:
        return self.store.get(key)

    def set(self, key: str, value: Any) -> None:
        self.store[key] = value


@pytest.fixture(autouse=True)
def _no_pygame_display(monkeypatch):
    import pygame

    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0)


def _build_diamond_chain(doc: md.MasonDoc, depth: int) -> None:
    """``T0..T{depth}``, where ``Ti`` places *two* instances of ``T{i-1}``.

    Only ``depth + 1`` distinct templates exist, and each is a handful of
    literal nodes -- cheap to author regardless of ``depth``. What blows up
    is *checking* or *expanding* this DAG without memory of a template
    already visited: the number of root-to-leaf paths is ``2**depth``, so a
    naive walk that re-explores a shared template once per path costs
    exponentially more per added level -- the shape both mason-01 (checking)
    and mason-02 (expanding) hit.
    """
    leaf = nd.GroupNode(uid=nd.new_uid(), name="T0")
    leaf.children.append(_box())
    doc.define_prefab("T0", leaf)
    for i in range(1, depth + 1):
        group = nd.GroupNode(uid=nd.new_uid(), name=f"T{i}")
        prev = f"T{i - 1}"
        group.children.append(nd.PrefabNode(uid=nd.new_uid(), name="a", template=prev))
        group.children.append(nd.PrefabNode(uid=nd.new_uid(), name="b", template=prev))
        doc.define_prefab(f"T{i}", group)


def _counting_walk(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Monkeypatch ``nodes.walk`` (the one per-segment traversal both
    ``document._prefab_refers_to_memo`` and ``scene._walk_segment`` call) to
    count its own invocations, and hand back a one-element mutable box
    holding the running count -- ``[calls]`` -- so a caller can read it after
    the fact without a ``nonlocal`` per test.
    """
    calls = [0]
    real_walk = nd.walk

    def counting(roots: Any) -> Any:
        calls[0] += 1
        return real_walk(roots)

    monkeypatch.setattr(nd, "walk", counting)
    return calls


# --- mason-01: MasonDoc.define_prefab / _prefab_refers_to ------------------


def test_define_prefab_completes_quickly_for_a_branching_prefab_reference_chain_instead_of_reexploring_it_exponentially(  # noqa: E501
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_prefab_refers_to`` used to re-walk every referenced template with
    no memo, so a diamond of references (two prefabs each placing a third)
    re-walked the shared one once per parent that reached it -- cost doubles
    per nesting level: 2.5 s at 20, 20.2 s at 23, on the frame thread
    (``define_prefab`` runs synchronously). Pinned here by counting
    ``nodes.walk`` calls building a depth-``DEPTH`` diamond chain, which is
    exact and depth-independent-of-clock, rather than by wall time.
    """
    DEPTH = 12

    fixed_doc = md.MasonDoc()
    fixed_calls = _counting_walk(monkeypatch)
    _build_diamond_chain(fixed_doc, DEPTH)
    fixed_total = fixed_calls[0]

    # Against the pre-fix, unmemoised method this chain costs one nodes.walk
    # call per root-to-leaf path -- 2**DEPTH of them, i.e. thousands at
    # DEPTH=12 -- rather than roughly one per template (O(DEPTH)). See the
    # fixer's return for the actual pre-fix call count measured at this
    # DEPTH via a scratchpad reproduction of the unfixed method.
    assert fixed_total < 20 * DEPTH, (
        "MasonDoc.define_prefab's memoised cycle check should cost roughly "
        f"one nodes.walk call per template (O(DEPTH)) building this chain, "
        f"not one per root-to-leaf path -- expected under {20 * DEPTH} calls "
        f"at nesting depth {DEPTH}, got {fixed_total}"
    )


# --- mason-02: scene.resolved_count / MasonDoc.resolved_growth -------------


def test_resolved_growth_refuses_quickly_instead_of_counting_an_exponential_prefab_expansion_to_completion(  # noqa: E501
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``resolved_count`` used to call ``_walk_segment`` directly, so it
    never got ``walk``'s own ``_bounded``/``MAX_PLACED`` stop -- the
    pre-flight a write-side caller (``resolved_growth``, ``resolved_total``)
    calls *before* attaching anything counted an exponential prefab
    expansion all the way to completion (45.9 s at 8.4 million items) instead
    of refusing once the count was already well past the ceiling.
    """
    DEPTH = 15  # resolves to roughly 3 * 2**15 ~= 98,000 items, fully expanded

    doc = md.MasonDoc()
    _build_diamond_chain(doc, DEPTH)
    node = nd.PrefabNode(uid=nd.new_uid(), name="probe", template=f"T{DEPTH}")

    # A small ceiling makes the fixed method refuse almost immediately,
    # visiting only a small fraction of what the tree actually expands to
    # (~98,000 items at DEPTH=15). Against the pre-fix method -- which called
    # _walk_segment directly, with no ceiling at all -- this would count the
    # entire expansion regardless of MAX_PLACED; see the fixer's return for
    # the actual pre-fix item and call counts measured at this DEPTH via a
    # scratchpad reproduction of the unfixed method.
    monkeypatch.setattr(msc, "MAX_PLACED", 50)
    fixed_calls = _counting_walk(monkeypatch)
    with pytest.raises(ValueError, match="MAX_PLACED"):
        msc.resolved_count(doc, [node])
    fixed_total = fixed_calls[0]
    monkeypatch.undo()  # restore nd.walk

    # The same refusal reached through the document-side wrapper the write
    # doors actually call (docs-01's own subject: MasonDoc.resolved_growth).
    monkeypatch.setattr(msc, "MAX_PLACED", 50)
    with pytest.raises(ValueError, match="MAX_PLACED"):
        doc.resolved_growth([node])
    monkeypatch.undo()

    assert fixed_total < 5000, (
        "resolved_count should refuse shortly after crossing MAX_PLACED, not "
        f"after visiting most of a ~98,000-item expansion -- got "
        f"{fixed_total} nodes.walk calls before it raised"
    )


# --- docs-01 item 1: add_node/add_nodes charge a prefab instance's resolved
# size, not just its tree size -------------------------------------------


def test_a_prefab_instance_attached_via_add_node_directly_is_still_charged_at_its_resolved_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only ``mode.place_prefab`` ever charged a placed prefab instance's
    *resolved* (expanded) size against ``scene.MAX_PLACED``; ``MasonDoc.
    add_node``/``add_nodes`` charged the one tree node it is. A door that
    reaches ``add_node``/``add_nodes`` some other way -- straight from the
    engine, an agent call, a future controller -- could still attach an
    instance whose template alone resolves past the ceiling while the
    tree-side check saw a single, unremarkable node.
    """
    doc = md.MasonDoc()
    # One ordinary node already in the scene -- needed so the *combined*
    # total (existing + growth), not the growth alone, is what crosses the
    # ceiling below; see this test's own docstring in the fixer's return for
    # why (``resolved_growth`` is itself bounded by mason-02's fix, so a
    # growth that alone exceeds the ceiling raises scene.py's own generic
    # refusal before ``_check_resolved_placed``'s combined check runs).
    doc.add_node(_box())
    template = nd.GroupNode(uid=nd.new_uid(), name="cluster")
    for _ in range(5):
        template.children.append(_box())
    doc.define_prefab("Cluster", template)  # 6 items resolved: group + 5 meshes

    monkeypatch.setattr(msc, "MAX_PLACED", 6)
    node = nd.PrefabNode(uid=nd.new_uid(), name="probe", template="Cluster")

    # The tree-side check alone would pass this (one PrefabNode is one tree
    # node, and 1 existing + 1 new = 2, comfortably under 6), so a failure
    # here means only the resolved-size charge (1 existing + 6 resolved = 7)
    # caught it.
    with pytest.raises(ValueError, match="resolved size"):
        doc.add_node(node)

    assert len(doc.all_nodes()) == 1, "refused before the instance was attached"


def test_add_nodes_also_charges_a_prefab_instance_at_its_resolved_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The batched sibling of the test above -- ``add_nodes`` is the door a
    prefab-instance array or a multi-node paste would actually go through.
    """
    doc = md.MasonDoc()
    doc.add_node(_box())  # see the sibling test's docstring for why
    template = nd.GroupNode(uid=nd.new_uid(), name="cluster")
    for _ in range(5):
        template.children.append(_box())
    doc.define_prefab("Cluster", template)

    monkeypatch.setattr(msc, "MAX_PLACED", 6)
    node = nd.PrefabNode(uid=nd.new_uid(), name="probe", template="Cluster")

    with pytest.raises(ValueError, match="resolved size"):
        doc.add_nodes([node])

    assert len(doc.all_nodes()) == 1


def test_place_prefab_still_toasts_a_friendly_message_rather_than_letting_add_nodes_own_refusal_escape(  # noqa: E501
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``mode.place_prefab``'s own resolved-size pre-check (mason-01, first
    run) still runs *before* ``add_node``, so its friendly toast -- not
    ``add_node``'s own generic ``ValueError`` -- is what the user sees. This
    is the "every door refuses" claim's other half: docs-01 must not have
    turned place_prefab's toast into an uncaught crash.
    """
    ctx = _FakeCtx()
    tab = mason_mode.new_document(ctx)
    doc = tab.doc
    template = nd.GroupNode(uid=nd.new_uid(), name="cluster")
    for _ in range(5):
        template.children.append(_box())
    doc.define_prefab("Cluster", template)

    monkeypatch.setattr(msc, "MAX_PLACED", 3)
    result = mason_mode.place_prefab(ctx, "Cluster")

    assert result is None
    assert doc.all_nodes() == []
    assert any(kind == "error" for _msg, kind in ctx.toasts)


# --- mason-03: Tools column buttons grey out with a reason ------------------


def test_align_distribute_drop_and_array_buttons_grey_out_with_a_reason_when_the_selection_is_too_small():  # noqa: E501
    """Align, Distribute, Drop to ground and both Array buttons used to call
    ``widgets.disabled_button`` with no ``reason=`` -- greyed out with
    nothing to say why. Checked by source, the same way
    ``test_audit_2026_09_23_packwright.py``'s reason-coverage test is: no
    live imgui window is needed to assert a call passes the kwarg.
    """
    import inspect

    from realmspinner.studio.modes.mason.ui.panes import tools as mason_tools

    placement_source = inspect.getsource(mason_tools._placement)
    array_source = inspect.getsource(mason_tools._array)

    assert placement_source.count("Align##masonalign") == 1
    assert "reason=" in placement_source.split("Align##masonalign", 1)[1].split(
        "):", 1
    )[0], "the Align button must grey out with a reason"

    assert placement_source.count("Distribute##masondistribute") == 1
    assert "reason=" in placement_source.split("Distribute##masondistribute", 1)[1].split(
        "):", 1
    )[0], "the Distribute button must grey out with a reason"

    assert placement_source.count("Drop selection to ground##masondrop") == 1
    assert "reason=" in placement_source.split(
        "Drop selection to ground##masondrop", 1
    )[1].split("):", 1)[0], "the Drop to ground button must grey out with a reason"

    assert array_source.count("Array (linear)##masonarraylinear") == 1
    assert "reason=" in array_source.split("Array (linear)##masonarraylinear", 1)[1].split(
        "):", 1
    )[0], "the Array (linear) button must grey out with a reason"

    assert array_source.count("Array (radial)##masonarrayradial") == 1
    assert "reason=" in array_source.split("Array (radial)##masonarrayradial", 1)[1].split(
        "):", 1
    )[0], "the Array (radial) button must grey out with a reason"


# --- mason-04: the Tools column docstring and the duplicated pivot ---------


def test_mason_tools_docstring_agrees_with_what_mason_tools_body_actually_draws() -> None:
    """The module docstring used to say the tool grid, pivot and snap all
    moved to ``mason_header`` -- only pivot did, and ``_body`` drew it a
    *second* time regardless, so it appeared twice in one frame. The
    docstring is corrected to describe what actually moved (pivot only), and
    the duplicate pivot control is removed from ``_body`` -- this pins both:
    the docstring must not claim the tool grid or snap moved (they are still
    drawn here), and ``_body`` must draw pivot at most once, i.e. not at all,
    since ``header.py`` owns it now.
    """
    import inspect

    from realmspinner.studio.modes.mason.ui.panes import header as mason_header
    from realmspinner.studio.modes.mason.ui.panes import tools as mason_tools

    doc = mason_tools.__doc__ or ""
    # The pre-fix wording, verbatim: "Mason's transform tools, pivot and
    # snap are exactly that same kind of between-clicks setting and belong
    # on ``mason_header`` instead" -- claiming a three-item move when only
    # pivot happened. The docstring must no longer say all three belong on
    # the header; it must instead say the move has not happened for the
    # other two.
    assert "transform tools, pivot and snap" not in doc, (
        "the docstring must not claim the transform tool grid, pivot and "
        "snap all belong on mason_header -- only pivot actually moved there"
    )
    assert "has not happened" in doc.lower() or "did not" in doc.lower(), (
        "the docstring must say the tool-grid/snap move to the header did "
        "not happen, since _tool_grid and _snap are still drawn in this "
        "module's own _body"
    )

    body_source = inspect.getsource(mason_tools._body)
    assert "_pivot(" not in body_source, (
        "_body must not draw a pivot control of its own -- it moved to "
        "mason_header, and drawing it here too is the duplicate the "
        "2026-09-23 audit's mason-04 found"
    )
    assert not hasattr(mason_tools, "_pivot"), (
        "the dead _pivot function should be removed along with its call, "
        "not left behind unused"
    )

    # header.py still draws exactly one pivot control -- the move mason-04
    # found genuinely happened for this one field.
    header_source = inspect.getsource(mason_header)
    assert header_source.count('"mason-header-pivot"') == 1
