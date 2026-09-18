"""Chapter 17's "right-click it. Choose Make prefab and give it a name."

The 2026-09-12 audit's docs-03 found two things wrong with the Make-prefab
gesture: neither context-menu call site (``panes/mason_menu.py``,
``panes/mason_outliner.py``) ever passed a name, so
``mason_mode.define_prefab_from_selection`` silently defaulted to the
selected node's own name with no step the reader could see as "naming" it;
and a name that collided with a template the selection already places -- the
recursion ``document.define_prefab`` itself refuses -- raised a ``ValueError``
that was swallowed with no toast at either site, so the gesture did nothing
visible at all.

Headless: this is ``mason_mode``'s controller layer and ``dialogs.PromptQueue``
is plain data plus a deque, so nothing here needs imgui or a GL context.
"""

from __future__ import annotations

from typing import Any

from warlock.studio import dialogs
from warlock.studio.modes.mason import mode as mason_mode
from warlock.studio.modes.mason.engine import nodes as nd


class _FakeCtx:
    """The minimal ctx ``mason_mode.new_document`` and
    ``define_prefab_from_selection``/``prompt_define_prefab_from_selection``
    need. ``tests/test_mason_mode.py`` carries an equivalent ``FakeCtx`` for
    that file's own suite; restated here rather than imported across test
    modules, which each fixer's tests avoid doing for the same reason two
    unrelated audit fixers never share a scratch file.
    """

    def __init__(self) -> None:
        self.svc = None
        self.state = type("S", (), {"mason": None, "mode": "home"})()
        self.settings = _Settings()
        self.prompts = dialogs.PromptQueue()
        self.toasts: list[tuple[str, str]] = []

    def toast(self, message: str, kind: str = "info", *_a: Any, **_k: Any) -> None:
        self.toasts.append((message, kind))


class _Settings:
    def __init__(self) -> None:
        self.store: dict[str, Any] = {}

    def get(self, key: str) -> Any:
        return self.store.get(key)

    def set(self, key: str, value: Any) -> None:
        self.store[key] = value


def _ctx() -> _FakeCtx:
    ctx = _FakeCtx()
    mason_mode.new_document(ctx)
    return ctx


def test_make_prefab_from_the_context_menu_lets_the_user_name_the_new_template() -> None:
    """The gesture asks first, rather than minting a name from the node.

    A reader following Chapter 17 word for word -- right-click, choose Make
    prefab, type a name they chose themselves -- must see that name on the
    template afterward, not whatever the selected node happened to be called.
    """
    ctx = _ctx()
    doc = mason_mode.ensure(ctx).active.doc
    node = doc.add_node(nd.MeshNode(uid=nd.new_uid(), name="Barrel"))
    doc.select([node.uid])

    mason_mode.prompt_define_prefab_from_selection(ctx)

    prompt = ctx.prompts.pending
    assert prompt is not None, "Make prefab must open a naming prompt"
    assert prompt.value == "Barrel"  # seeded from the node, not left blank

    prompt.on_accept("Ale Barrel")

    assert "Ale Barrel" in doc.prefabs
    assert "Barrel" not in doc.prefabs
    instances = [n for n in doc.all_nodes() if isinstance(n, nd.PrefabNode)]
    assert len(instances) == 1
    assert instances[0].template == "Ale Barrel"


def test_naming_a_prefab_after_one_it_already_places_is_refused_with_a_toast() -> None:
    """The silent half of docs-03: a colliding name used to vanish with no
    sign anything happened. ``document.define_prefab`` refuses a template
    that would contain an instance of itself (directly, or through another
    prefab it places) -- building that exact shape and naming the new prefab
    with the name it already places must surface the refusal as a toast, not
    swallow it.
    """
    ctx = _ctx()
    doc = mason_mode.ensure(ctx).active.doc

    post = doc.add_node(nd.MeshNode(uid=nd.new_uid(), name="Post"))
    doc.select([post.uid])
    assert mason_mode.define_prefab_from_selection(ctx) == "Post"

    group = doc.add_node(nd.GroupNode(uid=nd.new_uid(), name="Fence"))
    instance = nd.PrefabNode(uid=nd.new_uid(), name="Post instance", template="Post")
    doc.add_node(instance, parent_uid=group.uid)
    doc.select([group.uid])

    name = mason_mode.define_prefab_from_selection(ctx, "Post")

    assert name == ""
    assert ctx.toasts, "a colliding prefab name must toast, not do nothing silently"
    message, level = ctx.toasts[-1]
    assert level == "error"
    # A bare ``str(exc)`` is library text with no subject in front of it
    # (``tests/test_ux_todo_fixes.py::test_no_toast_forwards_a_bare_exception``'s
    # own rule) -- the toast must read as a sentence about the gesture that
    # failed, not just the document layer's own words for why.
    assert message.startswith("Could not make a prefab:")
