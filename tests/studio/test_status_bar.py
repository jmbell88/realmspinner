"""``status_bar.items``' document-mode branch, and the registry it must agree with.

shell-08 (the 2026-09-08 audit): ``items()`` hand-wrote its own tuple of
document modes (``"inker", "clay", "plotter", "packwright", "sirens"``) as a
second, unguarded copy of ``palette._DOC_MODES``'s key set (``poser`` handled
separately in both). A document mode added to ``_DOC_MODES`` in future had no
test tying it to this tuple, so the status bar would silently stop showing
that mode's document name, tool and zoom.
"""

from __future__ import annotations

from types import SimpleNamespace

from realmspinner.studio import mode_manifest, palette, status_bar


def test_status_bar_document_modes_match_the_doc_mode_registry(monkeypatch):
    """A mode registered in :data:`mode_manifest.DOC_MODES` (other than
    "poser", which is its own special case in both places) must be one
    ``status_bar.items`` draws a document row for -- proven by adding a mode
    neither module has ever heard of and checking the row appears, rather than
    by comparing the two sets structurally, which a hand-written tuple that
    happened to be copied correctly would also pass.

    The registry moved from ``palette._DOC_MODES`` to the manifest on
    2026-09-17, and so did the *module lookup*: the bar used to spell the
    module out of the mode key as ``f".{mode}_mode"``, which only held while
    every mode module sat directly under ``studio/``. The injected row below
    therefore names its module rather than implying it.
    """
    fake_tab = SimpleNamespace(label="Widget##pd9", dirty=False, view=None)
    fake_module = SimpleNamespace(active=lambda ctx: fake_tab)

    gizmo = mode_manifest.ModeManifest(
        "gizmo", "gizmo_mode", "gizmo", "Export Gizmo", "gizmo_mode"
    )
    monkeypatch.setattr(
        mode_manifest, "DOC_MODES", (*mode_manifest.DOC_MODES, gizmo)
    )

    real_import_module = mode_manifest.import_module

    def fake_import_module(name, package=None):
        if name.endswith(".gizmo_mode"):
            return fake_module
        return real_import_module(name, package)

    monkeypatch.setattr(mode_manifest, "import_module", fake_import_module)

    ctx = SimpleNamespace(state=SimpleNamespace(mode="gizmo"), cache=SimpleNamespace(jobs=[]))
    items = {item.key: item.text for item in status_bar.items(ctx)}
    assert items.get("document") == "Widget"


def test_poser_is_excluded_from_the_derived_set_and_kept_as_its_own_branch():
    """``poser`` is a real key of ``_DOC_MODES`` (with an empty export label,
    since a pose is saved rather than exported) but has no tab and no
    ``dirty`` -- it is handled by ``poser_mode.document_label`` instead, so it
    must never join the derived document-mode set."""
    assert "poser" in palette._DOC_MODES
    assert "poser" not in status_bar._document_modes()


def test_the_derived_set_matches_the_registry_minus_poser():
    assert status_bar._document_modes() == frozenset(palette._DOC_MODES) - {"poser"}


def test_resource_item_docstring_does_not_claim_protection_status_drop_order_denies():
    """``resource_item``'s docstring used to argue it follows
    ``overlay.doctor_banner``'s rule to "reserve the trailing item before
    trimming the leading detail" -- i.e. that being kept out of :func:`items`
    and right-anchored *protects* the resource meter from being the first
    thing dropped as the status group runs out of room. But
    ``menus.STATUS_DROP_ORDER`` lists ``"resources"`` first, so
    ``fit_status_rows`` drops the (right-anchored, trailing) resource meter
    *before* any of the left-hand ``items()`` keys -- the opposite priority
    from what the docstring argued for. The 2026-09-16 audit found the
    docstring stale against ``STATUS_DROP_ORDER``.
    """
    from realmspinner.studio import menus

    # The real priority the docstring must not contradict: "resources" is the
    # *first* key given up, not the reserved one.
    assert menus.STATUS_DROP_ORDER[0] == "resources"

    doc = status_bar.resource_item.__doc__ or ""
    assert "reserve the trailing item" not in doc, (
        "resource_item's docstring still claims right-anchoring reserves the "
        "meter ahead of the leading detail, which STATUS_DROP_ORDER denies -- "
        "\"resources\" is the first key fit_status_rows drops"
    )
    assert "STATUS_DROP_ORDER" in doc, (
        "resource_item's docstring should name the mechanism that actually "
        "decides drop priority"
    )
