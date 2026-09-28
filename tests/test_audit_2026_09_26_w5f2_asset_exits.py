"""Regression tests for the 2026-09-26 audit, findings create-brief-04 and
create-brief-05, both in ``studio/asset_exits.py``.
"""

from __future__ import annotations

import inspect


def test_reference_stages_is_pinned_to_service_files_editable_stages():
    """create-brief-04: ``asset_exits._REFERENCE_STAGES`` used to be a hand
    copy of ``service.files.EDITABLE_STAGES`` -- two identical-looking tuples
    that happened to agree, with nothing making them keep agreeing. Pinned
    means the same object, not merely an equal one.
    """
    from realmspinner.service import files as svc_files
    from realmspinner.studio import asset_exits

    assert asset_exits._REFERENCE_STAGES is svc_files.EDITABLE_STAGES


def test_asset_exits_module_scope_carries_no_studio_submodule_import():
    """create-brief-05: the module docstring promises "a mode's own UI
    submodule is imported lazily inside each function that needs it", but
    ``from .modes.create.ui.stages import IMAGE_STAGES`` used to sit at
    module scope, contradicting it. ``icons``/``modes``/``verbs`` are the
    documented, deliberate exception (plain data, needed to build
    ``_MODE_ICONS`` once) and stay.
    """
    from realmspinner.studio import asset_exits

    source = inspect.getsource(asset_exits)
    header = source.split("class Exit", 1)[0]
    assert ".modes." not in header, (
        "a mode's own UI submodule must not be imported at module scope -- "
        f"found one in:\n{header}"
    )


def test_plotter_add_and_packwright_add_still_import_image_stages_lazily():
    """The other half of create-brief-05: moving the import off module scope
    must not just delete the name -- ``_plotter_add``/``_packwright_add``
    still need ``IMAGE_STAGES`` to gate the near-miss button, now imported
    inside each function body.
    """
    from realmspinner.studio import asset_exits

    for fn in (asset_exits._plotter_add, asset_exits._packwright_add):
        source = inspect.getsource(fn)
        assert "from .modes.create.ui.stages import IMAGE_STAGES" in source, (
            f"{fn.__name__} lost its lazy IMAGE_STAGES import"
        )
