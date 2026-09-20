"""``filetypes.describe``'s worked example against the suffix list it derives from.

shell-XX (the 2026-09-16 audit): ``IMAGE_SUFFIXES`` carries six formats
including ``.gif``, but ``describe``'s own docstring example --
``"Images" -> "Images (*.png *.jpg *.jpeg *.webp *.bmp)"`` -- shows only five
and omits ``.gif``. A worked example that disagrees with what the function
actually returns is the "docs describe code that does not exist" defect
class.
"""

from __future__ import annotations

from realmspinner.studio import filetypes


def test_describe_docstring_example_matches_image_suffixes():
    """The docstring's worked example for ``describe("Images")`` must show
    every suffix ``IMAGE_SUFFIXES`` actually carries -- it silently dropped
    ``.gif`` even though ``IMAGE_SUFFIXES`` has carried it since the file was
    written, which is exactly what ``describe`` itself is built to prevent
    (its own docstring: "a label and a pattern list maintained separately is
    exactly how a dialog comes to refuse what it accepts")."""
    doc = filetypes.describe.__doc__ or ""
    real = filetypes.describe("Images")
    assert real == "Images (*.png *.jpg *.jpeg *.webp *.bmp *.gif)"
    assert real in doc, (
        f"describe()'s docstring example does not match what describe() "
        f"actually returns ({real!r}); the worked example in the docstring "
        f"is stale against IMAGE_SUFFIXES"
    )
