"""T8: ``studio/familiar/doors.py`` -- the pure prompt/schema/parse logic
behind Familiar's navigate and create-draft skills.

``realmspinner.familiar.doors`` does not exist on the pre-T8 tree, so
every test below fails with an ``ImportError``/``ModuleNotFoundError``
before its first assertion runs against the unmodified code.
"""

from __future__ import annotations

from realmspinner.familiar import doors
from realmspinner.studio.modes.create.engine import assets as create_assets


def test_a_navigate_target_outside_the_offered_keys_is_none():
    """A destination the model was never offered -- or an explicit "none",
    or malformed output -- must never be trusted: only a key actually in
    *keys* is a real answer."""
    keys = ("go:clay", "go:mason")

    assert doors.parse_target('{"target": "go:other"}', keys) is None
    assert doors.parse_target('{"target": "none"}', keys) is None
    assert doors.parse_target("not json at all", keys) is None
    assert doors.parse_target('{"target": "go:clay"}', keys) == "go:clay"


def test_a_navigate_target_inside_a_code_fence_is_still_parsed():
    """Constrained decoding still sometimes rides inside a code fence in
    practice -- the same tolerance :func:`~.router.parse_route` keeps."""
    keys = ("go:clay",)
    fenced = '```json\n{"target": "go:clay"}\n```'
    assert doors.parse_target(fenced, keys) == "go:clay"


def test_the_navigate_schema_offers_exactly_the_destinations_plus_none():
    keys = ("go:clay", "manual")
    schema = doors.navigate_schema(keys)
    assert schema["properties"]["target"]["enum"] == ["go:clay", "manual", "none"]
    assert schema["required"] == ["target"]
    assert schema["additionalProperties"] is False


def test_the_create_schema_offers_exactly_the_asset_types_and_caps_the_prompt():
    """The 2026-09-20 audit (familiar-06): ``doors.create_schema`` was
    exercised only in the gpu lane -- its enum and 1000-char prompt cap had
    no CPU-lane assertion, while the sibling ``navigate_schema`` already had
    one (``test_the_navigate_schema_offers_exactly_the_destinations_plus_
    none``, above). Mirrors that test's own shape for ``create``."""
    asset_types = (("image", "Image"), ("3d_model", "3D Model"))
    schema = doors.create_schema(asset_types)
    assert schema["properties"]["asset_type"]["enum"] == ["image", "3d_model"]
    assert schema["properties"]["prompt"]["type"] == "string"
    assert schema["properties"]["prompt"]["maxLength"] == 1000
    assert schema["required"] == ["asset_type", "prompt"]
    assert schema["additionalProperties"] is False


def test_a_draft_with_an_unknown_asset_type_is_refused():
    asset_types = (("image", "Image"), ("3d_model", "3D Model"))

    unknown = '{"asset_type": "sprite_sheet", "prompt": "a fox"}'
    assert doors.parse_draft(unknown, asset_types) is None
    # A blank prompt is refused too -- a caller must never draft a request
    # with nothing in it to generate.
    assert doors.parse_draft('{"asset_type": "image", "prompt": "   "}', asset_types) is None
    assert doors.parse_draft("garbage", asset_types) is None

    result = doors.parse_draft('{"asset_type": "image", "prompt": "a fox"}', asset_types)
    assert result == ("image", "a fox")


def test_create_messages_name_every_asset_type():
    """Every asset type Create actually offers must appear in the user turn
    -- both halves, key and label -- or the model is choosing from a list it
    was never shown."""
    asset_types = create_assets.ASSET_TYPE_OPTIONS

    messages = doors.build_create_messages("make a lantern", asset_types)

    assert messages[0] == {"role": "system", "content": doors.CREATE_SYSTEM}
    user = messages[1]["content"]
    for key, label in asset_types:
        assert key in user
        assert label in user
    assert "make a lantern" in user


def test_navigate_messages_name_every_destination():
    destinations = (
        doors.Destination(key="go:clay", label="Go to Clay"),
        doors.Destination(key="manual", label="Open the manual"),
    )

    messages = doors.build_navigate_messages("open clay", destinations)

    assert messages[0] == {"role": "system", "content": doors.NAV_SYSTEM}
    user = messages[1]["content"]
    assert "go:clay: Go to Clay" in user
    assert "manual: Open the manual" in user
    assert "open clay" in user
