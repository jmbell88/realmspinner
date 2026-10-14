"""``clay_scene`` shows an agent a rig: which objects are bones, which meshes are weighted, and
the animations the document holds.

Tranche 3 of the 2026-10-13 Clay rig plan. Read-only: an agent marks bones and plays with ops
through ``clay_op`` as before; what it could not do was *see* the result.
"""

from __future__ import annotations

import numpy as np

from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay.agent import dispatch as agent_clay

from ._rigkit import skin_for, weighted
from .test_agent_clay import _Ctx, _new_agent_tab, _payload

Q90 = [0.0, 0.0, float(np.sin(np.pi / 4)), float(np.cos(np.pi / 4))]


def _scene(ctx, session) -> dict:
    return _payload(agent_clay.call(ctx, session, "clay_scene", {}))


def test_a_document_with_no_rig_reports_no_bones_and_no_animations() -> None:
    ctx, session = _Ctx(), agent_clay.Session()
    _new_agent_tab(ctx, session)
    scene = _scene(ctx, session)
    assert scene["rig"] == {
        "bones": 0, "fps": 24, "active_animation": None, "animations": [],
    }
    assert scene["objects"][0]["bone"] is False and "skin" not in scene["objects"][0]


def test_a_marked_bone_a_weighted_mesh_and_a_clip_are_all_visible_to_the_agent() -> None:
    ctx, session = _Ctx(), agent_clay.Session()
    first = _new_agent_tab(ctx, session)
    selected = agent_clay.call(ctx, session, "clay_select", {"uids": [first]})
    assert selected["isError"] is False, selected
    result = agent_clay.call(ctx, session, "clay_op", {"name": "mark-bone"})
    assert result["isError"] is False, result
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    doc = tab.doc
    body = doc.add_object(
        type(doc.objects[0])(
            uid=999, name="Body", mesh=weighted(doc.objects[0].mesh), skin=skin_for(first)
        )
    ).uid
    doc.add_animation("Wave")
    doc.set_animation_loop(0, False)
    doc.set_rig_time(2.0)
    doc.key_pose({first: {"rotation": Q90}})

    scene = _scene(ctx, session)
    rows = {row["uid"]: row for row in scene["objects"]}
    assert rows[first]["bone"] is True and "skin" not in rows[first]
    assert rows[body]["bone"] is False and rows[body]["skin"] == [first]
    assert scene["rig"]["bones"] == 1
    assert scene["rig"]["active_animation"] == 0
    assert scene["rig"]["animations"] == [
        {"name": "Wave", "duration": 2.0, "loop": False, "interpolation": "linear", "bones": 1}
    ]


def test_the_scene_row_and_the_rig_block_are_declared_in_the_output_schema() -> None:
    tools = {t.name: t for t in agent_clay.tools()}
    schema = tools["clay_scene"].output_schema
    row = schema["properties"]["objects"]["items"]["properties"]
    assert row["bone"] == {"type": "boolean"} and row["skin"]["type"] == "array"
    assert set(schema["properties"]["rig"]["properties"]) == {
        "bones", "fps", "active_animation", "animations",
    }
