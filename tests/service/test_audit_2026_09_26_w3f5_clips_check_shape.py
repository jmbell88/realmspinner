"""Regression tests for the 2026-09-26 audit, finding poser-poses-01.

``service.clips._check_shape`` built a clip's ``segments`` with a bare
``int(n)`` and iterated ``payload["poses"]``/``payload["clips"]`` calling
``.get`` on each entry with no check that the entry was an object at all.
Both silently misbehaved instead of refusing by field: a non-numeric segment
(a string, ``None``) raised a raw ``TypeError``/``ValueError`` with no field,
a fractional one (``2.9``) truncated to 2 with no error whatsoever, and a
non-object pose or clip entry (a bare string in a hand-edited or
agent-written payload) raised a raw ``AttributeError``.
"""

from __future__ import annotations

import pytest

from realmspinner.service import Invalid
from realmspinner.service import clips as svc_clips

TEMPLATE = "humanoid"


def _as_payload(view: dict) -> dict:
    return {"space": view["space"], "poses": view["poses"], "clips": view["clips"]}


def _shipped(svc) -> dict:
    return svc_clips.library(svc, TEMPLATE)


def test_check_shape_refuses_a_non_numeric_or_fractional_segment_with_a_field(svc):
    payload = _as_payload(_shipped(svc))
    payload["clips"][0]["segments"][0] = "two"
    with pytest.raises(Invalid) as caught:
        svc_clips.save(svc, TEMPLATE, payload)
    assert caught.value.field == "segments"


def test_check_shape_refuses_a_fractional_segment_instead_of_truncating_it(svc):
    payload = _as_payload(_shipped(svc))
    payload["clips"][0]["segments"][0] = 2.9
    with pytest.raises(Invalid) as caught:
        svc_clips.save(svc, TEMPLATE, payload)
    assert caught.value.field == "segments"
    # It must never have been silently accepted as int(2.9) == 2: a whole,
    # in-band value like 2 is a legal segment and would raise nothing at all
    # if this fell back to truncating instead of refusing.
    assert "2.9" in caught.value.message


def test_check_shape_refuses_a_non_object_pose_entry_with_a_field(svc):
    payload = _as_payload(_shipped(svc))
    payload["poses"].append("not an object")
    with pytest.raises(Invalid) as caught:
        svc_clips.save(svc, TEMPLATE, payload)
    assert caught.value.field == "poses"


def test_check_shape_refuses_a_non_object_clip_entry_with_a_field(svc):
    payload = _as_payload(_shipped(svc))
    payload["clips"].append("not an object")
    with pytest.raises(Invalid) as caught:
        svc_clips.save(svc, TEMPLATE, payload)
    assert caught.value.field == "clips"
