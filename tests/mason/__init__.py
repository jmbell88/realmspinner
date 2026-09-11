"""Makes Mason's tests a package, so their basenames cannot collide.

``conftest.py`` sets no ``importmode``, so under pytest's default prepend mode a
test module's *basename* must be unique across the whole tree -- a duplicate is
a hard ``import file mismatch`` collection **error**, not a skip. This marker
makes these modules import as ``mason.test_*`` instead of bare ``test_*``.

It is here for the reason ``tests/clay/__init__.py`` is, only more so: Mason's
module names are the most generic set anyone has added to this tree --
``test_document``, ``test_scene``, ``test_ops``, ``test_terrain``, ``test_pick``
-- and three of those already exist under ``tests/clay`` or would collide with
``studio/plotter``'s own resolver tests the day somebody adds them.
"""
