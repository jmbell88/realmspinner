"""Makes ``tests/modes/`` a package, so a mode's suite is importable from anywhere.

Several tests share a fake or a builder with another mode's suite rather than
duplicate it (``test_frame_thread_doors.py`` drives three modes through their own
suites' ``_ctx``). Before P8 of the restructure every such helper lived at the
``tests/`` root, which ``conftest.py`` always puts on ``sys.path``; below it, a
bare ``from test_sirens_mode import`` only resolves if pytest happens to have
collected that directory first. As a package the import is
``modes.sirens.test_sirens_mode`` -- the very module object pytest imported --
whatever order, or single file, the run collects.
"""
