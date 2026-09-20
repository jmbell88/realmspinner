"""The desktop app: one pygame window, one ModernGL context, imgui panels.

Importing this package must not import pygame or moderngl -- ``realmspinner doctor``
and the test suite reach for :mod:`realmspinner.kernels.geom3d.math3d` and friends on
machines with no display. The window lives in :mod:`realmspinner.studio.main` and is
only built when someone actually runs it.
"""
