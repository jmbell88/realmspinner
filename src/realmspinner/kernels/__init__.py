"""Layer 1: the shared domain kernels -- geometry, meshes, pixels, tile
grids, rigs, palettes, audio and the Manual's text.

A kernel is pure: it computes, it refuses, and it has no window, no job
queue and no child process. That is what lets the same code answer to a
mode, to a service door, to a pipeline worker and to an agent tool call
without any of them importing each other.

These packages lived under ``studio/`` until 2026-09-17 and the cost was
visible in the tree: ``characters/`` reached into Clay's mesh engine through
function-scope imports to dodge a layering rule, ``pipelines/pixel.py``
carried a second palette parser because it could not import the first, and
``pipelines/tilemask.py`` restated a bit table it could not reach. Nothing
here imports ``studio``; ``tests/test_layering.py`` is what says so.
"""
