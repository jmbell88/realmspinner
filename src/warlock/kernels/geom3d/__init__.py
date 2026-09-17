"""3D arithmetic and glTF, with no viewport attached.

Matrices, quaternions, camera maths, and the reader and writer that turn a
GLB into arrays and back. Clay's meshes, Mason's scenes, Poser's rigs, the
character families and the sheet renderer all need these, and exactly none
of them need a GL context to need them -- which is why this half was pulled
out of ``studio/viewer/`` (2026-09-17) and the drawing half was left behind.
"""
