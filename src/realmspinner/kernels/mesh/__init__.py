"""Clay's geometry engine -- the pure half.

Everything under this package is arrays and arithmetic. It may import numpy
and, **lazily inside the functions that need them**, Pillow (``PIL``, for the
texture members of a saved document), ``trimesh`` and ``manifold3d`` (the
boolean kernel) and ``scipy`` (a KD-tree and connected components). From the
project it may reach exactly three leaves: the shared history engine
``realmspinner.core.undo``, the bounded container readers
``realmspinner.core.safeio``, and ``realmspinner.kernels.geom3d`` -- of which
only ``gltf``, ``math3d`` and ``glbio``, the GLB parser the importer's
preflight reads. Nothing else: no ``imgui``, no ``moderngl``, no ``pygame``,
nothing from ``service``, ``queue`` or ``pipelines``, and no sibling pure
package. That is not tidiness for its own sake. A user who knows the shape
they want gets a path to it that is not "install Blender", and the only way
that path stays trustworthy is if every rule about geometry -- what a face is,
which way a normal points, what a mirror does to winding, what an extrude does
to a texture seam -- is assertable headlessly, without a window and without a GPU.

That list is pinned by ``tests/modes/clay/test_clay_imports.py``, the way the other
pure packages pin theirs, so the next outward import is a decision rather
than a discovery. ``gltf`` in particular is reached for rather than mirrored
because a Clay material *is* a ``gltf.Material`` -- see :mod:`.document`.
"""
