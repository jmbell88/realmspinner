"""Mason -- the 3D scene editor's engine.

Pure in the way ``studio/clay/`` and ``studio/plotter/`` are: stdlib plus numpy
plus a lazy Pillow, no imgui, no moderngl, no pygame, no ``service``. What that
buys here is what it bought there, in a document whose subject is an
*arrangement* rather than an asset: every rule about what a group's transform
reaches, which flag a hidden ancestor overrides, where the six copies of one
prefab actually land and what a sculpt brush costs the undo budget is assertable
with no window and no GPU.

The outward set is :mod:`warlock.studio.undo` -- the history engine the raster
editor, Clay and Plotter already share -- three modules of the viewer, the
container-level GLB reader, and the four guard leaves. It is pinned exactly by
``tests/mason/test_mason_imports.py``, which was written before this package
existed, so the next outward import is a decision rather than a discovery.

**The one that is deliberately absent is ``clay``.** Mason places Clay's
primitives, and importing ``clay.primitives`` to build them is the obvious move
and the wrong one: a library asset has to resolve through a callback whatever
happens -- it lives behind a job id this package may not turn into a path -- and
one resolution mechanism beats two. So both kinds of reference go through the
same :class:`~.refs.GeometrySource` door, and this package's reach stays a leaf
rather than a chain.

**A scene links; only an export embeds.** The document stores a job id and a
generator's parameters, never geometry: sixty textured GLBs written into every
save is hundreds of megabytes a keystroke, and re-exporting an asset from Clay
would leave the scene showing a stale copy with nothing in the file to say why.
"""
