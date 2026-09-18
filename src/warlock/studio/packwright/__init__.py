"""Packwright -- the atlas packer's engine.

Pure in the way ``studio/inker/``, ``studio/clay/`` and ``studio/plotter/`` are:
no imgui, no moderngl, no pygame, no ``service``. It reaches outward exactly
eight times, across six modules: ``document.py`` for :mod:`warlock.core.undo`
(the shared history engine); ``layout.py`` for :mod:`warlock.kernels.sheet`
(the authority on the atlas ceiling and on what "trim" means); ``tsxout.py``
twice, for :mod:`warlock.kernels.grid2d.tileset` and
:mod:`warlock.studio.plotter.tsx` (the one ``.tsx`` writer in the repo, and the
type it writes); ``compose.py`` and ``wpack.py`` for
:mod:`warlock.studio.plotter.pngio` (the one RGBA-to-PNG encoder, which four
byte-identical copies used to spell); ``sources.py`` for
:mod:`warlock.kernels.grid2d.tileset` again, for ``frozen_rgba``; and
``wpack.py`` for :mod:`warlock.core.safeio.zipguard` (the shared bounded-zip reader
four container doors now share, so the ``file_size`` ceiling a ``.wpack``'s own
directory is checked against is not a fourth private copy of that bound). All
eight are pinned exactly, and at that granularity, by
``tests/packwright/test_packwright_imports.py``.

**The raster editor is deliberately not among them.** A clip's frames are read
through duck typing: :mod:`.sources` takes *a document* and asks it for frames,
because a packer is routinely handed loose PNG files and no document at all, and
an import would drag the whole raster editor in for the case that has none.

All but the first are the ``sheetout.py`` argument repeated: reach for the
module that *owns* a definition rather than restating it, so there is one answer
to "how big may an atlas be", one to "where does the alpha stop" and one to
"what does a ``.tsx`` look like".

**A layout is derived, never stored.** The document holds sources and settings;
the packer is deterministic, so re-deriving is what makes re-export of an
unchanged document byte-identical without a cache anybody has to invalidate.
"""

from __future__ import annotations
