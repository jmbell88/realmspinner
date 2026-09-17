"""The rig kernel: skeleton templates, pose/clip validation and storage, and
the pure spec builders for the Blender subprocess.

Layer 1 ("pure domain, no UI, no process control", ``dev/RESTRUCTURE.md``'s
table) -- this is the P4 split of the former ``src/warlock/rigging.py``
(2,849 lines) into one module per concern. **No symbol is re-exported here**:
a caller imports the submodule it needs and addresses the symbol through it
(``from warlock.kernels.rig import skeleton`` / ``skeleton.fit_template``),
the same rule the other ``kernels/`` packages already follow -- gathering
120 names back onto this file would just be the god file wearing a directory.

* :mod:`.templates` -- the skeleton template registry (``Template``,
  ``TEMPLATE_DIR``, ``get_template``, ``catalog``) and the limb presets
  grafted onto one (``get_limb_preset``); the shared capped-JSON reader and
  the shared acyclic-parent-graph check both live here too, since this
  module's own parsers are what need them first.
* :mod:`.cliplib` -- the shipped/user *clip library* (an ordered list of key
  poses that plays back as a walk cycle). Named ``cliplib``, not ``clips``:
  ``warlock/clips.py`` is a different module already, and this name is the
  one that will not collide with it once a later restructure wave folds
  ``clips.py`` into this package too.
* :mod:`.skeleton` -- fitting a template onto a mesh's bounding box, and
  Poser's skeleton editor (add/split/remove a bone, graft a limb, mirror
  pairs, the whole-skeleton validator queued as a re-rig).
* :mod:`.poses` -- a pose payload (a bone -> local quaternion map), the two
  rotation frames Poser and a clip library each speak, and the shipped pose
  libraries (the picker's presets, the deformation-QA battery).
* :mod:`.store` -- everything about a rig/pose/sheet/sprite-draft that is a
  path or a record on disk: resource ids, the rig's temp/served-name
  convention, and one file per artifact under a job directory. Deliberately
  a leaf: :mod:`.templates` and :mod:`.skeleton` both import ``RigError``
  from here, so this module imports neither of them back.
* :mod:`.blender_spec` -- the worker specs: pure dict construction, one
  function per op the Blender subprocess understands. Spawning and
  supervising that subprocess is *not* here -- it is process control, so it
  lives one layer up, at ``warlock.pipelines.blender_run``.
"""

from __future__ import annotations
