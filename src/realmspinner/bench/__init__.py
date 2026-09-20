"""A fixed benchmark: the same prompts, the same seeds, measured the same way.

This package exists to answer one question the app cannot: did a change make
the assets better, or does it only feel that way? It is a developer tool, not
a feature -- nothing in ``studio/`` or ``service/`` reaches into it to run a
suite or score anything, and it reaches the app the same way a user does,
through ``service.jobs.create_job`` on a real ``studio.runtime.Runtime``.
There are two exceptions, both readers rather than runners. ``bench.findings``
is a pure-stdlib, torch-free *reader* of ``findings.json`` that the generate
panes import to show an "accept 6/8" hint next to a field -- reading a report
is not running a benchmark. What writes that file lives in
``service/findings.py``, because verdicts are rows in the job DB now and the
sweep that produces them runs on the live queue: the parameter-sweep half of
this package (``sweep.py``, ``verdicts.py``, ``report.py``) has moved out to
``service/sweeps.py``, ``service/verdicts.py`` and ``service/findings.py``.
``bench.metrics`` is the second (2026-09-13): ``studio.agent_clay`` lazily
imports ``silhouette_iou_masks``/``mask_aspect`` (by way of
``compare_silhouette``) to answer ``clay_render``'s ``compare`` header with a
shape IoU between a stored reference and a live render -- the same arithmetic
this package's own ``silhouette_iou`` runs on two files on disk, reused rather
than copied so the two could not quietly drift apart. Reading a pure function
of two masks is not running a benchmark either. What is left here beyond
those two readers is the regression half -- the same prompts, the same seeds,
measured the same way.

The split mirrors kernels/sheet.py: everything decidable (the suite, the
recipes, the manifest, the grid, the aggregation in ``score.py``) is pure and
torch-free, and only the three stages that genuinely need a GPU -- running
jobs, rendering views, and the metrics themselves, which import torch inside
their own functions -- touch one. That is what lets a metric change be
re-applied to every past run without regenerating anything: ``bench score``
reads the views off disk.

The full A/B is therefore three commands: ``run --stage model --render`` twice
under two recipes, then ``score <b> --against <a>``.

Entry point: ``python -m realmspinner.bench <subcommand>``. Deliberately not
``realmspinner``'s own CLI -- that is the user-facing binary, and its flat
``choices=[...]`` would have to become a subparser tree to host this.
"""

from __future__ import annotations
