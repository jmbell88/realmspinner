"""The Familiar programme's headless half.

The GL-side preview/apply mechanics (a scratch clone, the sandboxed ``ctx``
an agent tool call runs against, and transplanting an accepted preview onto
the real document) live in :mod:`~warlock.studio.familiar_preview` instead --
that module reaches ``agent_clay``, which imports ``clay_view`` (``moderngl``)
and ``panes.clay_tools`` (``imgui_bundle``), and this package must never
carry that import, even by way of a relative-import chain two hops long (see
``tests/_pure_packages.py::_module_roots`` and
``tests/familiar/test_familiar_imports.py``).

This package holds the parts of Familiar that need no window and no service
door: authoring a training card, retrieval over prior sessions, routing an
incoming message, the thread/turn bookkeeping around a conversation -- and,
since the 2026-09-17 restructure, :mod:`.llama_client`, the one module here
that *does* touch the network. It moved down from ``pipelines/`` because it
needs :mod:`.contract`'s sizing tables and ``contract`` may not be promoted
the other way (``contract.derive_clay_card`` depends on the live
``agent_clay`` tool surface, layer 5). Every other module here stays
importable by a training script with no network stack, which is what
``tests/familiar/test_familiar_imports.py``'s httpx ban protects; that test
carries the one recorded exemption ``llama_client.py`` needs to be what it
is -- a real HTTP client -- without weakening the ban for anything else.
"""

from __future__ import annotations
