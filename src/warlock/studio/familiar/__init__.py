"""The Familiar programme's headless half.

The GL-side preview/apply mechanics (a scratch clone, the sandboxed ``ctx``
an agent tool call runs against, and transplanting an accepted preview onto
the real document) live in :mod:`~warlock.studio.familiar_preview` instead --
that module reaches ``agent_clay``, which imports ``clay_view`` (``moderngl``)
and ``panes.clay_tools`` (``imgui_bundle``), and this package must never
carry that import, even by way of a relative-import chain two hops long (see
``tests/_pure_packages.py::_module_roots`` and
``tests/familiar/test_familiar_imports.py``).

This package holds -- and, as T3 lands, will hold -- the parts of Familiar
that need no window, no service door and no network: authoring a training
card, retrieval over prior sessions, routing an incoming message, and the
thread/turn bookkeeping around a conversation.
"""

from __future__ import annotations
