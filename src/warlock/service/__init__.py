"""The app's business logic, independent of any transport.

Everything here used to live inside ``app.py``'s route bodies. It is plain
synchronous Python that raises :mod:`~warlock.service.errors` exceptions
instead of ``HTTPException``, so the desktop UI can call it from a worker
thread. The HTTP routes that shape was originally kept for are gone -- the app
serves no HTTP and opens no listening socket. It opens two, both loopback
clients of a resident child this process itself spawned: ``pipelines/trellis.py``
talks to the local ``trellis-server.exe`` subprocess, and
``familiar/llama_client.py`` talks to Familiar's resident ``llama-server.exe``
(the 2026-09-18 audit, finding docs-04 -- the second arrived with Familiar and
this sentence never grew to name it). So the exceptions are the whole contract
now rather than an intermediate form on the way back to status codes.

**And then a second caller arrived, which is why that sentence is narrower than
it was.** :mod:`warlock.mcp` lets an external agent drive Clay over a local
named pipe, so the app does now accept an inbound connection -- but a named
pipe is not a socket, carries no port, and is reachable by nothing off the
machine. The shape this layer was given for the routes turned out to be exactly
what an agent wants, and better: ``field`` on a refusal is the address of the
argument that was wrong, and HTTP had nowhere to put it.

Functions take a :class:`~warlock.service.core.WarlockService` as their first
argument rather than hanging off it as methods: the split into modules is by
subject (jobs, rigs, sheets, ...) and a single 700-line class would only put
that structure back into one file.
"""

from .core import WarlockService
from .errors import (
    Conflict,
    Failed,
    Invalid,
    NotFound,
    NotReady,
    ServiceError,
    TooLarge,
)

__all__ = [
    "Conflict",
    "Failed",
    "Invalid",
    "NotFound",
    "NotReady",
    "ServiceError",
    "TooLarge",
    "WarlockService",
]
