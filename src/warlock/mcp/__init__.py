"""MCP (Model Context Protocol) support for Warlock Studio.

**A leaf, on purpose.** Every module under here (`protocol.py`, `rpc.py`,
`pipe.py`, `bridge.py`) is stdlib-only and imports nothing else from
`warlock` -- `bridge.py`'s one exception, `warlock.config`, is display-free
and does not pull in pygame, moderngl or torch. That is what lets `warlock
mcp` run as a tiny child process on a machine with no GPU and no window,
exactly like `doctor` and `sweep`, and it is enforced the same way those
constraints usually are in this codebase: don't add an import here that
would break it.

The layering this buys:

- `protocol.py` speaks MCP over JSON-RPC 2.0 and knows nothing about Clay,
  Inker or any other mode -- it takes a tool list and a call function as
  arguments and has no opinion about what they do.
- `rpc.py` speaks a second, private wire format -- Warlock's own versioned
  RPC v1 -- over the same pipe; `Tool`, `ok`, `fail`, `text`, `image_png` and
  `MAX_FRAME` live here and `protocol.py` re-exports them, since both wire
  formats share that vocabulary. See its module docstring for the shape and
  the versioning rule.
- `pipe.py` is the transport: a named pipe (Windows) or a Unix socket
  (everywhere else) via `multiprocessing.connection`, guarded by a token so
  a stray local connection cannot drive the Studio.
- `bridge.py` is `warlock mcp` itself -- a dumb byte relay between an agent's
  stdio and that pipe. It contains no MCP semantics; all of those live in
  `protocol.py`, which runs inside the app (see `studio/agent_host.py`),
  because the app is the only side that can actually act on a tool call.
  `studio/agent_host.py` also sniffs a connection's first frame for `rpc.py`'s
  format and answers it directly -- `bridge.py` itself is
  unchanged and does not yet speak it.
"""

from __future__ import annotations
