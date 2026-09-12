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
- `bridge.py` is `warlock mcp` itself -- the real MCP server. It speaks
  `rpc.py`'s private RPC v1 to Studio (`hello`, `catalogue`, `call`) and
  dual-era MCP (`protocol.bridge_dispatch`) to whatever client dialled its
  stdio: legacy, `initialize`-first JSON-RPC (with batching only for the one
  legacy revision that still had it) and a newer "modern" era that drops
  `initialize` for `server/discover` and versions each request through
  `params._meta`. Studio keeps all per-call state (dedup, replay,
  `warlock_status`, transcript, timeouts) behind the RPC v1 `call` op; this
  module never re-parses a tool result, only splices its raw bytes into
  whichever MCP envelope the connection's era calls for. `WARLOCK_MCP_RELAY=1`
  keeps the old dumb byte-relay behaviour available as an escape hatch.
  `studio/agent_host.py` still sniffs a connection's first frame to serve
  the old, unwrapped MCP path directly too, for a bridge that has not been
  updated to speak RPC v1 yet.
"""

from __future__ import annotations
