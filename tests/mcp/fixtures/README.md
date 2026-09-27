# MCP / ext-tasks schema fixtures

Both files here are trimmed copies of upstream JSON Schema documents, kept
offline so `tests/mcp/test_schema_conformance.py` never touches the network
(the whole suite runs offline by design -- see the root `CLAUDE.md`).

- `ext_tasks_2026-07-28.json` -- trimmed from
  https://raw.githubusercontent.com/modelcontextprotocol/ext-tasks/main/schema/2026-07-28/schema.json,
  fetched 2026-09-26. Covers `CreateTaskResult`, `GetTaskResult`,
  `CancelTaskResult`, the `Task`/`WorkingTask`/`CompletedTask`/
  `FailedTask`/`CancelledTask`/`InputRequiredTask` shapes they compose from,
  and the `Result`/`ResultMetaObject`/`Implementation`/`Icon`/`Error` shapes
  those in turn reference (this schema document defines its own copies of
  those core shapes rather than importing the core schema). Notably: a
  task's `taskId`/`status`/`createdAt`/`lastUpdatedAt`/`ttlMs` sit **flat**
  on the result object (not nested under a `task` key), and
  `createdAt`/`lastUpdatedAt` are both required ISO-8601 timestamp strings.
- `mcp_2026-07-28.json` -- trimmed from
  https://raw.githubusercontent.com/modelcontextprotocol/modelcontextprotocol/main/schema/2026-07-28/schema.json,
  fetched 2026-09-26. Covers `DiscoverResult`, `ServerCapabilities`,
  `Implementation`, `Result`, `ResultMetaObject`, `Error`, `JSONObject` and
  `Icon`. Notably: `ResultMetaObject`'s only reserved key is
  `io.modelcontextprotocol/serverInfo` (an `Implementation`, i.e.
  `{"name": ..., "version": ...}`) -- not a bare `serverInfo` key inside
  `_meta`.

Each file keeps only the `$defs` this repo's tests actually validate
against; everything else in the upstream schema (requests, notifications,
sampling/elicitation types the Tasks extension merely references) is
dropped, and each retained `$def`'s own `$ref`s were checked to still
resolve inside the trimmed file (see the file's own `$comment`). Re-fetch
both documents from their source URLs and re-run the trim if this repo ever
needs a newer protocol revision's shapes.
