# Security policy

## Reporting a vulnerability

**Please do not open a public issue for a security problem.** Use GitHub's
[private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
on this repository (Security -> Report a vulnerability), which opens a private
advisory only the maintainers can see.

Expect an acknowledgement within a week. There is no bounty; this is one
person's project.

## What is in scope

Realmspinner is an offline desktop application. It has no server, no account
system and no network listener beyond `127.0.0.1`, so the realistic threat is
**a malicious file**, not a malicious peer. In scope:

- **Any file the app opens.** `.ora`, `.aseprite`, `.tmx`/`.tsx` and their JSON
  spellings `.tmj`/`.tsj`, `.rmap`, `.rblk`, `.rpack`, `.rscn`, `.rsng`, `.glb`, and
  every image format Pillow handles.
  These are files people download from asset sites, so a crafted one reaching
  code execution, a decompression bomb, or a write outside the chosen directory
  is a real finding. So is a hang or an unbounded allocation.
- **Path traversal** through any archive member, external tileset reference or
  export template.
- **Anything that makes the app reach the network.** The offline guarantee
  (`HF_HUB_OFFLINE=1`, set before any import) is a security property here, not
  only a convenience. There are three user-initiated exceptions, each its own
  subprocess: the `fetch_worker` (model weights, from Hugging Face), the
  `pack_worker` (`src/realmspinner/pipelines/pack_worker.py`, spawned by
  `src/realmspinner/service/packs.py`, downloading and installing optional
  dependency packs — Create/Muse/rigging extras — from Settings -> Packs), and
  the `update_worker` (`src/realmspinner/pipelines/update_worker.py`, reading the
  release feed and fetching an installer when the user asks Settings to check
  for an update).
- **The MCP agent bridge.** `src/realmspinner/mcp/` listens on a named pipe
  (Unix socket elsewhere) and accepts JSON-RPC from another process running as
  the same user on the same machine, so an agent can drive Clay and, since the
  character pipeline landed, a second, narrower surface. It is **off
  until switched on in Settings**, it is inbound only — no model, no inference,
  no socket, `HF_HUB_OFFLINE` untouched. Scope is two-part: an agent gets its
  own Clay tab and can address no other; against the character pipeline it may
  read any Library row by job id, but write is additive only — new mesh, rig
  or charsheet rows minted through the same service doors a pane uses, derived
  artifacts, and copies into the configured export folder — never a rewrite of
  an existing row, never a path (every argument is an id or an enum), and it
  may cancel only the jobs it started on its own connection. Every file an
  agent exports is named from `characters.agent_export_stem` (the asset's own
  export name plus its job id, and a sheet id too where the format needs one)
  rather than the plain display name a pane's export uses, so an agent's
  export of a copy can only ever overwrite an earlier export *it* made of the
  same asset, never a human's export of a same-named one. It is still an
  untrusted-input parser like any file format above: its framing
  (`src/realmspinner/mcp/protocol.py`) and its pipe (`pipe.py`) are in scope, and so
  is anything reachable through either derived tool surface that escapes its
  own bound — Clay's one tab, or the character pipeline's read-any/write-
  additive-only rule.
- **Familiar's local listener.** `src/realmspinner/pipelines/llama.py` spawns
  `llama-server.exe`, a second loopback HTTP listener distinct from the named
  pipe above, bound to `127.0.0.1` only. It exists only while Familiar is in
  use — spawned on demand, not on startup — and every spawn writes a fresh
  API key to a key file (`--api-key-file`, never on the command line, where
  any other process on the machine could read it) rather than reusing one
  across restarts. In scope: anything reachable through that HTTP surface, and
  the key file's own handling.
- **Subprocess handling.** Heavy or privileged work is never done inline in the
  main process: reconstruction (`trellis-server.exe`), the Blender worker,
  the matting worker, the music and stem-separation workers, LoRA training,
  `doctor`'s load probe, the fetch worker, the pack worker and the update
  worker all run as a child process inside the `winjob` kill-on-close job, so a
  crash or a forced close of the main window cannot leave one running orphaned.

## What is not in scope

- **Model weights and what they generate.** The app runs whatever checkpoint you
  point it at, in-process, by design. A malicious `.safetensors` is a supply
  chain question about where you downloaded it from.
- **`REALMSPINNER_*` environment variables.** They are configuration, set by whoever
  is already running the process.
- Anything requiring an attacker who already has code execution as your user.
- The known, documented traversal allowance in `.tmx`/`.tsx` external
  references, and the same allowance in their `.tmj`/`.tsj` JSON spellings:
  Tiled's real folder layouts use `../`, so relative traversal is permitted
  deliberately while absolute and UNC paths are refused. It is a same-user
  read in an offline app; if you have a way to turn it into something more,
  that *is* in scope.

## Supported versions

The latest release only.
