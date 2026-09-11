# Security policy

## Reporting a vulnerability

**Please do not open a public issue for a security problem.** Use GitHub's
[private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
on this repository (Security -> Report a vulnerability), which opens a private
advisory only the maintainers can see.

Expect an acknowledgement within a week. There is no bounty; this is one
person's project.

## What is in scope

Warlock Studio is an offline desktop application. It has no server, no account
system and no network listener beyond `127.0.0.1`, so the realistic threat is
**a malicious file**, not a malicious peer. In scope:

- **Any file the app opens.** `.ora`, `.aseprite`, `.tmx`/`.tsx` and their JSON
  spellings `.tmj`/`.tsj`, `.wmap`, `.wblk`, `.wpack`, `.wsng`, `.glb`, and
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
  `pack_worker` (`src/warlock/pipelines/pack_worker.py`, spawned by
  `src/warlock/service/packs.py`, downloading and installing optional
  dependency packs — Create/Muse/rigging extras — from Settings -> Packs), and
  the `update_worker` (`src/warlock/pipelines/update_worker.py`, reading the
  release feed and fetching an installer when the user asks Settings to check
  for an update).
- **The MCP agent bridge.** `src/warlock/mcp/` listens on a named pipe
  (Unix socket elsewhere) and accepts JSON-RPC from another process running as
  the same user on the same machine, so an agent can drive Clay. It is **off
  until switched on in Settings**, it is inbound only — no model, no inference,
  no socket, `HF_HUB_OFFLINE` untouched — and an agent gets its own Clay tab and
  can address no other. It is still an untrusted-input parser like any file
  format above: its framing (`src/warlock/mcp/protocol.py`) and its pipe
  (`pipe.py`) are in scope, and so is anything reachable through the derived
  tool surface that escapes that one tab.
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
- **`WARLOCK_*` environment variables.** They are configuration, set by whoever
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
