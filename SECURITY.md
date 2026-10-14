# Security policy

## Reporting a vulnerability

**Please do not open a public issue for a security problem.** Use GitHub's
[private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
on this repository (Security -> Report a vulnerability), which opens a private
advisory only the maintainers can see.

Expect an acknowledgement within a week. There is no bounty; this is one
person's project.

## What is in scope

Realmspinner is an offline desktop application. It has no server and no account
system, and no inbound listener beyond three local kinds: the reconstruction
engine's loopback port (`127.0.0.1:17971` by default, `REALMSPINNER_TRELLIS_PORT`),
open only while `trellis-server.exe` is resident; Familiar's loopback ports, open only while
its `llama-server.exe` children are running (below); and the named pipe (a local
socket where there is no named pipe) the MCP bridge uses once you switch it on,
guarded by a token. So the realistic threat is **a malicious file**, not a
malicious peer. In scope:

- **Any file the app opens.** `.ora`, `.aseprite` (and its `.ase` spelling), `.tmx`/`.tsx` and their JSON
  spellings `.tmj`/`.tsj`, `.rmap`, `.rblk`, `.rpack`, `.rscn`, `.rsng`, `.glb`, and
  every image format Pillow handles. Also the import-only formats two parsers
  read but never write: Clay's mesh importer takes `.obj`, `.stl` and `.ply`
  (`kernels.mesh.meshimport.SUPPORTED_SUFFIXES`, alongside the `.glb` already
  named above), and Clay's Import Animation takes `.glb` only
  (`kernels.mesh.animimport`) -- the 2026-10-14 audit, finding docs-A-B-02:
  the list names only formats a door in the app can reach. The 2026-09-20
  audit, finding docs-01: an import-only parser is exactly as reachable by a crafted
  third-party file as a native document format, and the inventory above had
  only ever been swept from `service.files`'s save-format constants, which a
  read-only importer never appears in. Three more were missing for the same
  reason (the 2026-10-03 audit, finding docs-35): Sirens' sample import reads
  `.wav` (`kernels.audio.wavout.read_wav`), the palette importers read `.hex`,
  `.gpl`, `.pal` and `.txt` (`service.palettes.SUFFIXES`), and Inker opens the
  `.ase` suffix as well as `.aseprite`
  (`kernels.pixel.asein.ASEPRITE_SUFFIXES`).
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
  the same user on the same machine, so an agent can drive Clay. It is **off
  until switched on in Settings**, it is inbound only — it carries no model of its own,
  makes no outbound connection, and leaves `HF_HUB_OFFLINE` untouched. Its scope
  is one thing: an agent gets its own Clay tab and can address no other. The
  only doors out of that tab are reading a Library job's reference image by id
  (`clay_reference_add`) and `clay_export`'s one-shot mint of a Library row; it
  never reaches text-to-image or the reconstruction engine. (The character-pipeline tool surface that once sat beside it was
  removed on 2026-10-14; the 2026-10-14 audit, finding docs-A-B-01.) It is still an
  untrusted-input parser like any file format above: its framing
  (`src/realmspinner/mcp/protocol.py`) and its pipe (`pipe.py`) are in scope, and so
  is anything reachable through the derived Clay tool surface that escapes its
  own bound — Clay's one tab.
- **Familiar's local listeners.** `src/realmspinner/pipelines/llama.py` spawns
  `llama-server.exe`, a loopback HTTP listener distinct from the named pipe above,
  bound to `127.0.0.1` only (`REALMSPINNER_FAMILIAR_PORT`, 17972 by default). A second
  instance of the same binary serves the optional retrieval model (EmbeddingGemma 2),
  on the CPU, on its own loopback port (`REALMSPINNER_FAMILIAR_EMBED_PORT`, 17973 by
  default), with its own key file and its own port-owner claim. Both exist only while Familiar is in use —
  spawned on demand, not on startup (and the language-model one is stopped before any GPU job
  runs) — and both are started with `--offline`, so neither can fetch anything. Every spawn writes a
  fresh API key to a key file (`--api-key-file`, never on the command line, where
  any other process on the machine could read it) rather than reusing one across
  restarts. Familiar is not an egress path: the weights come through the
  `fetch_worker` above, and the three network exceptions stay three. In scope:
  anything reachable through those HTTP surfaces, and the key files' own handling.
- **Subprocess handling.** Heavy or privileged work is never done inline in the
  main process: every child process the app starts — reconstruction
  (`trellis-server.exe`), Familiar's `llama-server.exe` children, the
  text-to-image worker (`t2i_client`), the Blender worker, the matting worker,
  the music and stem-separation workers, LoRA training, `gltfpack`
  (`pipelines/optimize.py`), Flourish's text model (`recipe_worker`),
  `doctor`'s load probe, the fetch worker, the pack worker and the update
  worker — runs inside the `winjob` kill-on-close job, so a
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
