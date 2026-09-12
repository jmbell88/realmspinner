# 2026-09-12 — clean-machine fetch pipeline and pack recovery paths

Continuation of the 2026-08-26/2026-09-05 clean-machine programme (P1 in
TODO.md), on the same second Windows PC (8 GiB card, below `vram.TRELLIS_GIB`
= 16.0). Warlock Studio 0.0.46.

## Fetch pipeline (P1.4)

- dinov2 (0.4 GB): downloaded, installed, recognized immediately.
- SDXL 1.0 + Hyper-SD: downloaded, installed, recognized immediately.
- Image Generation pack: installed on first prompt from Create, correctly.
- TRELLIS 2 GGUF: an existing `~/.warlock/models/trellis2-gguf` was copied in
  from the dev machine over USB; the app recognized it immediately but still
  required running the download (already-present files did not short-circuit
  install). The download was cancelled mid-flight — cancelled cleanly, no
  stuck state. Retried after a two-minute wait and completed successfully.
  F1/F2's fix holds outside the test suite: a cancelled fetch does not corrupt
  the retry.

## Engine launch (P1.5, partial)

- `warlock doctor` has no PATH entry on an installed machine; use the Start
  Menu shortcut "Warlock Doctor" or run `<InstallDir>\bin\warlock-doctor.cmd`
  directly. Ran clean: exit 0, engine row `[SETUP]`.
- Remove on the TRELLIS engine stopped the resident `trellis-server.exe` and
  deleted it successfully — no locked-DLL failure.
- Not reached on this machine: Create ungreying into a real generation and a
  textured GLB coming out the other end. The 8 GiB card is below
  `vram.TRELLIS_GIB` (16.0) and the VRAM door (`service.validation.check_vram`)
  correctly refuses before attempting reconstruction. This is the design
  working as intended, not a defect. The generation proof still needs a
  bigger card — the laptop (P1.7) or the dev machine.

## Pack recovery paths (P1.5's three recovery scenarios + P1.6)

All four run against a real pack (not a fake), on real hardware, for the
first time:

- **Cancel during download**: quit mid-download — cancelled cleanly on the
  worker's acknowledgement, no partial pack left registered as installed.
- **Quit during commit (pip install)**: held/handled correctly — no
  half-installed pack state observed.
- **Repair on a healthy pack**: reinstalled pinned wheels, came back green.
- **Repair on a hand-damaged pack** (deleted `.dist-info`/a top-level module):
  detected and repaired, came back green.
- **Upgrade over an existing install** (0.0.46 over itself, packs already
  installed): all previously-installed packs came back via the Restore packs
  banner, without incident.

All five outcomes were "as far as I can tell" / "as it should" — no error
banners, no manual cleanup required, packs immediately usable after each.

## What's still open in P1

- Actual end-to-end generation (image -> TRELLIS -> textured GLB) on hardware
  with >=16 GiB VRAM: the laptop, or this dev machine.
- The laptop entirely (P1.7) — GPU unknown.
