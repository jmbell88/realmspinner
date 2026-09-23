"""Central configuration. Every path/port is env-overridable with REALMSPINNER_* vars."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The one blessed base download, and therefore what an unconfigured job runs.
# Lived in ``models.py`` until the 2026-09-17 restructure: which checkpoint
# the app defaults to when nobody names one is a configuration decision, not
# registry data, and leaving it in ``models`` was the one thing that made
# ``config.py`` -- layer 0, imported by nearly everything -- reach up into
# ``models`` -- layer 2, the weights registry -- for a single string.
#
# Not turbo, and measured rather than preferred: in the 2026-08-09 re-baseline
# `base_model=turbo` took 1 of 5 -- the weakest checkpoint that produced data --
# while `base_model=sdxl_cfg` took 3 of 3, the best survival rate of any arm
# with data. The two rank oppositely on refusals and on quality (both of that
# run's composition refusals fell on SDXL-family arms at full CFG), and this
# picks quality: a refusal is a sentence the user can act on, a poor mesh is two
# minutes of the serial worker spent on something they will discard.
#
# It is also the distribution decision. One 7 GiB download of
# stabilityai/stable-diffusion-xl-base-1.0 is shared by four recipes -- sdxl,
# sdxl_cfg, pixel, lightning -- so shipping this one entry unlocks the speed
# recipes for a small LoRA each, and pixel sheets for 0.2 GB more. Recorded in
# dev/measurements/2026-08-11-default-base-model.md.
#
# The key is only meaningful against ``models.BASE_MODELS``, which this module
# may not import (that would just move the layer violation rather than close
# it) -- callers that need the full registry entry do
# ``models.BASE_MODELS[config.DEFAULT_BASE_MODEL]`` themselves.
DEFAULT_BASE_MODEL = "sdxl_cfg"


def source_checkout() -> bool:
    """Whether this package is running from a source checkout.

    Realmspinner requires a checkout-shaped runtime, produced by either a source
    clone or the Windows installer (DST-01, D4). Every native default below resolves against
    ``PROJECT_ROOT`` -- ``Path(__file__).parents[2]`` -- which is the repository
    root for an editable install and, inside a wheel, points somewhere under the
    environment (``Lib/vendor``). The wheel carries the package, the manual and
    the changelog but *not* ``vendor/``, which is git-ignored in full: the
    binaries are one-time manual downloads. So a non-editable install can never
    find the reconstruction engine, and the failure would arrive as a missing
    file at the first job rather than as anything anyone could act on.

    Supporting wheels properly means installing pinned native assets under a
    versioned app-data location or shipping them as package resources resolved
    through ``importlib.resources``. That is a distribution project, not a
    bugfix, and nothing asks for it today -- so the supported model is stated
    and checked instead of half-implemented.

    The probe is the runtime-layout marker: ``pyproject.toml`` beside the
    package's parent. Deliberately not "is vendor/ present" -- a layout
    that has not downloaded the binaries yet is a *supported* state with its own
    Doctor rows, and conflating the two would refuse to start over a missing
    optional file.
    """
    return (PROJECT_ROOT / "pyproject.toml").is_file()

#: The engine binary's filename, spelled once because three things resolve it:
#: the download's presence probe, ``Config.resolve_trellis_exe`` and doctor.
TRELLIS_SERVER_NAME = "trellis-server.exe"

# None on purpose, and confirmed by measurement -- see Config.trellis_band.
DEFAULT_TRELLIS_BAND: int | None = None


def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, default)).resolve()


def _env_opt_path(name: str) -> Path | None:
    """A path variable with no default, so "unset" is answerable.

    ``_env_path`` cannot express that: it needs a default, and a default is
    exactly what an *override* does not have. ``REALMSPINNER_TRELLIS_EXE`` names a
    copy of the engine the user is pointing at by hand, and the difference
    between "pointing at one" and "not" is what ``Config.resolve_trellis_exe``
    branches on. An empty or whitespace-only value reads as unset, matching
    ``_env_opt_float``: a variable set to nothing is a variable somebody meant
    to clear.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    return Path(raw.strip()).resolve()


def _home() -> Path:
    """The one directory the app owns on this machine.

    A module-level helper rather than a field other factories read, because a
    ``default_factory`` lambda cannot see its siblings -- every root below
    resolves this independently, and ``REALMSPINNER_HOME`` therefore moves all of
    them at once while each per-root variable still wins over it.
    """
    return _env_path("REALMSPINNER_HOME", Path.home() / ".realmspinner")


# Every environment variable this process could not make sense of, as
# ``(name, raw value, what was expected)``. Collected rather than raised, and
# this is the whole of RUN-03's fix.
#
# The failure it replaces: several fields called bare ``int()``/``float()``
# inside their ``default_factory``, so one typo in a port, a timeout or a
# threshold raised out of ``get_config()`` -- which runs before ``studio.main``
# has a log handler, before Doctor exists, and before there is a window to say
# anything in. The user got a traceback naming ``int()`` and no indication of
# which variable was wrong. Meanwhile *other* fields (the optional VRAM floats)
# silently turned a malformed value into ``None``, so an explicit safety limit
# with a typo in it became "unset" -- the two halves of one policy disagreeing
# about whether bad input is fatal or invisible.
#
# So: malformed input never raises and never vanishes. It falls back to the
# documented default and is recorded here, and Doctor reports every one of them
# in a single actionable row.
INVALID_ENV: list[tuple[str, str, str]] = []


def _note_invalid(name: str, raw: str, expected: str) -> None:
    entry = (name, raw, expected)
    if entry not in INVALID_ENV:
        INVALID_ENV.append(entry)


def _env_int(name: str, default: int) -> int:
    """``int`` from the environment, or the default with a note. Never raises."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        _note_invalid(name, raw, "a whole number")
        return default


def _env_float(name: str, default: float) -> float:
    """``float`` from the environment, or the default with a note. Never raises."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError:
        _note_invalid(name, raw, "a number")
        return default


def _env_opt_int(name: str, default: int | None) -> int | None:
    """Like int(os.environ[name]) but with an explicit "leave it to the exe" value.

    Empty or "auto" means None, i.e. omit the flag entirely rather than passing
    a number, so trellis-server.exe applies its own heuristic.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    text = raw.strip().lower()
    if text in ("", "auto"):
        return None
    try:
        return int(text)
    except ValueError:
        # Recorded and defaulted, like every other numeric field. This used to
        # raise out of ``get_config()`` -- before any log handler, Doctor or
        # window existed -- over a typo in a texture resolution (RUN-03).
        _note_invalid(name, raw, "a whole number, or 'auto'")
        return default


def _env_bool(name: str, default: bool) -> bool:
    """``bool`` from the environment, or the default with a note. Never raises.

    One vocabulary for every boolean variable -- this used to differ per field,
    so ``REALMSPINNER_TRELLIS_WEBP=yes`` silently meant *off*.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    text = raw.strip().lower()
    if text in ("1", "true", "on", "yes"):
        return True
    if text in ("0", "false", "off", "no"):
        return False
    _note_invalid(name, raw, "1/true/on/yes or 0/false/off/no")
    return default


def _env_opt_bool(name: str) -> bool | None:
    """A tri-state flag: unset (None) is distinguishable from an explicit off.

    Auto-detection has to be able to tell "the user chose coexist" from "the
    user chose nothing", and a bool with a default cannot.
    """
    raw = os.environ.get(name)
    if raw is None:
        return None
    text = raw.strip().lower()
    if text == "":
        return None
    if text in ("1", "true", "on", "yes"):
        return True
    if text in ("0", "false", "off", "no"):
        return False
    # Recorded and defaulted like every other field (RUN-03): a typo used to
    # read as an *explicit* "coexist", which suppressed ``vram.plan``'s
    # auto-detection on exactly the switch whose wrong value OOMs a small card.
    _note_invalid(name, raw, "1/true/on/yes or 0/false/off/no")
    return None


def _env_opt_float(name: str) -> float | None:
    """An optional float: unset is None, and so is a malformed value.

    ``None`` is still the fallback -- these are the VRAM limits, where "unset"
    genuinely is the default and there is no better number to substitute. What
    changed is that a malformed one is no longer *silent*: an explicit safety
    limit with a typo in it used to become "unset" with nothing said anywhere,
    which is the same policy inconsistency RUN-03 names from the other side.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        return float(raw.strip())
    except ValueError:
        _note_invalid(name, raw, "a number")
        return None


@dataclass(slots=True)
class Config:
    # Everything the app generates lives under here -- assets, bench runs,
    # palettes and the model weights. A user's work is not a part of the source
    # tree: it has to survive a reinstall, a second checkout and a `git clean`,
    # and it has to be findable by somebody who never cloned the repo. The one
    # exception is the vendored native binaries (gltfpack.exe, realmspinnerc.dll),
    # which ship *with* the checkout and so stay repo-relative below. The
    # reconstruction engine used to be the third of them and is not any more:
    # it is 838 MB, it is downloaded from Settings -> Models like a model, and
    # ``resolve_trellis_exe`` below is where the checkout and the download meet.
    home: Path = field(default_factory=_home)
    data_dir: Path = field(
        default_factory=lambda: _env_path("REALMSPINNER_DATA_DIR", _home() / "assets")
    )
    # Configurable apart from data_dir on purpose, and documented as such in
    # docs/manual/41-configuration.md: the library and its index are two
    # separate choices (an SSD index over a spinning-disk library is the case).
    db_path: Path = field(
        default_factory=lambda: _env_path("REALMSPINNER_DB", _home() / "assets" / "jobs.sqlite")
    )
    # **An override, not a location.** ``None`` means "nobody has said where
    # the engine is", which is the ordinary state: ``resolve_trellis_exe``
    # then looks in the downloaded runtime directory and falls back to the
    # checkout's ``vendor/``. Read this field only to ask whether the user
    # pointed somewhere by hand; every consumer that wants the *path* calls
    # the method, because the answer changes the moment a download lands and a
    # value resolved at startup would still name the empty vendor directory an
    # hour later.
    trellis_server_exe: Path | None = field(
        default_factory=lambda: _env_opt_path("REALMSPINNER_TRELLIS_EXE")
    )
    # Optional: a project folder assets can be copied straight into (e.g. a
    # Godot project's assets/). Unset means the feature is off and its routes
    # 404 -- writing outside data_dir is opt-in, never a default.
    export_dir: Path | None = field(
        default_factory=lambda: (
            _env_path("REALMSPINNER_EXPORT_DIR", PROJECT_ROOT)
            if os.environ.get("REALMSPINNER_EXPORT_DIR")
            else None
        )
    )
    # Vendored like trellis-server.exe: a pinned native binary, never downloaded
    # at runtime. Missing it costs you the triangle budgets, not the app --
    # jobs then ship the reconstruction as the engine returned it, which is
    # what they did before.
    gltfpack_exe: Path = field(
        default_factory=lambda: _env_path(
            "REALMSPINNER_GLTFPACK", PROJECT_ROOT / "vendor" / "gltfpack" / "gltfpack.exe"
        )
    )
    # Default triangle profile for a new job. See pipelines/optimize.PROFILES.
    #
    # "standard" (50,000 triangles), not "raw", since 2026-09-23
    # (dev/measurements/2026-09-23-default-mesh-budget.md): every generated
    # mesh landed at ~267k-298k faces -- trellis-server's own quadric simplify
    # to ~300k, with no second pass -- which is a source mesh, not a game
    # asset, and the complaint that triggered this change said so in as many
    # words. The per-tier corpus qualification that used to hold every named
    # tier out of the generate forms never produced a usable result (0 of 20
    # accepted on 2026-08-13), so it is retired in favour of a check on every
    # job instead of a sample of three: `tiercheck.compare` now guards every
    # `pipelines.optimize.run` pass and refuses to publish a tier that dropped
    # UVs, a material assignment or a PBR texture. `resolve_profile`
    # (service/_jobs_create.py) downgrades a *defaulted* tier to raw when
    # gltfpack is not installed, rather than refusing the job outright.
    # **What is still unguarded is silhouette damage** -- no number here
    # scores a mangled blade or a lost finger; the eye catches that, and
    # Retarget -> Raw rebuilds from source.glb, which is never touched. Set
    # REALMSPINNER_MESH_PROFILE=raw to go back to the old default.
    mesh_profile: str = field(
        default_factory=lambda: os.environ.get("REALMSPINNER_MESH_PROFILE", "standard")
    )
    # The game-ready remesh's default triangle budget, when nothing asks for a
    # gltfpack tier by name and Blender is on this machine
    # (dev/measurements/2026-09-23-default-mesh-budget.md).
    # ``service._jobs_create.resolve_lowpoly`` is the door that reads this;
    # 5000 is the measured pick (4,996 triangles achieved on the raccoon, good
    # fidelity, ~6 s). 0 turns the remesh off -- a model job then falls back to
    # ``mesh_profile``'s gltfpack tier exactly as it did before this existed.
    lowpoly_triangles: int = field(
        default_factory=lambda: _env_int("REALMSPINNER_LOWPOLY_TRIANGLES", 5000)
    )
    # Whether a finished reference is scored against the run's conditioning
    # reference -- ref.png, which is the active profile's style anchor when one
    # is set and otherwise whatever image the user attached -- as well as
    # against its own composition report. On by default: the composition half
    # costs nothing (the report is already measured) and the anchor half only
    # runs when there is a ref.png and DINOv2 is on disk.
    rank_candidates: bool = field(
        default_factory=lambda: os.environ.get("REALMSPINNER_RANK", "on").lower()
        not in ("0", "false", "off", "no")
    )
    # How many extra times a text job may redraw its reference when the
    # composition report refuses the one it just drew.
    #
    # Was 0 -- off -- on the reasoning that a retry is four seconds of GPU
    # nobody asked for and the report's rules are heuristics, so a refusal is a
    # strong hint rather than a fact. Both halves are still true and neither is
    # what the setting costs. The 2026-08-07 rogue sweep refused 17 of 100
    # units at the composition gate, and a model-stage refusal is not a hint at
    # all there: the job *fails*. Seed 11's baseline was one of them, and losing
    # it did not cost one mesh -- ``findings.comparisons`` pairs rows sharing a
    # sweep, a source and a seed, so that baseline was one side of nine
    # prospective pairs and every one went with it.
    #
    # 2, and the reroll holds ``mesh_seed`` fixed and moves only
    # the reference seed, which is the point: a unit's identity in the corpus
    # is its config vector and its mesh seed, and which of three reference
    # draws avoided a character sheet is not a setting anyone is measuring.
    # The ceiling still ends the loop rather than a verdict, so past it the
    # user gets the last draw and the model stage gets its ordinary refusal.
    reference_retries: int = field(
        default_factory=lambda: max(0, _env_int("REALMSPINNER_REFERENCE_RETRIES", 2))
    )
    # How many extra times the trellis stage may run when the finished mesh
    # audits worse than mesh_hole_max. 0 -- off -- because a retry is two
    # minutes of GPU the user did not ask for, and a remesh is a reroll rather
    # than a repair: the second reconstruction can perfectly well be worse than
    # the first, which is why the loop keeps whichever attempt measured best
    # rather than whichever came last. Graded on 2026-09-02
    # (dev/measurements/2026-09-02-hole-audit-vs-grade.md): the loop was run
    # for real on the five meshes the trigger still fires on under v0.6.0 and
    # the reviewer graded the kept attempts 0 of 5 usable -- the trigger sees
    # perforated skins, which that release fixed, and what remains is open
    # forms a reroll cannot close.
    mesh_retries: int = field(
        default_factory=lambda: max(0, _env_int("REALMSPINNER_MESH_RETRIES", 0))
    )
    # The worst-view see-through fraction past which a mesh is worth redoing.
    # 0.07 is measured, not guessed, the same way trellis_band was settled by a
    # sweep: dev/measurements/2026-08-04-hole-rate-baseline.md ran the whole
    # core-v1 suite at two seeds and found the hole rate sharply bimodal, with
    # *nothing at all* between 0.0308 and 0.1010 -- 22 of 37 meshes below the
    # gap and 15 above it, out to 0.556. 0.07 is the midpoint of that empty gap
    # rather than a percentile, so it is the value furthest from either cluster
    # and the one least disturbed by a future sample shifting one of them; any
    # threshold inside the gap selects the identical fifteen meshes.
    mesh_hole_max: float = field(
        default_factory=lambda: _env_float("REALMSPINNER_MESH_HOLE_MAX", 0.07)
    )
    # Where `python -m realmspinner.bench` writes its runs. A sibling of data_dir
    # rather than a child, on purpose: a benchmark run copies its artifacts
    # rather than referencing them, precisely so it survives prune_jobs -- and
    # a prune (or a Clean) that walked the library would collect them if they
    # lived inside it.
    bench_dir: Path = field(
        default_factory=lambda: _env_path("REALMSPINNER_BENCH_DIR", _home() / "bench")
    )
    # Where a bulk delete puts the evidence before it removes the assets. A
    # sibling of data_dir for bench_dir's reason exactly, and the reason it
    # exists at all is measured: on 2026-09-07 the mesh probe found 0 of 9
    # model-stage verdicts still carrying a source.glb, because finishing a
    # blind grading pass is also what deletes the meshes it graded
    # (dev/measurements/2026-09-07-mesh-probe-preregistration.md). Retention
    # keeps an accept *in place*; this keeps everything that was judged
    # *somewhere*.
    evidence_dir: Path = field(
        default_factory=lambda: _env_path("REALMSPINNER_EVIDENCE_DIR", _home() / "evidence")
    )
    # Where pixel-art palette files live (.hex from Lospec, .gpl from GIMP).
    # Ships empty: a palette is the user's own art direction, and a bundled one
    # would be a default nobody chose. Absent or empty simply means the palette
    # control offers nothing, never an error -- the same rule every optional
    # model directory follows.
    palette_dir: Path = field(
        default_factory=lambda: _env_path("REALMSPINNER_PALETTE_DIR", _home() / "palettes")
    )
    trellis_models_dir: Path = field(
        default_factory=lambda: _env_path(
            "REALMSPINNER_TRELLIS_MODELS", _home() / "models" / "trellis2-gguf"
        )
    )
    # The engine's own binaries -- trellis-server.exe, ggml, and the three
    # NVIDIA CUDA redistributables -- once they have been downloaded.
    #
    # Under ``engine/`` rather than ``models/`` because it is not a model and
    # nothing about it is loadable: ``models/`` is walked by the disk-usage
    # report, by the sweeps and by ``verify_all``, all of which reason about
    # weights. It is under the *home* rather than the install root for the
    # reason every other download is -- it has to survive a reinstall, and an
    # upgrade that replaced the app runtime would otherwise cost 838 MB again.
    trellis_runtime_dir: Path = field(
        default_factory=lambda: _env_path(
            "REALMSPINNER_TRELLIS_RUNTIME", _home() / "engine" / "trellis"
        )
    )
    trellis_port: int = field(
        default_factory=lambda: _env_int("REALMSPINNER_TRELLIS_PORT", 17971)
    )
    # Seconds of queue inactivity before the trellis server is stopped to free VRAM.
    trellis_idle_timeout: float = field(
        default_factory=lambda: _env_float("REALMSPINNER_TRELLIS_IDLE", 600.0)
    )
    # Familiar's own two directories, deliberately never reused from trellis'
    # -- llama.cpp and trellis.cpp ship their own, differently built
    # ``ggml*.dll`` (see fetch.py's ``familiar_runtime_dir``), so publishing
    # one engine's binaries into the other's directory would silently mix DLL
    # builds the moment both engines are installed.
    familiar_runtime_dir: Path = field(
        default_factory=lambda: _env_path(
            "REALMSPINNER_FAMILIAR_RUNTIME", _home() / "engine" / "llama"
        )
    )
    familiar_models_dir: Path = field(
        default_factory=lambda: _env_path(
            "REALMSPINNER_FAMILIAR_MODELS", _home() / "models" / "familiar"
        )
    )
    familiar_port: int = field(
        default_factory=lambda: _env_int("REALMSPINNER_FAMILIAR_PORT", 17972)
    )
    # Seconds of inactivity before the Familiar child is stopped to free VRAM.
    familiar_idle_timeout: float = field(
        default_factory=lambda: _env_float("REALMSPINNER_FAMILIAR_IDLE", 300.0)
    )
    # Where every image model lives: models.BASE_MODELS[k].dir_name resolves
    # against this, and style LoRAs against its loras/ subdirectory. All
    # downloaded once by hand (see README) -- the app never downloads, and
    # loads are local_files_only.
    t2i_model_root: Path = field(
        default_factory=lambda: _env_path("REALMSPINNER_T2I_ROOT", _home() / "models")
    )
    # Pre-registry override, still honoured: points the *turbo* entry at an
    # arbitrary local diffusers dir so existing setups keep working. Other base
    # models always resolve under t2i_model_root.
    t2i_turbo_dir: Path | None = field(
        default_factory=lambda: (
            _env_path("REALMSPINNER_T2I_DIR", PROJECT_ROOT)
            if os.environ.get("REALMSPINNER_T2I_DIR")
            else None
        )
    )
    # Base model used when a job doesn't name one. Sampler settings are not
    # configurable here on purpose -- they belong to the checkpoint (models.py).
    t2i_model: str = field(
        default_factory=lambda: os.environ.get(
            "REALMSPINNER_T2I_MODEL", DEFAULT_BASE_MODEL
        )
    )
    # trellis-server.exe's WebP textures declare EXT_texture_webp as required,
    # which Godot's glTF importer does not implement (it refuses the file
    # rather than skip the extension). Off is the correct default.
    trellis_webp: bool = field(
        default_factory=lambda: _env_bool("REALMSPINNER_TRELLIS_WEBP", False)
    )
    # trellis-server.exe's "auto" texture PBR resolution bakes visible per-texel
    # noise into the baseColor atlas at --res 1024/1536 (reproduced via
    # trellis-cli.exe: default auto is noise, explicit --tex-res 512 is clean).
    # Pin it to 512 until upstream fixes the auto heuristic.
    trellis_tex_res: int = field(
        default_factory=lambda: _env_int("REALMSPINNER_TRELLIS_TEX_RES", 512)
    )
    # Width of the narrow band the DC remesh runs over. The exe defaults it to
    # res/512 when the flag is absent, which is what None gives you.
    #
    # Measured 2026-08-01 and written up in
    # dev/measurements/2026-08-01-trellis-band.md (the document was backfilled
    # on 2026-09-11 after an audit found this table living only here, in the one
    # constant CLAUDE.md names as the example of a corpus-keyed one).
    # (`realmspinner sweep`, one reference image, seed 42, res 1024,
    # hole_fraction @ 1024) -- worst-view see-through fraction:
    #
    #     auto  0.0077   267,360 faces   123 s   (res/512 == band 2 here)
    #     2     0.0077   266,632 faces   143 s
    #     4     0.0167   290,774 faces   124 s
    #     8     0.0110   297,898 faces   136 s
    #     16    0.0125   289,586 faces   193 s
    #
    # Two conclusions, both against the earlier guess that a wider band would
    # help: the heuristic is already the best of the ladder, and widening makes
    # the surface *more* perforated while adding faces and time. So the flag
    # stays off. The run also puts a floor under what counts as a real
    # difference: auto and 2 are the same setting, and still disagreed by 728
    # faces, so anything under ~0.3% is noise.
    #
    # This supersedes an earlier note claiming the default left "~1300
    # disconnected plates, 7-31% of the silhouette". Nothing in this sweep
    # measured worse than 1.7%, so whatever produced those numbers was not
    # this exe at these settings. Re-measure before acting on that claim.
    trellis_band: int | None = field(
        default_factory=lambda: _env_opt_int("REALMSPINNER_TRELLIS_BAND", DEFAULT_TRELLIS_BAND)
    )
    # The three launch flags the exe accepts that Realmspinner never passed until
    # 2026-09-02: --gss / --gsh (guidance strengths for the sparse-structure
    # and structured-latent stages) and --max-tokens (the high-resolution token
    # budget). None omits the flag, so the exe's own default runs -- and that
    # default is *not printed by --help*, so the first rung of any sweep over
    # these must be "omitted", never a number copied from a guess. They exist
    # as sweep axes for the props-v1 hole question (dev/measurements/
    # 2026-08-30-sdxl-cfg-props.md); a winning rung becomes a default here by
    # a measurement doc, not before.
    trellis_gss: float | None = field(
        default_factory=lambda: _env_opt_float("REALMSPINNER_TRELLIS_GSS")
    )
    trellis_gsh: float | None = field(
        default_factory=lambda: _env_opt_float("REALMSPINNER_TRELLIS_GSH")
    )
    trellis_max_tokens: int | None = field(
        default_factory=lambda: _env_opt_int("REALMSPINNER_TRELLIS_MAX_TOKENS", None)
    )
    # Two more the exe accepts and Realmspinner never passed until 2026-09-03, and
    # the two that decide how much of the reconstruction survives into
    # source.glb. --decim: the exe quadric-simplifies every mesh to 300K faces
    # at res 1024 (150K at 512) *before* writing the GLB, so "raw" has never
    # been the untouched reconstruction -- a res-1024 run is ~15M faces before
    # that pass (tests/fixtures/trellis_1024_v060.log). ``0`` turns the pass
    # off; a positive value selects the legacy cluster-grid decimation at that
    # grid. --atlas: the UV atlas edge in px (the exe defaults 2048 at res
    # 1024, 1024 at 512), the texture-detail twin of trellis_tex_res. None
    # omits the flag. Both are sweep axes for dev/measurements/
    # 2026-09-03-trellis-detail-sweep.md; a default moves by that document.
    trellis_decim: int | None = field(
        default_factory=lambda: _env_opt_int("REALMSPINNER_TRELLIS_DECIM", None)
    )
    trellis_atlas: int | None = field(
        default_factory=lambda: _env_opt_int("REALMSPINNER_TRELLIS_ATLAS", None)
    )
    # Tri-state, and the None is the point. Unset means "decide from the card":
    # studio.runtime resolves it through vram.plan() at startup and writes a
    # plain bool back here before the Worker exists, so queue.py keeps reading
    # a bool and the "read once at startup" contract is unchanged. Set
    # explicitly (1/true/on or 0/false/off) it is honoured verbatim and
    # auto-detection never overrides it.
    #
    # False (coexist) keeps an SDXL-class pipe (~7 GB) and trellis-server (~16 GB)
    # resident together, which is right on a 32 GB card and is what a 32 GB
    # card still auto-selects. True restores the stop-trellis -> run-SDXL ->
    # unload -> restart handoff for OOM situations (resolution 1536, smaller
    # GPUs, a resident Flux).
    # Run the image pipeline in *this* process instead of a child. The escape
    # hatch, off by default, for the one situation the child makes harder:
    # attaching a debugger to the sampling loop, or reading a torch traceback
    # without the marker framing in the way.
    #
    # It is not a performance switch. The child costs one process spawn per
    # unload cycle and buys back what measurement showed the in-process loader
    # cannot give: klein charged +21.1 GiB of host commit on 2026-08-22 and
    # `unload()` returned 0.1 GiB of it, because the allocator's arenas outlive
    # every reference. Turning this on restores that leak
    # (dev/measurements/2026-08-22-trampoline-child-pids.md).
    t2i_in_process: bool = field(
        default_factory=lambda: _env_bool("REALMSPINNER_T2I_IN_PROCESS", False)
    )
    vram_exclusive: bool | None = field(
        default_factory=lambda: _env_opt_bool("REALMSPINNER_VRAM_EXCLUSIVE")
    )
    # Whether the value above came from the environment. Captured because the
    # resolve destroys the evidence: writing a plain bool back over the
    # tri-state makes "was this chosen or configured" underivable, so every
    # later plan() -- every health poll -- reported an auto-selected mode as
    # set by an environment variable nobody set. None means "not resolved yet",
    # i.e. infer it from vram_exclusive.
    vram_exclusive_explicit: bool | None = None
    # What one job may ask the card for, in GiB. None = device total minus
    # vram.HEADROOM_GIB, which is what you want unless the driver misreports.
    vram_budget_gib: float | None = field(
        default_factory=lambda: _env_opt_float("REALMSPINNER_VRAM_BUDGET")
    )
    # Pretend the card is this large. The escape hatch for a torch-less install
    # -- and what makes the whole admission gate testable without a GPU.
    vram_total_gib: float | None = field(
        default_factory=lambda: _env_opt_float("REALMSPINNER_VRAM_TOTAL")
    )
    # Whether a rig may read its joint positions off the reference image the
    # mesh was reconstructed from (pipelines/pose2d) instead of scaling the
    # template onto the bounding box.
    #
    # On, and it is a kill-switch rather than an opt-in because the thing it
    # switches off is already the fallback: with no weights on disk, a
    # non-humanoid template, no reference image, or a detection that fails any
    # sanity gate, a rig is byte-for-byte what it was before this existed. So
    # there is no state in which turning it *on* is a risk the user should have
    # to accept deliberately -- but a wrong skeleton is hard to see and easy to
    # blame on something else, so there has to be one flag that takes the whole
    # feature out of the picture while it is being judged.
    pose_fit: bool = field(
        default_factory=lambda: os.environ.get("REALMSPINNER_POSE_FIT", "on").lower()
        not in ("0", "false", "off", "no")
    )
    # Skeleton template a rig request falls back to when it doesn't name one.
    # Validated against templates.templates() at request time, not here -- config
    # is imported by everything and must not pull the template registry in.
    rig_template: str = field(
        default_factory=lambda: os.environ.get("REALMSPINNER_RIG_TEMPLATE", "humanoid")
    )
    # Wall-clock ceiling for one Blender subprocess. Automatic weights on a
    # 300k-face mesh are minutes of CPU; anything past this is a hang, and a
    # hung bpy holds the single-worker queue against every other job.
    rig_timeout: float = field(
        default_factory=lambda: _env_float("REALMSPINNER_RIG_TIMEOUT", 1800.0)
    )
    # Baking a pose is an import, a handful of quaternion assignments and an
    # export -- seconds, not minutes. It gets its own much tighter ceiling
    # because it runs inline in a request rather than on the queue.
    pose_timeout: float = field(
        default_factory=lambda: _env_float("REALMSPINNER_POSE_TIMEOUT", 300.0)
    )
    # Render the deformation battery after a rig. A kill-switch rather than an
    # opt-in, for the reason pose_fit is one: with it off a rig is byte for
    # byte what it was before this existed, so there is no state in which
    # turning it *on* is a risk to accept deliberately -- but it is a dozen
    # extra EEVEE frames on the serial queue, and a user who does not review
    # rigs should be able to stop paying for them.
    deform_qa: bool = field(
        default_factory=lambda: os.environ.get("REALMSPINNER_DEFORM_QA", "on").lower()
        not in ("0", "false", "off", "no")
    )
    # A sheet is one EEVEE render per cell -- 8 yaws times however many poses.
    # Generous because the cell count is user-chosen, but still bounded: this
    # runs on the serial queue and a hang would block every later job.
    sheet_timeout: float = field(
        default_factory=lambda: _env_float("REALMSPINNER_SHEET_TIMEOUT", 1800.0)
    )
    # Wall-clock ceiling for the stem-separation subprocess (``_q_music._separate``).
    # The 2026-09-08 audit, finding muse-02: this job used to borrow
    # ``pose_timeout`` (300s), a ceiling ``pose_timeout``'s own docstring says
    # is sized for an inline bake that runs in seconds, not minutes -- but
    # separation is a *queued* job that can process up to a 600-second take
    # (``service._jobs_music.MAX_DURATION``) in 10-second chunks and explicitly
    # falls back to CPU with no card present, so a legitimately-progressing
    # job on a long take or a CPU-only host was killed exactly like a hung
    # one. Sized like ``rig_timeout``/``sheet_timeout`` -- generous because a
    # full-length take on CPU is genuinely slow, but still bounded because this
    # runs on the serial queue and a hang would block every later job.
    separation_timeout: float = field(
        default_factory=lambda: _env_float("REALMSPINNER_SEPARATION_TIMEOUT", 1800.0)
    )

    def resolve_trellis_exe(self) -> Path:
        """Where ``trellis-server.exe`` actually is, asked fresh every time.

        Three places, in this order, and the order is the design:

        1. ``REALMSPINNER_TRELLIS_EXE``, if the user set it. An explicit answer wins
           over both discovered ones -- it is the sideload path for a machine
           that cannot reach GitHub, and the way a developer runs against a
           build of their own.
        2. ``trellis_runtime_dir``, the downloaded engine. Probed rather than
           assumed, because the row can be removed again from Settings.
        3. ``vendor/trellis/`` in the checkout, which is where a developer's
           copy has always lived and still does. The installer no longer stages
           it, so on a packaged install this is a directory that does not
           exist -- and a path that does not exist is the right answer to
           return, because ``doctor`` and the mode gate both report *that*
           rather than guessing.

        **Called at use, never cached.** ``Config`` is built once at startup and
        the engine is a download, so a value resolved into a field would name
        the empty vendor path for the whole session after a successful install
        -- the "restart before it can use it" failure the pack landing exists
        to avoid, arrived at for free by asking the filesystem instead.
        """
        if self.trellis_server_exe is not None:
            return self.trellis_server_exe
        downloaded = self.trellis_runtime_dir / TRELLIS_SERVER_NAME
        if downloaded.is_file():
            return downloaded
        return PROJECT_ROOT / "vendor" / "trellis" / TRELLIS_SERVER_NAME

    @property
    def autosave_dir(self) -> Path:
        """Where Inker's crash-safety copies live.

        Under ``data_dir`` rather than beside each document: a document may
        have no path at all, and one that does may sit on a network share or a
        read-only directory -- neither of which is a good place to be writing
        every two minutes. One directory also means recovery is a listing
        rather than a search.
        """
        return self.data_dir / "autosave"

    def job_dir(self, job_id: str) -> Path:
        return self.data_dir / job_id


# Which environment variable each field answers to (S140). A table rather than
# something derived, because the derivation does not exist: every default is a
# ``default_factory`` lambda closing over its own variable name, and there is no
# way to ask a dataclass field which string its factory read.
#
# The pairing is asserted in both directions by a test -- a field with no entry
# and an entry naming no field both fail -- which is what keeps this from
# becoming the usual stale second copy. ``vram_exclusive_explicit`` is the one
# deliberate omission: it is not configured, it is *derived* by the startup
# resolve, and listing it beside real settings would present a computed answer
# as something the user set.
SETTINGS: tuple[tuple[str, str], ...] = (
    ("home", "REALMSPINNER_HOME"),
    ("data_dir", "REALMSPINNER_DATA_DIR"),
    ("db_path", "REALMSPINNER_DB"),
    ("bench_dir", "REALMSPINNER_BENCH_DIR"),
    ("evidence_dir", "REALMSPINNER_EVIDENCE_DIR"),
    ("palette_dir", "REALMSPINNER_PALETTE_DIR"),
    ("export_dir", "REALMSPINNER_EXPORT_DIR"),
    ("trellis_server_exe", "REALMSPINNER_TRELLIS_EXE"),
    ("trellis_models_dir", "REALMSPINNER_TRELLIS_MODELS"),
    ("trellis_runtime_dir", "REALMSPINNER_TRELLIS_RUNTIME"),
    ("trellis_port", "REALMSPINNER_TRELLIS_PORT"),
    ("trellis_idle_timeout", "REALMSPINNER_TRELLIS_IDLE"),
    ("trellis_webp", "REALMSPINNER_TRELLIS_WEBP"),
    ("trellis_tex_res", "REALMSPINNER_TRELLIS_TEX_RES"),
    ("trellis_band", "REALMSPINNER_TRELLIS_BAND"),
    ("trellis_gss", "REALMSPINNER_TRELLIS_GSS"),
    ("trellis_gsh", "REALMSPINNER_TRELLIS_GSH"),
    ("trellis_max_tokens", "REALMSPINNER_TRELLIS_MAX_TOKENS"),
    ("trellis_decim", "REALMSPINNER_TRELLIS_DECIM"),
    ("trellis_atlas", "REALMSPINNER_TRELLIS_ATLAS"),
    ("familiar_runtime_dir", "REALMSPINNER_FAMILIAR_RUNTIME"),
    ("familiar_models_dir", "REALMSPINNER_FAMILIAR_MODELS"),
    ("familiar_port", "REALMSPINNER_FAMILIAR_PORT"),
    ("familiar_idle_timeout", "REALMSPINNER_FAMILIAR_IDLE"),
    ("gltfpack_exe", "REALMSPINNER_GLTFPACK"),
    ("mesh_profile", "REALMSPINNER_MESH_PROFILE"),
    ("lowpoly_triangles", "REALMSPINNER_LOWPOLY_TRIANGLES"),
    ("mesh_retries", "REALMSPINNER_MESH_RETRIES"),
    ("mesh_hole_max", "REALMSPINNER_MESH_HOLE_MAX"),
    ("reference_retries", "REALMSPINNER_REFERENCE_RETRIES"),
    ("rank_candidates", "REALMSPINNER_RANK"),
    ("t2i_model", "REALMSPINNER_T2I_MODEL"),
    ("t2i_model_root", "REALMSPINNER_T2I_ROOT"),
    ("t2i_turbo_dir", "REALMSPINNER_T2I_DIR"),
    ("t2i_in_process", "REALMSPINNER_T2I_IN_PROCESS"),
    ("vram_exclusive", "REALMSPINNER_VRAM_EXCLUSIVE"),
    ("vram_budget_gib", "REALMSPINNER_VRAM_BUDGET"),
    ("vram_total_gib", "REALMSPINNER_VRAM_TOTAL"),
    ("pose_fit", "REALMSPINNER_POSE_FIT"),
    ("pose_timeout", "REALMSPINNER_POSE_TIMEOUT"),
    ("rig_template", "REALMSPINNER_RIG_TEMPLATE"),
    ("rig_timeout", "REALMSPINNER_RIG_TIMEOUT"),
    ("deform_qa", "REALMSPINNER_DEFORM_QA"),
    ("sheet_timeout", "REALMSPINNER_SHEET_TIMEOUT"),
    ("separation_timeout", "REALMSPINNER_SEPARATION_TIMEOUT"),
)

# The other half of the environment, and the reason the readout was incomplete.
#
# ``SETTINGS`` can only name ``Config`` fields -- the test above it asserts that
# in both directions -- so four variables that change what the process does were
# invisible to ``realmspinner doctor`` and to the Settings pane simply because no
# field carries them. They are read at their point of use, from ``os.environ``
# directly: ``migrate.py`` for the two home-migration switches, ``native.py``
# for the two kernel ones. A host that silently ran the numpy fallbacks, or that
# skipped a migration, gave the same readout as one that did not, which is
# exactly the failure ``from_env`` exists to prevent.
#
# They are a separate table rather than entries in ``SETTINGS`` because their
# value is the raw string in the environment, not a resolved field: there is no
# default to report, only set or unset. ``effective`` appends them last.
SWITCHES: tuple[tuple[str, str], ...] = (
    ("no_migrate", "REALMSPINNER_NO_MIGRATE"),
    ("migrate_keep", "REALMSPINNER_MIGRATE_KEEP"),
    ("native", "REALMSPINNER_NATIVE"),
    ("native_dll", "REALMSPINNER_NATIVE_DLL"),
)


@dataclass(frozen=True, slots=True)
class Setting:
    """One configured value, and where it came from."""

    name: str
    env: str
    value: str
    from_env: bool


def effective(config: Config | None = None) -> list[Setting]:
    """What this process is actually running on, field by field (S140).

    The point is the ``from_env`` column, not the values: "trellis port 17971"
    is not a diagnosis, and "trellis port 17971, set by REALMSPINNER_TRELLIS_PORT"
    is -- a host whose behaviour disagrees with the documentation almost always
    disagrees because something in its environment says so, and until this
    there was no way to see that from inside the app.

    Reads the environment rather than comparing against a freshly constructed
    ``Config``: the comparison would report "default" for a variable explicitly
    set to the default value, which is exactly the case where a user is asking
    whether their setting took.

    Both tables, ``SETTINGS`` first and then ``SWITCHES``: a field-backed
    setting reports its resolved value, an env-only switch reports the raw
    string or ``(unset)``, and the ``from_env`` column means the same thing in
    either case.

    Pure, in the ``vram.py`` sense -- stdlib only, no imports from ``service``,
    ``queue`` or ``studio`` -- because both the CLI and a pane read it.
    """
    config = config or get_config()
    out: list[Setting] = []
    for name, env in SETTINGS:
        value = getattr(config, name, None)
        out.append(
            Setting(
                name=name,
                env=env,
                value="(unset)" if value is None else str(value),
                from_env=env in os.environ,
            )
        )
    for name, env in SWITCHES:
        raw = os.environ.get(env)
        out.append(
            Setting(
                name=name,
                env=env,
                value="(unset)" if raw is None else raw,
                from_env=raw is not None,
            )
        )
    return out


_config: Config | None = None


def get_config() -> Config:
    global _config
    if _config is None:
        cfg = Config()
        # Before any mkdir: the migration's "destination already exists and is
        # non-empty, so leave it alone" rule has to see the destination as the
        # user left it, and creating data_dir first would make every first run
        # look like a half-finished move.
        from . import migrate

        migrate.run(cfg)
        # palette_dir and t2i_model_root are created here and not lazily: on a
        # fresh install both are directories the user has to put files into by
        # hand, and an existing empty folder is the only instruction that
        # survives the README being closed. bench_dir stays lazy -- its two
        # writers already mkdir, and an empty bench/ says nothing to anyone.
        for d in (cfg.home, cfg.data_dir, cfg.palette_dir, cfg.t2i_model_root):
            d.mkdir(parents=True, exist_ok=True)
        _config = cfg
    return _config
