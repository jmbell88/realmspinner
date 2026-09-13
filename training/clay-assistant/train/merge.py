"""Merge a run's LoRA adapter into the local base checkpoint, offline.

``train_a.py`` saves the adapter first and merges last, and run A's merge step failed
because Unsloth's ``save_pretrained_merged`` re-checks the Hugging Face hub for the base
model name even with the weights on disk and ``HF_HUB_OFFLINE=1`` set. This does the merge
with transformers + PEFT straight from the local snapshot directory, so nothing here can
reach for the network. Run with Studio's Python::

    ~/.unsloth/studio/unsloth_studio/Scripts/python.exe train/merge.py run-A
"""

from __future__ import annotations

import os
import pathlib
import sys

os.environ["HF_HUB_OFFLINE"] = "1"

HERE = pathlib.Path(__file__).resolve().parent
PKG = HERE.parent


def local_snapshot(repo: str) -> pathlib.Path:
    """The one snapshot directory the HF cache holds for *repo*."""
    hub = pathlib.Path.home() / ".cache" / "huggingface" / "hub"
    snaps = hub / ("models--" + repo.replace("/", "--")) / "snapshots"
    dirs = [p for p in snaps.iterdir() if p.is_dir()]
    if len(dirs) != 1:
        raise SystemExit(f"expected one snapshot under {snaps}, found {len(dirs)}")
    return dirs[0]


def main() -> int:
    run = sys.argv[1] if len(sys.argv) > 1 else "run-A"
    out = PKG / "out" / run
    lora = out / "lora"
    target = out / "merged-16bit"

    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    base_dir = local_snapshot("unsloth/gemma-4-E2B-it")
    print(f"base: {base_dir}", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        str(base_dir), dtype=torch.bfloat16, device_map="cpu"
    )
    model = PeftModel.from_pretrained(model, str(lora))
    model = model.merge_and_unload()
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(target), safe_serialization=True)
    AutoProcessor.from_pretrained(str(base_dir)).save_pretrained(str(target))
    total = sum(p.stat().st_size for p in target.glob("*.safetensors")) / 2**30
    print(f"merged -> {target} ({total:.2f} GiB of safetensors)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
