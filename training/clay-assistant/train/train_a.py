"""Run A: LoRA fine-tune of Gemma 4 E2B-it on the verified Clay-assistant dataset.

Driven with Unsloth's own library from the Unsloth Studio environment
(``~/.unsloth/studio/unsloth_studio/Scripts/python.exe``) rather than through the Studio
GUI, because the GUI could not be driven from this session; the recipe is the one the
Studio would have run (LoRA on bf16 weights, responses-only loss, the Gemma 4 template).

Choices, each with its reason:

* **bf16 LoRA, not QLoRA** -- 32 GB is plenty for a ~5B-raw-parameter model, and 4-bit
  quantisation of the base costs accuracy the dataset then has to pay back.
* **Inline fenced JSON, not native tool-call tokens** -- the assistant turns in the export
  carry both; we train on ``content`` only (the ```` ```json ```` fence) and drop
  ``tool_calls``, so the model emits exactly what ``baseline/score_baseline.py`` and the
  later integration parse. Native tool tokens would need the runtime to render ``tools``
  through the jinja template on every call; the fence works with any runtime.
* **Responses-only loss** -- the ~2k-token system turn is identical on every row; training
  on it would spend most of the gradient learning to reproduce our own tool card.
* **max_seq_length 8192** -- measured rows are 2.2k-4.8k tokens (p95 ~3k).
* **Evaluation on val each epoch** -- loss only; the door-acceptance eval is
  ``eval/run_val.py``, run after export, because loss says nothing about whether a batch
  is accepted.

Outputs under ``training/clay-assistant/out/run-A/`` (gitignored): the LoRA adapter,
the merged 16-bit model, ``trainer_state.json``, and ``config.json`` with every value
below. GGUF export is a separate step (``export_gguf.py``) so an export failure cannot
cost the run.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "0")

HERE = pathlib.Path(__file__).resolve().parent
PKG = HERE.parent
DATASET = PKG / "dataset"
OUT = PKG / "out" / "run-A"
OUT.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "base_model": "unsloth/gemma-4-E2B-it",
    "max_seq_length": 8192,
    "load_in_4bit": False,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.0,
    "target": "language attention + mlp, no vision",
    "learning_rate": 2e-4,
    "epochs": 3,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "warmup_ratio": 0.03,
    "lr_scheduler": "cosine",
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    "seed": 3407,
    "chat_template": "gemma-4",
    "loss": "responses only (<|turn>model)",
    "assistant_turn": "content (fenced json) only; tool_calls dropped",
    "train_rows": None,
    "val_rows": None,
}


def _rows(path: pathlib.Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        msgs: list[dict] = []
        for m in r["messages"]:
            # Drop the tool_calls channel; the fenced JSON in content is the target.
            content = m.get("content") or ""
            # The Gemma 4 template refuses two consecutive turns of one role, and an
            # edit/query row carries the scene turn and the request as two user turns
            # (convert.to_messages keeps them apart so a runtime can choose). Fold them
            # into one user turn here, scene first, blank line, then the request.
            if msgs and msgs[-1]["role"] == m["role"]:
                msgs[-1]["content"] += "\n\n" + content
            else:
                msgs.append({"role": m["role"], "content": content})
        rows.append({"conversations": msgs})
    return rows


def main() -> int:
    from unsloth import FastModel  # noqa: I001  (must be imported before transformers)
    from unsloth.chat_templates import get_chat_template, train_on_responses_only

    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    t0 = time.time()
    model, tokenizer = FastModel.from_pretrained(
        model_name=CONFIG["base_model"],
        dtype=None,
        max_seq_length=CONFIG["max_seq_length"],
        load_in_4bit=CONFIG["load_in_4bit"],
        full_finetuning=False,
    )
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"],
        bias="none",
        random_state=CONFIG["seed"],
    )
    tokenizer = get_chat_template(tokenizer, chat_template=CONFIG["chat_template"])

    train_rows = _rows(DATASET / "unsloth_train.jsonl")
    val_rows = _rows(DATASET / "unsloth_val.jsonl")
    CONFIG["train_rows"] = len(train_rows)
    CONFIG["val_rows"] = len(val_rows)

    def fmt(examples):
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            ).removeprefix("<bos>")
            for convo in examples["conversations"]
        ]
        return {"text": texts}

    train_ds = Dataset.from_list(train_rows).map(fmt, batched=True)
    val_ds = Dataset.from_list(val_rows).map(fmt, batched=True)

    sample = train_ds[0]["text"]
    (OUT / "sample_rendered.txt").write_text(sample, encoding="utf-8")
    # Gemma 4's "tokenizer" is the multimodal processor; plain text goes to its inner one.
    text_tok = getattr(tokenizer, "tokenizer", tokenizer)
    lens = [len(text_tok(t)["input_ids"]) for t in train_ds["text"]]
    CONFIG["token_len_min_med_max"] = [min(lens), sorted(lens)[len(lens) // 2], max(lens)]
    print(f"tokens min/med/max {CONFIG['token_len_min_med_max']}", flush=True)
    assert max(lens) <= CONFIG["max_seq_length"], "a row exceeds max_seq_length"

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=SFTConfig(
            output_dir=str(OUT / "checkpoints"),
            dataset_text_field="text",
            per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
            num_train_epochs=CONFIG["epochs"],
            learning_rate=CONFIG["learning_rate"],
            warmup_ratio=CONFIG["warmup_ratio"],
            lr_scheduler_type=CONFIG["lr_scheduler"],
            optim=CONFIG["optim"],
            weight_decay=CONFIG["weight_decay"],
            logging_steps=5,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            bf16=True,
            fp16=False,
            seed=CONFIG["seed"],
            report_to="none",
            max_length=CONFIG["max_seq_length"],
            dataloader_num_workers=0,
        ),
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|turn>user\n",
        response_part="<|turn>model\n",
    )

    # Prove the mask: the first row's labels must be -100 everywhere except the model turn.
    batch = trainer.train_dataset[0]
    labels = batch["labels"]
    kept = sum(1 for x in labels if x != -100)
    CONFIG["mask_check_first_row"] = {"tokens": len(labels), "trained_on": kept}
    print(f"mask check: {kept}/{len(labels)} tokens carry loss on row 0", flush=True)
    assert 0 < kept < len(labels) * 0.5, "responses-only mask did not apply"

    (OUT / "config.json").write_text(json.dumps(CONFIG, indent=2), encoding="utf-8")
    print(f"setup {time.time() - t0:.0f}s; training...", flush=True)

    stats = trainer.train()
    (OUT / "train_result.json").write_text(json.dumps(stats.metrics, indent=2), encoding="utf-8")
    (OUT / "trainer_state.json").write_text(
        json.dumps(trainer.state.log_history, indent=2), encoding="utf-8"
    )

    model.save_pretrained(str(OUT / "lora"))
    tokenizer.save_pretrained(str(OUT / "lora"))
    # The merge is a separate step (merge.py): Unsloth's save_pretrained_merged re-checks
    # the hub for the base model name and fails under HF_HUB_OFFLINE, which cost run A its
    # merge after a clean 58-minute train. The adapter above is the run's real output.
    peak_gib = torch.cuda.max_memory_reserved() / 2**30
    print(f"done in {time.time() - t0:.0f}s; peak VRAM {peak_gib:.1f} GiB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
