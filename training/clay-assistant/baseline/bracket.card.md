# Baseline: bracket

Subject: a mechanical bracket: a flat plate with two bolt holes and a right-angle gusset
Variant: compact tool card (gen/convert.compact_tools) as system turn
Model: unsloth/gemma-4-E2B-it-GGUF, gemma-4-E2B-it-BF16.gguf, untuned
Runtime: llama-server (Unsloth build 10909, commit 329b6160f), --jinja, ctx 16384, -ngl 99
Sampling: {"temperature": 1.0, "top_k": 64, "top_p": 0.95, "max_tokens": 4096}
Date: 2026-09-12
Tokens: prompt 2120, completion 1300; wall 7.1 s; finish_reason stop

## Prompt (user turn)

Answer with exactly one clay_batch tool call, as a JSON object with a "calls" list, that builds: a mechanical bracket: a flat plate with two bolt holes and a right-angle gusset

## Raw reply (verbatim)

```json
{
  "calls": [
    {
      "name": "clay_add_primitive",
      "params": {
        "generator": "box",
        "name": "Bracket_Plate",
        "params": {
          "extents": [
            1.0,
            0.1,
            10.0
          ]
        },
        "translation": [
          0,
          0.05,
          0
        ]
      }
    },
    {
      "name": "clay_add_primitive",
      "params": {
        "generator": "box",
        "name": "Bracket_Gusset",
        "params": {
          "extents": [
            0.5,
            0.1,
            0.5
          ]
        },
        "translation": [
          5.0,
          0.05,
          0
        ]
      }
    }
  ]
}
```
