# Baseline: serpent

Subject: a serpentine creature whose body tapers from head to tail, on four short legs
Variant: compact tool card (gen/convert.compact_tools) as system turn
Model: unsloth/gemma-4-E2B-it-GGUF, gemma-4-E2B-it-BF16.gguf, untuned
Runtime: llama-server (Unsloth build 10909, commit 329b6160f), --jinja, ctx 16384, -ngl 99
Sampling: {"temperature": 1.0, "top_k": 64, "top_p": 0.95, "max_tokens": 4096}
Date: 2026-09-12
Tokens: prompt 2118, completion 1321; wall 7.2 s; finish_reason stop

## Prompt (user turn)

Answer with exactly one clay_batch tool call, as a JSON object with a "calls" list, that builds: a serpentine creature whose body tapers from head to tail, on four short legs

## Raw reply (verbatim)

```json
{
  "calls": [
    {
      "name": "clay_add_primitive",
      "params": {
        "generator": "cylinder",
        "name": "serpentine_body",
        "params": {
          "radius": 0.2,
          "height": 4.0,
          "translation": [0, 2.0, 0]
        },
        "rotation": [0, 0, 0],
        "scale": [1, 1, 1]
      }
    },
    {
      "name": "clay_add_primitive",
      "params": {
        "generator": "cylinder",
        "name": "leg_1",
        "params": {
          "radius": 0.1,
          "height": 1.0,
          "translation": [-1.0, 0.5, 0]
        },
        "rotation": [0, 0, 0],
        "scale": [1, 1, 1]
      }
    },
    {
      "name": "clay_add_primitive",
      "params": {
        "generator": "cylinder",
        "name": "leg_2",
        "params": {
          "radius": 0.1,
          "height": 1.0,
          "translation": [1.0, 0.5, 0]
        },
        "rotation": [0, 0, 0],
        "scale": [1, 1, 1]
      }
    },
    {
      "name": "clay_add_primitive",
      "params": {
        "generator": "cylinder",
        "name": "leg_3",
        "params": {
          "radius": 0.1,
          "height": 1.0,
          "translation": [-1.0, -0.5, 0]
        },
        "rotation": [0, 0, 0],
        "scale": [1, 1, 1]
      }
    },
    {
      "name": "clay_add_primitive",
      "params": {
        "generator": "cylinder",
        "name": "leg_4",
        "params": {
          "radius": 0.1,
          "height": 1.0,
          "translation": [1.0, -0.5, 0]
        },
        "rotation": [0, 0, 0],
        "scale": [1, 1, 1]
      }
    }
  ]
}
```
