# Baseline: chair

Subject: a four-legged wooden chair with a slatted back
Variant: compact tool card (gen/convert.compact_tools) as system turn
Model: unsloth/gemma-4-E2B-it-GGUF, gemma-4-E2B-it-BF16.gguf, untuned
Runtime: llama-server (Unsloth build 10909, commit 329b6160f), --jinja, ctx 16384, -ngl 99
Sampling: {"temperature": 1.0, "top_k": 64, "top_p": 0.95, "max_tokens": 4096}
Date: 2026-09-12
Tokens: prompt 2113, completion 1553; wall 8.4 s; finish_reason stop

## Prompt (user turn)

Answer with exactly one clay_batch tool call, as a JSON object with a "calls" list, that builds: a four-legged wooden chair with a slatted back

## Raw reply (verbatim)

```json
{
  "calls": [
    {
      "name": "clay_add_primitive",
      "params": {
        "generator": "box",
        "name": "chair_seat",
        "params": {
          "extents": [0.5, 0.1, 0.5]
        },
        "translation": [0, 0.05, 0]
      }
    },
    {
      "name": "clay_add_primitive",
      "params": {
        "generator": "cylinder",
        "name": "leg_front_left",
        "params": {
          "radius": 0.05,
          "height": 0.5
        },
        "translation": [-0.25, 0.25, 0]
      }
    },
    {
      "name": "clay_add_primitive",
      "params": {
        "generator": "cylinder",
        "name": "leg_front_right",
        "params": {
          "radius": 0.05,
          "height": 0.5
        },
        "translation": [0.25, 0.25, 0]
      }
    },
    {
      "name": "clay_add_primitive",
      "params": {
        "generator": "cylinder",
        "name": "leg_back_left",
        "params": {
          "radius": 0.05,
          "height": 0.5
        },
        "translation": [-0.25, 0.25, 0]
      }
    },
    {
      "name": "clay_add_primitive",
      "params": {
        "generator": "cylinder",
        "name": "leg_back_right",
        "params": {
          "radius": 0.05,
          "height": 0.5
        },
        "translation": [0.25, 0.25, 0]
      }
    },
    {
      "name": "clay_add_primitive",
      "params": {
        "generator": "box",
        "name": "back_slat_1",
        "params": {
          "extents": [0.4, 0.02, 0.05]
        },
        "translation": [0, 0.25, 0.25]
      }
    },
    {
      "name": "clay_add_primitive",
      "params": {
        "generator": "box",
        "name": "back_slat_2",
        "params": {
          "extents": [0.4, 0.02, 0.05]
        },
        "translation": [0, 0.25, 0.45]
      }
    },
    {
      "name": "clay_add_primitive",
      "params": {
        "generator": "box",
        "name": "back_slat_3",
        "params": {
          "extents": [0.4, 0.02, 0.05]
        },
        "translation": [0, 0.25, 0.65]
      }
    },
    {
      "name": "clay_material",
      "params": {
        "uids": [
          1,
          2,
          3,
          4,
          5,
          6,
          7,
          8
        ],
        "color": [
          0.4,
          0.2,
          0.05,
          0.0
        ],
        "metallic": 0.1,
        "roughness": 0.8,
        "name": "wood_texture"
      }
    }
  ]
}
```
