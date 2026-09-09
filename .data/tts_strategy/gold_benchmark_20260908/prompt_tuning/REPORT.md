# Gemini 3.5 Flash-Lite Prompt Tuning on the 288-Clip Gold Set

## Result

Attempt 2 (`prompt_v2_defect_first.txt`) is the selected prompt. Compared with
the original prompt, it raised agreement with the accepted Gemini 3.8 Flash
MEDIUM labels from 182/288 (63.2%) to 206/288 (71.5%), raised F1 from 73.5% to
78.6%, reduced false passes from 94 to 74, and reduced false rejects from 12 to
8. All 288 requests succeeded.

The reference dataset is labeled by `gemini-3.8-flash` at MEDIUM reasoning. It
is not labeled by a 3.8 Flash-Lite model.

| Version | Prompt strategy | Agreement | TP / TN | False pass / false reject | Pass / reject | F1 | Thinking tokens | Cost |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| V0 | Original three-category prompt | 182/288 (63.2%) | 147 / 35 | 94 / 12 | 241 / 47 | 73.5% | not recorded | not recorded |
| V1 | Nine-dimension rubric alignment | 176/288 (61.1%) | 158 / 18 | 111 / 1 | 269 / 19 | 73.8% | 748 | $0.1748823 |
| **V2** | **English defect-first listening sweeps** | **206/288 (71.5%)** | **151 / 55** | **74 / 8** | **225 / 63** | **78.6%** | **39,831** | **$0.2009490** |
| V3 | Vietnamese defect-first | 173/288 (60.1%) | 155 / 18 | 111 / 4 | 266 / 22 | 72.9% | 915 | $0.1388146 |

All candidate runs used `gemini-3.5-flash-lite`, temperature 0, and MEDIUM
reasoning effort. New-run API metadata identifies the returned model version as
`gemini-3.5-flash-lite` for every successful response. The three new full runs
used 876,949 total tokens and cost an estimated $0.5146459 in aggregate.

## Why the Original Prompt Failed

The original prompt did not match the teacher rubric. It exposed three broad
categories, while the 3.8 teacher independently judged nine dimensions. In
particular, the old output vocabulary had no `sound_effect` code and collapsed
noise and reverberation into one coarse class. The teacher's labels contain 16
sound-effect defects, 17 reverberation defects, and three excessive-noise
defects.

Flash-Lite also used an overall-cleanliness shortcut. On 94 teacher rejects it
returned pass with generic explanations such as “single clean speaker” even
when the teacher identified specific continuous music, a second voice, or a
cut boundary. Original prompt recall by teacher defect was especially weak:
1/13 clipped starts, 4/29 clipped ends, 3/17 reverberation, 21/61 music, and
14/37 secondary speakers.

Merely listing all nine dimensions did not fix the shortcut. V1 produced only
19 rejects and 748 thinking tokens across the whole set. The Vietnamese V3
prompt behaved similarly, with 22 rejects and 915 thinking tokens. V2's short,
ordered, defect-first sweeps caused the model to spend 39,831 thinking tokens
and improved recall across the main defects: music 28/61, secondary speaker
19/37, clipped end 12/29, sound effect 11/16, reverberation 6/17, overlap 5/8,
and excessive noise 1/3. Clipped-start recall remained poor at 2/13, which is a
model limitation that prompt wording did not overcome.

V2's main trade-off is music sensitivity: all eight false rejects claimed
music on clips the teacher passed. Even with that cost, it improved both sides
of the original confusion matrix: 27 original false passes became correct
rejects and eight original false rejects became correct passes; seven original
true rejects and four original true passes moved the wrong way. Net agreement
rose by 24 clips.

## Saved Artifacts

- `prompt_v0_original.txt` and the pre-existing V0 JSON/CSV/Markdown under
  `../reports/gemini35_flash_lite_medium_vs_gemini38_medium.*`
- `prompt_v1_rubric_alignment.txt` and `v1_rubric_alignment.{json,csv,md}`
- `prompt_v2_defect_first.txt` and `v2_defect_first.{json,csv,md}`
- `prompt_v3_vietnamese_defect_first.txt`, the initial partial result,
  `v3_retry.{json,csv,md}`, and merged complete
  `v3_vietnamese_defect_first_complete.{json,csv,md}`

Prompt SHA-256 values:

- V0: `55bc6ec5e02008b3a3cffda3f588ce2eba6e5052e4f8920b1c193a2c0bc94ab4`
- V1: `c89d48e26d7f5375d5bf9ec5ad747fa79896ae94be125c3d873c80c4b8defb14`
- V2: `7fbdb3276c6be62358880259f9c8b28ae4fa07d0c8edcc7ec2c4ef6099765b5`
- V3: `264fba3977dd655ed8fab214ed5ca8a65fcfcce96eac491b676332784eb0df66`

## Selected Invocation

```bash
uv run --no-sync python scripts/evaluate_verifier.py \
  --backend gemini \
  --model gemini-3.5-flash-lite \
  --reasoning-effort medium \
  --concurrency 8 \
  --input .data/tts_strategy/gold_benchmark_20260908/eval_input.jsonl \
  --prompt-file .data/tts_strategy/gold_benchmark_20260908/prompt_tuning/prompt_v2_defect_first.txt \
  --output-report .data/tts_strategy/gold_benchmark_20260908/prompt_tuning/v2_defect_first.md \
  --output-json .data/tts_strategy/gold_benchmark_20260908/prompt_tuning/v2_defect_first.json \
  --export-csv .data/tts_strategy/gold_benchmark_20260908/prompt_tuning/v2_defect_first.csv
```
