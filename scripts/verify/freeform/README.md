# Freeform audio-model experiments

This folder is intentionally separate from the production verifiers. It is for
testing what an audio-capable model understands and for iterating on prompt and
answer formats. Its outputs are observations, not pass/reject verdicts, and are
not accepted by `scripts/verify/evaluate_verifier.py`.

Available execution paths:

- `gemini.sh`: Gemini native audio API, without `responseMimeType` or a schema.
- `endpoint.sh`: OpenAI-compatible audio endpoints, including served vLLM,
  Unsloth, Gemma, and other compatible models.
- `hf.sh`: local Hugging Face audio models, including Gemma 4 checkpoints and
  optional LoRA adapters.

None of these commands parses the generated answer. The model may return prose,
Markdown, JSON, XML, a transcript, or any other text requested by the prompt.

For each input it writes:

- `<stem>_gemini.txt`: the first candidate's text, character-for-character as
  returned in its text parts (multiple text parts are concatenated without
  separators).
- `<stem>_gemini.json`: experiment metadata, the exact prompts and generation
  settings, latency, usage/cost estimates, and the complete Gemini response body.
  This sidecar preserves non-text parts, finish reasons, safety feedback, and any
  additional candidates even when the text file is empty.

Runtime outputs belong under `.data/`. Keep prompt files in a named experiment
directory when they are temporary, or under `prompts/` when they are useful,
reviewed project inputs. Give each prompt/model/settings variant a separate
output directory; matching completed runs are reused, while conflicting outputs
require `--overwrite`.

```bash
bash scripts/verify/freeform/gemini.sh \
  --input-dir .data/clips \
  --output-dir .data/verify/freeform-gemini/baseline \
  --prompt-file .data/prompts/describe_audio.txt \
  --model gemini-3.8-flash \
  --reasoning-effort medium \
  --temperature 0.2 \
  --max-tokens 4096
```

The equivalent non-Gemini entry points are:

```bash
bash scripts/verify/freeform/endpoint.sh \
  --input-file .data/clips/example.wav \
  --output-dir .data/verify/freeform-endpoint/baseline \
  --endpoint http://localhost:8000/v1/chat/completions \
  --model google/gemma-4-E2B-it \
  --prompt-file .data/prompts/describe_audio.txt

bash scripts/verify/freeform/hf.sh \
  --input-file .data/clips/example.wav \
  --output-dir .data/verify/freeform-hf/baseline \
  --model-id google/gemma-4-E2B-it \
  --prompt-file .data/prompts/describe_audio.txt
```

Use `--system-prompt-file` to test system/user prompt separation and
`--audio-position before|after` to test whether part ordering changes behavior.
`--top-p` and `--top-k` are optional sampling controls. `--input-file` takes
precedence over `--input-dir`; with one input, `--output-file` is the exact text
destination. The `.json` suffix is reserved for its metadata sidecar.

Provider-specific generation remains in the existing verifier runner when one
exists. `endpoint.py` and `hf.py` expose raw `generate()` methods used by both
their production verifier and these experiments. `_common.py` owns only the
shared experiment artifact/resume contract.
