# Agent behavior development

This command family separates behavior exploration from hardened behaviors.
The commands in this directory send user-owned prompts to audio-capable models
and preserve their unparsed responses. Once a behavior and response contract
are stable, its task-specific parsing and validation belong in a subdirectory
such as `verifier/`.

Available raw generation backends:

- `gemini.sh`: Gemini native audio generation without a response schema.
- `endpoint.sh`: OpenAI-compatible audio endpoints, including served vLLM and
  Unsloth models.
- `hf.sh`: local Hugging Face audio models with optional LoRA adapters.

All three reuse the same production generation paths as their hardened verifier
counterparts. The raw commands do not parse the answer: it may be prose,
Markdown, JSON, XML, a transcript, or another format requested by the prompt.

## Artifact contract

Every successful raw generation creates a pair in the same directory:

- `<stem>_<backend>.txt`: the exact generated text, including an empty response.
- `<stem>_<backend>.json`: prompts, generation settings, provider details,
  latency, usage/cost data when available, and the text file's path and digest.

The text file is written before its metadata sidecar, and the command reports
both paths. A cached run is reused only when both files exist and the metadata
and text digest match. An incomplete or conflicting pair requires
`--overwrite`.

With no explicit output path, raw outputs are grouped by backend and inferred
audio family:

```text
.data/agent/<backend>/<family>/
```

For example, a Gemini run may produce:

```text
.data/agent/gemini/<family>/<stem>_gemini.txt
.data/agent/gemini/<family>/<stem>_gemini.json
```

`--input-file --output-file result.txt` writes exactly `result.txt` and its
sibling `result.json`. A `.json` raw output is rejected because that suffix is
reserved for metadata. Directory input requires `--output-dir` or uses the
default above while preserving relative input directories.

## Examples

```bash
bash scripts/agent/gemini.sh \
  --input-dir .data/clips \
  --prompt-file .data/prompts/describe_audio.txt \
  --model gemini-3.8-flash \
  --reasoning-effort medium \
  --temperature 0.2 \
  --max-tokens 4096

bash scripts/agent/endpoint.sh \
  --input-file .data/clips/example.wav \
  --endpoint http://localhost:8000/v1/chat/completions \
  --model google/gemma-4-E2B-it \
  --prompt-file .data/prompts/describe_audio.txt

bash scripts/agent/hf.sh \
  --input-file .data/clips/example.wav \
  --model-id google/gemma-4-E2B-it \
  --prompt-file .data/prompts/describe_audio.txt
```

Use `--system-prompt-file` to test system/user prompt separation. The user
message places the prompt before the audio. `--top-p` and `--top-k` are exposed
where supported. Give named prompt/model/settings experiments explicit output
directories under `.data/agent/`; runtime artifacts must stay out of git.

Hardened acoustic verification is documented in
[`verifier/README.md`](verifier/README.md).
