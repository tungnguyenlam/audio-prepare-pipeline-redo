# Agent generation and hardened verifiers

[← Index](README.md) · [Data contract §5–8](data_contract.md#5-agent-response-pair-scriptsagent) · [Cookbook](commands.md#agent-generation-and-verification)

`scripts/agent/` sends a user-owned prompt plus one audio clip to an audio-capable
model and preserves the unparsed answer. `scripts/agent/verifier/` reuses the same
generation clients, requests JSON, parses it, and enforces a pass/reject verdict
schema. Neither directory orchestrates other pipeline stages.

## Raw generation (`scripts/agent/{gemini,endpoint,hf}.sh`)

| Backend | Transport | Notes |
|---|---|---|
| `gemini` | Google Gemini API | Batch API by default (`--inference-mode batch`, ≤ `--batch-size 10` requests per job, split further under the 20 MB inline limit, polled up to `--batch-timeout-s 86400`); `--inference-mode standard` for synchronous calls. Default `--model gemini-3.8-flash --reasoning-effort medium` (`none/low/medium/high` → `thinkingLevel`). Needs `GEMINI_API_KEY`. |
| `endpoint` | OpenAI-compatible `/v1/chat/completions` | Served vLLM, Unsloth, etc. Optional `OPENAI_API_KEY`. |
| `hf` | Local `transformers` model | Multimodal `AutoProcessor` / `AutoModelForMultimodalLM` path for Gemma 4; `--model-id`, `--adapter-path` (LoRA), `--load-in-4bit/-8bit`, `--device`. |

- `--prompt-file` is required; `--system-prompt-file` is optional. The user message
  places the prompt before the audio.
- Output is the pair `<stem>_<backend>.txt` + `.json` described in the
  [data contract](data_contract.md#5-agent-response-pair-scriptsagent). No parsing
  is applied: the text may be prose, JSON, XML, a transcript, or anything the prompt
  asked for.
- Interrupted Gemini Batch runs resume from `work/batch_jobs/` when re-invoked with
  identical arguments; `--overwrite` submits a fresh job.

## Hardened verifiers (`scripts/agent/verifier/*.sh`)

| Backend | Environment | Model / options |
|---|---|---|
| `gemini` | `.venvs/verify` | as above; adds `_usage`, `_cost` (`paid_batch` / `paid_standard`) |
| `hf` | `.venvs/verify` | Gemma 4 E2B / E4B / 12B or any HF audio LLM, optional LoRA adapter; `--max-new-tokens 1024` by default |
| `endpoint` | `.venvs/verify` | any OpenAI-compatible server |
| `unsloth` | `.venvs/verify` | Unsloth Studio (`--model`, `--gguf-variant`, `--payload-mode`, `UNSLOTH_*` env) |
| `vllm` | `.venvs/vllm` | offline vLLM engine (`--dtype`, `--tensor-parallel-size`, …) or `--endpoint` server |
| `moss` | `.venvs/moss` | MOSS-Audio (`--model-id`, `--torch-dtype`, `--trust-remote-code`) |
| `minicpm` | `.venvs/minicpmo` | MiniCPM-o (Python 3.11 env) |
| `kimi` | `.venvs/kimi` | Kimi-Audio (Python 3.11 env) |
| `vibevoice` | `.venvs/vibevoice` | VibeVoice-ASR speaker counting; `--min-secondary-speech-s` separates `reject` from `uncertain` |

- `--prompt-file` defaults to `prompts/acoustic_defect-3.txt`. The prompt text selects
  the validation profile (`acoustic_defect_v3`, `speaker_purity_v1`,
  `word_boundary_v1`, or `custom`) — see
  [data contract §6](data_contract.md#6-verifier-verdict-scriptsagentverifier).
- Generation, parse, or schema failure is recorded per input
  (`status: "fail"`, `error.stage`) and never stops later inputs.
- Failure JSON records a stable `error.code`, explanatory `error.message`, and a
  safe exception class for generation failures. Schema failures also retain the
  parsed-but-invalid object as `invalid_verdict`; exact model text remains in the
  sibling `.txt` response artifact.
- Every verifier backend that accepts prompts uses the same runtime parser and
  schema validator. Acoustic v3 pass responses require a nonempty transcript;
  reject responses must omit it. Verifiers never move or delete audio.
- `vibevoice` remains a prompt-free speaker-count verifier and therefore does not
  run the acoustic v3 rubric or emit its transcript field.

Prompts in `prompts/`: `acoustic_defect-3.txt` is the active verifier default;
`acoustic_defect.txt` and `acoustic_defect-2.txt` are retained unchanged as
deprecated, reference-only revisions and are not registered validation profiles.
`speaker_purity.txt` and `word_boundary.txt` are narrower verifier alternatives.
Free-form transcript/description prompts (`prompt-transcripts-*.txt`,
`vi-prompt-alam*.txt`) remain available for `scripts/agent/`.
`acoustic_defect-3.txt` combines acoustic verification with conditional
Vietnamese/English transcription and an audible emotion/speaking-style label. It
accepts faint non-intrusive background noise, adds `unsupported_language` and
`singing` eligibility failures, requires nonempty `emotion` and final `transcript`
fields for pass, and forbids the transcript field for reject.

The HF backend requires a multimodal processor. Gemma 4 is detected from its model
configuration and is loaded only through `AutoModelForMultimodalLM`; it does not
fall back to a text-only loader. Messages, audio loading, tokenization, and feature
construction run together through the processor's multimodal chat template. The
verify environment requires `transformers>=5.10.1` for this API and model family.
If a parsed HF pass omits `emotion` or `transcript`, the backend makes one targeted
repair generation and records `_schema_retry` in the final verdict. Other parse or
schema failures remain failures and retain their raw response for diagnosis.

## Offline tools (no model calls)

### `analysis.sh --input-dir DIR` (alias `analyze.sh --verdict-dir DIR`)

Reads every verifier JSON under `DIR` recursively (skipping `work/`, `plot*/`,
`comparisons/`, `experiments/`, hidden dirs), joins diarization manifests found
beside the recorded source audio (or given via `--input-manifest` /
`--manifest-dir`), and writes `DIR/plot/` — see
[data contract §7](data_contract.md#7-verifier-analysis-analysissh---input-dir-dir--dirplot).
Start with `plot/report.md`. Point it at one model/effort directory for
single-variant figures; a directory with several configurations still gets
per-model statistics and `by_model.png`. Acoustic v3 transcript text, character and
word counts plus emotion labels are exported to both sample CSVs and summarized in
`analysis.json`; `transcripts.png` shows contract outcomes and pass transcript
lengths, while `emotions.png` and `emotions_by_speaker.png` show emotion counts and
durations when available. Rerunning refreshes `plot/`; a nonempty
custom `--output-dir` needs `--overwrite`. Detected defects are model labels, not
errors against a reference.

### `compare.sh --reference-dir REF --candidates-dir CAND [...]`

Compares saved runs. Each `--candidates-dir` is one candidate; the reference is
whatever you choose, not an implied teacher. Inputs may be a flat directory of
verdict JSON or a run root with family subdirectories; a flat reference matches the
same family name under a candidate root (`medium/example` ↔ `low` finds
`low/example`). Clip keys are filename stems minus the backend suffix; recorded
source SHA-256 values must agree when both exist. Only reference clips define
scope; failed, uncertain, invalid, missing or hash-mismatched pairs are reported but
excluded from metrics. Output goes to a new directory under
`.data/agent/verifier/comparisons/` (or an empty `--output-dir` outside all inputs);
files are listed in [data contract §8](data_contract.md#8-verifier-comparison-comparesh--dataagentverifiercomparisonsutc-hash).
Exit code 2 on missing inputs, no valid reference, duplicate keys, or a candidate
with zero valid matches. `pairs.csv` preserves both transcripts and emotion labels
and records their exact, different, missing-candidate, or not-applicable status.
Transcript differences are also written separately; neither text nor emotion
agreement changes verifier decision metrics.

### `evaluate_verifier.sh --predictions-dir P --reference-dir R --output-file F`

Pairs verdict JSON files in two flat directories by clip stem (all verifier backend
suffixes stripped), validates each artifact against its prompt profile, and reports
accuracy, reject precision/recall/F1, false-rejection rate, latency percentiles,
transcript exact-match counts, and emotion exact-match counts as JSON. Use
`compare.sh` for family trees, other backends, and defect-level disagreement.

### `scaffold_experiment.sh --name NAME`

Creates an empty verifier-development workspace
`.data/agent/verifier/experiments/NAME/` (`experiment.json`, `templates.json`,
`manifests/*.jsonl`, `audio/`, `reports/`, `checkpoints/`). It refuses an existing
target and performs no model or data operations.
