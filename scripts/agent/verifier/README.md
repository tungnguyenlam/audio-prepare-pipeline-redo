# Hardened audio verifier behavior

These commands turn raw audio-model generation into validated pass/reject
verdict artifacts. They use the shared generation clients in `scripts/agent/`,
add the acoustic verifier prompt/response constraints, preserve exact model
text, parse model output, and enforce the core verdict contract.

Supported verifier backends are Gemini, OpenAI-compatible endpoint, Hugging
Face, Unsloth, vLLM, MOSS-Audio, MiniCPM-o, Kimi-Audio, and VibeVoice-ASR.
`evaluate_verifier.py` compares pairwise verdict files against reference verdicts.
`compare.sh` benchmarks candidate verifiers against the reference teacher (Gemini 3.8 Flash Medium)
globally or locally per diarization folder, breaks down defect criteria (clipped words,
secondary speakers, music bleed), renders visual comparison plots, and exports conflict cases.
`analyze.py` joins a verifier directory to diarization manifests, tolerates
individual invalid or missing responses, exports two flat CSV tables, and
renders coverage, decision, defect, dimension, speaker, and timeline plots.
`scaffold_experiment.py` creates local workspaces for verifier behavior
development under `.data/agent/verifier/experiments/`.

Default outputs follow the hardened behavior hierarchy:

```text
.data/agent/verifier/<backend>/<family>/
.data/agent/verifier/gemini/<model>/<reasoning-effort>/<family>/
```

New verifier outputs are a sibling pair: `<stem>_<backend>.txt` contains the
exact unparsed assistant response and `<stem>_<backend>.json` contains request
provenance, response identity, status, and either a validated verdict or a
stable failure stage/code. Generation, parsing, or schema failure is isolated
to that input; later inputs are still processed. Legacy verdict-only JSON is
still accepted by the analyzer, with raw-response columns left empty.

```bash
bash scripts/agent/verifier/gemini.sh --input-dir .data/clips
bash scripts/agent/verifier/hf.sh --input-dir .data/clips
bash scripts/agent/verifier/endpoint.sh --input-dir .data/clips

# Compare all verifiers against reference teacher (Global):
bash scripts/agent/verifier/compare.sh

# Compare verifiers for one specific diarization folder (Local):
bash scripts/agent/verifier/compare.sh khanhvy

# Analyze one diarization result. The output defaults to <verdict-dir>/plot/.
bash scripts/agent/verifier/analyze.sh \
  --verdict-dir .data/agent/verifier/gemini/gemini-3-8-flash/medium/example \
  --input-manifest .data/diarize/sortformer/example/segments.json
```

The Gemini verifier uses Google's asynchronous Batch API by default and waits
for the job to finish. This preserves the same raw-generation and verdict
parser path while applying Batch pricing. Use `--inference-mode standard` only
when an immediate synchronous response is required. Gemini artifacts include
`_inference_mode`, `_batch_job`, `_batch_request_key`, `_response_id`, exact
usage/cache counters, and the matching `paid_batch` or `paid_standard` estimate.
Submitted Batch state is kept under the selected model/reasoning variant's
`.data/.../work/batch_jobs/` path for interruption-safe resume.

The analysis directory contains `all_samples.csv`, which includes every
expected turn plus unmatched artifacts, and `successful_samples.csv`, which
contains only schema-valid `pass`/`reject` results. Both have identical,
pandas-friendly columns. Their first columns are `audio_path`, `final_verdict`,
and `assistant_raw_response`; known response fields and failure-code indicator
columns are also promoted while the lossless parsed JSON remains available.
Plots are read back from these CSV files so the tables are the analysis source
of truth. Invalid and missing responses are withheld from accepted yield but
are never silently counted as model rejects.

Every command remains standalone; this directory does not orchestrate agent
exploration, verification, or any other pipeline stage.
