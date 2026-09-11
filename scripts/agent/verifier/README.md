# Hardened audio verifier behavior

These commands turn raw audio-model generation into validated pass/reject
verdict artifacts. They use the shared generation clients in `scripts/agent/`,
add the acoustic verifier prompt/response constraints, preserve exact model
text, parse model output, and enforce the core verdict contract.

Supported verifier backends are Gemini, OpenAI-compatible endpoint, Hugging
Face, Unsloth, vLLM, MOSS-Audio, MiniCPM-o, Kimi-Audio, and VibeVoice-ASR.
`evaluate_verifier.py` compares their verdict files against reference verdicts.
`analyze.py` joins a verifier directory to diarization manifests, tolerates
individual invalid or missing responses, exports two flat CSV tables, and
renders coverage, decision, defect, dimension, speaker, and timeline plots.
`scaffold_experiment.py` creates local workspaces for verifier behavior
development under `.data/agent/verifier/experiments/`.

Default outputs follow the hardened behavior hierarchy:

```text
.data/agent/verifier/<backend>/<family>/
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

# Analyze one diarization result. The output defaults to <verdict-dir>/plot/.
bash scripts/agent/verifier/analyze.sh \
  --verdict-dir .data/agent/verifier/gemini/example \
  --input-manifest .data/diarize/sortformer/example/segments.json
```

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
