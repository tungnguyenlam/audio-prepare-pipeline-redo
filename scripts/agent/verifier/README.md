# Hardened audio verifier behavior

These commands turn raw audio-model generation into validated pass/reject
verdict artifacts. They use the shared generation clients in `scripts/agent/`,
add the acoustic verifier prompt/response constraints, preserve exact model
text, parse model output, and enforce the core verdict contract.

Supported verifier backends are Gemini, OpenAI-compatible endpoint, Hugging
Face, Unsloth, vLLM, MOSS-Audio, MiniCPM-o, Kimi-Audio, and VibeVoice-ASR.
`evaluate_verifier.py` compares pairwise verdict files against reference verdicts.
`compare.sh` compares saved candidate runs against an explicitly selected reference,
reports matching coverage and acoustic defect disagreements, and exports tables and plots.
It runs offline and does not call any models.
`analysis.sh --input-dir DIR` (also available as `analyze.sh --verdict-dir DIR`) joins a verifier directory to diarization manifests, tolerates
individual invalid or missing responses, exports two flat CSV tables, and
renders coverage, decision, defect, dimension, measurement, speaker, and timeline plots.
It also writes readable error cases and per-category statistics inside `DIR/plot/`.
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

# Compare one video against the matching family inside a candidate run:
bash scripts/agent/verifier/compare.sh \
  --reference-dir .data/agent/verifier/gemini/gemini-3-8-flash/medium/example \
  --candidates-dir .data/agent/verifier/gemini/gemini-3-8-flash/low

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

## Analyze one model run

```bash
# One model + reasoning effort, across every family beneath that directory:
bash scripts/agent/verifier/analysis.sh \
  --input-dir .data/agent/verifier/gemini/gemini-3-8-flash/low

# One video only:
bash scripts/agent/verifier/analysis.sh \
  --input-dir .data/agent/verifier/gemini/gemini-3-8-flash/low/UuQgxxfU_Hc_VLOG-Lan-d

# Other backends use the same interface:
bash scripts/agent/verifier/analysis.sh --input-dir .data/agent/verifier/hf
```

No reference model or extra flags are required. Choose the saved model/effort
folder you want to inspect. The command reads artifacts recursively and saves
its results in `<input-dir>/plot/`. Repeating the command refreshes generated
reports and removes obsolete PNGs recorded by the previous analysis. Other files
are retained. `--output-dir` is optional; refreshing a nonempty custom destination
requires `--overwrite`. The existing `analyze.sh` and `--verdict-dir` remain aliases
for the same implementation.

Open `plot/report.md` first. It includes model identity and configuration, pass/
reject/invalid/missing counts, error statistics and cases grouped by category with
links to audio and verdict JSON. `error_cases.csv` includes the exact raw response
for each case; a clip with multiple defects has multiple rows. `error_stats.csv`
contains category counts, denominators and rates for each model configuration.
Acoustic rates use valid artifacts; processing-error rates use all artifacts in
that model group. `reject_without_defect_code` retains rejected clips from models
that do not emit acoustic codes. Detected defects are model labels, not measured
false accepts/rejects against a reference; use `compare.sh` for that comparison.

Plots include `coverage.png`, `decisions.png`, `defects.png`, and, when relevant,
`dimensions.png`, `measurements.png` (latency, confidence, speaker count, secondary
speech), `processing_errors.png`, `by_speaker.png`, and `timeline*.png`. Figures
summarize the supplied directory. If it contains several configurations, the
report/CSV statistics separate them and `by_model.png` shows their status counts;
select one variant directory for figures specific to that variant. `analysis.json`
also contains `model_stats`, `error_stats`, report paths and the current plot list.

The analyzer automatically finds diarization manifests beside recorded source
audio when accessible. Without a manifest, it can still analyze verdicts and raw
responses, but cannot count inputs that never produced artifacts or show their
speaker timeline/duration. Supply `--input-manifest` when source paths have moved.
A copied sibling `.txt` is used when its recorded path no longer exists, and its
hash is checked when supplied. Legacy verdict-only JSON is accepted. Runtime
`work/`, plots and other internal directories are excluded from artifact scanning.

## Compare saved verifier runs

Both required inputs accept a flat directory of verdict JSON files or a run root
containing family subdirectories. Every `--candidates-dir` is one candidate run;
repeat the flag to compare multiple backends or model/effort variants. The reference
is chosen by you and is not assumed to be ground truth.

```bash
# All families in medium, against low and HF:
bash scripts/agent/verifier/compare.sh \
  --reference-dir .data/agent/verifier/gemini/gemini-3-8-flash/medium \
  --candidates-dir .data/agent/verifier/gemini/gemini-3-8-flash/low \
  --candidates-dir .data/agent/verifier/hf

# Two flat folders (their directory names may differ):
bash scripts/agent/verifier/compare.sh \
  --reference-dir .data/verdicts/reference \
  --candidates-dir .data/verdicts/candidate \
  --output-dir .data/agent/verifier/comparisons/manual-review \
  --no-plots
```

For the command in the reported failure, use:

```bash
bash scripts/agent/verifier/compare.sh \
  --reference-dir .data/agent/verifier/gemini/gemini-3-8-flash/medium/UuQgxxfU_Hc_VLOG-Lan-d \
  --candidates-dir .data/agent/verifier/gemini/gemini-3-8-flash/low
```

Matching rules:

- A run root preserves relative family paths; a flat input uses its folder name.
  Two flat inputs share the reference family name. Thus `medium/example` matches
  `low/example` when you pass `low` as the candidate root.
- Clip keys are JSON filename stems with only the final known backend suffix
  removed: Gemini, HF, vLLM, Unsloth, endpoint, MOSS, MiniCPM, Kimi and VibeVoice.
  Matching does not depend on absolute source paths from a particular machine.
  Both recorded source SHA-256 hashes, when available, must agree.
- Only reference clips define the evaluation scope. Candidate-only clips are
  listed separately. Select a single backend/model/effort root, not a directory
  containing several runs. Duplicate normalized keys are an error.
- Scanning includes inputs under `.data/` and prunes child `work`, `plot`, `plots`,
  `comparisons`, `experiments`, `__pycache__` and hidden directories. Unrelated
  JSON objects are ignored and counted; malformed JSON is retained as invalid.
- Current wrapped verdicts and legacy verdict-only JSON are accepted. The analyzer
  and comparator share prompt-aware schema checks. Failed, uncertain, missing,
  schema-invalid or hash-mismatched pairs never enter pass/reject metrics. Comparison
  uses JSON only; audio and raw-response TXT files do not need to exist locally.

Terminal output shows the resolved reference, input counts, matching coverage,
recall, false rejection rate, agreement and bad accepts. Coverage is matched valid
pairs divided by valid reference clips. Metrics use each candidate's own matched
subset, so check coverage before comparing scores. Zero denominators are `N/A`
(`null` in JSON), including missing latency. Defect codes describe the annotations
available in the verdict: a speaker-only backend does not assess every acoustic
criterion. No defect annotations are invented for such backends.

Outputs default to a new directory under
`.data/agent/verifier/comparisons/<UTC-timestamp>-<selection-hash>/`:

| File | Contents |
| --- | --- |
| `report.md` | Readable aggregate comparison and metric definitions |
| `summary.json` | Input inventories, invalid counts, extra clip paths, aggregate and per-family metrics |
| `pairs.csv` | Every reference clip for each candidate, including missing/invalid/audio-mismatched cases |
| `conflicts.csv`, `conflicts.md` | Bad accepts, false rejects and defect-code disagreements with reasons |
| `plots/candidate_<index>_defects.png` | Caught/missed defect counts in candidate argument order |

`--no-plots` skips plotting; missing matplotlib produces a warning while retaining
all tables. `--output-dir` is an exact destination and must be new or empty, outside
all input trees. stdout contains only the successful output path; diagnostics and
the terminal report go to stderr. Missing directories, no valid reference, duplicate
keys, or any candidate with zero valid matches exit with code 2 before writing a
report. Partial coverage is allowed and explicitly reported.

The former no-argument global scan, positional family argument and `--base-dir`
were removed. Select the reference and each candidate explicitly using the examples
above; no model is silently chosen as teacher.
