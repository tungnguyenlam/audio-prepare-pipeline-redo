# Agent generation and hardened verifiers

[← Overview](../README.md) · [Data contract §5–8](data_contract.md#5-agent-response-pair-scriptsagent) · [Cookbook](commands.md#agent-generation-and-verification)

`scripts/s4-agent/` sends a user-owned prompt plus one audio clip to an audio-capable
model and preserves the unparsed answer. `scripts/s4-agent/verifier/` reuses the same
generation clients, requests JSON, parses it, and enforces a pass/reject verdict
schema. Neither directory orchestrates other pipeline stages.

## Raw generation (`scripts/s4-agent/{gemini,endpoint,hf}.sh`)

| Backend | Transport | Notes |
|---|---|---|
| `gemini` | Google Gemini API | Standard synchronous API by default; opt into Batch (`--inference-mode batch`, ≤ `--batch-size 10` requests per job, split further under the 20 MB inline limit, polled up to `--batch-timeout-s 86400`); `--inference-mode flex` for synchronous calls on Google's Flex tier (same 50% discount as Batch, 1–15 min latency, defaults `--timeout-s 900 --max-retry 11` because Flex answers 503 when capacity is short); `--inference-mode standard` for full-price synchronous calls. Default `--model gemini-3.8-flash --reasoning-effort medium` (`low/medium/high` → `thinkingLevel`) and `--max-tokens 65536` (the model's 64k ceiling). Sampling controls are omitted as required by the Gemini 3.8 migration guide. No forced JSON response mode: the raw command preserves free-form answers and the verifier validates the requested JSON afterward. AI Studio settings must be matched explicitly. Needs `GEMINI_API_KEY`. |
| `endpoint` | OpenAI-compatible `/v1/chat/completions` | Served vLLM, Unsloth, etc. Optional `OPENAI_API_KEY`. |
| `hf` | Local `transformers` model | Multimodal `AutoProcessor` / `AutoModelForMultimodalLM` path for Gemma 4; `--model-id`, `--adapter-path` (LoRA), `--load-in-4bit/-8bit`, `--device`. |

- Raw endpoint/HF commands require `--prompt-file`; `--system-prompt-file` is
  optional. Their user message places the prompt before the audio.
- Raw Gemini loads `prompts/full-tags-prompt.md` as a system instruction, overridable
  with `GEMINI_SYSTEM_PROMPT` in the environment or repository `.env`. It has no
  system-prompt file CLI flag. The Gemini verifier retains `--prompt-file` and sends that
  prompt as a system instruction too. User content contains audio followed by
  `--user-prompt` text (default: "Analyze the attached audio according to the system
  instructions."). The Gemini 3.8 migration guide requires nonempty text in the
  final user turn; blank user instructions are rejected locally.
  Standard uses `google-genai`; Flex/Batch use REST. The verify requirements now
  declare the SDK; existing environments need it provisioned separately.
- Both Gemini commands accept `--max-retry N`: at most N additional attempts per
  transient failed HTTP request (network errors or HTTP 429/500/502/503/504).
  `0` means one attempt, with no retries; defaults are 4 retries for Standard/Batch
  and 11 for Flex. SDK internal retries are disabled so they cannot multiply this
  limit. Retry delays respect a longer provider `Retry-After` header when present.
  Legacy `--max-retries` still means total attempts, including the first;
  the two flags are mutually exclusive. Nonretryable errors stop immediately.
  Cache creation and Batch submission do not retry ambiguous network failures,
  preventing duplicate resources/jobs. Batch polling requests use the same limit,
  but failed Batch results are not automatically resubmitted by this flag.
- A valid verifier `reject` is a successful completed verdict, never a retry trigger.
  `--max-response-retries N` separately retries empty or transient incomplete
  provider answers in Standard/Flex (default: 3 additional generations; 0 disables
  response retries). It handles HTTP 200 responses with no final text, missing
  completion reason, or transient OTHER/malformed/tool-call finish reasons.
  Each generation has its own HTTP retry budget: with defaults Standard can make
  up to 4 × 5 generation HTTP attempts, plus preflight/cache requests. Retries can
  incur charges; they stop at the configured limit, not an unbounded loop.
  Prompt blocks, safety/recitation blocks, unknown terminal finish reasons, and
  MAX_TOKENS fail explicitly without identical automatic resubmission. Review the
  recorded reason/configuration; retry cannot guarantee a usable response.
  Nonempty STOP text that fails verifier JSON/schema validation remains a failure;
  response retries do not repair task output or reroll valid acoustic rejections.
  Batch answers receive the same completion checks but are not automatically
  resubmitted; saved failures replay on `--continue`, requiring fresh work via
  `--overwrite` (scope input to the affected clips to avoid rerunning successes).
- Gemini raw failures retain their text/provider evidence with `status: fail`.
  Raw generation artifacts retain received retry responses in `attempts` and sum
  their usage/cost. Verifier JSON uses the compact verdict layout: source,
  parameters, status, response-file metadata, and verdict (or error/invalid_verdict
  on failure). It does not embed `generation`, provider bodies, thought signatures,
  or retry response copies. Final unparsed answer text remains in the sibling
  `.txt`; verdict metadata retains latency, aggregate usage/cost, model version
  and response ID. Provider completion failures keep their safe code/message in
  `error`; generation evidence is used internally for retry and run cost reporting.
  Exhausted response retries are failures, never successful empty output.
  Existing continuation and overwrite rules below apply.
- Verifier metadata now records `prompt_role=system` so old user-prompt runs cannot
  silently mix with the new generation behavior. Older artifacts require a separate
  output directory or explicit `--overwrite`.
- Output is the pair `<stem>_<backend>.txt` + `.json` described in the
  [data contract](data_contract.md#5-agent-response-pair-scriptsagent). No parsing
  is applied: the text may be prose, JSON, XML, a transcript, or anything the prompt
  asked for.
- Both Gemini commands accept `--continue` in Standard, Flex, and Batch mode.
  Completed valid output pairs are skipped even if made with another prompt,
  source digest, or settings (the run logs how many), so deleting bad outputs and
  rerunning with `--continue` regenerates only those. Without `--continue`, such
  conflicts require `--overwrite` or a new output directory.
  `--continue` and `--overwrite` cannot be combined.
- Continuation retries missing, failed, or incomplete outputs. Standard/Flex
  requests cannot recover an in-flight response after a local interruption;
  retrying that clip may incur another charge.
- Batch continuation reconnects to saved jobs under `work/batch_jobs/` and
  republishes their responses when the run manifest (root, paths, source digests,
  output destinations, prompts, settings, batch size) is unchanged and covers
  every missing output. Otherwise the missing outputs are submitted as a new Batch
  run. Without `--continue`, pending work submits a fresh Batch job.
  `--overwrite` regenerates all outputs with fresh requests.
  Failed provider jobs are resubmitted while successful jobs are retained.
  Saved per-request errors or responses that fail verdict validation are replayed
  on continuation; use `--overwrite` to request new responses for those clips.
- Both Gemini commands share one mode selector and request implementation.
  Standard is the default. Python callers use
  `inference_mode="batch" | "flex" | "standard"`; call `generate_batch()` for Batch
  or `generate()` / `verify()` for synchronous modes. The Python Batch method
  retains its `reuse_state=True` default; the CLI passes the explicit flag value.
- Explicit prompt caching is **off by default**. Add `--cache-prompt` to create a
  Google `cachedContents` resource containing the system prompt text. Audio is sent separately on every request. The cache is
  reused within that command invocation, including concurrent workers. It is
  created only when a new provider request needs it; resumed Batch jobs reuse
  their existing requests. New invocations create their own cache for new work.
- `--cache-ttl-s` defaults to 3,600 seconds for Standard/Flex and 90,000 seconds
  (25 hours) for Batch. Batch requires at least 90,000 seconds to cover its queue
  window. Caches expire automatically, including after interruption; storage is
  billed for the full TTL. Caches are replaced before new submissions when near
  expiry (or when less than 24 hours remain for Batch).
  Google rejects prompts below its model-specific minimum or unsupported caching
  combinations; Standard/Flex log a warning and send the prompt inline after a
  cache-creation HTTP 400. Batch stops instead.
- Implicit caching can still occur without `--cache-prompt`. Usage-based estimates
  distinguish cached reads from uncached input and include output/thinking tokens.
  Explicit caches contain only text, so cached tokens are priced as text even if
  provider modality details apportion hits across audio and text.
- Cost estimates include `cache_storage_usd` once per created cache, using its
  returned token count and full TTL. Storage is included in the run total and
  assigned to the first subsequent priced response for offline artifact sums;
  per-clip totals therefore include that shared overhead on one clip. If no
  response is produced, storage appears only in the run summary. Unknown storage
  rates/counts are reported as `unpriced_caches`, not zero-priced caches. Cached
  Batch jobs retain their recorded storage estimate on resume alongside inference
  estimates; resumed totals describe the job, not solely new charges.
- Synchronous pricing uses the returned Standard/Flex service tier when present,
  otherwise the requested mode; Batch always uses Batch pricing. The September
  21, 2026 rate card includes the introductory Flash storage rate and the distinct
  Gemini 3.5 Flash Flex cached-read rate. Sources:
  [REST Flex](https://ai.google.dev/gemini-api/docs/generate-content/flex-inference),
  [caching](https://ai.google.dev/gemini-api/docs/generate-content/caching), and
  [pricing](https://ai.google.dev/gemini-api/docs/pricing).
- Cache settings are part of artifact identity. Existing verifier outputs from
  earlier versions may require `--overwrite` or a new output directory.

## Gemini launcher dependency errors

The Gemini launchers select `VERIFIER_PYTHON` when set, otherwise
`.venvs/verify/bin/python`, then `.venvs/main/bin/python`. Activating Conda
`(base)` does not override that selection. If startup reports missing `httpx` or
`google.genai`, check the selected interpreter; both dependencies are already
listed in `envs/requirements-verify.txt`.

If the active Python already has both dependencies, select it explicitly:

```bash
export VERIFIER_PYTHON="$(command -v python)"
"$VERIFIER_PYTHON" -c 'import httpx; from google import genai'
bash scripts/s4-agent/verifier/gemini.sh --help
```

Then rerun the normal command in that shell. Quote input paths containing spaces.
To provision the dedicated environment instead, use the existing
`bash envs/setup_worker_envs.sh verify` command; it installs the full verifier
stack, including local HF dependencies, not only the Gemini client.

## Comparing Gemini with Google AI Studio

Reviewed against Google's [Gemini 3.8 migration guide](https://ai.google.dev/gemini-api/docs/generate-content/latest-model)
and [model specification](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash)
on 2026-09-23. The documented generateContent path still includes Gemini 3.8;
Google labels it Legacy while also offering Interactions. That label alone does
not establish a model-quality difference. Model ID, medium thinking by default,
65,536 output tokens, and omitted sampling controls match the model guidance.
The audio-only final user turn was the concrete request mismatch corrected here.

For an informative comparison, use the exact same audio bytes, model version,
system rubric, user text, thinking level, output limit, safety configuration,
response format and tools, with a fresh AI Studio conversation. Compare the code
exported from AI Studio with these settings. Text pasted into its user chat is
not the same message layout as the rubric in this command's system instruction.
The raw command honors GEMINI_SYSTEM_PROMPT; the verifier instead uses its
--prompt-file/default, so check both resolved prompts. No sampling/JSON/safety
settings or conversation history are added by this command. This audit did not
have the user's AI Studio request export and did not run paid comparisons, so it
cannot attribute remaining quality differences or acoustic false rejections.

Google's [response reference](https://ai.google.dev/api/generate-content#FinishReason)
distinguishes provider completion/block reasons from a JSON verdict's `reject`.
Use the saved generation evidence to identify which happened before changing the
rubric. See the [troubleshooting guide](https://ai.google.dev/gemini-api/docs/troubleshooting)
for request errors.

All agent and verifier progress is written to stderr so stdout remains a clean
stream of successful artifact paths. Each run logs its backend/model, item start
and completion, latency, and response size or verifier decision. Token usage and
per-item cost are included when returned by the provider. The final `TOTAL_COST`
line reports the cumulative estimated USD cost and pricing tier; local models and
endpoints without pricing metadata explicitly report cost as unavailable.

## Hardened verifiers (`scripts/s4-agent/verifier/*.sh`)

| Backend | Environment | Model / options |
|---|---|---|
| `gemini` | `.venvs/verify` | as above; adds `_usage`, `_cost` (`paid_batch` / `paid_flex` / `paid_standard`) |
| `hf` | `.venvs/verify` | Gemma 4 E2B / E4B / 12B or any HF audio LLM, optional LoRA adapter; `--max-new-tokens 1024` by default |
| `endpoint` | `.venvs/verify` | any OpenAI-compatible server |
| `unsloth` | `.venvs/verify` | Unsloth Studio (`--model`, `--gguf-variant`, `--payload-mode`, `UNSLOTH_*` env) |
| `vllm` | `.venvs/vllm` | offline vLLM engine (`--dtype`, `--tensor-parallel-size`, …) or `--endpoint` server |
| `moss` | `.venvs/moss` | MOSS-Audio (`--model-id`, `--torch-dtype`, `--trust-remote-code`) |
| `minicpm` | `.venvs/minicpmo` | MiniCPM-o (Python 3.11 env) |
| `kimi` | `.venvs/kimi` | Kimi-Audio (Python 3.11 env) |
| `vibevoice` | `.venvs/vibevoice` | VibeVoice-ASR speaker counting; `--quantization none` / `int8` / `nf4` (`int4` = NF4); `--min-secondary-speech-s` separates `reject` from `uncertain`. INT8/NF4 need NVIDIA CUDA + bitsandbytes |

- `--prompt-file` defaults to `prompts/full-tags-prompt.md`. The prompt text selects
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

Prompts in `prompts/`: `full-tags-prompt.md` is the active verifier default.
It accepts transcribable multilingual speech, including foreign names and code
switching. Each occurrence gets IPA only when no audible deviation from native
American English is heard; other foreign pronunciations use ViePhoneme written
directly from the heard sounds. Brackets come from the ear only: each occurrence
is treated as meaningless sound from a stranger, and reasoning about a word's
language, spelling, romanization, dictionary/IPA or common Vietnamized reading is
treated as knowledge overriding listening. Speakers may mix reading styles within
one word, so no style is assumed. ViePhoneme is a script a Vietnamese reader can
read aloud to mimic the speaker; it need not form valid Vietnamese syllables. A
one-letter-one-sound consonant reading key replaces Vietnamese letters with
dialect-dependent or merged readings (`s`/`x`, `d`/`gi`, `ch`/`tr`); codas and
audible post-coda frication outside the Vietnamese set stay as hyphenated
consonant blocks. No word-specific examples.
The IPA/ViePhoneme choice is made first, per occurrence, by phonemes: an English
word whose sounds, codas/clusters, syllable count and stress match American
English must get IPA regardless of Vietnamese voice quality; it becomes
ViePhoneme only when a concrete Vietnamese-style cue is heard (Vietnamese tone,
substituted sound, dropped/changed coda or cluster, inserted vowel, flattened or
misplaced stress). The ear-only rule applies to writing ViePhoneme. Both the IPA choice and ViePhoneme writing require a
focused second listen to the word's own segment, syllable by syllable
(including aspiration). `unsupported_language` applies only when language content
cannot be reliably transcribed, not merely because it is outside Vietnamese or
English. Audible fillers, including vague ones, remain in order after a dedicated
word-boundary sweep; a pause is silence or a breath intake, so `~` marks
short/medium unexpected pauses and breath catches and `*` marks long pauses within an unfinished sentence. Punctuation
requires audible phrasing or sentence closure. Emotion is decided from the voice before transcription, and follows the voice
when it differs from the content. The default keeps emotion labels
inside `transcript`, with no separate `emotion` field. These are prompt
instructions; the runtime schema validator does not verify phonetic accuracy.

`acoustic_defect.txt` and `acoustic_defect-2.txt` are retained unchanged as
deprecated, reference-only revisions and are not registered validation profiles.
`speaker_purity.txt` and `word_boundary.txt` are narrower verifier alternatives.
Free-form transcript/description prompts (`prompt-transcripts-*.txt`,
`vi-prompt-alam*.txt`) remain available for `scripts/s4-agent/`.
`acoustic_defect-3.txt` combines acoustic verification with conditional
Vietnamese/English transcription and an audible emotion/speaking-style label. It
accepts faint non-intrusive background noise, adds `unsupported_language` and
`singing` eligibility failures, requires nonempty `emotion` and final `transcript`
fields for pass, and forbids the transcript field for reject.
Word boundaries allow tight crops of silence or faint tails after speech sounds
have finished, but still reject clearly truncated phonemes. Music tolerance is
limited to barely perceptible traces that neither mask nor compete with speech
and have no clearly discernible beat or melody; even quiet identifiable beats or
melodies still fail. Other rejection criteria are unchanged.

The HF backend requires a multimodal processor. Gemma 4 is detected from its model
configuration and is loaded only through `AutoModelForMultimodalLM`; it does not
fall back to a text-only loader. Messages, audio loading, tokenization, and feature
construction run together through the processor's multimodal chat template. The
verify environment requires `transformers>=5.10.1` for this API and model family.
If a parsed HF pass omits `emotion` or `transcript`, the backend makes one targeted
repair generation and records `_schema_retry` in the final verdict. Other parse or
schema failures remain failures and retain their raw response for diagnosis.

## Offline tools (no model calls)

### `plot_verifier_analysis.sh --input-dir DIR` (aliases: `analysis.sh`, `analyze.sh`)

All verifiers invoke this script automatically upon completing a run unless `--skip-analysis` / `--no-analyze` is passed. It can also be run or re-run standalone at any time.

All verifier backends update `plot/sample_costs.md` atomically after each saved
result, including failed results, even with `--skip-analysis`. Continuing a run
includes previously saved artifacts and replaces the row for a retried/overwritten
artifact, so samples are not duplicated. The summary reflects the currently saved
artifacts, not a history of overwritten attempts. A partial run therefore retains
its completed results without waiting for plotting; an in-flight request without
a saved artifact is not included. During processing the report lists saved results;
full analysis also adds missing turns from diarization manifests. Standalone
analysis writes this report before rendering plots.

Automatic analysis writes `plot/` inside the run's verdict output directory:
the parent of an explicit `--output-file`, an explicit `--output-dir`, or the
default audio-family directory (for example,
`.data/s4-agent/verifier/gemini/<model>/<effort>/<family>/plot/`). Directory
inputs include nested verdict folders under that output root. To aggregate
multiple families, run analysis explicitly on their parent directory.

Reads every verifier JSON under `DIR` recursively (skipping `work/`, `plot*/`,
`comparisons/`, `experiments/`, hidden dirs), joins diarization manifests found
beside the recorded source audio (or given via `--input-manifest` /
`--manifest-dir`), and writes `DIR/plot/` — see
[data contract §7](data_contract.md#7-verifier-analysis-plot_verifier_analysissh---input-dir-dir--dirplot).
Start with `plot/report.md`. Point it at one model/effort directory for
single-variant figures; a directory with several configurations still gets
per-model statistics and `by_model.png`. Acoustic v3 transcript text, character and
word counts plus emotion labels are exported to both sample CSVs and summarized in
`analysis.json`; `transcripts.png` shows contract outcomes and pass transcript
lengths, while `emotions.png` and `emotions_by_speaker.png` show emotion counts and
durations when available. When cost metadata is available, separate dead-simple plots `cost_distribution.png`
(per-sample cost histogram with mean and median markers) and `cost_total.png` (input, output, and total expenditure bar chart)
are generated alongside `sample_costs.md` (top summary followed by a timestamp-ordered table containing path, start time, end time, pass/not pass, transcripts, and cost). Rerunning refreshes `plot/`; a nonempty
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
`.data/s4-agent/verifier/comparisons/` (or an empty `--output-dir` outside all inputs);
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
`.data/s4-agent/verifier/experiments/NAME/` (`experiment.json`, `templates.json`,
`manifests/*.jsonl`, `audio/`, `reports/`, `checkpoints/`). It refuses an existing
target and performs no model or data operations.

## Handing a run to a transcript reviewer

Use `scripts/s5-export/export_verifier_handoff.sh` with a verifier directory and its
expected audio inventory. The manifest selects its clips recursively; the default
ZIP is `.data/s5-export/<manifest-family>_v1.zip`. An explicit destination is optional. See
[command example](commands.md#dataset-export-s5-export). This standalone offline
command reads existing artifacts; it never regenerates verifier results or provisions
an environment. Existing Python is enough. Each selected clip must have one result;
multiple settings across different clips are recorded, while competing results
produce actionable `--configuration HASH` choices or folders to select.

A finished verifier run can include rejected clips. The export keeps two distinct
states: verifier processing complete/partial, and human transcript review pending.
A complete expected inventory plus valid results proves processing completion;
discovered files alone cannot prove coverage. Resolve missing/failed results using
the original verifier command and settings before export. For Gemini, existing
`--continue` behavior applies; saved Batch failures may require a fresh explicit
regeneration as described above. Export itself never retries or spends model usage.
Alternatively, `--allow-partial` issues a visibly labeled partial review package.

The recipient extracts the ZIP and opens `START_HERE.html` to listen, search,
filter and enter corrected transcripts/review notes. They save progress JSON to
resume and download a review CSV to return. HTML changes are not saved automatically.
Alternatively they edit `clip_catalog.xlsx` in Excel or `clip_catalog.csv`, both
of which include relative audio paths and the original transcripts. XLSX paths are
clickable links. Keep catalogs at the package root; keep all audio folders together.
Choose one editing method; spreadsheet changes do not sync to HTML.

Rejected/unresolved audio is separated in `audio/needs_attention/`, with an audit
CSV. Unavailable/changed non-pass audio is omitted in a partial delivery but its
issue remains visible. Missing transcripts explicitly say `needs_transcription`:
the reviewer can enter a full transcript or exclude the clip. Missing emotion is
shown as not provided. Narrow/custom verifier profiles and VibeVoice do not promise
the acoustic v3 transcript/emotion fields. No invalid model transcript is promoted
into the original transcript column.

Review status is `pending`, `approved` (original text correct), `corrected` (full
replacement supplied), or `exclude`. Human feedback never changes original model
outcomes or resolves processing errors; the sender must assess corrections and
resolve remaining processing separately. This package is a review handoff, not a
declaration that every delivered clip is approved training data.
