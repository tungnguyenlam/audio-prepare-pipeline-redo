# Verifier Batch review — 2026-09-25

Historical review of the implementation before incremental retrieval. The
subsequent implementation now checkpoints each completed job and publishes
verifier samples while other jobs run. `--continue` retries already-published
failed samples once per invocation and reconnects any in-flight retries. Available
failure usage/cost is preserved, Batch item failures log as failures and verifier
exit status is nonzero. Poll intervals are no longer capped at 60 seconds. CLI
state reuse is gated by the full run signature. See the current
[verifier guide](agent_verifier.md) for behavior and retry/cost limitations.

The findings and example below describe the original review snapshot; statements
about waiting for all jobs and replaying published failed samples are superseded.

## Assessment

Python syntax and the Gemini launcher/help wiring pass. The successful Gemini
Batch path uses the currently documented REST API and correct current list-price
arithmetic. Failure reporting, resume identity and large-run handling have gaps;
this is not an end-to-end certification.

Only Gemini implements provider-side asynchronous Batch. Endpoint, HF, Kimi,
MiniCPM, MOSS, Unsloth, VibeVoice and vLLM verifiers share local item grouping;
`--batch-size` there does not enable Google's Batch pricing or scheduling.

## Findings

1. **Local dependency blocker:** `.venvs/verify` has httpx 0.28.1 but no
   `google-genai` distribution. The Gemini constructor imports the SDK even in
   REST Batch mode, so this environment cannot run it yet. `--help` exits before
   constructor initialization. Requirements specify `google-genai>=1.56.0`, not
   an exact latest version; PyPI reports 2.25.0 at review time. Remote environments
   and a `VERIFIER_PYTHON` override were not inspected.
2. **Resume identity can reuse the wrong answer:** `GeminiAgent.generate_batch`
   replaces all `contents` with empty parts when hashing request identity, losing
   `user_prompt`. Changing only that prompt changes the outer run manifest but
   can still find the old inner session and replay its results. The inner session
   also omits batch size; changing group sizes can reuse completed results or
   resubmit differently grouped work while old jobs remain active.
3. **Failed-response metadata is lost:** `verdict_processor` initializes
   `generation=None` and never captures `exc.generation`. JSON parsing failures
   and provider completion failures therefore lose usage/cost in verdict artifacts
   and item logs. Schema failures preserve `invalid_verdict`, but the item log
   omits it. Terminal provider totals can include costs absent from per-item plots.
4. **Failure exit status is misleading:** handled model/validation failures write
   `status=fail` artifacts and return normally. The shared `batch()` helper counts
   them as `ITEM_DONE`, can print `BATCH_COMPLETE ... 0 failed`, and returns zero.
   `TOTAL_COST`'s failed count and artifact status are more informative. A valid
   acoustic `reject` is correctly treated as a successful verdict.
5. **Interrupted-run costs are incomplete:** requests are counted only when
   responses are converted after all jobs finish. Timeout/interruption can print
   `$0 ... no model requests` despite submitted, potentially billable jobs. Missing
   usage metadata becomes a zero-dollar estimate rather than unknown. Replaying a
   saved batch counts its old response costs again in that invocation's summary;
   do not add summaries from resumed invocations to estimate new spending.
6. **Polling is not an exact deadline:** sleep is `min(interval, 60)`, so values
   above 60 seconds are silently shortened. The timeout starts after submission
   and is checked after a whole polling sweep; HTTP waits/retries can overrun it.
   All jobs must finish before verdicts and sample-cost files are published.
   Per-item latency is the shared polling duration, not individual inference time;
   sample-cost table start/end values are audio clip positions, not job timestamps.
7. **Large runs are not bounded by batch size:** all audio is base64-encoded and
   retained before submission. All groups are submitted before polling, without
   a cap on outstanding jobs; `--concurrency` does not throttle provider jobs.
   Large runs can exhaust memory or provider job/enqueued-token quotas. The
   documented concurrent Batch limit is 100; actual project limits also apply.
8. **Continuation has limits:** a successful job containing an individual request
   error, invalid JSON or invalid verdict is replayed on `--continue`, not newly
   generated. This is documented but broader CLI help suggests failed outputs
   will be retried. Whole failed provider jobs are resubmitted. A malformed output
   JSON can also abort preflight because `read_json` errors are not handled there.
9. **Submission is not exactly-once:** network failures are deliberately not
   retried for POST submission, but retryable HTTP status codes still are. There
   is no idempotency/reconciliation mechanism for acceptance before a lost reply
   or interruption before the job name is saved. Inspect provider jobs before
   restarting an ambiguous submission.

## Running the current implementation

Use an existing environment containing the declared verifier dependencies and
provide `GEMINI_API_KEY` through the environment or the local `.env`; do not put
credentials in CLI flags. No environment was installed by this review.

Start with a small input directory. From the repository root:

```bash
scripts/s4-agent/verifier/gemini.sh \
  --input-dir .data/my-clips \
  --output-dir .data/verifier-batch-review/verdicts \
  --work-dir .data/verifier-batch-review/work \
  --prompt-file prompts/full-tags-prompt.md \
  --inference-mode batch \
  --batch-size 10 \
  --batch-poll-interval-s 60 \
  --batch-timeout-s 90000 \
  --timeout-s 120 \
  --max-retry 4 \
  --reasoning-effort medium \
  --max-tokens 65536 \
  --concurrency 1
```

`--batch-size 10` is a maximum requests/job, not a minimum, GPU batch size, or cap
on outstanding jobs. Requests are split again at an internal 18 MiB serialized
entry budget below Google's 20 MB inline limit. An individually oversized request
is rejected locally; this implementation does not upload JSONL batch files.

The default batch timeout is 86400 seconds (24 hours); the example allows 25
hours. Google's target is 24 hours, often sooner, not an exact completion promise.
Polling more frequently does not speed inference. `--timeout-s` controls each
HTTP request; `--max-retry 4` permits up to five HTTP attempts and does not retry
bad Batch answers. `--max-response-retries` also does not resubmit Batch answers.
Leave prompt caching off initially: enabling it requires a minimum 90000-second
TTL, sufficient cacheable prompt size, and incurs full-TTL storage charges.

Expect configuration/preflight logs, one submitted-job line per group, then job
state lines each polling sweep on stderr. `VERIFIER_START`, per-item progress,
verdict/text output pairs, `plot/sample_costs.md` and final cost reporting follow
only after all jobs terminate. Successful artifact publication paths go to stdout.
Post-run analysis runs by default; `--skip-analysis` disables it. There is no
provider-completion percentage or ETA in the polling log.

For an interrupted wait, preserve inputs, prompts, settings, output and work
paths and rerun the same command with `--continue`. Local timeout/Ctrl-C does not
cancel the remote jobs. Do not use `--overwrite` to resume: it creates fresh paid
requests for every selected input. For a bad individual response, select only
that clip with `--input-file` and its exact `--output-file`, then use `--overwrite`
to regenerate it; overwriting the original directory run regenerates all clips.
Do not change user prompt or batch size while resuming due to finding 2.

## Costs and API currency

For Gemini 3.8 Flash, current Batch estimates are:

```text
USD = uncached_input_tokens * 0.375 / 1,000,000
    + cached_input_tokens * 0.0375 / 1,000,000
    + (output_tokens + thinking_tokens) * 1.875 / 1,000,000
    + explicit_cache_tokens * TTL_hours * 0.50 / 1,000,000
```

The default has no explicit cache storage. Rates match Google's introductory
pricing through December 31, 2026. The hardcoded rate card has no automatic date
rollover for January 2027. These are estimates, not billing-ledger totals.

Batch uses httpx with `v1beta/models/{model}:batchGenerateContent` and GET polling;
it does not use `client.batches.create()`. This endpoint and inline payload are
still documented. Standard inference uses the official Google Gen AI SDK.
A REST implementation is not obsolete simply because it does not use SDK Batch.

Sources checked: [Batch API](https://ai.google.dev/gemini-api/docs/batch-api),
[pricing](https://ai.google.dev/gemini-api/docs/pricing),
[rate limits](https://ai.google.dev/gemini-api/docs/rate-limits),
[Google Gen AI SDK distribution](https://pypi.org/project/google-genai/).

Validation: AST parsing of all 24 Python files under `scripts/s4-agent` plus
`scripts/_common/files.py`, Bash syntax for both Gemini launchers, and verifier
launcher `--help` passed. No tests were written or run; actual submission,
polling, response parsing, cancellation and account billing remain unverified.
