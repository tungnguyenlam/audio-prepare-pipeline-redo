# GOAL

Read this before any research task. It defines what "done" means, what doesn't
count, and how to work. Repository rules in `AGENTS.md` still apply. When this
file and `research/GRAND_PLAN.md` conflict, this file wins; the plan is the menu
of experiments, this file is the acceptance test.

## 1. The goal

**Build a curation method that rejects more defective clips without discarding more
good speech. Show the gain on human-labelled clips from recordings never used during
development, then show that the gain carries over to a TTS model trained on equal
hours of the curated data.**

A gain counts only if it comes from a better method (better signals, features,
boundaries, or decision procedure). Moving a threshold, choosing a friendlier
metric, or getting closer to Gemini's labels does not count.

### Acceptance criteria (all must hold)

| ID | Criterion | Evidence required |
|---|---|---|
| G1 | **Ground truth exists and is frozen first.** A real test set of ≥ 300 clips from ≥ 3 recordings that no development step has touched, covering at least one studio and one music-backed source. Two human annotators label it blind to every model using the rubric in `docs/experiments.md`. | `.data/goal/test_manifest.jsonl` with recording IDs and SHA-256 per clip, annotation files, Cohen's κ per dimension. Frozen (commit the manifest's hash in `WORKLOG.md`) before any method is tuned. |
| G2 | **The method beats every baseline on the test set.** It achieves lower R = bad accepted / accepted at matched accepted-minute yield (±10 %), or its R–yield curve dominates the baseline's. | Full R–yield curves; 95 % bootstrap CI resampled **by recording**, and the CI of the R reduction must exclude 0. Baselines: no gate (`recipe_nolock`), signal baselines (GRAND_PLAN Phase 2), Gemini 3.5 Flash-Lite with the V2 defect-first prompt, Gemini 3.8 Flash MEDIUM, and Gemma 4 E2B V3. |
| G3 | **The mechanism is understood.** An ablation shows which component produces the gain. On ViCut-Bench, the method's detection-vs-severity curve is compared with the human perceptual threshold for each defect. | Ablation table plus per-defect detection curves with clean-twin false-reject rates. |
| G4 | **It matters downstream.** Train TTS on **equal hours** curated by the new gate and by the best baseline gate (same speaker pool, text, steps, ≥ 2 seeds), then compare the pre-declared primary metrics: CER (PhoASR-whisper-small) and truncated-ending rate on 200 held-out sentences, plus a small listening test. | Every run is reported, including runs that show no gain. A null G4 is still a result: report it and don't rescue it by switching metrics. |
| G5 | **Reproducible.** Every number traces back to a command, its input hashes, and raw outputs under `.data/`. | `WORKLOG.md` entries; unparsed model responses kept. |

### Milestones (Interspeech 2027 deadline ≈ early Feb 2027)

| By | Milestone | Decision if missed |
|---|---|---|
| 2026-10-31 | G1 test set chosen, hashed and frozen; annotation sheets ready for the humans. Free Gemini quota measured. | Tell the user immediately: humans must annotate, and you cannot. |
| 2026-11-15 | Baselines scored on **dev** data; ViCut-Bench defect clips generated. | — |
| 2026-12-01 | A candidate method beats the best baseline on **dev** with a CI excluding 0. | Stop chasing a method. Write the benchmark/audit paper in GRAND_PLAN with honest negative results. |
| 2026-12-20 | One-shot test-set evaluation (G2), ablations (G3), TTS runs started (G4). | — |
| 2027-01 | Writing, reruns, buffer. | — |

## 2. What does not count

Each of these has already happened in this project or is an easy trap. Any one of
them invalidates the claim.

1. **Touching the test set during development.** You may not choose thresholds,
   prompts, checkpoints, seeds, features or hyperparameters from test results.
   Evaluate on the test set once per frozen method and log every evaluation. If you
   run it again, say so and report both runs.
2. **Treating Gemini as ground truth.** The 288-clip gold set
   (`.data/tts_strategy/gold_benchmark_20260908/`) was labelled by Gemini 3.8 Flash
   and is not source-disjoint from old training data, so it is **dev and
   diagnostic data only**. Higher agreement with Gemini is not an improvement. You
   are also Gemini 3.8 Flash, so agreeing with yourself is not evidence.
3. **Buying R with yield.** Rejecting more always lowers R. Every R must come with
   its yield, and accepting nothing is invalid.
4. **Synthetic-only wins.** ViCut-Bench shows mechanism (G3), not the headline
   claim. Synthetic defects carry shortcuts: digital zero pads (PhoAudiobook's 50 ms
   pads), discontinuities at splice points, resampling or level changes. Process the
   clean twin through the same pipeline (same resample, re-encode and gain) and
   check that the detector stays silent on it.
5. **Leakage.** Split by recording and speaker before any augmentation; derived
   audio inherits its parent's split. Check SHA-256 overlap between splits, since 15
   of 31 Khanh Vy cuts were byte-identical to training audio.
6. **Silent failure that looks like a result.** The E2B V3 student passed 288/288
   clips with a low validation loss. Before trusting any model output, check that
   weights and adapters actually loaded, that the input affects the output (silence
   or shuffled audio must change predictions), and that the class distribution is
   not degenerate.
7. **Single runs and cherry-picking.** Report n, the CI, every seed and every
   failed configuration. Use examples only to illustrate an aggregate result.
8. **Invented numbers or citations.** Every claim about a paper needs the PDF in
   `research/papers/`, extracted text in `research/notes/`, and the supporting
   sentence quoted in your notes. Every number must come from a file you produced
   or read. If you don't know something, write "unknown".
9. **Changing the target after seeing results.** If a metric, split or label must
   change, log why, re-freeze it, and report the before and after numbers.

## 3. How to work: be critical

Question every step, including this file, `GRAND_PLAN.md`, earlier numbers in
`docs/`, and your own previous conclusions. If something is wrong, write it in
`WORKLOG.md` and propose a change instead of following it silently.

**Before each experiment**, append to `WORKLOG.md`:

- Hypothesis, and the result that would **refute** it.
- Data split, metric, baseline, and expected cost (GPU-h, Gemini calls).
- Why this is the most informative next experiment.

**After each experiment**, append:

- Result versus prediction, with n and the CI.
- **At least two alternative explanations** and the check that rules each one in
  or out. Default checks: a trivial baseline (duration, RMS, edge energy), label
  shuffle or random features, another recording, and a clean-twin control.
- What you will do next and why.

**Treat these as bugs until proven otherwise:**

- A result far better than every previous system (e.g. > 90 % reject recall when
  the best judge reached ~43 %).
- Perfect separation, or one constant prediction.
- A gain that disappears on another recording or seed.
- A gain that a duration or loudness baseline also achieves.

**Stop and ask the user** when ground truth needs human listening, when a test set
may be contaminated, when a decision changes the paper's claim, or before spending
paid API credit.

## 4. When you hit a dead end

A dead end means two consecutive experiments that don't move the dev metric
outside its CI, or an approach with no concrete next step.

1. **Write it down.** Record what was tried, the numbers, and your best explanation
   of the failure.
2. **Diagnose before switching.** Inspect ≥ 20 failures and sort them into data,
   labels, features, model or decision rule. Measure first: edge energy,
   spectrogram, SNR, overlap detector, speaker-embedding drift.
3. **Read.** Search arXiv, the ISCA Archive (Interspeech) and the ACL Anthology for
   the specific failure, not the general topic. Download useful papers to
   `research/papers/`, extract their text to `research/notes/`, and add a row to
   `research/README.md` stating what each paper does and does not support. Read
   the method and results sections, not only the abstract.
4. **Generate ≥ 3 alternative hypotheses.** Include at least one from a different
   field (e.g. onset detection in MIR, overlapped-speech detection,
   source-separation quality metrics, anomaly detection on clean twins).
5. **Run the cheapest discriminating experiment.** Keep it to ≤ 1 GPU-hour, dev
   data only, and ≤ 20 Gemini calls. Pick the experiment whose outcome would most
   change your next decision.
6. **Timebox.** Give each approach a budget up front. When the budget is spent,
   write up the negative result and move on. A documented dead end is useful in
   the paper.

**Leads already established (verify before relying on them):**

- Aligned word gaps contain almost none of the real pauses. Search around word
  edges instead (`docs/tts_segmentation_testing.md`).
- Energy-valley snapping reduced clipped words (37.5 → 25 % for DiariZen).
- Music bleed and secondary speech, not boundaries, cause most rejects on vlogs.
- Flash-Lite misses quiet music and exact word-edge clipping.
- Whether defect information survives in frozen audio encoders is untested
  (GRAND_PLAN Phase 3). A probe head would be a cheap local gate.

## 5. Checking audio with Gemini 3.8 Flash (free key)

You cannot listen. Use deterministic measurements first: energy at the edges,
spectrograms, SNR, overlap detection and speaker embeddings. Call Gemini only when
those cannot answer the question **and** the answer would change your next action.
Before calling, write the question and the decision it controls in `WORKLOG.md`.

**Allowed:**

- Check whether a generated defect is audible or an anchor is actually clean.
- Spot-check ≤ 10 clips behind a surprising result.
- Run the pre-declared judge-audit subset (GRAND_PLAN Phase 2), trickled over days.

**Not allowed:**

- Labelling G1 test clips.
- Iterating prompts against test data.
- Bulk runs that are not in the plan.

**How:**

```bash
# free-form question about one clip
scripts/s4-agent/gemini.sh -i clip.wav -m gemini-3.8-flash \
  --inference-mode standard --reasoning-effort medium -c 1 \
  --user-prompt "Does the last word end completely, or is its final consonant cut? Give the time." \
  -od .data/goal/gemini_checks/<topic>

# rubric verdicts for a directory (resume across days)
scripts/s4-agent/verifier/gemini.sh -id <dir> -m gemini-3.8-flash \
  --inference-mode standard --reasoning-effort medium -c 1 --continue \
  -od .data/goal/gemini_checks/<topic>
```

- The key comes from `GEMINI_API_KEY` in `.env`. Never print, log or commit it.
- Use the default concurrency of 1 on the free tier. On `RESOURCE_EXHAUSTED`, stop
  and resume the next day with `--continue`; don't retry in a loop.
- Until the daily quota is measured, make ≤ 20 ad-hoc audio calls per day. Log
  each day's call count, clips and purpose in `WORKLOG.md`.
- Treat the verdict as one noisy signal from a model with known blind spots. When
  it disagrees with a measurement, check the measurement again before believing
  either. Keep the raw responses.

## 6. Resources

One RX 9060 XT (16 GB, ROCm), the NVIDIA server `vsf-242` for separation and
diarization, a free Gemini key, an optional OpenRouter key, and the authors' time
for listening. Details are in GRAND_PLAN §1 and `CURRENT_MACHINE.md`. Methods that
need more compute than this are out of scope.
