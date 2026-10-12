# Documentation index

[← Repository README](../README.md)

The repository is a set of standalone commands that communicate through files under
`.data/`. There is no orchestrator, web app or Python package API. If a document
mentions `src/`, `ManagedModel`, `SonicStudio` or a web backend, it is outdated.

## Goal and research

| Document | Contents |
|---|---|
| [`../GOAL.md`](../GOAL.md) | Acceptance criteria and working protocol for research. Read first. |
| [`../research/GRAND_PLAN.md`](../research/GRAND_PLAN.md) | Interspeech 2027 experiment plan: ViCut-Bench, judge audit, probes, TTS dose-response. |
| [`../research/README.md`](../research/README.md) | Downloaded papers and what each actually supports. |
| [`experiments.md`](experiments.md) | Past measurements, metric R, gold set rules, lessons learned. |

## Using the commands

| Document | Contents |
|---|---|
| [`commands.md`](commands.md) | Environments, CLI rules, cookbook for every command group. |
| [`data_contract.md`](data_contract.md) | File formats: sidecars, manifests, verifier outputs, export ZIPs. |
| [`agent_verifier.md`](agent_verifier.md) | Raw audio-LLM generation and pass/reject verifiers. |
| [`hardware.md`](hardware.md) | Machines, ROCm/CUDA notes, sync scripts. |

## Topic notes

| Document | Contents |
|---|---|
| [`tts_segmentation_testing.md`](tts_segmentation_testing.md) | Current sentence-end TTS cutter (`experimental-v3`) recipe. |
| [`tts_segmentation_verification.md`](tts_segmentation_verification.md) | Superseded `experimental-v2` results and why they failed. |
| [`tts_segmentation_proposal.md`](tts_segmentation_proposal.md) | Target design for acoustic-aware segmentation (partly unimplemented). |
| [`vad_segmentation_baseline.md`](vad_segmentation_baseline.md) | One recorded run of the VAD-only baseline. |
| [`tts_training_and_evaluation.md`](tts_training_and_evaluation.md) | TTS training, Vietnamese evaluation metrics, ASR models. |
| [`verifier_batch_review.md`](verifier_batch_review.md) | Historical review of Gemini Batch verifier behavior. |
| [`gemini_batch_debugging.md`](gemini_batch_debugging.md) | Read-only snippets for diagnosing Gemini Batch/quota failures. |
| [`verifier_cost_summary_vi.md`](verifier_cost_summary_vi.md) | Vietnamese management summary of yield and Gemini cost (2026-09-25). |
