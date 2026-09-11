# Hardened audio verifier behavior

These commands turn raw audio-model generation into validated pass/reject
verdict JSON. They use the shared generation clients in `scripts/agent/`, add
the acoustic verifier prompt/response constraints, parse model output, and
enforce the verdict contract.

Supported verifier backends are Gemini, OpenAI-compatible endpoint, Hugging
Face, Unsloth, vLLM, MOSS-Audio, MiniCPM-o, Kimi-Audio, and VibeVoice-ASR.
`evaluate_verifier.py` compares their verdict files against reference verdicts.
`scaffold_experiment.py` creates local workspaces for verifier behavior
development under `.data/agent/verifier/experiments/`.

Default outputs follow the hardened behavior hierarchy:

```text
.data/agent/verifier/<backend>/<family>/
```

Verifier commands produce verdict JSON only. Raw `.txt` plus `.json` response
pairs are produced by the parent commands such as `scripts/agent/gemini.sh`.

```bash
bash scripts/agent/verifier/gemini.sh --input-dir .data/clips
bash scripts/agent/verifier/hf.sh --input-dir .data/clips
bash scripts/agent/verifier/endpoint.sh --input-dir .data/clips
```

Every command remains standalone; this directory does not orchestrate agent
exploration, verification, or any other pipeline stage.
