# Restructuring Completion and Agent Continuation Log

## Status

The migration to standalone audio-processing commands is complete. All 10 command groups have been fully implemented, legacy applications and classes in `src/` have been removed, documentation has been rewritten, and all files have passed static verification.

## Engineering Ideology & Principles

- **No Implicit Orchestration:** No command chains download → separate → diarize → mix. Callers and scripts compose standalone tools through the filesystem.
- **File-Backed Interfaces:** Data passes through standard WAV files, sibling JSON metadata (`recording.wav` → `recording.json`), and segment manifests (`segments.json`). No in-memory audio waveforms pass across commands.
- **CLI Invariants:**
  - `--input-file` takes precedence over `--input-dir`.
  - For single-output operations, `--output-file` is the exact target destination.
  - Sibling metadata sidecars track provenance, model identity, execution parameters, and output audio properties.
  - Diarization output produces `<stem>/segments.json` and individual turn clips.
  - Manifest-editing purity stages output modified manifests; `scripts/audio/export_segments.py` renders the resulting clips.
- **No Background Workers or Queues:** All commands run sequentially, loading models once per invocation.

## Command Groups Inventory

1. **Download:** `scripts/download/{youtube,playlist,channel}.py`
2. **Separation:** `scripts/separate/{htdemucs,htdemucs_ft,bs_roformer,mel_roformer,mvsep_mdx23}.sh`
3. **Diarization:** `scripts/diarize/{sortformer,pyannote_community1,pyannote_31,clustering,threed_speaker,diarizen}.sh`
4. **Audio Utilities:** `scripts/audio/{info,convert,cut,export_segments,compare_waveforms,compare_spectrograms}.py`
5. **Speaker Operations:** `scripts/speaker/{enroll,score,filter,purity}.py`, `scripts/speaker/{score,purity}.sh`
6. **Purity Stages:** `scripts/purity/{consensus,cleanup,collar,snap,align,segment}.py`, `scripts/purity/align.sh`
7. **Verification:** `scripts/verify/{hf,gemini,endpoint,unsloth,moss,minicpm,kimi,vibevoice}.sh`
8. **Mixing:** `scripts/mix/mix.py`
9. **Evaluation & Plotting:** `scripts/evaluate/{separation,diarization,plot_diarization,plot_metrics}.py`
10. **Dataset Utilities:** `scripts/dataset/{index,filter,export,bundle}.py`

## Documentation & Contracts

- [Command Cookbook and Contracts](scripts/COMMANDS.md)
- [CLI Contract Gateway](docs/api_contract.md)
- [Data & File Contract Gateway](docs/data_contract.md)
- [Repository README](README.md)
- [Scripts Index](scripts/README.md)
- [AGENTS.md Guidelines](AGENTS.md)
