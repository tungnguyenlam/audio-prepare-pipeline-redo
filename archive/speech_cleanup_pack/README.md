# Speech cleanup experiment pack

This pack adds the acoustic-cleaning stages missing from the current diarization pipeline.
It does **not** replace Mel/BS-RoFormer, DiariZen, pyannote, NeMo clustering, Sortformer,
3D-Speaker, consensus, collar, align, snap, or segment.

## Recommended topology

```text
RAW
  -> Mel/BS-RoFormer (existing)              # remove most music
  -> denoise_deepfilternet.py                # hiss / broadband noise
  -> enhance_clearvoice.py                   # residual contamination / speech enhancement
  -> vad_gate_silero.py                      # mute non-speech SFX/music/noise regions
  -> diarization (existing)
  -> consensus + cleanup + collar + align + snap + segment (existing)
  -> for OVERLAP clips only: separate_overlap_clearvoice.py
  -> difficult final clips only: restore_voicefixer.py
```

## Why this order

- Source separation removes accompaniment, but the vocal stem can still contain noise/reverb/artifacts.
- DeepFilterNet is a denoiser, not a music separator.
- ClearVoice enhancement can further improve speech but should be A/B tested for speaker fidelity.
- Silero VAD only mutes regions where nobody is speaking. It cannot remove noise mixed under speech.
- MossFormer2 speech separation is only for overlapping human speakers; do not run it on every clip.
- VoiceFixer uses neural-vocoder restoration and can change timbre. Treat it as an aggressive fallback.

## Suggested isolated environments

### DeepFilterNet

```bash
python -m venv .venv-df
source .venv-df/bin/activate
pip install torch torchaudio deepfilternet soundfile scipy
```

### ClearVoice

```bash
python -m venv .venv-clearvoice
source .venv-clearvoice/bin/activate
pip install clearvoice soundfile scipy
```

### Silero VAD

```bash
python -m venv .venv-vad
source .venv-vad/bin/activate
pip install silero-vad soundfile scipy
```

### VoiceFixer (optional/aggressive)

```bash
python -m venv .venv-voicefixer
source .venv-voicefixer/bin/activate
pip install 'git+https://github.com/haoheliu/voicefixer.git' soundfile scipy
```

## A/B test one vocal stem

Assume the existing RoFormer stage produced `vocals.wav`.

### 1. Denoise

```bash
.venv-df/bin/python denoise_deepfilternet.py \
  -i vocals.wav -o 01_df.wav --post-filter
```

If speech gets too dry, limit attenuation, e.g.:

```bash
.venv-df/bin/python denoise_deepfilternet.py \
  -i vocals.wav -o 01_df.wav --atten-lim-db 18
```

### 2. Enhance residual contamination

```bash
.venv-clearvoice/bin/python enhance_clearvoice.py \
  -i 01_df.wav -o 02_cv.wav \
  --model MossFormer2_SE_48K --device cuda --preserve-sr
```

### 3. Remove non-speech regions from waveform

```bash
.venv-vad/bin/python vad_gate_silero.py \
  -i 02_cv.wav -o 03_speech_only.wav \
  --speech-pad-ms 80 --fade-ms 8 --write-json
```

Use `03_speech_only.wav` as the diarization input for an experiment. Compare it with
`vocals.wav` and `01_df.wav`; aggressive enhancement can occasionally hurt embeddings.

### 4. Separate an overlap clip

```bash
.venv-clearvoice/bin/python separate_overlap_clearvoice.py \
  -i overlap_001.wav -o separated_overlap --device cuda
```

This produces `overlap_001_spk1.wav` and `overlap_001_spk2.wav`. The order is permutation
ambiguous; map the streams to your diarized speaker IDs with TitaNet/CAM++/ECAPA embeddings.

### 5. Aggressive restoration only for bad final clips

```bash
.venv-voicefixer/bin/python restore_voicefixer.py \
  -i bad_clip.wav -o restored.wav --mode 0 --cuda --preserve-sr
```

## One-command cascade

Balanced mode:

```bash
python speech_cleanup_cascade.py \
  -i vocals.wav -o speech_only.wav --mode balanced \
  --df-python .venv-df/bin/python \
  --cv-python .venv-clearvoice/bin/python \
  --vad-python .venv-vad/bin/python \
  --device cuda
```

Aggressive mode adds VoiceFixer:

```bash
python speech_cleanup_cascade.py \
  -i vocals.wav -o speech_only_aggressive.wav --mode aggressive \
  --df-python .venv-df/bin/python \
  --cv-python .venv-clearvoice/bin/python \
  --vad-python .venv-vad/bin/python \
  --vf-python .venv-voicefixer/bin/python \
  --device cuda --keep-intermediate
```

## What each stage targets

| Contamination | Main stage |
|---|---|
| Background music | Existing Mel/BS-RoFormer / HTDemucs / MVSEP |
| Residual music | ClearVoice enhancement + VAD gate when non-speech |
| SFX without speech | Silero VAD gate |
| SFX overlapping speech | ClearVoice may reduce it; no generic model can guarantee perfect removal |
| Hiss / white noise | DeepFilterNet |
| Room reverb / echo | VoiceFixer fallback; evaluate speaker identity carefully |
| Other speaker, non-overlap | Existing diarization + consensus/collar |
| Other speaker, overlap | MossFormer2_SS_16K separation |
| Separator artifacts | ClearVoice; VoiceFixer only for severe cases |
| Boundary bleed | Existing consensus + cleanup + collar + snap |
| General speech quality | ClearVoice enhancement |

## Evaluation rule

Do not select the cleanest-sounding output by ear alone. For each stage compare:

1. speaker embedding cosine similarity to the pre-enhancement vocal stem,
2. ASR WER/CER or transcript stability,
3. DNSMOS/NISQA if available,
4. residual music/noise by listening to a small stratified sample.

If quality improves but speaker similarity drops sharply, reject that enhancement setting for
speaker-diarization/training data.
