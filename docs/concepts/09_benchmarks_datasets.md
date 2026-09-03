# 09. Benchmarks & Datasets (reading DER tables without fooling yourself)

[← Concepts Index](README.md) | [Main docs: benchmarks](../bench-paper-diarize.md) | [Mixing](../05_benchmark_and_mixing.md)

A DER number is meaningless without its **protocol** (collar? overlap scored?)
and its **dataset** (meeting room? YouTube? Vietnamese?). This guide teaches
both.

## 1. The protocol triple

```mermaid
flowchart TD
    D["DER value"] --> C["Collar: 0 s (strict) vs 0.25 s (forgiving)"]
    D --> O["Overlap: included (hard) vs skipped (easier)"]
    D --> A["Automatic (no oracle count/VAD) vs oracle-aided"]
```

Rule: only compare rows that share all three. Pyannote community-1's card
(0 s collar, overlap included, fully automatic) vs Sortformer's DIHARD3-Eval
≤4-speaker subset is **not** apples-to-apples — the subset restriction alone
removes the hardest files.

## 2. Dataset personalities

```mermaid
flowchart TD
    AMI["AMI (Edinburgh meetings): far-field SDM vs headset IHM — tests reverb"]
    ALI["AliMeeting / AISHELL-4 (Mandarin meetings): tonal language, overlap-heavy"]
    VOX["VoxConverse (YouTube talk): in-the-wild noise, many speakers"]
    DIH["DIHARD 3 (deliberately hard): clinical, telephone, restaurant chaos"]
    CALL["CALLHOME (phone calls, 2–4 speakers): narrowband 8 kHz legacy"]
    VIYT["ViYT-Diar (100 Vietnamese YouTube files): OUR target domain"]
```

| Dataset | What stresses | Why it matters here |
|---|---|---|
| AMI SDM / IHM | Distant mics vs headsets | Ingest is YouTube-compressed, closer to SDM |
| AliMeeting / AISHELL-4 | Overlap + tonal speech | Closest public proxy for Vietnamese talk shows |
| VoxConverse | Wild acoustics, speaker count | Tests `max_speakers` discipline |
| DIHARD 3 | Everything adversarial | General robustness ceiling |
| CALLHOME | Telephone band | Mostly legacy; less relevant to 16–44.1 kHz YouTube |
| ViYT-Diar | Vietnamese YouTube, manual labels | The benchmark to run next — no public checkpoint has a verified number |

## 3. The three-way picture (shared protocol)

On the four common sets (lower = better), DiariZen v2 leads, 3D-Speaker and
Pyannote trade blows:

```mermaid
flowchart LR
    DZ["DiariZen: 10.1 / 10.8 / 13.9 / 9.1"]
    PY["Pyannote C-1: 11.7 / 20.3 / 19.9 / 11.2"]
    TD["3D-Speaker: 10.3 / 19.7 / 21.8 / 11.8"]
    DZ --> NOTE["Strongest on overlap-heavy AliMeeting"]
```

Sets: AISHELL-4 / AliMeeting / AMI SDM / VoxConverse. Full table with sources
→ `../bench-paper-diarize.md`.

## 4. Why the clustering row is blank (a lesson in oracle vs real)

NeMo publishes stellar TitaNet-clustering DER (~1% on AMI Lapel) — but with
**oracle VAD** (perfect speech boundaries handed to the clusterer). This repo
runs **MarbleNet-predicted VAD**, whose errors are the dominant term. Copying
oracle numbers into our table would be dishonest; the blank row *is* the
correct documentation.

```mermaid
flowchart TD
    OR["Oracle VAD + clustering: ~1% DER (lab)"] --> REAL["MarbleNet VAD + clustering: much higher (real)"]
    REAL --> L["Lesson: VAD errors dominate; never compare across VAD conditions"]
```

## 5. Separation benchmarks are mixtures, not datasets

`AudioMixer` *manufactures* tests: clean speech + music at calibrated SMRs
(`[-5, 0, 5, 10]` dB) with explicitly seeded music crops (e.g. `seed=42` →
bit-exact reruns; `seed` is a required argument, no default).
Harder SMR = lower vocals SDR expected. See `01_audio_fundamentals.md` §4 for
the dB math.

## Where to go next

- Scoring mechanics → `04_evaluation_der_jer_hungarian.md`.
- Running mixes → `../05_benchmark_and_mixing.md`.
