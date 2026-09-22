# 08. Boundary Hygiene (cutting without wounding syllables or leaking voices)

[← Concepts Index](README.md) | [Main docs: 03 §3, 04 Stage 3](../03_speaker_diarization.md)

Neural boundaries are approximate (±100–300 ms). Boundary hygiene is the set
of deterministic repairs that (a) shave risky edges near other speakers and
(b) extend or nudge edges to land on silence, syllable ends, and acoustic
zeros.

```mermaid
flowchart TD
    RAW["Raw turn 1.25–5.40"] --> COL["Collar: shave risky edges"]
    COL --> ALI["Alignment: snap outward to syllables"]
    ALI --> VAL["Valley snap: nudge to energy minimum"]
    VAL --> CLEAN["Clean cut"]
```

## 1. Collars: safety margins

A **collar** shaves time inward from a boundary. Two moods:

- **Evaluation collar** (scoring forgiveness, see `04_...`): *ignore* ±collar
  around reference edges when computing DER.
- **Erosion collar** (pipeline action): *delete* ±collar from the turn itself
  (`boundary_collar_s=0.35` in zero-contamination vs 0.04 in light cleanup).

```mermaid
flowchart LR
    subgraph BEFORE["Before: A ends 5.40, B starts 5.45"]
        direction LR
        A1["A ...████"] --> G["5 ms gap"] --> B1["████... B"]
    end
    BEFORE --> AFTER["After 40 ms collar: A ends 5.36, B starts 5.49 — cross-talk sliver gone"]
```

## 2. Context-aware collar: shave near people, extend into silence

`apply_context_aware_collar` looks at *what surrounds* the edge:

```mermaid
flowchart TD
    EDGE["Turn edge"] --> NEAR{"Other speaker within handoff_risk_distance (0.80 s)?"}
    NEAR -->|yes| SHAVE["Shave inward: handoff likely, bleed risk"]
    NEAR -->|no, silence| EXTEND["Grant silence_tail_buffer (+0.027 s): rescue codas/reverb"]
```

**Worked example.** Turn ends 5.40; next speaker starts 5.70 (0.30 < 0.80) →
shave to ~5.05. Same turn ending into 3 s of silence → extend to 5.55, keeping
the `-ng` resonance. Vietnamese codas survive exactly because of the second
branch.

## 3. Pre/post-roll with blockers (cutting for export)

When slicing WAVs, `pad_and_merge_intervals` expands by `pre_roll/post_roll`
to keep natural attack/decay — but expansion **halts at blocker intervals**
(neighboring other-speaker turns) when `stop_at_other_speakers=True`:

```mermaid
flowchart LR
    T["Turn 10.0–14.0"] --> EX["Expand → 9.88–14.20"]
    B["Blocker: other speaker 14.10–16.00"] --> HALT["Halted → 9.88–14.10 (no theft)"]
    EX & HALT --> OUT["Exported clip"]
```

Reverb preserved; neighbor's voice never stolen.

## 4. Energy-valley snapping: cut where the air is still

`snap_boundaries_to_acoustic_valleys` scans ±150 ms (`energy_search_window_s`)
with 2 ms hops, measuring local energy, and moves the edge to the quietest
trough below −30 dB (`energy_valley_floor_db`) — a vocal-fold closure /
zero-crossing:

```mermaid
flowchart TD
    E["Edge at 5.40 (mid-vibration, amplitude 0.35)"] --> SCAN["Scan 5.25–5.55, 2 ms steps"]
    SCAN --> V["Trough at 5.47 (−34 dB, near zero)"]
    V --> S["Snap → 5.47: no click/pop on cut"]
```

Cutting at a vibration peak leaves a discontinuity the ear hears as a click;
cutting at zero is inaudible. Guaranteed local-minimum search; whether a click
*was* audible is empirical.

## 5. Jitter, merging, dropping (light cleanup)

`clean_speaker_turns` (defaults: `min_turn_duration_s=0.5`,
`merge_same_speaker_gap_s=1.0`, `boundary_collar_s=0.04`,
`jitter_max_duration_s=3.0`):

```mermaid
flowchart TD
    J["A-B-A flicker (B < 3 s, no overlap) → relabel B as A"]
    M["A …(0.6 s pause)… A → merge into one turn"]
    D["Leftover < 0.5 s → drop (coughs, blips)"]
```

Order matters: de-jitter → trim collars → merge gaps → drop shorts. The result
is a *derived view*; raw canonical turns are never mutated.

## Where to go next

- Syllable-aware snapping → `06_asr_forced_alignment.md`.
- Parameter directions → `../08_model_parameters_and_tradeoffs.md` §3, §5.
