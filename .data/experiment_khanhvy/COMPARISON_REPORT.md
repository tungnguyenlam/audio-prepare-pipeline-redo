# 3-Way Acoustic Verification Benchmark: Gemini 3.8 Flash vs Gemma 4 12B vs Gemma 4 E4B

- **Source Audio:** `VLOG_Tháp_tùng_chị_gái_nhận_hàm_PGS_Đằng_sau_các_thành_tựu_viral_là_gì_(ft.YouTube_Works_Awards)__H0VpjeULCck.wav`
- **Total Candidate Turns:** 31
- **Gemini 3.8 Flash vs Gemma 4 12B Agreement:** **8/31 (25.8%)**
- **Gemma 4 E4B vs Gemma 4 12B Agreement:** **21/31 (67.7%)**
- **Gemini 3.8 Flash vs Gemma 4 E4B Agreement:** **16/31 (51.6%)**
- **Full 3-Way Consensus (All Models Agree):** **8/31 (25.8%)**

## Summary Metrics

| Metric | Google Gemini 3.8 Flash (`thinkingLevel=LOW`) | Gemma 4 12B (Local Unsloth / ROCm) | Gemma 4 E4B (Local Unsloth / ROCm) |
| :--- | :--- | :--- | :--- |
| Passed Turns | 12 (38.7%) | 24 (77.4%) | 27 (87.1%) |
| Rejected Turns | 19 (61.3%) | 7 (22.6%) | 4 (12.9%) |
| Avg Latency | 2.66s | 14.92s | 5.56s |
| Hardware | Google Cloud API | AMD Radeon (12.87 GB VRAM) | AMD Radeon (9.47 GB VRAM) |

## Turn-by-Turn Audit Table

| Turn | Dur | Audio Cut | Gemini 3.8 Flash | Gemma 4 12B | Gemma 4 E4B | Consensus |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: |
| `turn_001_spk_00_0.10-1.85` | 1.75s | [turn_001_spk_00_0.10-1.85.wav](cuts/turn_001_spk_00_0.10-1.85.wav) | **reject** (The final syllable is cut off abruptly before the natural acoustic decay of the vowel/coda finishes.) | **pass** (Single speaker present with clear word boundaries at both start and end.) | **reject** (The speech ends abruptly, cutting off the final word before its full tonal contour and coda closure are completed.) | 🤝 2-1 |
| `turn_002_spk_00_2.19-5.28` | 3.08s | [turn_002_spk_00_2.19-5.28.wav](cuts/turn_002_spk_00_2.19-5.28.wav) | **pass** (The utterance is clear with a single speaker throughout, containing natural acoustic boundaries without clipping or background speaker intrusion.) | **error** | **pass** (Single speaker is present throughout. The audio starts with a clear phonetic onset and ends with the final word completing its tonal contour and coda into natural silence.) | 🤝 2-1 |
| `turn_003_spk_01_6.85-7.76` | 0.92s | [turn_003_spk_01_6.85-7.76.wav](cuts/turn_003_spk_01_6.85-7.76.wav) | **pass** (The utterance starts cleanly with a complete onset and ends naturally with fully articulated coda and trailing silence, featuring only a single speaker.) | **pass** (Single speaker with clear phonetic onset and completion of the final word.) | **pass** (The audio contains a single, clear speaker with no background noise or secondary voices. The utterance starts with a full phonetic onset and ends with a complete closure into natural silence.) | ✅ All 3 |
| `turn_004_spk_00_8.56-10.44` | 1.87s | [turn_004_spk_00_8.56-10.44.wav](cuts/turn_004_spk_00_8.56-10.44.wav) | **pass** (The audio contains a single speaker with natural silences at both boundaries and no clipping or secondary voices.) | **pass** (The audio features a single speaker with clear, complete word boundaries at both the beginning and the end.) | **pass** (The audio contains a single speaker with no secondary voices or tail intrusions. Both the start and end of the speech are acoustically complete and natural.) | ✅ All 3 |
| `turn_005_spk_00_10.83-19.84` | 9.01s | [turn_005_spk_00_10.83-19.84.wav](cuts/turn_005_spk_00_10.83-19.84.wav) | **reject** (The audio cuts off abruptly at the end while the final word is still actively being articulated, cutting its acoustic decay and coda short.) | **pass** (Single speaker with clear word boundaries at both start and end.) | **pass** (Single speaker is present throughout. Speech starts and ends on complete, natural acoustic word boundaries with no clipped segments or trailing secondary voices.) | 🤝 2-1 |
| `turn_006_spk_02_20.87-21.92` | 1.05s | [turn_006_spk_02_20.87-21.92.wav](cuts/turn_006_spk_02_20.87-21.92.wav) | **pass** (The clip contains only one speaker with complete phonetic boundaries and no tail intrusion or clipping.) | **error** | **pass** (The audio contains a single speaker with no background noise or secondary voices. Both the start and end of the speech are acoustically complete, with no clipped boundaries.) | 🤝 2-1 |
| `turn_007_spk_00_23.08-28.06` | 4.97s | [turn_007_spk_00_23.08-28.06.wav](cuts/turn_007_spk_00_23.08-28.06.wav) | **reject** (An initial secondary speaker is audible before the main speaker, and the final word is clipped abruptly at the end.) | **error** | **reject** (The audio ends abruptly mid-word ('Cách'), failing to complete the final consonant closure and tonal contour into natural silence.) | 🤝 2-1 |
| `turn_008_spk_00_28.16-36.02` | 7.86s | [turn_008_spk_00_28.16-36.02.wav](cuts/turn_008_spk_00_28.16-36.02.wav) | **reject** (The initial syllable is abruptly truncated at the start of the recording, cutting off its initial consonant onset.) | **pass** (The audio contains a single speaker with clear, complete phonetic boundaries at both the start and the end, with no tail intrusions.) | **pass** (The audio contains a single, clear speaker with no background noise or secondary voices. Both the start and end of the speech are acoustically complete, with full phonetic onsets and natural silence following the final word.) | 🤝 2-1 |
| `turn_009_spk_00_36.28-40.70` | 4.42s | [turn_009_spk_00_36.28-40.70.wav](cuts/turn_009_spk_00_36.28-40.70.wav) | **reject** (The audio cuts off abruptly at the very end while the final syllable's tone and vowel are still actively sounding, cutting off the natural decay.) | **pass** (Single speaker with clear phonetic onset and coda closure at both boundaries.) | **pass** (Single speaker with no background noise. Speech starts and ends on complete word boundaries.) | 🤝 2-1 |
| `turn_010_spk_00_41.19-42.76` | 1.57s | [turn_010_spk_00_41.19-42.76.wav](cuts/turn_010_spk_00_41.19-42.76.wav) | **reject** (The final syllable is abruptly cut off mid-vocalization at the end of the clip.) | **pass** (Single speaker with clear, complete word boundaries at both start and end.) | **pass** (The audio contains a single speaker with no overlap or tail intrusion. Both the start and end of the speech are acoustically complete.) | 🤝 2-1 |
| `turn_011_spk_02_43.75-46.72` | 2.97s | [turn_011_spk_02_43.75-46.72.wav](cuts/turn_011_spk_02_43.75-46.72.wav) | **reject** (A second speaker speaks clearly in the latter half of the clip.) | **pass** (Single speaker with clear, complete word boundaries at both start and end.) | **pass** (The audio contains a single speaker with no background noise or secondary voices. Both the start and end of the speech are complete and natural.) | 🤝 2-1 |
| `turn_012_spk_00_48.36-51.29` | 2.93s | [turn_012_spk_00_48.36-51.29.wav](cuts/turn_012_spk_00_48.36-51.29.wav) | **pass** (The audio contains a single speaker throughout with clean, complete start and end acoustic boundaries and no secondary speaker intrusion.) | **pass** (Single speaker with clear, complete word boundaries at both start and end.) | **pass** (The audio contains a single speaker with clear, complete word boundaries from start to finish, and no secondary vocal intrusions are present.) | ✅ All 3 |
| `turn_013_spk_00_52.71-53.94` | 1.23s | [turn_013_spk_00_52.71-53.94.wav](cuts/turn_013_spk_00_52.71-53.94.wav) | **reject** (The final syllable is abruptly cut off before its coda closure and vowel decay are complete.) | **pass** (Single speaker with clear phonetic onset and complete coda closure at the end.) | **pass** (The audio contains a single speaker with no background noise or secondary voices. Both the start and end of the speech are acoustically complete, adhering to natural word boundaries.) | 🤝 2-1 |
| `turn_014_spk_00_59.87-74.94` | 15.07s | [turn_014_spk_00_59.87-74.94.wav](cuts/turn_014_spk_00_59.87-74.94.wav) | **pass** (The speech is cleanly delivered by a single speaker with full phonetic word boundaries at both ends.) | **pass** (Single speaker with clear word boundaries at both start and end.) | **pass** (The audio contains a single speaker with no overlapping speech or secondary vocal intrusions, including the final 500ms. Both the beginning and end of the speech occur on complete, natural acoustic word boundaries.) | ✅ All 3 |
| `turn_015_spk_02_80.51-84.50` | 3.99s | [turn_015_spk_02_80.51-84.50.wav](cuts/turn_015_spk_02_80.51-84.50.wav) | **reject** (The final syllable is cut off abruptly before its full tone and acoustic energy decay naturally.) | **error** | **pass** (The audio contains a single speaker with no overlap or tail intrusion. Both the start and end of the speech segment are on complete, natural word boundaries.) | ❌ Divergent |
| `turn_016_spk_02_87.24-88.96` | 1.72s | [turn_016_spk_02_87.24-88.96.wav](cuts/turn_016_spk_02_87.24-88.96.wav) | **reject** (The final syllable is abruptly cut off mid-vocalization before reaching natural acoustic decay.) | **pass** (Single speaker with complete phonetic boundaries at both start and end.) | **pass** (The audio contains a single speaker with no overlap or trailing vocal intrusions. Both the start and end of the speech occur on complete, natural word boundaries.) | 🤝 2-1 |
| `turn_017_spk_00_99.36-101.73` | 2.38s | [turn_017_spk_00_99.36-101.73.wav](cuts/turn_017_spk_00_99.36-101.73.wav) | **reject** (The initial word is cut off abruptly mid-syllable at the onset, and multiple background voices/laughter are audible throughout the clip.) | **pass** (Single speaker with clear phonetic boundaries at both start and end.) | **pass** (The audio contains a single speaker with no overlaps or secondary voices. The speech starts with a full phonetic onset and ends with a complete word closure into natural silence.) | 🤝 2-1 |
| `turn_018_spk_01_104.28-106.01` | 1.73s | [turn_018_spk_01_104.28-106.01.wav](cuts/turn_018_spk_01_104.28-106.01.wav) | **reject** (There is a prominent secondary speaker or background voice audible at the start of the audio clip.) | **pass** (Single speaker with complete word boundaries at both start and end.) | **reject** (The audio is pure, but the final word is clipped at the end of the segment.) | 🤝 2-1 |
| `turn_019_spk_02_107.94-109.57` | 1.63s | [turn_019_spk_02_107.94-109.57.wav](cuts/turn_019_spk_02_107.94-109.57.wav) | **reject** (There are two distinct speakers taking turns in this short recording.) | **pass** (Single speaker throughout with no tail intrusions; speech starts and ends on complete word boundaries.) | **pass** (Single speaker with no overlap or tail intrusion. Speech starts and ends on complete word boundaries.) | 🤝 2-1 |
| `turn_020_spk_01_108.79-111.29` | 2.51s | [turn_020_spk_01_108.79-111.29.wav](cuts/turn_020_spk_01_108.79-111.29.wav) | **reject** (There is overlapping speech between multiple speakers at the beginning, and the speech cuts off abruptly mid-vowel at the end.) | **pass** (Single speaker with complete word boundaries at start and end.) | **pass** (The audio contains a single speaker with no overlaps or secondary voices. The speech starts and ends on complete, natural acoustic word boundaries.) | 🤝 2-1 |
| `turn_021_spk_01_113.12-114.25` | 1.12s | [turn_021_spk_01_113.12-114.25.wav](cuts/turn_021_spk_01_113.12-114.25.wav) | **pass** (The speech starts and ends cleanly with clear word boundaries, natural decay, and no secondary speaker or overlapping speech.) | **error** | **pass** (Single speaker is present, and both the start and end of the speech are acoustically complete.) | 🤝 2-1 |
| `turn_022_spk_00_115.32-118.62` | 3.31s | [turn_022_spk_00_115.32-118.62.wav](cuts/turn_022_spk_00_115.32-118.62.wav) | **reject** (Heavy background chatter and overlapping secondary voices are present throughout the recording.) | **error** | **pass** (The audio contains a single speaker with no secondary voices or tail intrusion. Both the start and end of the speech are acoustically complete.) | ❌ Divergent |
| `turn_023_spk_00_119.25-121.27` | 2.01s | [turn_023_spk_00_119.25-121.27.wav](cuts/turn_023_spk_00_119.25-121.27.wav) | **pass** (The utterance is produced by a single speaker with cleanly preserved acoustic boundaries at both onset and offset.) | **error** | **pass** (The audio contains a single speaker, and both the start and end of the speech are acoustically complete and natural.) | 🤝 2-1 |
| `turn_024_spk_01_122.05-123.22` | 1.17s | [turn_024_spk_01_122.05-123.22.wav](cuts/turn_024_spk_01_122.05-123.22.wav) | **pass** (The audio contains a single speaker throughout with clear acoustic onset and natural ending fade without clipping.) | **pass** (Single speaker with clear word boundaries at both start and end.) | **pass** (Single speaker with no overlap or trailing voices. Speech starts and ends on complete word boundaries.) | ✅ All 3 |
| `turn_025_spk_01_125.51-126.74` | 1.23s | [turn_025_spk_01_125.51-126.74.wav](cuts/turn_025_spk_01_125.51-126.74.wav) | **reject** (Multiple speakers are audible with overlapping voices in the background, and the audio abruptly cuts off mid-syllable at the end.) | **pass** (Single speaker with clear word boundaries at both start and end.) | **pass** (The audio contains a single speaker with no overlap or tail intrusion. Both the start and end of the speech are on complete, natural word boundaries.) | 🤝 2-1 |
| `turn_026_spk_00_130.36-139.84` | 9.48s | [turn_026_spk_00_130.36-139.84.wav](cuts/turn_026_spk_00_130.36-139.84.wav) | **pass** (The recording contains a single female speaker throughout with clean, complete start and end boundaries.) | **pass** (Single speaker with complete word boundaries at both start and end.) | **pass** (Single speaker throughout the clip with no secondary voices or tail intrusion. Both the start and end of the speech are acoustically complete.) | ✅ All 3 |
| `turn_027_spk_00_154.39-157.66` | 3.27s | [turn_027_spk_00_154.39-157.66.wav](cuts/turn_027_spk_00_154.39-157.66.wav) | **reject** (The final syllable is cut off abruptly before the vowel/coda completes naturally.) | **pass** (Single speaker with clear phonetic onset and full coda closure at both boundaries.) | **pass** (The audio features a single speaker with no overlap or tail intrusion. Both the start and end of the speech occur on complete, natural word boundaries.) | 🤝 2-1 |
| `turn_028_spk_01_158.46-161.50` | 3.05s | [turn_028_spk_01_158.46-161.50.wav](cuts/turn_028_spk_01_158.46-161.50.wav) | **reject** (The final syllable is cut off abruptly while vocal fold vibration is still actively sustained.) | **pass** (Single speaker with clear word boundaries at both start and end.) | **reject** (The audio cuts off abruptly while the final word is still in flight, failing to complete its tonal contour and coda closure.) | 🤝 2-1 |
| `turn_029_spk_01_163.21-164.51` | 1.3s | [turn_029_spk_01_163.21-164.51.wav](cuts/turn_029_spk_01_163.21-164.51.wav) | **pass** (The clip contains a single speaker with fully articulated onsets and endings without any acoustic clipping or intruding voices.) | **pass** (Single speaker with clear word boundaries at both start and end.) | **pass** (Single speaker is present throughout. Both the start and end of the speech are acoustically complete with no clipped boundaries or secondary speaker intrusion.) | ✅ All 3 |
| `turn_030_spk_00_175.90-177.58` | 1.68s | [turn_030_spk_00_175.90-177.58.wav](cuts/turn_030_spk_00_175.90-177.58.wav) | **pass** (The audio contains a single female speaker with clearly defined word boundaries at both the start and end of the segment, with no clipped speech or secondary speaker intrusion.) | **pass** (Single speaker with clear phonetic onset and coda closure at both boundaries.) | **pass** (The audio contains a single speaker with complete word boundaries at the start and end, and no secondary vocal intrusions.) | ✅ All 3 |
| `turn_031_spk_02_178.18-180.00` | 1.82s | [turn_031_spk_02_178.18-180.00.wav](cuts/turn_031_spk_02_178.18-180.00.wav) | **reject** (The final word is cut off abruptly while vocalization and tone are still actively ongoing.) | **pass** (Single speaker with clear, complete word boundaries at both start and end.) | **pass** (The audio contains a single speaker, and both the start and end of the speech segment are acoustically complete and natural.) | 🤝 2-1 |

## Disagreement Deep Dive

### Turn `turn_001_spk_00_0.10-1.85` (1.75s)
- **Audio Cut:** [Listen](data/experiment_khanhvy/cuts/turn_001_spk_00_0.10-1.85.wav)
- **Gemini 3.8 Flash:** `reject` — The final syllable is cut off abruptly before the natural acoustic decay of the vowel/coda finishes.
- **Gemma 4 12B:** `pass` — Single speaker present with clear word boundaries at both start and end.
- **Gemma 4 E4B:** `reject` — The speech ends abruptly, cutting off the final word before its full tonal contour and coda closure are completed.

### Turn `turn_002_spk_00_2.19-5.28` (3.08s)
- **Audio Cut:** [Listen](data/experiment_khanhvy/cuts/turn_002_spk_00_2.19-5.28.wav)
- **Gemini 3.8 Flash:** `pass` — The utterance is clear with a single speaker throughout, containing natural acoustic boundaries without clipping or background speaker intrusion.
- **Gemma 4 12B:** `error` — None
- **Gemma 4 E4B:** `pass` — Single speaker is present throughout. The audio starts with a clear phonetic onset and ends with the final word completing its tonal contour and coda into natural silence.

### Turn `turn_005_spk_00_10.83-19.84` (9.01s)
- **Audio Cut:** [Listen](data/experiment_khanhvy/cuts/turn_005_spk_00_10.83-19.84.wav)
- **Gemini 3.8 Flash:** `reject` — The audio cuts off abruptly at the end while the final word is still actively being articulated, cutting its acoustic decay and coda short.
- **Gemma 4 12B:** `pass` — Single speaker with clear word boundaries at both start and end.
- **Gemma 4 E4B:** `pass` — Single speaker is present throughout. Speech starts and ends on complete, natural acoustic word boundaries with no clipped segments or trailing secondary voices.

### Turn `turn_006_spk_02_20.87-21.92` (1.05s)
- **Audio Cut:** [Listen](data/experiment_khanhvy/cuts/turn_006_spk_02_20.87-21.92.wav)
- **Gemini 3.8 Flash:** `pass` — The clip contains only one speaker with complete phonetic boundaries and no tail intrusion or clipping.
- **Gemma 4 12B:** `error` — None
- **Gemma 4 E4B:** `pass` — The audio contains a single speaker with no background noise or secondary voices. Both the start and end of the speech are acoustically complete, with no clipped boundaries.

### Turn `turn_007_spk_00_23.08-28.06` (4.97s)
- **Audio Cut:** [Listen](data/experiment_khanhvy/cuts/turn_007_spk_00_23.08-28.06.wav)
- **Gemini 3.8 Flash:** `reject` — An initial secondary speaker is audible before the main speaker, and the final word is clipped abruptly at the end.
- **Gemma 4 12B:** `error` — None
- **Gemma 4 E4B:** `reject` — The audio ends abruptly mid-word ('Cách'), failing to complete the final consonant closure and tonal contour into natural silence.

### Turn `turn_008_spk_00_28.16-36.02` (7.86s)
- **Audio Cut:** [Listen](data/experiment_khanhvy/cuts/turn_008_spk_00_28.16-36.02.wav)
- **Gemini 3.8 Flash:** `reject` — The initial syllable is abruptly truncated at the start of the recording, cutting off its initial consonant onset.
- **Gemma 4 12B:** `pass` — The audio contains a single speaker with clear, complete phonetic boundaries at both the start and the end, with no tail intrusions.
- **Gemma 4 E4B:** `pass` — The audio contains a single, clear speaker with no background noise or secondary voices. Both the start and end of the speech are acoustically complete, with full phonetic onsets and natural silence following the final word.

### Turn `turn_009_spk_00_36.28-40.70` (4.42s)
- **Audio Cut:** [Listen](data/experiment_khanhvy/cuts/turn_009_spk_00_36.28-40.70.wav)
- **Gemini 3.8 Flash:** `reject` — The audio cuts off abruptly at the very end while the final syllable's tone and vowel are still actively sounding, cutting off the natural decay.
- **Gemma 4 12B:** `pass` — Single speaker with clear phonetic onset and coda closure at both boundaries.
- **Gemma 4 E4B:** `pass` — Single speaker with no background noise. Speech starts and ends on complete word boundaries.

### Turn `turn_010_spk_00_41.19-42.76` (1.57s)
- **Audio Cut:** [Listen](data/experiment_khanhvy/cuts/turn_010_spk_00_41.19-42.76.wav)
- **Gemini 3.8 Flash:** `reject` — The final syllable is abruptly cut off mid-vocalization at the end of the clip.
- **Gemma 4 12B:** `pass` — Single speaker with clear, complete word boundaries at both start and end.
- **Gemma 4 E4B:** `pass` — The audio contains a single speaker with no overlap or tail intrusion. Both the start and end of the speech are acoustically complete.

### Turn `turn_011_spk_02_43.75-46.72` (2.97s)
- **Audio Cut:** [Listen](data/experiment_khanhvy/cuts/turn_011_spk_02_43.75-46.72.wav)
- **Gemini 3.8 Flash:** `reject` — A second speaker speaks clearly in the latter half of the clip.
- **Gemma 4 12B:** `pass` — Single speaker with clear, complete word boundaries at both start and end.
- **Gemma 4 E4B:** `pass` — The audio contains a single speaker with no background noise or secondary voices. Both the start and end of the speech are complete and natural.

### Turn `turn_013_spk_00_52.71-53.94` (1.23s)
- **Audio Cut:** [Listen](data/experiment_khanhvy/cuts/turn_013_spk_00_52.71-53.94.wav)
- **Gemini 3.8 Flash:** `reject` — The final syllable is abruptly cut off before its coda closure and vowel decay are complete.
- **Gemma 4 12B:** `pass` — Single speaker with clear phonetic onset and complete coda closure at the end.
- **Gemma 4 E4B:** `pass` — The audio contains a single speaker with no background noise or secondary voices. Both the start and end of the speech are acoustically complete, adhering to natural word boundaries.

### Turn `turn_015_spk_02_80.51-84.50` (3.99s)
- **Audio Cut:** [Listen](data/experiment_khanhvy/cuts/turn_015_spk_02_80.51-84.50.wav)
- **Gemini 3.8 Flash:** `reject` — The final syllable is cut off abruptly before its full tone and acoustic energy decay naturally.
- **Gemma 4 12B:** `error` — None
- **Gemma 4 E4B:** `pass` — The audio contains a single speaker with no overlap or tail intrusion. Both the start and end of the speech segment are on complete, natural word boundaries.

### Turn `turn_016_spk_02_87.24-88.96` (1.72s)
- **Audio Cut:** [Listen](data/experiment_khanhvy/cuts/turn_016_spk_02_87.24-88.96.wav)
- **Gemini 3.8 Flash:** `reject` — The final syllable is abruptly cut off mid-vocalization before reaching natural acoustic decay.
- **Gemma 4 12B:** `pass` — Single speaker with complete phonetic boundaries at both start and end.
- **Gemma 4 E4B:** `pass` — The audio contains a single speaker with no overlap or trailing vocal intrusions. Both the start and end of the speech occur on complete, natural word boundaries.

### Turn `turn_017_spk_00_99.36-101.73` (2.38s)
- **Audio Cut:** [Listen](data/experiment_khanhvy/cuts/turn_017_spk_00_99.36-101.73.wav)
- **Gemini 3.8 Flash:** `reject` — The initial word is cut off abruptly mid-syllable at the onset, and multiple background voices/laughter are audible throughout the clip.
- **Gemma 4 12B:** `pass` — Single speaker with clear phonetic boundaries at both start and end.
- **Gemma 4 E4B:** `pass` — The audio contains a single speaker with no overlaps or secondary voices. The speech starts with a full phonetic onset and ends with a complete word closure into natural silence.

### Turn `turn_018_spk_01_104.28-106.01` (1.73s)
- **Audio Cut:** [Listen](data/experiment_khanhvy/cuts/turn_018_spk_01_104.28-106.01.wav)
- **Gemini 3.8 Flash:** `reject` — There is a prominent secondary speaker or background voice audible at the start of the audio clip.
- **Gemma 4 12B:** `pass` — Single speaker with complete word boundaries at both start and end.
- **Gemma 4 E4B:** `reject` — The audio is pure, but the final word is clipped at the end of the segment.

### Turn `turn_019_spk_02_107.94-109.57` (1.63s)
- **Audio Cut:** [Listen](data/experiment_khanhvy/cuts/turn_019_spk_02_107.94-109.57.wav)
- **Gemini 3.8 Flash:** `reject` — There are two distinct speakers taking turns in this short recording.
- **Gemma 4 12B:** `pass` — Single speaker throughout with no tail intrusions; speech starts and ends on complete word boundaries.
- **Gemma 4 E4B:** `pass` — Single speaker with no overlap or tail intrusion. Speech starts and ends on complete word boundaries.

### Turn `turn_020_spk_01_108.79-111.29` (2.51s)
- **Audio Cut:** [Listen](data/experiment_khanhvy/cuts/turn_020_spk_01_108.79-111.29.wav)
- **Gemini 3.8 Flash:** `reject` — There is overlapping speech between multiple speakers at the beginning, and the speech cuts off abruptly mid-vowel at the end.
- **Gemma 4 12B:** `pass` — Single speaker with complete word boundaries at start and end.
- **Gemma 4 E4B:** `pass` — The audio contains a single speaker with no overlaps or secondary voices. The speech starts and ends on complete, natural acoustic word boundaries.

### Turn `turn_021_spk_01_113.12-114.25` (1.12s)
- **Audio Cut:** [Listen](data/experiment_khanhvy/cuts/turn_021_spk_01_113.12-114.25.wav)
- **Gemini 3.8 Flash:** `pass` — The speech starts and ends cleanly with clear word boundaries, natural decay, and no secondary speaker or overlapping speech.
- **Gemma 4 12B:** `error` — None
- **Gemma 4 E4B:** `pass` — Single speaker is present, and both the start and end of the speech are acoustically complete.

### Turn `turn_022_spk_00_115.32-118.62` (3.31s)
- **Audio Cut:** [Listen](data/experiment_khanhvy/cuts/turn_022_spk_00_115.32-118.62.wav)
- **Gemini 3.8 Flash:** `reject` — Heavy background chatter and overlapping secondary voices are present throughout the recording.
- **Gemma 4 12B:** `error` — None
- **Gemma 4 E4B:** `pass` — The audio contains a single speaker with no secondary voices or tail intrusion. Both the start and end of the speech are acoustically complete.

### Turn `turn_023_spk_00_119.25-121.27` (2.01s)
- **Audio Cut:** [Listen](data/experiment_khanhvy/cuts/turn_023_spk_00_119.25-121.27.wav)
- **Gemini 3.8 Flash:** `pass` — The utterance is produced by a single speaker with cleanly preserved acoustic boundaries at both onset and offset.
- **Gemma 4 12B:** `error` — None
- **Gemma 4 E4B:** `pass` — The audio contains a single speaker, and both the start and end of the speech are acoustically complete and natural.

### Turn `turn_025_spk_01_125.51-126.74` (1.23s)
- **Audio Cut:** [Listen](data/experiment_khanhvy/cuts/turn_025_spk_01_125.51-126.74.wav)
- **Gemini 3.8 Flash:** `reject` — Multiple speakers are audible with overlapping voices in the background, and the audio abruptly cuts off mid-syllable at the end.
- **Gemma 4 12B:** `pass` — Single speaker with clear word boundaries at both start and end.
- **Gemma 4 E4B:** `pass` — The audio contains a single speaker with no overlap or tail intrusion. Both the start and end of the speech are on complete, natural word boundaries.

### Turn `turn_027_spk_00_154.39-157.66` (3.27s)
- **Audio Cut:** [Listen](data/experiment_khanhvy/cuts/turn_027_spk_00_154.39-157.66.wav)
- **Gemini 3.8 Flash:** `reject` — The final syllable is cut off abruptly before the vowel/coda completes naturally.
- **Gemma 4 12B:** `pass` — Single speaker with clear phonetic onset and full coda closure at both boundaries.
- **Gemma 4 E4B:** `pass` — The audio features a single speaker with no overlap or tail intrusion. Both the start and end of the speech occur on complete, natural word boundaries.

### Turn `turn_028_spk_01_158.46-161.50` (3.05s)
- **Audio Cut:** [Listen](data/experiment_khanhvy/cuts/turn_028_spk_01_158.46-161.50.wav)
- **Gemini 3.8 Flash:** `reject` — The final syllable is cut off abruptly while vocal fold vibration is still actively sustained.
- **Gemma 4 12B:** `pass` — Single speaker with clear word boundaries at both start and end.
- **Gemma 4 E4B:** `reject` — The audio cuts off abruptly while the final word is still in flight, failing to complete its tonal contour and coda closure.

### Turn `turn_031_spk_02_178.18-180.00` (1.82s)
- **Audio Cut:** [Listen](data/experiment_khanhvy/cuts/turn_031_spk_02_178.18-180.00.wav)
- **Gemini 3.8 Flash:** `reject` — The final word is cut off abruptly while vocalization and tone are still actively ongoing.
- **Gemma 4 12B:** `pass` — Single speaker with clear, complete word boundaries at both start and end.
- **Gemma 4 E4B:** `pass` — The audio contains a single speaker, and both the start and end of the speech segment are acoustically complete and natural.
