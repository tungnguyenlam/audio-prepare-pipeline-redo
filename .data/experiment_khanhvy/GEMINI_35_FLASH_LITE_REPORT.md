# Benchmark: Gemini 3.5 Flash-Lite on 31 Khanh Vy Benchmark Turns

- **Evaluator:** `gemini-3.5-flash-lite`
- **Total Benchmark Turns:** 31
- **Pass Count:** 23 (74.2%)
- **Reject Count:** 8 (25.8%)
- **Agreement with Gemini 3.8 Flash:** 16/31 (51.6%)
- **Agreement with Gemma 4 E4B:** 21/31 (67.7%)
- **Average Latency:** 1.79s per clip

## Comparison Table

| Index | Turn ID | Dur (s) | 3.5 Flash-Lite | 3.8 Flash | Gemma 4 E4B | 3.5 Latency | Purity | Completeness | Boundary Issue | Reason |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | `turn_001_spk_00_0.10-1.85` | 1.75s | **pass** | `reject` | `reject` | 1.81s | `pure` | `complete` | `none` | The audio contains a single speaker with clear, complete word boundaries at both the start and the end, and no tail speaker intrusion. |
| 2 | `turn_002_spk_00_2.19-5.28` | 3.08s | **pass** | `pass` | `pass` | 1.74s | `pure` | `complete` | `none` | The audio contains a single speaker with clean acoustic boundaries at both the start and the end, with no tail intrusion or overlapping speech. |
| 3 | `turn_003_spk_01_6.85-7.76` | 0.92s | **pass** | `pass` | `pass` | 1.36s | `pure` | `complete` | `none` | The audio contains a single speaker with no background voices or tail intrusions, and the speech starts and ends cleanly with complete acoustic boundaries. |
| 4 | `turn_004_spk_00_8.56-10.44` | 1.87s | **pass** | `pass` | `pass` | 2.07s | `pure` | `complete` | `none` | The audio contains a single speaker with no overlapping speech or tail intrusions. Both the start and end of the utterance are acoustically complete with natural boundaries. |
| 5 | `turn_005_spk_00_10.83-19.84` | 9.01s | **pass** | `reject` | `pass` | 2.25s | `pure` | `complete` | `none` | The audio contains a single speaker throughout the clip with no secondary speaker intrusion at the tail. The speech starts cleanly and ends with complete natural acoustic word boundaries. |
| 6 | `turn_006_spk_02_20.87-21.92` | 1.05s | **pass** | `pass` | `pass` | 1.89s | `pure` | `complete` | `none` | The audio contains a single speaker with no background voices or tail intrusion. The speech starts and ends cleanly on natural acoustic word boundaries with complete phonetic onset and coda. |
| 7 | `turn_007_spk_00_23.08-28.06` | 4.97s | **pass** | `reject` | `reject` | 1.8s | `pure` | `complete` | `none` | The audio contains a single speaker with clear acoustic boundaries at both the start and the end, with no intrusions or overlapping speech. |
| 8 | `turn_008_spk_00_28.16-36.02` | 7.86s | **pass** | `reject` | `pass` | 1.9s | `pure` | `complete` | `none` | The audio contains a single speaker with no background voices or tail intrusion. The speech begins and ends cleanly on natural acoustic word boundaries without any clipping. |
| 9 | `turn_009_spk_00_36.28-40.70` | 4.42s | **pass** | `reject` | `pass` | 1.89s | `pure` | `complete` | `none` | The audio contains a single speaker with no background voices or tail intrusions, and the speech starts and ends cleanly with complete natural word boundaries. |
| 10 | `turn_010_spk_00_41.19-42.76` | 1.57s | **pass** | `reject` | `pass` | 1.89s | `pure` | `complete` | `none` | The audio contains a single speaker with no background or tail intrusions, and both the start and end words are acoustically complete with natural onsets and decays. |
| 11 | `turn_011_spk_02_43.75-46.72` | 2.97s | **reject** | `reject` | `pass` | 1.53s | `impure` | `complete` | `none` | A secondary speaker is audible in the background alongside the main speaker. |
| 12 | `turn_012_spk_00_48.36-51.29` | 2.93s | **pass** | `pass` | `pass` | 1.62s | `pure` | `complete` | `none` | The audio contains a single speaker with clear, natural acoustic boundaries at both the start and end, with no background voices or tail intrusion. |
| 13 | `turn_013_spk_00_52.71-53.94` | 1.23s | **pass** | `reject` | `pass` | 1.46s | `pure` | `complete` | `none` | The audio contains a single speaker with clean acoustic boundaries at both the start and the end, with no tail intrusions. |
| 14 | `turn_014_spk_00_59.87-74.94` | 15.07s | **pass** | `pass` | `pass` | 2.25s | `pure` | `complete` | `none` | The audio contains a single speaker throughout with clean acoustic boundaries at both the start and the end. |
| 15 | `turn_015_spk_02_80.51-84.50` | 3.99s | **pass** | `reject` | `pass` | 1.88s | `pure` | `complete` | `none` | The audio contains a single speaker with clear acoustic boundaries at both the start and the end, with no overlapping speech or tail intrusions. |
| 16 | `turn_016_spk_02_87.24-88.96` | 1.72s | **pass** | `reject` | `pass` | 1.63s | `pure` | `complete` | `none` | The audio contains a single speaker with no background voices or tail intrusion. Both the start and end words are acoustically complete with natural onsets and codas. |
| 17 | `turn_017_spk_00_99.36-101.73` | 2.38s | **reject** | `reject` | `pass` | 1.72s | `impure` | `complete` | `none` | There are multiple speakers laughing and talking in the background overlapping with the main speaker. |
| 18 | `turn_018_spk_01_104.28-106.01` | 1.73s | **reject** | `reject` | `reject` | 1.64s | `impure` | `complete` | `none` | A secondary speaker utters a trailing vocalization at the very end of the clip, causing tail intrusion. |
| 19 | `turn_019_spk_02_107.94-109.57` | 1.63s | **reject** | `reject` | `pass` | 1.61s | `impure` | `complete` | `none` | A secondary speaker's voice is audible overlapping or immediately following the primary speaker in the audio. |
| 20 | `turn_020_spk_01_108.79-111.29` | 2.51s | **reject** | `reject` | `pass` | 1.9s | `impure` | `incomplete` | `clipped_end` | The main speaker is interrupted and followed by a secondary speaker at the tail end, and the final word is abruptly cut off before natural decay. |
| 21 | `turn_021_spk_01_113.12-114.25` | 1.12s | **reject** | `pass` | `pass` | 1.63s | `impure` | `complete` | `none` | There are multiple speakers talking simultaneously in the audio clip. |
| 22 | `turn_022_spk_00_115.32-118.62` | 3.31s | **reject** | `reject` | `pass` | 1.62s | `impure` | `complete` | `none` | There are multiple speakers talking simultaneously in the background throughout the audio clip. |
| 23 | `turn_023_spk_00_119.25-121.27` | 2.01s | **pass** | `pass` | `pass` | 3.1s | `pure` | `complete` | `none` | The audio contains a single speaker with clear acoustic boundaries at both the start and the end, with no overlapping speech, background voices, or clipped words. |
| 24 | `turn_024_spk_01_122.05-123.22` | 1.17s | **pass** | `pass` | `pass` | 1.46s | `pure` | `complete` | `none` | The audio contains a single speaker with clear acoustic boundaries at both the start and the end, with no overlapping voices or tail intrusions. |
| 25 | `turn_025_spk_01_125.51-126.74` | 1.23s | **pass** | `reject` | `pass` | 1.46s | `pure` | `complete` | `none` | The audio contains a single speaker with no background voices or tail intrusion, and the speech starts and ends cleanly on complete acoustic word boundaries. |
| 26 | `turn_026_spk_00_130.36-139.84` | 9.48s | **pass** | `pass` | `pass` | 2.0s | `pure` | `complete` | `none` | The audio contains a single speaker with clean boundaries at the start and end, and no tail intrusion or overlapping speech. |
| 27 | `turn_027_spk_00_154.39-157.66` | 3.27s | **pass** | `reject` | `pass` | 1.83s | `pure` | `complete` | `none` | The speech is delivered by a single speaker with no background voices or tail intrusion. Both the start and the end of the utterance have complete acoustic boundaries and natural consonant/tonal transitions into silence. |
| 28 | `turn_028_spk_01_158.46-161.50` | 3.05s | **pass** | `reject` | `reject` | 1.65s | `pure` | `complete` | `none` | The audio contains a single speaker with clean boundaries at both the start and end, and no trailing speaker intrusion. |
| 29 | `turn_029_spk_01_163.21-164.51` | 1.3s | **pass** | `pass` | `pass` | 1.65s | `pure` | `complete` | `none` | The audio contains a single speaker with no background voices or tail intrusion. Both the start and end words are fully articulated with natural acoustic boundaries and complete consonant/tonal closure. |
| 30 | `turn_030_spk_00_175.90-177.58` | 1.68s | **reject** | `pass` | `pass` | 1.56s | `impure` | `complete` | `none` | A secondary speaker is heard uttering a trailing voice at the tail end of the audio clip. |
| 31 | `turn_031_spk_02_178.18-180.00` | 1.82s | **pass** | `reject` | `pass` | 1.61s | `pure` | `complete` | `none` | The audio contains a single speaker throughout with no overlap or tail intrusion. Both the start and end words are acoustically complete with natural onsets and decays. |
