# Speech Verifier Benchmark Report

- **Evaluated Model:** `gemini-3.5-flash-lite` (gemini)
- **LoRA Adapter:** `None (Base Model)`
- **Reasoning effort:** `medium`
- **Evaluation Dataset:** `.data/tts_strategy/gold_benchmark_20260908/eval_input.jsonl`
- **Total Evaluated:** 288
- **Successful:** 288 (100.0%)
- **Average Latency:** 2.16s

## Verdict Distribution

| Verdict | Count | Percentage |
|---|---|---|
| **Pass** | 225 | 78.1% |
| **Reject** | 63 | 21.9% |
| **Total** | 288 | 100.0% |


## API Usage & Estimated Cost

- **Requests priced:** 288 (unpriced: 0)
- **Tokens:** prompt=210905, audio_in=39545, output=15240, thinking=39831, total=265976
- **Estimated USD (paid Standard, as of 2026-09-04):** input=$0.063271, output+thinking=$0.137678, **total=$0.200949**


## Benchmark vs Gemini 3.8 Flash Teacher

- **Total Ground-Truth Reference Items:** 288
- **Overall Agreement Rate:** **71.5%** (206/288)
- **Precision (TTS Purity Confidence):** 67.1%
- **Recall (Passing Yield Retention):** 95.0%
- **F1 Score:** 78.6%

### Confusion Matrix

| Reference \ Model | Model Pass | Model Reject | Total Reference |
|---|---|---|---|
| **Gemini Pass** | 151 (True Pass) | 8 (False Reject) | 159 |
| **Gemini Reject** | 74 (Contamination Leak) | 55 (True Reject) | 129 |
| **Total Model** | 225 | 63 | 288 |


## Disagreements with Gemini 3.8 Flash (82 samples)

| Sample ID | Gemini 3.8 Flash | Candidate Model | Diagnostic | Gemini Reason | Candidate Model Reason |
|---|---|---|---|---|---|
| `QBml8L3wS3Q_dddbc34270a0` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Noticeable hollow room reverberation and echo during 'quý vị thấy một cái không gian hoàn toàn mới nha' following an abrupt acoustic shift. | The background is clean, there is only one speaker, and both file edges are intact without clipping. |
| `i0zYcXBjytE_523d9406419e` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible background music/instrumental track playing underneath the speech throughout the clip. | The background is clean, there is only one speaker, and both file edges are intact without truncation. All checks passed. |
| `lfIbjICmfW0_1f2563edaa07` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Background acoustic guitar music plays underneath the speech throughout the entire clip. | The background is clean, there is only one speaker with no overlap or secondary voice, and both the start and end of the file are free of clipping. |
| `lfIbjICmfW0_7dd16cb044a7` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible background music track playing continuously throughout the clip.; The final word/syllable ('vi...') is abruptly cut off before completion at the end of the clip. | The background, speakers, and both file edges were checked. The speech is clean with no background music, effects, noise, or secondary speakers, and the edges are intact without clipping. |
| `lfIbjICmfW0_ff70a78d5f7c` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible background music plays underneath the speech throughout the entire audio.; The final word 'một' is cut off abruptly at the end of the clip. | The audio has clean foreground speech without background music, sound effects, or secondary speakers. Both the start and end edges are intact, and no acoustic defects were found. |
| `j83rzAzRDAI_59b8f959fdaf` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible cinematic background music plays throughout the entire recording.; The final word 'cũng' is cut off abruptly before completion. | The speech is clean with no background music, sound effects, or noise. There are no secondary speakers, and both file edges start and end cleanly. |
| `lfIbjICmfW0_a6c4d7397ce7` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible background music plays continuously throughout the entire clip. | The speech is clean with no background music, effects, or secondary speakers. Both the start and end edges are intact without clipping. |
| `PXEtB-CsvSw_04e7792dd749` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible background music/ambient soundtrack playing beneath the speech throughout the entire clip.; Noticeable hall reverberation and PA system room echo on the speech. | The clip has clean foreground speech without background music, sound effects, or excessive noise. No secondary speaker is present, and both file edges are intact with no clipping. |
| `lfIbjICmfW0_3a265423d415` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible background music plays continuously beneath the vocal track throughout the clip.; Speech is cut off abruptly mid-utterance at the end of the audio. | The background, single speaker, and both file edges were checked and found to be completely clean without any defects. |
| `fwN5VT_QxkY_6d101f6ecdaa` | **REJECT** | **PASS** | False Pass (Contamination Leak) | The initial consonant of the first word 'trời' is clipped off at the very beginning of the audio.; Audible hollow room reverberation and echo on the speech throughout the entire clip. | The speech starts cleanly at the beginning, ends naturally, and the background is free of music, noise, or secondary speakers. |
| `QBml8L3wS3Q_3a06b1cd3ae5` | **PASS** | **REJECT** | False Reject (Overly Strict) | pass | There is audible background music accompanying the speech throughout the clip. |
| `QBml8L3wS3Q_04b2ee6d579b` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Acoustic guitar background music plays continuously throughout the speech. | The clip has clean foreground speech without background music, noise, or secondary speakers. Both file edges are complete and free from clipping. |
| `j83rzAzRDAI_f017a0ae4cd7` | **REJECT** | **PASS** | False Pass (Contamination Leak) | A second female speaker enters saying 'mày không gọi, để chị gọi'.; The second speaker overlaps with the primary speaker starting 'Đừng có làm'. | The background is clean, there is only one speaker performing the dialogue, and both file edges are clean without clipping. |
| `lfIbjICmfW0_fa646b94bd27` | **REJECT** | **PASS** | False Pass (Contamination Leak) | The initial consonant of the first word 'khẩn' is cut abruptly at the start of the audio file. | The background is clean, there is only one speaker with no secondary voices or interruptions, and both file edges are clean and complete. |
| `i0zYcXBjytE_96905135c83e` | **REJECT** | **PASS** | False Pass (Contamination Leak) | A secondary voice speaks in the background during 'gai góc'.; Secondary speaker speech overlaps directly with the primary speaker's words.; The final word 'cá' is abruptly cut off before its natural completion. | The background is clean, there is only a single speaker throughout the clip without any interference, and both the beginning and the end of the file are acoustically complete. |
| `i0zYcXBjytE_8652b02f4f76` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible ambient background music / synth pad playing continuously beneath the speech. | The background, speakers, and both file edges were checked and found to be clean of any defects. |
| `j83rzAzRDAI_75a698aa3318` | **REJECT** | **PASS** | False Pass (Contamination Leak) | A female voice speaks in the first half of the clip followed by a different male speaker at around 2.5 seconds. | The audio is clean with no background music, sound effects, or secondary speakers. Both file edges are complete and free of clipping. |
| `QBml8L3wS3Q_17b27eb3a783` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Persistent loud ambient noise resembling rushing water from a waterfall is audible throughout the entire recording. | The background is clean, there is only one speaker with no overlaps, and both file edges were checked and found to be intact without truncation. |
| `fwN5VT_QxkY_1ee7c2c834b8` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Background instrumental music plays throughout the entire audio clip. | The background, speakers, and both file edges were checked. The speech is clean, uninterrupted, has no secondary speakers or background noise, and starts and ends cleanly. |
| `j83rzAzRDAI_da5d334353b0` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible double clapping/slapping sound effect.; A secondary male speaker enters saying 'Nên là...' after the primary female speaker.; The trailing speech is cut off abruptly at the end of the audio. | The background, speakers, and both file edges were checked. No background music, noise, or secondary speakers were found, and the audio is clean with natural onsets and offsets. |
| `Oa-mVxGS4cw_495699047b62` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Continuous instrumental background music plays underneath the speech throughout the entire clip. | The background is clean, there is only one primary speaker with no secondary voices or interruptions, and both file edges are clean and complete. All checks passed. |
| `j83rzAzRDAI_c6c05afb5834` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible low-pitched chuckle/snicker intrusion from an off-mic secondary person right after the final word. | The background is clean, there is only one speaker, and both file edges are intact. Checked the background, speakers, and both file edges. |
| `QBml8L3wS3Q_2585cbcbf9c3` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible room reverberation and hollow acoustic reflections throughout the clip from recording in an untreated space. | The background is clean with no music or distracting noise, there is only a single speaker throughout the clip, and both the start and end of the file are acoustically complete with no clipping. |
| `lfIbjICmfW0_deb876c311ed` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Background instrumental/melodic music is audible beneath the speech throughout the entire clip. | The background is clean without music or artifacts, there is only one speaker with no interruptions, and both the start and end of the file are acoustically complete. |
| `fwN5VT_QxkY_432970843d6b` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible background music plays continuously throughout the entire clip. | The background is clean, there is only one speaker with no secondary voices, and both file edges are clean and unclipped. |
| `QBml8L3wS3Q_defe3cfd2bf5` | **PASS** | **REJECT** | False Reject (Overly Strict) | pass | The clip contains audible instrumental music playing in the background. |
| `lfIbjICmfW0_c70127e5f0c2` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Continuous background music plays underneath the speech throughout the entire audio. | The clip contains clean foreground speech without background music, sound effects, or secondary speakers. Both file edges are clean and intact. |
| `QBml8L3wS3Q_626bca788f54` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible background instrumental music plays throughout the entire recording. | The background, speakers, and both file edges were checked and found to be clean of any defects. |
| `i0zYcXBjytE_3779be889c1f` | **REJECT** | **PASS** | False Pass (Contamination Leak) | The initial consonant of the following word is cut off abruptly at the end of the audio. | The background, speakers, and both file edges were checked and found to be free of defects. |
| `fwN5VT_QxkY_d8e1416c96f8` | **REJECT** | **PASS** | False Pass (Contamination Leak) | A different speaker is heard uttering a truncated phrase before the primary male speaker begins.; The initial speech at the very start of the clip is cut off mid-syllable. | The background is clean, there is only a single speaker, and both file edges are intact. |
| `PXEtB-CsvSw_5562a479358c` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible instrumental background music plays underneath the speech throughout the entire clip. | The background is clean, there is only one speaker without any interruptions or background voices, and both the start and end of the file are acoustically complete. |
| `PXEtB-CsvSw_ad7ba1b905b0` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Continuous dramatic background music plays throughout the entire audio clip. | The clip has clean foreground speech without background music or sound effects, no secondary speakers, and both file edges are intact. |
| `Oa-mVxGS4cw_629e20b1f079` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Continuous background music is present throughout the entire recording. | The background, speaker, and both file edges were checked; no music, secondary speakers, noise, or clipped words were found. |
| `3Nll-JLzvvE_cadda063f046` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible room reverberation and hollow acoustic reflections throughout the recording. | The clip has clean foreground speech with no background music, sound effects, excessive noise, or secondary speakers. Both the start and end of the audio file are clean without any clipping of the speech. |
| `Oa-mVxGS4cw_08e43a00e843` | **REJECT** | **PASS** | False Pass (Contamination Leak) | A continuous, prominent typewriter or keyboard clicking sound effect plays throughout the clip. | The background is clean, there is only one speaker with no interruptions, and both file edges are clean and complete. All checks passed. |
| `3Nll-JLzvvE_fa3154617901` | **PASS** | **REJECT** | False Reject (Overly Strict) | pass | There is a distinct audible music track (instrumental beat/melody) playing in the background throughout the entire clip. |
| `3Nll-JLzvvE_484f50812020` | **REJECT** | **PASS** | False Pass (Contamination Leak) | The audio cuts in abruptly at the start during an ongoing vowel sound, cutting off the onset of the initial word. | The background is clean, there is only one primary speaker with no secondary voices, and both file edges are clean without any clipping or truncation. |
| `3Nll-JLzvvE_b7108ac193ce` | **PASS** | **REJECT** | False Reject (Overly Strict) | pass | Audible music (piano notes) is present in the background. |
| `NI8JVXNWlN8_a922963f0836` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible background acoustic guitar music plays continuously throughout the recording. | The background, speakers, and both file edges were checked and found to be free of defects. |
| `NI8JVXNWlN8_41516b38ef10` | **REJECT** | **PASS** | False Pass (Contamination Leak) | The final word 'bạn' is cut off abruptly mid-syllable as the recording ends. | The background is clean with only faint room tone, there is only a single speaker with no secondary voices or interruptions, and both file edges capture complete speech onsets and releases. |
| `zK3qFnKZFRo_3eabdb8239e5` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible background music plays continuously throughout the entire clip.; The final word is cut off mid-syllable as the recording ends. | The audio is clean foreground speech with no background music, sound effects, or secondary speakers. Both the start and end of the file are intact without any clipping. |
| `zK3qFnKZFRo_cd631106dfb9` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible background music plays continuously throughout the entire clip.; A second female speaker answers the initial question starting at around 2.8s ('Chúng mình được học...'). | The foreground speech is clear and clean with no background music, sound effects, excessive noise, or reverberation. There are no secondary speakers or overlapping speech, and both the start and end of the file are intact without clipping. |
| `zK3qFnKZFRo_d7723da54617` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible background chatter from secondary speakers in the room throughout the recording. | The background is clean with no music, sound effects, or excessive noise. There is only a single speaker throughout the clip. Both file edges are clean, with the speech starting properly on the first word and the final word concluding acoustically before the file ends. |
| `zK3qFnKZFRo_682aac453db4` | **REJECT** | **PASS** | False Pass (Contamination Leak) | An audio fragment of an earlier word is clipped right as the clip begins.; Audible background music plays underneath the speech across the entire clip.; A distinct second speaker begins speaking after the first speaker finishes.; The speech cuts off mid-word at the very end of the clip. | The speech is clean and clear with no background music, secondary speakers, excessive noise, or clipped edges. |
| `zK3qFnKZFRo_3d09def2ae6f` | **PASS** | **REJECT** | False Reject (Overly Strict) | pass | There is audible background music playing throughout the clip. |
| `PXEtB-CsvSw_4b7db2a5bf27` | **REJECT** | **PASS** | False Pass (Contamination Leak) | The final word 'có' is cut off abruptly mid-phonation before finishing naturally.; Audible room reflections and hollow reverberation present throughout the recording. | The audio is clean with no background music, effects, or noise, there is only one speaker, and both file edges are clean. |
| `3Nll-JLzvvE_269a2efb80f4` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible background music and rhythmic beat playing continuously throughout the entire utterance. | The background, speakers, and both file edges were checked and found to be clean of any defects. |
| `3Nll-JLzvvE_825c989c1c0a` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Background music with melodic piano/keyboard chords is audible throughout the entire recording. | The audio is clean with no background music, sound effects, excessive noise, or reverberation. Only one speaker is present throughout. Both file edges are clean with no clipped words at the start or end. |
| `3Nll-JLzvvE_307d277028a7` | **REJECT** | **PASS** | False Pass (Contamination Leak) | A digital swoosh / transition sound effect plays underneath the speech during 'là battle'. | The background, speakers, and both file edges were checked; the speech is clean, uninterrupted, single-speaker, and free of background noise or music. |
| `PXEtB-CsvSw_edfa1f05041a` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Continuous instrumental background music is audible throughout the entire speech recording. | The background is clean, there is only a single speaker, and both file edges are intact without truncation. The clip has been thoroughly checked and meets all criteria. |
| `3Nll-JLzvvE_5e2eb7696a66` | **REJECT** | **PASS** | False Pass (Contamination Leak) | The initial syllable/filler at the start of the audio is abruptly cut into without a natural onset. | The clip has clean foreground speech with no background music, sound effects, or excessive noise. No secondary speakers are present, and both file edges are clean without any clipping of the start or end syllables. |
| `PXEtB-CsvSw_07ad42b90f66` | **REJECT** | **PASS** | False Pass (Contamination Leak) | The final word 'Airbus' is cut off abruptly before the final syllable/consonant is completed.; Faint background music/synth pad is audible behind the voice. | The audio is clean with no background music, sound effects, or excessive noise. Only one speaker is present, and both the start and end edges are clean and uncut. |
| `PXEtB-CsvSw_4a7222b7e6cf` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Faint secondary chuckle/laughter audible in the background between 'cái' and 'ô cửa'. | The background is clean, no secondary speakers are present, and both file edges are intact. |
| `3Nll-JLzvvE_ab7847350244` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Noticeable hollow room reverberation and reflections coloring the speaker's voice throughout the clip. | The clip has clean foreground speech with no background music, sound effects, excessive noise, or reverberation. There is only one speaker and no overlapping speech. Both file edges are clean with no clipped words. |
| `3Nll-JLzvvE_60d59ffec8bb` | **REJECT** | **PASS** | False Pass (Contamination Leak) | The initial word at the very beginning of the audio is truncated mid-syllable, cutting off the initial consonant onset. | The background is clean, there is only a single speaker, and both file edges are clean and unclipped. |
| `3Nll-JLzvvE_ccee8b3cad81` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Background instrumental beat/music is audible throughout the clip. | The background, speakers, and both file edges were checked and found to be clean of any defects. |
| `3Nll-JLzvvE_63c370036991` | **PASS** | **REJECT** | False Reject (Overly Strict) | pass | An audible instrumental beat/music is present in the background throughout the clip. |
| `PXEtB-CsvSw_804b39bc7047` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible background music plays continuously beneath the spoken audio throughout the clip. | The background, speakers, and both file edges were checked and found free of defects. |
| `PXEtB-CsvSw_5eb0ecf1cad9` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible secondary chuckle/laughter intrusion between 'cái' and 'ô cửa'. | The background, speakers, and both file edges were checked and found to be clean and free of defects. |
| `3Nll-JLzvvE_82b7f679dd4e` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible acoustic guitar background music plays beneath the speech throughout the clip. | The background is clean, there is only one speaker, and both file edges are clean and intact after careful listening. |
| `3Nll-JLzvvE_32ba902adc90` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible background music/synthesizer track playing continuously throughout the recording. | The audio clip has clean foreground speech without background noise, music, or effects. There are no secondary speakers, and both the start and end of the file are intact without clipping. |
| `3Nll-JLzvvE_521ae4587a8c` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible background music beat playing throughout the speech. | The background, single speaker, and both file edges were checked and found to be completely clean without any defects. |
| `3Nll-JLzvvE_8202666b8c80` | **PASS** | **REJECT** | False Reject (Overly Strict) | pass | Audible background music/beats are present throughout the clip. |
| `PXEtB-CsvSw_0407475b8f1c` | **REJECT** | **PASS** | False Pass (Contamination Leak) | The final word cuts off abruptly mid-syllable ('được ph...'). | The background is clean with no music, sound effects, excessive noise, or reverberation. Only one speaker is audible with no secondary voices or overlaps. Both file edges are clean with no clipped words at the start or end. |
| `H0VpjeULCck_147db6aa6fbb` | **REJECT** | **PASS** | False Pass (Contamination Leak) | A secondary male speaker takes over and says 'hai hai rồi hai hai rồi' after the initial female speaker finishes her phrase. | The background is clean, there is only a single speaker, and both file edges are intact without truncation. |
| `H0VpjeULCck_c6b6c8d9be6d` | **REJECT** | **PASS** | False Pass (Contamination Leak) | A truncated syllable is cut off mid-word at the very beginning of the clip right before 'cho người tàn tật'. | The background is clean, there is only one speaker without any overlapping voices, and both file edges are complete without truncation. The clip was thoroughly checked and meets all criteria for a pass. |
| `H0VpjeULCck_6483a2a02485` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Pronounced auditorium/PA system reverberation and room echo audible across the entire recording.; Audience applause and clapping clearly audible in the background under the speech. | The background is clean with only faint natural room acoustic, there is no secondary speaker, and both the start and end file edges are complete and cleanly articulated. Checked background, speakers, and file edges. |
| `H0VpjeULCck_1175ba7b50f4` | **REJECT** | **PASS** | False Pass (Contamination Leak) | The initial consonant onset of the first word 'về' is abruptly cut off at the start of the audio file. | The background, speakers, and both file edges were checked and found to be clean with no defects. |
| `H0VpjeULCck_2b47dea53175` | **REJECT** | **PASS** | False Pass (Contamination Leak) | A secondary, quieter voice enters speaking ('mà chị Anh Tú...').; The speech is cut off abruptly mid-utterance at the end of the audio. | The background, single speaker, and both file edges were carefully checked and found to be free of defects. |
| `H0VpjeULCck_884be69c1d71` | **REJECT** | **PASS** | False Pass (Contamination Leak) | The final word 'đồng' is abruptly cut off mid-phonation before full articulation. | The background is clean, there is only one speaker with no secondary voices, and both the start and end of the file are complete and uninterrupted. The audio passed all inspection sweeps. |
| `H0VpjeULCck_d22bb70da87b` | **REJECT** | **PASS** | False Pass (Contamination Leak) | A second distinct speaker enters and speaks ('Em rùa...') immediately following the first speaker's phrase. | The background, speakers, and both file edges were checked and found to be clean without any audible defects. |
| `H0VpjeULCck_2e87f4943945` | **REJECT** | **PASS** | False Pass (Contamination Leak) | The initial word is cut off at the very beginning of the audio.; Upbeat acoustic guitar background music plays continuously throughout the clip. | The background, speakers, and both file edges were checked. The speech is clean, clear, free of background music, noise, or artifacts, and has complete word onsets and offsets. |
| `H0VpjeULCck_2fea93237c62` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible background chatter and murmurs from other people in the venue throughout the recording.; Persistent ambient hall noise and crowd atmosphere typical of an unconditioned public event recording.; Noticeable acoustic reverberation from the large hall/room environment. | The audio is clean with no background music, sound effects, or secondary speakers. Both the start and end of the file are intact and free from clipping. |
| `H0VpjeULCck_30e23469a5a1` | **REJECT** | **PASS** | False Pass (Contamination Leak) | A distinct secondary voice exclaims/laughs at the very beginning before the primary speaker begins. | The background, speakers, and both file edges were checked and found to be clean without any audible defects. |
| `H0VpjeULCck_9c94fe9c4eb5` | **PASS** | **REJECT** | False Reject (Overly Strict) | pass | There is noticeable background instrumental music playing throughout the clip. |
| `H0VpjeULCck_f6d56b6c4b68` | **REJECT** | **PASS** | False Pass (Contamination Leak) | A high-pitched child's voice calls out ('Ơi') before a different female speaker says 'em chạy qua đây nè'.; Audible hollow room reverberation throughout the recording. | The background, speakers, and both file edges were checked. The speech is clean, free of background noise, music, or secondary speakers, and starts and ends naturally without clipping. |
| `H0VpjeULCck_09b6a4096e6d` | **REJECT** | **PASS** | False Pass (Contamination Leak) | The initial syllable at the very beginning of the audio is cut off abruptly mid-articulation. | The background is clean, there are no secondary speakers, and both file edges are intact without clipping. |
| `H0VpjeULCck_a8f7261a357f` | **REJECT** | **PASS** | False Pass (Contamination Leak) | A prominent cartoon sound effect / squeak audio cue plays immediately after the speech. | The clip has clean foreground speech with no background music, sound effects, or excessive noise. No secondary speaker is audible. Both file edges are clean and free of truncation. |
| `H0VpjeULCck_fc1b74737c62` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible hall reverberation and PA system room reflections throughout the clip. | The background is clean, there is only one primary speaker, and both the start and end of the file are free of clipping or truncation. |
| `H0VpjeULCck_8f88007f4ee1` | **REJECT** | **PASS** | False Pass (Contamination Leak) | The final word 'đã' is cut off unnaturally mid-phonation at the end of the clip. | The background is clean, there is only one primary speaker with no interruptions, and both file edges are intact. Checked carefully. |
| `H0VpjeULCck_f5a056e32ba2` | **REJECT** | **PASS** | False Pass (Contamination Leak) | The final word is abruptly cut off mid-syllable at the end of the audio clip. | The speech is clean and clear with no background music, sound effects, excessive noise, or reverberation. Only one speaker is present throughout the audio. Both file edges are clean with no clipped words at the start or end. |
| `H0VpjeULCck_c9ff6a0ff305` | **REJECT** | **PASS** | False Pass (Contamination Leak) | A secondary male speaker is clearly audible speaking at the beginning of the clip.; The male speaker's voice overlaps with the primary female speaker's utterance. | The background is clean, there is only one primary speaker with no overlapping voices or secondary sounds, and both file edges are intact. |


## Evaluation Prompt

```text
This is a strict reject-defect search for a Vietnamese TTS clip. Do not rate the overall clarity. Search for every audible disqualifier first; PASS is allowed only after all searches below are negative. Clean foreground speech does not cancel a quiet background defect. Both PASS and REJECT are common in this batch, so never default to PASS.

Make three careful listening sweeps:

SWEEP A — BACKGROUND AND EDITS, for the whole clip:
- Reject for any audible music: quiet instrumental bed, melody, beat, bass, drone, synth pad, jingle, or tonal accompaniment. Music under a loud voice still rejects.
- Reject for an important edited or non-speech effect: chime, whoosh, notification, clap/impact, transition, or similarly conspicuous effect.
- Reject for prominent environmental noise, strong hollow echo/reverberation, distortion, separation/phase artifacts, or damaged/muffled voice. Faint harmless room tone is allowed.

SWEEP B — PEOPLE, for the whole clip and especially gaps/tail:
- Reject if any other person is audible, including one quiet word, laugh, chuckle, whisper, breath, shout, or vocalization.
- Reject if another person overlaps the primary speaker. Sequential speech from two different people also rejects.

SWEEP C — EXACT FILE EDGES:
- Listen to the first 500 ms. Reject if speech was already in progress when the file began: missing consonant/onset, entry mid-vowel, or entry mid-syllable.
- Listen to the last 500 ms. Reject if the file stops while the final actually spoken word is still sounding: interrupted vowel/tone, missing coda/release, or speech cut mid-syllable.
- Do not use grammar. An incomplete sentence may pass when its last audible word finishes acoustically. Natural Vietnamese unreleased final stops and naturally abrupt delivery may pass. Reject clipping only from audible edge evidence.

Final check before answering: ask “Did I overlook quiet music, a brief second voice, or a cut first/last syllable?” If any answer is yes, REJECT. If evidence of a defect is truly absent, PASS.

Return exactly one JSON object and nothing else:
{
  "decision": "pass" | "reject",
  "failure_codes": ["secondary_speaker", "overlapping_speech", "clipped_word_start", "clipped_word_end", "music", "sound_effect", "excessive_noise", "reverberation", "voice_damage"],
  "reason": "Brief concrete acoustic evidence. For a pass, state that the background, speakers, and both file edges were checked."
}

For PASS, failure_codes must be []. For REJECT, include each defect heard. Never report a defect solely to satisfy the expected class balance.
```
