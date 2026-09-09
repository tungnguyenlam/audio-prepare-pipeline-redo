# Speech Verifier Benchmark Report

- **Evaluated Model:** `gemini-3.5-flash-lite` (gemini)
- **LoRA Adapter:** `None (Base Model)`
- **Reasoning effort:** `medium`
- **Evaluation Dataset:** `.data/tts_strategy/gold_benchmark_20260908/eval_input.jsonl`
- **Total Evaluated:** 288
- **Successful:** 288 (100.0%)
- **Average Latency:** 2.07s

## Verdict Distribution

| Verdict | Count | Percentage |
|---|---|---|
| **Pass** | 269 | 93.4% |
| **Reject** | 19 | 6.6% |
| **Total** | 288 | 100.0% |


## API Usage & Estimated Cost

- **Requests priced:** 288 (unpriced: 0)
- **Tokens:** prompt=253241, audio_in=39545, output=38816, thinking=748, total=292805
- **Estimated USD (paid Standard, as of 2026-09-04):** input=$0.075972, output+thinking=$0.098910, **total=$0.174882**


## Benchmark vs Gemini 3.8 Flash Teacher

- **Total Ground-Truth Reference Items:** 288
- **Overall Agreement Rate:** **61.1%** (176/288)
- **Precision (TTS Purity Confidence):** 58.7%
- **Recall (Passing Yield Retention):** 99.4%
- **F1 Score:** 73.8%

### Confusion Matrix

| Reference \ Model | Model Pass | Model Reject | Total Reference |
|---|---|---|---|
| **Gemini Pass** | 158 (True Pass) | 1 (False Reject) | 159 |
| **Gemini Reject** | 111 (Contamination Leak) | 18 (True Reject) | 129 |
| **Total Model** | 269 | 19 | 288 |


## Disagreements with Gemini 3.8 Flash (112 samples)

| Sample ID | Gemini 3.8 Flash | Candidate Model | Diagnostic | Gemini Reason | Candidate Model Reason |
|---|---|---|---|---|---|
| `lfIbjICmfW0_7dd16cb044a7` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible background music track playing continuously throughout the clip.; The final word/syllable ('vi...') is abruptly cut off before completion at the end of the clip. | All nine acoustic dimensions are acceptable with no detectable defects, secondary speakers, noise, or voice damage. |
| `i0zYcXBjytE_523d9406419e` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible background music/instrumental track playing underneath the speech throughout the clip. | All nine acoustic dimensions are acceptable with clear speech and no discernible defects. |
| `lfIbjICmfW0_ff70a78d5f7c` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible background music plays underneath the speech throughout the entire audio.; The final word 'một' is cut off abruptly at the end of the clip. | All nine acoustic dimensions are acceptable with clear speech, no background noise, music, or artifacts. |
| `lfIbjICmfW0_1f2563edaa07` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Background acoustic guitar music plays underneath the speech throughout the entire clip. | All nine acoustic dimensions are acceptable with clear speech and no audible defects. |
| `QBml8L3wS3Q_dddbc34270a0` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Noticeable hollow room reverberation and echo during 'quý vị thấy một cái không gian hoàn toàn mới nha' following an abrupt acoustic shift. | All nine acoustic dimensions are acceptable with no detectable defects, secondary speakers, noise, or processing artifacts. |
| `j83rzAzRDAI_5023d310e3fb` | **REJECT** | **PASS** | False Pass (Contamination Leak) | A secondary female voice intrudes and starts speaking ('Mà nguyên tắc của...').; The incoming speech at the end of the clip is cut off abruptly. | All nine acoustic dimensions are acceptable with clear speech and no significant defects. |
| `QBml8L3wS3Q_80a5643ac71d` | **REJECT** | **PASS** | False Pass (Contamination Leak) | An audible transition whoosh/chime sound effect plays between the spoken phrases.; The final utterance at the end of the clip is abruptly cut off mid-speech. | All nine acoustic dimensions are acceptable with clear speech and no noticeable defects. |
| `QBml8L3wS3Q_72f64084799d` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Prominent cartoon foley and sound effects (squeaks, rattles, whooshes) accompanying the character vocalization. | All nine audio quality dimensions are acceptable with no detectable defects. |
| `i0zYcXBjytE_086b0a8709c8` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Background music plays continuously throughout the entire clip. | All nine acoustic dimensions are acceptable with no detectable defects. |
| `j83rzAzRDAI_59b8f959fdaf` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible cinematic background music plays throughout the entire recording.; The final word 'cũng' is cut off abruptly before completion. | All nine acoustic dimensions are acceptable with no detectable defects. |
| `PXEtB-CsvSw_04e7792dd749` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible background music/ambient soundtrack playing beneath the speech throughout the entire clip.; Noticeable hall reverberation and PA system room echo on the speech. | All nine acoustic dimensions are acceptable with clear speech and no audible defects. |
| `fwN5VT_QxkY_51c1a609b4c7` | **REJECT** | **PASS** | False Pass (Contamination Leak) | A secondary speaker chuckles/giggles right after the primary speaker finishes saying 'cũng tới'. | All nine acoustic dimensions are acceptable with clear speech, no background noise, music, or interference. |
| `fwN5VT_QxkY_6d101f6ecdaa` | **REJECT** | **PASS** | False Pass (Contamination Leak) | The initial consonant of the first word 'trời' is clipped off at the very beginning of the audio.; Audible hollow room reverberation and echo on the speech throughout the entire clip. | All nine acoustic dimensions are acceptable with clear speech and no detectable defects. |
| `lfIbjICmfW0_a6c4d7397ce7` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible background music plays continuously throughout the entire clip. | All nine acoustic dimensions are acceptable with no detectable defects. |
| `j83rzAzRDAI_f017a0ae4cd7` | **REJECT** | **PASS** | False Pass (Contamination Leak) | A second female speaker enters saying 'mày không gọi, để chị gọi'.; The second speaker overlaps with the primary speaker starting 'Đừng có làm'. | All nine acoustic dimensions are acceptable with no detectable defects, interference, or damage. |
| `i0zYcXBjytE_ae723bd8d792` | **REJECT** | **PASS** | False Pass (Contamination Leak) | The final word/syllable is cut off abruptly mid-articulation at the end of the audio.; Audible dramatic background music/drone is present throughout the audio. | All nine acoustic dimensions are acceptable with no detectable defects. |
| `lfIbjICmfW0_32377309f798` | **REJECT** | **PASS** | False Pass (Contamination Leak) | A prominent digital notification/ding sound effect plays between the spoken phrases.; The speech cuts off abruptly at the end of the clip mid-thought. | All nine acoustic dimensions are acceptable with clear speech and no audible defects. |
| `lfIbjICmfW0_3a265423d415` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible background music plays continuously beneath the vocal track throughout the clip.; Speech is cut off abruptly mid-utterance at the end of the audio. | All nine acoustic dimensions are acceptable with clear speech and no audible defects. |
| `QBml8L3wS3Q_b9a399aa1915` | **REJECT** | **PASS** | False Pass (Contamination Leak) | The initial syllable at the beginning of the clip is cut off abruptly without pre-speech silence.; Audible room reverberation and hollow reflections throughout the recording. | All nine acoustic dimensions are acceptable with clear speech and no significant defects. |
| `i0zYcXBjytE_73a8564afd82` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible background chatter and secondary speech clearly audible underneath the primary speaker.; Secondary voices overlap with the speaker's utterance towards the end of the clip. | All nine acoustic dimensions are acceptable with no detectable defects. |
| `QBml8L3wS3Q_04b2ee6d579b` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Acoustic guitar background music plays continuously throughout the speech. | All nine acoustic dimensions are acceptable with no detectable defects, interference, or artifacts. |
| `lfIbjICmfW0_fa646b94bd27` | **REJECT** | **PASS** | False Pass (Contamination Leak) | The initial consonant of the first word 'khẩn' is cut abruptly at the start of the audio file. | All nine acoustic dimensions are acceptable with clear speech, no background noise, music, or artifacts. |
| `i0zYcXBjytE_96905135c83e` | **REJECT** | **PASS** | False Pass (Contamination Leak) | A secondary voice speaks in the background during 'gai góc'.; Secondary speaker speech overlaps directly with the primary speaker's words.; The final word 'cá' is abruptly cut off before its natural completion. | All nine acoustic dimensions are acceptable with no detectable defects. |
| `i0zYcXBjytE_8652b02f4f76` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible ambient background music / synth pad playing continuously beneath the speech. | The audio is clean, clear, and free of defects across all nine evaluated dimensions. |
| `QBml8L3wS3Q_17b27eb3a783` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Persistent loud ambient noise resembling rushing water from a waterfall is audible throughout the entire recording. | All nine acoustic dimensions are acceptable with no detectable defects. |
| `i0zYcXBjytE_987d3a350c56` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Loud, disruptive coughing and heavy wheezing.; A different speaker interrupts and speaks ('Đó, đúng rồi...'). | All nine acoustic dimensions are acceptable with clear speech and no significant defects. |
| `j83rzAzRDAI_75a698aa3318` | **REJECT** | **PASS** | False Pass (Contamination Leak) | A female voice speaks in the first half of the clip followed by a different male speaker at around 2.5 seconds. | All nine acoustic dimensions are acceptable with no detectable defects. |
| `PXEtB-CsvSw_9e503946b3bf` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Continuous electronic synth music track throughout the entire audio with no spoken words. | All nine acoustic dimensions are acceptable with no detectable defects, music, noise, or voice damage. |
| `i0zYcXBjytE_b489b3e393b8` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Background instrumental music/piano accompaniment plays throughout the clip. | All nine acoustic dimensions are acceptable with no detectable flaws. |
| `fwN5VT_QxkY_1ee7c2c834b8` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Background instrumental music plays throughout the entire audio clip. | All nine acoustic dimensions are acceptable with no detectable defects. |
| `j83rzAzRDAI_da5d334353b0` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible double clapping/slapping sound effect.; A secondary male speaker enters saying 'Nên là...' after the primary female speaker.; The trailing speech is cut off abruptly at the end of the audio. | All nine acoustic dimensions are acceptable with clear speech and no significant defects. |
| `QBml8L3wS3Q_2585cbcbf9c3` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible room reverberation and hollow acoustic reflections throughout the clip from recording in an untreated space. | All nine acoustic dimensions are acceptable with no detectable defects. |
| `Oa-mVxGS4cw_495699047b62` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Continuous instrumental background music plays underneath the speech throughout the entire clip. | All nine acoustic dimensions are acceptable with clear speech, no background noise, music, or artifacts. |
| `i0zYcXBjytE_076d41144ed6` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Dramatic instrumental background music swells up and continues until the end of the clip. | All nine acoustic dimensions are acceptable with no detectable defects. |
| `fwN5VT_QxkY_432970843d6b` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible background music plays continuously throughout the entire clip. | All nine acoustic dimensions are acceptable with no detectable defects. |
| `Oa-mVxGS4cw_1b88b40eeadc` | **REJECT** | **PASS** | False Pass (Contamination Leak) | The final word 'cử' is cut off abruptly before the vowel and tone finish naturally. | All nine acoustic dimensions are acceptable with clear speech and no defects. |
| `j83rzAzRDAI_c6c05afb5834` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible low-pitched chuckle/snicker intrusion from an off-mic secondary person right after the final word. | All nine acoustic dimensions are acceptable with no detectable defects. |
| `j83rzAzRDAI_9932eff26455` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Background instrumental music plays audibly throughout the clip. | All nine acoustic dimensions are acceptable with clear speech and no significant defects. |
| `lfIbjICmfW0_deb876c311ed` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Background instrumental/melodic music is audible beneath the speech throughout the entire clip. | The audio is clean, clear, and meets all criteria across all nine dimensions without any detectable defects. |
| `j83rzAzRDAI_6b7288640713` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible background music plays continuously throughout the entire audio clip.; The final word 'chưa' is cut off abruptly before the vowel completes. | All nine acoustic dimensions are acceptable with no detectable defects. |
| `i0zYcXBjytE_78f3a276fe13` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible background music plays underneath the speech throughout the entire clip. | All nine acoustic dimensions are acceptable with clear speech and no audible defects. |
| `QBml8L3wS3Q_defe3cfd2bf5` | **PASS** | **REJECT** | False Reject (Overly Strict) | pass | The audio contains distinct instrumental music playing throughout. |
| `QBml8L3wS3Q_626bca788f54` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible background instrumental music plays throughout the entire recording. | All nine acoustic dimensions are acceptable with clear speech and no audible defects. |
| `lfIbjICmfW0_c70127e5f0c2` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Continuous background music plays underneath the speech throughout the entire audio. | All nine acoustic dimensions are acceptable and no defects are detected. |
| `i0zYcXBjytE_3779be889c1f` | **REJECT** | **PASS** | False Pass (Contamination Leak) | The initial consonant of the following word is cut off abruptly at the end of the audio. | All nine acoustic dimensions are acceptable with no detectable defects, secondary speakers, noise, or processing artifacts. |
| `lfIbjICmfW0_28f7bd68f798` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible background music plays underneath the speech throughout the entire clip. | All nine acoustic dimensions are acceptable with no detectable defects. |
| `fwN5VT_QxkY_d8e1416c96f8` | **REJECT** | **PASS** | False Pass (Contamination Leak) | A different speaker is heard uttering a truncated phrase before the primary male speaker begins.; The initial speech at the very start of the clip is cut off mid-syllable. | All nine acoustic dimensions are acceptable with no detectable defects. |
| `Oa-mVxGS4cw_629e20b1f079` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Continuous background music is present throughout the entire recording. | The audio is clean, clear, and meets all criteria across the nine dimensions without any detectable defects. |
| `PXEtB-CsvSw_5562a479358c` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible instrumental background music plays underneath the speech throughout the entire clip. | All nine acoustic dimensions are acceptable with clear speech and no audible defects. |
| `PXEtB-CsvSw_ad7ba1b905b0` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Continuous dramatic background music plays throughout the entire audio clip. | The audio is clean, clear, and meets all criteria across all nine dimensions without any defects. |
| `3Nll-JLzvvE_cadda063f046` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible room reverberation and hollow acoustic reflections throughout the recording. | All nine acoustic dimensions are acceptable with clear speech and no audible defects. |
| `Oa-mVxGS4cw_08e43a00e843` | **REJECT** | **PASS** | False Pass (Contamination Leak) | A continuous, prominent typewriter or keyboard clicking sound effect plays throughout the clip. | All nine acoustic dimensions are acceptable with clear speech and no audible defects. |
| `3Nll-JLzvvE_aec44b1111c4` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible background music track plays continuously underneath the speech. | All nine acoustic dimensions are acceptable with clear speech and no detectable defects. |
| `3Nll-JLzvvE_484f50812020` | **REJECT** | **PASS** | False Pass (Contamination Leak) | The audio cuts in abruptly at the start during an ongoing vowel sound, cutting off the onset of the initial word. | The audio is clean, clear, and meets all criteria across all nine dimensions without any detectable defects. |
| `NI8JVXNWlN8_41516b38ef10` | **REJECT** | **PASS** | False Pass (Contamination Leak) | The final word 'bạn' is cut off abruptly mid-syllable as the recording ends. | All nine acoustic dimensions are acceptable with no detectable defects. |
| `NI8JVXNWlN8_a922963f0836` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible background acoustic guitar music plays continuously throughout the recording. | The audio is clean, clear, and meets all criteria across all nine dimensions without any detectable defects. |
| `zK3qFnKZFRo_59831b7cc175` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Loud background music plays continuously under the speech throughout the entire clip. | All nine acoustic dimensions are acceptable with clear speech and no significant defects. |
| `zK3qFnKZFRo_3eabdb8239e5` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible background music plays continuously throughout the entire clip.; The final word is cut off mid-syllable as the recording ends. | All nine acoustic dimensions are acceptable with clear speech and no significant defects. |
| `zK3qFnKZFRo_d7723da54617` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible background chatter from secondary speakers in the room throughout the recording. | The audio is clean, clear, free of background noise, music, and secondary speakers, with acceptable room acoustics and no clipping or processing damage. |
| `zK3qFnKZFRo_d46f0ab286fc` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Continuous background music plays underneath the speech throughout the entire audio. | All nine acoustic dimensions are acceptable with no detectable defects. |
| `zK3qFnKZFRo_682aac453db4` | **REJECT** | **PASS** | False Pass (Contamination Leak) | An audio fragment of an earlier word is clipped right as the clip begins.; Audible background music plays underneath the speech across the entire clip.; A distinct second speaker begins speaking after the first speaker finishes.; The speech cuts off mid-word at the very end of the clip. | All nine acoustic dimensions are acceptable with clear speech, no background music or noise, and no clipping at the start or end. |
| `zK3qFnKZFRo_f7a4508f9b00` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible hollow room reverberation and reflections throughout the entire clip.; Audio cuts off abruptly mid-vocalization right after the final word. | All nine acoustic dimensions are acceptable with clear speech, no unwanted noise, music, or distortion. |
| `zK3qFnKZFRo_cd631106dfb9` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible background music plays continuously throughout the entire clip.; A second female speaker answers the initial question starting at around 2.8s ('Chúng mình được học...'). | All nine acoustic dimensions are acceptable with no detectable defects. |
| `PXEtB-CsvSw_b4efb3957c8e` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible room reflections and hollow reverberation throughout the entire recording. | The audio is clean, clear, and meets all criteria across all nine dimensions without any detectable defects. |
| `PXEtB-CsvSw_4b7db2a5bf27` | **REJECT** | **PASS** | False Pass (Contamination Leak) | The final word 'có' is cut off abruptly mid-phonation before finishing naturally.; Audible room reflections and hollow reverberation present throughout the recording. | All nine acoustic dimensions are acceptable with no defects detected. |
| `3Nll-JLzvvE_269a2efb80f4` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible background music and rhythmic beat playing continuously throughout the entire utterance. | All nine acoustic dimensions are acceptable with clear speech and no noticeable defects. |
| `3Nll-JLzvvE_825c989c1c0a` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Background music with melodic piano/keyboard chords is audible throughout the entire recording. | The audio is clean, clear, and meets all quality standards across all nine dimensions. |
| `3Nll-JLzvvE_307d277028a7` | **REJECT** | **PASS** | False Pass (Contamination Leak) | A digital swoosh / transition sound effect plays underneath the speech during 'là battle'. | All nine acoustic dimensions are acceptable with no detectable defects. |
| `3Nll-JLzvvE_5e2eb7696a66` | **REJECT** | **PASS** | False Pass (Contamination Leak) | The initial syllable/filler at the start of the audio is abruptly cut into without a natural onset. | All nine acoustic dimensions are acceptable with clear speech, no background noise, music, or artifacts. |
| `PXEtB-CsvSw_3dcd448ab798` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Background instrumental music plays audibly throughout the entire clip. | All nine acoustic dimensions are acceptable with clear speech, no unwanted noise, music, distortion, or truncation. |
| `PXEtB-CsvSw_edfa1f05041a` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Continuous instrumental background music is audible throughout the entire speech recording. | The audio file is clean, clear, and free from any noticeable defects across all nine dimensions. |
| `PXEtB-CsvSw_07ad42b90f66` | **REJECT** | **PASS** | False Pass (Contamination Leak) | The final word 'Airbus' is cut off abruptly before the final syllable/consonant is completed.; Faint background music/synth pad is audible behind the voice. | All nine acoustic dimensions are acceptable with no detectable defects, noise, or damage. |
| `PXEtB-CsvSw_4a7222b7e6cf` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Faint secondary chuckle/laughter audible in the background between 'cái' and 'ô cửa'. | All nine acoustic dimensions are acceptable with clear speech and no detectable defects. |
| `3Nll-JLzvvE_ab7847350244` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Noticeable hollow room reverberation and reflections coloring the speaker's voice throughout the clip. | All nine acoustic dimensions are acceptable with no defects detected. |
| `3Nll-JLzvvE_60d59ffec8bb` | **REJECT** | **PASS** | False Pass (Contamination Leak) | The initial word at the very beginning of the audio is truncated mid-syllable, cutting off the initial consonant onset. | All nine acoustic dimensions are acceptable with clear speech and no audible defects. |
| `3Nll-JLzvvE_ccee8b3cad81` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Background instrumental beat/music is audible throughout the clip. | All nine acoustic dimensions are acceptable with no detectable defects. |
| `PXEtB-CsvSw_fcc07fffd479` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Background piano instrumental music plays continuously throughout the entire clip. | All nine acoustic dimensions are acceptable with clear speech, no background noise, music, or artifacts. |
| `PXEtB-CsvSw_804b39bc7047` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible background music plays continuously beneath the spoken audio throughout the clip. | The audio is clean, clear, and meets all criteria across all nine dimensions. |
| `3Nll-JLzvvE_2464a29c335b` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Continuous ambient background music with rhythmic synth pulses is audible beneath the speech throughout the entire clip. | All nine acoustic dimensions are acceptable with clear speech, no background noise, music, or other defects. |
| `3Nll-JLzvvE_82b7f679dd4e` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible acoustic guitar background music plays beneath the speech throughout the clip. | All nine acoustic dimensions are acceptable with no detectable defects. |
| `PXEtB-CsvSw_5eb0ecf1cad9` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible secondary chuckle/laughter intrusion between 'cái' and 'ô cửa'. | All nine acoustic dimensions are acceptable with clear speech and no audible defects. |
| `3Nll-JLzvvE_32ba902adc90` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible background music/synthesizer track playing continuously throughout the recording. | All nine acoustic dimensions are acceptable with clear speech and no defects. |
| `3Nll-JLzvvE_521ae4587a8c` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible background music beat playing throughout the speech. | All nine acoustic dimensions are acceptable with clear speech and no defects. |
| `PXEtB-CsvSw_0407475b8f1c` | **REJECT** | **PASS** | False Pass (Contamination Leak) | The final word cuts off abruptly mid-syllable ('được ph...'). | All nine acoustic dimensions are acceptable with clear speech and no discernible defects. |
| `H0VpjeULCck_6483a2a02485` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Pronounced auditorium/PA system reverberation and room echo audible across the entire recording.; Audience applause and clapping clearly audible in the background under the speech. | All nine acoustic dimensions are acceptable with no detectable defects, interference, or damage. |
| `H0VpjeULCck_c6b6c8d9be6d` | **REJECT** | **PASS** | False Pass (Contamination Leak) | A truncated syllable is cut off mid-word at the very beginning of the clip right before 'cho người tàn tật'. | All nine acoustic dimensions are acceptable with clear speech and no audible defects. |
| `H0VpjeULCck_446b7139ca8b` | **REJECT** | **PASS** | False Pass (Contamination Leak) | A secondary child voice is heard speaking before the main adult female speaker begins.; The final word 'cách' is abruptly cut off before completion at the end of the audio. | The audio is clean, clear, and meets all criteria for TTS training without any audible defects. |
| `H0VpjeULCck_147db6aa6fbb` | **REJECT** | **PASS** | False Pass (Contamination Leak) | A secondary male speaker takes over and says 'hai hai rồi hai hai rồi' after the initial female speaker finishes her phrase. | All nine acoustic dimensions are acceptable with no detectable defects. |
| `H0VpjeULCck_66f6f7e37d62` | **REJECT** | **PASS** | False Pass (Contamination Leak) | A different speaker (child voice) speaks at the beginning before the main speaker begins. | All nine acoustic dimensions are acceptable with clear speech and no audible defects. |
| `H0VpjeULCck_2b47dea53175` | **REJECT** | **PASS** | False Pass (Contamination Leak) | A secondary, quieter voice enters speaking ('mà chị Anh Tú...').; The speech is cut off abruptly mid-utterance at the end of the audio. | All nine acoustic dimensions are acceptable with no detectable defects. |
| `H0VpjeULCck_1175ba7b50f4` | **REJECT** | **PASS** | False Pass (Contamination Leak) | The initial consonant onset of the first word 'về' is abruptly cut off at the start of the audio file. | All nine acoustic dimensions are acceptable with clear speech and no audible defects. |
| `H0VpjeULCck_884be69c1d71` | **REJECT** | **PASS** | False Pass (Contamination Leak) | The final word 'đồng' is abruptly cut off mid-phonation before full articulation. | All nine acoustic dimensions are acceptable with clear speech and no discernible defects. |
| `H0VpjeULCck_2e87f4943945` | **REJECT** | **PASS** | False Pass (Contamination Leak) | The initial word is cut off at the very beginning of the audio.; Upbeat acoustic guitar background music plays continuously throughout the clip. | All nine acoustic dimensions are acceptable with no defects detected. |
| `H0VpjeULCck_981d551b4cc8` | **REJECT** | **PASS** | False Pass (Contamination Leak) | The initial word 'Thôi' is clipped at the start of the audio, cutting off its initial consonant onset. | All nine acoustic dimensions are acceptable with no detectable defects, secondary speakers, noise, or processing damage. |
| `H0VpjeULCck_2fea93237c62` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible background chatter and murmurs from other people in the venue throughout the recording.; Persistent ambient hall noise and crowd atmosphere typical of an unconditioned public event recording.; Noticeable acoustic reverberation from the large hall/room environment. | All nine acoustic dimensions are acceptable. The speech is clear and free of defects. |
| `H0VpjeULCck_30e23469a5a1` | **REJECT** | **PASS** | False Pass (Contamination Leak) | A distinct secondary voice exclaims/laughs at the very beginning before the primary speaker begins. | All nine acoustic dimensions are acceptable with no detectable defects. |
| `H0VpjeULCck_d22bb70da87b` | **REJECT** | **PASS** | False Pass (Contamination Leak) | A second distinct speaker enters and speaks ('Em rùa...') immediately following the first speaker's phrase. | All nine acoustic dimensions are acceptable with clear speech and no audible defects. |
| `H0VpjeULCck_aac9e759c83e` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible acoustic guitar background music accompanies the speech throughout the entire clip. | All nine audio quality dimensions are acceptable with no detectable defects. |
| `H0VpjeULCck_7cbb2f9aea00` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible boxy room reflections and hollow reverberation throughout the recording, characteristic of an untreated room with a distant microphone. | All nine acoustic dimensions are acceptable with clear speech and no audible defects. |
| `H0VpjeULCck_f6d56b6c4b68` | **REJECT** | **PASS** | False Pass (Contamination Leak) | A high-pitched child's voice calls out ('Ơi') before a different female speaker says 'em chạy qua đây nè'.; Audible hollow room reverberation throughout the recording. | All nine acoustic dimensions are acceptable and no defects are detected. |
| `H0VpjeULCck_90ef6eda856e` | **REJECT** | **PASS** | False Pass (Contamination Leak) | The final syllable is abruptly cut off mid-vocalization. | All nine acoustic dimensions are acceptable with clear speech and no discernible defects. |
| `H0VpjeULCck_5022a6284d52` | **REJECT** | **PASS** | False Pass (Contamination Leak) | The final word 'sư' is abruptly truncated before full pronunciation. | All nine acoustic dimensions are acceptable with clear speech and no audible defects. |
| `H0VpjeULCck_321ce95ffd67` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Loud audience applause begins immediately following the speaker's announcement.; Audience cheering and shouting sounds alongside applause.; Audible public address system reverberation and room echo throughout the speech. | All nine acoustic dimensions are acceptable with clear speech and no significant defects. |
| `H0VpjeULCck_a8f7261a357f` | **REJECT** | **PASS** | False Pass (Contamination Leak) | A prominent cartoon sound effect / squeak audio cue plays immediately after the speech. | All nine acoustic dimensions are acceptable with clear speech and no audible defects. |
| `H0VpjeULCck_09b6a4096e6d` | **REJECT** | **PASS** | False Pass (Contamination Leak) | The initial syllable at the very beginning of the audio is cut off abruptly mid-articulation. | All nine acoustic dimensions are acceptable with no detectable defects. |
| `H0VpjeULCck_9f6758e67fc9` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible secondary giggling/laughter in the background during the initial phrase.; Overlapping laughter underneath the primary speaker's voice at the beginning of the clip. | All nine acoustic dimensions are acceptable with no detectable defects. |
| `H0VpjeULCck_f5a056e32ba2` | **REJECT** | **PASS** | False Pass (Contamination Leak) | The final word is abruptly cut off mid-syllable at the end of the audio clip. | The audio is clean, clear, and meets all criteria for speech quality without any noticeable defects across the nine dimensions. |
| `H0VpjeULCck_8f88007f4ee1` | **REJECT** | **PASS** | False Pass (Contamination Leak) | The final word 'đã' is cut off unnaturally mid-phonation at the end of the clip. | The audio is clean, clear, and meets all criteria for a successful recording without any acoustic defects. |
| `H0VpjeULCck_fc1b74737c62` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible hall reverberation and PA system room reflections throughout the clip. | All nine acoustic dimensions are acceptable with no detectable defects, interference, or damage. |
| `H0VpjeULCck_4171b84d5281` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Faint secondary speech/hall announcements are audible in the background behind the primary speaker.; Strong room reverberation and echo characteristic of an untreated event hall/auditorium.; Prominent public venue ambient noise throughout the recording. | All nine acoustic dimensions are acceptable. The speech is clear and free of defects. |
| `H0VpjeULCck_88e4e5b4ea9d` | **REJECT** | **PASS** | False Pass (Contamination Leak) | A secondary male speaker's voice intervenes to say 'tự hào' in the middle of the primary female speaker's sentence. | All nine acoustic dimensions are acceptable with clear speech and no audible defects. |
| `H0VpjeULCck_fef4e9ab4468` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Background acoustic guitar music is audible throughout the entire recording underneath the speech. | All nine acoustic dimensions are acceptable with clear speech, no unwanted secondary voices, background noise, music, sound effects, or processing damage. |


## Evaluation Prompt

```text
Listen directly to this exact audio candidate for Vietnamese Text-to-Speech training. Judge only what is acoustically audible; do not infer defects from meaning, grammar, topic, or sentence completeness.

Audit these nine dimensions independently. A strong primary voice must not hide a quieter defect:

1. speaker_purity: defective if any different person's speech, whisper, breath, laugh, chuckle, or vocalization is audible, even briefly or quietly.
2. overlap: defective if two people speak or vocalize at the same time.
3. word_start: defective only when the recording begins after the first spoken word or syllable has already started, cutting its acoustic onset.
4. word_end: defective only when the recording ends before the last actually spoken word or syllable acoustically finishes, cutting its vowel, tone contour, or coda.
5. music: defective if any instrumental music, beat, melody, drone, synth pad, or musical bed is audible anywhere, however quiet.
6. sound_effect: defective if an important non-speech effect such as a chime, whoosh, notification, clap, impact, or edited transition is audible.
7. noise: defective for prominent or disruptive environmental noise such as traffic, machinery, rushing water, coughing, or similar interference. Do not reject faint harmless room tone.
8. reverberation: defective for clearly hollow, echoing, or strongly reverberant speech from an untreated space. Do not reject mild natural room ambience.
9. voice_damage: defective for clipping distortion, separation artifacts, phase damage, severe muffling, or other processing damage to the voice.

Inspect the first 500 ms and last 500 ms especially carefully for clipped speech and a second person's intrusion. Also inspect the entire clip for low-level continuous music and brief secondary voices. Natural Vietnamese unreleased final stops, same-speaker breaths, a grammatically incomplete phrase, or a naturally abrupt speaking style are not automatically clipping.

Decision rule: return "pass" only when all nine dimensions are "acceptable". Return "reject" when at least one dimension is "defective". If evidence is borderline, use your best acoustic judgment; do not call a defect merely because it is possible.

Return one valid JSON object only, with no markdown or text outside it:
{
  "decision": "pass" | "reject",
  "dimensions": {
    "speaker_purity": "acceptable" | "defective",
    "overlap": "acceptable" | "defective",
    "word_start": "acceptable" | "defective",
    "word_end": "acceptable" | "defective",
    "music": "acceptable" | "defective",
    "sound_effect": "acceptable" | "defective",
    "noise": "acceptable" | "defective",
    "reverberation": "acceptable" | "defective",
    "voice_damage": "acceptable" | "defective"
  },
  "failure_codes": ["secondary_speaker", "overlapping_speech", "clipped_word_start", "clipped_word_end", "music", "sound_effect", "excessive_noise", "reverberation", "voice_damage"],
  "reason": "Concise acoustic evidence for every defective dimension, or a concise statement that all nine checks are acceptable."
}

For a pass, failure_codes must be empty. For a reject, include every audible defect code that caused rejection.
```
