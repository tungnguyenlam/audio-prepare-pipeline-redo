# Gemini 3.5 Flash-Lite Prompt Attempt 3 — Vietnamese Defect-First

- Reference: Gemini 3.8 Flash, `thinkingLevel=MEDIUM`, rubric `tts-v1`
- Candidate: `gemini-3.5-flash-lite`, reasoning effort `medium`
- Dataset: `gold_benchmark_20260908` (288 clips; 159 pass / 129 reject)
- Successful verdicts: 288 / 288
- Pass / reject: 266 / 22
- Agreement: 173 / 288 (60.1%)
- True pass / true reject: 155 / 18
- False pass / false reject: 111 / 4
- Precision / recall / F1: 58.3% / 97.5% / 72.9%
- Mean latency: 1.75 s
- Usage: 298,457 prompt + 18,796 answer + 915 thinking = 318,168 tokens
- Estimated paid-Standard cost: $0.1388146

One initial response contained an invalid JSON escape. It is preserved in
`v3_vietnamese_defect_first.json`; the one-item successful retry is preserved in
`v3_retry.json`. `v3_vietnamese_defect_first_complete.json` and its CSV merge the
retry and contain all 288 successful verdicts.

The Vietnamese prompt did not improve the Lite model. It returned to a strong
pass bias and used only 915 thinking tokens across 288 requests. Its defect
recall was 9/61 music, 11/37 secondary speaker, 3/29 clipped end, 0/13 clipped
start, 6/16 sound effect, 2/17 reverberation, 3/8 overlap, and 1/3 excessive
noise.

The exact prompt is saved in `prompt_v3_vietnamese_defect_first.txt`.

