# Speech Verifier Benchmark Report

- **Evaluated Model:** `gemini-3.5-flash-lite` (gemini)
- **LoRA Adapter:** `None (Base Model)`
- **Reasoning effort:** `medium`
- **Evaluation Dataset:** `.data/tts_strategy/gold_benchmark_20260908/eval_input.jsonl`
- **Total Evaluated:** 1
- **Successful:** 1 (100.0%)
- **Average Latency:** 1.66s

## Verdict Distribution

| Verdict | Count | Percentage |
|---|---|---|
| **Pass** | 1 | 100.0% |
| **Reject** | 0 | 0.0% |
| **Total** | 1 | 100.0% |


## API Usage & Estimated Cost

- **Requests priced:** 1 (unpriced: 0)
- **Tokens:** prompt=1083, audio_in=184, output=62, thinking=0, total=1145
- **Estimated USD (paid Standard, as of 2026-09-04):** input=$0.000325, output+thinking=$0.000155, **total=$0.000480**


## Benchmark vs Gemini 3.8 Flash Teacher

- **Total Ground-Truth Reference Items:** 1
- **Overall Agreement Rate:** **0.0%** (0/1)
- **Precision (TTS Purity Confidence):** 0.0%
- **Recall (Passing Yield Retention):** 0.0%
- **F1 Score:** 0.0%

### Confusion Matrix

| Reference \ Model | Model Pass | Model Reject | Total Reference |
|---|---|---|---|
| **Gemini Pass** | 0 (True Pass) | 0 (False Reject) | 0 |
| **Gemini Reject** | 1 (Contamination Leak) | 0 (True Reject) | 1 |
| **Total Model** | 1 | 0 | 1 |


## Disagreements with Gemini 3.8 Flash (1 samples)

| Sample ID | Gemini 3.8 Flash | Candidate Model | Diagnostic | Gemini Reason | Candidate Model Reason |
|---|---|---|---|---|---|
| `j83rzAzRDAI_c6c05afb5834` | **REJECT** | **PASS** | False Pass (Contamination Leak) | Audible low-pitched chuckle/snicker intrusion from an off-mic secondary person right after the final word. | Đã kiểm tra kỹ nền âm không có tạp âm hay âm nhạc, không phát hiện giọng nói thứ hai, và hai mép file hoàn chỉnh không bị cắt lẹm. |


## Evaluation Prompt

```text
Đây là bước tìm lỗi nghiêm ngặt cho một đoạn âm thanh dùng huấn luyện TTS tiếng Việt. Chỉ đánh giá tín hiệu âm thanh thực sự nghe thấy; không dựa vào nội dung, ngữ pháp hoặc việc câu nói có trọn ý hay không. Đừng đánh giá theo cảm giác “giọng chính khá rõ”. Hãy tìm từng lỗi loại bỏ trước; chỉ được PASS sau khi tất cả lượt kiểm tra dưới đây đều không phát hiện lỗi. Cả PASS và REJECT đều thường gặp, vì vậy không được mặc định PASS.

Nghe kỹ theo ba lượt độc lập:

LƯỢT A — NỀN ÂM VÀ ÂM THANH BIÊN TẬP, trên toàn bộ đoạn:
- REJECT nếu thực sự nghe thấy nhạc ở bất kỳ mức âm lượng nào: nhạc nền nhỏ, giai điệu, nhịp/beat, bass, tiếng đàn, drone, synth pad hoặc jingle. Giọng chính lớn không làm nhạc nền trở thành chấp nhận được. Tuy nhiên, không được nhầm cao độ tự nhiên của giọng nói, tiếng ù đều không có tính nhạc hoặc suy đoán mơ hồ với nhạc.
- REJECT nếu nghe thấy hiệu ứng đáng kể như chuông/chime, whoosh, tiếng thông báo, vỗ tay/va đập, âm chuyển cảnh hoặc hiệu ứng biên tập rõ ràng.
- REJECT nếu có tiếng ồn môi trường nổi bật gây cản trở, tiếng phòng rỗng/vang/echo rõ rệt, méo tiếng, lỗi tách nguồn/lệch pha, hoặc giọng bị hỏng/nghẹt nặng. Âm phòng rất nhẹ, vô hại được chấp nhận.

LƯỢT B — NGƯỜI NÓI, trên toàn bộ đoạn và đặc biệt trong khoảng nghỉ/cuối đoạn:
- REJECT nếu nghe thấy bất kỳ người thứ hai nào, kể cả chỉ một từ nhỏ, tiếng cười/khúc khích, thì thầm, hơi thở, tiếng hét hoặc âm giọng rất ngắn.
- REJECT nếu hai người nói/phát âm đồng thời. Hai người nói nối tiếp nhau nhưng không chồng lấn vẫn là lỗi secondary_speaker.

LƯỢT C — ĐÚNG MÉP FILE, tập trung mạnh vào 500 ms đầu và 500 ms cuối:
- clipped_word_start: REJECT khi file bắt đầu giữa một âm tiết/từ đang được phát ra, làm mất phần khởi âm/phụ âm đầu hoặc bật vào giữa nguyên âm. Từ đầu tiên bắt đầu sát mép file nhưng vẫn có đầy đủ khởi âm thì được chấp nhận.
- clipped_word_end: REJECT khi file dừng trong lúc âm tiết/từ cuối thực sự vẫn đang phát, làm đứt nguyên âm, đường thanh điệu hoặc âm cuối. Một câu chưa trọn ý vẫn PASS nếu từ cuối nghe thấy đã hoàn tất về âm học.
- Âm tắc cuối không bật hơi tự nhiên của tiếng Việt, hơi thở cùng người nói và lối nói dừng đột ngột tự nhiên không tự động là lẹm chữ. Chỉ báo lỗi khi nghe được bằng chứng cắt ở mép file.

Trước khi trả lời, tự kiểm lại ba lỗi thường bị bỏ sót: (1) nhạc nền nhỏ liên tục, (2) giọng người thứ hai rất ngắn, (3) âm tiết đầu hoặc cuối bị cắt. Nếu nghe thấy bất kỳ lỗi nào thì REJECT. Nếu không thực sự nghe thấy bằng chứng lỗi thì PASS; không bịa lỗi để cân bằng số nhãn.

Chỉ trả về đúng một JSON hợp lệ, không markdown và không thêm chữ bên ngoài:
{
  "decision": "pass" | "reject",
  "failure_codes": ["secondary_speaker", "overlapping_speech", "clipped_word_start", "clipped_word_end", "music", "sound_effect", "excessive_noise", "reverberation", "voice_damage"],
  "reason": "Bằng chứng âm học cụ thể, ngắn gọn bằng tiếng Việt. Nếu pass, xác nhận đã kiểm tra nền âm, người nói và cả hai mép file."
}

Nếu PASS thì failure_codes phải là []. Nếu REJECT thì liệt kê mọi mã lỗi thực sự nghe thấy.
```
