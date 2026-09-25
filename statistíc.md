# Thống kê s4-agent cập nhật

Ngày đối chiếu: 25/09/2026. Phạm vi: toàn bộ artifact hiện có trong `data/s4-agent/`, ghép với đúng 17 manifest nguồn trong `data/s3-diarize/`. Số thập phân dùng dấu chấm; thời lượng tính bằng giây trước khi đổi sang giờ. Chi phí là USD, cộng từ số chưa làm tròn, không cộng các bảng tổng với nhau.

**Kết quả mới: 1.633 đoạn, 1.626 pass, 2 reject và 5 lỗi xử lý. Tổng chi phí ước tính gồm cache batch và khoản testing riêng 7.4007 USD là 23.760045250 USD, làm tròn 23.7600 USD.** Đây là tổng các khoản truy được và khoản người dùng xác nhận, không phải hóa đơn xác nhận toàn bộ request đã phát sinh.

## 1. Thời lượng và nguyên nhân hao hụt

Tỷ lệ giữ lại dưới đây là tỷ lệ **thời lượng**, không phải tỷ lệ số mẫu. “Audio gốc” trong báo cáo là WAV sau Mel-RoFormer, trước diarization; không phải tổng mọi file trong thư mục s2.

| Giai đoạn | Số lượng | Tổng thời lượng | Tỷ lệ so với audio gốc | Trung bình | Min-max | Ghi chú hao hụt |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Sau Mel-RoFormer, trước diarization | 17 WAV | 16957.1855 s = 4.710329 giờ | 100% | 997.4815 s/file | 7.1866-3243.9030 s | Mẫu số chung của 17 nguồn có verifier |
| Các lượt nói raw của diarizer | 4.805 lượt | 11900.6200 s = 3.305728 giờ | 70.1804% | 2.4767 s | 0.0200-12.3600 s | 5056.5655 s nằm ngoài các vùng diarizer nhận là speech |
| Sau merge, trước lọc độ dài | 1.718 đoạn | 13656.8200 s = 3.793561 giờ | 80.5371% | 7.9493 s | 0.1200-15.0000 s | Merge giữ lại thêm 1756.2000 s khoảng nghỉ giữa các lượt nói |
| Sau diarization/lọc, đầu vào verifier | 1.633 đoạn | 13578.9400 s = 3.771928 giờ | 80.0778% | 8.3153 s | 1.5000-15.0000 s | Loại 85 đoạn dưới 1.5 s, tổng 77.8800 s; không có đoạn bị loại vì quá dài |
| Sau verifier, chỉ lấy pass | 1.626 đoạn | 13537.2400 s = 3.760344 giờ | 79.8319% | 8.3255 s | 1.5000-15.0000 s | Giữ 99.6929% thời lượng đầu vào verifier; mất 7.9000 s do reject và 33.8000 s do lỗi xử lý |

**Vì sao từ 100% còn khoảng 80%?** Phần lớn là các vùng ngoài speech mà diarizer không giữ lại, sau khi đã trừ khoảng nghỉ được merge thu hồi. Không thể gọi toàn bộ phần này là “silence được nghe kiểm chứng”: nó có thể gồm im lặng, tiếng phi lời hoặc speech diarizer bỏ sót. Báo cáo `duration_loss.json` gọi trường này là `silence_lost_s`; đây là phép tính theo timeline, không phải nhãn chuẩn do người nghe xác nhận.

| Nguyên nhân | Thời lượng mất | Điểm phần trăm so với audio gốc |
| --- | ---: | ---: |
| Ngoài vùng speech, sau khi trừ phần merge giữ lại | 5056.5655 - 1756.2000 = 3300.3655 s | 19.4629 |
| Lọc đoạn ngắn dưới 1.5 s | 77.8800 s, 85 đoạn | 0.4593 |
| Loại vì quá dài | 0 s | 0 |
| Verifier reject | 7.9000 s, 2 đoạn | 0.0466 |
| Verifier lỗi xử lý, chưa có verdict hợp lệ | 33.8000 s, 5 đoạn | 0.1993 |
| Tổng không nằm trong tập pass | 3419.9455 s | 20.1681 |

Phần ngoài speech sau merge chiếm **97.6947% hao hụt trước verifier**; lọc đoạn ngắn chiếm 2.3053%. Không được cộng thẳng `silence_lost_s` raw với loss do clip ngắn mà bỏ qua 1756.2000 s merge thu hồi. Tổng thời lượng lượt nói và hợp các khoảng thời gian bằng nhau ở 17 audit này, nên không có phần overlap bị cộng đôi trong các tổng trên.

Đầu vào verifier có 147 đoạn dưới 3 s, 256 đoạn từ 3 đến dưới 5 s, 638 đoạn từ 5 đến dưới 10 s, 592 đoạn từ 10 đến 15 s. Trung vị: 8.3200 s. Các đoạn ngắn này đã qua ngưỡng 1.5 s, không đồng nghĩa với mẫu bị loại.

## 2. Chi phí cập nhật và đối chiếu bảng cũ

### Các khoản có nguồn

| Khoản | USD chưa làm tròn | Nguồn / cách tính |
| --- | ---: | --- |
| Input không cache | 0.264294000 | 352.392 token x 0.75 / triệu |
| Input đọc cache | 0.736708500 | 9.822.780 token x 0.075 / triệu |
| Tổng input | 1.001002500 | Tổng hai dòng trên |
| Output văn bản | 0.616305000 | 164.348 token x 3.75 / triệu |
| Thinking | 14.554038750 | 3.881.077 token x 3.75 / triệu |
| Tổng output tính phí | 15.170343750 | Output văn bản + thinking |
| Cache storage đã gắn vào verdict | 0.083511500 | Tổng `_cost.cache_storage_usd` |
| Tổng từ JSON từng mẫu | 16.254857750 | Input + output + cache đã gắn vào verdict |
| Cache storage của batch chưa hoàn tất | 0.104487500 | Trường `cache_storage_usd` trong state batch; trình bày riêng bên dưới |
| Chi phí cơ sở gồm cache batch | 16.359345250 | 16.254857750 + 0.104487500 |
| Testing bổ sung, ngoài các JSON | 7.400700000 | Người dùng xác nhận là khoản riêng cần cộng thêm |
| **Tổng gồm testing và cache batch** | **23.760045250** | **16.359345250 + 7.400700000** |

State [batch_a37e4157d704293ac5bf.json](data/s4-agent/verifier/gemini/gemini-3-8-flash/medium/work/batch_jobs/batch_a37e4157d704293ac5bf.json) có `status=submitting`, 107 nhóm `BATCH_STATE_PENDING`, lưu cache storage 0.1044875 USD, chưa có kết quả generation trong state. Các verdict hiện tại đều là `standard`; khoản cache batch này được cộng riêng theo state, không giả định batch đã chạy xong hoặc không phát sinh phí generation. Trạng thái trên là snapshot local, chưa đối chiếu API/billing. **Nếu chỉ tính verdict và testing, chưa đưa cache batch pending vào tổng, con số là 23.655557750 USD.**

Chưa có bảng kê input/output/cache của riêng testing 7.4007 USD. Vì vậy không thể chia chính xác toàn bộ 23.760045250 USD thành chỉ ba dòng input/output/cache. Cơ cấu đúng theo bằng chứng hiện có là: input 1.001002500 USD; output 15.170343750 USD; cache storage 0.187999000 USD; testing chưa phân loại 7.400700000 USD.

| Thành phần trong tổng gồm testing | USD | Tỷ trọng |
| --- | ---: | ---: |
| Input xác định được | 1.001002500 | 4.2130% |
| Output xác định được, gồm thinking | 15.170343750 | 63.8481% |
| Cache storage, gồm batch pending | 0.187999000 | 0.7912% |
| Testing riêng, chưa có phân rã | 7.400700000 | 31.1477% |
| Tổng | 23.760045250 | 100% |

### Đơn giá theo sản lượng hiện tại

| Chỉ số | Cơ sở gồm cache batch, chưa testing | Gồm testing riêng 7.4007 USD |
| --- | ---: | ---: |
| Tổng USD | 16.359345250 | 23.760045250 |
| USD / mẫu đầu vào, 1.633 mẫu | 0.01001797 | 0.01454994 |
| USD / mẫu pass, 1.626 mẫu | 0.01006110 | 0.01461257 |
| USD / phút audio đầu vào verifier | 0.07228552 | 0.10498630 |
| USD / giờ audio đầu vào verifier | 4.33713109 | 6.29917821 |
| USD / giờ audio pass | 4.35049116 | 6.31858214 |

Đơn giá gồm testing phân bổ chi phí thử nghiệm vào sản lượng hiện có; không có nghĩa mỗi request production thực sự tốn mức đó. Chi phí từng mẫu trên 1.628 verdict có giá: trung bình 0.00998456 USD; trung vị 0.00634166 USD; min 0.001265325 USD; **max 0.094422825 USD**, thay cho 0.0916 USD trong bảng cũ. Năm mẫu lỗi không có `_usage`/`_cost`; không tính chúng là request miễn phí.

### Những số cũ cần thay

Snapshot [plot/all_samples.csv](data/s4-agent/verifier/gemini/gemini-3-8-flash/medium/plot/all_samples.csv) mới có 1.348 mẫu. Đọc `cost_json` chưa làm tròn của snapshot cho tổng **13.568842375 USD**. 285 artifact mới đóng góp **2.686015375 USD**, gồm input 0.156950625, output 2.523753750, cache 0.005311000 USD. Tổng hiện tại từ verdict là 16.254857750 USD. Hai nguồn bổ sung là `sep17-freestyle-marry` (127 đoạn) và `sep17-script-full` (158 đoạn).

Nếu cộng cache batch 0.104487500 USD vào snapshot cũ, cơ sở cũ truy được là **13.673329875 USD**, không phải 13.6738 USD: còn chênh **0.000470125 USD** chưa có chứng từ để quy vào khoản nào. Không lấy số cũ đã làm tròn làm điểm xuất phát cho tổng mới.

Trong bảng người dùng đưa, `1.2959 + 19.6911 + 0.0882 = 21.0752 USD`, không bằng `21.0745 USD`; lệch 0.0007 USD. Vì chưa có chi tiết testing và các thành phần cũ không tự khớp, báo cáo này không suy ngược tỷ trọng testing từ bảng đó.

Các tổng tiền này là ước tính theo artifact còn lưu. Request retry, lần generate bị ghi đè, lỗi không lưu usage và generation batch chưa có kết quả local có thể không nằm trong tổng. Muốn chốt số tiền thanh toán tuyệt đối cần đối chiếu billing/log request; không thể phục hồi toàn bộ lịch sử chỉ từ verdict cuối.

## 3. Output đắt có phải do thinking?

**Có.** Cấu hình của cả 1.633 artifact là `gemini-3.8-flash`, `reasoning_effort=medium`, `inference_mode=standard`. 1.628 verdict thành công lưu tổng token như sau:

| Loại token | Số token | Giải thích |
| --- | ---: | --- |
| Prompt tổng | 10.175.172 | Gồm text và audio; không chỉ system prompt |
| Trong đó text | 9.836.108 | Bao gồm phần đọc từ cache |
| Trong đó audio | 339.064 | Không cache |
| Prompt đã cache | 9.822.780 | 96.5367% prompt; là tập con, không cộng thêm vào prompt |
| Output văn bản | 164.348 | Câu trả lời được xuất |
| Thinking | 3.881.077 | Phần suy luận tính theo đơn giá output |
| Output tính phí | 4.045.425 | Văn bản + thinking |
| Tổng token | 14.220.597 | Prompt + văn bản + thinking |

Thinking chiếm **95.9374% output token tính phí**, và **89.5365% tổng phí trong verdict** (mẫu số 16.254857750 USD, chưa testing/cache batch). Output tổng tốn 15.1552 lần input tổng dù đơn giá output chỉ gấp 5 lần input thường, vì phần lớn input được cache với giá thấp hơn và output gồm rất nhiều thinking.

Bảng giá [Google Gemini API](https://ai.google.dev/gemini-api/docs/pricing#gemini-3.8-flash), đối chiếu ngày 25/09/2026: standard input 0.75 USD/triệu token; output **bao gồm thinking** 3.75 USD/triệu; cache read 0.075 USD/triệu; cache storage 0.50 USD/triệu token/giờ. Các mức này khớp price card local `2026-09-21`. Output gấp 5 lần input thường nhưng gấp 50 lần token đọc cache.

Mẫu đắt nhất: `sep17-script-full_mel_roformer_diarizen_spk00_000120652-000130052_0014_gemini.json`, 9.40 s audio, 109 output token văn bản nhưng **24.915 thinking token**, tổng 0.094422825 USD. Đây là bằng chứng trực tiếp chi phí cao không xuất phát chủ yếu từ transcript dài.

Giảm thinking có thể giảm chi phí, nhưng dữ liệu hiện tại chỉ dùng medium, không đủ để kết luận cấu hình thấp hơn giữ nguyên chất lượng. Không có lần gọi model hoặc thử nghiệm trả phí mới trong lần thống kê này.

## 4. Verifier hoạt động đúng không? Những mẫu nào không pass?

Về tính nhất quán artifact: 1.628 verdict đều qua validator hiện có (`acoustic_defect_v3`), không thấy mâu thuẫn decision/failure_codes, thiếu transcript ở pass hoặc transcript xuất hiện ở reject. Cả 1.628 raw response đều tồn tại và khớp SHA-256 đã lưu. 1.633 source clip ghép đủ vào manifest tương ứng và source SHA-256 khớp clip SHA-256 trong manifest; không thiếu mẫu trong 17 nguồn này. Đây là đối chiếu metadata, không phải hash lại toàn bộ WAV hoặc đánh giá bằng tai.

| Kết quả | Số đoạn | Tổng giây | Trung bình | Min-max | Ý nghĩa |
| --- | ---: | ---: | ---: | ---: | --- |
| Pass | 1.626 | 13537.24 | 8.3255 s | 1.50-15.00 s | Có transcript và verdict hợp lệ |
| Reject | 2 | 7.90 | 3.9500 s | 2.62-5.28 s | Model xác định không đạt tiêu chí |
| Lỗi xử lý | 5 | 33.80 | 6.7600 s | 4.42-9.88 s | Chưa có kết luận chất lượng audio |

Tỷ lệ pass theo **số mẫu đầu vào**: 1626/1633 = **99.5713%**. Nếu chỉ lấy verdict hợp lệ làm mẫu số: 1626/1628 = **99.8771%**. Hai tỷ lệ này khác tỷ lệ giữ **thời lượng** 99.6929%.

### Hai mẫu bị reject

| Nguồn và ID đoạn | Vị trí / thời lượng | Failure code | Lý do model ghi |
| --- | --- | --- | --- |
| `sep-16-sample-diemtinh-vieclam`, `0047` | 521.572-526.852 s; 5.28 s | `secondary_speaker` | “The audio contains two distinct speakers in an interview dialogue.” |
| `sep17-script-full`, `0033` | 289.652-292.272 s; 2.62 s | `singing` | “The speaker is singing a melodic phrase.” |

Artifact: [vieclam 0047](data/s4-agent/verifier/gemini/gemini-3-8-flash/medium/sep-16-sample-diemtinh-vieclam_mel_roformer/sep-16-sample-diemtinh-vieclam_mel_roformer_diarizen_spk00_000521572-000526852_0047_gemini.json), [script-full 0033](data/s4-agent/verifier/gemini/gemini-3-8-flash/medium/sep17-script-full_mel_roformer/sep17-script-full_mel_roformer_diarizen_spk00_000289652-000292272_0033_gemini.json).

Hai quyết định khớp quy tắc prompt: có người nói thứ hai hoặc hát đều reject. Trường hợp hát vẫn có thể `pure + complete + studio_clean` nhưng reject do eligibility `singing`; đó không phải mâu thuẫn. Chưa nghe kiểm và chưa có nhãn chuẩn độc lập, nên **chưa thể kết luận model nghe đúng, tính precision/recall, hoặc khẳng định không bỏ lọt lỗi trong các mẫu pass**. Tỷ lệ pass rất cao không chứng minh độ chính xác cao.

### Năm mẫu lỗi xử lý

| Nguồn | ID | Vị trí (s) | Dài (s) | Mã lỗi |
| --- | --- | --- | ---: | --- |
| `assistant-response-1` | `0145` | 1489.992-1499.872 | 9.88 | `invalid_json`, `VerifierResponseError` |
| `assistant-response-1` | `0241` | 2440.392-2445.732 | 5.34 | `invalid_json`, `VerifierResponseError` |
| `assistant-response-1` | `0198` | 2028.032-2032.732 | 4.70 | `invalid_json`, `VerifierResponseError` |
| `raw-assistant-response-2-2-4` | `0023` | 231.012-235.432 | 4.42 | `generation_failed`, `GeminiResponseError` |
| `raw-assistant-response-p5-10` | `0048` | 457.132-466.592 | 9.46 | `generation_failed`, `GeminiResponseError` |

Cả năm file `.txt` tương ứng đều rỗng. JSON không lưu usage/cost hoặc chi tiết provider đủ để quy nguyên nhân sâu hơn; không suy đoán là silence, cắt từ hay audio lỗi. Các mẫu này cần xử lý lại trước khi quyết định giữ/loại. Verifier chỉ ghi verdict, không tự xóa WAV.

## 5. Chi tiết 17 nguồn

Tên nguồn dưới đây bỏ hậu tố `_mel_roformer`. Mọi thời lượng là giây. “Ngoài speech” đã trừ khoảng nghỉ merge thu hồi; “ngắn” là thời lượng bị lọc dưới 1.5 s. Cột P/R/F lần lượt pass/reject/lỗi xử lý. USD chỉ gồm chi phí verdict của nguồn, chưa phân bổ cache batch/testing.

| Nguồn | Đoạn | Trước diarization | Đầu vào verifier | Giữ lại | TB đoạn | Min-max | Ngoài speech | Ngắn | P/R/F | USD |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | ---: |
| 2speaker | 63 | 685.31 | 611.56 | 89.24% | 9.71 | 1.60-15.00 | 69.81 | 3.94 | 63/0/0 | 0.9428 |
| PO | 31 | 249.54 | 199.60 | 79.99% | 6.44 | 1.56-14.92 | 44.64 | 5.30 | 31/0/0 | 0.3384 |
| assistant-response-1 | 319 | 3243.90 | 2528.28 | 77.94% | 7.93 | 1.50-14.94 | 709.30 | 6.32 | 316/0/3 | 1.9486 |
| assistant-response-27min | 142 | 1559.60 | 1204.32 | 77.22% | 8.48 | 1.82-15.00 | 355.28 | 0.00 | 142/0/0 | 1.6015 |
| assistant-response-28min | 141 | 1614.28 | 1240.70 | 76.86% | 8.80 | 1.56-14.90 | 368.12 | 5.46 | 141/0/0 | 1.6083 |
| assistant-response-5min | 25 | 271.64 | 204.56 | 75.31% | 8.18 | 1.66-14.46 | 67.08 | 0.00 | 25/0/0 | 0.2694 |
| emotional-intelligence | 101 | 1074.90 | 917.40 | 85.35% | 9.08 | 1.56-14.96 | 143.68 | 13.82 | 101/0/0 | 1.5093 |
| ngalevi_assistant1 | 125 | 1607.09 | 1161.30 | 72.26% | 9.29 | 4.52-14.18 | 444.43 | 1.36 | 125/0/0 | 0.9266 |
| raw-assistant-response-1 | 20 | 216.82 | 177.90 | 82.05% | 8.90 | 3.30-14.02 | 38.92 | 0.00 | 20/0/0 | 0.1483 |
| raw-assistant-response-2-2-4 | 29 | 275.43 | 216.38 | 78.56% | 7.46 | 1.96-14.88 | 57.61 | 1.44 | 28/0/1 | 0.1656 |
| raw-assistant-response-p5-10 | 91 | 928.91 | 735.14 | 79.14% | 8.08 | 1.90-14.82 | 189.83 | 3.94 | 90/0/1 | 0.9728 |
| sep-16-sample-diemtinh-vieclam | 98 | 1016.89 | 876.92 | 86.24% | 8.95 | 1.60-14.98 | 127.39 | 12.58 | 97/1/0 | 1.4980 |
| sep-16-sample-share-conversation | 87 | 804.38 | 678.62 | 84.37% | 7.80 | 1.64-14.54 | 119.86 | 5.90 | 87/0/0 | 0.9804 |
| sep16-sample-diemtinh-select | 1 | 7.19 | 6.14 | 85.44% | 6.14 | 6.14-6.14 | 1.05 | 0.00 | 1/0/0 | 0.0156 |
| sep17-butterfly-effect | 75 | 728.95 | 632.42 | 86.76% | 8.43 | 1.84-15.00 | 95.45 | 1.08 | 75/0/0 | 0.6431 |
| sep17-freestyle-marry | 127 | 1225.18 | 1054.44 | 86.06% | 8.30 | 1.54-14.96 | 158.70 | 12.04 | 127/0/0 | 1.3213 |
| sep17-script-full | 158 | 1447.17 | 1133.26 | 78.31% | 7.17 | 1.62-14.92 | 309.21 | 4.70 | 157/1/0 | 1.3647 |

## 6. Phương pháp và giới hạn

- Chỉ cộng JSON có `operation=verify`, mỗi artifact một lần. Không cộng lại 18 JSON phân tích, CSV, Markdown hoặc batch request như mẫu đầu vào mới.
- Ghép source theo tên clip trong manifest duy nhất của từng nguồn, đối chiếu SHA-256 metadata. Đường dẫn cũ `.data/...` và thư mục nguồn từng được di chuyển không được coi là nguồn mới.
- Tổng giây mỗi clip lấy từ `end_s - start_s` trong `segments.json`; thời lượng WAV nguồn lấy `duration_loss.json` đã có `probed_audio=true`. Raw/merged lấy manifest tương ứng; không suy ngược thời lượng từ tỷ lệ đã làm tròn.
- Có 10 nội dung prompt khác nhau trong cùng cấu hình model/medium/standard. Đây là tổng của các lần chạy đã lưu, không phải benchmark một prompt cố định.
- Chi phí cộng trường `_cost` bằng số thập phân chính xác và đối chiếu công thức token. Cache batch 0.1044875 USD là khoản state riêng, testing 7.4007 USD là khoản người dùng xác nhận; chưa có phân rã token của testing.
- Đã đọc launcher, runner Gemini, helper pricing, validator và tài liệu verifier. Không sửa pipeline, không chạy test (người dùng không yêu cầu), không gọi model trả phí, không khẳng định đã nghe kiểm audio.
- Các báo cáo cũ trong `plot/` là snapshot 1.348 mẫu; dùng báo cáo này cho số liệu cập nhật, không cộng lẫn hai snapshot.
