# Báo cáo tóm tắt xử lý audio và chi phí verifier

Số liệu ngày 25/09/2026, sau khi bổ sung dữ liệu: **17 file nguồn, 1.633 đoạn đầu vào, thu được 1.626 đoạn pass tương đương 3,7603 giờ. Tổng chi phí ước tính gồm testing: 23,7600 USD.**

## 1. Sản lượng và nguyên nhân hao hụt

| Giai đoạn | Số lượng | Thời lượng | Giữ lại so với nguồn | Độ dài trung bình; min-max | Ghi chú hao hụt |
| --- | ---: | ---: | ---: | --- | --- |
| Sau tách Mel-RoFormer, trước diarization | 17 file | 4,7103 giờ | 100% | 997,48 s/file; 7,19-3.243,90 s | Mốc thời lượng ban đầu |
| Sau diarization, đầu vào verifier | 1.633 đoạn | 3,7719 giờ | 80,08% | 8,32 s/đoạn; 1,50-15,00 s | Mất 55,01 phút ngoài vùng lời nói được giữ lại và 1,30 phút do lọc đoạn ngắn |
| Sau verifier, các mẫu pass | 1.626 đoạn | 3,7603 giờ | 79,83% | 8,33 s/đoạn; 1,50-15,00 s | Giảm thêm 41,70 s: 2 mẫu reject và 5 mẫu lỗi xử lý |

**Hao hụt chủ yếu đến từ các vùng diarizer không giữ lại, không phải do nhiều clip ngắn bị lọc.** Sau khi tính cả khoảng nghỉ được ghép lại, phần ngoài lời nói mất 3.300,37 s, chiếm **97,69% hao hụt trước verifier**. Lọc 85 đoạn dưới 1,5 s chỉ mất 77,88 s, chiếm 2,31% hao hụt; không có đoạn bị loại vì quá dài. Phần ngoài lời nói có thể gồm silence, âm phi lời hoặc lời nói bị diarizer bỏ sót; chưa nghe kiểm để khẳng định tất cả là silence.

Verifier giữ **99,69% thời lượng** đầu vào. Theo số lượng mẫu, tỷ lệ pass là **99,57%** (1.626/1.633). Các tỷ lệ này cập nhật thay cho bảng cũ 1.348 đoạn.

## 2. Vì sao output đắt dù đơn giá chỉ gấp 5 input?

**Đúng là đang dùng nhiều thinking token.** Với cấu hình thinking `medium`, có **3,881 triệu thinking token**, so với chỉ **0,164 triệu token câu trả lời**. Thinking chiếm **95,94% output token tính phí** và tốn **14,5540 USD**; câu trả lời chỉ tốn **0,6163 USD**.

Input rẻ còn vì **96,54% prompt token được đọc từ cache**. Theo bảng giá đã đối chiếu ngày 25/09/2026, input thường là 0,75 USD/triệu token, input cache là 0,075 USD/triệu, output gồm cả thinking là 3,75 USD/triệu. Vì vậy, tổng phí output/input thực tế là **15,16 lần**, dù đơn giá output chỉ gấp 5 input thường. Nguồn: [Google Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing#gemini-3.8-flash).

| Khoản chi phí | USD |
| --- | ---: |
| Input | 1,0010 |
| Output, gồm 14,5540 USD thinking + 0,6163 USD câu trả lời | 15,1703 |
| Cache storage, gồm 0,1045 USD của batch còn pending | 0,1880 |
| **Chi phí cơ sở** | **16,3593** |
| Testing riêng, đã xác nhận cộng thêm | 7,4007 |
| **Tổng ước tính gồm testing** | **23,7600** |

Tính cả testing: **0,01461 USD/mẫu pass**, tương đương **6,3186 USD/giờ audio pass**. Chưa tính testing: 4,3505 USD/giờ pass. Đây là ước tính từ dữ liệu đã lưu và khoản testing được xác nhận, chưa phải hóa đơn cuối; chưa đủ thông tin chia riêng testing thành input/output/cache hoặc tính hết các request lỗi/retry. Nếu chưa cộng cache batch pending, tổng là **23,6556 USD**.

## 3. Verifier có đúng không, vì sao có mẫu không pass?

**Kết quả nhất quán với quy tắc kiểm tra, nhưng chưa đủ căn cứ khẳng định độ chính xác khi nghe audio.** Cả 1.628 kết quả có verdict đều qua kiểm tra định dạng và tính nhất quán; không có mâu thuẫn giữa quyết định pass/reject và mã lỗi.

| Nhóm không pass | Số mẫu | Thời lượng | Lý do |
| --- | ---: | ---: | --- |
| Reject: `vieclam`, đoạn `0047` | 1 | 5,28 s | Model báo có người nói thứ hai (`secondary_speaker`) |
| Reject: `script-full`, đoạn `0033` | 1 | 2,62 s | Model báo có hát (`singing`) |
| Lỗi xử lý | 5 | 33,80 s | 3 lỗi JSON, 2 lỗi generation; chưa có kết luận về chất lượng audio |

Hai mẫu reject phù hợp tiêu chí loại của prompt; cần nghe lại để xác nhận model nhận định đúng. Năm mẫu lỗi cần xử lý lại, không nên coi là audio kém chất lượng. **Tỷ lệ pass cao không đồng nghĩa độ chính xác cao**: cần nghe kiểm cả mẫu pass và reject để đánh giá lỗi bỏ lọt/loại nhầm. Hướng tối ưu chi phí là đánh giá thinking thấp hơn trên tập đã nghe kiểm trước khi đổi cấu hình.

Nguồn số liệu và chi tiết đối chiếu: [statistíc.md](statistíc.md).
