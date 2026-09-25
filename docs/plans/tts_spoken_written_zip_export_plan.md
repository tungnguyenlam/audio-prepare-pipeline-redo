# Kế hoạch export ZIP cho TTS: spoken form và written form

Trạng thái: đề xuất ngày 2026-09-25; chưa triển khai code.
Phạm vi hiện tại là lập plan implementation, CLI và quy tắc tag.

## 1. Contract đầu ra

Một file `.zip` dưới `.data/`. Khi giải nén, ngay tại thư mục đích có:

```text
audio/
    clip_<id>.wav
    clip_<id>.flac
spoken.csv
written.csv
```

Không thêm thư mục dataset bao ngoài, HTML, XLSX, manifest hay JSON vào ZIP.
Mỗi CSV có đúng hai cột, header `audio_path,transcript`, UTF-8 không BOM,
dấu phân cách comma; dùng `csv.writer` để escape dấu phẩy, quote và newline.
Không dùng quy tắc thêm apostrophe chống công thức của CSV review vì sẽ thay đổi
text huấn luyện. Hai CSV có cùng số dòng, thứ tự và audio_path duy nhất.
Đường dẫn là POSIX relative từ thư mục chứa CSV, ví dụ `audio/clip_abc.wav`.
Audio giữ nguyên byte, codec và extension; không resample hay đổi tên đuôi giả WAV.
Chỉ đưa audio có đủ cả hai transcript vào ZIP; không xuất dataset rỗng.

## 2. Nguồn dữ liệu và hiện trạng

- `index.py` tạo inventory audio/hash/metadata, không thu thập transcript.
- `export.py` chỉ xuất bảng manifest JSONL/CSV; `bundle.py` đóng audio và
  `manifest.json`. Hai lệnh này không chọn verifier pass và không tách transcript.
- `export_verifier_handoff.py` đã có `expected_inputs`, `collect_run`, kiểm tra
  source hash, raw response và verdict qua `_verifier_artifacts`/`_verdicts`.
  Package hiện tại phục vụ review, gồm cả reject và các file HTML/XLSX.
- `prompts/full-tags-prompt.md` mục 2–4 quy định emotion `[label]`, event/filler
  `<...>`, `word[ViePhoneme]`, `word[/IPA/]`, `_`, `-` và dấu nghỉ.

Đề xuất thêm command độc lập `export_tts_zip.py/.sh`, dùng kết quả verifier có sẵn
và inventory audio. Đây là đầu ra TTS mới; command review vẫn phục vụ review.
Không đổi tên hàng loạt các command ngắn đã được chọn trước đây.
Không gọi model, không chạy nối tiếp các stage khác, không cài package.
Không tự nhập `corrected_transcript` từ CSV review trong phiên bản này vì chưa có
contract hợp nhất review. Nếu cần dùng bản sửa của người, phải chốt thêm nguồn vào.

## 3. Chọn clip

1. Đọc inventory từ `--input-manifest` (repeatable): indexed `entries` hoặc exported
   `turns`; path/hash theo contract hiện tại. Yêu cầu inventory complete.
2. Tìm verdict trong `--input-dir`, chọn inventory trước khi xử lý configuration.
3. Tái sử dụng validator production cho mọi backend/profile được hỗ trợ.
   Không viết parser response riêng hoặc đoán kết quả mới nhất.
4. `pass` + transcript không rỗng + audio/hash hợp lệ: đưa vào export.
   `reject` hợp lệ: loại khỏi tập TTS và đếm trong log, không coi là run lỗi.
5. Missing/failed/invalid/uncertain/incomplete, hash lệch hoặc nhiều verdict cạnh
   tranh: dừng với lỗi cụ thể. Dùng `--configuration` hoặc thu hẹp input để chọn.
   Không có `--allow-partial` hoặc cơ chế âm thầm skip lỗi cho output training.
6. Profile chỉ kiểm tra speaker, không có transcript, không đủ điều kiện export;
   báo rõ thay vì tạo text hay CSV trống. Transcript legacy không theo grammar
   hiện tại phải báo chưa hỗ trợ, không tự suy diễn cặp written/spoken.

## 4. Quy tắc transcript và tag

Spoken là **toàn bộ câu**, thay mỗi written token đã có annotation bằng nội dung
phiên âm; không chỉ gom các đoạn trong ngoặc. Written giữ written token và bỏ
annotation của token đó. Chữ không có annotation giữ ở cả hai bản.

| Đầu vào | Spoken | Written |
|---|---|---|
| `Xin chào` | `Xin chào` | `Xin chào` |
| `AI[ây_ai]` | `ây_ai` | `AI` |
| `hello[/həˈləʊ/]` | `/həˈləʊ/` | `hello` |
| `9:15[chín_giờ_mười_lăm]` | `chín_giờ_mười_lăm` | `9:15` |
| `[neutral]` | xóa | xóa |
| `[happy]`, `[sad]` và emotion hợp lệ khác | giữ | giữ |
| `<laugh>`, `<sigh>`, `<hesitation>`, `<tounge_click>`… | giữ | giữ |
| Filler mô phỏng `<uh>`, `<mmm>`… | giữ | giữ |
| Dấu câu/dấu nghỉ `. , ? ! ~ *` | giữ | giữ |

Ví dụ theo mặc định đề xuất:

```text
Gốc:    [neutral] Tôi dùng AI[ây_ai], [happy] hello[/həˈləʊ/]! <laugh>
Spoken: Tôi dùng ây_ai, [happy] /həˈləʊ/! <laugh>
Written:Tôi dùng AI, [happy] hello! <laugh>
```

Hai chi tiết đã hỏi người dùng và còn chờ xác nhận:
- Giữ nguyên nội dung trong ngoặc gồm `/.../`, `_`, `-` (đề xuất), hay bỏ `/`
  bao IPA và đổi `_` thành space. Không tự đổi IPA sang chữ Việt.
- Giữ emotion khác `[neutral]` và event/filler (đề xuất), hay bỏ thêm emotion.

Mặc định đề xuất chưa phải lựa chọn đã được người dùng xác nhận. Việc bỏ
`[neutral]` áp dụng ở mọi vị trí, kể cả điểm trở về giọng nền sau emotion khác;
điều này bỏ marker reset của quy ước emotion nguồn theo đúng yêu cầu.

Parser triển khai dạng scanner trái sang phải, phân biệt emotion độc lập với
annotation dính written token. Xác định ranh giới token theo grammar, giữ ký hiệu
bên trong số/ngày giờ như `9:15`, không ăn dấu câu đứng ngoài token. Phân loại
emotion theo vị trí và catalog, không chỉ theo nội dung ngoặc. Cặp annotation
phải có written token, payload không rỗng; ngoặc lồng, ngoặc thiếu, annotation
mồ côi hoặc cấu trúc nhiều từ mơ hồ phải báo clip/vị trí lỗi, không đoán scope.
Giữ nguyên payload và Unicode; chỉ dọn whitespace ở chỗ xóa tag và đầu/cuối câu.
Không dùng regex xóa toàn bộ `[...]`; không xóa nhầm emotion hoặc mất lời.
Không xác nhận lại độ đúng ngữ âm bằng model; đây là chuyển đổi text có cấu trúc.

## 5. CLI dự kiến

```bash
bash scripts/s5-export/export_tts_zip.sh \
  --input-dir .data/s4-agent/verifier/<backend>/<model> \
  --input-manifest .data/clips/<family>/segments.json \
  --output-file .data/s5-export/tts_dataset.zip
```

| Flag | Hoạt động |
|---|---|
| `--input-dir`, `-id` | Bắt buộc; thư mục verdict, duyệt đệ quy |
| `--input-manifest`, `-im` | Bắt buộc; lặp lại được để chọn nhiều inventory không trùng |
| `--output-file`, `-of` | Bắt buộc; đường dẫn ZIP chính xác dưới `.data/` |
| `--configuration` | Chọn settings hash/prefix duy nhất khi có nhiều verdict |
| `--overwrite` | Cho phép thay ZIP đã tồn tại |

Hai CSV luôn được tạo cùng nhau; chưa cần flag chọn riêng spoken/written hoặc
flag biến đổi từng loại tag. Launcher chọn `AUDIO_PYTHON`, môi trường có sẵn,
rồi `python3`, theo hành vi nhẹ của handoff; truyền nguyên arguments.

Log stderr dự kiến: `TTS_EXPORT_SCAN`, `TTS_EXPORT_SELECT`, `TTS_EXPORT_PARSE`,
`TTS_EXPORT_AUDIO`, `TTS_EXPORT_DONE`. Báo số pass/reject, số dòng và lỗi theo clip;
stdout chỉ in đường dẫn ZIP khi thành công. Không dump toàn bộ transcript/raw
response vào log; không đưa credentials/config payload vào package.

## 6. Implementation theo thứ tự

1. Chốt hai quy tắc text còn mở và thêm ví dụ vào contract.
2. Tách phần inventory/thu thập verdict hiện dùng ở handoff sang helper riêng
   `scripts/s5-export/_verifier_export_inventory.py`, để handoff và TTS cùng dùng.
   Giữ folder/catalog/progress đặc thù handoff trong command handoff; helper trả
   rows trung lập. Chỉ chia sẻ hành vi hai consumer hiện tại cùng cần.
3. Trong `export_tts_zip.py`, viết hàm thuần `split_transcript_forms(text)` trả
   spoken/written; parser chỉ phục vụ command này nên không tạo framework chung.
4. Tạo một danh sách record chuẩn gồm source/hash, clip_id, archive_path,
   spoken/written. Tái sử dụng cách tạo clip ID hiện có, kiểm tra collision;
   đường dẫn trong ZIP là `audio/<clip_id><suffix>`, không path tuyệt đối/`..`.
5. Validate toàn bộ trước khi publish. Ghi audio và hai CSV từ cùng record list
   vào ZIP tạm dưới thư mục output; kiểm tra hash audio khi đóng gói để bắt thay
   đổi sau scan. Dùng ZIP64 cho dữ liệu lớn. Không overwrite input/source;
   cleanup file tạm khi lỗi, chỉ thay đích sau khi hoàn tất bằng atomic replace.
   Không cần giải nén/copy toàn bộ audio ra thư mục trung gian.
6. Bổ sung launcher cùng tên, tái sử dụng chọn interpreter nhẹ nếu hai command
   cần chung helper; không dùng launcher có khả năng tự provision package.
7. Cập nhật `docs/commands.md`, `docs/data_contract.md`, README export và workflow
   TTS liên quan; ghi rõ khác biệt dataset training với package review.
   Plan handoff cũ giữ làm lịch sử; đính chính mục đổi tên đã được đảo lại ngày
   2026-09-24 nếu chỉnh vào phần này, không đưa quyết định cũ vào code mới.

## 7. Tiêu chí hoàn thành và validation

- ZIP có đúng `audio/`, `spoken.csv`, `written.csv`; mọi CSV path trỏ tới member
  tồn tại, không thừa audio, không trùng path, hai bảng khớp 1:1.
- Text đúng ví dụ đã chốt; bỏ mọi `[neutral]`; bảo toàn Unicode IPA, dấu tiếng Việt,
  punctuation và các tag giữ lại. Không mất written token hoặc lời ngoài ngoặc.
- Lỗi input/parse/hash không tạo package hoàn chỉnh giả; audio giữ nguyên hash.
- Kiểm tra diff, syntax Python/Bash và `--help` qua launcher, không gọi model.
- Không viết/chạy tests khi chưa có yêu cầu. Các case mixed IPA/ViePhoneme,
  emotion giữa câu, tag lỗi, CSV escaping, basename trùng, ZIP lớn và giải nén
  chuyển máy là acceptance cases cần xác minh khi được yêu cầu, không tuyên bố
  đã được runtime kiểm chứng chỉ nhờ đọc code hoặc syntax/help.
- Cập nhật WORKLOG bằng Bash, commit và push các file thuộc task.
