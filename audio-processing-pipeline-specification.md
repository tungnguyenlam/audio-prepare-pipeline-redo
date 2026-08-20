# 📋 BẢN ĐẶC TẢ YÊU CẦU BÀI TOÁN (SOFTWARE REQUIREMENTS SPECIFICATION)

**Tên dự án:** Audio Processing & Clean Segmentation Pipeline  
**Mã tài liệu:** SPEC-AUDIO-PIPE-01  
**Ngày tạo:** 2026-08-15  
**Trạng thái:** Approved / Ready for Implementation  
**Mục tiêu:** Xây dựng hệ thống pipeline xử lý âm thanh tự động trích xuất các phân đoạn âm thanh chuẩn chất lượng cao phục vụ huấn luyện mô hình AI (TTS, Voice Cloning, ASR).

---

## 1. TỔNG QUAN HỆ THỐNG (SYSTEM OVERVIEW)

Hệ thống nhận vào file âm thanh thô với bất kỳ định dạng, chất lượng hoặc đặc tính âm học nào (nhiều người nói, ồn ào, tạp âm môi trường phức tạp). Hệ thống có nhiệm vụ tự động tiền xử lý, phân đoạn người nói, nhận diện ranh giới từ vựng, lọc nhiễu thông minh và xuất ra tập hợp các file âm thanh ngắn (segments) thỏa mãn các điều kiện kỹ thuật nghiêm ngặt.

```mermaid
flowchart LR
    RawAudio[Raw Input Audio\nĐa người, tạp âm, ồn] --> Pipeline[Audio Processing Pipeline\nRTX 3090 GPU]
    Pipeline --> Segments[Clean Audio Segments\n3s < Duration < 30s\n1 Speaker | Clean | Natural]
    Pipeline --> Meta[Metadata File\nJSON / CSV]
```

---

## 2. PHẠM VI HỆ THỐNG (SCOPE)

### 2.1. Trong phạm vi (In-Scope):
* Nhận dạng và chuyển đổi tự động mọi định dạng audio phổ biến (`.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`).
* Phân tách người nói (Speaker Diarization) và lọc lấy các phân đoạn chỉ có **1 người nói duy nhất**.
* Nhận diện ranh giới từ vựng chính xác đến mili-giây (Word-level timestamps) để đảm bảo không bị cắt cụt từ.
* Khử hoàn toàn tiếng ồn môi trường nhưng bảo tồn tuyệt đối các âm thanh con người (cười, khóc, ho, thở dài, hét...).
* Đảm bảo chất âm tự nhiên, không biến dạng phổ (no phase distortion/robot artifacts).
* Xuất các segment theo ngưỡng thời lượng cấu hình được ($3s < x < 30s$) kèm file metadata.

### 2.2. Ngoài phạm vi / Tạm hoãn (Deferred / Out-of-Scope in v1):
* Tách nguồn âm (Source Separation) cho các đoạn nhiều người nói đè lên nhau cùng một thời điểm (Overlapping Speech) $\rightarrow$ Đánh dấu và lưu trữ riêng để xử lý ở phiên bản sau.
* Tự động nhận diện cảm xúc (Emotion recognition) hoặc dịch ngôn ngữ.

---

## 3. ĐẶC TẢ YÊU CẦU CHỨC NĂNG (FUNCTIONAL REQUIREMENTS)

### [FR-01] Tiền xử lý & Chuẩn hóa Âm thanh (Audio Ingestion & Normalization)
* **FR-01.1:** Hệ thống phải hỗ trợ nạp các file âm thanh có sample rate, bit depth và số kênh bất kỳ (Stereo, 5.1...).
* **FR-01.2:** Tự động chuyển đổi âm thanh đầu vào về chuẩn nội bộ: **Sample Rate: 16,000 Hz**, **Channel: Single Channel (Mono)**, **Format: Float32 Tensor**.

### [FR-02] Phân đoạn Người nói & Lọc Đơn Âm (Speaker Diarization & Single-Speaker Filtering)
* **FR-02.1:** Tự động phân đoạn và gán nhãn người nói cho toàn bộ audio: `[Start_Time, End_Time, Speaker_ID]`.
* **FR-02.2:** Phát hiện vùng có 2 hoặc nhiều người nói đồng thời (**Overlapping Speech**).
* **FR-02.3:** Loại bỏ các vùng Overlap ra khỏi luồng xử lý chính và ghi log riêng. Chỉ giữ lại các phân đoạn thuộc về **duy nhất 1 người nói**.

### [FR-03] Nhận diện Ranh giới Từ & Không Cắt Cụt Chữ (Word Boundary Alignment)
* **FR-03.1:** Sử dụng cơ chế ASR + Forced Alignment (WhisperX + Wav2Vec2 CTC) để xác định thời điểm bắt đầu (`word.start`) và kết thúc (`word.end`) của từng từ phát ra.
* **FR-03.2:** **Quy tắc bất biến (Hard Rule):** Điểm cắt đầu (cut-in) và điểm cắt cuối (cut-out) của một segment **chỉ được phép rơi vào khoảng trống giữa 2 từ liền kề** ($[Word_{i}.end, Word_{i+1}.start]$).
* **FR-03.3:** Chấp nhận phân đoạn bị mất nghĩa ngữ cảnh (cắt giữa câu), nhưng **tuyệt đối không được cắt vào giữa một từ làm mất âm tiết hoặc cụt chữ**.

### [FR-04] Gom phân đoạn & Kiểm soát Độ dài (Smart Chunking Engine)
* **FR-04.1:** Cấu hình thời lượng segment:
  * `MIN_DURATION`: Mặc định $3.0$ giây (có thể điều chỉnh qua config).
  * `MAX_DURATION`: Mặc định $30.0$ giây (có thể điều chỉnh qua config).
* **FR-04.2:** Thuật toán gom từ liên tiếp thuộc cùng một người nói:
  * Nếu thời lượng tích lũy $\ge MIN\_DURATION$ VÀ xuất hiện khoảng ngắt nghỉ (pause giữa 2 từ $\ge 0.3s$) $\rightarrow$ Chốt phân đoạn tại `word.end`.
  * Nếu người nói liên tục và thời lượng đạt ngưỡng an toàn (ví dụ: $25s - 28s$) $\rightarrow$ Tìm ranh giới từ gần nhất trước $30.0s$ để ngắt segment.
* **FR-04.3:** Loại bỏ hoàn toàn các phân đoạn có tổng thời lượng $\le MIN\_DURATION$.

### [FR-05] Khử Nhiễu Môi trường & Bảo tồn Âm thanh Tự nhiên (Speech Enhancement)
* **FR-05.1:** Triệt tiêu các loại nhiễu môi trường: tiếng ồn nền (background noise), tiếng quạt gió, tiếng giao thông, tiếng ồn trắng, tiếng vang phòng (reverberation).
* **FR-05.2:** Bảo tồn các âm thanh phát ra từ cơ thể người (Human Non-verbal Vocalizations) bao gồm: tiếng ho, tiếng cười, tiếng thở, tiếng khóc, tiếng hét...
* **FR-05.3:** Đảm bảo âm sắc của giọng nói giữ nguyên độ dày, độ ấm tự nhiên, **không xuất hiện tiếng kim loại (metallic artifacts) hay giọng méo robot**.

### [FR-06] Xuất Dữ liệu & Metadata (Export & Artifacts)
* **FR-06.1:** Lưu từng segment thành file riêng biệt định dạng `.wav` (PCM 16-bit, 16kHz/24kHz Mono).
* **FR-06.2:** Sinh file `metadata.json` / `metadata.csv` chứa thông tin chi tiết từng segment.

---

## 4. ĐẶC TẢ DỮ LIỆU ĐẦU VÀO / ĐẦU RA (DATA CONTRACTS)

### 4.1. Cấu trúc Thư mục Đầu ra (Output Directory Structure)
```text
output_dir/
├── segments/
│   ├── audio_001_spk0_seg0001.wav
│   ├── audio_001_spk0_seg0002.wav
│   ├── audio_001_spk1_seg0001.wav
│   └── ...
├── deferred_overlaps/
│   └── audio_001_overlap_001.wav
└── metadata.json
```

### 4.2. Schema File Metadata (`metadata.json`)
```json
[
  {
    "segment_id": "audio_001_spk0_seg0001",
    "file_path": "segments/audio_001_spk0_seg0001.wav",
    "source_file": "raw_samples/meeting_noisy.mp3",
    "speaker_id": "SPEAKER_00",
    "start_time_sec": 12.45,
    "end_time_sec": 22.18,
    "duration_sec": 9.73,
    "transcript": "chúng ta sẽ bắt đầu buổi họp ngay bây giờ",
    "sample_rate": 16000,
    "channels": 1
  }
]
```

---

## 5. YÊU CẦU PHI CHỨC NĂNG (NON-FUNCTIONAL REQUIREMENTS)

| Nhóm | Yêu cầu kỹ thuật | Tiêu chuẩn đo lường |
| :--- | :--- | :--- |
| **Hiệu năng & Tốc độ** | Tận dụng GPU NVIDIA RTX 3090 (24GB VRAM) với PyTorch CUDA, xử lý theo cơ chế batching. | **Real-Time Factor (RTF) $\le 0.3$** *(1 giờ audio xử lý xong trong $\le 18$ phút)*. |
| **Bộ nhớ & Tài nguyên** | Kiểm soát GPU VRAM không bị tràn bộ nhớ (Out-Of-Memory - OOM). | VRAM sử dụng $\le 18$ GB khi chạy đồng thời Diarization + WhisperX + Denoise. |
| **Độ chính xác Ranh giới** | Độ lệch timestamp ranh giới từ vựng. | Sai số ranh giới từ $\le 20ms$, **0% trường hợp cắt cụt âm tiết**. |
| **Khả năng Cấu hình** | Cho phép tùy chỉnh tham số qua file `.yaml` hoặc biến môi trường. | Dễ dàng đổi `min_duration`, `max_duration`, `batch_size`, `device`. |

---

## 6. MA TRẬN KIỂM THỬ & TIÊU CHÍ NGHIỆM THU (ACCEPTANCE CRITERIA)

| Mã kiểm thử | Kịch bản kiểm thử (Test Scenario) | Kết quả mong đợi (Expected Result) |
| :--- | :--- | :--- |
| **TC-01** | Input audio có tiếng quạt gió và tiếng xe cộ rất to. | Segment đầu ra sạch ồn nền, giọng người nghe rõ ràng, không bị biến dạng kim loại. |
| **TC-02** | Input có 2 người nói tranh nhau (nói đè lên nhau). | Đoạn nói đè bị tách riêng/bỏ qua, các segment xuất ra chỉ có giọng của đúng 1 người. |
| **TC-03** | Người nói nói một tràng liên tục không ngừng nghỉ trong 45s. | Segment được chia thành các đoạn $< 30s$, điểm chia rơi đúng vào ranh giới giữa 2 từ, không bị cụt chữ. |
| **TC-04** | Người nói đang nói thì cười lớn hoặc ho. | Tiếng cười và tiếng ho được giữ nguyên vẹn trong segment, không bị thuật toán khử ồn xóa mất. |
| **TC-05** | File audio dài 2.5s (dưới ngưỡng tối thiểu 3s). | Hệ thống bỏ qua, không xuất segment rác. |
| **TC-06** | File audio không có tiếng người (chỉ có tiếng nhạc hoặc tiếng ồn). | Hệ thống phát hiện không có speech và kết thúc an toàn, không sinh segment lỗi. |

---

## 7. CẤU HÌNH THAM SỐ HỆ THỐNG (SYSTEM CONFIGURATION DEFAULTS)

```yaml
pipeline:
  device: "cuda"
  sample_rate: 16000
  
segmentation:
  min_duration_sec: 3.0
  max_duration_sec: 30.0
  pause_threshold_sec: 0.3
  
models:
  diarization: "pyannote/speaker-diarization-3.1"
  asr_alignment: "whisperx:large-v2"
  enhancement: "DeepFilterNet3"
```
