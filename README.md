# 🎵 SonicCrawl • YouTube Audio Ingestion Pipeline

Ứng dụng Web tải và trích xuất dữ liệu âm thanh từ YouTube tự động sang `.wav` (PCM 16-bit, mặc định 44.1kHz stereo) để bảo toàn chất lượng cho source separation trước khi tạo dữ liệu Speech / TTS / ASR.

---

## 🌟 Tính Năng Chính

- **YouTube Ingestion Engine:** Tích hợp `yt-dlp` và `ffmpeg`, hỗ trợ mọi định dạng link YouTube (Video, Shorts, `youtu.be`).
- **WAV chất lượng cao mặc định:** Crawl thành PCM 16-bit, **44.1 kHz stereo** để giữ phổ tần và thông tin kênh cho source separation. Có thể chọn 16/24/48 kHz hoặc mono khi cần.
- **Vocal Separation trong Web UI:** Chọn audio đã crawl và chạy **HT Demucs**, **Mel-RoFormer**, hoặc cả hai từ tab *Source Separation*. Nghe, tải và so sánh output trực tiếp trên trình duyệt.
- **Lưu trữ Tự động:** Mọi audio được lưu trực tiếp vào thư mục `audio_crawl/` kèm file `metadata.json`.
- **Giao diện Trực quan:** 
  - Dark Mode hiện đại, hiệu ứng Glassmorphism.
  - Sắp xếp chuẩn 6 tab theo đúng thứ tự Pipeline xử lý âm thanh.
  - Trình phát nhạc tích hợp (Audio Player) hỗ trợ Seek, Play/Pause, Âm lượng, Tua nhanh/chậm.
  - Nút Copy Path nhanh đường dẫn file trên server (`audio_crawl/...`).
  - Nút Download trực tiếp file `.wav`.
  - Quản lý và xóa file tiện lợi.

---

## 🚀 Hướng Dẫn Cài Đặt & Khởi Chạy

### 1. Cài đặt Môi trường (uv / Python 3.10+)

Các lệnh dưới đây phải chạy trong thư mục chứa `pyproject.toml`:

```bash
# Nếu bạn đang ở workspace cha, vào project trước:
cd audio-prepare-pipeline-redo

# Tạo môi trường ảo và cài dependencies cơ bản
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

Python 3.11 là lựa chọn khuyến nghị cho các dependency ML/CUDA. Python 3.10–3.12 cũng phù hợp nếu bản PyTorch CUDA tương ứng đã được cài.

### 2. Cài source-separation models (tuỳ chọn)

Cài HT Demucs để chạy model này từ tab **Source Separation**:

```bash
uv pip install -e '.[separation]'
```

Mel-RoFormer dùng checkpoint và YAML cấu hình theo từng model, nên cần checkout
repository [Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training), cài dependencies của nó, rồi cấu hình các biến môi trường trước khi chạy web:

```bash
export MEL_ROFORMER_DIR=/absolute/path/to/Music-Source-Separation-Training
export MEL_ROFORMER_CONFIG=/absolute/path/to/model_mel_band_roformer.yaml
export MEL_ROFORMER_CHECKPOINT=/absolute/path/to/model_mel_band_roformer.ckpt
```

Kết quả tách nguồn được lưu tại `processed_audio/<model>/<run_id>/` và có thể nghe/tải trực tiếp trong tab Source Separation.

### 3. Khởi chạy Web Application

```bash
bash scripts/start_crawler.sh
```

Mở trình duyệt tại: **`http://localhost:8567`**

---

## 🎚️ Dùng Vocal Separation

1. Crawl audio ở **44.1 kHz / Stereo** (giá trị mặc định), hoặc chọn một file đã tồn tại trong `audio_crawl/`.
2. Mở tab **Source Separation** và chọn audio đầu vào.
3. Chọn `HT Demucs`, `Mel-RoFormer`, hoặc chọn cả hai để benchmark A/B.
4. Nhấn **Tách vocal**. Các model chạy tuần tự để không tranh chấp GPU.
5. Khi hoàn tất, mỗi model hiện một thẻ kết quả. Bấm **Nghe** để phát từ backend hoặc **Tải WAV** để tải stem về máy.

`HT Demucs` được gọi với `--two-stems=vocals`, nên output là vocal và accompaniment. Mel-RoFormer phụ thuộc checkpoint đã cấu hình; các stem WAV mà checkpoint tạo ra đều được hiển thị.

> Source separation không tách từng người nói đang nói chồng lên nhau. Sau bước này vẫn cần diarization để loại overlap. Chỉ chuyển output xuống 16 kHz mono ở bước sau, khi chuẩn bị audio cho ASR/TTS.

### Trạng thái model

Tab Source Separation hiển thị trạng thái trước khi chạy:

- **HT Demucs:** sẵn sàng khi package `demucs` đã được cài.
- **Mel-RoFormer:** sẵn sàng khi repo inference, YAML config và checkpoint tồn tại ở ba biến môi trường nêu trên.

Nếu chọn cả hai mà một model chưa sẵn sàng, model còn lại vẫn có thể hoàn tất và kết quả của nó vẫn được hiển thị.

---

## 📁 Cấu Trúc Module Dự Án

```
.
├── audio_crawl/                      # Thư mục lưu trữ các file .wav và metadata
│   ├── <title>__<video_id>.wav
│   ├── <title>__<video_id>.json      # Sidecar metadata
│   └── metadata.json                 # Metadata toàn bộ dataset
├── processed_audio/                  # Output source separation theo model/run
│   └── <model>/<run_id>/...wav
│
├── docs/
│   └── 2026-08-15_audio-processing-pipeline.md
│
├── scripts/
│   └── start_crawler.sh              # Script chạy ứng dụng Web (port 8567)
│
├── src/
│   ├── web/                          # 🌐 Giao diện Web & REST API Server
│   │   ├── __init__.py
│   │   ├── app.py                    # FastAPI Backend & Endpoints
│   │   └── static/                   # Frontend UI (HTML, CSS, JS)
│   │       ├── index.html
│   │       ├── style.css
│   │       └── app.js
│   │
│   ├── crawler/                      # 📥 Module Crawl Audio từ YouTube
│   │   ├── __init__.py
│   │   ├── downloader.py             # Engine tải & trích xuất WAV
│   │   └── storage.py                # Quản lý audio_crawl/ & metadata
│   │
│   ├── denoise/                      # 🧹 Enhancement & vocal-separation workflow
│   │   ├── __init__.py
│   │   ├── deepfilter.py
│   │   └── source_separation.py      # HT Demucs / Mel-RoFormer runners
│   │
│   ├── separation/                   # 🎚️ Module Tách Nguồn Âm Thanh (Demucs/RoFormer)
│   │   ├── __init__.py
│   │   └── base.py
│   │
│   ├── diarization/                  # 👥 Module Phân Đoạn Người Nói (PyAnnote 3.1)
│   │   ├── __init__.py
│   │   └── pyannote_diarizer.py
│   │
│   ├── alignment/                    # ⏱️ Module Gióng Hàng Từ & VAD (WhisperX)
│   │   ├── __init__.py
│   │   └── whisperx_aligner.py
│   │
│   ├── chunking/                     # ✂️ Module Gom Đoạn Thông Minh (3s < x < 30s)
│   │   ├── __init__.py
│   │   └── smart_chunker.py
│   │
│   └── utils/                        # 🛠️ Dataclass Audio & Tiện ích chung
│       ├── __init__.py
│       └── AudioClass.py
│
├── pyproject.toml                    # Cấu hình dự án & thư viện
└── README.md
```
