// ==========================================================================
// SonicStudio • Audio Processing Pipeline Client Application
// ==========================================================================

// Theme Management (Light & Dark Mode)
const themeToggleBtn = document.getElementById("themeToggleBtn");

function initTheme() {
    const savedTheme = localStorage.getItem("theme");
    const systemPrefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    const theme = savedTheme || (systemPrefersDark ? "dark" : "light");
    document.documentElement.setAttribute("data-theme", theme);
}

function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute("data-theme") || "dark";
    const newTheme = currentTheme === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", newTheme);
    localStorage.setItem("theme", newTheme);
    showToast(`Đã chuyển sang ${newTheme === "dark" ? "Dark Mode (Tối)" : "Light Mode (Sáng)"}`, "success", 2000);
}

if (themeToggleBtn) {
    themeToggleBtn.addEventListener("click", toggleTheme);
}

// ==========================================================================
// Tab Switching Management
// ==========================================================================
const tabButtons = document.querySelectorAll(".tab-btn");
const tabPanes = document.querySelectorAll(".tab-pane");

function switchTab(tabId) {
    tabButtons.forEach(btn => {
        if (btn.dataset.tab === tabId) {
            btn.classList.add("active");
        } else {
            btn.classList.remove("active");
        }
    });

    tabPanes.forEach(pane => {
        if (pane.id === `tab-${tabId}`) {
            pane.classList.add("active");
        } else {
            pane.classList.remove("active");
        }
    });

    if (tabId === "separation") {
        loadSeparationInputs();
        loadSeparationModelStatus();
        loadSeparationHistory();
    } else if (tabId === "benchmark") {
        loadBenchmarkInputs();
        loadBenchmarkHistory();
    } else if (tabId === "diarization") {
        loadDiarizationSources();
        loadDiarizationHistory();
        initDiarizationState();
    }
}

tabButtons.forEach(btn => {
    btn.addEventListener("click", () => {
        const tabId = btn.dataset.tab;
        switchTab(tabId);
    });
});

// ==========================================================================
// State & Audio Player Elements
// ==========================================================================
let audioList = [];
let currentlyPlaying = null;
let isAudioPlaying = false;

let separationTimerInterval = null;
let separationStartTime = null;

let benchmarkTimerInterval = null;
let benchmarkStartTime = null;

// Stage 1 Crawl Form Elements
const crawlForm = document.getElementById("crawlForm");
const youtubeUrlInput = document.getElementById("youtubeUrl");
const sampleRateSelect = document.getElementById("sampleRateSelect");
const channelSelect = document.getElementById("channelSelect");
const bypassSelect = document.getElementById("bypassSelect");
const submitCrawlBtn = document.getElementById("submitCrawlBtn");
const pasteBtn = document.getElementById("pasteBtn");
const clearInputBtn = document.getElementById("clearInputBtn");
const progressBox = document.getElementById("progressBox");

// Stage 2 Source Separation Elements
const separationForm = document.getElementById("separationForm");
const separationInputSelect = document.getElementById("separationInputSelect");
const directAudioUpload = document.getElementById("directAudioUpload");
const uploadAudioBtn = document.getElementById("uploadAudioBtn");
const runSeparationBtn = document.getElementById("runSeparationBtn");
const separationProgress = document.getElementById("separationProgress");
const sepProgressTimer = document.getElementById("sepProgressTimer");
const separationResults = document.getElementById("separationResults");
const htdemucsStatus = document.getElementById("htdemucsStatus");
const melRoformerStatus = document.getElementById("melRoformerStatus");
const separationHistoryContainer = document.getElementById("separationHistoryContainer");
const refreshHistoryBtn = document.getElementById("refreshHistoryBtn");

// Stage 3 Separation Benchmark Elements
const benchmarkForm = document.getElementById("benchmarkForm");
const benchmarkInputSelect = document.getElementById("benchmarkInputSelect");
const benchmarkDurationSelect = document.getElementById("benchmarkDurationSelect");
const runBenchmarkBtn = document.getElementById("runBenchmarkBtn");
const benchmarkProgress = document.getElementById("benchmarkProgress");
const benchProgressTimer = document.getElementById("benchProgressTimer");
const benchmarkResults = document.getElementById("benchmarkResults");
const benchmarkHistoryContainer = document.getElementById("benchmarkHistoryContainer");
const refreshBenchmarkHistoryBtn = document.getElementById("refreshBenchmarkHistoryBtn");

// Header Stats
const statTotalFiles = document.getElementById("statTotalFiles");
const statTotalDuration = document.getElementById("statTotalDuration");
const statTotalSize = document.getElementById("statTotalSize");

// Library Elements
const audioListContainer = document.getElementById("audioListContainer");
const emptyState = document.getElementById("emptyState");
const searchInput = document.getElementById("searchInput");
const refreshBtn = document.getElementById("refreshBtn");
const toastContainer = document.getElementById("toastContainer");

// Player Elements
const playerBar = document.getElementById("playerBar");
const globalAudio = document.getElementById("globalAudio");
const playerThumb = document.getElementById("playerThumb");
const playerTitle = document.getElementById("playerTitle");
const playerUploader = document.getElementById("playerUploader");
const playerPlayBtn = document.getElementById("playerPlayBtn");
const playIcon = document.getElementById("playIcon");
const pauseIcon = document.getElementById("pauseIcon");
const playerSeekBack = document.getElementById("playerSeekBack");
const playerSeekFwd = document.getElementById("playerSeekFwd");
const scrubberBar = document.getElementById("scrubberBar");
const scrubberFill = document.getElementById("scrubberFill");
const scrubberHandle = document.getElementById("scrubberHandle");
const playerCurrentTime = document.getElementById("playerCurrentTime");
const playerTotalTime = document.getElementById("playerTotalTime");
const playerVolume = document.getElementById("playerVolume");
const playerMuteBtn = document.getElementById("playerMuteBtn");
const playerDownloadBtn = document.getElementById("playerDownloadBtn");

// ==========================================================================
// Helper Utilities
// ==========================================================================

function formatTime(seconds) {
    if (isNaN(seconds) || seconds < 0) return "00:00";
    const totalSecs = Math.floor(seconds);
    const hrs = Math.floor(totalSecs / 3600);
    const mins = Math.floor((totalSecs % 3600) / 60);
    const secs = totalSecs % 60;
    if (hrs > 0) {
        return `${String(hrs).padStart(2, '0')}:${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
    }
    return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
}

function showToast(message, type = "success", duration = 4000) {
    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;
    
    const icon = type === "success" 
        ? `<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"></polyline></svg>`
        : `<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>`;

    toast.innerHTML = `
        <span class="toast-icon">${icon}</span>
        <span>${message}</span>
    `;

    toastContainer.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = "0";
        toast.style.transform = "translateX(20px)";
        toast.style.transition = "all 0.3s ease";
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

// ==========================================================================
// Audio Library (Stage 1) Data & Rendering
// ==========================================================================

async function loadAudioLibrary() {
    try {
        const res = await fetch("/api/audio");
        if (!res.ok) throw new Error("Không thể tải danh sách audio.");
        const data = await res.json();
        
        audioList = data.items || [];
        
        if (data.stats) {
            statTotalFiles.textContent = data.stats.total_files || 0;
            statTotalDuration.textContent = data.stats.total_duration_formatted || "00:00";
            statTotalSize.textContent = data.stats.total_size_formatted || "0 MB";
        }
        
        renderAudioList(audioList);
        loadSeparationInputs();
        loadBenchmarkInputs();
    } catch (err) {
        console.error(err);
        showToast(err.message, "error");
    }
}

function renderAudioList(items) {
    const query = searchInput.value.trim().toLowerCase();
    const filtered = items.filter(item => {
        if (!query) return true;
        const title = (item.title || "").toLowerCase();
        const uploader = (item.uploader || "").toLowerCase();
        return title.includes(query) || uploader.includes(query);
    });

    audioListContainer.innerHTML = "";

    if (filtered.length === 0) {
        emptyState.classList.remove("hidden");
        return;
    }

    emptyState.classList.add("hidden");

    filtered.forEach(item => {
        const isCurrent = currentlyPlaying && currentlyPlaying.filename === item.filename;
        const card = document.createElement("div");
        card.className = `audio-item-card ${isCurrent && isAudioPlaying ? 'active-playing' : ''}`;
        card.dataset.filename = item.filename;

        const thumbSrc = item.thumbnail || "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='80' height='50' viewBox='0 0 80 50'%3E%3Crect width='80' height='50' fill='%231e293b'/%3E%3Ctext x='50%25' y='50%25' dominant-baseline='middle' text-anchor='middle' fill='%2364748b' font-size='12'%3EWAV%3C/text%3E%3C/svg%3E";

        card.innerHTML = `
            <div class="item-left">
                <div class="item-thumb-wrapper" onclick="playAudio('${item.filename}')">
                    <img class="item-thumb" src="${thumbSrc}" alt="thumb" onerror="this.src='data:image/svg+xml,%3Csvg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'80\\' height=\\'50\\' viewBox=\\'0 0 80 50\\'%3E%3Crect width=\\'80\\' height=\\'50\\' fill=\\'%231e293b\\'/ %3E%3C/svg%3E'">
                    <div class="thumb-play-overlay">
                        <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
                            <polygon points="5 3 19 12 5 21 5 3"></polygon>
                        </svg>
                    </div>
                </div>
                <div class="item-details">
                    <div class="item-title" title="${item.title}">${item.title}</div>
                    <div class="item-meta">
                        <span class="item-uploader">${item.uploader || "YouTube"}</span>
                        <span class="badge badge-wav">WAV</span>
                        <span class="badge badge-spec">${(item.sample_rate || 16000) / 1000}kHz • ${item.channels || 'mono'}</span>
                        <span class="badge badge-size">${item.filesize_formatted || '0 MB'}</span>
                        <span class="item-uploader">⏱️ ${item.duration_formatted || '00:00'}</span>
                    </div>
                </div>
            </div>

            <div class="item-actions">
                <button class="btn-play-item" onclick="playAudio('${item.filename}')" title="Phát audio">
                    ${isCurrent && isAudioPlaying 
                        ? `<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect></svg>`
                        : `<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>`
                    }
                </button>
                <button class="btn-icon-action" onclick="copyFilePath('audio_crawl/${item.filename}')" title="Sao chép đường dẫn file">
                    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
                        <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                    </svg>
                </button>
                <a href="/api/audio/${item.filename}" download="${item.filename}" class="btn-icon-action" title="Tải xuống WAV">
                    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                        <polyline points="7 10 12 15 17 10"></polyline>
                        <line x1="12" y1="15" x2="12" y2="3"></line>
                    </svg>
                </a>
                <button class="btn-icon-action" onclick="deleteAudio('${item.filename}')" title="Xóa audio">
                    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="3 6 5 6 21 6"></polyline>
                        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                    </svg>
                </button>
            </div>
        `;

        audioListContainer.appendChild(card);
    });
}

// ==========================================================================
// Stage 2: Source Separation Functions
// ==========================================================================

function loadSeparationInputs(selectFilename = null) {
    if (!separationInputSelect) return;
    const currentVal = selectFilename || separationInputSelect.value;
    separationInputSelect.innerHTML = "";
    
    if (audioList.length === 0) {
        separationInputSelect.add(new Option("Chưa có audio trong thư viện (hãy crawl hoặc upload)", ""));
        return;
    }

    separationInputSelect.add(new Option("Chọn audio cần tách vocal…", ""));
    audioList.forEach(item => {
        const label = `${item.title || item.filename} (${item.duration_formatted || "00:00"})`;
        const option = new Option(label, item.filename, false, item.filename === currentVal);
        separationInputSelect.add(option);
    });
}

const htThenMelStatus = document.getElementById("htThenMelStatus");
const melThenHtStatus = document.getElementById("melThenHtStatus");
const deepfilternetStatus = document.getElementById("deepfilternetStatus");

function getModelDisplay(modelKey) {
    switch (modelKey) {
        case "htdemucs":
            return { label: "⚡ HT Demucs (v4)", badgeClass: "badge-ht" };
        case "mel_roformer":
            return { label: "💎 Mel-Band RoFormer (SOTA)", badgeClass: "badge-roformer" };
        case "deepfilternet":
            return { label: "🛡️ DeepFilterNet (v3)", badgeClass: "badge-df" };
        case "ht_then_mel":
            return { label: "🔄 HT Demucs ➔ Mel-RoFormer (Cascade)", badgeClass: "badge-cascade" };
        case "mel_then_ht":
            return { label: "🔁 Mel-RoFormer ➔ HT Demucs (Cascade)", badgeClass: "badge-cascade" };
        default:
            return { label: modelKey, badgeClass: "badge-ht" };
    }
}

async function loadSeparationModelStatus() {
    if (!htdemucsStatus || !melRoformerStatus) return;
    try {
        const res = await fetch("/api/separation/models");
        if (!res.ok) throw new Error("Không thể kiểm tra trạng thái model.");
        const data = await res.json();
        const models = data.models || {};
        
        const ht = models.htdemucs;
        const mel = models.mel_roformer;
        const df = models.deepfilternet;

        if (ht) {
            htdemucsStatus.textContent = ht.available ? "Sẵn sàng (CUDA)" : ht.message;
            htdemucsStatus.style.color = ht.available ? "var(--accent-emerald)" : "var(--accent-amber)";
        }
        if (mel) {
            melRoformerStatus.textContent = mel.available ? "Sẵn sàng (SOTA)" : mel.message;
            melRoformerStatus.style.color = mel.available ? "var(--accent-cyan)" : "var(--accent-amber)";
        }
        if (deepfilternetStatus && df) {
            deepfilternetStatus.textContent = df.available ? "Sẵn sàng (Denoise 48kHz)" : df.message;
            deepfilternetStatus.style.color = df.available ? "#ec4899" : "var(--accent-amber)";
        }
        if (htThenMelStatus && models.ht_then_mel) {
            htThenMelStatus.textContent = models.ht_then_mel.available ? "Sẵn sàng (Tối ưu Cache)" : models.ht_then_mel.message;
            htThenMelStatus.style.color = models.ht_then_mel.available ? "#fbbf24" : "var(--accent-amber)";
        }
        if (melThenHtStatus && models.mel_then_ht) {
            melThenHtStatus.textContent = models.mel_then_ht.available ? "Sẵn sàng (Tối ưu Cache)" : models.mel_then_ht.message;
            melThenHtStatus.style.color = models.mel_then_ht.available ? "#fbbf24" : "var(--accent-amber)";
        }
    } catch (err) {
        htdemucsStatus.textContent = "Không thể kiểm tra";
        melRoformerStatus.textContent = "Không thể kiểm tra";
    }
}

async function loadSeparationHistory() {
    if (!separationHistoryContainer) return;
    try {
        const res = await fetch("/api/separation/history");
        if (!res.ok) throw new Error("Không thể tải lịch sử tách nguồn.");
        const data = await res.json();
        renderSeparationHistory(data.history || []);
    } catch (err) {
        console.error(err);
    }
}

function renderSeparationHistory(history) {
    if (!separationHistoryContainer) return;
    separationHistoryContainer.innerHTML = "";

    if (history.length === 0) {
        separationHistoryContainer.innerHTML = `
            <div class="empty-state" style="padding: 24px 0;">
                <p style="color: var(--text-dim);">Chưa có bản ghi tách nguồn nào trong <code>processed_audio/</code>.</p>
            </div>
        `;
        return;
    }

    history.forEach(item => {
        const card = document.createElement("div");
        card.className = "history-item-card";

        const { label: modelLabel, badgeClass } = getModelDisplay(item.model);
        const createdFormatted = item.created_at ? new Date(item.created_at).toLocaleString("vi-VN") : "--";

        let stemsHtml = "";
        (item.stems || []).forEach(stem => {
            stemsHtml += `
                <button class="btn-secondary" style="font-size: 0.8rem; padding: 4px 8px;" onclick="playProcessedAudio('${stem.url}', '${item.model} • Vocal', '${item.input_filename || item.run_id}', '${stem.filename}')">
                    ▶ <span class="stem-badge-vocal">VOCAL</span>
                </button>
                <button class="btn-stage5-handover" style="font-size: 0.74rem; padding: 4px 8px;" onclick="useVocalForDiarization('${item.model}', '${item.run_id}', '${item.input_filename || item.run_id}')" title="Phân đoạn người nói với Diarization (Stage 4)">
                    👥 Diarization
                </button>
            `;
        });

        card.innerHTML = `
            <div class="history-left">
                <div class="history-info-title"><span class="model-badge ${badgeClass}" style="font-size: 0.75rem;">${modelLabel}</span> • <span style="font-family: var(--font-mono); color: var(--accent-cyan);">${item.run_id}</span></div>
                <div class="history-info-meta">
                    <span>File gốc: <strong>${item.input_filename || "Audio"}</strong></span>
                    ${item.cached_pass1_run_id ? `<span style="color: #fbbf24;">• ⚡ Dùng Pass 1 từ Run: ${item.cached_pass1_run_id}</span>` : ''}
                    ${item.atten_lim_db !== undefined && item.atten_lim_db !== null ? `<span style="color: #ec4899;">• 🛡️ Lọc: ${item.atten_lim_db >= 100 ? 'Max 100 dB' : item.atten_lim_db + ' dB'}${item.post_filter ? ' + PostFilter' : ''}</span>` : ''}
                    <span>• ${createdFormatted}</span>
                </div>
            </div>
            <div class="history-actions">
                <div style="display: flex; gap: 6px;">${stemsHtml}</div>
                <button class="btn-delete-history" onclick="deleteSeparationRun('${item.model}', '${item.run_id}')" title="Xóa bản ghi này">
                    <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="3 6 5 6 21 6"></polyline>
                        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                    </svg>
                </button>
            </div>
        `;

        separationHistoryContainer.appendChild(card);
    });
}

async function deleteSeparationRun(model, runId) {
    if (!confirm(`Bạn có chắc muốn xóa bản lưu tách nguồn ${runId} của model ${model}?`)) return;
    try {
        const res = await fetch(`/api/separation/${model}/${runId}`, { method: "DELETE" });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Không thể xóa bản ghi.");
        showToast(data.message || "Đã xóa bản ghi.", "success");
        loadSeparationHistory();
    } catch (err) {
        showToast(err.message, "error");
    }
}

function renderSeparationResults(results) {
    separationResults.innerHTML = "";

    results.forEach(result => {
        const card = document.createElement("section");
        card.className = "separation-result-card";

        const { label: modelLabel, badgeClass } = getModelDisplay(result.model);

        const dfConfigText = result.atten_lim_db !== undefined && result.atten_lim_db !== null
            ? `<span style="font-size: 0.78rem; color: #ec4899; margin-left: 8px;">🛡️ (Mức lọc: ${result.atten_lim_db >= 100 ? 'Max 100 dB' : result.atten_lim_db + ' dB'}${result.post_filter ? ' + PostFilter' : ''})</span>`
            : '';

        const header = document.createElement("div");
        header.className = "separation-result-header";
        header.innerHTML = `
            <div class="result-model-title">
                <span class="model-badge ${badgeClass}">${modelLabel}</span>
                <span>Kết Quả Tách Vocal</span>
                ${result.cached_pass1_run_id ? `<span style="font-size: 0.78rem; color: #fbbf24; margin-left: 8px;">⚡ (Tối ưu: Tái sử dụng Pass 1 từ Run ${result.cached_pass1_run_id})</span>` : ''}
                ${dfConfigText}
            </div>
            <div class="result-run-id">Run ID: ${result.run_id}</div>
        `;
        card.appendChild(header);

        (result.stems || []).forEach(stem => {
            const row = document.createElement("div");
            row.className = "separation-stem-row";

            const info = document.createElement("div");
            info.innerHTML = `
                <div class="separation-stem-name">
                    <span>🎙️</span>
                    <span class="stem-badge-vocal">VOCAL STEM</span>
                </div>
                <div class="separation-stem-meta">${stem.filename} • ${stem.filesize_formatted}</div>
            `;

            const actions = document.createElement("div");
            actions.className = "separation-actions";

            // Play Button
            const playBtn = document.createElement("button");
            playBtn.type = "button";
            playBtn.className = "btn-secondary";
            playBtn.innerHTML = `▶ Nghe Vocal`;
            playBtn.addEventListener("click", () => {
                playProcessedAudio(stem.url, `${result.model.toUpperCase()} • Vocal`, "Separated Vocal", stem.filename);
            });

            // Download Button
            const dlBtn = document.createElement("a");
            dlBtn.className = "btn-secondary";
            dlBtn.href = stem.download_url;
            dlBtn.download = stem.filename;
            dlBtn.innerHTML = `📥 Tải WAV`;

            // Stage 4 Diarization Handover button
            const stage4Btn = document.createElement("button");
            stage4Btn.type = "button";
            stage4Btn.className = "btn-stage3-action";
            stage4Btn.innerHTML = `🚀 Dùng Vocal cho Stage 4 (Diarization)`;
            stage4Btn.title = "Chuyển stem vocal này sang Stage 4 (Speaker Diarization)";
            stage4Btn.addEventListener("click", () => {
                useVocalForDiarization(result.model, result.run_id, result.input_filename || result.run_id);
            });

            actions.append(playBtn, dlBtn, stage4Btn);
            row.append(info, actions);
            card.appendChild(row);
        });

        separationResults.appendChild(card);
    });

    separationResults.classList.remove("hidden");
}

// Upload direct audio file
if (uploadAudioBtn && directAudioUpload) {
    uploadAudioBtn.addEventListener("click", () => {
        directAudioUpload.click();
    });

    directAudioUpload.addEventListener("change", async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append("file", file);

        uploadAudioBtn.disabled = true;
        uploadAudioBtn.innerHTML = `<span>Đang tải lên & chuyển đổi WAV...</span>`;

        try {
            const res = await fetch("/api/upload", {
                method: "POST",
                body: formData
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || "Tải lên tệp thất bại.");

            showToast(data.message || "Tải lên file thành công!", "success");
            await loadAudioLibrary();
            loadSeparationInputs(data.data?.filename);
            loadBenchmarkInputs(data.data?.filename);
        } catch (err) {
            console.error(err);
            showToast(err.message, "error");
        } finally {
            uploadAudioBtn.disabled = false;
            uploadAudioBtn.innerHTML = `
                <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                    <polyline points="17 8 12 3 7 8"></polyline>
                    <line x1="12" y1="3" x2="12" y2="15"></line>
                </svg>
                <span>Upload Tệp Audio</span>
            `;
            directAudioUpload.value = "";
        }
    });
}

// DeepFilterNet UI Helpers
window.updateDFAttenDisplay = function(val) {
    const valNum = parseInt(val, 10);
    const badge = document.getElementById("dfAttenVal");
    if (!badge) return;

    document.querySelectorAll(".df-preset-btn").forEach(btn => btn.classList.remove("active"));

    if (valNum >= 100) {
        badge.innerHTML = "⚡ Tối đa (Max 100 dB)";
        const maxBtn = document.querySelector(".df-preset-btn:last-child");
        if (maxBtn) maxBtn.classList.add("active");
    } else {
        badge.innerHTML = `${valNum} dB`;
        const matchingBtn = Array.from(document.querySelectorAll(".df-preset-btn")).find(b => b.textContent.includes(`${valNum} dB`));
        if (matchingBtn) matchingBtn.classList.add("active");
    }
};

window.setDFAttenPreset = function(val) {
    const slider = document.getElementById("dfAttenSlider");
    if (slider) {
        slider.value = val;
        window.updateDFAttenDisplay(val);
    }
};

// Toggle DeepFilterNet Config Box visibility based on model selection
const modelDfChk = document.getElementById("model_deepfilternet");
const dfCfgBox = document.getElementById("deepfilterConfigBox");
if (modelDfChk && dfCfgBox) {
    const syncDfVisibility = () => {
        dfCfgBox.style.display = modelDfChk.checked ? "block" : "none";
    };
    modelDfChk.addEventListener("change", syncDfVisibility);
    syncDfVisibility();
}

// Separation Form Submit
if (separationForm) {
    separationForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const filename = separationInputSelect.value;
        const models = [...document.querySelectorAll('input[name="separationModel"]:checked')]
            .map(input => input.value);

        if (!filename) {
            showToast("Hãy chọn audio đầu vào cần tách vocal.", "error");
            return;
        }
        if (models.length === 0) {
            showToast("Hãy chọn ít nhất một model tách âm.", "error");
            return;
        }

        const dfAttenInput = document.getElementById("dfAttenSlider");
        const dfPostFilterInput = document.getElementById("dfPostFilter");
        const deepfilternet_atten_lim_db = dfAttenInput ? parseFloat(dfAttenInput.value) : 100;
        const deepfilternet_post_filter = dfPostFilterInput ? dfPostFilterInput.checked : false;

        runSeparationBtn.disabled = true;
        runSeparationBtn.querySelector(".btn-text").classList.add("hidden");
        runSeparationBtn.querySelector(".btn-spinner").classList.remove("hidden");
        separationProgress.classList.remove("hidden");
        separationResults.classList.add("hidden");

        separationStartTime = Date.now();
        sepProgressTimer.textContent = "00:00";
        if (separationTimerInterval) clearInterval(separationTimerInterval);
        separationTimerInterval = setInterval(() => {
            const elapsed = Math.floor((Date.now() - separationStartTime) / 1000);
            sepProgressTimer.textContent = formatTime(elapsed);
        }, 1000);

        try {
            const res = await fetch("/api/separation", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    filename,
                    models,
                    deepfilternet_atten_lim_db,
                    deepfilternet_post_filter
                }),
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || "Không thể chạy source separation.");

            renderSeparationResults(data.results || []);
            (data.errors || []).forEach(err => showToast(`${err.model}: ${err.message}`, "error", 7000));
            showToast("Hoàn tất bóc tách nguồn âm thanh!", "success");
            loadSeparationHistory();
        } catch (err) {
            console.error(err);
            showToast(err.message, "error", 7000);
        } finally {
            clearInterval(separationTimerInterval);
            runSeparationBtn.disabled = false;
            runSeparationBtn.querySelector(".btn-text").classList.remove("hidden");
            runSeparationBtn.querySelector(".btn-spinner").classList.add("hidden");
            separationProgress.classList.add("hidden");
        }
    });
}

if (refreshHistoryBtn) {
    refreshHistoryBtn.addEventListener("click", () => {
        loadSeparationHistory();
        showToast("Đã làm mới lịch sử tách nguồn.", "success", 2000);
    });
}

// ==========================================================================
// Stage 3: Separation Benchmark Functions
// ==========================================================================
let benchmarkSourcesList = [];
const benchmarkSourceMeta = document.getElementById("benchmarkSourceMeta");

async function loadBenchmarkInputs(selectFilename = null) {
    if (!benchmarkInputSelect) return;
    benchmarkInputSelect.innerHTML = "";

    try {
        const res = await fetch("/api/benchmark/sources");
        if (!res.ok) throw new Error("Không thể tải danh sách bản tách nguồn.");
        const data = await res.json();
        benchmarkSourcesList = data.sources || [];

        if (benchmarkSourcesList.length === 0) {
            benchmarkInputSelect.add(new Option("Chưa có file nào được tách vocal từ Tab 2 (hãy sang Tab 2 tách trước)", ""));
            if (benchmarkSourceMeta) benchmarkSourceMeta.classList.add("hidden");
            return;
        }

        benchmarkInputSelect.add(new Option("Chọn audio đã tách vocal từ Tab 2 để chấm điểm…", ""));
        benchmarkSourcesList.forEach(item => {
            const isSelected = item.input_filename === selectFilename;
            const option = new Option(item.label, item.input_filename, false, isSelected);
            benchmarkInputSelect.add(option);
        });

        updateBenchmarkSourceMeta();
    } catch (err) {
        console.error(err);
        benchmarkInputSelect.add(new Option("Lỗi tải danh sách bản tách nguồn", ""));
    }
}

function updateBenchmarkSourceMeta() {
    if (!benchmarkSourceMeta || !benchmarkInputSelect) return;
    const selectedName = benchmarkInputSelect.value;
    const item = benchmarkSourcesList.find(s => s.input_filename === selectedName);

    if (!item) {
        benchmarkSourceMeta.classList.add("hidden");
        return;
    }

    const availableTags = [];
    if (item.latest_htdemucs_run_id) availableTags.push("⚡ HT Demucs");
    if (item.latest_mel_roformer_run_id) availableTags.push("💎 Mel-RoFormer");
    if (item.latest_deepfilternet_run_id) availableTags.push("🛡️ DeepFilterNet3");
    if (item.latest_ht_then_mel_run_id) availableTags.push("🔄 Cascade HT➔Mel");
    if (item.latest_mel_then_ht_run_id) availableTags.push("🔁 Cascade Mel➔HT");

    let statusText = `✅ <strong>Sẵn sàng so sánh đối đầu (${availableTags.length} nguồn):</strong> ${availableTags.join(" • ")}`;
    benchmarkSourceMeta.innerHTML = statusText;
    benchmarkSourceMeta.classList.remove("hidden");
}

if (benchmarkInputSelect) {
    benchmarkInputSelect.addEventListener("change", updateBenchmarkSourceMeta);
}

async function loadBenchmarkHistory() {
    if (!benchmarkHistoryContainer) return;
    try {
        const res = await fetch("/api/benchmark/history");
        if (!res.ok) throw new Error("Không thể tải lịch sử benchmark.");
        const data = await res.json();
        renderBenchmarkHistory(data.history || []);
    } catch (err) {
        console.error(err);
    }
}

let activeBenchmarkCharts = [];
let visibleMetrics = {
    sim: true,
    sig: true,
    bak: true
};

window.toggleBenchmarkMetric = function(metricKey, isVisible) {
    visibleMetrics[metricKey] = isVisible;
    const indexMap = { sim: 0, sig: 1, bak: 2 };
    const dsIndex = indexMap[metricKey];

    activeBenchmarkCharts.forEach(chart => {
        if (typeof dsIndex === "number") {
            chart.setDatasetVisibility(dsIndex, isVisible);
            if (chart.options.scales?.ySim) {
                chart.options.scales.ySim.display = visibleMetrics.sim;
            }
            if (chart.options.scales?.yMos) {
                chart.options.scales.yMos.display = (visibleMetrics.sig || visibleMetrics.bak);
            }
            chart.update();
        }
    });
};

function destroyActiveCharts() {
    activeBenchmarkCharts.forEach(c => {
        try { c.destroy(); } catch (e) { console.warn("Chart destroy error:", e); }
    });
    activeBenchmarkCharts = [];
}

function renderBenchmarkHistory(history) {
    if (!benchmarkHistoryContainer) return;
    benchmarkHistoryContainer.innerHTML = "";

    if (history.length === 0) {
        benchmarkHistoryContainer.innerHTML = `
            <div class="empty-state" style="padding: 24px 0;">
                <p style="color: var(--text-dim);">Chưa có bản ghi so sánh benchmark nào trong <code>benchmark_results/</code>.</p>
            </div>
        `;
        return;
    }

    history.forEach(item => {
        const card = document.createElement("div");
        card.className = "history-item-card";

        const models = item.models || {};
        const createdFormatted = item.created_at ? new Date(item.created_at).toLocaleString("vi-VN") : "--";

        let modelsSummaryHtml = "";
        Object.entries(models).forEach(([key, m]) => {
            const stats = m.stats || {};
            const sim = stats.avg_similarity || m.speaker_similarity?.similarity_percent || "--";
            const sig = stats.avg_sig || m.dnsmos?.sig || "--";
            const bak = stats.avg_bak || m.dnsmos?.bak || "--";
            const badgeClass = m.badge_class || (key.includes("ht") ? "badge-ht" : "badge-roformer");
            modelsSummaryHtml += `
                <span style="margin-right: 12px; font-size: 0.76rem;">
                    <span class="model-badge ${badgeClass}" style="font-size: 0.7rem; padding: 2px 6px;">${m.badge || m.name}</span> 
                    Sim: <strong>${sim}%</strong> | SIG: <strong>${sig}</strong> | BAK: <strong>${bak}</strong>
                </span>
            `;
        });

        card.innerHTML = `
            <div class="history-left">
                <div class="history-info-title">
                    <span>📈 Biểu Đồ Chuỗi Thời Gian (${Object.keys(models).length} nguồn): <strong>${item.input_filename}</strong> (${item.eval_duration || item.clip_duration}s)</span>
                </div>
                <div class="history-info-meta" style="display: flex; flex-wrap: wrap; gap: 4px; align-items: center; margin-top: 4px;">
                    ${modelsSummaryHtml}
                    <span style="color: var(--text-dim);">• ${createdFormatted}</span>
                </div>
            </div>
            <div class="history-actions">
                <button class="btn-secondary" onclick='renderBenchmarkResults(${JSON.stringify(item)})' title="Xem biểu đồ">
                    📊 Xem Biểu Đồ
                </button>
                <button class="btn-delete-history" onclick="deleteBenchmarkRun('${item.benchmark_id}')" title="Xóa benchmark này">
                    <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="3 6 5 6 21 6"></polyline>
                        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                    </svg>
                </button>
            </div>
        `;
        benchmarkHistoryContainer.appendChild(card);
    });
}

async function deleteBenchmarkRun(benchmarkId) {
    if (!confirm(`Bạn có chắc chắn muốn xóa bản ghi benchmark [${benchmarkId}] này không?`)) {
        return;
    }
    try {
        const res = await fetch(`/api/benchmark/${benchmarkId}`, {
            method: "DELETE"
        });
        const data = await res.json();
        if (!res.ok) {
            throw new Error(data.detail || "Không thể xóa bản ghi benchmark.");
        }
        showToast(data.message || "Đã xóa bản ghi benchmark thành công.", "success", 3000);
        if (benchmarkResults && !benchmarkResults.classList.contains("hidden")) {
            destroyActiveCharts();
            benchmarkResults.classList.add("hidden");
            benchmarkResults.innerHTML = "";
        }
        loadBenchmarkHistory();
    } catch (err) {
        showToast(err.message, "error", 5000);
    }
}

window.deleteBenchmarkRun = deleteBenchmarkRun;

let currentlyPlayingModelKey = null;

function updateChartPlayButtons() {
    document.querySelectorAll(".chart-play-btn").forEach(btn => {
        const key = btn.dataset.modelKey;
        const card = btn.closest(".benchmark-chart-card");
        if (key === currentlyPlayingModelKey && isAudioPlaying) {
            btn.innerHTML = `
                <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
                    <rect x="6" y="4" width="4" height="16"></rect>
                    <rect x="14" y="4" width="4" height="16"></rect>
                </svg>
            `;
            btn.classList.add("playing");
            if (card) card.classList.add("is-playing");
        } else {
            btn.innerHTML = `
                <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
                    <polygon points="5 3 19 12 5 21 5 3"></polygon>
                </svg>
            `;
            btn.classList.remove("playing");
            if (card) card.classList.remove("is-playing");
        }
    });
}

function togglePlayModelVocal(key, vocalUrl, modelName, originalFilename) {
    if (currentlyPlayingModelKey === key && isAudioPlaying) {
        globalAudio.pause();
        return;
    }

    currentlyPlayingModelKey = key;
    playProcessedAudio(vocalUrl, `${modelName} • Vocal`, originalFilename, `${key}_vocals.wav`);
    updateChartPlayButtons();
    activeBenchmarkCharts.forEach(c => c.draw());
}

function renderBenchmarkResults(data) {
    if (!benchmarkResults) return;
    destroyActiveCharts();
    benchmarkResults.innerHTML = "";

    const models = data.models || {};
    const modelKeys = Object.keys(models);
    const timeline = data.timeline || { labels: [], timestamps: [] };
    const labels = timeline.labels || [];
    const timestamps = timeline.timestamps || [];
    const clipDuration = data.clip_duration || 0;

    // 1. Meta Bar
    const metaBar = document.createElement("div");
    metaBar.className = "benchmark-meta-bar";
    metaBar.innerHTML = `
        <div>
            <span>File thử nghiệm: <strong>${data.input_filename}</strong></span>
            <span style="margin-left: 12px; color: var(--text-dim);">Độ dài: <strong>${formatTime(clipDuration)} (${clipDuration}s)</strong> • Đang so sánh: <strong>${modelKeys.length} nguồn Vocal</strong></span>
        </div>
        <div style="font-family: var(--font-mono); color: var(--accent-cyan);">Benchmark ID: ${data.benchmark_id}</div>
    `;
    benchmarkResults.appendChild(metaBar);

    // 2. Multi-Way Player Reference Bar
    const playerCard = document.createElement("div");
    playerCard.className = "three-way-player-card";
    
    let channelsHtml = `
        <div class="three-way-channel">
            <span class="channel-name">1. Audio Gốc (Mix)</span>
            <button class="btn-secondary" onclick="playProcessedAudio('${data.reference_audio_url}', 'Reference Audio (Mix)', '${data.input_filename}', 'reference.wav')">
                ▶ Nghe Mix Gốc
            </button>
        </div>
    `;

    let channelIdx = 2;
    Object.entries(models).forEach(([key, m]) => {
        channelsHtml += `
            <div class="three-way-channel">
                <span class="channel-name">${channelIdx++}. ${m.badge || m.name}</span>
                <button class="btn-secondary" onclick="togglePlayModelVocal('${key}', '${m.vocal_url}', '${m.name}', '${data.input_filename}')">
                    ▶ Phát Vocal
                </button>
            </div>
        `;
    });

    playerCard.innerHTML = `
        <div class="three-way-title">🎧 Danh Sách Nhanh Các Kênh Âm Thanh (${modelKeys.length + 1} Nguồn)</div>
        <div class="three-way-grid" style="grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));">${channelsHtml}</div>
    `;
    benchmarkResults.appendChild(playerCard);

    // 2.5 Metrics Display Filter Controls (Toggle checkboxes)
    const filterCard = document.createElement("div");
    filterCard.className = "metrics-filter-card";
    filterCard.innerHTML = `
        <div class="filter-header-left">
            <span class="filter-icon">📊</span>
            <span class="filter-title">Tùy Chọn Thông Số Hiển Thị:</span>
            <span class="filter-hint">(Bật/tắt đường biểu đồ trên tất cả các nguồn output)</span>
        </div>
        <div class="filter-options-group">
            <label class="metric-toggle-label metric-sim">
                <input type="checkbox" id="chkMetricSim" ${visibleMetrics.sim ? "checked" : ""} onchange="toggleBenchmarkMetric('sim', this.checked)">
                <span>🗣️ Speaker Similarity (%)</span>
            </label>
            <label class="metric-toggle-label metric-sig">
                <input type="checkbox" id="chkMetricSig" ${visibleMetrics.sig ? "checked" : ""} onchange="toggleBenchmarkMetric('sig', this.checked)">
                <span>🎙️ DNSMOS SIG (Speech Quality)</span>
            </label>
            <label class="metric-toggle-label metric-bak">
                <input type="checkbox" id="chkMetricBak" ${visibleMetrics.bak ? "checked" : ""} onchange="toggleBenchmarkMetric('bak', this.checked)">
                <span>🛡️ DNSMOS BAK (Noise Suppression)</span>
            </label>
        </div>
    `;
    benchmarkResults.appendChild(filterCard);

    // 3. One Integrated Player-Chart Card Per Model Output
    const chartsWrapper = document.createElement("div");
    chartsWrapper.className = "benchmark-charts-wrapper";

    Object.entries(models).forEach(([key, m]) => {
        const stats = m.stats || {};
        const badgeClass = m.badge_class || (key.includes("ht") ? "badge-ht" : "badge-roformer");
        const canvasId = `chart_${key}`;

        const avgSim = parseFloat(stats.avg_similarity) || 0;
        const avgSig = parseFloat(stats.avg_sig) || 0;
        const avgBak = parseFloat(stats.avg_bak) || 0;

        const simColor = avgSim < 90 ? "#f43f5e" : "#10b981";
        let sigColor = "#38bdf8";
        if (avgSig < 3.0) sigColor = "#f43f5e";
        else if (avgSig < 4.2) sigColor = "#fbbf24";

        let bakColor = "#10b981";
        if (avgBak < 3.5) bakColor = "#f43f5e";
        else if (avgBak < 4.0) bakColor = "#fbbf24";

        const modelChartCard = document.createElement("div");
        modelChartCard.className = "benchmark-chart-card";
        modelChartCard.id = `chartCard_${key}`;
        modelChartCard.innerHTML = `
            <div class="chart-card-header">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <button class="chart-play-btn" id="chartPlayBtn_${key}" data-model-key="${key}" title="Phát / Dừng kênh vocal này" onclick="togglePlayModelVocal('${key}', '${m.vocal_url}', '${m.name}', '${data.input_filename}')">
                        <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
                            <polygon points="5 3 19 12 5 21 5 3"></polygon>
                        </svg>
                    </button>
                    <div class="chart-title-area">
                        <div class="chart-title">
                            <span class="model-badge ${badgeClass}" style="font-size: 0.85rem; padding: 4px 10px;">${m.badge || m.name}</span>
                            <span>Thanh Phát Âm Thanh & Biểu Đồ Chất Lượng</span>
                        </div>
                        <div class="chart-desc">Vạch phát nhạc đồng bộ thời gian thực từ 00:00 ➔ ${labels[labels.length - 1] || formatTime(clipDuration)}</div>
                    </div>
                </div>

                <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
                    <span id="chartTimer_${key}" class="chart-playback-timer">00:00 / ${formatTime(clipDuration)}</span>
                    <div class="chart-summary-pill">
                        <span class="dot" style="background: ${simColor};"></span>
                        <span>Sim:</span>
                        <strong style="color: ${simColor};">${stats.avg_similarity || '--'}%</strong>
                    </div>
                    <div class="chart-summary-pill">
                        <span class="dot" style="background: ${sigColor};"></span>
                        <span>SIG:</span>
                        <strong style="color: ${sigColor};">${stats.avg_sig || '--'}</strong>
                    </div>
                    <div class="chart-summary-pill">
                        <span class="dot" style="background: ${bakColor};"></span>
                        <span>BAK:</span>
                        <strong style="color: ${bakColor};">${stats.avg_bak || '--'}</strong>
                    </div>
                </div>
            </div>
            
            <div class="chart-canvas-box" id="canvasBox_${key}" style="height: 290px;">
                <canvas id="${canvasId}"></canvas>
            </div>

            <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.75rem; color: var(--text-muted); padding-top: 4px; flex-wrap: wrap; gap: 8px;">
                <span>🎨 <strong>Quy chuẩn đổi màu:</strong> <span style="color: #f43f5e; font-weight: 600;">■ Đỏ (Kém)</span>: Sim &lt;90% | SIG &lt;3.0 | BAK &lt;3.5 • <span style="color: #fbbf24; font-weight: 600;">■ Vàng</span>: SIG 3.0-4.2 | BAK 3.5-4.0 • <span style="color: #10b981; font-weight: 600;">■ Xanh (Tốt)</span></span>
                <span>🎯 <em>Nhấp/kéo trên biểu đồ để nhảy phát nhạc</em></span>
            </div>
        `;

        chartsWrapper.appendChild(modelChartCard);
    });

    benchmarkResults.appendChild(chartsWrapper);
    benchmarkResults.classList.remove("hidden");

    // Render Chart.js instances with Dynamic Segment Threshold Colors & Live Playhead Plugin
    if (typeof Chart !== "undefined") {
        Object.entries(models).forEach(([key, m]) => {
            const canvasId = `chart_${key}`;
            const canvasEl = document.getElementById(canvasId);
            if (!canvasEl) return;

            const ts = m.time_series || {};
            const ctx = canvasEl.getContext("2d");

            const datasets = [
                // 1. Speaker Similarity (%) - Right Y-Axis: < 90% -> Red (#f43f5e), >= 90% -> Green (#10b981)
                {
                    label: '🗣️ Speaker Similarity (%)',
                    data: ts.similarity || [],
                    yAxisID: 'ySim',
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.08)',
                    fill: true,
                    tension: 0.35,
                    borderWidth: 2.5,
                    pointRadius: 0,
                    pointHoverRadius: 6,
                    pointHitRadius: 12,
                    segment: {
                        borderColor: (ctx) => {
                            const y0 = ctx.p0?.parsed?.y;
                            const y1 = ctx.p1?.parsed?.y;
                            if (y0 !== undefined && y1 !== undefined) {
                                return (y0 < 90 || y1 < 90) ? '#f43f5e' : '#10b981';
                            }
                            return '#10b981';
                        },
                        backgroundColor: (ctx) => {
                            const y0 = ctx.p0?.parsed?.y;
                            const y1 = ctx.p1?.parsed?.y;
                            if (y0 !== undefined && y1 !== undefined) {
                                return (y0 < 90 || y1 < 90) ? 'rgba(244, 63, 94, 0.08)' : 'rgba(16, 185, 129, 0.08)';
                            }
                            return 'rgba(16, 185, 129, 0.08)';
                        }
                    }
                },
                // 2. DNSMOS SIG (Speech Quality 1.0 - 5.0) - Left Y-Axis: < 3.0 -> Red (#f43f5e), 3.0 - 4.2 -> Yellow (#fbbf24), >= 4.2 -> Cyan (#38bdf8)
                {
                    label: '🎙️ DNSMOS SIG (Speech Quality)',
                    data: ts.sig || [],
                    yAxisID: 'yMos',
                    borderColor: '#38bdf8',
                    backgroundColor: 'transparent',
                    tension: 0.35,
                    borderWidth: 2.5,
                    pointRadius: 0,
                    pointHoverRadius: 6,
                    pointHitRadius: 12,
                    segment: {
                        borderColor: (ctx) => {
                            const y0 = ctx.p0?.parsed?.y;
                            const y1 = ctx.p1?.parsed?.y;
                            if (y0 !== undefined && y1 !== undefined) {
                                const minVal = Math.min(y0, y1);
                                if (minVal < 3.0) return '#f43f5e';
                                if (minVal < 4.2) return '#fbbf24';
                                return '#38bdf8';
                            }
                            return '#38bdf8';
                        }
                    }
                },
                // 3. DNSMOS BAK (Background Suppression 1.0 - 5.0) - Left Y-Axis: < 3.5 -> Red (#f43f5e), < 4.0 -> Yellow (#fbbf24), >= 4.0 -> Green (#10b981)
                {
                    label: '🛡️ DNSMOS BAK (Noise Suppression)',
                    data: ts.bak || [],
                    yAxisID: 'yMos',
                    borderColor: '#fbbf24',
                    backgroundColor: 'transparent',
                    tension: 0.35,
                    borderWidth: 2.5,
                    pointRadius: 0,
                    pointHoverRadius: 6,
                    pointHitRadius: 12,
                    segment: {
                        borderColor: (ctx) => {
                            const y0 = ctx.p0?.parsed?.y;
                            const y1 = ctx.p1?.parsed?.y;
                            if (y0 !== undefined && y1 !== undefined) {
                                const minVal = Math.min(y0, y1);
                                if (minVal < 3.5) return '#f43f5e';
                                if (minVal < 4.0) return '#fbbf24';
                                return '#10b981';
                            }
                            return '#fbbf24';
                        }
                    }
                }
            ];

            // Custom Playhead Plugin for Live Audio Synchronization
            const livePlayheadPlugin = {
                id: `playhead_${key}`,
                afterDraw: (chart) => {
                    if (currentlyPlayingModelKey === key && globalAudio && !isNaN(globalAudio.currentTime)) {
                        const chartArea = chart.chartArea;
                        if (!chartArea) return;

                        const cur = globalAudio.currentTime;
                        const dur = globalAudio.duration || clipDuration || 1;
                        const progress = Math.max(0, Math.min(1, cur / dur));
                        const x = chartArea.left + progress * (chartArea.right - chartArea.left);

                        const c = chart.ctx;
                        c.save();

                        // 1. Shaded playback progress tint
                        c.fillStyle = 'rgba(56, 189, 248, 0.08)';
                        c.fillRect(chartArea.left, chartArea.top, x - chartArea.left, chartArea.bottom - chartArea.top);

                        // 2. Glowing Neon Playhead Line
                        c.beginPath();
                        c.moveTo(x, chartArea.top);
                        c.lineTo(x, chartArea.bottom);
                        c.lineWidth = 2.5;
                        c.strokeStyle = '#38bdf8';
                        c.shadowColor = '#38bdf8';
                        c.shadowBlur = 10;
                        c.stroke();

                        // 3. Playhead Knob at the top
                        c.beginPath();
                        c.arc(x, chartArea.top + 3, 5, 0, 2 * Math.PI);
                        c.fillStyle = '#38bdf8';
                        c.shadowBlur = 12;
                        c.fill();

                        c.restore();
                    }
                }
            };

            const chartInstance = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: datasets
                },
                plugins: [livePlayheadPlugin],
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        mode: 'index',
                        intersect: false,
                    },
                    plugins: {
                        legend: {
                            position: 'top',
                            labels: {
                                color: '#94a3b8',
                                font: { family: "'Outfit', sans-serif", size: 12, weight: '500' },
                                usePointStyle: true,
                                pointStyle: 'circle',
                                padding: 16,
                            }
                        },
                        tooltip: {
                            backgroundColor: 'rgba(15, 23, 42, 0.96)',
                            titleColor: '#38bdf8',
                            bodyColor: '#f1f5f9',
                            borderColor: 'rgba(255, 255, 255, 0.18)',
                            borderWidth: 1,
                            padding: 12,
                            boxPadding: 6,
                            usePointStyle: true,
                            titleFont: { family: "'JetBrains Mono', monospace", size: 13, weight: '600' },
                            bodyFont: { family: "'Outfit', sans-serif", size: 12 },
                            callbacks: {
                                title: function(context) {
                                    if (!context || !context.length) return '';
                                    return `⏱️ Mốc thời gian: ${context[0].label}`;
                                },
                                label: function(context) {
                                    const val = context.parsed.y;
                                    if (context.datasetIndex === 0 || context.dataset.yAxisID === 'ySim') {
                                        return ` 🗣️ Speaker Similarity: ${val}%`;
                                    }
                                    if (context.datasetIndex === 1) {
                                        return ` 🎙️ DNSMOS SIG: ${val.toFixed(2)} / 5.0`;
                                    }
                                    if (context.datasetIndex === 2) {
                                        return ` 🛡️ DNSMOS BAK: ${val.toFixed(2)} / 5.0`;
                                    }
                                    return ` ${context.dataset.label}: ${val}`;
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            grid: { color: 'rgba(255, 255, 255, 0.05)' },
                            ticks: {
                                color: '#64748b',
                                font: { family: "'JetBrains Mono', monospace", size: 11 },
                                maxRotation: 0,
                                autoSkip: true,
                                maxTicksLimit: 14
                            }
                        },
                        yMos: {
                            type: 'linear',
                            position: 'left',
                            min: 1.0,
                            max: 5.0,
                            grid: { color: 'rgba(255, 255, 255, 0.05)' },
                            title: {
                                display: true,
                                text: 'DNSMOS MOS (1.0 - 5.0)',
                                color: '#94a3b8',
                                font: { family: "'Outfit', sans-serif", size: 11, weight: '600' }
                            },
                            ticks: {
                                color: '#94a3b8',
                                font: { family: "'JetBrains Mono', monospace", size: 11 },
                                stepSize: 0.5,
                                callback: val => `${val.toFixed(1)}`
                            }
                        },
                        ySim: {
                            type: 'linear',
                            position: 'right',
                            min: 0,
                            max: 100,
                            grid: { drawOnChartArea: false },
                            title: {
                                display: true,
                                text: 'Speaker Similarity (%)',
                                color: '#10b981',
                                font: { family: "'Outfit', sans-serif", size: 11, weight: '600' }
                            },
                            ticks: {
                                color: '#10b981',
                                font: { family: "'JetBrains Mono', monospace", size: 11 },
                                callback: val => `${val}%`
                            }
                        }
                    },
                    onClick: (e) => {
                        const chartArea = chartInstance.chartArea;
                        if (!chartArea) return;
                        const canvasPos = Chart.helpers?.getRelativePosition ? Chart.helpers.getRelativePosition(e, chartInstance) : { x: e.offsetX, y: e.offsetY };
                        const clickX = canvasPos.x;
                        if (clickX >= chartArea.left && clickX <= chartArea.right) {
                            const frac = (clickX - chartArea.left) / (chartArea.right - chartArea.left);
                            const dur = globalAudio.duration || clipDuration || 1;
                            const targetSec = Math.max(0, Math.min(dur, frac * dur));

                            if (currentlyPlayingModelKey !== key) {
                                togglePlayModelVocal(key, m.vocal_url, m.name, data.input_filename);
                            }
                            globalAudio.currentTime = targetSec;
                            if (globalAudio.paused) globalAudio.play();
                            showToast(`Đã nhảy tới ${formatTime(targetSec)} (${m.name})`, "info", 1500);
                        }
                    }
                }
            });

            // Apply initial metric visibility from checkbox filter state
            chartInstance.setDatasetVisibility(0, visibleMetrics.sim);
            chartInstance.setDatasetVisibility(1, visibleMetrics.sig);
            chartInstance.setDatasetVisibility(2, visibleMetrics.bak);
            if (chartInstance.options.scales?.ySim) {
                chartInstance.options.scales.ySim.display = visibleMetrics.sim;
            }
            if (chartInstance.options.scales?.yMos) {
                chartInstance.options.scales.yMos.display = (visibleMetrics.sig || visibleMetrics.bak);
            }
            chartInstance.update('none');

            activeBenchmarkCharts.push(chartInstance);
        });
    }
}

// Benchmark Form Submit
if (benchmarkForm) {
    benchmarkForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const filename = benchmarkInputSelect.value;
        const selectedSource = benchmarkSourcesList.find(s => s.input_filename === filename);

        if (!filename || !selectedSource) {
            showToast("Hãy chọn một bản audio đã bóc tách từ Tab 2.", "error");
            return;
        }

        runBenchmarkBtn.disabled = true;
        runBenchmarkBtn.querySelector(".btn-text").classList.add("hidden");
        runBenchmarkBtn.querySelector(".btn-spinner").classList.remove("hidden");
        benchmarkProgress.classList.remove("hidden");
        benchmarkResults.classList.add("hidden");

        benchmarkStartTime = Date.now();
        benchProgressTimer.textContent = "00:00";
        if (benchmarkTimerInterval) clearInterval(benchmarkTimerInterval);
        benchmarkTimerInterval = setInterval(() => {
            const elapsed = Math.floor((Date.now() - benchmarkStartTime) / 1000);
            benchProgressTimer.textContent = formatTime(elapsed);
        }, 1000);

        try {
            const res = await fetch("/api/benchmark/evaluate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    input_filename: selectedSource.input_filename,
                    htdemucs_run_id: selectedSource.latest_htdemucs_run_id,
                    mel_roformer_run_id: selectedSource.latest_mel_roformer_run_id,
                    deepfilternet_run_id: selectedSource.latest_deepfilternet_run_id,
                    ht_then_mel_run_id: selectedSource.latest_ht_then_mel_run_id,
                    mel_then_ht_run_id: selectedSource.latest_mel_then_ht_run_id,
                }),
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || "Không thể chấm điểm benchmark.");

            renderBenchmarkResults(data.data);
            showToast("Chấm điểm benchmark hoàn tất!", "success");
            loadBenchmarkHistory();
        } catch (err) {
            console.error(err);
            showToast(err.message, "error", 7000);
        } finally {
            clearInterval(benchmarkTimerInterval);
            runBenchmarkBtn.disabled = false;
            runBenchmarkBtn.querySelector(".btn-text").classList.remove("hidden");
            runBenchmarkBtn.querySelector(".btn-spinner").classList.add("hidden");
            benchmarkProgress.classList.add("hidden");
        }
    });
}

if (refreshBenchmarkHistoryBtn) {
    refreshBenchmarkHistoryBtn.addEventListener("click", () => {
        loadBenchmarkInputs();
        loadBenchmarkHistory();
        showToast("Đã làm mới danh sách bản tách & lịch sử benchmark.", "success", 2000);
    });
}

// ==========================================================================
// Global Player Logic
// ==========================================================================

function playAudio(filename) {
    const item = audioList.find(a => a.filename === filename);
    if (!item) return;

    if (currentlyPlaying && currentlyPlaying.filename === filename) {
        if (isAudioPlaying) {
            globalAudio.pause();
        } else {
            globalAudio.play();
        }
        return;
    }

    currentlyPlaying = item;
    globalAudio.src = `/api/audio/${filename}`;
    playerBar.classList.remove("hidden");

    playerThumb.src = item.thumbnail || "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='80' height='50' viewBox='0 0 80 50'%3E%3Crect width='80' height='50' fill='%231e293b'/%3E%3C/svg%3E";
    playerTitle.textContent = item.title;
    playerUploader.textContent = item.uploader || "YouTube";
    playerDownloadBtn.href = `/api/audio/${filename}`;
    playerDownloadBtn.download = item.filename;

    globalAudio.play().catch(e => console.log("Play interrupted:", e));
}

function playProcessedAudio(url, title, subtitle, filename) {
    currentlyPlaying = { filename: url, title, uploader: subtitle, duration: 0 };
    globalAudio.src = url;
    playerBar.classList.remove("hidden");
    playerThumb.src = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='80' height='50' viewBox='0 0 80 50'%3E%3Crect width='80' height='50' fill='%236366f1'/%3E%3Ctext x='50%25' y='50%25' dominant-baseline='middle' text-anchor='middle' fill='%23ffffff' font-size='11'%3ESTEM%3C/text%3E%3C/svg%3E";
    playerTitle.textContent = title;
    playerUploader.textContent = subtitle;
    playerDownloadBtn.href = url;
    playerDownloadBtn.download = filename;
    globalAudio.play().catch(e => console.log("Play interrupted:", e));
}

globalAudio.addEventListener("play", () => {
    isAudioPlaying = true;
    playIcon.classList.add("hidden");
    pauseIcon.classList.remove("hidden");
    renderAudioList(audioList);
    updateChartPlayButtons();
    activeBenchmarkCharts.forEach(c => c.draw());
});

globalAudio.addEventListener("pause", () => {
    isAudioPlaying = false;
    playIcon.classList.remove("hidden");
    pauseIcon.classList.add("hidden");
    renderAudioList(audioList);
    updateChartPlayButtons();
    activeBenchmarkCharts.forEach(c => c.draw());
});

globalAudio.addEventListener("timeupdate", () => {
    const cur = globalAudio.currentTime;
    const dur = globalAudio.duration || (currentlyPlaying ? currentlyPlaying.duration : 0);
    playerCurrentTime.textContent = formatTime(cur);
    playerTotalTime.textContent = formatTime(dur);

    if (dur > 0) {
        const pct = (cur / dur) * 100;
        scrubberFill.style.width = `${pct}%`;
        scrubberHandle.style.left = `${pct}%`;
    }

    if (currentlyPlayingModelKey) {
        const timerEl = document.getElementById(`chartTimer_${currentlyPlayingModelKey}`);
        if (timerEl) {
            timerEl.textContent = `${formatTime(cur)} / ${formatTime(dur)}`;
        }
        activeBenchmarkCharts.forEach(c => c.draw());
    }
});

globalAudio.addEventListener("ended", () => {
    isAudioPlaying = false;
    playIcon.classList.remove("hidden");
    pauseIcon.classList.add("hidden");
    scrubberFill.style.width = "0%";
    scrubberHandle.style.left = "0%";
    renderAudioList(audioList);
    currentlyPlayingModelKey = null;
    updateChartPlayButtons();
    activeBenchmarkCharts.forEach(c => c.draw());
});

playerPlayBtn.addEventListener("click", () => {
    if (!currentlyPlaying) return;
    if (isAudioPlaying) {
        globalAudio.pause();
    } else {
        globalAudio.play();
    }
});

playerSeekBack.addEventListener("click", () => {
    globalAudio.currentTime = Math.max(0, globalAudio.currentTime - 5);
});

playerSeekFwd.addEventListener("click", () => {
    globalAudio.currentTime = Math.min(globalAudio.duration || 99999, globalAudio.currentTime + 5);
});

scrubberBar.addEventListener("click", (e) => {
    const rect = scrubberBar.getBoundingClientRect();
    const pos = (e.clientX - rect.left) / rect.width;
    const dur = globalAudio.duration || (currentlyPlaying ? currentlyPlaying.duration : 0);
    if (dur > 0) {
        globalAudio.currentTime = pos * dur;
    }
});

playerVolume.addEventListener("input", (e) => {
    globalAudio.volume = parseFloat(e.target.value);
});

playerMuteBtn.addEventListener("click", () => {
    globalAudio.muted = !globalAudio.muted;
    playerMuteBtn.style.opacity = globalAudio.muted ? "0.4" : "1";
});

// Crawl Form Submit (Stage 1)
crawlForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const url = youtubeUrlInput.value.trim();
    if (!url) {
        showToast("Vui lòng nhập link YouTube.", "error");
        return;
    }

    const sampleRate = parseInt(sampleRateSelect.value, 10);
    const mono = channelSelect.value === "true";
    const cookiesFromBrowser = bypassSelect ? (bypassSelect.value.trim() || null) : null;

    submitCrawlBtn.disabled = true;
    submitCrawlBtn.querySelector(".btn-text").classList.add("hidden");
    submitCrawlBtn.querySelector(".btn-spinner").classList.remove("hidden");
    progressBox.classList.remove("hidden");

    try {
        const response = await fetch("/api/crawl", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                url: url,
                sample_rate: sampleRate,
                mono: mono,
                cookies_from_browser: cookiesFromBrowser
            })
        });

        const result = await response.json();
        if (!response.ok) {
            throw new Error(result.detail || "Đã xảy ra lỗi khi crawl video.");
        }

        showToast(result.message || "Tải và trích xuất WAV thành công!", "success");
        youtubeUrlInput.value = "";
        clearInputBtn.classList.add("hidden");

        await loadAudioLibrary();
        if (result.data && result.data.filename) {
            playAudio(result.data.filename);
        }
    } catch (err) {
        console.error(err);
        showToast(err.message, "error");
    } finally {
        submitCrawlBtn.disabled = false;
        submitCrawlBtn.querySelector(".btn-text").classList.remove("hidden");
        submitCrawlBtn.querySelector(".btn-spinner").classList.add("hidden");
        progressBox.classList.add("hidden");
    }
});

// Copy File Path
function copyFilePath(path) {
    navigator.clipboard.writeText(path).then(() => {
        showToast(`Đã sao chép đường dẫn: ${path}`, "success", 2000);
    }).catch(() => {
        showToast("Không thể sao chép đường dẫn.", "error");
    });
}

// Delete Audio (Stage 1)
async function deleteAudio(filename) {
    if (!confirm(`Bạn có chắc muốn xóa tệp ${filename}?`)) return;

    try {
        const res = await fetch(`/api/audio/${filename}`, { method: "DELETE" });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Xóa tệp thất bại.");

        showToast(data.message || "Đã xóa audio thành công.", "success");
        if (currentlyPlaying && currentlyPlaying.filename === filename) {
            globalAudio.pause();
            playerBar.classList.add("hidden");
            currentlyPlaying = null;
        }
        await loadAudioLibrary();
    } catch (err) {
        showToast(err.message, "error");
    }
}

// Paste & Clear Helpers
pasteBtn.addEventListener("click", async () => {
    try {
        const text = await navigator.clipboard.readText();
        if (text) {
            youtubeUrlInput.value = text;
            clearInputBtn.classList.remove("hidden");
            youtubeUrlInput.focus();
            showToast("Đã dán link từ Clipboard!", "success", 2000);
        }
    } catch (err) {
        showToast("Vui lòng dán thủ công bằng Ctrl+V", "error");
    }
});

youtubeUrlInput.addEventListener("input", () => {
    if (youtubeUrlInput.value.length > 0) {
        clearInputBtn.classList.remove("hidden");
    } else {
        clearInputBtn.classList.add("hidden");
    }
});

clearInputBtn.addEventListener("click", () => {
    youtubeUrlInput.value = "";
    clearInputBtn.classList.add("hidden");
    youtubeUrlInput.focus();
});

searchInput.addEventListener("input", () => {
    renderAudioList(audioList);
});

refreshBtn.addEventListener("click", () => {
    loadAudioLibrary();
    showToast("Đã làm mới danh sách audio.", "success", 2000);
});

// ==========================================================================
// STAGE 4: SPEAKER DIARIZATION (src/diarization)
// ==========================================================================

const diarizationForm = document.getElementById("diarizationForm");
const runDiarizationBtn = document.getElementById("runDiarizationBtn");
const diarizationProgress = document.getElementById("diarizationProgress");
const diarProgressTimer = document.getElementById("diarProgressTimer");
const diarizationResults = document.getElementById("diarizationResults");
const diarizationHistoryContainer = document.getElementById("diarizationHistoryContainer");
const refreshDiarHistoryBtn = document.getElementById("refreshDiarHistoryBtn");
const hfTokenInput = document.getElementById("hfTokenInput");

const SPEAKER_PALETTE = [
    { name: "Tím (Speaker 0)", hex: "#a855f7", bg: "rgba(168, 85, 247, 0.12)", border: "#a855f7" },
    { name: "Cyan (Speaker 1)", hex: "#38bdf8", bg: "rgba(56, 189, 248, 0.12)", border: "#38bdf8" },
    { name: "Cam (Speaker 2)", hex: "#f97316", bg: "rgba(249, 115, 22, 0.12)", border: "#f97316" },
    { name: "Hồng (Speaker 3)", hex: "#ec4899", bg: "rgba(236, 72, 153, 0.12)", border: "#ec4899" },
    { name: "Lục (Speaker 4)", hex: "#10b981", bg: "rgba(16, 185, 129, 0.12)", border: "#10b981" },
    { name: "Vàng (Speaker 5)", hex: "#fbbf24", bg: "rgba(251, 191, 36, 0.12)", border: "#fbbf24" },
    { name: "Xanh Dương (Speaker 6)", hex: "#3b82f6", bg: "rgba(59, 130, 246, 0.12)", border: "#3b82f6" },
];

function getSpeakerColorObj(speakerId, fallbackIndex = 0) {
    if (!speakerId || speakerId.toLowerCase().includes("overlap")) {
        return { name: "Overlap (Đè giọng)", hex: "#f43f5e", bg: "rgba(244, 63, 94, 0.2)", border: "#f43f5e" };
    }
    const numMatch = speakerId.match(/\d+/);
    const idx = numMatch ? parseInt(numMatch[0], 10) : fallbackIndex;
    return SPEAKER_PALETTE[idx % SPEAKER_PALETTE.length];
}

function initDiarizationState() {
    const savedToken = localStorage.getItem("hf_token");
    if (savedToken && hfTokenInput) {
        hfTokenInput.value = savedToken;
    }
}

function toggleDiarSourceType(type) {
    const procContainer = document.getElementById("diarProcessedSourceContainer");
    const crawlContainer = document.getElementById("diarCrawlSourceContainer");
    const procSelect = document.getElementById("diarVocalSelect");
    const crawlSelect = document.getElementById("diarCrawlSelect");

    if (type === "processed") {
        procContainer.classList.remove("hidden");
        crawlContainer.classList.add("hidden");
        if (procSelect) procSelect.required = true;
        if (crawlSelect) crawlSelect.required = false;
    } else {
        procContainer.classList.add("hidden");
        crawlContainer.classList.remove("hidden");
        if (procSelect) procSelect.required = false;
        if (crawlSelect) crawlSelect.required = true;
    }
}

function toggleDiarEngine(engine) {
    const hfContainer = document.getElementById("hfTokenContainer");
    const labelOffline = document.getElementById("labelEngineOffline");
    const labelPyannote = document.getElementById("labelEnginePyannote");

    if (engine === "pyannote") {
        if (hfContainer) hfContainer.classList.remove("hidden");
        if (labelPyannote) labelPyannote.classList.add("active");
        if (labelOffline) labelOffline.classList.remove("active");
    } else {
        if (hfContainer) hfContainer.classList.add("hidden");
        if (labelOffline) labelOffline.classList.add("active");
        if (labelPyannote) labelPyannote.classList.remove("active");
    }
}

function toggleTokenVisibility() {
    if (!hfTokenInput) return;
    if (hfTokenInput.type === "password") {
        hfTokenInput.type = "text";
    } else {
        hfTokenInput.type = "password";
    }
}

// Fetch sources for Diarization
async function loadDiarizationSources(preselectVal = null) {
    const vocalSelect = document.getElementById("diarVocalSelect");
    const crawlSelect = document.getElementById("diarCrawlSelect");
    if (!vocalSelect || !crawlSelect) return;

    try {
        const res = await fetch("/api/diarization/sources");
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Không thể tải danh sách nguồn audio.");

        // 1. Render Processed Vocals
        const vocals = data.processed_vocals || [];
        if (vocals.length === 0) {
            vocalSelect.innerHTML = `<option value="">⚠️ Chưa có bản vocal nào từ Stage 2 (Hãy tách âm trước)</option>`;
        } else {
            vocalSelect.innerHTML = `<option value="">-- Chọn bản tách vocal (${vocals.length} bản khả dụng) --</option>` +
                vocals.map(v => {
                    const modelNameMap = {
                        "htdemucs": "HT Demucs",
                        "mel_roformer": "Mel-RoFormer",
                        "deepfilternet": "DeepFilterNet",
                        "ht_then_mel": "Cascade HT ➔ Mel",
                        "mel_then_ht": "Cascade Mel ➔ HT"
                    };
                    const mName = modelNameMap[v.model] || v.model;
                    const val = `${v.model}::${v.run_id}`;
                    const isSel = (preselectVal && preselectVal === val) ? "selected" : "";
                    return `<option value="${val}" ${isSel}>[${mName}] ${v.input_filename} (${v.filesize_formatted})</option>`;
                }).join("");
        }

        // 2. Render Crawled Audios
        const crawls = data.crawl_audios || [];
        if (crawls.length === 0) {
            crawlSelect.innerHTML = `<option value="">⚠️ Chưa có tệp crawl nào trong audio_crawl/</option>`;
        } else {
            crawlSelect.innerHTML = `<option value="">-- Chọn file audio gốc (${crawls.length} tệp) --</option>` +
                crawls.map(c => {
                    return `<option value="${c.filename}">${c.title || c.filename} (${c.duration_formatted || "N/A"})</option>`;
                }).join("");
        }
    } catch (err) {
        console.error("Error loading diarization sources:", err);
    }
}

// Handover shortcut from Stage 2 / Stage 3 to Stage 4
function useVocalForDiarization(model, runId, filename) {
    switchTab("diarization");
    const radio = document.querySelector('input[name="diarSourceType"][value="processed"]');
    if (radio) {
        radio.checked = true;
        toggleDiarSourceType("processed");
    }
    const val = `${model}::${runId}`;
    loadDiarizationSources(val);
    showToast(`Đã chọn vocal "${filename || runId}" cho Stage 4!`, "success", 3000);
}

// Run Diarization Handler
if (diarizationForm) {
    diarizationForm.addEventListener("submit", async (e) => {
        e.preventDefault();

        const sourceType = document.querySelector('input[name="diarSourceType"]:checked')?.value || "processed";
        const engine = document.querySelector('input[name="diarEngine"]:checked')?.value || "offline_clustering";
        const numSpeakersVal = document.getElementById("diarNumSpeakers")?.value;
        const minDurationVal = parseFloat(document.getElementById("diarMinDuration")?.value || "0.5");
        const filterOverlap = document.getElementById("diarFilterOverlap")?.checked ?? true;
        const hfToken = hfTokenInput?.value?.trim() || null;

        if (hfToken) {
            localStorage.setItem("hf_token", hfToken);
        }

        let payload = {
            source_type: sourceType,
            engine: engine,
            hf_token: hfToken,
            num_speakers: numSpeakersVal ? parseInt(numSpeakersVal, 10) : null,
            min_duration_s: minDurationVal,
            filter_overlap: filterOverlap
        };

        if (sourceType === "processed") {
            const vocalVal = document.getElementById("diarVocalSelect")?.value;
            if (!vocalVal || !vocalVal.includes("::")) {
                showToast("Vui lòng chọn một bản tách Vocal từ danh sách!", "error");
                return;
            }
            const [model, runId] = vocalVal.split("::");
            payload.processed_model = model;
            payload.processed_run_id = runId;
        } else {
            const filename = document.getElementById("diarCrawlSelect")?.value;
            if (!filename) {
                showToast("Vui lòng chọn file audio từ thư viện crawl!", "error");
                return;
            }
            payload.filename = filename;
        }

        // UI State: Starting Diarization
        runDiarizationBtn.disabled = true;
        runDiarizationBtn.querySelector(".btn-text")?.classList.add("hidden");
        runDiarizationBtn.querySelector(".btn-spinner")?.classList.remove("hidden");
        diarizationProgress?.classList.remove("hidden");
        diarizationResults?.classList.add("hidden");

        let sec = 0;
        diarProgressTimer.textContent = "00:00";
        const timerInterval = setInterval(() => {
            sec++;
            const m = String(Math.floor(sec / 60)).padStart(2, "0");
            const s = String(sec % 60).padStart(2, "0");
            diarProgressTimer.textContent = `${m}:${s}`;
        }, 1000);

        try {
            const res = await fetch("/api/diarization", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            const data = await res.json();
            if (!res.ok) {
                throw new Error(data.detail || "Quá trình Speaker Diarization thất bại.");
            }

            showToast("🎉 Phân đoạn người nói thành công!", "success", 4000);
            renderDiarizationResults(data.data);
            await loadDiarizationHistory();
        } catch (err) {
            console.error("Diarization error:", err);
            showToast(err.message, "error", 6000);
        } finally {
            clearInterval(timerInterval);
            runDiarizationBtn.disabled = false;
            runDiarizationBtn.querySelector(".btn-text")?.classList.remove("hidden");
            runDiarizationBtn.querySelector(".btn-spinner")?.classList.add("hidden");
            diarizationProgress?.classList.add("hidden");
        }
    });
}

// Render Diarization Results UI
function renderDiarizationResults(data) {
    if (!diarizationResults || !data) return;

    const totalDur = data.total_duration_s || 1.0;
    const speakers = data.speakers || [];
    const turns = data.turns || [];

    const engineNameMap = {
        "offline_clustering": "SpeechBrain ECAPA-TDNN (Offline SOTA)",
        "pyannote": "PyAnnote Audio 3.1 Neural Diarization"
    };
    const engineDisplayName = engineNameMap[data.engine] || data.engine;

    // Build Speaker Cards HTML
    const speakerCardsHtml = speakers.map((spk, idx) => {
        const color = getSpeakerColorObj(spk.speaker_id, idx);
        const formatTime = (s) => {
            const m = Math.floor(s / 60);
            const sec = Math.floor(s % 60);
            return `${m}p ${sec}s (${s.toFixed(1)}s)`;
        };

        return `
            <div class="speaker-summary-card" style="border-left: 4px solid ${color.border};">
                <div class="spk-card-top">
                    <div class="spk-avatar-title">
                        <div class="spk-avatar" style="background: ${color.hex};">
                            ${spk.speaker_id.replace("SPEAKER_", "S")}
                        </div>
                        <div>
                            <span class="spk-name">${spk.speaker_id}</span>
                            <span style="font-size: 0.76rem; color: var(--text-dim);">${color.name}</span>
                        </div>
                    </div>
                    <span class="spk-percent-badge" style="background: ${color.bg}; color: ${color.hex}; border: 1px solid ${color.border};">
                        ${spk.percentage}%
                    </span>
                </div>

                <div class="spk-stats-rows">
                    <div class="spk-stats-row">
                        <span>Thời lượng nói:</span>
                        <strong style="color: var(--text-main); font-family: var(--font-mono);">${formatTime(spk.total_time_s)}</strong>
                    </div>
                    <div class="spk-stats-row">
                        <span>Số lượt nói (Turns):</span>
                        <strong style="color: var(--text-main); font-family: var(--font-mono);">${spk.turn_count} câu thoại</strong>
                    </div>
                </div>

                <div class="spk-actions-bar">
                    ${spk.sample_audio_url ? `
                        <button type="button" class="btn-secondary" style="font-size: 0.76rem; padding: 5px 9px;" onclick="playProcessedAudio('${spk.sample_audio_url}', '${spk.speaker_id} Preview', 'sample.wav')">
                            ▶ Nghe Mẫu Giọng
                        </button>
                    ` : ""}
                    ${spk.master_audio_url ? `
                        <a href="${spk.master_audio_url}" download="${spk.speaker_id}_full.wav" class="btn-secondary" style="font-size: 0.76rem; padding: 5px 9px; text-decoration: none;">
                            📥 Tải Toàn Bộ Giọng
                        </a>
                    ` : ""}
                    <button type="button" class="btn-stage5-handover" onclick="handoverToStage5('${data.run_id}', '${spk.speaker_id}', '${spk.master_audio_url || ""}')">
                        🚀 Dùng Cho Stage 5
                    </button>
                </div>
            </div>
        `;
    }).join("");

    // Build Interactive Timeline Blocks
    const timelineBlocksHtml = turns.map(t => {
        const color = getSpeakerColorObj(t.speaker_id);
        const leftPct = (t.start_s / totalDur) * 100;
        const widthPct = Math.max(0.4, (t.duration_s / totalDur) * 100);
        const formatSec = (s) => {
            const m = String(Math.floor(s / 60)).padStart(2, "0");
            const sec = String(Math.floor(s % 60)).padStart(2, "0");
            return `${m}:${sec}`;
        };

        const tip = `${t.speaker_id} [${formatSec(t.start_s)} - ${formatSec(t.end_s)}] (${t.duration_s.toFixed(1)}s)${t.is_overlap ? ' - Overlap!' : ''}`;

        return `
            <div class="timeline-block" 
                 style="position: absolute; left: ${leftPct}%; width: ${widthPct}%; background: ${t.is_overlap ? '#f43f5e' : color.hex};" 
                 title="${tip}"
                 onclick="seekAndPlayDiarTurn(${t.start_s})">
            </div>
        `;
    }).join("");

    // Build Legend HTML
    const legendItemsHtml = speakers.map(s => {
        const color = getSpeakerColorObj(s.speaker_id);
        return `
            <div class="legend-item">
                <span class="legend-dot" style="background: ${color.hex};"></span>
                <span><strong>${s.speaker_id}</strong> (${s.percentage}%)</span>
            </div>
        `;
    }).join("") + (data.overlap_filtered ? `
        <div class="legend-item" style="margin-left: auto;">
            <span class="legend-dot" style="background: #f43f5e;"></span>
            <span style="color: #f43f5e;">🛡️ Đã lọc bỏ Overlap</span>
        </div>
    ` : "");

    // Build Turns Table HTML
    const spkTurnCounters = {};
    const turnsTableHtml = turns.map((t, idx) => {
        const color = getSpeakerColorObj(t.speaker_id);
        spkTurnCounters[t.speaker_id] = (spkTurnCounters[t.speaker_id] || 0) + 1;
        const spkTurnIdx = spkTurnCounters[t.speaker_id];
        const turnFilename = t.turn_filename || `turn_${String(spkTurnIdx).padStart(3, "0")}_${String(t.start_s.toFixed(2)).padStart(6, "0")}-${String(t.end_s.toFixed(2)).padStart(6, "0")}.wav`;
        const clipUrl = t.clip_url || `/api/diarized/${data.run_id}/speakers/${t.speaker_id}/${turnFilename}`;

        return `
            <tr>
                <td style="font-family: var(--font-mono); color: var(--text-dim);">${idx + 1}</td>
                <td>
                    <span class="spk-percent-badge" style="background: ${color.bg}; color: ${color.hex}; border: 1px solid ${color.border};">
                        ${t.speaker_id}
                    </span>
                </td>
                <td style="font-family: var(--font-mono);">${t.start_s.toFixed(2)}s ➔ ${t.end_s.toFixed(2)}s</td>
                <td style="font-family: var(--font-mono); font-weight: 600;">${t.duration_s.toFixed(2)}s</td>
                <td>
                    ${t.is_overlap 
                        ? `<span style="color: #f43f5e; font-weight: 600; font-size: 0.74rem;">⚠️ Overlap</span>` 
                        : `<span style="color: #10b981; font-weight: 600; font-size: 0.74rem;">✓ Đơn giọng</span>`}
                </td>
                <td>
                    <button type="button" class="btn-secondary" style="font-size: 0.74rem; padding: 3px 8px;" onclick="playProcessedAudio('${clipUrl}', '${t.speaker_id} #${idx + 1}', '${turnFilename}')">
                        ▶ Nghe Đoạn
                    </button>
                </td>
            </tr>
        `;
    }).join("");

    diarizationResults.innerHTML = `
        <div class="diar-overview-card">
            <div class="diar-overview-header">
                <div>
                    <h3 style="font-size: 1.15rem; color: var(--text-main); display: flex; align-items: center; gap: 8px;">
                        <span>👥 Kết Quả Phân Đoạn: ${speakers.length} Người Nói</span>
                        <span class="badge" style="background: rgba(99, 102, 241, 0.15); color: var(--accent-cyan); font-size: 0.74rem;">${engineDisplayName}</span>
                    </h3>
                    <p style="font-size: 0.8rem; color: var(--text-dim); margin-top: 2px;">
                        Tệp nguồn: <strong>${data.input_filename}</strong> | Tổng thời lượng: <strong>${totalDur.toFixed(1)}s</strong> | Tổng lượt nói: <strong>${turns.length} turns</strong>
                    </p>
                </div>
            </div>

            <!-- Speaker Summary Grid -->
            <div class="speaker-summary-grid">
                ${speakerCardsHtml}
            </div>
        </div>

        <!-- Interactive Timeline Track -->
        <div class="diar-timeline-card">
            <div class="diar-timeline-header">
                <span>⏱️ Dòng Thời Gian Lượt Nói (Speaker Timeline):</span>
                <span style="font-size: 0.76rem; color: var(--text-muted);">Nhấp vào dải màu bất kỳ để nghe trực tiếp mốc thời gian đó</span>
            </div>
            <div class="diar-timeline-track" style="position: relative;">
                ${timelineBlocksHtml}
            </div>
            <div class="diar-legend-bar">
                ${legendItemsHtml}
            </div>
        </div>

        <!-- Detailed Turns Table -->
        <div class="diar-turns-card">
            <h4 style="font-size: 0.95rem; color: var(--text-main); margin-bottom: 8px;">
                📋 Danh Sách Chi Tiết Từng Câu Thoại (${turns.length} câu đã phân đoạn):
            </h4>
            <div class="diar-table-responsive">
                <table class="diar-turns-table">
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Người Nói</th>
                            <th>Mốc Thời Gian</th>
                            <th>Thời Lượng</th>
                            <th>Trạng Thái</th>
                            <th>Thao Tác</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${turnsTableHtml}
                    </tbody>
                </table>
            </div>
        </div>
    `;

    diarizationResults.classList.remove("hidden");
    diarizationResults.scrollIntoView({ behavior: "smooth", block: "start" });
}

function seekAndPlayDiarTurn(startSeconds) {
    if (globalAudio && !isNaN(globalAudio.duration)) {
        globalAudio.currentTime = startSeconds;
        globalAudio.play();
        showToast(`Đang phát từ mốc ${startSeconds.toFixed(1)}s`, "info", 1500);
    }
}

// Load & Render Diarization History
async function loadDiarizationHistory() {
    if (!diarizationHistoryContainer) return;

    try {
        const res = await fetch("/api/diarization/history");
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Không thể tải lịch sử diarization.");

        renderDiarizationHistory(data.history || []);
    } catch (err) {
        console.error("Error loading diarization history:", err);
    }
}

function renderDiarizationHistory(history) {
    if (!diarizationHistoryContainer) return;

    if (history.length === 0) {
        diarizationHistoryContainer.innerHTML = `
            <div class="empty-state" style="padding: 24px 0;">
                <p>Chưa có bản ghi phân đoạn nào trong <code>diarized_audio/</code>.</p>
            </div>
        `;
        return;
    }

    diarizationHistoryContainer.innerHTML = history.map(item => {
        const createdDate = item.created_at ? new Date(item.created_at).toLocaleString("vi-VN") : "N/A";
        const speakers = item.speakers || [];
        const spkBadges = speakers.map(s => {
            const color = getSpeakerColorObj(s.speaker_id);
            return `<span class="spk-percent-badge" style="background: ${color.bg}; color: ${color.hex}; border: 1px solid ${color.border}; font-size: 0.72rem;">${s.speaker_id} (${s.percentage}%)</span>`;
        }).join(" ");

        return `
            <div class="separation-history-card">
                <div class="history-card-left">
                    <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                        <span class="badge badge-model">${item.engine || "Diarization"}</span>
                        <strong style="color: var(--text-main);">${item.input_filename}</strong>
                        <span style="font-family: var(--font-mono); font-size: 0.76rem; color: var(--text-muted);">${item.run_id}</span>
                    </div>
                    <div style="margin-top: 6px; display: flex; gap: 6px; flex-wrap: wrap; align-items: center;">
                        <span style="font-size: 0.78rem; color: var(--text-muted);">${createdDate} | ${item.num_speakers} Người nói | ${item.turns?.length || 0} turns</span>
                        ${spkBadges}
                    </div>
                </div>

                <div class="history-card-actions">
                    <button type="button" class="btn-secondary" style="font-size: 0.78rem; padding: 6px 12px;" onclick='renderDiarizationResults(${JSON.stringify(item)})'>
                        🔍 Xem Chi Tiết
                    </button>
                    <button type="button" class="btn-delete" style="font-size: 0.78rem; padding: 6px 12px;" onclick="deleteDiarizationRun('${item.run_id}')">
                        🗑️ Xóa
                    </button>
                </div>
            </div>
        `;
    }).join("");
}

async function deleteDiarizationRun(runId) {
    if (!confirm(`Bạn có chắc chắn muốn xóa bản ghi Diarization ${runId}?`)) return;

    try {
        const res = await fetch(`/api/diarization/${runId}`, { method: "DELETE" });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Xóa bản ghi thất bại.");

        showToast(data.message || "Đã xóa bản ghi Diarization thành công.", "success");
        await loadDiarizationHistory();
    } catch (err) {
        showToast(err.message, "error");
    }
}

if (refreshDiarHistoryBtn) {
    refreshDiarHistoryBtn.addEventListener("click", () => {
        loadDiarizationHistory();
        showToast("Đã làm mới lịch sử diarization.", "success", 1500);
    });
}

// ============================================================================
// STAGE 5: WORD ALIGNMENT & VAD LOGIC
// ============================================================================

let currentAlignmentResult = null;
let alignTimerInterval = null;

const alignmentForm = document.getElementById("alignmentForm");
const runAlignmentBtn = document.getElementById("runAlignmentBtn");
const alignmentProgress = document.getElementById("alignmentProgress");
const alignProgressTimer = document.getElementById("alignProgressTimer");
const alignmentResults = document.getElementById("alignmentResults");
const alignmentHistoryContainer = document.getElementById("alignmentHistoryContainer");
const refreshAlignHistoryBtn = document.getElementById("refreshAlignHistoryBtn");

const alignDiarizedGroup = document.getElementById("alignDiarizedGroup");
const alignProcessedGroup = document.getElementById("alignProcessedGroup");
const alignCrawlGroup = document.getElementById("alignCrawlGroup");

const alignDiarizedSelect = document.getElementById("alignDiarizedSelect");
const alignProcessedSelect = document.getElementById("alignProcessedSelect");
const alignCrawlSelect = document.getElementById("alignCrawlSelect");

function initAlignmentState() {
    // Source radio change
    const sourceRadios = document.querySelectorAll('input[name="alignSourceType"]');
    sourceRadios.forEach(radio => {
        radio.addEventListener("change", () => {
            const val = radio.value;
            if (alignDiarizedGroup) alignDiarizedGroup.classList.toggle("hidden", val !== "diarized");
            if (alignProcessedGroup) alignProcessedGroup.classList.toggle("hidden", val !== "processed");
            if (alignCrawlGroup) alignCrawlGroup.classList.toggle("hidden", val !== "crawl");
        });
    });

    // Model card click styling
    const modelCards = document.querySelectorAll('#tab-alignment .model-card');
    modelCards.forEach(card => {
        card.addEventListener("click", () => {
            modelCards.forEach(c => c.classList.remove("active"));
            card.classList.add("active");
            const radio = card.querySelector('input[type="radio"]');
            if (radio) radio.checked = true;
        });
    });

    // Form submission
    if (alignmentForm) {
        alignmentForm.addEventListener("submit", runAlignmentHandler);
    }

    // Refresh history
    if (refreshAlignHistoryBtn) {
        refreshAlignHistoryBtn.addEventListener("click", () => {
            loadAlignmentHistory();
            showToast("Đã làm mới lịch sử Word Alignment.", "success", 1500);
        });
    }

    // Audio timeupdate synchronization for Karaoke Word Highlight
    if (globalAudio) {
        globalAudio.addEventListener("timeupdate", () => {
            const currentTab = document.querySelector(".tab-pane.active");
            if (currentTab && currentTab.id === "tab-alignment") {
                syncKaraokeWordHighlight(globalAudio.currentTime);
            }
        });
    }

    // Initial load
    loadAlignmentSources();
    loadAlignmentHistory();
}

async function loadAlignmentSources() {
    try {
        const res = await fetch("/api/alignment/sources");
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Không thể tải danh sách nguồn audio.");

        // 1. Diarized Speakers
        if (alignDiarizedSelect) {
            const diarList = data.diarized_speakers || [];
            if (diarList.length === 0) {
                alignDiarizedSelect.innerHTML = `<option value="">-- Chưa có bản Diarization nào. Vui lòng chạy Stage 4 trước --</option>`;
            } else {
                alignDiarizedSelect.innerHTML = diarList.map(s => `
                    <option value="${s.run_id}__${s.speaker_id}">
                        🎙️ [${s.run_id.substring(0, 6)}] ${s.speaker_id} (${s.speaker_label}) | ${s.input_filename} | ${s.duration_formatted}
                    </option>
                `).join("");
            }
        }

        // 2. Processed Vocals
        if (alignProcessedSelect) {
            const procList = data.processed_vocals || [];
            if (procList.length === 0) {
                alignProcessedSelect.innerHTML = `<option value="">-- Chưa có bản tách vocal nào trong Stage 2 --</option>`;
            } else {
                alignProcessedSelect.innerHTML = procList.map(p => `
                    <option value="${p.model}__${p.run_id}">
                        🎚️ [${p.model}] ${p.input_filename} (${p.duration_formatted} - ${p.filesize_formatted})
                    </option>
                `).join("");
            }
        }

        // 3. Crawl Audios
        if (alignCrawlSelect) {
            const crawlList = data.crawl_audios || [];
            if (crawlList.length === 0) {
                alignCrawlSelect.innerHTML = `<option value="">-- Chưa có audio nào trong thư viện Crawl --</option>`;
            } else {
                alignCrawlSelect.innerHTML = crawlList.map(c => `
                    <option value="${c.filename}">
                        📁 ${c.title} (${c.duration_formatted} - ${c.filesize_formatted})
                    </option>
                `).join("");
            }
        }
    } catch (err) {
        console.error("Error loading alignment sources:", err);
    }
}

async function runAlignmentHandler(e) {
    e.preventDefault();

    const sourceType = document.querySelector('input[name="alignSourceType"]:checked')?.value || "diarized";
    const modelSize = document.querySelector('input[name="alignModelSize"]:checked')?.value || "large-v3";
    const language = document.getElementById("alignLanguage")?.value || "auto";
    const vadFilter = document.getElementById("alignVadFilter")?.checked ?? true;
    const wordTimestamps = document.getElementById("alignWordTimestamps")?.checked ?? true;
    const initialPrompt = document.getElementById("alignInitialPrompt")?.value?.trim() || null;

    const payload = {
        source_type: sourceType,
        model_size: modelSize,
        language: language,
        vad_filter: vadFilter,
        beam_size: 5,
        word_timestamps: wordTimestamps,
        initial_prompt: initialPrompt,
    };

    if (sourceType === "diarized") {
        const val = alignDiarizedSelect?.value;
        if (!val || !val.includes("__")) {
            showToast("Vui lòng chọn một Giọng Speaker từ danh sách Diarization.", "error");
            return;
        }
        const [rId, spkId] = val.split("__");
        payload.diarized_run_id = rId;
        payload.diarized_speaker_id = spkId;
    } else if (sourceType === "processed") {
        const val = alignProcessedSelect?.value;
        if (!val || !val.includes("__")) {
            showToast("Vui lòng chọn một bản tách Vocal từ Stage 2.", "error");
            return;
        }
        const [pModel, rId] = val.split("__");
        payload.processed_model = pModel;
        payload.processed_run_id = rId;
    } else {
        const filename = alignCrawlSelect?.value;
        if (!filename) {
            showToast("Vui lòng chọn một file từ thư viện crawl.", "error");
            return;
        }
        payload.filename = filename;
    }

    // UI Loading state
    if (runAlignmentBtn) {
        runAlignmentBtn.disabled = true;
        runAlignmentBtn.querySelector(".btn-text")?.classList.add("hidden");
        runAlignmentBtn.querySelector(".btn-spinner")?.classList.remove("hidden");
    }
    if (alignmentProgress) alignmentProgress.classList.remove("hidden");
    if (alignmentResults) alignmentResults.classList.add("hidden");

    const statusTextEl = document.getElementById("alignProgressStatusText");
    const pctBadgeEl = document.getElementById("alignProgressPctBadge");
    const barFillEl = document.getElementById("alignProgressBarFill");
    const detailTextEl = document.getElementById("alignProgressDetailText");
    const speedBadgeEl = document.getElementById("alignProgressSpeedBadge");
    const statAudioTimeEl = document.getElementById("alignStatAudioTime");
    const statWordsCountEl = document.getElementById("alignStatWordsCount");
    const statEtaEl = document.getElementById("alignStatEta");
    const liveSnippetTextEl = document.getElementById("alignLiveSnippetText");

    if (statusTextEl) statusTextEl.textContent = "Đang nạp mô hình Whisper...";
    if (pctBadgeEl) pctBadgeEl.textContent = "0%";
    if (barFillEl) barFillEl.style.width = "0%";
    if (speedBadgeEl) speedBadgeEl.textContent = "⚡ 0.0x GPU";
    if (statAudioTimeEl) statAudioTimeEl.textContent = "00:00 / 00:00";
    if (statWordsCountEl) statWordsCountEl.textContent = "0 từ";
    if (statEtaEl) statEtaEl.textContent = "~0s";
    if (liveSnippetTextEl) liveSnippetTextEl.textContent = "Đang chuẩn bị âm thanh và nạp trọng số lên GPU...";

    let sec = 0;
    if (alignProgressTimer) alignProgressTimer.textContent = "00:00";
    if (alignTimerInterval) clearInterval(alignTimerInterval);
    alignTimerInterval = setInterval(() => {
        sec++;
        const m = String(Math.floor(sec / 60)).padStart(2, "0");
        const s = String(sec % 60).padStart(2, "0");
        if (alignProgressTimer) alignProgressTimer.textContent = `${m}:${s}`;
    }, 1000);

    const formatSecondsToMinSec = (totalSec) => {
        if (!totalSec || isNaN(totalSec)) return "00:00";
        const m = Math.floor(totalSec / 60);
        const s = Math.floor(totalSec % 60);
        return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
    };

    // Start realtime progress polling
    let progressPollInterval = setInterval(async () => {
        try {
            const pRes = await fetch("/api/alignment/progress");
            if (!pRes.ok) return;
            const pData = await pRes.json();
            const prog = pData.progress;
            if (!prog) return;

            const pct = Math.min(100, Math.max(0, prog.progress_percent || 0));
            if (barFillEl) barFillEl.style.width = `${pct}%`;
            if (pctBadgeEl) pctBadgeEl.textContent = `${Math.round(pct)}%`;
            if (statusTextEl && prog.message) statusTextEl.textContent = prog.message;

            if (prog.status === "downloading_model") {
                if (speedBadgeEl && prog.speed_mb_s !== undefined) {
                    speedBadgeEl.textContent = `📥 ${prog.speed_mb_s.toFixed(2)} MB/s`;
                }
                if (statAudioTimeEl && prog.total_mb) {
                    statAudioTimeEl.textContent = `${prog.downloaded_mb || 0} MB / ${prog.total_mb} MB`;
                }
                if (statWordsCountEl) {
                    statWordsCountEl.textContent = `File: ${prog.file_name || "weights"}`;
                }
                if (statEtaEl) {
                    statEtaEl.textContent = prog.eta_s ? `~${Math.round(prog.eta_s)}s` : "~0s";
                }
                if (liveSnippetTextEl) {
                    liveSnippetTextEl.textContent = `Đang tải ${prog.file_name || "mô hình"} từ CDN trực tiếp vào thư mục cache...`;
                }
                if (detailTextEl) {
                    detailTextEl.textContent = `Đã tải: ${prog.downloaded_mb || 0} MB / ${prog.total_mb || 0} MB • Tốc độ: ${prog.speed_mb_s || 0} MB/s`;
                }
            } else if (prog.status === "loading_gpu") {
                if (speedBadgeEl) speedBadgeEl.textContent = "⚡ GPU Alloc";
                if (statAudioTimeEl) statAudioTimeEl.textContent = "Nạp VRAM";
                if (statWordsCountEl) statWordsCountEl.textContent = `${prog.model_size || "Whisper"}`;
                if (statEtaEl) statEtaEl.textContent = "~2s";
                if (liveSnippetTextEl) liveSnippetTextEl.textContent = `Đang khởi tạo Tensor CTranslate2 trên GPU NVIDIA RTX 3090 (float16)...`;
                if (detailTextEl) detailTextEl.textContent = `Đang nạp mô hình ${prog.model_size || ""} vào bộ nhớ GPU...`;
            } else {
                // Aligning / Transcribing phase
                if (speedBadgeEl && prog.speed_x) {
                    speedBadgeEl.textContent = `⚡ ${prog.speed_x.toFixed(1)}x GPU`;
                }

                if (statAudioTimeEl) {
                    const curStr = formatSecondsToMinSec(prog.current_time_s || 0);
                    const totStr = formatSecondsToMinSec(prog.total_time_s || 0);
                    statAudioTimeEl.textContent = `${curStr} / ${totStr}`;
                }

                if (statWordsCountEl) {
                    statWordsCountEl.textContent = `${prog.words_count || 0} từ (${prog.segments_count || 0} câu)`;
                }

                if (statEtaEl) {
                    statEtaEl.textContent = prog.eta_s ? `~${Math.round(prog.eta_s)}s` : "~0s";
                }

                if (liveSnippetTextEl && prog.latest_snippet) {
                    liveSnippetTextEl.textContent = `"${prog.latest_snippet}"`;
                }

                if (detailTextEl) {
                    if (prog.total_time_s > 0) {
                        detailTextEl.textContent = `Đã gióng: ${prog.words_count || 0} từ • ${prog.current_time_s || 0}s / ${prog.total_time_s || 0}s âm thanh`;
                    } else {
                        detailTextEl.textContent = prog.message || "Đang xử lý...";
                    }
                }
            }
        } catch (e) {
            // ignore polling errors
        }
    }, 600);

    try {
        const res = await fetch("/api/alignment", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Gióng hàng từ (Alignment) thất bại.");

        if (barFillEl) barFillEl.style.width = "100%";
        if (pctBadgeEl) pctBadgeEl.textContent = "100%";
        if (statusTextEl) statusTextEl.textContent = "Hoàn tất gióng hàng từ!";

        currentAlignmentResult = data.data;
        renderAlignmentResults(data.data);
        showToast(`Gióng hàng thành công: ${data.data.total_words} từ trong ${data.data.total_segments} câu!`, "success", 4000);
        await loadAlignmentHistory();
    } catch (err) {
        showToast(err.message, "error", 6000);
    } finally {
        if (progressPollInterval) clearInterval(progressPollInterval);
        if (alignTimerInterval) clearInterval(alignTimerInterval);
        if (runAlignmentBtn) {
            runAlignmentBtn.disabled = false;
            runAlignmentBtn.querySelector(".btn-text")?.classList.remove("hidden");
            runAlignmentBtn.querySelector(".btn-spinner")?.classList.add("hidden");
        }
        if (alignmentProgress) alignmentProgress.classList.add("hidden");
    }
}

function renderAlignmentResults(data) {
    if (!alignmentResults || !data) return;
    alignmentResults.classList.remove("hidden");

    const totalDur = data.total_duration_s || 0;
    const durFormat = Math.floor(totalDur / 60) + ":" + String(Math.floor(totalDur % 60)).padStart(2, "0");

    const words = data.words || [];
    const segments = data.segments || [];

    // Build Word Badges HTML
    const wordBadgesHtml = words.map(w => {
        const confColor = w.probability >= 0.8 ? "rgba(16, 185, 129, 0.15)" : "rgba(245, 158, 11, 0.15)";
        const confBorder = w.probability >= 0.8 ? "rgba(16, 185, 129, 0.3)" : "rgba(245, 158, 11, 0.3)";

        return `
            <span class="word-badge" 
                  data-start="${w.start_s}" 
                  data-end="${w.end_s}" 
                  style="border-color: ${confBorder};"
                  title="Từ: ${w.word} | [${w.start_s.toFixed(2)}s - ${w.end_s.toFixed(2)}s] | Độ tin cậy: ${(w.probability * 100).toFixed(0)}%"
                  onclick="seekAndPlayAlignWord(${w.start_s})">
                ${w.word}
                <span class="word-time">${w.start_s.toFixed(1)}s</span>
            </span>
        `;
    }).join("");

    // Build Paragraph View HTML
    const paragraphsHtml = segments.map(seg => `
        <div class="paragraph-item">
            <div class="paragraph-time" onclick="seekAndPlayAlignWord(${seg.start_s})">
                ⏱️ [${seg.start_s.toFixed(2)}s ➔ ${seg.end_s.toFixed(2)}s] (Câu #${seg.id})
            </div>
            <div>${seg.text}</div>
        </div>
    `).join("");

    // Build SRT Subtitle text preview
    let srtText = "";
    segments.forEach(seg => {
        const formatTs = (sec) => {
            const m = Math.floor(sec / 60);
            const s = Math.floor(sec % 60);
            const ms = Math.floor((sec - Math.floor(sec)) * 1000);
            return `00:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")},${String(ms).padStart(3, "0")}`;
        };
        srtText += `${seg.id}\n${formatTs(seg.start_s)} --> ${formatTs(seg.end_s)}\n${seg.text}\n\n`;
    });

    alignmentResults.innerHTML = `
        <div class="diar-overview-card">
            <div class="diar-overview-header">
                <div>
                    <h3 style="font-size: 1.15rem; color: var(--text-main); display: flex; align-items: center; gap: 8px;">
                        <span>⏱️ Kết Quả Word Alignment: ${data.total_words} Từ</span>
                        <span class="badge badge-accent">${data.model_size || "large-v3"}</span>
                        <span class="badge" style="background: rgba(16, 185, 129, 0.15); color: #10b981;">Ngôn ngữ: ${data.language?.toUpperCase() || "VI"} (${((data.language_probability || 1.0) * 100).toFixed(0)}%)</span>
                    </h3>
                    <p style="font-size: 0.8rem; color: var(--text-dim); margin-top: 2px;">
                        Tệp nguồn: <strong>${data.input_filename}</strong> | Thời lượng: <strong>${durFormat}</strong> | Tốc độ: <strong>${data.words_per_minute || 0} từ/phút</strong>
                    </p>
                </div>

                <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                    <a href="/api/aligned/${data.run_id}/subtitles.srt" download="subtitles_${data.run_id}.srt" class="btn-secondary" style="font-size: 0.78rem; padding: 6px 12px; text-decoration: none;">
                        📥 Tải SRT
                    </a>
                    <a href="/api/aligned/${data.run_id}/subtitles.vtt" download="subtitles_${data.run_id}.vtt" class="btn-secondary" style="font-size: 0.78rem; padding: 6px 12px; text-decoration: none;">
                        📥 Tải VTT
                    </a>
                    <a href="/api/aligned/${data.run_id}/words.json" download="words_${data.run_id}.json" class="btn-secondary" style="font-size: 0.78rem; padding: 6px 12px; text-decoration: none;">
                        📥 Tải Words JSON
                    </a>
                    <button type="button" class="btn-stage6-handover" onclick="handoverToStage6('${data.run_id}')">
                        🚀 Dùng Cho Stage 6 (Smart Chunking)
                    </button>
                </div>
            </div>

            <!-- Alignment Metric Stats Grid -->
            <div class="align-metrics-grid" style="margin-top: 16px;">
                <div class="align-metric-card">
                    <div class="align-metric-label">Tổng Số Từ</div>
                    <div class="align-metric-val">${data.total_words}</div>
                </div>
                <div class="align-metric-card">
                    <div class="align-metric-label">Số Câu / Đoạn</div>
                    <div class="align-metric-val">${data.total_segments}</div>
                </div>
                <div class="align-metric-card">
                    <div class="align-metric-label">Tốc Độ Nói</div>
                    <div class="align-metric-val">${data.words_per_minute} <span style="font-size: 0.8rem; font-weight: normal;">wpm</span></div>
                </div>
                <div class="align-metric-card">
                    <div class="align-metric-label">Ngôn Ngữ</div>
                    <div class="align-metric-val" style="color: var(--accent-indigo);">${data.language?.toUpperCase() || "VI"}</div>
                </div>
                <div class="align-metric-card">
                    <div class="align-metric-label">Thời Lượng Audio</div>
                    <div class="align-metric-val" style="color: var(--text-main);">${durFormat}</div>
                </div>
            </div>

            <!-- Audio Master Player Bar -->
            <div style="margin-bottom: 16px; background: rgba(0, 0, 0, 0.2); border: 1px solid var(--border-color); border-radius: var(--radius-sm); padding: 12px 16px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <button type="button" class="btn-primary" style="padding: 6px 14px; font-size: 0.82rem;" onclick="playProcessedAudio('/api/aligned/${data.run_id}/audio.wav', 'Alignment: ${data.input_filename}', '${data.input_filename}')">
                        ▶ Phát Toàn Bộ Track
                    </button>
                    <span style="font-size: 0.8rem; color: var(--text-muted);">💡 Nhấp vào bất kỳ từ nào dưới đây để phát chính xác từ mili-giây đó.</span>
                </div>
            </div>

            <!-- View Mode Switcher -->
            <div class="align-view-nav">
                <button type="button" class="btn-align-view active" onclick="switchAlignView('words', this)">
                    🔤 Xem Từng Từ (Word Badges - Karaoke Player)
                </button>
                <button type="button" class="btn-align-view" onclick="switchAlignView('paragraph', this)">
                    📜 Xem Đoạn Văn (Paragraph Transcript)
                </button>
                <button type="button" class="btn-align-view" onclick="switchAlignView('srt', this)">
                    🎬 Xem Phụ Đề (SRT Subtitles)
                </button>
            </div>

            <!-- View 1: Word Badges Karaoke Container -->
            <div id="alignViewWords" class="karaoke-container">
                ${wordBadgesHtml || `<p style="color: var(--text-dim);">Không có từ nào được tìm thấy.</p>`}
            </div>

            <!-- View 2: Paragraph View Container -->
            <div id="alignViewParagraph" class="paragraph-container hidden">
                ${paragraphsHtml || `<p style="color: var(--text-dim);">Không có nội dung văn bản.</p>`}
            </div>

            <!-- View 3: SRT Code Box Container -->
            <div id="alignViewSrt" class="srt-preview-box hidden">
${srtText || "Chưa có phụ đề SRT"}
            </div>
        </div>
    `;

    // Auto trigger audio player on result ready
    playProcessedAudio(`/api/aligned/${data.run_id}/audio.wav`, `Alignment: ${data.input_filename}`, data.input_filename);
}

function switchAlignView(mode, btnEl) {
    const viewWords = document.getElementById("alignViewWords");
    const viewParagraph = document.getElementById("alignViewParagraph");
    const viewSrt = document.getElementById("alignViewSrt");

    if (viewWords) viewWords.classList.toggle("hidden", mode !== "words");
    if (viewParagraph) viewParagraph.classList.toggle("hidden", mode !== "paragraph");
    if (viewSrt) viewSrt.classList.toggle("hidden", mode !== "srt");

    const btns = document.querySelectorAll(".btn-align-view");
    btns.forEach(b => b.classList.remove("active"));
    if (btnEl) btnEl.classList.add("active");
}

function seekAndPlayAlignWord(startSec) {
    if (!globalAudio) return;
    globalAudio.currentTime = startSec;
    if (globalAudio.paused) {
        globalAudio.play().catch(() => {});
    }
}

function syncKaraokeWordHighlight(currentTime) {
    const wordBadges = document.querySelectorAll("#alignViewWords .word-badge");
    if (!wordBadges || wordBadges.length === 0) return;

    let activeFound = false;
    wordBadges.forEach(badge => {
        const start = parseFloat(badge.getAttribute("data-start") || "0");
        const end = parseFloat(badge.getAttribute("data-end") || "0");

        if (currentTime >= start && currentTime <= end) {
            badge.classList.add("active");
            activeFound = true;
            // Smooth scroll active word into viewport
            badge.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "nearest" });
        } else {
            badge.classList.remove("active");
        }
    });
}

function handoverToStage5(runId, speakerId, masterUrl) {
    // 1. Switch to Tab 5 (Alignment)
    const alignTabBtn = document.querySelector('.tab-btn[data-tab="alignment"]');
    if (alignTabBtn) alignTabBtn.click();

    // 2. Select Diarized radio
    const diarRadio = document.querySelector('input[name="alignSourceType"][value="diarized"]');
    if (diarRadio) {
        diarRadio.checked = true;
        diarRadio.dispatchEvent(new Event("change"));
    }

    // 3. Reload sources and preselect this speaker
    loadAlignmentSources().then(() => {
        if (alignDiarizedSelect) {
            const targetVal = `${runId}__${speakerId}`;
            alignDiarizedSelect.value = targetVal;
        }
        showToast(`Đã chuyển giọng ${speakerId} sang Stage 5 (Word Alignment).`, "success");
    });
}

function handoverToStage6(alignmentRunId) {
    // Switch to Tab 6 (Chunking) when implemented
    const chunkTabBtn = document.querySelector('.tab-btn[data-tab="chunking"]');
    if (chunkTabBtn) chunkTabBtn.click();
    showToast(`Đã chọn bản ghi Alignment ${alignmentRunId} cho Stage 6 (Smart Chunking).`, "success");
}

async function loadAlignmentHistory() {
    if (!alignmentHistoryContainer) return;

    try {
        const res = await fetch("/api/alignment/history");
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Không thể tải lịch sử alignment.");

        const history = data.history || [];
        if (history.length === 0) {
            alignmentHistoryContainer.innerHTML = `
                <div class="empty-state" style="padding: 24px 0;">
                    <p>Chưa có bản ghi Word Alignment nào. Hãy chọn một nguồn audio và bắt đầu gióng hàng.</p>
                </div>
            `;
            return;
        }

        alignmentHistoryContainer.innerHTML = history.map(item => {
            const createdDate = item.created_at ? new Date(item.created_at).toLocaleString("vi-VN") : "Gần đây";
            return `
                <div class="history-card">
                    <div class="history-card-info">
                        <div class="history-card-title">
                            <span>⏱️ ${item.input_filename}</span>
                            <span class="badge badge-accent">${item.model_size || "large-v3"}</span>
                            <span class="badge" style="background: rgba(16, 185, 129, 0.15); color: #10b981;">${item.language?.toUpperCase() || "VI"}</span>
                        </div>
                        <div class="history-card-meta">
                            <span>${createdDate} | ${item.total_words || 0} từ | ${item.total_segments || 0} câu | ${item.total_duration_formatted || "0s"}</span>
                        </div>
                    </div>

                    <div class="history-card-actions">
                        <button type="button" class="btn-secondary" style="font-size: 0.78rem; padding: 6px 12px;" onclick='renderAlignmentResults(${JSON.stringify(item)})'>
                            🔍 Xem Chi Tiết
                        </button>
                        <a href="/api/aligned/${item.run_id}/subtitles.srt" download="subtitles_${item.run_id}.srt" class="btn-secondary" style="font-size: 0.78rem; padding: 6px 12px; text-decoration: none;">
                            📥 SRT
                        </a>
                        <button type="button" class="btn-delete" style="font-size: 0.78rem; padding: 6px 12px;" onclick="deleteAlignmentRun('${item.run_id}')">
                            🗑️ Xóa
                        </button>
                    </div>
                </div>
            `;
        }).join("");
    } catch (err) {
        alignmentHistoryContainer.innerHTML = `<div class="empty-state"><p style="color: #f43f5e;">Lỗi tải lịch sử: ${err.message}</p></div>`;
    }
}

async function deleteAlignmentRun(runId) {
    if (!confirm(`Bạn có chắc chắn muốn xóa bản ghi Word Alignment ${runId}?`)) return;

    try {
        const res = await fetch(`/api/alignment/${runId}`, { method: "DELETE" });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Xóa bản ghi thất bại.");

        showToast(data.message || "Đã xóa bản ghi Word Alignment thành công.", "success");
        await loadAlignmentHistory();
    } catch (err) {
        showToast(err.message, "error");
    }
}

// Initialize on page load
document.addEventListener("DOMContentLoaded", () => {
    initTheme();
    loadAudioLibrary();
    loadSeparationModelStatus();
    initDiarizationState();
    initAlignmentState();
});


