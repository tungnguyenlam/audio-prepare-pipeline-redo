/**
 * SonicStudio — Audio Preparation & Separation Studio Web Application
 * Modular ES6 Vanilla JavaScript Frontend Architecture
 */

// ==================== STATE MANAGEMENT ====================

const state = {
  activeAudio: null,       // Currently selected Audio metadata
  activePeaks: [],         // Downsampled waveform peaks
  selection: { start: 0, end: 0, active: false },
  zoom: 1.0,
  audioList: [],           // All registered Audio items
  serverFiles: [],         // Files on disk from /api/library
  systemStatus: null,

  // Live A/B state
  ab: {
    trackAId: null,
    trackBId: null,
    currentTrack: 'A',     // 'A' or 'B'
  },

  // Master Player State
  player: {
    isPlaying: false,
    duration: 0,
    currentTime: 0,
    volume: 1.0,
    playbackRate: 1.0,
    loop: false,
    previewEnd: null,      // Used for previewing cuts
    showRemainingTime: false,
  },
};

// DOM Elements Cache
const el = {
  // Navigation
  tabs: document.querySelectorAll('.nav-tab'),
  tabPanes: document.querySelectorAll('.tab-pane'),
  deviceLabel: document.getElementById('device-label'),
  btnThemeToggle: document.getElementById('btn-theme-toggle'),
  iconThemeSun: document.getElementById('icon-theme-sun'),
  iconThemeMoon: document.getElementById('icon-theme-moon'),
  btnOpenShortcutsModal: document.getElementById('btn-open-shortcuts-modal'),

  // Master Player
  audio: document.getElementById('global-audio-element'),
  btnPlayPause: document.getElementById('btn-player-play-pause'),
  iconPlay: document.getElementById('icon-play'),
  iconPause: document.getElementById('icon-pause'),
  btnSkipBack: document.getElementById('btn-player-skip-back'),
  btnSkipFwd: document.getElementById('btn-player-skip-fwd'),
  btnLoop: document.getElementById('btn-player-loop'),
  btnMute: document.getElementById('btn-player-mute'),
  iconVol: document.getElementById('icon-vol'),
  iconVolMute: document.getElementById('icon-vol-mute'),
  volumeSlider: document.getElementById('player-volume-slider'),
  speedSelect: document.getElementById('player-speed-select'),
  scrubWrapper: document.getElementById('player-scrub-wrapper'),
  scrubProgress: document.getElementById('player-scrub-progress'),
  scrubHandle: document.getElementById('player-scrub-handle'),
  timeCurrent: document.getElementById('player-time-current'),
  timeTotal: document.getElementById('player-time-total'),
  playerTitle: document.getElementById('player-track-title'),
  playerSub: document.getElementById('player-track-sub'),

  // Workspace
  dropzone: document.getElementById('audio-dropzone'),
  fileInput: document.getElementById('file-input'),
  ytUrlInput: document.getElementById('yt-url-input'),
  btnYtDownload: document.getElementById('btn-yt-download'),
  btnYtPasteWorkspace: document.getElementById('btn-yt-paste-workspace'),
  btnBrowseLibrary: document.getElementById('btn-browse-library-modal'),
  activeSection: document.getElementById('active-audio-section'),
  metaSourceType: document.getElementById('meta-source-type'),
  metaTitle: document.getElementById('meta-title'),
  metaId: document.getElementById('meta-id'),
  metaDuration: document.getElementById('meta-duration'),
  metaSampleRate: document.getElementById('meta-sample-rate'),
  metaNativeRate: document.getElementById('meta-native-rate'),
  metaChannels: document.getElementById('meta-channels'),
  metaFormat: document.getElementById('meta-format'),
  metaSize: document.getElementById('meta-size'),
  metaFingerprint: document.getElementById('meta-fingerprint'),
  historyTagsList: document.getElementById('history-tags-list'),
  btnQuickSave: document.getElementById('btn-quick-save'),
  btnSaveToDialog: document.getElementById('btn-save-to-dialog'),
  btnDownloadAudio: document.getElementById('btn-download-audio'),
  btnSendToSep: document.getElementById('btn-send-to-separation'),

  // Waveform & Canvas
  waveformViewport: document.getElementById('waveform-viewport'),
  waveformCanvas: document.getElementById('waveform-canvas'),
  playheadLine: document.getElementById('playhead-line'),
  selectionOverlay: document.getElementById('selection-overlay'),
  selectionRangeLabel: document.getElementById('selection-range-label'),
  selectionActionsBar: document.getElementById('selection-actions-bar'),
  btnAuditionSelection: document.getElementById('btn-audition-selection'),
  btnClearSelection: document.getElementById('btn-clear-selection'),
  timeTooltip: document.getElementById('waveform-time-tooltip'),
  rulerMid: document.getElementById('ruler-mid'),
  rulerEnd: document.getElementById('ruler-end'),
  btnZoomIn: document.getElementById('btn-zoom-in'),
  btnZoomOut: document.getElementById('btn-zoom-out'),
  btnResetZoom: document.getElementById('btn-reset-zoom'),
  zoomLabel: document.getElementById('zoom-level-label'),
  btnToggleSpec: document.getElementById('btn-toggle-spectrogram'),
  spectrogramPanel: document.getElementById('spectrogram-panel'),
  specImage: document.getElementById('spec-image'),
  specLoader: document.getElementById('spec-loader'),
  btnRefreshSpec: document.getElementById('btn-refresh-spec'),

  // Audio Cutter
  cutStartInput: document.getElementById('cut-start-input'),
  cutEndInput: document.getElementById('cut-end-input'),
  btnUseSelection: document.getElementById('btn-use-selection'),
  btnPreviewCut: document.getElementById('btn-preview-cut'),
  btnApplyCut: document.getElementById('btn-apply-cut'),
  cutUnitRadios: document.querySelectorAll('input[name="cut_unit"]'),

  // YouTube Crawler Studio
  ytTabUrlInput: document.getElementById('yt-tab-url-input'),
  btnYtPasteTab: document.getElementById('btn-yt-paste-tab'),
  btnYtTabInspect: document.getElementById('btn-yt-tab-inspect'),
  btnYtTabDownload: document.getElementById('btn-yt-tab-download'),
  ytSampleRateSelect: document.getElementById('yt-sample-rate-select'),
  ytFormatSelect: document.getElementById('yt-format-select'),
  ytTaskProgressBox: document.getElementById('yt-task-progress-box'),
  ytTaskTitle: document.getElementById('yt-task-title'),
  ytTaskTimer: document.getElementById('yt-task-timer'),
  ytProgressBar: document.getElementById('yt-progress-bar'),
  ytTaskStatusText: document.getElementById('yt-task-status-text'),
  ytPreviewCard: document.getElementById('yt-preview-card'),
  ytPreviewThumb: document.getElementById('yt-preview-thumb'),
  ytPreviewDuration: document.getElementById('yt-preview-duration'),
  ytPreviewTitle: document.getElementById('yt-preview-title'),
  ytPreviewChannel: document.getElementById('yt-preview-channel'),
  ytPreviewViews: document.getElementById('yt-preview-views'),
  ytPreviewDesc: document.getElementById('yt-preview-desc'),
  ytPreviewLink: document.getElementById('yt-preview-link'),
  ytEmptyPlaceholder: document.getElementById('yt-empty-placeholder'),
  ytInspectBadge: document.getElementById('yt-inspect-badge'),
  ytVaultList: document.getElementById('yt-vault-list'),
  btnRefreshYtHistory: document.getElementById('btn-refresh-yt-history'),

  // Separation Studio
  sepInputSelect: document.getElementById('sep-input-select'),
  modelCards: document.querySelectorAll('.model-card[data-model]'),
  sepDeviceSelect: document.getElementById('sep-device-select'),
  sepStemsSelect: document.getElementById('sep-stems-select'),
  roformerPresetGroup: document.getElementById('roformer-preset-group'),
  roformerCheckpointInput: document.getElementById('roformer-checkpoint-input'),
  btnRunSeparation: document.getElementById('btn-run-separation'),
  sepTaskProgressBox: document.getElementById('sep-task-progress-box'),
  sepTaskTitle: document.getElementById('sep-task-title'),
  sepTaskTimer: document.getElementById('sep-task-timer'),
  sepProgressBar: document.getElementById('sep-progress-bar'),
  sepTaskStatusText: document.getElementById('sep-task-status-text'),
  sepResultsList: document.getElementById('sep-results-list'),

  // Diarization Studio
  diarInputSelect: document.getElementById('diar-input-select'),
  diarModelCards: document.querySelectorAll('.model-card[data-diar-model]'),
  hfTokenInput: document.getElementById('hf-token-input'),
  diarDeviceSelect: document.getElementById('diar-device-select'),
  btnRunDiarization: document.getElementById('btn-run-diarization'),
  diarTaskProgressBox: document.getElementById('diar-task-progress-box'),
  diarTaskTimer: document.getElementById('diar-task-timer'),
  diarProgressBar: document.getElementById('diar-progress-bar'),
  diarTaskStatusText: document.getElementById('diar-task-status-text'),
  diarResultsWrapper: document.getElementById('diar-results-wrapper'),
  diarEmptyPlaceholder: document.getElementById('diar-empty-placeholder'),
  speakersSummaryRow: document.getElementById('speakers-summary-row'),
  diarTimeline: document.getElementById('diar-timeline'),
  turnsTableBody: document.getElementById('turns-table-body'),
  diarModelBadge: document.getElementById('diar-model-badge'),

  // Comparison Studio
  compareTrackASelect: document.getElementById('compare-track-a-select'),
  compareTrackBSelect: document.getElementById('compare-track-b-select'),
  btnGenerateComparison: document.getElementById('btn-generate-comparison'),
  abActiveLabel: document.getElementById('ab-active-label'),
  btnAbSwitchA: document.getElementById('btn-ab-switch-a'),
  btnAbSwitchB: document.getElementById('btn-ab-switch-b'),
  btnAbToggleInstant: document.getElementById('btn-ab-toggle-instant'),
  imgCompareSpectrogram: document.getElementById('img-compare-spectrogram'),
  imgCompareWaveform: document.getElementById('img-compare-waveform'),
  spectrogramCompareBox: document.getElementById('spectrogram-compare-box'),
  waveformCompareBox: document.getElementById('waveform-compare-box'),

  // Benchmark Mixer
  mixerSpeechSelect: document.getElementById('mixer-speech-select'),
  mixerMusicSelect: document.getElementById('mixer-music-select'),
  mixerSmrSlider: document.getElementById('mixer-smr-slider'),
  mixerSmrVal: document.getElementById('mixer-smr-val'),
  mixerSeedInput: document.getElementById('mixer-seed-input'),
  btnRunMixer: document.getElementById('btn-run-mixer'),
  mixerResultBox: document.getElementById('mixer-result-box'),
  mixStemsGrid: document.getElementById('mix-stems-grid'),

  // Library & History
  serverFilesList: document.getElementById('server-files-list'),
  btnRefreshLibrary: document.getElementById('btn-refresh-library'),
  sessionHistoryList: document.getElementById('session-history-list'),
  sessionCountBadge: document.getElementById('session-count-badge'),

  // Modals & Toasts
  modalSaveTo: document.getElementById('modal-save-to'),
  inputSavePath: document.getElementById('input-save-path'),
  btnCancelSave: document.getElementById('btn-cancel-save'),
  btnConfirmSave: document.getElementById('btn-confirm-save'),
  btnCloseSaveModal: document.getElementById('btn-close-save-modal'),
  modalLibrary: document.getElementById('modal-library'),
  modalLibraryItems: document.getElementById('modal-library-items'),
  btnCloseLibraryModal: document.getElementById('btn-close-library-modal'),
  modalShortcuts: document.getElementById('modal-shortcuts'),
  btnCloseShortcutsModal: document.getElementById('btn-close-shortcuts-modal'),
  toastContainer: document.getElementById('toast-container'),
};

// ==================== UTILITY FUNCTIONS ====================

function formatTime(seconds) {
  if (isNaN(seconds) || seconds < 0) return "00:00";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}

function formatTimePrecise(seconds) {
  if (isNaN(seconds) || seconds < 0) return "00:00.00";
  const m = Math.floor(seconds / 60);
  const sInt = Math.floor(seconds % 60).toString().padStart(2, '0');
  const sFrac = (seconds % 1).toFixed(2).substring(2);
  return `${m.toString().padStart(2, '0')}:${sInt}.${sFrac}`;
}

function formatBytes(bytes) {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
}

function showToast(message, type = "info") {
  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  
  let icon = 'ℹ️';
  if (type === 'success') icon = '✓';
  if (type === 'error') icon = '✕';
  if (type === 'warning') icon = '⚠️';

  toast.innerHTML = `<span style="font-weight: 700;">${icon}</span><span>${message}</span>`;
  el.toastContainer.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateX(100%)";
    setTimeout(() => toast.remove(), 250);
  }, 3800);
}

// ==================== MASTER AUDIO PLAYER ENGINE ====================

function initPlayer() {
  el.btnPlayPause.addEventListener('click', togglePlayPause);
  el.btnSkipBack.addEventListener('click', () => seekRelative(-5));
  el.btnSkipFwd.addEventListener('click', () => seekRelative(5));

  el.btnLoop.addEventListener('click', () => {
    state.player.loop = !state.player.loop;
    el.audio.loop = state.player.loop;
    el.btnLoop.classList.toggle('active', state.player.loop);
    showToast(state.player.loop ? "Loop playback enabled" : "Loop playback disabled", "info");
  });

  el.speedSelect.addEventListener('change', (e) => {
    state.player.playbackRate = parseFloat(e.target.value);
    el.audio.playbackRate = state.player.playbackRate;
  });

  el.volumeSlider.addEventListener('input', (e) => {
    state.player.volume = parseFloat(e.target.value);
    el.audio.volume = state.player.volume;
    updateVolumeIcon();
  });

  el.btnMute.addEventListener('click', () => {
    if (el.audio.volume > 0) {
      el.audio.volume = 0;
      el.volumeSlider.value = 0;
    } else {
      el.audio.volume = state.player.volume || 1.0;
      el.volumeSlider.value = el.audio.volume;
    }
    updateVolumeIcon();
  });

  // Toggle remaining time vs total duration on click
  el.timeTotal.addEventListener('click', () => {
    state.player.showRemainingTime = !state.player.showRemainingTime;
    onTimeUpdate();
  });

  // Scrub bar interaction
  el.scrubWrapper.addEventListener('click', (e) => {
    if (!state.player.duration) return;
    const rect = el.scrubWrapper.getBoundingClientRect();
    const pos = (e.clientX - rect.left) / rect.width;
    const seekTime = Math.max(0, Math.min(pos * state.player.duration, state.player.duration));
    seekTo(seekTime);
  });

  // Audio element events
  el.audio.addEventListener('timeupdate', onTimeUpdate);
  el.audio.addEventListener('loadedmetadata', onLoadedMetadata);
  el.audio.addEventListener('ended', onEnded);
  el.audio.addEventListener('play', () => setPlayingUI(true));
  el.audio.addEventListener('pause', () => setPlayingUI(false));
  el.audio.addEventListener('error', () => {
    const mediaError = el.audio.error;
    const message = mediaError?.code === MediaError.MEDIA_ERR_SRC_NOT_SUPPORTED
      ? "This audio format is not supported by your browser"
      : "The audio stream could not be loaded";
    showToast(message, "error");
    setPlayingUI(false);
  });

  // Global keyboard shortcuts engine (DAW Style)
  window.addEventListener('keydown', handleGlobalKeydown);
}

function handleGlobalKeydown(e) {
  // If typing inside an input, textarea or select, ignore hotkeys
  if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;

  // Space: Play / Pause (or A/B switch if on comparison tab)
  if (e.code === 'Space') {
    e.preventDefault();
    const activeTab = document.querySelector('.nav-tab.active')?.dataset.tab;
    if (activeTab === 'tab-comparison' && state.ab.trackAId && state.ab.trackBId) {
      toggleABInstant();
    } else {
      togglePlayPause();
    }
    return;
  }

  // Arrow Left / Right Seek
  if (e.code === 'ArrowLeft') {
    e.preventDefault();
    seekRelative(e.shiftKey ? -5 : -2);
    return;
  }
  if (e.code === 'ArrowRight') {
    e.preventDefault();
    seekRelative(e.shiftKey ? 5 : 2);
    return;
  }

  // J / K / L Seek & Pause
  if (e.key === 'j' || e.key === 'J') {
    seekRelative(-2);
    return;
  }
  if (e.key === 'k' || e.key === 'K') {
    if (!el.audio.paused) el.audio.pause();
    return;
  }
  if (e.key === 'l' || e.key === 'L') {
    seekRelative(2);
    return;
  }

  // [ and ]: Set Cut Start / End to current playhead
  if (e.key === '[') {
    if (state.activeAudio) {
      el.cutStartInput.value = (el.audio.currentTime || 0).toFixed(2);
      showToast(`Set Cut Start bound to ${el.cutStartInput.value}s`, "info");
    }
    return;
  }
  if (e.key === ']') {
    if (state.activeAudio) {
      el.cutEndInput.value = (el.audio.currentTime || 0).toFixed(2);
      showToast(`Set Cut End bound to ${el.cutEndInput.value}s`, "info");
    }
    return;
  }

  // C: Apply Cut
  if (e.key === 'c' || e.key === 'C') {
    if (!e.metaKey && !e.ctrlKey && state.activeAudio && !el.btnApplyCut.disabled) {
      el.btnApplyCut.click();
    }
    return;
  }

  // M: Toggle Mute
  if (e.key === 'm' || e.key === 'M') {
    if (!e.metaKey && !e.ctrlKey) {
      el.btnMute.click();
    }
    return;
  }

  // Z / Shift+Z / 0: Zoom
  if (e.key === 'z' || e.key === 'Z') {
    if (e.shiftKey) {
      setZoom(Math.max(1.0, state.zoom / 1.5));
    } else {
      setZoom(state.zoom * 1.5);
    }
    return;
  }
  if (e.key === '0') {
    setZoom(1.0);
    return;
  }

  // 1 through 7: Tab navigation
  if (e.key >= '1' && e.key <= '7') {
    const tabIndex = parseInt(e.key) - 1;
    if (el.tabs[tabIndex]) {
      switchTab(el.tabs[tabIndex].dataset.tab);
    }
    return;
  }

  // ?: Toggle Shortcuts Modal
  if (e.key === '?' || (e.shiftKey && e.key === '/')) {
    toggleShortcutsModal();
    return;
  }

  // Escape: Close open modals / clear selection
  if (e.code === 'Escape') {
    closeAllModals();
    if (state.selection.active) {
      clearSelection();
    }
    return;
  }
}

function updateVolumeIcon() {
  const isMuted = el.audio.volume === 0;
  el.iconVol.classList.toggle('hidden', isMuted);
  el.iconVolMute.classList.toggle('hidden', !isMuted);
}

function setPlayingUI(isPlaying) {
  state.player.isPlaying = isPlaying;
  el.iconPlay.classList.toggle('hidden', isPlaying);
  el.iconPause.classList.toggle('hidden', !isPlaying);
}

function togglePlayPause() {
  if (!el.audio.src) {
    showToast("Load an audio track first", "info");
    return;
  }
  if (el.audio.paused) {
    playCurrentAudio();
  } else {
    el.audio.pause();
  }
}

function playCurrentAudio() {
  return el.audio.play().catch(err => {
    console.error("Play error:", err);
    if (err.name === "NotAllowedError") {
      showToast("Your browser blocked autoplay. Press Play to start audio.", "warning");
    } else {
      showToast(`Unable to play audio: ${err.message || "unknown playback error"}`, "error");
    }
  });
}

function seekTo(time) {
  if (!state.player.duration) return;
  el.audio.currentTime = time;
  updatePlayheadPosition(time);
}

function seekRelative(offset) {
  if (!state.player.duration) return;
  seekTo(Math.max(0, Math.min(el.audio.currentTime + offset, state.player.duration)));
}

function onLoadedMetadata() {
  state.player.duration = el.audio.duration || (state.activeAudio ? state.activeAudio.duration_s : 0);
  el.timeTotal.textContent = formatTime(state.player.duration);
  el.rulerEnd.textContent = formatTime(state.player.duration);
  el.rulerMid.textContent = formatTime(state.player.duration / 2);
}

function onTimeUpdate() {
  const cur = el.audio.currentTime;
  const dur = state.player.duration || 1;
  state.player.currentTime = cur;

  // Check if we hit cut preview boundary
  if (state.player.previewEnd !== null && cur >= state.player.previewEnd) {
    el.audio.pause();
    state.player.previewEnd = null;
  }

  el.timeCurrent.textContent = formatTime(cur);

  if (state.player.showRemainingTime) {
    const rem = Math.max(0, dur - cur);
    el.timeTotal.textContent = `-${formatTime(rem)}`;
  } else {
    el.timeTotal.textContent = formatTime(dur);
  }

  const pct = Math.min(100, Math.max(0, (cur / dur) * 100));
  el.scrubProgress.style.width = `${pct}%`;
  updatePlayheadPosition(cur);
}

function onEnded() {
  if (!state.player.loop) {
    setPlayingUI(false);
    seekTo(0);
  }
}

function loadAudioIntoPlayer(audioId, autoplay = false) {
  const item = state.audioList.find(a => a.id === audioId)
    || (state.activeAudio?.id === audioId ? state.activeAudio : null);

  el.audio.src = `/api/audio/${audioId}/stream`;
  el.audio.load();
  el.playerTitle.textContent = item?.title || item?.source_id || audioId;
  el.playerSub.textContent = item
    ? `${(item.format || "audio").toUpperCase()} • ${(item.sample_rate || 0).toLocaleString()}Hz • ${item.channels === 1 ? 'Mono' : 'Stereo'} • ID: ${audioId}`
    : `Audio ID: ${audioId}`;

  if (autoplay) {
    playCurrentAudio();
  }
}

// ==================== ACTIVE AUDIO & WORKSPACE ====================

async function setActiveAudio(audioId, options = { play: false }) {
  try {
    const res = await fetch(`/api/audio/${audioId}`);
    if (!res.ok) throw new Error("Failed to get audio metadata");
    const meta = await res.json();
    
    // Find registered wrapper
    const fullItem = state.audioList.find(a => a.id === audioId) || meta;
    state.activeAudio = { ...meta, id: audioId, source_type: fullItem.source_type || "local" };
    try {
      localStorage.setItem('sonic_active_audio_id', audioId);
    } catch (_) {}

    // Update Workspace UI
    el.activeSection.classList.remove('hidden');
    el.metaTitle.textContent = meta.title || meta.source_id;
    el.metaId.textContent = audioId;
    el.metaSourceType.textContent = fullItem.source_type || "Audio";
    el.metaDuration.textContent = `${(meta.duration_s || 0).toFixed(2)}s`;
    el.metaSampleRate.textContent = `${meta.sample_rate.toLocaleString()} Hz`;
    el.metaNativeRate.textContent = `${(meta.native_sample_rate || meta.sample_rate).toLocaleString()} Hz`;
    el.metaChannels.textContent = meta.channels === 1 ? "1 (Mono)" : `${meta.channels} (Stereo)`;
    el.metaFormat.textContent = (meta.format || "WAV").toUpperCase();
    el.metaSize.textContent = formatBytes(fullItem.file_size || 0);
    el.metaFingerprint.textContent = meta.fingerprint || "none";

    // Step history tags
    el.historyTagsList.innerHTML = "";
    const history = meta.history || [];
    if (history.length === 0) {
      el.historyTagsList.innerHTML = `<span class="history-tag base">raw_input</span>`;
    } else {
      history.forEach(tag => {
        const span = document.createElement("span");
        span.className = "history-tag";
        span.textContent = tag;
        el.historyTagsList.appendChild(span);
      });
    }

    // Set default cut bounds
    el.cutEndInput.value = (meta.duration_s || 10).toFixed(1);
    el.cutStartInput.value = "0.0";

    // Load into master player
    loadAudioIntoPlayer(audioId, options.play);

    // Fetch Waveform Peaks
    await loadWaveform(audioId);

    // Sync all dropdown selectors across tabs
    populateAllAudioSelects();

    // Reset spectrogram preview
    el.specImage.classList.add('hidden');
    el.specLoader.classList.remove('hidden');
    el.spectrogramPanel.classList.add('hidden');

    // Clear any previous selection
    clearSelection();

  } catch (err) {
    console.error("Error setting active audio:", err);
    showToast(err.message, "error");
  }
}

async function loadWaveform(audioId) {
  try {
    const res = await fetch(`/api/audio/${audioId}/waveform`);
    if (!res.ok) throw new Error("Failed to load waveform");
    const data = await res.json();
    state.activePeaks = data.peaks || [];
    renderWaveform();
  } catch (err) {
    console.error("Waveform load error:", err);
  }
}

// ==================== WAVEFORM CANVAS RENDERER ====================

function renderWaveform() {
  const canvas = el.waveformCanvas;
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.parentElement.getBoundingClientRect();
  const width = rect.width || 1200;
  const height = rect.height || 180;

  canvas.width = Math.floor(width * dpr);
  canvas.height = Math.floor(height * dpr);
  ctx.scale(dpr, dpr);

  ctx.clearRect(0, 0, width, height);

  const peaks = state.activePeaks;
  if (!peaks || peaks.length === 0) {
    ctx.fillStyle = "rgba(148, 163, 184, 0.6)";
    ctx.font = "12px JetBrains Mono";
    ctx.textAlign = "center";
    ctx.fillText("No waveform data loaded", width / 2, height / 2);
    return;
  }

  const numBars = peaks.length;
  const barWidth = Math.max(1.2, (width / numBars) * state.zoom);
  const centerY = height / 2;

  const isLight = document.documentElement.getAttribute('data-theme') === 'light';

  // Zero-crossing baseline
  ctx.strokeStyle = isLight ? "rgba(148, 163, 184, 0.4)" : "rgba(255, 255, 255, 0.1)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, centerY);
  ctx.lineTo(width, centerY);
  ctx.stroke();

  // Waveform vertical gradient
  const grad = ctx.createLinearGradient(0, 0, 0, height);
  if (isLight) {
    grad.addColorStop(0, "hsl(217, 91%, 50%)");
    grad.addColorStop(0.5, "hsl(190, 90%, 42%)");
    grad.addColorStop(1, "hsl(217, 91%, 50%)");
  } else {
    grad.addColorStop(0, "hsl(188, 86%, 56%)");
    grad.addColorStop(0.5, "hsl(217, 91%, 64%)");
    grad.addColorStop(1, "hsl(188, 86%, 50%)");
  }

  ctx.fillStyle = grad;

  for (let i = 0; i < numBars; i++) {
    const x = i * barWidth;
    if (x > width) break;
    const amp = peaks[i] || 0.01;
    const barHeight = Math.max(2, amp * (centerY - 10));

    // Draw symmetrical waveform bar
    ctx.fillRect(x, centerY - barHeight, Math.max(1, barWidth - 0.5), barHeight * 2);
  }
}

function updatePlayheadPosition(currentTime) {
  if (!state.player.duration || !el.waveformViewport) return;
  const pct = currentTime / state.player.duration;
  const width = el.waveformViewport.clientWidth;
  const pos = pct * width;
  el.playheadLine.style.transform = `translateX(${pos}px)`;
}

function clearSelection() {
  state.selection.active = false;
  state.selection.start = 0;
  state.selection.end = 0;
  if (el.selectionOverlay) el.selectionOverlay.classList.add('hidden');
  if (el.selectionActionsBar) el.selectionActionsBar.style.display = 'none';
}

function initWaveformInteractions() {
  let isDragging = false;
  let dragStartX = 0;

  const viewport = el.waveformViewport;

  viewport.addEventListener('mousedown', (e) => {
    if (!state.activeAudio) return;
    const rect = viewport.getBoundingClientRect();
    dragStartX = e.clientX - rect.left;
    isDragging = true;

    const time = (dragStartX / rect.width) * (state.activeAudio.duration_s || 1);
    seekTo(time);

    state.selection.start = time;
    state.selection.end = time;
    state.selection.active = true;
    updateSelectionOverlay(dragStartX, dragStartX, rect.width);
  });

  window.addEventListener('mousemove', (e) => {
    if (!state.activeAudio) return;
    const rect = viewport.getBoundingClientRect();
    const currentX = Math.max(0, Math.min(e.clientX - rect.left, rect.width));

    // Show hover time tooltip
    if (e.clientX >= rect.left && e.clientX <= rect.right && e.clientY >= rect.top && e.clientY <= rect.bottom) {
      const hoverTime = (currentX / rect.width) * (state.activeAudio.duration_s || 1);
      el.timeTooltip.classList.remove('hidden');
      el.timeTooltip.textContent = formatTimePrecise(hoverTime);
      el.timeTooltip.style.left = `${Math.min(currentX, rect.width - 60)}px`;
    } else {
      el.timeTooltip.classList.add('hidden');
    }

    if (!isDragging) return;

    const minX = Math.min(dragStartX, currentX);
    const maxX = Math.max(dragStartX, currentX);

    updateSelectionOverlay(minX, maxX, rect.width);

    const dur = state.activeAudio.duration_s || 1;
    state.selection.start = (minX / rect.width) * dur;
    state.selection.end = (maxX / rect.width) * dur;
  });

  window.addEventListener('mouseup', () => {
    if (isDragging) {
      isDragging = false;
      // If minimal drag, clear selection
      if (Math.abs(state.selection.end - state.selection.start) < 0.05) {
        clearSelection();
      } else {
        if (el.selectionActionsBar) el.selectionActionsBar.style.display = 'flex';
      }
    }
  });

  // Audition Selection button
  if (el.btnAuditionSelection) {
    el.btnAuditionSelection.addEventListener('click', () => {
      if (!state.activeAudio || !state.selection.active) return;
      seekTo(state.selection.start);
      state.player.previewEnd = state.selection.end;
      el.audio.play();
      showToast(`Auditioning selection: ${state.selection.start.toFixed(2)}s to ${state.selection.end.toFixed(2)}s`, "info");
    });
  }

  // Clear Selection button
  if (el.btnClearSelection) {
    el.btnClearSelection.addEventListener('click', clearSelection);
  }

  // Zoom controls
  el.btnZoomIn.addEventListener('click', () => setZoom(state.zoom * 1.5));
  el.btnZoomOut.addEventListener('click', () => setZoom(Math.max(1.0, state.zoom / 1.5)));
  el.btnResetZoom.addEventListener('click', () => setZoom(1.0));

  // Spectrogram Toggle
  el.btnToggleSpec.addEventListener('click', toggleSpectrogramPanel);
  el.btnRefreshSpec.addEventListener('click', loadSpectrogramImage);

  window.addEventListener('resize', renderWaveform);
}

function setZoom(newZoom) {
  state.zoom = Math.min(8.0, Math.max(1.0, newZoom));
  el.zoomLabel.textContent = `${Math.round(state.zoom * 100)}%`;
  renderWaveform();
  updatePlayheadPosition(state.player.currentTime);
}

function updateSelectionOverlay(minX, maxX, totalWidth) {
  el.selectionOverlay.classList.remove('hidden');
  el.selectionOverlay.style.left = `${minX}px`;
  el.selectionOverlay.style.width = `${maxX - minX}px`;

  const dur = state.activeAudio.duration_s || 1;
  const tStart = (minX / totalWidth) * dur;
  const tEnd = (maxX / totalWidth) * dur;
  el.selectionRangeLabel.textContent = `${tStart.toFixed(2)}s – ${tEnd.toFixed(2)}s (${(tEnd - tStart).toFixed(2)}s)`;
}

async function toggleSpectrogramPanel() {
  const isHidden = el.spectrogramPanel.classList.toggle('hidden');
  if (!isHidden && el.specImage.classList.contains('hidden')) {
    await loadSpectrogramImage();
  }
}

async function loadSpectrogramImage() {
  if (!state.activeAudio) return;
  el.specLoader.classList.remove('hidden');
  el.specImage.classList.add('hidden');

  try {
    el.specImage.src = `/api/audio/${state.activeAudio.id}/spectrogram?t=${Date.now()}`;
    el.specImage.onload = () => {
      el.specLoader.classList.add('hidden');
      el.specImage.classList.remove('hidden');
    };
  } catch (err) {
    el.specLoader.textContent = "Failed to load spectrogram.";
  }
}

// ==================== AUDIO CUTTER ACTIONS ====================

function initAudioCutter() {
  // Use Selection bounds
  el.btnUseSelection.addEventListener('click', () => {
    if (!state.selection.active) {
      showToast("Select a region on the waveform first", "info");
      return;
    }
    const unit = document.querySelector('input[name="cut_unit"]:checked').value;
    const dur = state.activeAudio.duration_s || 1;

    if (unit === 'seconds') {
      el.cutStartInput.value = state.selection.start.toFixed(2);
      el.cutEndInput.value = state.selection.end.toFixed(2);
    } else if (unit === 'minutes') {
      el.cutStartInput.value = (state.selection.start / 60).toFixed(3);
      el.cutEndInput.value = (state.selection.end / 60).toFixed(3);
    } else if (unit === 'percent') {
      el.cutStartInput.value = ((state.selection.start / dur) * 100).toFixed(1);
      el.cutEndInput.value = ((state.selection.end / dur) * 100).toFixed(1);
    } else if (unit === 'timestamp') {
      el.cutStartInput.value = formatTime(state.selection.start);
      el.cutEndInput.value = formatTime(state.selection.end);
    }
    showToast("Populated bounds from waveform selection", "success");
  });

  // Unit radio change styling
  el.cutUnitRadios.forEach(radio => {
    radio.addEventListener('change', () => {
      document.querySelectorAll('.radio-pill').forEach(p => p.classList.remove('active'));
      radio.closest('.radio-pill').classList.add('active');
    });
  });

  // Preview Cut
  el.btnPreviewCut.addEventListener('click', () => {
    if (!state.activeAudio) return;
    const startVal = parseFloat(el.cutStartInput.value) || 0;
    const endVal = parseFloat(el.cutEndInput.value) || (state.activeAudio.duration_s || 10);

    seekTo(startVal);
    state.player.previewEnd = endVal;
    el.audio.play();
    showToast(`Previewing cut: ${startVal}s to ${endVal}s`, "info");
  });

  // Apply Cut (AudioCutter API)
  el.btnApplyCut.addEventListener('click', async () => {
    if (!state.activeAudio) return;
    const start = el.cutStartInput.value.trim();
    const end = el.cutEndInput.value.trim();
    const unit = document.querySelector('input[name="cut_unit"]:checked').value;

    el.btnApplyCut.disabled = true;
    el.btnApplyCut.textContent = "Cutting...";

    try {
      const res = await fetch(`/api/audio/${state.activeAudio.id}/cut`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ start, end, unit }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Cut operation failed");

      showToast(`Audio cut successful! Created new clip ${data.audio_id}`, "success");
      await fetchAudioList();
      await setActiveAudio(data.audio_id, { play: true });
    } catch (err) {
      showToast(err.message, "error");
    } finally {
      el.btnApplyCut.disabled = false;
      el.btnApplyCut.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="6" cy="6" r="3"></circle><circle cx="6" cy="18" r="3"></circle><line x1="20" y1="4" x2="8.12" y2="15.88"></line><line x1="14.47" y1="14.48" x2="20" y2="20"></line><line x1="8.12" y1="8.12" x2="12" y2="12"></line></svg> <span>Apply Cut (AudioCutter)</span>`;
    }
  });
}

// ==================== INGEST & SAVE ACTIONS ====================

function initIngestAndSaves() {
  // Drag & Drop
  el.dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    el.dropzone.classList.add('dragover');
  });
  el.dropzone.addEventListener('dragleave', () => el.dropzone.classList.remove('dragover'));
  el.dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    el.dropzone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
      uploadFile(e.dataTransfer.files[0]);
    }
  });
  el.fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
      uploadFile(e.target.files[0]);
    }
  });

  // Paste from clipboard in workspace
  if (el.btnYtPasteWorkspace) {
    el.btnYtPasteWorkspace.addEventListener('click', async () => {
      try {
        const text = await navigator.clipboard.readText();
        if (text) {
          el.ytUrlInput.value = text.trim();
          showToast("Pasted link from clipboard", "info");
        }
      } catch (_) {
        showToast("Clipboard access denied or empty", "warning");
      }
    });
  }

  // Copy Audio ID and Fingerprint
  if (el.metaId) {
    el.metaId.addEventListener('click', () => {
      if (state.activeAudio) {
        navigator.clipboard.writeText(state.activeAudio.id);
        showToast(`Copied Audio ID '${state.activeAudio.id}' to clipboard`, "success");
      }
    });
  }

  if (el.metaFingerprint) {
    el.metaFingerprint.addEventListener('click', () => {
      if (state.activeAudio && state.activeAudio.fingerprint) {
        navigator.clipboard.writeText(state.activeAudio.fingerprint);
        showToast(`Copied fingerprint to clipboard`, "success");
      }
    });
  }

  // YouTube Ingest
  el.btnYtDownload.addEventListener('click', async () => {
    const url = el.ytUrlInput.value.trim();
    if (!url) {
      showToast("Please enter a YouTube video URL", "error");
      return;
    }
    el.btnYtDownload.disabled = true;
    el.btnYtDownload.innerHTML = `<span class="dot dot-pulse"></span> Fetching...`;

    try {
      const res = await fetch("/api/audio/youtube", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Failed to start YouTube crawl");

      showToast("YouTube crawl initiated in background...", "info");
      pollTask(data.task_id, async (result) => {
        showToast("YouTube download complete!", "success");
        await fetchAudioList();
        await setActiveAudio(result.audio_id, { play: true });
        el.ytUrlInput.value = "";
      });
    } catch (err) {
      showToast(err.message, "error");
    } finally {
      el.btnYtDownload.disabled = false;
      el.btnYtDownload.textContent = "Fetch & Normalize";
    }
  });

  // Quick Save (Audio.quick_save)
  el.btnQuickSave.addEventListener('click', async () => {
    if (!state.activeAudio) return;
    try {
      const res = await fetch(`/api/audio/${state.activeAudio.id}/quick-save`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Quick save failed");

      navigator.clipboard.writeText(data.saved_path);
      showToast(`Quick saved to: ${data.saved_path} (Path copied!)`, "success");
      await fetchAudioList();
    } catch (err) {
      showToast(err.message, "error");
    }
  });

  // Save To Modal
  el.btnSaveToDialog.addEventListener('click', () => {
    if (!state.activeAudio) return;
    el.inputSavePath.value = `benchmarks/separation/sources/speech/${state.activeAudio.title}.wav`;
    el.modalSaveTo.classList.remove('hidden');
  });
  el.btnCancelSave.addEventListener('click', () => el.modalSaveTo.classList.add('hidden'));
  el.btnCloseSaveModal.addEventListener('click', () => el.modalSaveTo.classList.add('hidden'));
  el.btnConfirmSave.addEventListener('click', async () => {
    const dest = el.inputSavePath.value.trim();
    if (!dest) return;
    try {
      const res = await fetch(`/api/audio/${state.activeAudio.id}/save-to`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dest }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Save failed");

      showToast(`Saved to: ${data.saved_path}`, "success");
      el.modalSaveTo.classList.add('hidden');
      await fetchAudioList();
    } catch (err) {
      showToast(err.message, "error");
    }
  });

  // Direct Download
  el.btnDownloadAudio.addEventListener('click', () => {
    if (!state.activeAudio) return;
    const link = document.createElement("a");
    link.href = `/api/audio/${state.activeAudio.id}/stream`;
    link.download = `${state.activeAudio.title || 'audio'}.${state.activeAudio.format || 'wav'}`;
    link.click();
  });

  // Send to Separation tab
  el.btnSendToSep.addEventListener('click', () => {
    switchTab('tab-separation');
    el.sepInputSelect.value = state.activeAudio.id;
  });

  // Sample Library Modal
  el.btnBrowseLibrary.addEventListener('click', openLibraryModal);
  el.btnCloseLibraryModal.addEventListener('click', () => el.modalLibrary.classList.add('hidden'));
}

// ==================== YOUTUBE CRAWLER STUDIO ====================

function initYouTubeCrawler() {
  // Paste button in tab
  if (el.btnYtPasteTab) {
    el.btnYtPasteTab.addEventListener('click', async () => {
      try {
        const text = await navigator.clipboard.readText();
        if (text) {
          el.ytTabUrlInput.value = text.trim();
          showToast("Pasted link from clipboard", "info");
        }
      } catch (_) {
        showToast("Clipboard access denied or empty", "warning");
      }
    });
  }

  // Inspect URL
  el.btnYtTabInspect.addEventListener('click', async () => {
    const url = el.ytTabUrlInput.value.trim();
    if (!url) {
      showToast("Please enter a YouTube video URL to inspect", "error");
      return;
    }
    el.btnYtTabInspect.disabled = true;
    el.btnYtTabInspect.textContent = "Inspecting...";

    try {
      const res = await fetch("/api/crawler/inspect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Failed to inspect YouTube URL");

      renderYouTubePreview(data);
      showToast(`Found: ${data.title}`, "success");
    } catch (err) {
      showToast(err.message, "error");
    } finally {
      el.btnYtTabInspect.disabled = false;
      el.btnYtTabInspect.textContent = "Inspect";
    }
  });

  // Download & Normalize
  el.btnYtTabDownload.addEventListener('click', async () => {
    const url = el.ytTabUrlInput.value.trim();
    if (!url) {
      showToast("Please enter a YouTube URL to download", "error");
      return;
    }

    const sampleRate = parseInt(el.ytSampleRateSelect.value) || 44100;
    const audioFormat = el.ytFormatSelect.value || "wav";

    el.btnYtTabDownload.disabled = true;
    el.ytTaskProgressBox.classList.remove('hidden');
    el.ytEmptyPlaceholder.classList.add('hidden');

    let startTime = Date.now();
    const timerInterval = setInterval(() => {
      el.ytTaskTimer.textContent = `${((Date.now() - startTime) / 1000).toFixed(1)}s`;
    }, 100);

    try {
      const res = await fetch("/api/audio/youtube", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          url,
          sample_rate: sampleRate,
          channels: 1,
          audio_format: audioFormat,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "YouTube crawl failed to start");

      pollTask(data.task_id, async (result) => {
        clearInterval(timerInterval);
        el.ytTaskProgressBox.classList.add('hidden');
        el.btnYtTabDownload.disabled = false;
        showToast(`YouTube download complete! Loaded '${result.metadata.title}'`, "success");
        await fetchAudioList();
        await fetchYouTubeVault();
        await setActiveAudio(result.audio_id, { play: true });
      }, (err) => {
        clearInterval(timerInterval);
        el.ytTaskProgressBox.classList.add('hidden');
        el.btnYtTabDownload.disabled = false;
        showToast(`YouTube ingestion error: ${err}`, "error");
      });

    } catch (err) {
      clearInterval(timerInterval);
      el.ytTaskProgressBox.classList.add('hidden');
      el.btnYtTabDownload.disabled = false;
      showToast(err.message, "error");
    }
  });

  el.btnRefreshYtHistory.addEventListener('click', fetchYouTubeVault);
}

function renderYouTubePreview(meta) {
  el.ytEmptyPlaceholder.classList.add('hidden');
  el.ytPreviewCard.classList.remove('hidden');
  el.ytInspectBadge.classList.remove('hidden');

  el.ytPreviewTitle.textContent = meta.title || "Unknown Video";
  el.ytPreviewChannel.textContent = meta.uploader || "YouTube Channel";
  el.ytPreviewViews.textContent = meta.view_count ? `${meta.view_count.toLocaleString()} views` : "";
  el.ytPreviewDuration.textContent = formatTime(meta.duration || 0);
  el.ytPreviewDesc.textContent = meta.description || "No description provided.";
  el.ytPreviewThumb.src = meta.thumbnail || "";
  el.ytPreviewLink.href = meta.webpage_url || "#";
}

async function fetchYouTubeVault() {
  try {
    const res = await fetch("/api/crawler/history");
    const data = await res.json();
    renderYouTubeVault(data.downloads || []);
  } catch (err) {
    console.error("Failed to fetch YouTube history:", err);
  }
}

function renderYouTubeVault(items) {
  const container = el.ytVaultList;
  container.innerHTML = "";

  if (!items || items.length === 0) {
    container.innerHTML = `<div class="empty-placeholder">No previously downloaded YouTube audio in .data/yt_crawler/downloads.</div>`;
    return;
  }

  items.forEach(item => {
    const card = document.createElement("div");
    card.className = "yt-vault-item";
    card.innerHTML = `
      <div class="file-details">
        <span class="file-name">${item.name}</span>
        <span class="file-path">${item.sample_rate.toLocaleString()}Hz • ${item.channels === 1 ? 'Mono' : 'Stereo'} • ${(item.duration_s || 0).toFixed(1)}s • ${formatBytes(item.size)}</span>
      </div>
      <div class="stem-actions">
        <button class="btn btn-sm btn-secondary btn-load-yt-workspace">🎛️ Workspace</button>
        <button class="btn btn-sm btn-secondary btn-yt-sep">🧪 Separate</button>
        <button class="btn btn-sm btn-secondary btn-yt-diar">👥 Diarize</button>
      </div>
    `;

    card.querySelector('.btn-load-yt-workspace').addEventListener('click', async () => {
      await loadServerFile(item.path);
    });

    card.querySelector('.btn-yt-sep').addEventListener('click', async () => {
      await loadServerFile(item.path);
      switchTab('tab-separation');
    });

    card.querySelector('.btn-yt-diar').addEventListener('click', async () => {
      await loadServerFile(item.path);
      switchTab('tab-diarization');
    });

    container.appendChild(card);
  });
}

async function uploadFile(file) {
  const formData = new FormData();
  formData.append('file', file);

  showToast(`Uploading ${file.name}...`, "info");
  try {
    const res = await fetch("/api/audio/upload", {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Upload failed");

    showToast(`Uploaded ${file.name} successfully!`, "success");
    await fetchAudioList();
    await setActiveAudio(data.audio_id, { play: true });
  } catch (err) {
    showToast(err.message, "error");
  }
}

// ==================== MODEL SEPARATION STUDIO ====================

function initSeparationStudio() {
  // Model Card Selection
  el.modelCards.forEach(card => {
    card.addEventListener('click', () => {
      el.modelCards.forEach(c => c.classList.remove('active'));
      card.classList.add('active');
      const model = card.dataset.model;
      el.roformerPresetGroup.style.display = (model === 'bs_roformer' || model === 'mel_roformer') ? 'block' : 'none';
    });
  });

  // Run Separation Button
  el.btnRunSeparation.addEventListener('click', async () => {
    const audioId = el.sepInputSelect.value;
    if (!audioId) {
      showToast("Please select an input audio to separate", "error");
      return;
    }

    const activeCard = document.querySelector('.model-card[data-model].active');
    const modelType = activeCard ? activeCard.dataset.model : "htdemucs";
    const device = el.sepDeviceSelect.value;
    const twoStems = el.sepStemsSelect.value;
    const modelName = el.roformerCheckpointInput.value.trim() || undefined;

    el.btnRunSeparation.disabled = true;
    el.sepTaskProgressBox.classList.remove('hidden');
    el.sepTaskTitle.textContent = `Running ${modelType.toUpperCase()} separation...`;

    let startTime = Date.now();
    const timerInterval = setInterval(() => {
      el.sepTaskTimer.textContent = `${((Date.now() - startTime) / 1000).toFixed(1)}s`;
    }, 100);

    try {
      const res = await fetch("/api/separation/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          audio_id: audioId,
          model_type: modelType,
          device: device,
          two_stems: twoStems,
          model_name: modelName,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Separation request failed");

      pollTask(data.task_id, (result) => {
        clearInterval(timerInterval);
        el.sepTaskProgressBox.classList.add('hidden');
        el.btnRunSeparation.disabled = false;
        showToast(`Separation completed in ${result.elapsed_s}s!`, "success");
        renderSeparationResultCard(result);
        fetchAudioList();
      }, (err) => {
        clearInterval(timerInterval);
        el.sepTaskProgressBox.classList.add('hidden');
        el.btnRunSeparation.disabled = false;
        showToast(`Separation failed: ${err}`, "error");
      });

    } catch (err) {
      clearInterval(timerInterval);
      el.sepTaskProgressBox.classList.add('hidden');
      el.btnRunSeparation.disabled = false;
      showToast(err.message, "error");
    }
  });
}

function renderSeparationResultCard(result) {
  const list = el.sepResultsList;
  if (list.querySelector('.empty-placeholder')) {
    list.innerHTML = "";
  }

  const meta = result.metadata;
  const audioId = result.separated_audio_id;

  const card = document.createElement("div");
  card.className = "stem-result-card";
  card.innerHTML = `
    <div class="stem-info">
      <span class="stem-title">${meta.title} (${result.model_type})</span>
      <span class="stem-meta">${meta.format.toUpperCase()} • ${meta.sample_rate.toLocaleString()}Hz • ${(meta.duration_s || 0).toFixed(2)}s • ${result.elapsed_s}s exec</span>
    </div>
    <div class="stem-actions">
      <button class="btn btn-sm btn-secondary btn-play-stem" data-id="${audioId}">▶ Play</button>
      <button class="btn btn-sm btn-secondary btn-load-workspace" data-id="${audioId}">🎛️ Workspace</button>
      <button class="btn btn-sm btn-accent btn-send-compare" data-id="${audioId}">⚖️ Compare</button>
    </div>
  `;

  card.querySelector('.btn-play-stem').addEventListener('click', () => loadAudioIntoPlayer(audioId, true));
  card.querySelector('.btn-load-workspace').addEventListener('click', () => {
    switchTab('tab-workspace');
    setActiveAudio(audioId, { play: true });
  });
  card.querySelector('.btn-send-compare').addEventListener('click', () => {
    switchTab('tab-comparison');
    el.compareTrackBSelect.value = audioId;
    if (state.activeAudio) {
      el.compareTrackASelect.value = state.activeAudio.id;
    }
  });

  list.prepend(card);
}

// ==================== SPEAKER DIARIZATION STUDIO ====================

function initDiarizationStudio() {
  // Model selection
  el.diarModelCards.forEach(card => {
    card.addEventListener('click', () => {
      el.diarModelCards.forEach(c => c.classList.remove('active'));
      card.classList.add('active');
    });
  });

  // Run Diarization
  el.btnRunDiarization.addEventListener('click', async () => {
    const audioId = el.diarInputSelect.value;
    if (!audioId) {
      showToast("Please select an input audio for diarization", "error");
      return;
    }

    const activeCard = document.querySelector('.model-card[data-diar-model].active');
    const modelType = activeCard ? activeCard.dataset.diarModel : "pyannote";
    const device = el.diarDeviceSelect.value;
    const token = el.hfTokenInput.value.trim() || undefined;

    el.btnRunDiarization.disabled = true;
    el.diarTaskProgressBox.classList.remove('hidden');

    let startTime = Date.now();
    const timerInterval = setInterval(() => {
      el.diarTaskTimer.textContent = `${((Date.now() - startTime) / 1000).toFixed(1)}s`;
    }, 100);

    try {
      const res = await fetch("/api/diarization/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          audio_id: audioId,
          model_type: modelType,
          device: device,
          token: token,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Diarization failed to start");

      pollTask(data.task_id, (result) => {
        clearInterval(timerInterval);
        el.diarTaskProgressBox.classList.add('hidden');
        el.btnRunDiarization.disabled = false;
        showToast(`Speaker Diarization completed in ${result.elapsed_s}s!`, "success");
        renderDiarizationResults(result.diarization, audioId);
      }, (err) => {
        clearInterval(timerInterval);
        el.diarTaskProgressBox.classList.add('hidden');
        el.btnRunDiarization.disabled = false;
        showToast(`Diarization failed: ${err}`, "error");
      });

    } catch (err) {
      clearInterval(timerInterval);
      el.diarTaskProgressBox.classList.add('hidden');
      el.btnRunDiarization.disabled = false;
      showToast(err.message, "error");
    }
  });
}

function renderDiarizationResults(diarization, audioId) {
  el.diarEmptyPlaceholder.classList.add('hidden');
  el.diarResultsWrapper.classList.remove('hidden');

  const audioItem = state.audioList.find(a => a.id === audioId);
  const totalAudioDuration = (audioItem ? audioItem.duration_s : 0) || 1;

  // Speaker Palette map
  const colors = ["var(--spk-0)", "var(--spk-1)", "var(--spk-2)", "var(--spk-3)", "var(--spk-4)"];
  const spkColorMap = {};
  (diarization.speakers || []).forEach((spk, idx) => {
    spkColorMap[spk.speaker_id] = colors[idx % colors.length];
  });

  // 1. Speakers Summary Cards
  el.speakersSummaryRow.innerHTML = "";
  (diarization.speakers || []).forEach(spk => {
    const card = document.createElement("div");
    card.className = "speaker-badge-card";
    const color = spkColorMap[spk.speaker_id] || "var(--accent-cyan)";
    const pct = ((spk.total_speech_s / totalAudioDuration) * 100).toFixed(1);

    card.innerHTML = `
      <div class="spk-color-indicator" style="background-color: ${color};"></div>
      <span class="spk-id">${spk.speaker_id}</span>
      <span class="spk-stats">${spk.total_speech_s.toFixed(2)}s (${pct}% • ${spk.turns_count} turns)</span>
    `;
    el.speakersSummaryRow.appendChild(card);
  });

  // 2. Interactive Timeline
  el.diarTimeline.innerHTML = "";
  (diarization.turns || []).forEach(turn => {
    const block = document.createElement("div");
    block.className = "diar-turn-block";
    const color = spkColorMap[turn.speaker_id] || "var(--accent-cyan)";
    const leftPct = (turn.start_s / totalAudioDuration) * 100;
    const widthPct = Math.max(0.5, ((turn.end_s - turn.start_s) / totalAudioDuration) * 100);

    block.style.left = `${leftPct}%`;
    block.style.width = `${widthPct}%`;
    block.style.backgroundColor = color;
    block.title = `${turn.speaker_id}: ${turn.start_s.toFixed(2)}s – ${turn.end_s.toFixed(2)}s (${turn.duration_s.toFixed(2)}s)`;

    block.addEventListener('click', () => {
      loadAudioIntoPlayer(audioId);
      seekTo(turn.start_s);
      el.audio.play();
    });

    el.diarTimeline.appendChild(block);
  });

  // 3. Turns Table
  el.turnsTableBody.innerHTML = "";
  (diarization.turns || []).forEach(turn => {
    const tr = document.createElement("tr");
    const color = spkColorMap[turn.speaker_id] || "var(--accent-cyan)";
    tr.innerHTML = `
      <td style="color: ${color}; font-weight: 700;">${turn.speaker_id}</td>
      <td>${turn.start_s.toFixed(2)}</td>
      <td>${turn.end_s.toFixed(2)}</td>
      <td>${turn.duration_s.toFixed(2)}s</td>
      <td><button class="btn btn-sm btn-ghost btn-play-turn" data-start="${turn.start_s}">▶ Seek</button></td>
    `;
    tr.querySelector('.btn-play-turn').addEventListener('click', () => {
      loadAudioIntoPlayer(audioId);
      seekTo(turn.start_s);
      el.audio.play();
    });
    el.turnsTableBody.appendChild(tr);
  });
}

// ==================== COMPARISON STUDIO ====================

function initComparisonStudio() {
  el.btnAbSwitchA.addEventListener('click', () => switchAB('A'));
  el.btnAbSwitchB.addEventListener('click', () => switchAB('B'));
  el.btnAbToggleInstant.addEventListener('click', toggleABInstant);

  el.compareTrackASelect.addEventListener('change', (e) => {
    state.ab.trackAId = e.target.value;
  });
  el.compareTrackBSelect.addEventListener('change', (e) => {
    state.ab.trackBId = e.target.value;
  });

  // Generate Comparison Visualizers
  el.btnGenerateComparison.addEventListener('click', async () => {
    const beforeId = el.compareTrackASelect.value;
    const afterId = el.compareTrackBSelect.value;

    if (!beforeId || !afterId) {
      showToast("Please select both Track A and Track B to compare", "error");
      return;
    }

    el.btnGenerateComparison.disabled = true;
    el.btnGenerateComparison.textContent = "Rendering Visualizers...";

    try {
      // Spectrogram Comparison
      const specRes = await fetch("/api/compare/spectrogram", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ before_id: beforeId, after_id: afterId }),
      });
      if (!specRes.ok) throw new Error("Spectrogram comparison failed");
      const specBlob = await specRes.blob();
      el.imgCompareSpectrogram.src = URL.createObjectURL(specBlob);
      el.imgCompareSpectrogram.classList.remove('hidden');
      el.spectrogramCompareBox.querySelector('.empty-placeholder')?.remove();

      // Waveform Comparison
      const waveRes = await fetch("/api/compare/waveform", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ before_id: beforeId, after_id: afterId }),
      });
      if (!waveRes.ok) throw new Error("Waveform comparison failed");
      const waveBlob = await waveRes.blob();
      el.imgCompareWaveform.src = URL.createObjectURL(waveBlob);
      el.imgCompareWaveform.classList.remove('hidden');
      el.waveformCompareBox.querySelector('.empty-placeholder')?.remove();

      showToast("Aligned Spectrograms & Waveforms generated!", "success");
    } catch (err) {
      showToast(err.message, "error");
    } finally {
      el.btnGenerateComparison.disabled = false;
      el.btnGenerateComparison.innerHTML = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="18" cy="5" r="3"></circle><circle cx="6" cy="12" r="3"></circle><circle cx="18" cy="19" r="3"></circle><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"></line><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"></line></svg> <span>Generate Aligned Visuals</span>`;
    }
  });
}

function switchAB(track) {
  state.ab.currentTrack = track;
  const targetId = track === 'A' ? el.compareTrackASelect.value : el.compareTrackBSelect.value;
  if (!targetId) return;

  const curTime = el.audio.currentTime;
  const wasPlaying = !el.audio.paused;

  el.btnAbSwitchA.classList.toggle('active', track === 'A');
  el.btnAbSwitchB.classList.toggle('active', track === 'B');
  el.abActiveLabel.textContent = `Playing Track ${track}`;

  loadAudioIntoPlayer(targetId, false);
  el.audio.currentTime = curTime;
  if (wasPlaying) el.audio.play();
}

function toggleABInstant() {
  switchAB(state.ab.currentTrack === 'A' ? 'B' : 'A');
}

// ==================== BENCHMARK MIXER ====================

function initBenchmarkMixer() {
  el.mixerSmrSlider.addEventListener('input', (e) => {
    el.mixerSmrVal.textContent = `${parseFloat(e.target.value).toFixed(1)} dB`;
  });

  el.btnRunMixer.addEventListener('click', async () => {
    const speechId = el.mixerSpeechSelect.value;
    const musicId = el.mixerMusicSelect.value;
    const targetSmrDb = parseFloat(el.mixerSmrSlider.value);
    const seed = parseInt(el.mixerSeedInput.value) || 42;

    if (!speechId || !musicId) {
      showToast("Please select both Speech and Music tracks", "error");
      return;
    }

    el.btnRunMixer.disabled = true;
    el.btnRunMixer.textContent = "Synthesizing Mixture...";

    try {
      const res = await fetch("/api/benchmark/mix", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          speech_id: speechId,
          music_id: musicId,
          target_smr_db: targetSmrDb,
          seed: seed,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Mixing failed");

      showToast("Mixture generated successfully!", "success");
      await fetchAudioList();
      renderMixerResults(data);
    } catch (err) {
      showToast(err.message, "error");
    } finally {
      el.btnRunMixer.disabled = false;
      el.btnRunMixer.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path></svg> <span>Synthesize Benchmark Mixture (AudioMixer.mix)</span>`;
    }
  });
}

function renderMixerResults(data) {
  el.mixerResultBox.classList.remove('hidden');
  el.mixStemsGrid.innerHTML = "";

  const items = [
    { title: "Synthetic Mixture", id: data.mixture_id, type: "Mixture" },
    { title: "Speech Reference", id: data.speech_ref_id, type: "Speech" },
    { title: "Music Reference", id: data.music_ref_id, type: "Music" },
  ];

  items.forEach(item => {
    const card = document.createElement("div");
    card.className = "stem-result-card";
    card.innerHTML = `
      <div class="stem-info">
        <span class="stem-title">${item.title}</span>
        <span class="stem-meta">ID: ${item.id}</span>
      </div>
      <div class="stem-actions">
        <button class="btn btn-sm btn-secondary btn-play-mix">▶ Play</button>
        <button class="btn btn-sm btn-secondary btn-load-mix">🎛️ Workspace</button>
      </div>
    `;
    card.querySelector('.btn-play-mix').addEventListener('click', () => loadAudioIntoPlayer(item.id, true));
    card.querySelector('.btn-load-mix').addEventListener('click', () => {
      switchTab('tab-workspace');
      setActiveAudio(item.id, { play: true });
    });
    el.mixStemsGrid.appendChild(card);
  });
}

// ==================== LIBRARY & HISTORY ====================

async function fetchAudioList() {
  try {
    const res = await fetch("/api/audio");
    const data = await res.json();
    state.audioList = data.audios || [];
    el.sessionCountBadge.textContent = `${state.audioList.length} items`;
    renderSessionHistory();
    populateAllAudioSelects();
  } catch (err) {
    console.error("Failed to fetch audio list:", err);
  }
}

function renderSessionHistory() {
  const container = el.sessionHistoryList;
  container.innerHTML = "";

  if (state.audioList.length === 0) {
    container.innerHTML = `<div class="empty-placeholder">No active audio in this session.</div>`;
    return;
  }

  state.audioList.forEach(item => {
    const card = document.createElement("div");
    card.className = "file-item-card";
    card.innerHTML = `
      <div class="file-details">
        <span class="file-name">${item.title}</span>
        <span class="file-path">${item.format.toUpperCase()} • ${item.sample_rate.toLocaleString()}Hz • ${(item.duration_s || 0).toFixed(2)}s • ${item.source_type}</span>
      </div>
      <div class="file-actions">
        <button class="btn btn-sm btn-secondary btn-load-session" data-id="${item.id}">Load</button>
      </div>
    `;
    card.querySelector('.btn-load-session').addEventListener('click', () => {
      switchTab('tab-workspace');
      setActiveAudio(item.id, { play: true });
    });
    container.appendChild(card);
  });
}

async function fetchServerFiles() {
  try {
    const res = await fetch("/api/library");
    const data = await res.json();
    state.serverFiles = data.files || [];
    renderServerFiles();
  } catch (err) {
    console.error("Failed to fetch library:", err);
  }
}

function renderServerFiles() {
  const container = el.serverFilesList;
  container.innerHTML = "";

  if (state.serverFiles.length === 0) {
    container.innerHTML = `<div class="empty-placeholder">No project audio files discovered.</div>`;
    return;
  }

  state.serverFiles.forEach(file => {
    const card = document.createElement("div");
    card.className = "file-item-card";
    card.innerHTML = `
      <div class="file-details">
        <span class="file-name">${file.name}</span>
        <span class="file-path">${file.category} • ${formatBytes(file.size)}</span>
      </div>
      <div class="file-actions">
        <button class="btn btn-sm btn-secondary btn-load-file" data-path="${file.path}">Load</button>
      </div>
    `;
    card.querySelector('.btn-load-file').addEventListener('click', () => loadServerFile(file.path));
    container.appendChild(card);
  });
}

async function loadServerFile(filePath) {
  try {
    showToast(`Loading ${filePath}...`, "info");
    const res = await fetch("/api/library/load", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: filePath }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Failed to load file");

    showToast(`Loaded ${filePath} successfully!`, "success");
    await fetchAudioList();
    switchTab('tab-workspace');
    await setActiveAudio(data.audio_id, { play: true });
    el.modalLibrary.classList.add('hidden');
  } catch (err) {
    showToast(err.message, "error");
  }
}

function openLibraryModal() {
  el.modalLibrary.classList.remove('hidden');
  el.modalLibraryItems.innerHTML = "";

  state.serverFiles.forEach(file => {
    const item = document.createElement("div");
    item.className = "file-item-card";
    item.innerHTML = `
      <div class="file-details">
        <span class="file-name">${file.name}</span>
        <span class="file-path">${file.category} • ${file.path}</span>
      </div>
      <button class="btn btn-sm btn-primary btn-modal-load" data-path="${file.path}">Load</button>
    `;
    item.querySelector('.btn-modal-load').addEventListener('click', () => loadServerFile(file.path));
    el.modalLibraryItems.appendChild(item);
  });
}

function toggleShortcutsModal() {
  if (el.modalShortcuts) {
    el.modalShortcuts.classList.toggle('hidden');
  }
}

function closeAllModals() {
  if (el.modalSaveTo) el.modalSaveTo.classList.add('hidden');
  if (el.modalLibrary) el.modalLibrary.classList.add('hidden');
  if (el.modalShortcuts) el.modalShortcuts.classList.add('hidden');
}

function populateAllAudioSelects() {
  const selects = [
    el.sepInputSelect,
    el.diarInputSelect,
    el.compareTrackASelect,
    el.compareTrackBSelect,
    el.mixerSpeechSelect,
    el.mixerMusicSelect,
  ];

  selects.forEach(select => {
    if (!select) return;
    const currentVal = select.value;
    select.innerHTML = '<option value="">-- Select Audio Track --</option>';

    state.audioList.forEach(item => {
      const opt = document.createElement("option");
      opt.value = item.id;
      opt.textContent = `${item.title} (${item.format.toUpperCase()}, ${(item.duration_s || 0).toFixed(1)}s, ${item.source_type})`;
      select.appendChild(opt);
    });

    if (currentVal && state.audioList.some(a => a.id === currentVal)) {
      select.value = currentVal;
    } else if (state.activeAudio && (select === el.sepInputSelect || select === el.diarInputSelect)) {
      select.value = state.activeAudio.id;
    }
  });
}

// ==================== TASK POLLING HELPER ====================

function pollTask(taskId, onComplete, onError) {
  const poll = async () => {
    try {
      const res = await fetch(`/api/tasks/${taskId}`);
      if (!res.ok) throw new Error("Task polling error");
      const task = await res.json();

      if (task.status === "completed") {
        if (onComplete) onComplete(task.result);
      } else if (task.status === "failed") {
        if (onError) onError(task.error);
      } else {
        setTimeout(poll, 600);
      }
    } catch (err) {
      if (onError) onError(err.message);
    }
  };
  poll();
}

// ==================== LIVE RELOAD SSE ====================

function initLiveReload() {
  const evtSource = new EventSource("/api/live-reload");
  evtSource.onmessage = (event) => {
    if (event.data === "reload") {
      console.log("Hot reload triggered from server!");
      window.location.reload();
    }
  };
  evtSource.onerror = () => {
    const badge = document.getElementById('live-reload-badge');
    if (badge) badge.style.opacity = '0.5';
  };
}

// ==================== SYSTEM STATUS ====================

async function fetchSystemStatus() {
  try {
    const res = await fetch("/api/system/status");
    const data = await res.json();
    state.systemStatus = data;
    el.deviceLabel.textContent = `${data.device_name.split(':')[0]}`;
  } catch (err) {
    el.deviceLabel.textContent = "Offline";
  }
}

// ==================== NAVIGATION TABS ====================

function initNavigation() {
  el.tabs.forEach(tab => {
    tab.addEventListener('click', () => switchTab(tab.dataset.tab));
  });

  if (el.btnOpenShortcutsModal) {
    el.btnOpenShortcutsModal.addEventListener('click', toggleShortcutsModal);
  }
  if (el.btnCloseShortcutsModal) {
    el.btnCloseShortcutsModal.addEventListener('click', () => el.modalShortcuts.classList.add('hidden'));
  }
}

function switchTab(tabId) {
  el.tabs.forEach(t => t.classList.toggle('active', t.dataset.tab === tabId));
  el.tabPanes.forEach(pane => pane.classList.toggle('active', pane.id === tabId));
  try {
    localStorage.setItem('sonic_active_tab', tabId);
  } catch (_) {}
  if (tabId === 'tab-workspace') {
    setTimeout(renderWaveform, 50);
  }
}

// ==================== THEME MANAGEMENT ====================

function initTheme() {
  const savedTheme = localStorage.getItem('sonic_theme') || (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
  applyTheme(savedTheme);

  if (el.btnThemeToggle) {
    el.btnThemeToggle.addEventListener('click', () => {
      const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
      const nextTheme = currentTheme === 'dark' ? 'light' : 'dark';
      applyTheme(nextTheme);
      showToast(`Switched to ${nextTheme} theme`, "info");
    });
  }
}

function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  try {
    localStorage.setItem('sonic_theme', theme);
  } catch (_) {}

  const isLight = theme === 'light';
  if (el.iconThemeSun) el.iconThemeSun.classList.toggle('hidden', !isLight);
  if (el.iconThemeMoon) el.iconThemeMoon.classList.toggle('hidden', isLight);
  renderWaveform();
}

// ==================== APP INITIALIZATION ====================

async function initApp() {
  initTheme();
  initPlayer();
  initWaveformInteractions();
  initAudioCutter();
  initIngestAndSaves();
  initYouTubeCrawler();
  initSeparationStudio();
  initDiarizationStudio();
  initComparisonStudio();
  initBenchmarkMixer();
  initNavigation();
  initLiveReload();

  el.btnRefreshLibrary.addEventListener('click', fetchServerFiles);

  await fetchSystemStatus();
  await fetchServerFiles();
  await fetchAudioList();
  await fetchYouTubeVault();

  // Restore saved active tab
  try {
    const savedTab = localStorage.getItem('sonic_active_tab');
    if (savedTab && document.getElementById(savedTab)) {
      switchTab(savedTab);
    }
  } catch (_) {}

  // Restore saved active audio or load first sample
  let targetAudioId = null;
  try {
    const savedId = localStorage.getItem('sonic_active_audio_id');
    if (savedId && state.audioList.some(a => a.id === savedId)) {
      targetAudioId = savedId;
    }
  } catch (_) {}

  if (targetAudioId) {
    await setActiveAudio(targetAudioId);
  } else if (state.audioList.length > 0) {
    await setActiveAudio(state.audioList[0].id);
  } else if (state.serverFiles.length > 0) {
    const firstSample = state.serverFiles.find(f => f.category === "Benchmark Speech") || state.serverFiles[0];
    if (firstSample) {
      await loadServerFile(firstSample.path);
    }
  }
}

document.addEventListener('DOMContentLoaded', initApp);
