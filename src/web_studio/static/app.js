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

  // Sample Library Modal Filter State
  libraryModalSearch: "",
  libraryModalCategory: "all",

  // Tab 7 Project Explorer Filter State
  tabLibrarySearch: "",
  tabLibraryCategory: "all",
  activeSavePreset: "speech",
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
  btnCutAndAudition: document.getElementById('btn-cut-and-audition'),
  btnCutAndRunModels: document.getElementById('btn-cut-and-run-models'),
  cutsTableBody: document.getElementById('cuts-table-body'),
  cutsCounterBadge: document.getElementById('cuts-counter-badge'),
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
  btnRunMultiSeparation: document.getElementById('btn-run-multi-separation'),
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

  // Audition & Scoring Hub
  auditionClipSelect: document.getElementById('audition-clip-select'),
  activeAuditionTrackName: document.getElementById('active-audition-track-name'),
  auditionTimeCurrent: document.getElementById('audition-time-current'),
  auditionTimeTotal: document.getElementById('audition-time-total'),
  auditionScrubber: document.getElementById('audition-scrubber'),
  btnAuditionPlay: document.getElementById('btn-audition-play'),
  iconAuditionPlay: document.getElementById('icon-audition-play'),
  iconAuditionPause: document.getElementById('icon-audition-pause'),
  btnAuditionLoop: document.getElementById('btn-audition-loop'),
  auditionVolumeSlider: document.getElementById('audition-volume-slider'),
  auditionTrackPills: document.getElementById('audition-track-pills'),
  btnBatchSeparateActiveClip: document.getElementById('btn-batch-separate-active-clip'),
  btnGenerateComparison: document.getElementById('btn-generate-comparison'),
  imgCompareSpectrogram: document.getElementById('img-compare-spectrogram'),
  spectrogramCompareBox: document.getElementById('spectrogram-compare-box'),
  scoringActiveModelLabel: document.getElementById('scoring-active-model-label'),
  currentEvalScoreBadge: document.getElementById('current-eval-score-badge'),
  starRatingWidget: document.getElementById('star-rating-widget'),
  starScoreText: document.getElementById('star-score-text'),
  sliderSubmetricClarity: document.getElementById('slider-submetric-clarity'),
  valSubmetricClarity: document.getElementById('val-submetric-clarity'),
  sliderSubmetricBleed: document.getElementById('slider-submetric-bleed'),
  valSubmetricBleed: document.getElementById('val-submetric-bleed'),
  sliderSubmetricArtifacts: document.getElementById('slider-submetric-artifacts'),
  valSubmetricArtifacts: document.getElementById('val-submetric-artifacts'),
  evalTagChips: document.getElementById('eval-tag-chips'),
  evalNotesInput: document.getElementById('eval-notes-input'),
  btnSaveEvaluation: document.getElementById('btn-save-evaluation'),
  btnCopyEvalNote: document.getElementById('btn-copy-eval-note'),

  // Evaluation Matrix & Notes Review
  btnExportEvalCsv: document.getElementById('btn-export-eval-csv'),
  btnExportEvalJson: document.getElementById('btn-export-eval-json'),
  btnRefreshEvalMatrix: document.getElementById('btn-refresh-eval-matrix'),
  kpiTotalEvals: document.getElementById('kpi-total-evals'),
  kpiTotalClips: document.getElementById('kpi-total-clips'),
  kpiTopModel: document.getElementById('kpi-top-model'),
  kpiAvgScore: document.getElementById('kpi-avg-score'),
  evalSearchInput: document.getElementById('eval-search-input'),
  evalModelFilterPills: document.getElementById('eval-model-filter-pills'),
  evaluationsTableBody: document.getElementById('evaluations-table-body'),

  // Side-by-Side Model Comparison Deck & Provenance
  activeModelProvenanceBanner: document.getElementById('active-model-provenance-banner'),
  provModelBadge: document.getElementById('prov-model-badge'),
  provDetailsText: document.getElementById('prov-details-text'),
  sbsColumnsDeck: document.getElementById('sbs-columns-deck'),
  sbsModelsCountBadge: document.getElementById('sbs-models-count-badge'),

  // Library & History
  serverFilesList: document.getElementById('server-files-list'),
  btnRefreshLibrary: document.getElementById('btn-refresh-library'),
  sessionHistoryList: document.getElementById('session-history-list'),
  sessionCountBadge: document.getElementById('session-count-badge'),

  // Modals & Toasts
  modalSaveTo: document.getElementById('modal-save-to'),
  inputSavePath: document.getElementById('input-save-path'),
  saveTargetPresets: document.getElementById('save-target-presets'),
  btnCancelSave: document.getElementById('btn-cancel-save'),
  btnConfirmSave: document.getElementById('btn-confirm-save'),
  btnCloseSaveModal: document.getElementById('btn-close-save-modal'),
  modalLibrary: document.getElementById('modal-library'),
  modalLibraryItems: document.getElementById('modal-library-items'),
  btnCloseLibraryModal: document.getElementById('btn-close-library-modal'),
  btnCancelLibraryModal: document.getElementById('btn-cancel-library-modal'),
  libraryModalSearch: document.getElementById('library-modal-search'),
  libraryModalCategories: document.getElementById('library-modal-categories'),
  libraryModalCount: document.getElementById('library-modal-count'),
  tabLibrarySearch: document.getElementById('tab-library-search'),
  tabLibraryCategories: document.getElementById('tab-library-categories'),
  modalShortcuts: document.getElementById('modal-shortcuts'),
  btnCloseShortcutsModal: document.getElementById('btn-close-shortcuts-modal'),
  toastContainer: document.getElementById('toast-container'),
};

// ==================== UTILITY FUNCTIONS ====================

function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

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

async function parseJsonResponse(res) {
  const text = await res.text();
  let data;
  try {
    data = JSON.parse(text);
  } catch (_) {
    if (res.status === 404) {
      throw new Error(`Endpoint not found (HTTP 404). Please ensure the backend server was restarted with the latest routes!`);
    }
    throw new Error(`Server returned HTTP ${res.status}: ${text.substring(0, 120) || res.statusText}`);
  }
  if (!res.ok) {
    throw new Error(data.error || `Request failed with status ${res.status}`);
  }
  return data;
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
  if (el.btnPlayPause) el.btnPlayPause.addEventListener('click', togglePlayPause);
  if (el.btnSkipBack) el.btnSkipBack.addEventListener('click', () => seekRelative(-5));
  if (el.btnSkipFwd) el.btnSkipFwd.addEventListener('click', () => seekRelative(5));

  if (el.btnLoop) {
    el.btnLoop.addEventListener('click', () => {
      state.player.loop = !state.player.loop;
      if (el.audio) el.audio.loop = state.player.loop;
      el.btnLoop.classList.toggle('active', state.player.loop);
      showToast(state.player.loop ? "Loop playback enabled" : "Loop playback disabled", "info");
    });
  }

  if (el.speedSelect) {
    el.speedSelect.addEventListener('change', (e) => {
      state.player.playbackRate = parseFloat(e.target.value);
      if (el.audio) el.audio.playbackRate = state.player.playbackRate;
    });
  }

  if (el.volumeSlider) {
    el.volumeSlider.addEventListener('input', (e) => {
      state.player.volume = parseFloat(e.target.value);
      if (el.audio) el.audio.volume = state.player.volume;
      updateVolumeIcon();
    });
  }

  if (el.btnMute) {
    el.btnMute.addEventListener('click', () => {
      if (!el.audio) return;
      if (el.audio.volume > 0) {
        el.audio.volume = 0;
        if (el.volumeSlider) el.volumeSlider.value = 0;
      } else {
        el.audio.volume = state.player.volume || 1.0;
        if (el.volumeSlider) el.volumeSlider.value = el.audio.volume;
      }
      updateVolumeIcon();
    });
  }

  // Toggle remaining time vs total duration on click
  if (el.timeTotal) {
    el.timeTotal.addEventListener('click', () => {
      state.player.showRemainingTime = !state.player.showRemainingTime;
      onTimeUpdate();
    });
  }

  // Scrub bar interaction
  if (el.scrubWrapper) {
    el.scrubWrapper.addEventListener('click', (e) => {
      if (!state.player.duration) return;
      const rect = el.scrubWrapper.getBoundingClientRect();
      const pos = (e.clientX - rect.left) / rect.width;
      const seekTime = Math.max(0, Math.min(pos * state.player.duration, state.player.duration));
      seekTo(seekTime);
    });
  }

  // Audio element events
  if (el.audio) {
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
  }

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

    // Update Model Provenance Banner
    if (el.activeModelProvenanceBanner) {
      const modelInfo = fullItem.model_info || meta.model_info;
      const tags = fullItem.tags || meta.tags || [];
      const isSeparated = fullItem.source_type === "separation" || tags.includes("separated") || (modelInfo && modelInfo.model_type);
      const isCut = fullItem.source_type === "cut" || tags.includes("cut");

      if (isSeparated) {
        el.activeModelProvenanceBanner.classList.remove('hidden');
        const modelLabel = modelInfo?.model_label || (
          tags.includes('htdemucs_ft') ? 'HTDemucs (Fine-Tuned)' :
          tags.includes('htdemucs') ? 'HTDemucs (Default v4)' :
          tags.includes('bs_roformer') ? 'BS-RoFormer (SOTA)' :
          tags.includes('mel_roformer') ? 'Mel-RoFormer (Mel-Band)' :
          tags.includes('mvsep_mdx23') ? 'MVSep MDX23' : 'Separation Model'
        );
        const stem = modelInfo?.stem ? modelInfo.stem.toUpperCase() : 'VOCALS';
        const parentTitle = modelInfo?.parent_title || (fullItem.parent_id ? `Parent Audio ${fullItem.parent_id}` : 'source clip');
        if (el.provModelBadge) el.provModelBadge.textContent = modelLabel;
        if (el.provDetailsText) el.provDetailsText.textContent = `${stem} Stem separated from "${parentTitle}"`;
      } else if (isCut) {
        el.activeModelProvenanceBanner.classList.remove('hidden');
        if (el.provModelBadge) el.provModelBadge.textContent = 'Audio Cut / Snippet';
        if (el.provDetailsText) el.provDetailsText.textContent = `Prepared snippet with background music`;
      } else {
        el.activeModelProvenanceBanner.classList.add('hidden');
      }
    }

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

  if (viewport) {
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
      if (el.timeTooltip) {
        if (e.clientX >= rect.left && e.clientX <= rect.right && e.clientY >= rect.top && e.clientY <= rect.bottom) {
          const hoverTime = (currentX / rect.width) * (state.activeAudio.duration_s || 1);
          el.timeTooltip.classList.remove('hidden');
          el.timeTooltip.textContent = formatTimePrecise(hoverTime);
          el.timeTooltip.style.left = `${Math.min(currentX, rect.width - 60)}px`;
        } else {
          el.timeTooltip.classList.add('hidden');
        }
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
  }

  // Audition Selection button
  if (el.btnAuditionSelection) {
    el.btnAuditionSelection.addEventListener('click', () => {
      if (!state.activeAudio || !state.selection.active) return;
      seekTo(state.selection.start);
      state.player.previewEnd = state.selection.end;
      if (el.audio) el.audio.play();
      showToast(`Auditioning selection: ${state.selection.start.toFixed(2)}s to ${state.selection.end.toFixed(2)}s`, "info");
    });
  }

  // Clear Selection button
  if (el.btnClearSelection) {
    el.btnClearSelection.addEventListener('click', clearSelection);
  }

  // Zoom controls
  if (el.btnZoomIn) el.btnZoomIn.addEventListener('click', () => setZoom(state.zoom * 1.5));
  if (el.btnZoomOut) el.btnZoomOut.addEventListener('click', () => setZoom(Math.max(1.0, state.zoom / 1.5)));
  if (el.btnResetZoom) el.btnResetZoom.addEventListener('click', () => setZoom(1.0));

  // Spectrogram Toggle
  if (el.btnToggleSpec) el.btnToggleSpec.addEventListener('click', toggleSpectrogramPanel);
  if (el.btnRefreshSpec) el.btnRefreshSpec.addEventListener('click', loadSpectrogramImage);

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
      const data = await parseJsonResponse(res);

      showToast(`Audio cut successful! Created new clip ${data.audio_id}`, "success");
      await fetchAudioList();
      addCutToRegistry(data.audio_id, start, end);
      await setActiveAudio(data.audio_id, { play: true });
    } catch (err) {
      showToast(err.message, "error");
    } finally {
      el.btnApplyCut.disabled = false;
      el.btnApplyCut.innerHTML = `<span>Apply Cut (Session)</span>`;
    }
  });

  // Cut & Send to Audition Hub
  if (el.btnCutAndAudition) {
    el.btnCutAndAudition.addEventListener('click', async () => {
      if (!state.activeAudio) {
        showToast("Please load an audio file first", "warning");
        return;
      }
      const start = el.cutStartInput.value.trim();
      const end = el.cutEndInput.value.trim();
      const unit = document.querySelector('input[name="cut_unit"]:checked').value;

      el.btnCutAndAudition.disabled = true;
      el.btnCutAndAudition.textContent = "Cutting...";

      try {
        const res = await fetch(`/api/audio/${state.activeAudio.id}/cut`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ start, end, unit }),
        });
        const data = await parseJsonResponse(res);

        showToast(`Snippet created! Switching to Audition & Scoring Hub...`, "success");
        await fetchAudioList();
        addCutToRegistry(data.audio_id, start, end);
        switchTab('tab-comparison');
        await loadClipForAudition(data.audio_id);
      } catch (err) {
        showToast(err.message, "error");
      } finally {
        el.btnCutAndAudition.disabled = false;
        el.btnCutAndAudition.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="6" cy="6" r="3"></circle><circle cx="6" cy="18" r="3"></circle><line x1="20" y1="4" x2="8.12" y2="15.88"></line><line x1="14.47" y1="14.48" x2="20" y2="20"></line><line x1="8.12" y1="8.12" x2="12" y2="12"></line></svg> <span>✂️ Cut & Send to Audition</span>`;
      }
    });
  }

  // Cut & Run All Demucs Models Suite
  if (el.btnCutAndRunModels) {
    el.btnCutAndRunModels.addEventListener('click', async () => {
      if (!state.activeAudio) {
        showToast("Please load an audio file first", "warning");
        return;
      }
      const start = el.cutStartInput.value.trim();
      const end = el.cutEndInput.value.trim();
      const unit = document.querySelector('input[name="cut_unit"]:checked').value;

      el.btnCutAndRunModels.disabled = true;
      el.btnCutAndRunModels.textContent = "Cutting & Queuing Models...";

      try {
        const res = await fetch(`/api/audio/${state.activeAudio.id}/cut`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ start, end, unit }),
        });
        const data = await parseJsonResponse(res);
        addCutToRegistry(data.audio_id, start, end);
        await fetchAudioList();

        showToast(`Running batch separation models on snippet...`, "info");
        await runBatchMultiModelSeparation(data.audio_id, true);
      } catch (err) {
        showToast(err.message, "error");
      } finally {
        el.btnCutAndRunModels.disabled = false;
        el.btnCutAndRunModels.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg> <span>🚀 Cut & Run All Demucs Models</span>`;
      }
    });
  }
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
      const data = await parseJsonResponse(res);

      navigator.clipboard.writeText(data.saved_path);
      showToast(`Quick saved to: ${data.saved_path} (Path copied!)`, "success");
      await fetchAudioList();
    } catch (err) {
      showToast(err.message, "error");
    }
  });

  // Save To Modal
  function updateSavePresetPath(folder) {
    if (!state.activeAudio) return;
    const baseTitle = state.activeAudio.title || state.activeAudio.source_id || "audio";
    const cleanTitle = baseTitle.replace(/[^a-zA-Z0-9_\-\.]+/g, "_").replace(/^_+|_+$/g, "").substring(0, 80);
    const fmt = state.activeAudio.format || "wav";
    if (el.inputSavePath) {
      el.inputSavePath.value = `${folder}/${cleanTitle}.${fmt}`;
    }
  }

  el.btnSaveToDialog.addEventListener('click', () => {
    if (!state.activeAudio) {
      showToast("Please load an audio file first", "warning");
      return;
    }
    updateSavePresetPath("benchmarks/separation/sources/speech");
    el.modalSaveTo.classList.remove('hidden');
    if (el.saveTargetPresets) {
      el.saveTargetPresets.querySelectorAll('.save-preset-chip').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.folder === 'benchmarks/separation/sources/speech');
      });
    }
  });

  if (el.saveTargetPresets) {
    el.saveTargetPresets.addEventListener('click', (e) => {
      const chip = e.target.closest('.save-preset-chip');
      if (!chip) return;
      el.saveTargetPresets.querySelectorAll('.save-preset-chip').forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      const folder = chip.dataset.folder || "benchmarks/separation/sources/speech";
      updateSavePresetPath(folder);
    });
  }

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
      const data = await parseJsonResponse(res);

      showToast(`Saved file & metadata to: ${data.saved_path}`, "success");
      el.modalSaveTo.classList.add('hidden');
      await fetchAudioList();
      await fetchServerFiles();
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

  // Run Single Separation Button
  el.btnRunSeparation.addEventListener('click', async () => {
    const audioId = el.sepInputSelect.value;
    if (!audioId) {
      showToast("Please select an input audio to separate", "error");
      return;
    }

    const activeCard = document.querySelector('.model-card[data-model].active');
    const modelType = activeCard ? activeCard.dataset.model : "htdemucs";
    const variant = activeCard ? activeCard.dataset.variant : undefined;
    const device = el.sepDeviceSelect.value;
    const twoStems = el.sepStemsSelect.value;
    const modelName = variant || el.roformerCheckpointInput.value.trim() || undefined;

    el.btnRunSeparation.disabled = true;
    el.sepTaskProgressBox.classList.remove('hidden');
    el.sepTaskTitle.textContent = `Running ${modelType.toUpperCase()} (${modelName || 'default'}) separation...`;

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
      const data = await parseJsonResponse(res);

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

  // Run Multi-Model Comparison Suite Button
  if (el.btnRunMultiSeparation) {
    el.btnRunMultiSeparation.addEventListener('click', async () => {
      const audioId = el.sepInputSelect.value;
      if (!audioId) {
        showToast("Please select an input audio to benchmark", "error");
        return;
      }
      await runBatchMultiModelSeparation(audioId, true);
    });
  }
}

function renderSeparationResultCard(result) {
  const list = el.sepResultsList;
  if (!list) return;
  if (list.querySelector('.empty-placeholder')) {
    list.innerHTML = "";
  }

  const meta = result.metadata || {};
  const audioId = result.separated_audio_id;
  const modelLabel = result.model_label || result.model_type || "Demucs";

  const card = document.createElement("div");
  card.className = "stem-result-card";
  card.innerHTML = `
    <div class="stem-info">
      <div style="display: flex; align-items: center; gap: 0.5rem;">
        <span class="badge badge-accent">${escapeHtml(modelLabel)}</span>
        <span class="stem-title">${escapeHtml(meta.title || audioId)}</span>
      </div>
      <span class="stem-meta">${(meta.format || "WAV").toUpperCase()} • ${(meta.sample_rate || 44100).toLocaleString()}Hz • ${(meta.duration_s || 0).toFixed(2)}s • ${result.elapsed_s || 0}s separation</span>
    </div>
    <div class="stem-actions">
      <button class="btn btn-sm btn-secondary btn-play-stem" data-id="${audioId}">▶ Play</button>
      <button class="btn btn-sm btn-secondary btn-load-workspace" data-id="${audioId}">🎛️ Workspace</button>
      <button class="btn btn-sm btn-primary btn-send-compare" data-id="${audioId}">⚖️ Side-by-Side Deck</button>
    </div>
  `;

  card.querySelector('.btn-play-stem').addEventListener('click', () => loadAudioIntoPlayer(audioId, true));
  card.querySelector('.btn-load-workspace').addEventListener('click', () => {
    switchTab('tab-workspace');
    setActiveAudio(audioId, { play: true });
  });
  card.querySelector('.btn-send-compare').addEventListener('click', async () => {
    switchTab('tab-comparison');
    const audioItem = state.audioList.find(a => a.id === audioId);
    const parentId = audioItem?.parent_id || state.activeAudio?.id;
    if (parentId) {
      await loadClipForAudition(parentId);
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

// ==================== CUTS MANAGER ====================

function initCutsManager() {
  renderCutsTable();
}

function addCutToRegistry(audioId, start, end) {
  const audio = state.audioList.find(a => a.id === audioId) || {
    id: audioId,
    title: state.activeAudio ? `${state.activeAudio.title}_cut_${start}_${end}` : audioId,
    duration_s: parseFloat(end) - parseFloat(start),
  };
  if (!state.cuts) state.cuts = [];
  state.cuts.unshift({
    id: audioId,
    title: audio.title || audioId,
    parentId: state.activeAudio ? state.activeAudio.id : null,
    start: parseFloat(start),
    end: parseFloat(end),
    duration: Math.max(0, parseFloat(end) - parseFloat(start)),
    created: Date.now(),
  });
  renderCutsTable();
}

function renderCutsTable() {
  if (!el.cutsTableBody) return;
  const parentId = state.activeAudio ? state.activeAudio.id : null;
  const cuts = (state.cuts || []).filter(c => !parentId || c.parentId === parentId || c.id === parentId);

  if (el.cutsCounterBadge) {
    el.cutsCounterBadge.textContent = `${cuts.length} Cut${cuts.length === 1 ? '' : 's'}`;
  }

  if (cuts.length === 0) {
    el.cutsTableBody.innerHTML = `<tr><td colspan="5" class="empty-table-msg">No cuts generated yet. Set start/end bounds above and click "✂️ Cut & Send to Audition".</td></tr>`;
    return;
  }

  el.cutsTableBody.innerHTML = "";
  cuts.forEach(cut => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><span class="file-name-text">${escapeHtml(cut.title)}</span></td>
      <td><code>${cut.start.toFixed(2)}s – ${cut.end.toFixed(2)}s</code></td>
      <td>${cut.duration.toFixed(2)}s</td>
      <td><span class="badge badge-sm badge-success">Ready</span></td>
      <td class="table-actions">
        <button class="btn btn-sm btn-ghost btn-play-cut" title="Play Cut">▶ Play</button>
        <button class="btn btn-sm btn-primary btn-audition-cut" title="Audition & Score Models">⚖️ Audition</button>
        <button class="btn btn-sm btn-accent btn-batch-cut" title="Run Demucs Models">🚀 Run Models</button>
      </td>
    `;
    tr.querySelector('.btn-play-cut').addEventListener('click', () => loadAudioIntoPlayer(cut.id, true));
    tr.querySelector('.btn-audition-cut').addEventListener('click', async () => {
      switchTab('tab-comparison');
      await loadClipForAudition(cut.id);
    });
    tr.querySelector('.btn-batch-cut').addEventListener('click', async () => {
      await runBatchMultiModelSeparation(cut.id, true);
    });
    el.cutsTableBody.appendChild(tr);
  });
}

// ==================== MULTI-MODEL AUDITION & SCORING HUB ====================

const auditionAudio = new Audio();
let auditionTracks = [];
let activeAuditionIndex = 0;
let activeScoreOverall = 5.0;
let activeScoreClarity = 5;
let activeScoreBleed = 5;
let activeScoreArtifacts = 5;
let activeSelectedTags = new Set();

function initAuditionHub() {
  auditionAudio.loop = true;
  auditionAudio.volume = 1.0;

  if (el.auditionClipSelect) {
    el.auditionClipSelect.addEventListener('change', (e) => {
      if (e.target.value) {
        loadClipForAudition(e.target.value);
      }
    });
  }

  if (el.btnAuditionPlay) {
    el.btnAuditionPlay.addEventListener('click', toggleAuditionPlay);
  }

  if (el.btnAuditionLoop) {
    el.btnAuditionLoop.addEventListener('click', () => {
      auditionAudio.loop = !auditionAudio.loop;
      el.btnAuditionLoop.classList.toggle('active', auditionAudio.loop);
      showToast(auditionAudio.loop ? "Loop playback enabled" : "Loop disabled", "info");
    });
    el.btnAuditionLoop.classList.add('active');
  }

  if (el.auditionVolumeSlider) {
    el.auditionVolumeSlider.addEventListener('input', (e) => {
      auditionAudio.volume = parseFloat(e.target.value);
    });
  }

  if (el.auditionScrubber) {
    el.auditionScrubber.addEventListener('input', (e) => {
      const pct = parseFloat(e.target.value);
      if (auditionAudio.duration) {
        auditionAudio.currentTime = (pct / 100) * auditionAudio.duration;
      }
    });
  }

  auditionAudio.addEventListener('timeupdate', () => {
    if (el.auditionTimeCurrent) {
      el.auditionTimeCurrent.textContent = formatTimePrecise(auditionAudio.currentTime);
    }
    if (el.auditionScrubber && auditionAudio.duration) {
      el.auditionScrubber.value = (auditionAudio.currentTime / auditionAudio.duration) * 100;
    }
  });

  auditionAudio.addEventListener('loadedmetadata', () => {
    if (el.auditionTimeTotal) {
      el.auditionTimeTotal.textContent = formatTimePrecise(auditionAudio.duration);
    }
  });

  auditionAudio.addEventListener('play', () => {
    if (el.iconAuditionPlay) el.iconAuditionPlay.classList.add('hidden');
    if (el.iconAuditionPause) el.iconAuditionPause.classList.remove('hidden');
  });

  auditionAudio.addEventListener('pause', () => {
    if (el.iconAuditionPlay) el.iconAuditionPlay.classList.remove('hidden');
    if (el.iconAuditionPause) el.iconAuditionPause.classList.add('hidden');
  });

  if (el.btnBatchSeparateActiveClip) {
    el.btnBatchSeparateActiveClip.addEventListener('click', async () => {
      const clipId = el.auditionClipSelect.value;
      if (!clipId) {
        showToast("Please select an audio clip first", "warning");
        return;
      }
      await runBatchMultiModelSeparation(clipId, false);
    });
  }

  if (el.btnGenerateComparison) {
    el.btnGenerateComparison.addEventListener('click', async () => {
      if (auditionTracks.length < 2) {
        showToast("Need at least original clip and 1 separated model stem to render comparison", "warning");
        return;
      }
      const origTrack = auditionTracks[0];
      const modelTrack = auditionTracks[activeAuditionIndex] || auditionTracks[1];

      el.btnGenerateComparison.disabled = true;
      el.btnGenerateComparison.textContent = "Rendering Mel Spectrograms...";

      try {
        const specRes = await fetch("/api/compare/spectrogram", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ before_id: origTrack.id, after_id: modelTrack.id }),
        });
        if (!specRes.ok) throw new Error("Spectrogram render failed");
        const specBlob = await specRes.blob();
        el.imgCompareSpectrogram.src = URL.createObjectURL(specBlob);
        el.imgCompareSpectrogram.classList.remove('hidden');
        el.spectrogramCompareBox.querySelector('.empty-placeholder')?.remove();
        showToast("Aligned Mel Spectrograms rendered!", "success");
      } catch (err) {
        showToast(err.message, "error");
      } finally {
        el.btnGenerateComparison.disabled = false;
        el.btnGenerateComparison.innerHTML = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="18" cy="5" r="3"></circle><circle cx="6" cy="12" r="3"></circle><circle cx="18" cy="19" r="3"></circle><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"></line><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"></line></svg> <span>Render Mel Spectrograms</span>`;
      }
    });
  }

  if (el.starRatingWidget) {
    const starBtns = el.starRatingWidget.querySelectorAll('.star-btn');
    starBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        const val = parseFloat(btn.dataset.value);
        setStarRating(val);
      });
      btn.addEventListener('mouseenter', () => {
        const val = parseFloat(btn.dataset.value);
        highlightStars(val);
      });
    });
    el.starRatingWidget.addEventListener('mouseleave', () => {
      highlightStars(activeScoreOverall);
    });
  }

  if (el.sliderSubmetricClarity) {
    el.sliderSubmetricClarity.addEventListener('input', (e) => {
      activeScoreClarity = parseInt(e.target.value);
      if (el.valSubmetricClarity) el.valSubmetricClarity.textContent = `${activeScoreClarity} / 5`;
    });
  }
  if (el.sliderSubmetricBleed) {
    el.sliderSubmetricBleed.addEventListener('input', (e) => {
      activeScoreBleed = parseInt(e.target.value);
      if (el.valSubmetricBleed) el.valSubmetricBleed.textContent = `${activeScoreBleed} / 5`;
    });
  }
  if (el.sliderSubmetricArtifacts) {
    el.sliderSubmetricArtifacts.addEventListener('input', (e) => {
      activeScoreArtifacts = parseInt(e.target.value);
      if (el.valSubmetricArtifacts) el.valSubmetricArtifacts.textContent = `${activeScoreArtifacts} / 5`;
    });
  }

  if (el.evalTagChips) {
    el.evalTagChips.addEventListener('click', (e) => {
      const chip = e.target.closest('.tag-chip');
      if (!chip) return;
      const tag = chip.dataset.tag;
      if (activeSelectedTags.has(tag)) {
        activeSelectedTags.delete(tag);
        chip.classList.remove('selected');
      } else {
        activeSelectedTags.add(tag);
        chip.classList.add('selected');
      }
    });
  }

  if (el.btnSaveEvaluation) {
    el.btnSaveEvaluation.addEventListener('click', saveCurrentEvaluation);
  }

  if (el.btnCopyEvalNote) {
    el.btnCopyEvalNote.addEventListener('click', () => {
      const note = el.evalNotesInput ? el.evalNotesInput.value : "";
      if (!note) {
        showToast("Note is empty", "warning");
        return;
      }
      navigator.clipboard.writeText(note);
      showToast("Evaluation notes copied to clipboard!", "success");
    });
  }
}

function setStarRating(score) {
  activeScoreOverall = score;
  highlightStars(score);
  if (el.starScoreText) {
    el.starScoreText.textContent = `${score.toFixed(1)} / 5.0`;
  }
}

function highlightStars(score) {
  if (!el.starRatingWidget) return;
  const btns = el.starRatingWidget.querySelectorAll('.star-btn');
  btns.forEach(btn => {
    const val = parseFloat(btn.dataset.value);
    btn.classList.toggle('active', val <= score);
  });
}

function toggleAuditionPlay() {
  if (auditionAudio.paused) {
    if (!auditionAudio.src || auditionAudio.src === window.location.href) {
      if (auditionTracks.length > 0) {
        switchAuditionTrack(activeAuditionIndex);
      }
    }
    el.audio.pause();
    auditionAudio.play().catch(e => console.error("Audition playback error:", e));
  } else {
    auditionAudio.pause();
  }
}

async function loadClipForAudition(clipId) {
  if (!clipId) return;

  let clipAudio = state.audioList.find(a => a.id === clipId);
  if (!clipAudio) return;

  // Auto-resolve to parent audio if user passed a separated stem!
  const isSeparated = clipAudio.source_type === "separation" || (clipAudio.tags && clipAudio.tags.includes("separated"));
  if (isSeparated && clipAudio.parent_id) {
    const parent = state.audioList.find(a => a.id === clipAudio.parent_id);
    if (parent) {
      clipAudio = parent;
      clipId = parent.id;
    }
  }

  if (el.auditionClipSelect) el.auditionClipSelect.value = clipId;

  auditionTracks = [
    {
      id: clipAudio.id,
      title: clipAudio.title,
      label: "Original Cut (with BG Music)",
      modelId: "original",
      modelName: "Original Mixture",
      stem: "mixture",
      isOriginal: true,
      path: clipAudio.path,
    }
  ];

  const childTracks = state.audioList.filter(a => 
    a.id !== clipId && (
      a.parent_id === clipId || 
      (a.tags && a.tags.includes("separated") && (a.title.includes(clipAudio.source_id || "") || a.title.includes(clipAudio.title || ""))) ||
      (a.model_info && (a.model_info.parent_title === clipAudio.title || a.parent_id === clipId))
    )
  );

  childTracks.forEach(ct => {
    let modelName = ct.model_info?.model_label || "Demucs Separated";
    let modelId = ct.model_info?.model_name || ct.model_info?.model_type || "demucs";
    if (ct.tags) {
      if (ct.tags.includes("htdemucs_ft")) { modelName = "HTDemucs (Fine-Tuned)"; modelId = "htdemucs_ft"; }
      else if (ct.tags.includes("htdemucs")) { modelName = "HTDemucs (Default v4)"; modelId = "htdemucs"; }
      else if (ct.tags.includes("bs_roformer")) { modelName = "BS-RoFormer (SOTA)"; modelId = "bs_roformer"; }
      else if (ct.tags.includes("mel_roformer")) { modelName = "Mel-RoFormer (Mel-Band)"; modelId = "mel_roformer"; }
      else if (ct.tags.includes("mvsep_mdx23")) { modelName = "MVSep MDX23"; modelId = "mvsep_mdx23"; }
    }
    auditionTracks.push({
      id: ct.id,
      title: ct.title,
      label: modelName,
      modelId: modelId,
      modelName: modelName,
      stem: ct.model_info?.stem || "vocals",
      isOriginal: false,
      path: ct.path,
    });
  });

  renderAuditionTrackPills();
  renderSideBySideDeck();
  switchAuditionTrack(0);
}

function renderAuditionTrackPills() {
  if (!el.auditionTrackPills) return;
  el.auditionTrackPills.innerHTML = "";

  auditionTracks.forEach((track, idx) => {
    const btn = document.createElement("button");
    btn.className = `track-pill-btn ${idx === activeAuditionIndex ? 'active' : ''}`;
    btn.dataset.index = idx;
    
    const existing = (state.evaluations || []).find(e => e.clip_id === (auditionTracks[0]?.id) && e.model_id === track.modelId);
    const scoreBadgeHtml = existing ? ` <span style="color:#fbbf24; font-weight:700;">★ ${existing.score_overall}</span>` : "";

    btn.innerHTML = `
      <span class="track-pill-key">${idx + 1}</span>
      <span>${escapeHtml(track.label)}${scoreBadgeHtml}</span>
    `;

    btn.addEventListener('click', () => switchAuditionTrack(idx));
    el.auditionTrackPills.appendChild(btn);
  });
}

function renderSideBySideDeck() {
  if (!el.sbsColumnsDeck) return;
  el.sbsColumnsDeck.innerHTML = "";

  if (!auditionTracks || auditionTracks.length === 0) {
    el.sbsColumnsDeck.innerHTML = `
      <div class="empty-placeholder">
        <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
        <span>Select an audio cut snippet above to view model outputs side-by-side.</span>
      </div>
    `;
    if (el.sbsModelsCountBadge) el.sbsModelsCountBadge.textContent = "0 Stems";
    return;
  }

  if (el.sbsModelsCountBadge) {
    el.sbsModelsCountBadge.textContent = `${auditionTracks.length} Stems Aligned`;
  }

  const clipId = auditionTracks[0]?.id;

  auditionTracks.forEach((track, idx) => {
    const card = document.createElement("div");
    card.className = `sbs-model-card ${idx === activeAuditionIndex ? 'active-audition' : ''}`;
    card.dataset.index = idx;

    const existing = (state.evaluations || []).find(e => e.clip_id === clipId && e.model_id === track.modelId);
    const audioObj = state.audioList.find(a => a.id === track.id) || {};
    const sampleRate = audioObj.sample_rate ? `${audioObj.sample_rate.toLocaleString()}Hz` : '44.1kHz';
    const dur = audioObj.duration_s ? `${audioObj.duration_s.toFixed(2)}s` : '--';
    const elapsed = audioObj.model_info?.elapsed_s ? `⏱️ ${audioObj.model_info.elapsed_s}s` : (track.isOriginal ? 'Input Clip' : 'Separated');

    const score = existing ? existing.score_overall : 5.0;
    const notes = existing ? (existing.notes || "") : "";

    let badgeClass = "badge-ghost";
    if (track.modelId === "htdemucs_ft") badgeClass = "badge-accent";
    else if (track.modelId === "htdemucs") badgeClass = "badge-primary";
    else if (track.modelId === "bs_roformer") badgeClass = "badge-info";
    else if (track.modelId === "mel_roformer") badgeClass = "badge-secondary";
    else if (track.isOriginal) badgeClass = "badge-warning";

    card.innerHTML = `
      <div class="sbs-card-header">
        <div class="sbs-title-group">
          <div class="sbs-model-name">${escapeHtml(track.label)}</div>
          <div class="sbs-model-desc">${track.isOriginal ? 'Original Mixture (with BG Music)' : (track.stem ? `Stem: ${track.stem}` : 'Estimated Stem')}</div>
        </div>
        <span class="badge ${badgeClass}">${track.isOriginal ? 'Reference' : (track.modelName || 'Model')}</span>
      </div>

      <button class="sbs-solo-btn ${idx === activeAuditionIndex ? 'active' : ''}" data-index="${idx}" title="Listen solo at current position (Shortcut: ${idx + 1})">
        <span>${idx === activeAuditionIndex ? '🔊 Active / Auditioning' : '▶ Solo / Audition'}</span>
        <kbd class="kbd" style="font-size: 0.7rem;">${idx + 1}</kbd>
      </button>

      <div class="sbs-meta-strip">
        <span>${sampleRate} • Mono</span>
        <span>${dur}</span>
        <span>${elapsed}</span>
      </div>

      ${!track.isOriginal ? `
        <div class="sbs-score-section">
          <div class="sbs-score-header">
            <span class="sbs-score-title">Human Quality Score:</span>
            <span class="sbs-score-val" id="sbs-score-val-${idx}">${score.toFixed(1)} / 5</span>
          </div>
          <div class="sbs-stars-row" data-index="${idx}">
            ${[1, 2, 3, 4, 5].map(v => `<button class="sbs-star-btn ${v <= score ? 'active' : ''}" data-val="${v}" data-index="${idx}">★</button>`).join('')}
          </div>
          <textarea class="sbs-notes-input" id="sbs-notes-${idx}" placeholder="Observation on vocal clarity, BG suppression, artifacts..." rows="2">${escapeHtml(notes)}</textarea>
          <div style="display: flex; justify-content: flex-end; margin-top: 4px;">
            <button class="btn btn-xs btn-primary sbs-btn-save" data-index="${idx}">💾 Save Score</button>
          </div>
        </div>
      ` : `
        <div class="sbs-score-section" style="opacity: 0.8; text-align: center; font-size: 0.78rem; padding: 1.2rem 0.5rem;">
          <span>🎯 <strong>Ground Reference</strong><br>Use this to compare suppression against original audio.</span>
        </div>
      `}

      <div class="sbs-card-footer">
        <span class="sbs-card-badge">${escapeHtml(track.title || track.id)}</span>
        <button class="btn btn-xs btn-ghost sbs-load-workspace" data-audio-id="${track.id}" title="Open in Workspace Studio">🎛️ Open</button>
      </div>
    `;

    // Solo button listener
    card.querySelector('.sbs-solo-btn').addEventListener('click', () => {
      switchAuditionTrack(idx);
    });

    // Star rating buttons listener
    card.querySelectorAll('.sbs-star-btn').forEach(starBtn => {
      starBtn.addEventListener('click', (e) => {
        const val = parseFloat(e.target.dataset.val);
        card.querySelectorAll('.sbs-star-btn').forEach(b => {
          b.classList.toggle('active', parseFloat(b.dataset.val) <= val);
        });
        const valEl = card.querySelector(`#sbs-score-val-${idx}`);
        if (valEl) valEl.textContent = `${val.toFixed(1)} / 5`;
        if (idx === activeAuditionIndex) {
          setStarRating(val);
        }
      });
    });

    // Save score listener
    const saveBtn = card.querySelector('.sbs-btn-save');
    if (saveBtn) {
      saveBtn.addEventListener('click', async () => {
        const activeStars = card.querySelectorAll('.sbs-star-btn.active');
        const scoreVal = activeStars.length > 0 ? parseFloat(activeStars[activeStars.length - 1].dataset.val) : 5.0;
        const notesText = card.querySelector(`#sbs-notes-${idx}`)?.value || "";

        const payload = {
          clip_id: clipId,
          model_id: track.modelId,
          model_name: track.modelName || track.label,
          stem_type: track.stem || "vocals",
          score_overall: scoreVal,
          score_vocal_clarity: scoreVal,
          score_bleed: scoreVal,
          score_artifacts: scoreVal,
          tags: Array.from(activeSelectedTags),
          notes: notesText,
          stem_audio_id: track.id,
        };

        try {
          const res = await fetch('/api/evaluations', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
          });
          if (!res.ok) throw new Error("Failed to save evaluation");
          showToast(`Score saved for ${track.label}!`, "success");
          await fetchEvaluations();
          renderAuditionTrackPills();
          renderSideBySideDeck();
        } catch (err) {
          showToast(err.message, "error");
        }
      });
    }

    // Open in workspace listener
    card.querySelector('.sbs-load-workspace')?.addEventListener('click', () => {
      setActiveAudio(track.id);
      switchTab('tab-workspace');
    });

    el.sbsColumnsDeck.appendChild(card);
  });
}

function switchAuditionTrack(idx) {
  if (idx < 0 || idx >= auditionTracks.length) return;
  activeAuditionIndex = idx;
  const track = auditionTracks[idx];

  const wasPlaying = !auditionAudio.paused;
  const curTime = auditionAudio.currentTime;

  auditionAudio.src = `/api/audio/${track.id}/stream`;
  auditionAudio.currentTime = curTime;
  if (wasPlaying) {
    auditionAudio.play().catch(e => console.error("Playback switch error:", e));
  }

  if (el.activeAuditionTrackName) {
    el.activeAuditionTrackName.textContent = track.label;
  }
  if (el.scoringActiveModelLabel) {
    el.scoringActiveModelLabel.textContent = `Evaluating: ${track.label} — ${track.stem || 'vocals'}`;
  }

  if (el.auditionTrackPills) {
    el.auditionTrackPills.querySelectorAll('.track-pill-btn').forEach((btn, i) => {
      btn.classList.toggle('active', i === idx);
    });
  }

  if (el.sbsColumnsDeck) {
    el.sbsColumnsDeck.querySelectorAll('.sbs-model-card').forEach((card, i) => {
      const isCurrent = i === idx;
      card.classList.toggle('active-audition', isCurrent);
      const soloBtn = card.querySelector('.sbs-solo-btn');
      if (soloBtn) {
        soloBtn.classList.toggle('active', isCurrent);
        soloBtn.querySelector('span').textContent = isCurrent ? '🔊 Active / Auditioning' : '▶ Solo / Audition';
      }
    });
  }

  const clipId = auditionTracks[0]?.id;
  const existing = (state.evaluations || []).find(e => e.clip_id === clipId && e.model_id === track.modelId);
  if (existing) {
    setStarRating(existing.score_overall);
    if (el.sliderSubmetricClarity) {
      el.sliderSubmetricClarity.value = existing.score_vocal_clarity;
      if (el.valSubmetricClarity) el.valSubmetricClarity.textContent = `${existing.score_vocal_clarity} / 5`;
    }
    if (el.sliderSubmetricBleed) {
      el.sliderSubmetricBleed.value = existing.score_bleed;
      if (el.valSubmetricBleed) el.valSubmetricBleed.textContent = `${existing.score_bleed} / 5`;
    }
    if (el.sliderSubmetricArtifacts) {
      el.sliderSubmetricArtifacts.value = existing.score_artifacts;
      if (el.valSubmetricArtifacts) el.valSubmetricArtifacts.textContent = `${existing.score_artifacts} / 5`;
    }
    if (el.evalNotesInput) el.evalNotesInput.value = existing.notes || "";
    activeSelectedTags = new Set(existing.tags || []);
    if (el.evalTagChips) {
      el.evalTagChips.querySelectorAll('.tag-chip').forEach(c => {
        c.classList.toggle('selected', activeSelectedTags.has(c.dataset.tag));
      });
    }
    if (el.currentEvalScoreBadge) {
      el.currentEvalScoreBadge.textContent = `★ ${existing.score_overall.toFixed(1)} Saved`;
      el.currentEvalScoreBadge.style.color = "#4ade80";
    }
  } else {
    setStarRating(5.0);
    if (el.sliderSubmetricClarity) {
      el.sliderSubmetricClarity.value = 5;
      if (el.valSubmetricClarity) el.valSubmetricClarity.textContent = "5 / 5";
    }
    if (el.sliderSubmetricBleed) {
      el.sliderSubmetricBleed.value = 5;
      if (el.valSubmetricBleed) el.valSubmetricBleed.textContent = "5 / 5";
    }
    if (el.sliderSubmetricArtifacts) {
      el.sliderSubmetricArtifacts.value = 5;
      if (el.valSubmetricArtifacts) el.valSubmetricArtifacts.textContent = "5 / 5";
    }
    if (el.evalNotesInput) el.evalNotesInput.value = "";
    activeSelectedTags.clear();
    if (el.evalTagChips) {
      el.evalTagChips.querySelectorAll('.tag-chip').forEach(c => c.classList.remove('selected'));
    }
    if (el.currentEvalScoreBadge) {
      el.currentEvalScoreBadge.textContent = "Unrated";
      el.currentEvalScoreBadge.style.color = "#fbbf24";
    }
  }
}

async function saveCurrentEvaluation() {
  if (auditionTracks.length === 0) {
    showToast("No track selected for evaluation", "error");
    return;
  }
  const origClip = auditionTracks[0];
  const activeTrack = auditionTracks[activeAuditionIndex];

  if (!activeTrack) return;

  const payload = {
    clip_id: origClip.id,
    clip_title: origClip.title,
    clip_path: origClip.path,
    model_id: activeTrack.modelId,
    model_name: activeTrack.modelName,
    stem: activeTrack.stem || "vocals",
    separated_audio_id: activeTrack.id,
    separated_audio_path: activeTrack.path,
    score_overall: activeScoreOverall,
    score_vocal_clarity: activeScoreClarity,
    score_bleed: activeScoreBleed,
    score_artifacts: activeScoreArtifacts,
    notes: el.evalNotesInput ? el.evalNotesInput.value.trim() : "",
    tags: Array.from(activeSelectedTags),
  };

  try {
    const res = await fetch("/api/evaluations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await parseJsonResponse(res);

    showToast(`Saved score (${activeScoreOverall.toFixed(1)} ★) and notes for ${activeTrack.label}!`, "success");
    if (el.currentEvalScoreBadge) {
      el.currentEvalScoreBadge.textContent = `★ ${activeScoreOverall.toFixed(1)} Saved`;
      el.currentEvalScoreBadge.style.color = "#4ade80";
    }
    await fetchEvaluations();
    renderAuditionTrackPills();
  } catch (err) {
    showToast(err.message, "error");
  }
}

async function runBatchMultiModelSeparation(audioId, jumpToAudition = false) {
  if (!audioId) return;

  const models = [
    { model_type: "htdemucs", model_name: "htdemucs", label: "HTDemucs (Default)" },
    { model_type: "htdemucs", model_name: "htdemucs_ft", label: "HTDemucs (Fine-Tuned)" },
    { model_type: "bs_roformer", model_name: null, label: "BS-RoFormer" },
    { model_type: "mel_roformer", model_name: null, label: "Mel-RoFormer" },
  ];

  showToast(`Initiated batch separation on models...`, "info");

  try {
    const res = await fetch("/api/separation/batch-compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        audio_id: audioId,
        models: models,
        two_stems: "vocals",
      }),
    });
    const data = await parseJsonResponse(res);

    pollTask(data.task_id, async (result) => {
      showToast(`Batch separation complete for '${result.clip_title}'!`, "success");
      await fetchAudioList();
      if (jumpToAudition) {
        switchTab('tab-comparison');
      }
      await loadClipForAudition(audioId);
    }, (err) => {
      showToast(`Batch separation error: ${err}`, "error");
    });
  } catch (err) {
    showToast(err.message, "error");
  }
}

// ==================== EVALUATION MATRIX & EXPORTS ====================

function initEvaluationMatrix() {
  if (el.btnExportEvalCsv) {
    el.btnExportEvalCsv.addEventListener('click', () => {
      window.location.href = "/api/evaluations/export?format=csv";
      showToast("Downloading evaluations CSV...", "info");
    });
  }

  if (el.btnExportEvalJson) {
    el.btnExportEvalJson.addEventListener('click', () => {
      window.location.href = "/api/evaluations/export?format=json";
      showToast("Downloading evaluations JSON...", "info");
    });
  }

  if (el.btnRefreshEvalMatrix) {
    el.btnRefreshEvalMatrix.addEventListener('click', fetchEvaluations);
  }

  if (el.evalSearchInput) {
    el.evalSearchInput.addEventListener('input', renderEvaluationsTable);
  }

  if (el.evalModelFilterPills) {
    el.evalModelFilterPills.addEventListener('click', (e) => {
      const pill = e.target.closest('.filter-pill');
      if (!pill) return;
      el.evalModelFilterPills.querySelectorAll('.filter-pill').forEach(p => p.classList.remove('active'));
      pill.classList.add('active');
      renderEvaluationsTable();
    });
  }
}

async function fetchEvaluations() {
  try {
    const res = await fetch("/api/evaluations");
    const data = await parseJsonResponse(res);
    state.evaluations = data.evaluations || [];
    updateMatrixKPIs();
    renderEvaluationsTable();
  } catch (err) {
    console.error("Failed to fetch evaluations:", err);
  }
}

function updateMatrixKPIs() {
  const evals = state.evaluations || [];
  if (el.kpiTotalEvals) el.kpiTotalEvals.textContent = evals.length;

  const uniqueClips = new Set(evals.map(e => e.clip_id)).size;
  if (el.kpiTotalClips) el.kpiTotalClips.textContent = uniqueClips;

  if (evals.length > 0) {
    const avg = evals.reduce((sum, e) => sum + (e.score_overall || 0), 0) / evals.length;
    if (el.kpiAvgScore) el.kpiAvgScore.textContent = `${avg.toFixed(2)} ★`;

    const modelScores = {};
    evals.forEach(e => {
      if (!modelScores[e.model_name]) modelScores[e.model_name] = [];
      modelScores[e.model_name].push(e.score_overall || 0);
    });

    let bestModel = "—";
    let bestAvg = -1;
    for (const [mName, scores] of Object.entries(modelScores)) {
      const mAvg = scores.reduce((a, b) => a + b, 0) / scores.length;
      if (mAvg > bestAvg) {
        bestAvg = mAvg;
        bestModel = `${mName} (${bestAvg.toFixed(1)}★)`;
      }
    }
    if (el.kpiTopModel) el.kpiTopModel.textContent = bestModel;
  } else {
    if (el.kpiAvgScore) el.kpiAvgScore.textContent = "0.0 ★";
    if (el.kpiTopModel) el.kpiTopModel.textContent = "—";
  }
}

function renderEvaluationsTable() {
  if (!el.evaluationsTableBody) return;
  const evals = state.evaluations || [];
  const query = el.evalSearchInput ? el.evalSearchInput.value.toLowerCase().trim() : "";
  const activePill = el.evalModelFilterPills ? el.evalModelFilterPills.querySelector('.filter-pill.active') : null;
  const filter = activePill ? activePill.dataset.filter : "all";

  const filtered = evals.filter(e => {
    const matchesQuery = !query ||
      (e.clip_title || "").toLowerCase().includes(query) ||
      (e.model_name || "").toLowerCase().includes(query) ||
      (e.notes || "").toLowerCase().includes(query) ||
      (e.tags || []).some(t => String(t).toLowerCase().includes(query));

    const matchesModel = filter === "all" || (e.model_id || "").toLowerCase().includes(filter);
    return matchesQuery && matchesModel;
  });

  if (filtered.length === 0) {
    el.evaluationsTableBody.innerHTML = `<tr><td colspan="9" class="empty-table-msg">No evaluations match your search filter.</td></tr>`;
    return;
  }

  el.evaluationsTableBody.innerHTML = "";
  filtered.forEach(item => {
    const tr = document.createElement("tr");
    const dateStr = item.updated_at ? new Date(item.updated_at * 1000).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : "—";
    const tagsHtml = (item.tags || []).map(t => `<span class="history-tag" style="margin-right:3px;">#${escapeHtml(t)}</span>`).join("");

    tr.innerHTML = `
      <td>
        <div style="font-weight:600; color:var(--text-primary);">${escapeHtml(item.clip_title || item.clip_id)}</div>
      </td>
      <td>
        <span class="badge badge-accent">${escapeHtml(item.model_name || item.model_id)}</span>
      </td>
      <td>
        <span style="color:#fbbf24; font-weight:800; font-size:1rem;">★ ${Number(item.score_overall).toFixed(1)}</span>
      </td>
      <td><code>${item.score_vocal_clarity}/5</code></td>
      <td><code>${item.score_bleed}/5</code></td>
      <td><code>${item.score_artifacts}/5</code></td>
      <td>
        <div style="max-width:280px; font-size:0.82rem; color:var(--text-secondary); line-height:1.3;">
          ${escapeHtml(item.notes || "—")}
          <div style="margin-top:4px;">${tagsHtml}</div>
        </div>
      </td>
      <td><span style="font-size:0.75rem; color:var(--text-muted);">${dateStr}</span></td>
      <td class="table-actions">
        <button class="btn btn-sm btn-ghost btn-play-stem" title="Play Stem">▶</button>
        <button class="btn btn-sm btn-secondary btn-rescore" title="Open in Audition Hub">⚖️</button>
        <button class="btn btn-sm btn-danger btn-delete-eval" title="Delete record">🗑️</button>
      </td>
    `;

    tr.querySelector('.btn-play-stem').addEventListener('click', () => {
      if (item.separated_audio_id) {
        loadAudioIntoPlayer(item.separated_audio_id, true);
      }
    });

    tr.querySelector('.btn-rescore').addEventListener('click', async () => {
      switchTab('tab-comparison');
      await loadClipForAudition(item.clip_id);
    });

    tr.querySelector('.btn-delete-eval').addEventListener('click', async () => {
      if (!confirm(`Delete evaluation for "${item.clip_title}" (${item.model_name})?`)) return;
      try {
        const res = await fetch(`/api/evaluations/${item.id}`, { method: "DELETE" });
        await parseJsonResponse(res);
        showToast("Deleted evaluation record", "info");
        await fetchEvaluations();
      } catch (err) {
        showToast(err.message, "error");
      }
    });

    el.evaluationsTableBody.appendChild(tr);
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
  if (!container) return;
  container.innerHTML = "";

  const query = (state.tabLibrarySearch || "").toLowerCase().trim();
  const category = (state.tabLibraryCategory || "all").toLowerCase();

  let filtered = (state.serverFiles || []).filter(file => {
    const fileCat = (file.category || "").toLowerCase();
    const matchesCategory = category === "all" || fileCat.includes(category);
    const matchesQuery = !query ||
      (file.name || "").toLowerCase().includes(query) ||
      (file.path || "").toLowerCase().includes(query) ||
      fileCat.includes(query);
    return matchesCategory && matchesQuery;
  });

  if (filtered.length === 0) {
    container.innerHTML = `<div class="empty-placeholder">No project audio files found matching filter.</div>`;
    return;
  }

  filtered.forEach(file => {
    const badgeInfo = getFileModelBadge(file);
    const card = document.createElement("div");
    card.className = "file-item-card";
    card.innerHTML = `
      <div class="file-details">
        <div style="display: flex; align-items: center; gap: 8px;">
          <span class="badge ${badgeInfo.class}" style="font-size: 0.72rem; font-weight: 700;">${escapeHtml(badgeInfo.label)}</span>
          <span class="file-name">${escapeHtml(file.name)}</span>
        </div>
        <span class="file-path">${escapeHtml(file.path)} • ${formatBytes(file.size || 0)}</span>
      </div>
      <div class="file-actions" style="display: flex; align-items: center; gap: 6px;">
        <button class="btn btn-sm btn-primary btn-load-file" data-path="${escapeHtml(file.path)}" title="Load into Studio Workspace">Load</button>
        <button class="btn btn-sm btn-ghost btn-delete-file" data-path="${escapeHtml(file.path)}" title="Delete file from disk">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path><line x1="10" y1="11" x2="10" y2="17"></line><line x1="14" y1="11" x2="14" y2="17"></line></svg>
        </button>
      </div>
    `;
    card.querySelector('.btn-load-file').addEventListener('click', () => loadServerFile(file.path));
    card.querySelector('.btn-delete-file').addEventListener('click', () => deleteServerFile(file.path, file.name));
    container.appendChild(card);
  });
}

async function deleteServerFile(filePath, fileName) {
  const displayName = fileName || filePath.split('/').pop();
  if (!confirm(`Are you sure you want to permanently delete "${displayName}" from disk?\n(Matching .json metadata sidecar will also be removed)`)) {
    return;
  }
  try {
    showToast(`Deleting ${displayName}...`, "info");
    const res = await fetch("/api/library/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: filePath }),
    });
    const data = await parseJsonResponse(res);

    showToast(`Deleted ${displayName} successfully!`, "success");
    await fetchServerFiles();
    renderLibraryModalItems();
  } catch (err) {
    showToast(`Delete failed: ${err.message}`, "error");
  }
}

async function loadServerFile(filePath) {
  try {
    showToast(`Loading ${filePath}...`, "info");
    const res = await fetch("/api/library/load", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: filePath }),
    });
    const data = await parseJsonResponse(res);

    if (data.audio_id) {
      await fetchAudioList();
      await setActiveAudio(data.audio_id, { play: true });
      showToast(`Loaded ${data.metadata?.title || filePath} into workspace!`, "success");
      closeAllModals();
    }
  } catch (err) {
    showToast(`Failed to load file: ${err.message}`, "error");
  }
}

async function openLibraryModal() {
  if (el.modalLibrary) el.modalLibrary.classList.remove('hidden');
  if (el.libraryModalSearch) {
    el.libraryModalSearch.value = "";
    state.libraryModalSearch = "";
    setTimeout(() => el.libraryModalSearch.focus(), 60);
  }
  state.libraryModalCategory = "all";
  if (el.libraryModalCategories) {
    el.libraryModalCategories.querySelectorAll('.pill-btn').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.category === 'all');
    });
  }

  if (!state.serverFiles || state.serverFiles.length === 0) {
    if (el.modalLibraryItems) {
      el.modalLibraryItems.innerHTML = '<div class="empty-placeholder">Scanning project directories for sample audio files...</div>';
    }
    await fetchServerFiles();
  }
  renderLibraryModalItems();
}

function renderLibraryModalItems() {
  if (!el.modalLibraryItems) return;
  el.modalLibraryItems.innerHTML = "";

  const query = (state.libraryModalSearch || "").toLowerCase().trim();
  const category = (state.libraryModalCategory || "all").toLowerCase();

  let filtered = (state.serverFiles || []).filter(file => {
    const fileCat = (file.category || "").toLowerCase();
    const matchesCategory = category === "all" || fileCat.includes(category);
    const matchesQuery = !query ||
      (file.name || "").toLowerCase().includes(query) ||
      (file.path || "").toLowerCase().includes(query) ||
      fileCat.includes(query);
    return matchesCategory && matchesQuery;
  });

  if (el.libraryModalCount) {
    el.libraryModalCount.textContent = `${filtered.length} of ${(state.serverFiles || []).length} sample files`;
  }

  if (filtered.length === 0) {
    el.modalLibraryItems.innerHTML = `
      <div class="empty-placeholder" style="padding: 2.5rem 1rem; text-align: center;">
        <span style="font-size: 2rem; display: block; margin-bottom: 0.5rem;">📂</span>
        <span>No sample audio files match your search filter.</span>
      </div>
    `;
    return;
  }

  filtered.forEach(file => {
    const badgeInfo = getFileModelBadge(file);
    const item = document.createElement("div");
    item.className = "file-item-card";
    item.innerHTML = `
      <div class="file-details">
        <div style="display: flex; align-items: center; gap: 8px;">
          <span class="badge ${badgeInfo.class}" style="font-size: 0.72rem; font-weight: 700;">${escapeHtml(badgeInfo.label)}</span>
          <span class="file-name">${escapeHtml(file.name)}</span>
        </div>
        <span class="file-path">${escapeHtml(file.path)} • ${formatBytes(file.size || 0)}</span>
      </div>
      <div class="file-actions" style="display: flex; align-items: center; gap: 6px;">
        <button class="btn btn-sm btn-primary btn-modal-load" data-path="${escapeHtml(file.path)}" title="Load audio into studio workspace">
          <span>Load</span>
        </button>
        <button class="btn btn-sm btn-ghost btn-delete-file" data-path="${escapeHtml(file.path)}" title="Delete file from disk">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path><line x1="10" y1="11" x2="10" y2="17"></line><line x1="14" y1="11" x2="14" y2="17"></line></svg>
        </button>
      </div>
    `;
    item.querySelector('.btn-modal-load').addEventListener('click', () => loadServerFile(file.path));
    item.querySelector('.btn-delete-file').addEventListener('click', () => deleteServerFile(file.path, file.name));
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

function initModals() {
  // Close buttons
  if (el.btnCloseLibraryModal) {
    el.btnCloseLibraryModal.addEventListener('click', () => {
      if (el.modalLibrary) el.modalLibrary.classList.add('hidden');
    });
  }
  if (el.btnCancelLibraryModal) {
    el.btnCancelLibraryModal.addEventListener('click', () => {
      if (el.modalLibrary) el.modalLibrary.classList.add('hidden');
    });
  }
  if (el.btnCloseSaveModal) {
    el.btnCloseSaveModal.addEventListener('click', () => {
      if (el.modalSaveTo) el.modalSaveTo.classList.add('hidden');
    });
  }
  if (el.btnCancelSave) {
    el.btnCancelSave.addEventListener('click', () => {
      if (el.modalSaveTo) el.modalSaveTo.classList.add('hidden');
    });
  }
  if (el.btnCloseShortcutsModal) {
    el.btnCloseShortcutsModal.addEventListener('click', () => {
      if (el.modalShortcuts) el.modalShortcuts.classList.add('hidden');
    });
  }

  // Close modals when clicking directly on backdrop outside modal card
  document.querySelectorAll('.modal-backdrop').forEach(backdrop => {
    backdrop.addEventListener('click', (e) => {
      if (e.target === backdrop) {
        backdrop.classList.add('hidden');
      }
    });
  });

  // Modal search input listener
  if (el.libraryModalSearch) {
    el.libraryModalSearch.addEventListener('input', (e) => {
      state.libraryModalSearch = e.target.value;
      renderLibraryModalItems();
    });
  }

  // Modal category pills listener
  if (el.libraryModalCategories) {
    el.libraryModalCategories.addEventListener('click', (e) => {
      const btn = e.target.closest('.pill-btn');
      if (!btn) return;
      el.libraryModalCategories.querySelectorAll('.pill-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.libraryModalCategory = btn.dataset.category || 'all';
      renderLibraryModalItems();
    });
  }
}

function populateAllAudioSelects() {
  const standardSelects = [
    el.sepInputSelect,
    el.diarInputSelect,
  ];

  standardSelects.forEach(select => {
    if (!select) return;
    const currentVal = select.value;
    select.innerHTML = '<option value="">-- Select Audio Track --</option>';

    state.audioList.forEach(item => {
      const opt = document.createElement("option");
      opt.value = item.id;
      let prefix = "";
      if (item.model_info?.model_label) {
        prefix = `[${item.model_info.model_label}] `;
      } else if (item.tags?.includes("htdemucs_ft")) {
        prefix = "[HTDemucs FT] ";
      } else if (item.tags?.includes("htdemucs")) {
        prefix = "[HTDemucs] ";
      } else if (item.tags?.includes("bs_roformer")) {
        prefix = "[BS-RoFormer] ";
      } else if (item.tags?.includes("mel_roformer")) {
        prefix = "[Mel-RoFormer] ";
      } else if (item.source_type === "cut" || item.tags?.includes("cut")) {
        prefix = "[Cut Snippet] ";
      }
      opt.textContent = `${prefix}${item.title} (${(item.duration_s || 0).toFixed(1)}s, ${item.format.toUpperCase()})`;
      select.appendChild(opt);
    });

    if (currentVal && state.audioList.some(a => a.id === currentVal)) {
      select.value = currentVal;
    } else if (state.activeAudio) {
      select.value = state.activeAudio.id;
    }
  });

  // Dedicated Audition Clip Select with categorized optgroups
  if (el.auditionClipSelect) {
    const curVal = el.auditionClipSelect.value;
    el.auditionClipSelect.innerHTML = '<option value="">-- Select Cut Snippet to Evaluate (with BG Music) --</option>';

    const cutsGroup = document.createElement("optgroup");
    cutsGroup.label = "✂️ Audio Cuts & Snippets (with Background Music)";

    const sourcesGroup = document.createElement("optgroup");
    sourcesGroup.label = "📥 Full YouTube & Ingest Audio Sources";

    const stemsGroup = document.createElement("optgroup");
    stemsGroup.label = "✨ Separated Vocal Stems (Auto-resolves to parent mixture)";

    state.audioList.forEach(item => {
      const opt = document.createElement("option");
      opt.value = item.id;
      const isSep = item.source_type === "separation" || item.tags?.includes("separated");
      const isCut = item.source_type === "cut" || item.tags?.includes("cut");

      let prefix = "";
      if (item.model_info?.model_label) prefix = `[${item.model_info.model_label}] `;
      else if (item.tags?.includes("htdemucs_ft")) prefix = "[HTDemucs FT] ";
      else if (item.tags?.includes("htdemucs")) prefix = "[HTDemucs] ";
      else if (item.tags?.includes("bs_roformer")) prefix = "[BS-RoFormer] ";
      else if (item.tags?.includes("mel_roformer")) prefix = "[Mel-RoFormer] ";
      else if (isCut) prefix = "[Audio Cut] ";
      else prefix = "[Raw Source] ";

      opt.textContent = `${prefix}${item.title} (${(item.duration_s || 0).toFixed(1)}s)`;

      if (isCut) {
        cutsGroup.appendChild(opt);
      } else if (isSep) {
        stemsGroup.appendChild(opt);
      } else {
        sourcesGroup.appendChild(opt);
      }
    });

    if (cutsGroup.children.length > 0) el.auditionClipSelect.appendChild(cutsGroup);
    if (sourcesGroup.children.length > 0) el.auditionClipSelect.appendChild(sourcesGroup);
    if (stemsGroup.children.length > 0) el.auditionClipSelect.appendChild(stemsGroup);

    if (curVal && state.audioList.some(a => a.id === curVal)) {
      el.auditionClipSelect.value = curVal;
    } else if (state.activeAudio) {
      const activeObj = state.audioList.find(a => a.id === state.activeAudio.id);
      if (activeObj && activeObj.parent_id && (activeObj.source_type === "separation" || activeObj.tags?.includes("separated"))) {
        el.auditionClipSelect.value = activeObj.parent_id;
      } else {
        el.auditionClipSelect.value = state.activeAudio.id;
      }
    }
  }
}

// ==================== KEYBOARD SHORTCUTS ====================

function initKeyboardShortcuts() {
  document.addEventListener('keydown', (e) => {
    const activeTag = document.activeElement ? document.activeElement.tagName.toLowerCase() : '';
    if (activeTag === 'input' || activeTag === 'textarea' || activeTag === 'select') {
      return;
    }

    const currentTab = document.querySelector('.nav-tab.active')?.dataset.tab;

    // Number keys 1-9 on Audition Hub
    if (currentTab === 'tab-comparison' && e.key >= '1' && e.key <= '9') {
      const trackIdx = parseInt(e.key, 10) - 1;
      if (trackIdx < auditionTracks.length) {
        e.preventDefault();
        switchAuditionTrack(trackIdx);
      }
    }

    // Space on Audition Hub
    if (currentTab === 'tab-comparison' && e.code === 'Space') {
      e.preventDefault();
      toggleAuditionPlay();
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

  // Tab 7 Project Explorer Search & Filters
  if (el.tabLibrarySearch) {
    el.tabLibrarySearch.addEventListener('input', (e) => {
      state.tabLibrarySearch = e.target.value;
      renderServerFiles();
    });
  }
  if (el.tabLibraryCategories) {
    el.tabLibraryCategories.addEventListener('click', (e) => {
      const btn = e.target.closest('.pill-btn');
      if (!btn) return;
      el.tabLibraryCategories.querySelectorAll('.pill-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.tabLibraryCategory = btn.dataset.category || 'all';
      renderServerFiles();
    });
  }
}

function switchTab(tabId) {
  el.tabs.forEach(t => t.classList.toggle('active', t.dataset.tab === tabId));
  el.tabPanes.forEach(pane => pane.classList.toggle('active', pane.id === tabId));

  try {
    localStorage.setItem('sonic_active_tab', tabId);
  } catch (_) {}

  if (tabId === 'tab-workspace') {
    renderWaveform();
    renderCutsTable();
  } else if (tabId === 'tab-comparison') {
    if (state.activeAudio && (!auditionTracks || auditionTracks.length === 0)) {
      loadClipForAudition(state.activeAudio.id);
    }
  } else if (tabId === 'tab-matrix') {
    fetchEvaluations();
  }
}

function toggleShortcutsModal() {
  el.modalShortcuts.classList.toggle('hidden');
}

// ==================== THEME MANAGEMENT ====================

function initTheme() {
  const savedTheme = localStorage.getItem('sonic_theme') || 'dark';
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
  console.log("SonicStudio initializing frontend...");

  try { initTheme(); } catch (e) { console.error("initTheme error:", e); }
  try { initPlayer(); } catch (e) { console.error("initPlayer error:", e); }
  try { initWaveformInteractions(); } catch (e) { console.error("initWaveformInteractions error:", e); }
  try { initAudioCutter(); } catch (e) { console.error("initAudioCutter error:", e); }
  try { initCutsManager(); } catch (e) { console.error("initCutsManager error:", e); }
  try { initIngestAndSaves(); } catch (e) { console.error("initIngestAndSaves error:", e); }
  try { initYouTubeCrawler(); } catch (e) { console.error("initYouTubeCrawler error:", e); }
  try { initSeparationStudio(); } catch (e) { console.error("initSeparationStudio error:", e); }
  try { initDiarizationStudio(); } catch (e) { console.error("initDiarizationStudio error:", e); }
  try { initAuditionHub(); } catch (e) { console.error("initAuditionHub error:", e); }
  try { initEvaluationMatrix(); } catch (e) { console.error("initEvaluationMatrix error:", e); }
  try { initKeyboardShortcuts(); } catch (e) { console.error("initKeyboardShortcuts error:", e); }
  try { initNavigation(); } catch (e) { console.error("initNavigation error:", e); }
  try { initModals(); } catch (e) { console.error("initModals error:", e); }
  try { initLiveReload(); } catch (e) { console.error("initLiveReload error:", e); }

  if (el.btnRefreshLibrary) {
    el.btnRefreshLibrary.addEventListener('click', fetchServerFiles);
  }

  try { await fetchSystemStatus(); } catch (e) { console.error("fetchSystemStatus error:", e); }
  try { await fetchServerFiles(); } catch (e) { console.error("fetchServerFiles error:", e); }
  try { await fetchAudioList(); } catch (e) { console.error("fetchAudioList error:", e); }
  try { await fetchYouTubeVault(); } catch (e) { console.error("fetchYouTubeVault error:", e); }
  try { await fetchEvaluations(); } catch (e) { console.error("fetchEvaluations error:", e); }

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

  try {
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
  } catch (e) {
    console.error("Audio auto-load error:", e);
  }

  console.log("SonicStudio initialized successfully!");
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initApp);
} else {
  initApp();
}
