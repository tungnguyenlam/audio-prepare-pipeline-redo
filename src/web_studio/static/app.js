/**
 * SonicStudio — Audio Preparation & Separation Studio Web Application
 * Modular ES6 Vanilla JavaScript Frontend Architecture
 */

// ==================== STATE MANAGEMENT ====================

const state = {
  activeTab: 'tab-workspace',
  activeAudio: null,       // Currently selected Audio metadata
  activePeaks: [],         // Downsampled waveform peaks
  selection: { start: 0, end: 0, active: false },
  cutUnit: 'seconds',
  zoom: 1.0,
  audioList: [],           // All registered Audio items
  serverFiles: [],         // Files on disk from /api/library
  systemStatus: null,
  selectedGpu: localStorage.getItem('sonic_selected_gpu') || 'cuda:0',

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

  // Diarization Studio State
  diarization: {
    audioId: null,
    data: null,
    speakers: [],
    turns: [],
    customNames: {},
    colors: {},
    zoom: 1.0,
    viewMode: 'multitrack',
    highlightOverlaps: true,
    activeTurnIndex: null,
    selectedTurnIndices: new Set(),
    loopTurn: false,
    followPlayhead: true,
    activeSpeakerFilter: 'all',
    minDurFilter: 0,
    searchQuery: '',
    sortMode: 'time-asc',
    activeExportTab: 'rttm',
    soloSpeaker: null,
    mutedSpeakers: new Set(),
    history: [],
    historySearch: '',
    dragTurnInfo: null,
    isScrubbing: false,
    selectionRange: null,
  },
};

// DOM Elements Cache
const el = {
  // Navigation
  tabs: document.querySelectorAll('.nav-tab'),
  tabPanes: document.querySelectorAll('.tab-pane'),
  deviceLabel: document.getElementById('device-label'),
  sidebarGpuSection: document.getElementById('sidebar-gpu-section'),
  sidebarGpuCardsList: document.getElementById('sidebar-gpu-cards-list'),
  sidebarSelectedGpuPill: document.getElementById('sidebar-selected-gpu-pill'),
  gpuLoadBadge: document.getElementById('gpu-load-badge'),
  gpuLoadLabel: document.getElementById('gpu-load-label'),
  headerGpuMeter: document.getElementById('header-gpu-meter'),
  queueLabel: document.getElementById('queue-label'),
  queueDot: document.getElementById('queue-dot'),
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
  btnPlayerStart: document.getElementById('btn-player-start'),
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
  selectionHelper: document.getElementById('selection-helper'),
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
  cutDurationDisplay: document.getElementById('cut-duration-display'),
  cutValidation: document.getElementById('cut-validation'),
  btnSetStartPlayhead: document.getElementById('btn-set-start-playhead'),
  btnSetEndPlayhead: document.getElementById('btn-set-end-playhead'),
  rangePresets: document.querySelectorAll('.range-preset'),
  btnUseSelection: document.getElementById('btn-use-selection'),
  btnPreviewCut: document.getElementById('btn-preview-cut'),
  btnApplyCut: document.getElementById('btn-apply-cut'),
  btnCutAndAudition: document.getElementById('btn-cut-and-audition'),
  btnCutAndRunModels: document.getElementById('btn-cut-and-run-models'),
  cutsTableBody: document.getElementById('cuts-table-body'),
  cutsCounterBadge: document.getElementById('cuts-counter-badge'),
  cutUnitRadios: document.querySelectorAll('input[name="cut_unit"]'),
  cutsCountBadge: document.getElementById('cuts-count-badge'),
  btnExportAllCuts: document.getElementById('btn-export-all-cuts'),
  btnClearAllCuts: document.getElementById('btn-clear-all-cuts'),

  // YouTube Ingest
  ytUrlInput: document.getElementById('yt-url-input'),
  ytTargetRate: document.getElementById('yt-target-rate'),
  ytMaxDuration: document.getElementById('yt-max-duration'),
  ytCookiesInput: document.getElementById('yt-cookies-input'),
  btnDownloadYt: document.getElementById('btn-download-yt'),
  btnIngestYt: document.getElementById('btn-ingest-yt'),
  ytTaskProgressBox: document.getElementById('yt-task-progress-box'),
  ytTaskTitle: document.getElementById('yt-task-title'),
  ytTaskTimer: document.getElementById('yt-task-timer'),
  ytProgressBar: document.getElementById('yt-progress-bar'),
  ytTaskStatusText: document.getElementById('yt-task-status-text'),
  ytPreviewCard: document.getElementById('yt-preview-card'),
  ytPreviewThumb: document.getElementById('yt-preview-thumb'),
  ytPreviewTitle: document.getElementById('yt-preview-title'),
  ytPreviewUploader: document.getElementById('yt-preview-uploader'),
  ytPreviewDuration: document.getElementById('yt-preview-duration'),
  ytPreviewId: document.getElementById('yt-preview-id'),
  ytPreviewLink: document.getElementById('yt-preview-link'),
  ytEmptyPlaceholder: document.getElementById('yt-empty-placeholder'),
  ytInspectBadge: document.getElementById('yt-inspect-badge'),
  ytVaultList: document.getElementById('yt-vault-list'),
  btnRefreshYtHistory: document.getElementById('btn-refresh-yt-history'),

  // Separation Studio
  sepInputSelect: document.getElementById('sep-input-select'),
  sepChildrenBox: document.getElementById('sep-children-box'),
  sepChildrenTitle: document.getElementById('sep-children-title'),
  sepChildrenCount: document.getElementById('sep-children-count'),
  sepChildrenList: document.getElementById('sep-children-list'),
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
  btnDiarShortcutsModal: document.getElementById('btn-diar-shortcuts-modal'),
  modalDiarShortcuts: document.getElementById('modal-diar-shortcuts'),
  btnCloseDiarShortcuts: document.getElementById('btn-close-diar-shortcuts'),
  diarInputSelect: document.getElementById('diar-input-select'),
  diarAudioMetaChip: document.getElementById('diar-audio-meta-chip'),
  diarChildrenBox: document.getElementById('diar-children-box'),
  diarChildrenTitle: document.getElementById('diar-children-title'),
  diarChildrenCount: document.getElementById('diar-children-count'),
  diarChildrenList: document.getElementById('diar-children-list'),
  diarInputPreviewPill: document.getElementById('diar-input-preview-pill'),
  btnDiarPreviewInput: document.getElementById('btn-diar-preview-input'),
  diarTrackTitleText: document.getElementById('diar-track-title-text'),
  diarTrackSpecChip: document.getElementById('diar-track-spec-chip'),
  diarModelCards: document.querySelectorAll('.model-card[data-diar-model]'),
  hfTokenInput: document.getElementById('hf-token-input'),
  btnToggleHfVis: document.getElementById('btn-toggle-hf-vis'),
  diarDeviceSelect: document.getElementById('diar-device-select'),
  btnRunDiarization: document.getElementById('btn-run-diarization'),
  btnDiarImport: document.getElementById('btn-diar-import'),
  diarFileImportInput: document.getElementById('diar-file-import-input'),
  btnDiarSample: document.getElementById('btn-diar-sample'),
  btnDiarReset: document.getElementById('btn-diar-reset'),
  diarTaskProgressBox: document.getElementById('diar-task-progress-box'),
  diarTaskTimer: document.getElementById('diar-task-timer'),
  diarProgressBar: document.getElementById('diar-progress-bar'),
  diarTaskStatusText: document.getElementById('diar-task-status-text'),
  diarResultsWrapper: document.getElementById('diar-results-wrapper'),
  diarEmptyPlaceholder: document.getElementById('diar-empty-placeholder'),
  btnDiarEmptyDemo: document.getElementById('btn-diar-empty-demo'),
  diarModelBadge: document.getElementById('diar-model-badge'),
  diarSpeakerCountBadge: document.getElementById('diar-speaker-count-badge'),
  diarTurnsCountBadge: document.getElementById('diar-turns-count-badge'),
  diarSpeechRatioBadge: document.getElementById('diar-speech-ratio-badge'),
  btnViewMultitrack: document.getElementById('btn-view-multitrack'),
  btnViewComposite: document.getElementById('btn-view-composite'),
  btnToggleOverlapHighlight: document.getElementById('btn-toggle-overlap-highlight'),
  btnOpenAutoMerge: document.getElementById('btn-open-auto-merge'),
  modalAutoMergeGaps: document.getElementById('modal-auto-merge-gaps'),
  btnCloseAutoMergeModal: document.getElementById('btn-close-auto-merge-modal'),
  btnCancelAutoMerge: document.getElementById('btn-cancel-auto-merge'),
  btnConfirmAutoMerge: document.getElementById('btn-confirm-auto-merge'),
  autoMergeThresholdInput: document.getElementById('auto-merge-threshold-input'),
  diarMinimapContainer: document.getElementById('diar-minimap-container'),
  diarMinimapTrack: document.getElementById('diar-minimap-track'),
  diarMinimapBlocksLayer: document.getElementById('diar-minimap-blocks-layer'),
  diarMinimapPlayhead: document.getElementById('diar-minimap-playhead'),
  diarMinimapViewportWindow: document.getElementById('diar-minimap-viewport-window'),
  btnDiarSkipBack: document.getElementById('btn-diar-skip-back'),
  btnDiarPlayToggle: document.getElementById('btn-diar-play-toggle'),
  iconDiarPlay: document.getElementById('icon-diar-play'),
  iconDiarPause: document.getElementById('icon-diar-pause'),
  btnDiarMuteAll: document.getElementById('btn-diar-mute-all'),
  iconDiarMuteAll: document.getElementById('icon-diar-mute-all'),
  diarMuteAllLabel: document.getElementById('diar-mute-all-label'),
  btnDiarSkipFwd: document.getElementById('btn-diar-skip-fwd'),
  btnDiarLoopTurn: document.getElementById('btn-diar-loop-turn'),
  btnDiarFollowPlayhead: document.getElementById('btn-diar-follow-playhead'),
  diarTimeCurrent: document.getElementById('diar-time-current'),
  diarTimeTotal: document.getElementById('diar-time-total'),
  btnDiarZoomOut: document.getElementById('btn-diar-zoom-out'),
  btnDiarZoomIn: document.getElementById('btn-diar-zoom-in'),
  btnDiarZoomFit: document.getElementById('btn-diar-zoom-fit'),
  diarZoomLevel: document.getElementById('diar-zoom-level'),
  diarSpeedSelect: document.getElementById('diar-speed-select'),
  diarMultitrackViewport: document.getElementById('diar-multitrack-viewport'),
  diarTimelineInner: document.getElementById('diar-timeline-inner'),
  diarLaneLabelsCol: document.getElementById('diar-lane-labels-col'),
  diarWaveformLabel: document.getElementById('diar-waveform-label'),
  diarCompositeLabel: document.getElementById('diar-composite-label'),
  diarSpkLabelsWrap: document.getElementById('diar-spk-labels-wrap'),
  diarTracksArea: document.getElementById('diar-tracks-area'),
  diarRulerTrack: document.getElementById('diar-ruler-track'),
  diarWaveformTrack: document.getElementById('diar-waveform-track'),
  diarWaveformCanvas: document.getElementById('diar-waveform-canvas'),
  diarCompositeTrack: document.getElementById('diar-composite-track'),
  diarSpeakerLanesWrap: document.getElementById('diar-speaker-lanes-wrap'),
  diarSelectionOverlay: document.getElementById('diar-selection-overlay'),
  diarSelectionActionsBar: document.getElementById('diar-selection-actions-bar'),
  btnSelPlay: document.getElementById('btn-sel-play'),
  btnSelAddTurn: document.getElementById('btn-sel-add-turn'),
  btnSelCut: document.getElementById('btn-sel-cut'),
  btnSelClear: document.getElementById('btn-sel-clear'),
  diarPlayheadLine: document.getElementById('diar-playhead-line'),
  diarPlayheadHandle: document.getElementById('diar-playhead-handle'),
  diarTurnTooltip: document.getElementById('diar-turn-tooltip'),
  diarSpeakersGrid: document.getElementById('diar-speakers-grid'),
  diarExtractModeSelect: document.getElementById('diar-extract-mode-select'),
  btnExtractAllSpeakers: document.getElementById('btn-extract-all-speakers'),
  btnAddTurnAtPlayhead: document.getElementById('btn-add-turn-at-playhead'),
  diarBatchBar: document.getElementById('diar-batch-bar'),
  diarBatchCount: document.getElementById('diar-batch-count'),
  diarBatchSpeakerSelect: document.getElementById('diar-batch-speaker-select'),
  btnBatchReassign: document.getElementById('btn-batch-reassign'),
  btnBatchCut: document.getElementById('btn-batch-cut'),
  btnBatchDelete: document.getElementById('btn-batch-delete'),
  btnBatchClear: document.getElementById('btn-batch-clear'),
  diarFilterSpeakerSelect: document.getElementById('diar-filter-speaker-select'),
  diarTurnsSearchInput: document.getElementById('diar-turns-search-input'),
  diarFilterMinDur: document.getElementById('diar-filter-min-dur'),
  diarSortTurnsSelect: document.getElementById('diar-sort-turns-select'),
  diarFilteredTurnsCount: document.getElementById('diar-filtered-turns-count'),
  thSelectAllTurns: document.getElementById('th-select-all-turns'),
  turnsTableBody: document.getElementById('turns-table-body'),
  exportNavTabs: document.querySelectorAll('.export-nav-tab'),
  exportFilenameLabel: document.getElementById('export-filename-label'),
  btnCopyExport: document.getElementById('btn-copy-export'),
  btnDownloadExport: document.getElementById('btn-download-export'),
  exportPreviewTextarea: document.getElementById('export-preview-textarea'),
  diarHistoryCountBadge: document.getElementById('diar-history-count-badge'),
  btnExportAllHistory: document.getElementById('btn-export-all-history'),
  btnClearDiarHistory: document.getElementById('btn-clear-diar-history'),
  diarHistorySearchInput: document.getElementById('diar-history-search-input'),
  diarHistoryList: document.getElementById('diar-history-list'),
  modalMergeSpeaker: document.getElementById('modal-merge-speaker'),
  btnCloseMergeModal: document.getElementById('btn-close-merge-modal'),
  mergeSourceSpkName: document.getElementById('merge-source-spk-name'),
  mergeTargetSpkSelect: document.getElementById('merge-target-spk-select'),
  btnCancelMerge: document.getElementById('btn-cancel-merge'),
  btnConfirmMerge: document.getElementById('btn-confirm-merge'),

  // Audition & Scoring Hub
  auditionClipSelect: document.getElementById('audition-clip-select'),
  activeAuditionTrackName: document.getElementById('active-audition-track-name'),
  auditionTimeCurrent: document.getElementById('audition-time-current'),
  auditionTimeTotal: document.getElementById('audition-time-total'),
  auditionScrubber: document.getElementById('audition-scrubber'),
  btnAuditionSkipBack: document.getElementById('btn-audition-skip-back'),
  btnAuditionStart: document.getElementById('btn-audition-start'),
  btnAuditionSkipFwd: document.getElementById('btn-audition-skip-fwd'),
  auditionSpeedSelect: document.getElementById('audition-speed-select'),
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
  btnClearSession: document.getElementById('btn-clear-session'),

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
  queueBadge: document.getElementById('queue-badge'),
  modalTaskQueue: document.getElementById('modal-task-queue'),
  btnCloseQueueModal: document.getElementById('btn-close-queue-modal'),
  btnCancelQueueModal: document.getElementById('btn-cancel-queue-modal'),
  btnRefreshQueueModal: document.getElementById('btn-refresh-queue-modal'),
  btnClearQueueFinished: document.getElementById('btn-clear-queue-finished'),
  studioQueueTaskList: document.getElementById('studio-queue-task-list'),
  queueStatRunning: document.getElementById('queue-stat-running'),
  queueStatQueued: document.getElementById('queue-stat-queued'),
  queueStatCompleted: document.getElementById('queue-stat-completed'),
  queueStatFailed: document.getElementById('queue-stat-failed'),
  queueModalFilters: document.getElementById('queue-modal-filters'),
  queueModalSubtitle: document.getElementById('queue-modal-subtitle'),
  queueGpuName: document.getElementById('queue-gpu-name'),
  queueGpuLoad: document.getElementById('queue-gpu-load'),
  queueGpuVram: document.getElementById('queue-gpu-vram'),
  queueGpuPower: document.getElementById('queue-gpu-power'),
  queueActiveSplit: document.getElementById('queue-active-split'),
  queueGpuDevicesGrid: document.getElementById('queue-gpu-devices-grid'),
  toastContainer: document.getElementById('toast-container'),
};

// Queue Modal state
state.queueModalFilter = 'all';
state.queuePollingInterval = null;

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
  const hundredths = Math.round(seconds * 100);
  const m = Math.floor(hundredths / 6000);
  const s = Math.floor((hundredths % 6000) / 100);
  const fraction = hundredths % 100;
  return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}.${fraction.toString().padStart(2, '0')}`;
}

function parseTimestampToSeconds(value) {
  const text = String(value).trim();
  if (!text) return NaN;
  if (text.includes(':')) {
    const parts = text.split(':').map(Number);
    if (parts.length < 2 || parts.length > 3 || parts.some(n => !Number.isFinite(n) || n < 0)) return NaN;
    if (parts.slice(1).some(n => n >= 60)) return NaN;
    return parts.length === 2
      ? parts[0] * 60 + parts[1]
      : parts[0] * 3600 + parts[1] * 60 + parts[2];
  }
  if (!/^\d+$/.test(text)) return NaN;
  if (text.length <= 2) return Number(text);
  const seconds = Number(text.slice(-2));
  const minutes = Number(text.slice(-4, -2) || text.slice(0, -2));
  const hours = text.length > 4 ? Number(text.slice(0, -4)) : 0;
  if (seconds >= 60 || minutes >= 60) return NaN;
  return hours * 3600 + minutes * 60 + seconds;
}

function cutValueToSeconds(value, unit, duration) {
  if (unit === 'timestamp') return parseTimestampToSeconds(value);
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return NaN;
  if (unit === 'minutes') return numeric * 60;
  if (unit === 'percent') return duration * numeric / 100;
  return numeric;
}

function secondsToCutValue(seconds, unit, duration) {
  if (unit === 'timestamp') return formatTimePrecise(seconds);
  if (unit === 'minutes') return (seconds / 60).toFixed(3);
  if (unit === 'percent') return duration ? (seconds / duration * 100).toFixed(1) : '0.0';
  return seconds.toFixed(2);
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

function isAuditionPlaybackActive() {
  return document.querySelector('.nav-tab.active')?.dataset.tab === 'tab-comparison'
    && auditionTracks.length > 0;
}

function getPlaybackAudio() {
  return isAuditionPlaybackActive() ? auditionAudio : el.audio;
}

function getPlaybackDuration(audio = getPlaybackAudio()) {
  if (Number.isFinite(audio?.duration) && audio.duration > 0) return audio.duration;
  if (audio === auditionAudio) {
    return state.audioList.find(item => item.id === auditionTracks[0]?.id)?.duration_s || 0;
  }
  return state.player.duration || state.activeAudio?.duration_s || state.diarization?.duration || 0;
}

function syncLoopControls(audio = getPlaybackAudio()) {
  const isLooping = Boolean(audio?.loop);
  el.btnLoop?.classList.toggle('active', isLooping && audio === getPlaybackAudio());
  el.btnAuditionLoop?.classList.toggle('active', isLooping && isAuditionPlaybackActive());
}

function syncSpeedControls(rate = getPlaybackAudio()?.playbackRate || state.player.playbackRate) {
  const numericRate = Number.parseFloat(rate) || 1;
  const setSelectValue = (select) => {
    if (!select) return;
    const matchingOption = Array.from(select.options).find(
      option => Number.parseFloat(option.value) === numericRate,
    );
    select.value = matchingOption?.value || String(numericRate);
  };
  setSelectValue(el.speedSelect);
  setSelectValue(el.auditionSpeedSelect);
  setSelectValue(el.diarSpeedSelect);
}

function syncVolumeControls(volume = getPlaybackAudio()?.volume ?? state.player.volume) {
  const value = Number.isFinite(volume) ? volume : 1;
  if (el.volumeSlider) el.volumeSlider.value = value;
  if (el.auditionVolumeSlider) el.auditionVolumeSlider.value = value;
  updateVolumeIcon(value);
}

function syncActivePlaybackControls() {
  const audio = getPlaybackAudio();
  syncLoopControls(audio);
  syncSpeedControls(audio?.playbackRate || state.player.playbackRate);
  syncVolumeControls(audio?.volume);
  setPlayingUI(audio && !audio.paused);
}

function setPlaybackRate(rate) {
  const parsedRate = Number.parseFloat(rate);
  if (!Number.isFinite(parsedRate) || parsedRate <= 0) return;
  state.player.playbackRate = parsedRate;
  if (el.audio) el.audio.playbackRate = parsedRate;
  if (auditionAudio) auditionAudio.playbackRate = parsedRate;
  syncSpeedControls(parsedRate);
}

function togglePlaybackLoop() {
  const audio = getPlaybackAudio();
  if (!audio) return;
  audio.loop = !audio.loop;
  if (audio === el.audio) state.player.loop = audio.loop;
  syncLoopControls(audio);
  showToast(audio.loop ? "Loop playback enabled" : "Loop playback disabled", "info");
}

function initPlayer() {
  if (el.btnPlayPause) el.btnPlayPause.addEventListener('click', togglePlayPause);
  if (el.btnSkipBack) el.btnSkipBack.addEventListener('click', () => seekRelative(-5));
  if (el.btnPlayerStart) el.btnPlayerStart.addEventListener('click', () => seekTo(0));
  if (el.btnSkipFwd) el.btnSkipFwd.addEventListener('click', () => seekRelative(5));

  if (el.btnLoop) {
    el.btnLoop.addEventListener('click', togglePlaybackLoop);
  }

  if (el.speedSelect) {
    el.speedSelect.addEventListener('change', (e) => setPlaybackRate(e.target.value));
  }

  if (el.volumeSlider) {
    el.volumeSlider.addEventListener('input', (e) => {
      state.player.volume = parseFloat(e.target.value);
      if (el.audio) el.audio.volume = state.player.volume;
      if (auditionAudio) auditionAudio.volume = state.player.volume;
      syncVolumeControls(state.player.volume);
    });
  }

  if (el.btnMute) {
    el.btnMute.addEventListener('click', () => {
      const audio = getPlaybackAudio();
      if (!audio) return;
      if (audio.volume > 0) {
        if (el.audio) el.audio.volume = 0;
        if (auditionAudio) auditionAudio.volume = 0;
        if (el.volumeSlider) el.volumeSlider.value = 0;
      } else {
        const restoredVolume = state.player.volume || 1.0;
        if (el.audio) el.audio.volume = restoredVolume;
        if (auditionAudio) auditionAudio.volume = restoredVolume;
        if (el.volumeSlider) el.volumeSlider.value = restoredVolume;
      }
      syncVolumeControls(getPlaybackAudio().volume);
    });
  }

  // Toggle remaining time vs total duration on click
  if (el.timeTotal) {
    el.timeTotal.addEventListener('click', () => {
      state.player.showRemainingTime = !state.player.showRemainingTime;
      if (isAuditionPlaybackActive()) updateAuditionTimeDisplays();
      else onTimeUpdate();
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
    el.audio.addEventListener('play', () => {
      if (!isAuditionPlaybackActive()) setPlayingUI(true);
    });
    el.audio.addEventListener('pause', () => {
      if (!isAuditionPlaybackActive()) setPlayingUI(false);
    });
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

  // Space: Play / Pause using whichever player is active.
  if (e.code === 'Space') {
    e.preventDefault();
    togglePlayPause();
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

  if (e.key === 'Home') {
    e.preventDefault();
    seekTo(0);
    return;
  }

  // Diarization Studio Context-Aware Hotkeys
  if (state.activeTab === 'tab-diarization') {
    if (e.key === '[') {
      e.preventDefault();
      navigateToAdjacentTurn(-1);
      return;
    }
    if (e.key === ']') {
      e.preventDefault();
      navigateToAdjacentTurn(1);
      return;
    }
    if (e.key === 'l' || e.key === 'L') {
      e.preventDefault();
      if (el.btnDiarLoopTurn) el.btnDiarLoopTurn.click();
      return;
    }
    if (e.key === 'f' || e.key === 'F') {
      e.preventDefault();
      if (el.btnDiarFollowPlayhead) el.btnDiarFollowPlayhead.click();
      return;
    }
    if (e.key === 's' || e.key === 'S') {
      e.preventDefault();
      splitActiveTurnAtPlayhead();
      return;
    }
    if (e.key === 'd' || e.key === 'D' || e.key === 'Delete') {
      e.preventDefault();
      deleteActiveTurn();
      return;
    }
    if (e.key === 'n' || e.key === 'N' || e.key === 'a' || e.key === 'A') {
      e.preventDefault();
      addTurnAtCursor();
      return;
    }
    if (e.key === 'z' || e.key === 'Z') {
      e.preventDefault();
      if (e.shiftKey) {
        setDiarZoom(Math.max(1.0, state.diarization.zoom / 1.5));
      } else {
        setDiarZoom(Math.min(10.0, state.diarization.zoom * 1.5));
      }
      return;
    }
    if (e.key === '0') {
      e.preventDefault();
      setDiarZoom(1.0);
      return;
    }
    if (e.key >= '1' && e.key <= '8') {
      if (state.diarization.activeTurnIndex !== null) {
        e.preventDefault();
        const spkIdx = parseInt(e.key) - 1;
        if (state.diarization.speakers[spkIdx]) {
          const targetSpk = state.diarization.speakers[spkIdx].speaker_id;
          state.diarization.turns[state.diarization.activeTurnIndex].speaker_id = targetSpk;
          detectTurnOverlaps();
          renderDiarizationWorkspace(state.diarization.data, state.diarization.audioId, false);
          showToast(`Turn assigned to ${getSpeakerName(targetSpk)}`, 'success');
        }
        return;
      }
    }
  }

  // J / K / L Seek & Pause (Non-diarization)
  if (e.key === 'j' || e.key === 'J') {
    seekRelative(-2);
    return;
  }
  if (e.key === 'k' || e.key === 'K') {
    const audio = getPlaybackAudio();
    if (audio && !audio.paused) audio.pause();
    return;
  }
  if (e.key === 'l' || e.key === 'L') {
    seekRelative(2);
    return;
  }

  // Workspace Cut Shortcuts: [ and ]: Set Cut Start / End to current playhead
  if (e.key === '[') {
    if (state.activeAudio) {
      const range = readCutRange();
      writeCutRange(el.audio.currentTime || 0, range.error ? state.activeAudio.duration_s : range.effectiveEnd);
      showToast(`Set clip start to ${formatTimePrecise(el.audio.currentTime || 0)}`, "info");
    }
    return;
  }
  if (e.key === ']') {
    if (state.activeAudio) {
      const range = readCutRange();
      writeCutRange(range.error ? 0 : range.start, el.audio.currentTime || 0);
      showToast(`Set clip end to ${formatTimePrecise(el.audio.currentTime || 0)}`, "info");
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

  // Shortcuts Cheatsheet
  if (e.key === '?' || (e.shiftKey && e.key === '/')) {
    if (el.modalDiarShortcuts && state.activeTab === 'tab-diarization') {
      el.modalDiarShortcuts.classList.toggle('hidden');
    } else if (el.btnOpenShortcutsModal) {
      el.btnOpenShortcutsModal.click();
    }
    return;
  }

  // Z / Shift+Z / 0: Zoom (Workspace)
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

  // Q: Toggle Task Queue Inspector Modal
  if ((e.key === 'q' || e.key === 'Q') && !e.metaKey && !e.ctrlKey && !e.altKey) {
    e.preventDefault();
    toggleQueueModal();
    return;
  }

  // Alt+P: Quick switch to SonicPipeline
  if (e.altKey && (e.key === 'p' || e.key === 'P')) {
    e.preventDefault();
    window.location.href = '/pipeline/';
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

function updateVolumeIcon(volume = getPlaybackAudio()?.volume ?? 0) {
  const isMuted = volume === 0;
  el.iconVol.classList.toggle('hidden', isMuted);
  el.iconVolMute.classList.toggle('hidden', !isMuted);
}

function setPlayingUI(isPlaying) {
  state.player.isPlaying = isPlaying;
  el.iconPlay?.classList.toggle('hidden', isPlaying);
  el.iconPause?.classList.toggle('hidden', !isPlaying);
  el.iconAuditionPlay?.classList.toggle('hidden', isPlaying);
  el.iconAuditionPause?.classList.toggle('hidden', !isPlaying);
  el.iconDiarPlay?.classList.toggle('hidden', isPlaying);
  el.iconDiarPause?.classList.toggle('hidden', !isPlaying);
}

function togglePlayPause() {
  if (isAuditionPlaybackActive()) {
    toggleAuditionPlay();
    return;
  }
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
  const audio = getPlaybackAudio();
  const duration = getPlaybackDuration(audio);
  if (!audio || !duration) return;
  const targetTime = Math.max(0, Math.min(Number(time) || 0, duration));
  if (audio.readyState === 0) {
    audio.addEventListener('loadedmetadata', () => seekTo(targetTime), { once: true });
    return;
  }
  audio.currentTime = targetTime;
  if (audio === auditionAudio) {
    updateAuditionTimeDisplays();
  } else {
    updatePlayheadPosition(targetTime);
  }
}

function seekRelative(offset) {
  const audio = getPlaybackAudio();
  if (!audio) return;
  seekTo(audio.currentTime + offset);
}

function onLoadedMetadata() {
  if (isAuditionPlaybackActive()) return;
  state.player.duration = el.audio.duration || (state.activeAudio ? state.activeAudio.duration_s : 0);
  el.timeTotal.textContent = formatTime(state.player.duration);
  el.rulerEnd.textContent = formatTime(state.player.duration);
  el.rulerMid.textContent = formatTime(state.player.duration / 2);
}

function onTimeUpdate() {
  if (isAuditionPlaybackActive()) return;
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
  updateDiarizationPlayhead(cur, dur);
}

function onEnded() {
  if (isAuditionPlaybackActive()) return;
  if (!state.player.loop) {
    setPlayingUI(false);
    seekTo(0);
  }
}

function loadAudioIntoPlayer(audioId, autoplay = false) {
  const currentStreamUrl = `/api/audio/${audioId}/stream`;
  const isDifferent = !el.audio.src || !el.audio.src.endsWith(currentStreamUrl);

  const item = state.audioList.find(a => a.id === audioId)
    || (state.activeAudio?.id === audioId ? state.activeAudio : null);

  if (isDifferent) {
    el.audio.src = currentStreamUrl;
    el.audio.playbackRate = state.player.playbackRate;
    el.audio.volume = state.player.volume;
    el.audio.loop = state.player.loop;
    el.audio.load();
  }
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
    const headerTrackEl = document.getElementById('header-active-track-name');
    if (headerTrackEl) headerTrackEl.textContent = `Active Track: ${meta.title || meta.source_id}`;
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

    // Set default cut bounds in the currently selected display format.
    writeCutRange(0, meta.duration_s || 10);

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
  if (el.selectionHelper) {
    el.selectionHelper.classList.remove('has-selection');
    el.selectionHelper.innerHTML = '<span class="selection-helper-icon">↔</span><span><strong>Select the useful moment.</strong> You can fine-tune its boundaries in the range panel.</span>';
  }
}

function initWaveformInteractions() {
  let isDragging = false;
  let dragStartX = 0;
  let dragMode = 'new';

  const viewport = el.waveformViewport;

  if (viewport) {
    viewport.addEventListener('mousedown', (e) => {
      if (!state.activeAudio) return;
      const rect = viewport.getBoundingClientRect();
      const handle = e.target.closest('.selection-handle');
      dragMode = handle?.dataset.handle || 'new';
      dragStartX = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
      isDragging = true;

      if (dragMode === 'new') {
        const time = (dragStartX / rect.width) * (state.activeAudio.duration_s || 1);
        seekTo(time);
        state.selection.start = time;
        state.selection.end = time;
        state.selection.active = true;
        updateSelectionOverlay(dragStartX, dragStartX, rect.width);
      }
      e.preventDefault();
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

      let minX;
      let maxX;
      if (dragMode === 'start') {
        minX = Math.min(currentX, (state.selection.end / (state.activeAudio.duration_s || 1)) * rect.width - 1);
        maxX = (state.selection.end / (state.activeAudio.duration_s || 1)) * rect.width;
      } else if (dragMode === 'end') {
        minX = (state.selection.start / (state.activeAudio.duration_s || 1)) * rect.width;
        maxX = Math.max(currentX, minX + 1);
      } else {
        minX = Math.min(dragStartX, currentX);
        maxX = Math.max(dragStartX, currentX);
      }

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
          populateCutBoundsFromSelection();
          if (el.selectionHelper) {
            el.selectionHelper.classList.add('has-selection');
            el.selectionHelper.innerHTML = `<span class="selection-helper-icon">✓</span><span><strong>${formatTimePrecise(state.selection.end - state.selection.start)} selected.</strong> Drag the blue edges or edit the values to refine it.</span>`;
          }
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

  window.addEventListener('resize', () => {
    renderWaveform();
    if (state.activeTab === 'tab-diarization') {
      setDiarZoom(state.diarization.zoom || 1.0);
    }
  });
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

function readCutRange(unit = state.cutUnit) {
  const duration = state.activeAudio?.duration_s || 0;
  const start = cutValueToSeconds(el.cutStartInput.value, unit, duration);
  const end = cutValueToSeconds(el.cutEndInput.value, unit, duration);
  let error = '';
  if (!Number.isFinite(start) || !Number.isFinite(end)) error = 'Enter a valid start and end value.';
  else if (start < 0 || end < 0) error = 'Range values cannot be negative.';
  else if (start >= end) error = 'End must be later than start.';
  else if (start >= duration) error = 'Start is outside this track.';
  return { start, end, effectiveEnd: Math.min(end, duration), duration, error };
}

function writeCutRange(start, end, unit = state.cutUnit) {
  const duration = state.activeAudio?.duration_s || 0;
  el.cutStartInput.value = secondsToCutValue(start, unit, duration);
  el.cutEndInput.value = secondsToCutValue(end, unit, duration);
  updateCutRangeUI();
}

function updateCutRangeUI(syncWaveform = false) {
  if (!state.activeAudio) return false;
  const range = readCutRange();
  const invalid = Boolean(range.error);
  el.cutStartInput.classList.toggle('is-invalid', invalid);
  el.cutEndInput.classList.toggle('is-invalid', invalid);
  el.btnPreviewCut.disabled = invalid;
  el.btnApplyCut.disabled = invalid;
  if (el.btnCutAndAudition) el.btnCutAndAudition.disabled = invalid;
  if (el.btnCutAndRunModels) el.btnCutAndRunModels.disabled = invalid;

  if (invalid) {
    el.cutValidation.textContent = range.error;
    el.cutValidation.classList.add('is-error');
    el.cutDurationDisplay.textContent = 'Invalid range';
    return false;
  }

  const clipDuration = Math.max(0, range.effectiveEnd - range.start);
  el.cutDurationDisplay.textContent = `${formatTimePrecise(clipDuration)} clip`;
  el.cutValidation.textContent = range.end > range.duration
    ? `Ends at track boundary (${formatTimePrecise(range.duration)}).`
    : `${formatTimePrecise(range.start)} → ${formatTimePrecise(range.effectiveEnd)}`;
  el.cutValidation.classList.remove('is-error');

  if (syncWaveform && el.waveformViewport) {
    state.selection = { start: range.start, end: range.effectiveEnd, active: true };
    const width = el.waveformViewport.clientWidth;
    updateSelectionOverlay(range.start / range.duration * width, range.effectiveEnd / range.duration * width, width);
    if (el.selectionActionsBar) el.selectionActionsBar.style.display = 'flex';
  }
  return true;
}

function populateCutBoundsFromSelection(showFeedback = false) {
  if (!state.activeAudio || !state.selection.active) return false;

  writeCutRange(state.selection.start, state.selection.end);

  if (showFeedback) {
    showToast("Using the selected waveform range", "success");
  }
  return true;
}

function initAudioCutter() {
  // Use Selection bounds
  el.btnUseSelection.addEventListener('click', () => {
    if (!populateCutBoundsFromSelection(true)) {
      showToast("Select a region on the waveform first", "info");
    }
  });

  // Unit radio change styling
  el.cutUnitRadios.forEach(radio => {
    radio.addEventListener('change', () => {
      const previousRange = readCutRange(state.cutUnit);
      state.cutUnit = radio.value;
      document.querySelectorAll('.radio-pill').forEach(p => p.classList.remove('active'));
      radio.closest('.radio-pill').classList.add('active');
      if (!previousRange.error) writeCutRange(previousRange.start, previousRange.effectiveEnd, state.cutUnit);
      else if (state.selection.active) populateCutBoundsFromSelection();
      else updateCutRangeUI();
    });
  });

  [el.cutStartInput, el.cutEndInput].forEach(input => {
    input.addEventListener('input', () => updateCutRangeUI(true));
  });

  el.btnSetStartPlayhead?.addEventListener('click', () => {
    const range = readCutRange();
    writeCutRange(state.player.currentTime || 0, range.error ? (state.activeAudio.duration_s || 0) : range.effectiveEnd);
  });

  el.btnSetEndPlayhead?.addEventListener('click', () => {
    const range = readCutRange();
    writeCutRange(range.error ? 0 : range.start, state.player.currentTime || 0);
  });

  el.rangePresets.forEach(button => {
    button.addEventListener('click', () => {
      const duration = state.activeAudio?.duration_s || 0;
      const preset = button.dataset.duration;
      const start = preset === 'all' ? 0 : Math.min(state.player.currentTime || 0, duration);
      const end = preset === 'all' ? duration : Math.min(duration, start + Number(preset));
      writeCutRange(start, end);
      updateCutRangeUI(true);
    });
  });

  // Preview Cut
  el.btnPreviewCut.addEventListener('click', () => {
    if (!state.activeAudio) return;
    const range = readCutRange();
    if (range.error) return updateCutRangeUI();

    seekTo(range.start);
    state.player.previewEnd = range.effectiveEnd;
    el.audio.play();
    showToast(`Previewing ${formatTimePrecise(range.effectiveEnd - range.start)} clip`, "info");
  });

  // Apply Cut (AudioCutter API)
  el.btnApplyCut.addEventListener('click', async () => {
    if (!state.activeAudio) return;
    if (!updateCutRangeUI()) return;
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
      addCutToRegistry(data.audio_id, start, end, unit);
      await setActiveAudio(data.audio_id, { play: true });
    } catch (err) {
      showToast(err.message, "error");
    } finally {
      el.btnApplyCut.disabled = false;
      el.btnApplyCut.innerHTML = `<span>Create Clip</span>`;
      updateCutRangeUI();
    }
  });

  // Cut & Send to Audition Hub
  if (el.btnCutAndAudition) {
    el.btnCutAndAudition.addEventListener('click', async () => {
      if (!state.activeAudio) {
        showToast("Please load an audio file first", "warning");
        return;
      }
      if (!updateCutRangeUI()) return;
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
        addCutToRegistry(data.audio_id, start, end, unit);
        switchTab('tab-comparison');
        await loadClipForAudition(data.audio_id);
      } catch (err) {
        showToast(err.message, "error");
      } finally {
        el.btnCutAndAudition.disabled = false;
        el.btnCutAndAudition.innerHTML = `<span>Create &amp; open Audition</span>`;
        updateCutRangeUI();
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
      if (!updateCutRangeUI()) return;
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
        addCutToRegistry(data.audio_id, start, end, unit);
        await fetchAudioList();

        showToast(`Running batch separation models on snippet...`, "info");
        await runBatchMultiModelSeparation(data.audio_id, true);
      } catch (err) {
        showToast(err.message, "error");
      } finally {
        el.btnCutAndRunModels.disabled = false;
        el.btnCutAndRunModels.innerHTML = `<span>Create &amp; run models</span>`;
        updateCutRangeUI();
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
  if (el.btnSendToSep) {
    el.btnSendToSep.addEventListener('click', () => {
      if (!state.activeAudio) return;
      switchTab('tab-separation');
      if (el.sepInputSelect) el.sepInputSelect.value = state.activeAudio.id;
    });
  }

  // Send to Diarization tab
  const btnSendToDiar = document.getElementById('btn-send-to-diarization');
  if (btnSendToDiar) {
    btnSendToDiar.addEventListener('click', () => {
      if (!state.activeAudio) return;
      switchTab('tab-diarization');
      if (el.diarInputSelect) {
        el.diarInputSelect.value = state.activeAudio.id;
        renderDiarizationChildren(state.activeAudio.id);
      }
    });
  }

  // Send to Audition tab
  const btnSendToAudition = document.getElementById('btn-send-to-audition');
  if (btnSendToAudition) {
    btnSendToAudition.addEventListener('click', () => {
      if (!state.activeAudio) return;
      switchTab('tab-comparison');
      if (el.auditionClipSelect) {
        el.auditionClipSelect.value = state.activeAudio.id;
        loadClipForAudition(state.activeAudio.id);
      }
    });
  }

  // Sample Library Modal
  el.btnBrowseLibrary.addEventListener('click', openLibraryModal);
  const btnBrowseTop = document.getElementById('btn-browse-library-top');
  if (btnBrowseTop) btnBrowseTop.addEventListener('click', openLibraryModal);
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
        <span class="file-name">${escapeHtml(item.name)}</span>
        <span class="file-path">${item.sample_rate.toLocaleString()}Hz • ${item.channels === 1 ? 'Mono' : 'Stereo'} • ${(item.duration_s || 0).toFixed(1)}s • ${formatBytes(item.size)}</span>
      </div>
      <div class="stem-actions" style="display: flex; align-items: center; gap: 6px;">
        <button class="btn btn-sm btn-secondary btn-load-yt-workspace">🎛️ Workspace</button>
        <button class="btn btn-sm btn-secondary btn-yt-sep">🧪 Separate</button>
        <button class="btn btn-sm btn-secondary btn-yt-diar">👥 Diarize</button>
        <button class="btn btn-sm btn-ghost btn-delete-yt" title="Delete downloaded audio file">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path><line x1="10" y1="11" x2="10" y2="17"></line><line x1="14" y1="11" x2="14" y2="17"></line></svg>
        </button>
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

    card.querySelector('.btn-delete-yt').addEventListener('click', async () => {
      if (confirm(`Delete YouTube downloaded audio "${item.name}" from disk?`)) {
        try {
          const res = await fetch("/api/crawler/delete", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ path: item.path }),
          });
          if (!res.ok) throw new Error("Failed to delete YouTube audio file");
          showToast(`Deleted ${item.name}`, "info");
          await fetchYouTubeVault();
          await fetchServerFiles();
        } catch (err) {
          showToast(err.message, "error");
        }
      }
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

function findChildAudios(parentAudioId) {
  if (!parentAudioId) return [];
  const parentItem = state.audioList.find(a => a.id === parentAudioId);
  if (!parentItem) return [];

  return state.audioList.filter(item => {
    if (item.id === parentAudioId) return false;
    if (item.parent_id === parentAudioId) return true;
    if (item.model_info && item.model_info.parent_title && parentItem.title && item.model_info.parent_title === parentItem.title) return true;
    if (item.source_id && parentItem.source_id && item.source_id.startsWith(parentItem.source_id + "_") && item.source_id !== parentItem.source_id) return true;
    return false;
  });
}

function renderSeparationChildren(selectedAudioId) {
  if (!el.sepChildrenBox || !el.sepChildrenList) return;
  if (!selectedAudioId) {
    el.sepChildrenBox.style.display = 'none';
    return;
  }

  const selectedItem = state.audioList.find(a => a.id === selectedAudioId);
  if (!selectedItem) {
    el.sepChildrenBox.style.display = 'none';
    return;
  }

  const children = findChildAudios(selectedAudioId);
  const sepStems = children.filter(c => c.source_type === 'separation' || c.tags?.includes('separated'));
  const cutClips = children.filter(c => c.source_type === 'cut' || c.tags?.includes('cut'));

  if (children.length === 0) {
    el.sepChildrenBox.style.display = 'none';
    return;
  }

  el.sepChildrenBox.style.display = 'block';
  if (el.sepChildrenCount) el.sepChildrenCount.textContent = `${children.length} derivative${children.length === 1 ? '' : 's'}`;

  let warningHtml = '';
  if (sepStems.length > 0) {
    el.sepChildrenBox.classList.add('has-warning');
    const modelNames = sepStems.map(s => s.model_info?.model_label || s.tags?.find(t => t.includes('demucs') || t.includes('roformer') || t.includes('mdx23')) || 'Separated Stem').filter(Boolean);
    const uniqueModels = [...new Set(modelNames)].join(', ');
    warningHtml = `
      <div class="child-warning-banner">
        <span>⚠️ <strong>Already Separated:</strong> ${escapeHtml(uniqueModels)} output${sepStems.length === 1 ? '' : 's'} exist for this track.</span>
      </div>
    `;
  } else {
    el.sepChildrenBox.classList.remove('has-warning');
  }

  let listHtml = warningHtml;
  listHtml += children.map(c => {
    const isStem = c.source_type === 'separation' || c.tags?.includes('separated');
    const isCut = c.source_type === 'cut' || c.tags?.includes('cut');
    let badgeClass = 'badge-primary';
    let badgeText = 'Child';
    if (isStem) {
      badgeClass = 'badge-accent';
      badgeText = c.model_info?.model_label || 'Stem';
    } else if (isCut) {
      badgeClass = 'badge-warning';
      badgeText = 'Cut';
    }

    return `
      <div class="child-chip-item">
        <div class="child-chip-left" title="${escapeHtml(c.title)}">
          <span class="child-chip-badge badge ${badgeClass}">${escapeHtml(badgeText)}</span>
          <span class="child-chip-title">${escapeHtml(c.title)}</span>
          <span class="text-muted font-mono" style="font-size: 0.65rem;">(${(c.duration_s || 0).toFixed(1)}s)</span>
        </div>
        <div class="child-chip-actions">
          <button class="child-chip-btn btn-play-child" data-id="${c.id}" title="Play audio snippet">▶ Play</button>
          <button class="child-chip-btn child-chip-btn-primary btn-select-child-sep" data-id="${c.id}" title="Select this child file as input">Use This</button>
        </div>
      </div>
    `;
  }).join('');

  el.sepChildrenList.innerHTML = listHtml;

  // Event handlers
  el.sepChildrenList.querySelectorAll('.btn-play-child').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      loadAudioIntoPlayer(btn.dataset.id, true);
    });
  });

  el.sepChildrenList.querySelectorAll('.btn-select-child-sep').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      el.sepInputSelect.value = btn.dataset.id;
      renderSeparationChildren(btn.dataset.id);
    });
  });

  // Also populate right-hand results panel with existing stems if results panel is currently empty or has placeholder
  if (sepStems.length > 0 && el.sepResultsList && el.sepResultsList.querySelector('.empty-placeholder')) {
    el.sepResultsList.innerHTML = '';
    sepStems.forEach(stem => {
      const modelLabel = stem.model_info?.model_label || stem.tags?.find(t => t.includes('demucs') || t.includes('roformer') || t.includes('mdx23')) || 'Separated Stem';
      renderSeparationResultCard({
        separated_audio_id: stem.id,
        metadata: stem,
        model_label: modelLabel,
        elapsed_s: stem.model_info?.elapsed_s || 0,
      });
    });
  }
}

function renderDiarizationChildren(selectedAudioId) {
  if (!el.diarChildrenBox || !el.diarChildrenList) return;
  if (!selectedAudioId) {
    el.diarChildrenBox.style.display = 'none';
    return;
  }

  const selectedItem = state.audioList.find(a => a.id === selectedAudioId);
  if (!selectedItem) {
    el.diarChildrenBox.style.display = 'none';
    return;
  }

  const children = findChildAudios(selectedAudioId);
  const vocalStems = children.filter(c => c.source_type === 'separation' || c.tags?.includes('separated') || c.tags?.includes('vocals'));
  const isVocalStem = selectedItem.source_type === 'separation' || selectedItem.tags?.includes('separated') || selectedItem.tags?.includes('vocals');

  if (children.length === 0 && !isVocalStem) {
    el.diarChildrenBox.style.display = 'none';
    return;
  }

  el.diarChildrenBox.style.display = 'block';
  if (el.diarChildrenCount) el.diarChildrenCount.textContent = `${children.length} derivative${children.length === 1 ? '' : 's'}`;

  let bannerHtml = '';
  if (vocalStems.length > 0) {
    el.diarChildrenBox.classList.add('has-warning');
    bannerHtml = `
      <div class="child-warning-banner">
        <span>💡 <strong>Clean Vocal Stems Available:</strong> Diarizing isolated vocals yields higher accuracy than noisy mixture audio.</span>
      </div>
    `;
  } else if (isVocalStem) {
    el.diarChildrenBox.classList.remove('has-warning');
    bannerHtml = `
      <div class="child-warning-banner" style="color: var(--accent-cyan); background: hsla(188, 86%, 53%, 0.1); border-left-color: var(--accent-cyan);">
        <span>✨ <strong>Separated Vocal Stem Selected:</strong> Ready for high-precision speaker diarization.</span>
      </div>
    `;
  } else {
    el.diarChildrenBox.classList.remove('has-warning');
  }

  let listHtml = bannerHtml;
  if (children.length > 0) {
    listHtml += children.map(c => {
      const isStem = c.source_type === 'separation' || c.tags?.includes('separated');
      const isCut = c.source_type === 'cut' || c.tags?.includes('cut');
      let badgeClass = 'badge-primary';
      let badgeText = 'Child';
      if (isStem) {
        badgeClass = 'badge-accent';
        badgeText = c.model_info?.model_label || 'Vocal Stem';
      } else if (isCut) {
        badgeClass = 'badge-warning';
        badgeText = 'Cut';
      }

      return `
        <div class="child-chip-item">
          <div class="child-chip-left" title="${escapeHtml(c.title)}">
            <span class="child-chip-badge badge ${badgeClass}">${escapeHtml(badgeText)}</span>
            <span class="child-chip-title">${escapeHtml(c.title)}</span>
            <span class="text-muted font-mono" style="font-size: 0.65rem;">(${(c.duration_s || 0).toFixed(1)}s)</span>
          </div>
          <div class="child-chip-actions">
            <button class="child-chip-btn btn-play-child" data-id="${c.id}" title="Play snippet">▶ Play</button>
            <button class="child-chip-btn child-chip-btn-primary btn-select-child-diar" data-id="${c.id}" title="Use this vocal stem for diarization">⚡ Diarize This</button>
          </div>
        </div>
      `;
    }).join('');
  }

  el.diarChildrenList.innerHTML = listHtml;

  // Event handlers
  el.diarChildrenList.querySelectorAll('.btn-play-child').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      loadAudioIntoPlayer(btn.dataset.id, true);
    });
  });

  el.diarChildrenList.querySelectorAll('.btn-select-child-diar').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      el.diarInputSelect.value = btn.dataset.id;
      renderDiarizationChildren(btn.dataset.id);
      updateDiarInputMeta(btn.dataset.id);
      loadDiarWaveform(btn.dataset.id);
    });
  });
}

function initSeparationStudio() {
  // Input track change listener
  if (el.sepInputSelect) {
    el.sepInputSelect.addEventListener('change', () => {
      renderSeparationChildren(el.sepInputSelect.value);
    });
  }

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
    const device = state.selectedGpu || (el.sepDeviceSelect ? el.sepDeviceSelect.value : 'auto');
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

const DIAR_PALETTE = [
  "hsl(188, 86%, 53%)", // Cyan
  "hsl(158, 64%, 52%)", // Emerald
  "hsl(38, 92%, 50%)",  // Amber
  "hsl(348, 83%, 60%)", // Rose
  "hsl(270, 75%, 65%)", // Purple
  "hsl(205, 90%, 55%)", // Sky Blue
  "hsl(84, 80%, 50%)",  // Lime
  "hsl(22, 90%, 55%)",  // Orange
];

let diarAuditionQueue = [];
let diarAuditionIndex = 0;
let diarAuditionActive = false;

function getSpeakerColor(speakerId) {
  if (state.diarization.colors[speakerId]) {
    return state.diarization.colors[speakerId];
  }
  const spkList = state.diarization.speakers || [];
  const idx = spkList.findIndex(s => s.speaker_id === speakerId);
  const color = DIAR_PALETTE[(idx >= 0 ? idx : 0) % DIAR_PALETTE.length];
  state.diarization.colors[speakerId] = color;
  return color;
}

function getSpeakerName(speakerId) {
  return state.diarization.customNames[speakerId] || speakerId;
}

function diarizationModelLabel(modelTypeOrBackend) {
  const key = String(modelTypeOrBackend || "").toLowerCase();
  if (key.includes("sortformer")) return "NeMo Sortformer";
  if (key.includes("clustering") || key.includes("cluster")) return "NeMo Clustering";
  if (key.includes("3.1") || key.includes("pyannote_31") || key.includes("pyannote_3")) return "Pyannote 3.1";
  if (key.includes("community") || key.includes("pyannote_community")) return "Pyannote Community-1";
  if (key.includes("pyannote")) return "Pyannote Community-1";
  return modelTypeOrBackend || "Pyannote";
}

function syncDiarModelOptions(modelType) {
  const hfGroup = document.getElementById("hf-token-group");
  if (hfGroup) {
    const isPyannote = modelType && modelType.startsWith("pyannote");
    hfGroup.style.display = isPyannote ? "" : "none";
  }
}

function initDiarizationStudio() {
  // Input track change listener
  if (el.diarInputSelect) {
    el.diarInputSelect.addEventListener('change', () => {
      const audioId = el.diarInputSelect.value;
      if (!audioId) {
        updateDiarInputMeta(null);
        renderDiarizationChildren(null);
        return;
      }
      renderDiarizationChildren(audioId);
      updateDiarInputMeta(audioId);
      loadDiarWaveform(audioId);
      loadAudioIntoPlayer(audioId, false);
      state.diarization.audioId = audioId;

      // Check if there is an existing diarization session saved for this audio
      const existingSession = (state.diarization.history || []).find(h => h.audio_id === audioId);
      if (existingSession) {
        loadHistorySession(existingSession.id);
      } else {
        // Reset the diarization result view to empty state for the new track
        state.diarization.data = null;
        state.diarization.turns = [];
        state.diarization.speakers = [];
        state.diarization.activeTurnIndex = null;
        state.diarization.selectedTurnIndices.clear();
        updateBatchBar();
        if (el.diarResultsWrapper) el.diarResultsWrapper.classList.add('hidden');
        if (el.diarEmptyPlaceholder) el.diarEmptyPlaceholder.classList.remove('hidden');
      }
    });
  }

  // Preview target input audio
  if (el.btnDiarPreviewInput) {
    el.btnDiarPreviewInput.addEventListener('click', () => {
      const audioId = el.diarInputSelect.value || state.activeAudio?.id;
      if (audioId) {
        loadAudioIntoPlayer(audioId, true);
      }
    });
  }

  // Model selection cards
  el.diarModelCards.forEach(card => {
    card.addEventListener('click', () => {
      el.diarModelCards.forEach(c => c.classList.remove('active'));
      card.classList.add('active');
      syncDiarModelOptions(card.dataset.diarModel);
    });
  });
  const initiallyActive = document.querySelector('.model-card[data-diar-model].active');
  syncDiarModelOptions(initiallyActive ? initiallyActive.dataset.diarModel : "pyannote_community");

  // HuggingFace Token visibility & persistence
  if (el.hfTokenInput) {
    const savedToken = localStorage.getItem('sonic_hf_token');
    if (savedToken) el.hfTokenInput.value = savedToken;
    el.hfTokenInput.addEventListener('change', () => {
      if (el.hfTokenInput.value.trim()) {
        localStorage.setItem('sonic_hf_token', el.hfTokenInput.value.trim());
      }
    });
  }

  if (el.btnToggleHfVis && el.hfTokenInput) {
    el.btnToggleHfVis.addEventListener('click', () => {
      const isPwd = el.hfTokenInput.type === 'password';
      el.hfTokenInput.type = isPwd ? 'text' : 'password';
      el.btnToggleHfVis.textContent = isPwd ? 'Hide' : 'Show';
    });
  }

  // Shortcuts Modal
  if (el.btnDiarShortcutsModal && el.modalDiarShortcuts) {
    el.btnDiarShortcutsModal.addEventListener('click', () => {
      el.modalDiarShortcuts.classList.remove('hidden');
    });
  }
  if (el.btnCloseDiarShortcuts && el.modalDiarShortcuts) {
    el.btnCloseDiarShortcuts.addEventListener('click', () => {
      el.modalDiarShortcuts.classList.add('hidden');
    });
  }

  // Auto-Merge Micro-Gaps Modal
  if (el.btnOpenAutoMerge && el.modalAutoMergeGaps) {
    el.btnOpenAutoMerge.addEventListener('click', () => {
      el.modalAutoMergeGaps.classList.remove('hidden');
    });
  }
  if (el.btnCloseAutoMergeModal && el.modalAutoMergeGaps) {
    el.btnCloseAutoMergeModal.addEventListener('click', () => {
      el.modalAutoMergeGaps.classList.add('hidden');
    });
  }
  if (el.btnCancelAutoMerge && el.modalAutoMergeGaps) {
    el.btnCancelAutoMerge.addEventListener('click', () => {
      el.modalAutoMergeGaps.classList.add('hidden');
    });
  }
  if (el.btnConfirmAutoMerge) {
    el.btnConfirmAutoMerge.addEventListener('click', () => {
      const threshold = parseFloat(el.autoMergeThresholdInput?.value) || 0.30;
      runAutoMergeMicroGaps(threshold);
      if (el.modalAutoMergeGaps) el.modalAutoMergeGaps.classList.add('hidden');
    });
  }

  // Run Diarization Button
  if (el.btnRunDiarization) {
    el.btnRunDiarization.addEventListener('click', async () => {
      const audioId = el.diarInputSelect.value;
      if (!audioId) {
        showToast("Please select a target audio track for diarization", "error");
        return;
      }

      const activeCard = document.querySelector('.model-card[data-diar-model].active');
      const modelType = activeCard ? activeCard.dataset.diarModel : "pyannote_community";
      const modelId = activeCard?.dataset.modelId || (modelType === "pyannote_31" ? "pyannote/speaker-diarization-3.1" : (modelType.startsWith("pyannote") ? "pyannote/speaker-diarization-community-1" : undefined));
      const device = state.selectedGpu || (el.diarDeviceSelect ? el.diarDeviceSelect.value : 'auto');
      const token = el.hfTokenInput.value.trim() || undefined;

      el.btnRunDiarization.disabled = true;
      el.diarTaskProgressBox.classList.remove('hidden');
      if (el.diarTaskStatusText) {
        el.diarTaskStatusText.textContent = `Running ${diarizationModelLabel(modelType)} diarization...`;
      }

      let startTime = Date.now();
      const timerInterval = setInterval(() => {
        if (el.diarTaskTimer) {
          el.diarTaskTimer.textContent = `${((Date.now() - startTime) / 1000).toFixed(1)}s`;
        }
      }, 100);

      try {
        const res = await fetch("/api/diarization/run", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            audio_id: audioId,
            model_type: modelType,
            model_id: modelId,
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
          renderDiarizationWorkspace(result.diarization, audioId, true);
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

  // Demo Data & Sample Generation
  const triggerDemo = () => {
    const audioId = el.diarInputSelect.value || state.activeAudio?.id || (state.audioList[0] ? state.audioList[0].id : null);
    if (!audioId) {
      showToast("Load an audio track into the workspace first to view diarization demo", "info");
      return;
    }
    generateDemoDiarization(audioId);
    showToast("Demo multi-speaker diarization loaded!", "success");
  };

  if (el.btnDiarSample) el.btnDiarSample.addEventListener('click', triggerDemo);
  if (el.btnDiarEmptyDemo) el.btnDiarEmptyDemo.addEventListener('click', triggerDemo);

  // Reset / Clear Diarization
  if (el.btnDiarReset) {
    el.btnDiarReset.addEventListener('click', () => {
      state.diarization.data = null;
      state.diarization.turns = [];
      state.diarization.speakers = [];
      state.diarization.activeTurnIndex = null;
      state.diarization.selectedTurnIndices.clear();
      updateBatchBar();
      if (el.diarResultsWrapper) el.diarResultsWrapper.classList.add('hidden');
      if (el.diarEmptyPlaceholder) el.diarEmptyPlaceholder.classList.remove('hidden');
      showToast("Diarization workspace reset", "info");
    });
  }

  // File Import Button
  if (el.btnDiarImport && el.diarFileImportInput) {
    el.btnDiarImport.addEventListener('click', () => el.diarFileImportInput.click());
    el.diarFileImportInput.addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (file) importAnnotationFile(file);
      e.target.value = '';
    });
  }

  // Transport Toolbar Controls
  if (el.btnDiarPlayToggle) {
    el.btnDiarPlayToggle.addEventListener('click', () => {
      const audioId = state.diarization.audioId || el.diarInputSelect.value;
      if (!el.audio.src && audioId) loadAudioIntoPlayer(audioId);
      togglePlayPause();
    });
  }

  if (el.btnDiarMuteAll) {
    el.btnDiarMuteAll.addEventListener('click', () => {
      toggleMuteAllSpeakers();
    });
  }

  if (el.btnDiarSkipBack) el.btnDiarSkipBack.addEventListener('click', () => seekRelative(-5));
  if (el.btnDiarSkipFwd) el.btnDiarSkipFwd.addEventListener('click', () => seekRelative(5));

  if (el.btnDiarLoopTurn) {
    el.btnDiarLoopTurn.addEventListener('click', () => {
      state.diarization.loopTurn = !state.diarization.loopTurn;
      el.btnDiarLoopTurn.classList.toggle('active', state.diarization.loopTurn);
      showToast(`Turn loop ${state.diarization.loopTurn ? 'enabled' : 'disabled'}`, 'info');
    });
  }

  if (el.btnDiarFollowPlayhead) {
    el.btnDiarFollowPlayhead.addEventListener('click', () => {
      state.diarization.followPlayhead = !state.diarization.followPlayhead;
      el.btnDiarFollowPlayhead.classList.toggle('active', state.diarization.followPlayhead);
      showToast(`Follow Playhead ${state.diarization.followPlayhead ? 'enabled' : 'disabled'}`, 'info');
    });
  }

  if (el.diarSpeedSelect) {
    el.diarSpeedSelect.addEventListener('change', (e) => {
      setPlaybackRate(parseFloat(e.target.value) || 1.0);
    });
  }

  // View Mode Toggles
  if (el.btnViewMultitrack && el.btnViewComposite) {
    el.btnViewMultitrack.addEventListener('click', () => {
      state.diarization.viewMode = 'multitrack';
      el.btnViewMultitrack.classList.add('active');
      el.btnViewComposite.classList.remove('active');
      if (el.diarSpeakerLanesWrap) el.diarSpeakerLanesWrap.style.display = 'flex';
      if (el.diarSpkLabelsWrap) el.diarSpkLabelsWrap.style.display = 'flex';
      if (el.diarCompositeTrack) el.diarCompositeTrack.style.display = 'block';
      if (el.diarCompositeLabel) el.diarCompositeLabel.style.display = 'flex';
    });

    el.btnViewComposite.addEventListener('click', () => {
      state.diarization.viewMode = 'composite';
      el.btnViewComposite.classList.add('active');
      el.btnViewMultitrack.classList.remove('active');
      if (el.diarSpeakerLanesWrap) el.diarSpeakerLanesWrap.style.display = 'none';
      if (el.diarSpkLabelsWrap) el.diarSpkLabelsWrap.style.display = 'none';
      if (el.diarCompositeTrack) el.diarCompositeTrack.style.display = 'block';
      if (el.diarCompositeLabel) el.diarCompositeLabel.style.display = 'flex';
    });
  }

  if (el.btnToggleOverlapHighlight) {
    el.btnToggleOverlapHighlight.addEventListener('click', () => {
      state.diarization.highlightOverlaps = !state.diarization.highlightOverlaps;
      el.btnToggleOverlapHighlight.classList.toggle('active', state.diarization.highlightOverlaps);
      document.querySelectorAll('.diar-turn-segment.has-overlap').forEach(seg => {
        seg.classList.toggle('overlap-active', state.diarization.highlightOverlaps);
      });
    });
  }

  // Zoom Controls
  if (el.btnDiarZoomIn) {
    el.btnDiarZoomIn.addEventListener('click', () => setDiarZoom(Math.min(10.0, state.diarization.zoom * 1.5)));
  }
  if (el.btnDiarZoomOut) {
    el.btnDiarZoomOut.addEventListener('click', () => setDiarZoom(Math.max(1.0, state.diarization.zoom / 1.5)));
  }
  if (el.btnDiarZoomFit) {
    el.btnDiarZoomFit.addEventListener('click', () => setDiarZoom(1.0));
  }

  // Add Turn at Cursor
  if (el.btnAddTurnAtPlayhead) {
    el.btnAddTurnAtPlayhead.addEventListener('click', addTurnAtCursor);
  }

  // Batch Toolbar Controls
  if (el.thSelectAllTurns) {
    el.thSelectAllTurns.addEventListener('change', (e) => {
      const checked = e.target.checked;
      const visibleTurns = getFilteredAndSortedTurns();
      if (checked) {
        visibleTurns.forEach(t => state.diarization.selectedTurnIndices.add(t.originalIndex));
      } else {
        visibleTurns.forEach(t => state.diarization.selectedTurnIndices.delete(t.originalIndex));
      }
      renderTurnsTable();
      updateBatchBar();
    });
  }

  if (el.btnBatchReassign) {
    el.btnBatchReassign.addEventListener('click', () => {
      const targetSpk = el.diarBatchSpeakerSelect?.value;
      if (!targetSpk || state.diarization.selectedTurnIndices.size === 0) return;
      const count = state.diarization.selectedTurnIndices.size;
      state.diarization.selectedTurnIndices.forEach(idx => {
        if (state.diarization.turns[idx]) {
          state.diarization.turns[idx].speaker_id = targetSpk;
        }
      });
      state.diarization.selectedTurnIndices.clear();
      detectTurnOverlaps();
      renderDiarizationWorkspace(state.diarization.data, state.diarization.audioId, false);
      showToast(`Reassigned ${count} turn(s) to ${getSpeakerName(targetSpk)}`, 'success');
    });
  }

  if (el.btnBatchCut) {
    el.btnBatchCut.addEventListener('click', () => {
      const audioId = state.diarization.audioId || el.diarInputSelect.value;
      let count = 0;
      state.diarization.selectedTurnIndices.forEach(idx => {
        const turn = state.diarization.turns[idx];
        if (turn) {
          addCutToRegistry(audioId, turn.start_s, turn.end_s, 'seconds');
          count++;
        }
      });
      state.diarization.selectedTurnIndices.clear();
      updateBatchBar();
      renderTurnsTable();
      showToast(`Added ${count} turns to Workspace Clips library!`, 'success');
    });
  }

  if (el.btnBatchDelete) {
    el.btnBatchDelete.addEventListener('click', () => {
      const count = state.diarization.selectedTurnIndices.size;
      if (count === 0) return;
      state.diarization.turns = state.diarization.turns.filter((_, idx) => !state.diarization.selectedTurnIndices.has(idx));
      state.diarization.selectedTurnIndices.clear();
      detectTurnOverlaps();
      renderDiarizationWorkspace(state.diarization.data, state.diarization.audioId, false);
      showToast(`Deleted ${count} turn(s)`, 'info');
    });
  }

  if (el.btnBatchClear) {
    el.btnBatchClear.addEventListener('click', () => {
      state.diarization.selectedTurnIndices.clear();
      if (el.thSelectAllTurns) el.thSelectAllTurns.checked = false;
      renderTurnsTable();
      updateBatchBar();
    });
  }

  // Region Selection Actions
  if (el.btnSelPlay) {
    el.btnSelPlay.addEventListener('click', () => {
      if (!state.diarization.selectionRange) return;
      const { start_s, end_s } = state.diarization.selectionRange;
      const audioId = state.diarization.audioId || el.diarInputSelect.value;
      loadAudioIntoPlayer(audioId);
      seekTo(start_s);
      state.player.previewEnd = end_s;
      el.audio.play();
    });
  }

  if (el.btnSelAddTurn) {
    el.btnSelAddTurn.addEventListener('click', () => {
      if (!state.diarization.selectionRange) return;
      const { start_s, end_s } = state.diarization.selectionRange;
      const spkId = state.diarization.speakers[0]?.speaker_id || "spk_00";
      state.diarization.turns.push({
        speaker_id: spkId,
        start_s: roundNum(start_s, 2),
        end_s: roundNum(end_s, 2),
        confidence: 1.0,
      });
      state.diarization.turns.sort((a, b) => a.start_s - b.start_s);
      clearTimelineSelection();
      detectTurnOverlaps();
      renderDiarizationWorkspace(state.diarization.data, state.diarization.audioId, true);
      showToast(`Created new speaker turn (${start_s.toFixed(2)}s – ${end_s.toFixed(2)}s)`, 'success');
    });
  }

  if (el.btnSelCut) {
    el.btnSelCut.addEventListener('click', () => {
      if (!state.diarization.selectionRange) return;
      const { start_s, end_s } = state.diarization.selectionRange;
      const audioId = state.diarization.audioId || el.diarInputSelect.value;
      addCutToRegistry(audioId, start_s, end_s, 'seconds');
      clearTimelineSelection();
      showToast(`Added region (${start_s.toFixed(2)}s – ${end_s.toFixed(2)}s) to Clips registry!`, 'success');
    });
  }

  if (el.btnSelClear) {
    el.btnSelClear.addEventListener('click', clearTimelineSelection);
  }

  // Filters & Search Toolbar
  if (el.diarFilterSpeakerSelect) {
    el.diarFilterSpeakerSelect.addEventListener('change', (e) => {
      state.diarization.activeSpeakerFilter = e.target.value;
      renderTurnsTable();
    });
  }

  if (el.diarTurnsSearchInput) {
    el.diarTurnsSearchInput.addEventListener('input', (e) => {
      state.diarization.searchQuery = e.target.value.toLowerCase().trim();
      renderTurnsTable();
    });
  }

  if (el.diarFilterMinDur) {
    el.diarFilterMinDur.addEventListener('change', (e) => {
      state.diarization.minDurFilter = parseFloat(e.target.value) || 0;
      renderTurnsTable();
    });
  }

  if (el.diarSortTurnsSelect) {
    el.diarSortTurnsSelect.addEventListener('change', (e) => {
      state.diarization.sortMode = e.target.value;
      renderTurnsTable();
    });
  }

  // Extract All Speakers Button
  if (el.btnExtractAllSpeakers) {
    el.btnExtractAllSpeakers.addEventListener('click', extractAllSpeakers);
  }

  // Export Navigation Tabs
  if (el.exportNavTabs) {
    el.exportNavTabs.forEach(tab => {
      tab.addEventListener('click', () => {
        el.exportNavTabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        state.diarization.activeExportTab = tab.dataset.exportTab;
        updateExportPreview();
      });
    });
  }

  // Copy & Download Export
  if (el.btnCopyExport) {
    el.btnCopyExport.addEventListener('click', () => {
      if (el.exportPreviewTextarea) {
        navigator.clipboard.writeText(el.exportPreviewTextarea.value).then(() => {
          showToast("Diarization annotation copied to clipboard!", "success");
        }).catch(() => {
          el.exportPreviewTextarea.select();
          document.execCommand('copy');
          showToast("Copied to clipboard!", "success");
        });
      }
    });
  }

  if (el.btnDownloadExport) {
    el.btnDownloadExport.addEventListener('click', downloadActiveExport);
  }

  // Diarization History Actions
  if (el.diarHistorySearchInput) {
    el.diarHistorySearchInput.addEventListener('input', (e) => {
      state.diarization.historySearch = e.target.value.toLowerCase().trim();
      renderDiarizationHistory();
    });
  }

  if (el.btnClearDiarHistory) {
    el.btnClearDiarHistory.addEventListener('click', clearDiarizationHistory);
  }

  if (el.btnExportAllHistory) {
    el.btnExportAllHistory.addEventListener('click', exportAllDiarizationHistory);
  }

  // Merge Speaker Modal Controls
  if (el.btnCloseMergeModal) el.btnCloseMergeModal.addEventListener('click', closeMergeModal);
  if (el.btnCancelMerge) el.btnCancelMerge.addEventListener('click', closeMergeModal);
  if (el.btnConfirmMerge) el.btnConfirmMerge.addEventListener('click', confirmMergeSpeakers);

  // Setup Global Drag, Scrub & Timeline Listeners
  setupTimelineDragListeners();
  setupMinimapListeners();

  // Diarization Quick Scroll Navigator Listeners
  const btnScrollTop = document.getElementById('btn-scroll-to-top');
  const btnScrollTimeline = document.getElementById('btn-scroll-to-timeline');
  const btnScrollTurns = document.getElementById('btn-scroll-to-turns');
  const btnScrollHistory = document.getElementById('btn-scroll-to-history');

  if (btnScrollTop) {
    btnScrollTop.addEventListener('click', () => {
      const topTarget = document.querySelector('#tab-diarization .diar-setup-card') || document.getElementById('tab-diarization');
      if (topTarget) topTarget.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  }
  if (btnScrollTimeline) {
    btnScrollTimeline.addEventListener('click', () => {
      const timelineTarget = document.querySelector('#tab-diarization .diar-timeline-card') || el.diarResultsWrapper;
      if (timelineTarget) timelineTarget.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  }
  if (btnScrollTurns) {
    btnScrollTurns.addEventListener('click', () => {
      const turnsTarget = document.querySelector('#tab-diarization .diar-turns-card');
      if (turnsTarget) turnsTarget.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  }
  if (btnScrollHistory) {
    btnScrollHistory.addEventListener('click', () => {
      const histTarget = document.querySelector('#tab-diarization .diar-history-card');
      if (histTarget) histTarget.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  }

  // Load Persisted Diarization History
  loadDiarizationHistory();
}

async function loadDiarWaveform(audioId) {
  if (!audioId) return;
  try {
    const res = await fetch(`/api/audio/${audioId}/waveform`);
    if (res.ok) {
      const data = await res.json();
      state.diarWaveformPeaks = data.peaks || [];
      renderDiarWaveform();
    }
  } catch (err) {
    console.warn("Could not fetch waveform peaks for diarization track:", err);
  }
}

function updateDiarInputMeta(audioId) {
  if (!audioId) {
    if (el.diarAudioMetaChip) el.diarAudioMetaChip.textContent = "No track selected";
    if (el.diarInputPreviewPill) el.diarInputPreviewPill.classList.add('hidden');
    return;
  }
  const item = state.audioList.find(a => a.id === audioId);
  if (item) {
    if (el.diarAudioMetaChip) {
      el.diarAudioMetaChip.textContent = `${(item.duration_s || 0).toFixed(1)}s • ${(item.sample_rate || 44100).toLocaleString()}Hz • ${(item.format || 'wav').toUpperCase()}`;
    }
    if (el.diarInputPreviewPill) {
      el.diarInputPreviewPill.classList.remove('hidden');
      if (el.diarTrackTitleText) el.diarTrackTitleText.textContent = item.title || item.source_id;
      if (el.diarTrackSpecChip) {
        el.diarTrackSpecChip.textContent = `${formatTime(item.duration_s || 0)} • ${item.channels === 1 ? 'Mono' : 'Stereo'}`;
      }
    }
  }
}

function setDiarZoom(zoom) {
  state.diarization.zoom = zoom;
  if (el.diarZoomLevel) el.diarZoomLevel.textContent = `${zoom.toFixed(1)}x`;

  const viewportWidth = el.diarMultitrackViewport ? el.diarMultitrackViewport.clientWidth : 1000;
  const labelColWidth = el.diarLaneLabelsCol ? el.diarLaneLabelsCol.offsetWidth : 200;
  const visibleTrackWidth = Math.max(300, viewportWidth - labelColWidth);
  const targetWidth = Math.round(visibleTrackWidth * zoom);
  if (el.diarTracksArea) {
    el.diarTracksArea.style.width = `${targetWidth}px`;
    el.diarTracksArea.style.minWidth = `${targetWidth}px`;
  }

  renderDiarWaveform();
  renderDiarRuler();
  redrawAllSpeakerSpectrograms();
  updateMinimapViewport();
}

function renderDiarizationWorkspace(diarization, audioId, addToHistory = true) {
  state.diarization.audioId = audioId;
  state.diarization.data = diarization;
  state.diarization.turns = (diarization.turns || []).map(t => ({
    ...t,
    start_s: roundNum(Number(t.start_s) || 0, 2),
    end_s: roundNum(Number(t.end_s) || 0, 2),
  }));
  state.diarization.speakers = diarization.speakers || [];

  if (el.diarEmptyPlaceholder) el.diarEmptyPlaceholder.classList.add('hidden');
  if (el.diarResultsWrapper) el.diarResultsWrapper.classList.remove('hidden');

  const audioItem = state.audioList.find(a => a.id === audioId) || state.activeAudio;
  const maxTurnEnd = state.diarization.turns.length > 0 ? Math.max(...state.diarization.turns.map(t => t.end_s)) : 10;
  const totalAudioDuration = (audioItem ? audioItem.duration_s : 0) || maxTurnEnd || 10;
  state.diarization.duration = totalAudioDuration;

  // Extract unique speakers if not present
  if (state.diarization.speakers.length === 0) {
    const uniqueSpkIds = Array.from(new Set(state.diarization.turns.map(t => t.speaker_id)));
    state.diarization.speakers = uniqueSpkIds.map(id => ({ speaker_id: id }));
  }

  // Assign default colors & custom names
  state.diarization.speakers.forEach((spk, idx) => {
    if (!state.diarization.colors[spk.speaker_id]) {
      state.diarization.colors[spk.speaker_id] = DIAR_PALETTE[idx % DIAR_PALETTE.length];
    }
    if (!state.diarization.customNames[spk.speaker_id]) {
      state.diarization.customNames[spk.speaker_id] = spk.speaker_id;
    }
  });

  // Check and flag overlaps
  detectTurnOverlaps();

  // Badges & Speech Ratio
  const totalSpeechS = state.diarization.turns.reduce((acc, t) => acc + Math.max(0, t.end_s - t.start_s), 0);
  const speechRatioPct = ((totalSpeechS / totalAudioDuration) * 100).toFixed(1);

  if (el.diarModelBadge) {
    const backend = diarization.model?.backend || diarization.model?.model_id || "Pyannote";
    el.diarModelBadge.textContent = diarizationModelLabel(backend);
  }
  if (el.diarSpeakerCountBadge) {
    el.diarSpeakerCountBadge.textContent = `${state.diarization.speakers.length} Speaker${state.diarization.speakers.length === 1 ? '' : 's'}`;
  }
  if (el.diarTurnsCountBadge) {
    el.diarTurnsCountBadge.textContent = `${state.diarization.turns.length} Turn${state.diarization.turns.length === 1 ? '' : 's'}`;
  }
  if (el.diarSpeechRatioBadge) {
    el.diarSpeechRatioBadge.textContent = `${speechRatioPct}% Speech (${totalSpeechS.toFixed(1)}s)`;
  }

  // Update Speaker Filter Dropdown & Batch Speaker Select
  if (el.diarFilterSpeakerSelect) {
    el.diarFilterSpeakerSelect.innerHTML = `<option value="all">All Speakers (${state.diarization.speakers.length})</option>` +
      state.diarization.speakers.map(s => `<option value="${s.speaker_id}">${escapeHtml(getSpeakerName(s.speaker_id))}</option>`).join('');
  }
  if (el.diarBatchSpeakerSelect) {
    el.diarBatchSpeakerSelect.innerHTML = state.diarization.speakers.map(s => `
      <option value="${s.speaker_id}">${escapeHtml(getSpeakerName(s.speaker_id))}</option>
    `).join('');
  }

  // 1. Fetch Spectrogram & Waveform
  loadDiarSpectrogram(audioId, () => redrawAllSpeakerSpectrograms());
  loadDiarWaveform(audioId);

  // 2. Render Minimap
  renderDiarMinimap();

  // 3. Render Time Ruler
  renderDiarRuler();

  // 4. Render Composite Lane
  renderCompositeLane();

  // 5. Render Multi-Track Swimlanes & Left Labels
  renderSpeakerSwimlanes();

  // 6. Render Speaker Profile Cards & Stats
  renderSpeakerProfiles();

  // 7. Render Turns Inspector Table
  renderTurnsTable();
  updateBatchBar();

  // 8. Update Export Preview
  updateExportPreview();

  // 9. Save to History (if new run)
  if (addToHistory) {
    saveDiarizationToHistory(diarization, audioId);
  } else {
    renderDiarizationHistory();
  }
}

function detectTurnOverlaps() {
  const turns = state.diarization.turns;
  turns.forEach(t => t.has_overlap = false);

  for (let i = 0; i < turns.length; i++) {
    for (let j = i + 1; j < turns.length; j++) {
      const a = turns[i];
      const b = turns[j];
      const overlapStart = Math.max(a.start_s, b.start_s);
      const overlapEnd = Math.min(a.end_s, b.end_s);
      if (overlapEnd > overlapStart + 0.02) {
        a.has_overlap = true;
        b.has_overlap = true;
      }
    }
  }
}

function renderDiarMinimap() {
  if (!el.diarMinimapBlocksLayer) return;
  el.diarMinimapBlocksLayer.innerHTML = "";
  const dur = state.diarization.duration || 1;

  state.diarization.turns.forEach(turn => {
    const leftPct = (turn.start_s / dur) * 100;
    const widthPct = Math.max(0.3, ((turn.end_s - turn.start_s) / dur) * 100);
    const color = getSpeakerColor(turn.speaker_id);

    const block = document.createElement("div");
    block.className = "diar-minimap-block";
    block.style.left = `${leftPct}%`;
    block.style.width = `${widthPct}%`;
    block.style.backgroundColor = color;
    el.diarMinimapBlocksLayer.appendChild(block);
  });

  updateMinimapViewport();
}

function updateMinimapViewport() {
  const windowEl = el.diarMinimapViewportWindow;
  const viewport = el.diarMultitrackViewport;
  if (!windowEl || !viewport) return;

  const totalWidth = viewport.scrollWidth || 1;
  const clientWidth = viewport.clientWidth || 1;
  const scrollLeft = viewport.scrollLeft || 0;

  const leftPct = (scrollLeft / totalWidth) * 100;
  const widthPct = Math.min(100, Math.max(3, (clientWidth / totalWidth) * 100));

  windowEl.style.left = `${leftPct}%`;
  windowEl.style.width = `${widthPct}%`;
}

function setupMinimapListeners() {
  const track = el.diarMinimapTrack;
  const viewport = el.diarMultitrackViewport;
  if (!track || !viewport) return;

  let isDraggingMinimap = false;

  const handleMinimapSeek = (e) => {
    const rect = track.getBoundingClientRect();
    const pos = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const dur = state.diarization.duration || 1;
    const seekSec = pos * dur;
    seekTo(seekSec);

    // Center viewport at seek position
    viewport.scrollLeft = pos * viewport.scrollWidth - viewport.clientWidth / 2;
    updateMinimapViewport();
  };

  track.addEventListener('mousedown', (e) => {
    isDraggingMinimap = true;
    handleMinimapSeek(e);
  });

  window.addEventListener('mousemove', (e) => {
    if (isDraggingMinimap) {
      handleMinimapSeek(e);
    }
  });

  window.addEventListener('mouseup', () => {
    isDraggingMinimap = false;
  });

  if (viewport) {
    viewport.addEventListener('scroll', () => {
      updateMinimapViewport();
    });
  }
}

function renderDiarRuler() {
  if (!el.diarRulerTrack) return;
  const dur = state.diarization.duration || 1;
  el.diarRulerTrack.innerHTML = "";

  const tracksArea = el.diarTracksArea;
  const trackWidth = tracksArea ? (tracksArea.clientWidth || 800) : 800;
  const pixelsPerSec = trackWidth / dur;

  const candidateSteps = [0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600];
  let stepSec = candidateSteps.find(s => s * pixelsPerSec >= 65) || 60;

  for (let t = 0; t <= dur; t += stepSec) {
    const pct = (t / dur) * 100;
    const tick = document.createElement("div");
    tick.className = "diar-ruler-tick";
    tick.style.left = `${pct}%`;
    tick.textContent = stepSec < 1 ? t.toFixed(2) + 's' : formatTime(t);
    el.diarRulerTrack.appendChild(tick);

    // Add sub-ticks
    if (stepSec >= 1 && stepSec <= 10) {
      const subStep = stepSec / 2;
      const subT = t + subStep;
      if (subT < dur) {
        const subPct = (subT / dur) * 100;
        const subTick = document.createElement("div");
        subTick.className = "diar-ruler-subtick";
        subTick.style.left = `${subPct}%`;
        el.diarRulerTrack.appendChild(subTick);
      }
    }
  }
}

function renderDiarWaveform() {
  const canvas = el.diarWaveformCanvas;
  if (!canvas || !el.diarWaveformTrack) return;

  const rect = el.diarWaveformTrack.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = el.diarWaveformTrack.clientWidth || rect.width || 800;
  const h = el.diarWaveformTrack.clientHeight || rect.height || 44;

  if (w <= 0 || h <= 0) return;

  canvas.width = Math.floor(w * dpr);
  canvas.height = Math.floor(h * dpr);

  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);

  const peaks = state.diarWaveformPeaks || state.activePeaks || [];
  if (peaks && peaks.length > 0) {
    const numBars = peaks.length;
    const barWidth = Math.max(1, w / numBars);
    const centerY = h / 2;

    // Gradient peak fill
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, "hsl(188, 86%, 58%)");
    grad.addColorStop(0.5, "hsl(217, 91%, 65%)");
    grad.addColorStop(1, "hsl(188, 86%, 58%)");

    ctx.fillStyle = grad;
    for (let i = 0; i < numBars; i++) {
      const val = peaks[i];
      const barH = Math.max(2, val * (h - 8));
      const y = centerY - barH / 2;
      ctx.fillRect(i * barWidth, y, Math.max(1, barWidth - 0.5), barH);
    }
  } else {
    // Subtle procedural waveform while loading
    const numBars = Math.floor(w / 4);
    ctx.fillStyle = "rgba(100, 149, 237, 0.4)";
    for (let i = 0; i < numBars; i++) {
      const s = Math.sin(i * 0.12) * 0.4 + Math.sin(i * 0.04) * 0.3 + 0.3;
      const barH = Math.max(2, s * (h - 10));
      const y = (h - barH) / 2;
      ctx.fillRect(i * 4, y, 2, barH);
    }
  }
}

function renderCompositeLane() {
  if (!el.diarCompositeTrack) return;
  el.diarCompositeTrack.innerHTML = "";
  const dur = state.diarization.duration || 1;

  state.diarization.turns.forEach((turn, idx) => {
    const color = getSpeakerColor(turn.speaker_id);
    const leftPct = (turn.start_s / dur) * 100;
    const widthPct = Math.max(0.4, ((turn.end_s - turn.start_s) / dur) * 100);

    const isSoloActive = Boolean(state.diarization.soloSpeaker);
    const isSolo = state.diarization.soloSpeaker === turn.speaker_id;
    const isMuted = state.diarization.mutedSpeakers.has(turn.speaker_id);
    const isDimmed = (isSoloActive && !isSolo) || isMuted;

    const seg = document.createElement("div");
    seg.className = `diar-turn-segment ${turn.has_overlap ? 'has-overlap' : ''} ${state.diarization.highlightOverlaps && turn.has_overlap ? 'overlap-active' : ''} ${state.diarization.activeTurnIndex === idx ? 'active-turn' : ''} ${isDimmed ? 'turn-dimmed' : ''}`;
    seg.style.left = `${leftPct}%`;
    seg.style.width = `${widthPct}%`;
    seg.style.backgroundColor = color;
    seg.dataset.index = idx;
    seg.innerHTML = `
      <div class="turn-handle turn-handle-left" data-index="${idx}"></div>
      <span class="turn-label-text">${escapeHtml(getSpeakerName(turn.speaker_id))} (${(turn.end_s - turn.start_s).toFixed(1)}s)</span>
      <div class="turn-handle turn-handle-right" data-index="${idx}"></div>
    `;

    attachTurnSegmentEvents(seg, turn, idx);
    el.diarCompositeTrack.appendChild(seg);
  });
}

const diarSpectrogramCache = new Map(); // audioId -> HTMLImageElement

function loadDiarSpectrogram(audioId, onLoaded) {
  if (!audioId) return;
  if (diarSpectrogramCache.has(audioId)) {
    const cached = diarSpectrogramCache.get(audioId);
    if (cached.complete && cached.naturalWidth > 0) {
      if (onLoaded) onLoaded(cached);
      redrawAllSpeakerSpectrograms();
      return;
    }
  }

  const img = new Image();
  img.crossOrigin = "anonymous";
  img.src = `/api/audio/${audioId}/spectrogram?raw=1&t=${Date.now()}`;
  img.onload = () => {
    diarSpectrogramCache.set(audioId, img);
    if (onLoaded) onLoaded(img);
    redrawAllSpeakerSpectrograms();
  };
  img.onerror = () => {
    console.warn("Could not load raw spectrogram image for:", audioId);
  };
}

function redrawAllSpeakerSpectrograms() {
  const audioId = state.diarization.audioId;
  const img = diarSpectrogramCache.get(audioId);
  const dur = state.diarization.duration || 1;

  const canvases = document.querySelectorAll('.diar-speaker-spectrogram-canvas');
  canvases.forEach(canvas => {
    const spkId = canvas.dataset.speaker;
    const spkTurns = state.diarization.turns.filter(t => t.speaker_id === spkId);
    drawSpeakerSpectrogramCanvas(canvas, img, spkTurns, dur);
  });
}

function drawSpeakerSpectrogramCanvas(canvas, img, spkTurns, dur) {
  const track = canvas.parentElement;
  if (!track) return;

  const rect = track.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = track.clientWidth || rect.width || 800;
  const h = track.clientHeight || rect.height || 64;

  if (w <= 0 || h <= 0) return;

  canvas.width = Math.floor(w * dpr);
  canvas.height = Math.floor(h * dpr);

  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);

  // 1. Draw raw full mel spectrogram
  if (img && img.complete && img.naturalWidth > 0) {
    ctx.drawImage(img, 0, 0, w, h);
  } else {
    // Fallback procedural spectrogram grid
    ctx.fillStyle = "#0c101c";
    ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = "rgba(100, 149, 237, 0.18)";
    for (let i = 0; i < w; i += 6) {
      const s = Math.sin(i * 0.08) * 0.5 + 0.5;
      ctx.fillRect(i, h - s * (h - 10), 3, s * (h - 10));
    }
  }

  // 2. Dim regions where this speaker is NOT speaking with semi-transparent dark shade
  ctx.fillStyle = "rgba(10, 14, 23, 0.78)";
  let lastEnd = 0;
  const sortedTurns = [...spkTurns].sort((a, b) => a.start_s - b.start_s);

  sortedTurns.forEach(turn => {
    if (turn.start_s > lastEnd) {
      const nonSpkLeft = (lastEnd / dur) * w;
      const nonSpkWidth = Math.max(0, ((turn.start_s - lastEnd) / dur) * w);
      ctx.fillRect(nonSpkLeft, 0, nonSpkWidth, h);
    }
    lastEnd = Math.max(lastEnd, turn.end_s);
  });

  if (lastEnd < dur) {
    const nonSpkLeft = (lastEnd / dur) * w;
    const nonSpkWidth = Math.max(0, ((dur - lastEnd) / dur) * w);
    ctx.fillRect(nonSpkLeft, 0, nonSpkWidth, h);
  }
}

function renderSpeakerSwimlanes() {
  if (!el.diarSpeakerLanesWrap || !el.diarSpkLabelsWrap) return;
  el.diarSpeakerLanesWrap.innerHTML = "";
  el.diarSpkLabelsWrap.innerHTML = "";
  const dur = state.diarization.duration || 1;

  state.diarization.speakers.forEach(spk => {
    const spkId = spk.speaker_id;
    const color = getSpeakerColor(spkId);
    const spkName = getSpeakerName(spkId);
    const spkTurns = state.diarization.turns.filter(t => t.speaker_id === spkId);
    const spkTotalSpeech = spkTurns.reduce((acc, t) => acc + Math.max(0, t.end_s - t.start_s), 0);

    const isSoloActive = Boolean(state.diarization.soloSpeaker);
    const isSolo = state.diarization.soloSpeaker === spkId;
    const isMuted = state.diarization.mutedSpeakers.has(spkId);
    const isDimmed = (isSoloActive && !isSolo) || isMuted;

    // 1. Left Sticky Label Row
    const labelRow = document.createElement("div");
    labelRow.className = `diar-spk-label-row ${isDimmed ? 'lane-dimmed' : ''}`;
    labelRow.dataset.speakerId = spkId;
    labelRow.innerHTML = `
      <div class="spk-label-left" title="Click to rename ${escapeHtml(spkName)}">
        <span class="spk-color-indicator" style="background-color: ${color}; width: 12px; height: 12px; border-radius: 50%; box-shadow: 0 0 6px ${color}; flex-shrink: 0;"></span>
        <div class="spk-name-wrap">
          <span class="lane-spk-name" style="color: ${color};" title="${escapeHtml(spkName)}">${escapeHtml(spkName)}</span>
          <span class="spk-stats-sub">${spkTurns.length} turns • ${spkTotalSpeech.toFixed(1)}s</span>
        </div>
      </div>
      <div class="spk-label-controls">
        <button class="spk-ctrl-btn btn-solo ${isSolo ? 'active' : ''}" data-speaker="${spkId}" title="Solo ${escapeHtml(spkName)} [S]">S</button>
        <button class="spk-ctrl-btn btn-mute ${isMuted ? 'active' : ''}" data-speaker="${spkId}" title="Mute ${escapeHtml(spkName)} [M]">M</button>
        <button class="spk-ctrl-btn btn-audition" data-speaker="${spkId}" title="Audition ${escapeHtml(spkName)} Segments">▶</button>
      </div>
    `;

    labelRow.querySelector('.spk-label-left').addEventListener('click', () => {
      const newName = prompt(`Rename speaker '${spkName}':`, spkName);
      if (newName && newName.trim() && newName.trim() !== spkName) {
        state.diarization.customNames[spkId] = newName.trim();
        renderSpeakerSwimlanes();
        renderCompositeLane();
        renderSpeakerProfiles();
        renderTurnsTable();
        updateExportPreview();
      }
    });

    labelRow.querySelector('.btn-solo').addEventListener('click', (e) => {
      e.stopPropagation();
      toggleSpeakerSolo(spkId);
    });
    labelRow.querySelector('.btn-mute').addEventListener('click', (e) => {
      e.stopPropagation();
      toggleSpeakerMute(spkId);
    });
    labelRow.querySelector('.btn-audition').addEventListener('click', (e) => {
      e.stopPropagation();
      auditionSpeakerTurns(spkId);
    });

    el.diarSpkLabelsWrap.appendChild(labelRow);

    // 2. Right Track Lane with Spectrogram Canvas + Speech Segment Color Indicators
    const track = document.createElement("div");
    track.className = `diar-speaker-lane-track ${isDimmed ? 'lane-dimmed' : ''}`;
    track.dataset.speaker = spkId;

    // Mel-Spectrogram Canvas Background
    const canvas = document.createElement("canvas");
    canvas.className = "diar-speaker-spectrogram-canvas";
    canvas.dataset.speaker = spkId;
    track.appendChild(canvas);

    // Speech Segments denoted by prominent color lines and glowing boundaries
    spkTurns.forEach(turn => {
      const idx = state.diarization.turns.indexOf(turn);
      const leftPct = (turn.start_s / dur) * 100;
      const widthPct = Math.max(0.4, ((turn.end_s - turn.start_s) / dur) * 100);

      const seg = document.createElement("div");
      seg.className = `diar-turn-segment ${turn.has_overlap ? 'has-overlap' : ''} ${state.diarization.highlightOverlaps && turn.has_overlap ? 'overlap-active' : ''} ${state.diarization.activeTurnIndex === idx ? 'active-turn' : ''} ${isDimmed ? 'turn-dimmed' : ''}`;
      seg.style.left = `${leftPct}%`;
      seg.style.width = `${widthPct}%`;
      seg.style.borderTop = `3px solid ${color}`;
      seg.style.borderBottom = `3px solid ${color}`;
      seg.style.borderLeft = `2px solid ${color}`;
      seg.style.borderRight = `2px solid ${color}`;
      seg.style.backgroundColor = `${color}28`;
      seg.style.boxShadow = `0 0 12px ${color}45`;
      seg.dataset.index = idx;
      seg.innerHTML = `
        <div class="turn-handle turn-handle-left" style="background-color: ${color};" data-index="${idx}"></div>
        <span class="turn-label-text">${(turn.end_s - turn.start_s).toFixed(1)}s</span>
        <div class="turn-handle turn-handle-right" style="background-color: ${color};" data-index="${idx}"></div>
      `;

      attachTurnSegmentEvents(seg, turn, idx);
      track.appendChild(seg);
    });

    el.diarSpeakerLanesWrap.appendChild(track);
  });

  // Render spectrogram image on all speaker canvases
  redrawAllSpeakerSpectrograms();
}

function attachTurnSegmentEvents(segEl, turn, idx) {
  segEl.addEventListener('mouseenter', (e) => {
    if (!state.diarization.dragTurnInfo) {
      showTurnTooltip(e, turn);
    }
  });

  segEl.addEventListener('mouseleave', () => {
    hideTurnTooltip();
  });

  segEl.addEventListener('click', (e) => {
    if (e.target.classList.contains('turn-handle')) return;
    e.stopPropagation();
    state.diarization.activeTurnIndex = idx;
    highlightActiveTurn(idx);

    // Also highlight row in turns table
    const row = document.getElementById(`turn-row-${idx}`);
    if (row) {
      document.querySelectorAll('.diar-turns-table tr').forEach(r => r.classList.remove('selected-row'));
      row.classList.add('selected-row');
      row.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    const audioId = state.diarization.audioId || el.diarInputSelect.value;
    if (audioId) {
      if (el.audio.src.indexOf(audioId) === -1) loadAudioIntoPlayer(audioId);
      seekTo(turn.start_s);
      el.audio.play();
    }
  });

  segEl.addEventListener('dblclick', (e) => {
    e.stopPropagation();
    state.diarization.activeTurnIndex = idx;
    highlightActiveTurn(idx);
    const row = document.getElementById(`turn-row-${idx}`);
    if (row) {
      row.scrollIntoView({ behavior: 'smooth', block: 'center' });
      row.classList.add('selected-row');
      setTimeout(() => row.classList.remove('selected-row'), 1500);
    }
  });

  // Drag Resize and Move Handler Initialization
  const leftHandle = segEl.querySelector('.turn-handle-left');
  const rightHandle = segEl.querySelector('.turn-handle-right');

  if (leftHandle) {
    leftHandle.addEventListener('mousedown', (e) => {
      e.stopPropagation();
      e.preventDefault();
      startTurnDrag(idx, 'resize-left', e.clientX);
    });
  }

  if (rightHandle) {
    rightHandle.addEventListener('mousedown', (e) => {
      e.stopPropagation();
      e.preventDefault();
      startTurnDrag(idx, 'resize-right', e.clientX);
    });
  }

  segEl.addEventListener('mousedown', (e) => {
    if (e.target.classList.contains('turn-handle')) return;
    e.stopPropagation();
    startTurnDrag(idx, 'move', e.clientX);
  });
}

function startTurnDrag(turnIndex, action, clientX) {
  const turn = state.diarization.turns[turnIndex];
  if (!turn) return;

  state.diarization.dragTurnInfo = {
    turnIndex,
    action,
    origStart: turn.start_s,
    origEnd: turn.end_s,
    startClientX: clientX,
  };
}

function getTrackTimeFromClientX(clientX) {
  const tracksArea = el.diarTracksArea;
  if (!tracksArea) return 0;
  const rect = tracksArea.getBoundingClientRect();
  const dur = state.diarization.duration || 1;
  const relX = Math.max(0, Math.min(rect.width, clientX - rect.left));
  return (relX / rect.width) * dur;
}

function setupTimelineDragListeners() {
  const tracksArea = el.diarTracksArea;
  const viewport = el.diarMultitrackViewport;
  if (!tracksArea || !viewport) return;

  let isSelectingRegion = false;
  let selectionStartSec = 0;

  // Window mousemove for turn dragging, playhead scrubbing, or region selection
  window.addEventListener('mousemove', (e) => {
    const dur = state.diarization.duration || 1;
    const tracksRect = tracksArea.getBoundingClientRect();
    const trackWidth = tracksRect.width || 800;

    // 1. Playhead Scrubbing
    if (state.diarization.isScrubbing) {
      const scrubSec = getTrackTimeFromClientX(e.clientX);
      seekTo(scrubSec);
      return;
    }

    // 2. Handling Turn Resizing or Moving (Smooth live DOM update without re-rendering everything!)
    if (state.diarization.dragTurnInfo) {
      const { turnIndex, action, origStart, origEnd, startClientX } = state.diarization.dragTurnInfo;
      const turn = state.diarization.turns[turnIndex];
      if (!turn) return;

      const deltaPx = e.clientX - startClientX;
      const deltaSec = (deltaPx / trackWidth) * dur;

      let tempStart = origStart;
      let tempEnd = origEnd;

      if (action === 'resize-left') {
        tempStart = Math.max(0, Math.min(origEnd - 0.1, origStart + deltaSec));
        if (e.shiftKey) tempStart = Math.round(tempStart * 10) / 10;
        turn.start_s = roundNum(tempStart, 2);
      } else if (action === 'resize-right') {
        tempEnd = Math.max(origStart + 0.1, Math.min(dur, origEnd + deltaSec));
        if (e.shiftKey) tempEnd = Math.round(tempEnd * 10) / 10;
        turn.end_s = roundNum(tempEnd, 2);
      } else if (action === 'move') {
        const segDur = origEnd - origStart;
        tempStart = Math.max(0, Math.min(dur - segDur, origStart + deltaSec));
        if (e.shiftKey) tempStart = Math.round(tempStart * 10) / 10;
        turn.start_s = roundNum(tempStart, 2);
        turn.end_s = roundNum(tempStart + segDur, 2);
      }

      // Update segment elements directly in DOM
      const leftPct = (turn.start_s / dur) * 100;
      const widthPct = Math.max(0.4, ((turn.end_s - turn.start_s) / dur) * 100);
      const durText = `${(turn.end_s - turn.start_s).toFixed(1)}s`;

      const matchingSegs = tracksArea.querySelectorAll(`.diar-turn-segment[data-index="${turnIndex}"]`);
      matchingSegs.forEach(seg => {
        seg.style.left = `${leftPct}%`;
        seg.style.width = `${widthPct}%`;
        const label = seg.querySelector('.turn-label-text');
        if (label) {
          if (seg.closest('.composite-row') || seg.closest('.diar-composite-track')) {
            label.textContent = `${getSpeakerName(turn.speaker_id)} (${durText})`;
          } else {
            label.textContent = durText;
          }
        }
      });

      showTurnTooltip(e, turn);
      return;
    }

    // 3. Handling Region Selection Dragging
    if (isSelectingRegion && el.diarSelectionOverlay) {
      const currentSec = getTrackTimeFromClientX(e.clientX);
      const s = Math.min(selectionStartSec, currentSec);
      const end = Math.max(selectionStartSec, currentSec);

      state.diarization.selectionRange = { start_s: s, end_s: end };
      const leftPct = (s / dur) * 100;
      const widthPct = ((end - s) / dur) * 100;

      el.diarSelectionOverlay.classList.remove('hidden');
      el.diarSelectionOverlay.style.left = `${leftPct}%`;
      el.diarSelectionOverlay.style.width = `${widthPct}%`;
    }
  });

  window.addEventListener('mouseup', () => {
    if (state.diarization.isScrubbing) {
      state.diarization.isScrubbing = false;
    }

    if (state.diarization.dragTurnInfo) {
      state.diarization.dragTurnInfo = null;
      hideTurnTooltip();
      state.diarization.turns.sort((a, b) => a.start_s - b.start_s);
      detectTurnOverlaps();
      renderSpeakerSwimlanes();
      renderCompositeLane();
      renderSpeakerProfiles();
      renderTurnsTable();
      renderDiarMinimap();
      updateExportPreview();
    }

    if (isSelectingRegion) {
      isSelectingRegion = false;
      if (state.diarization.selectionRange && (state.diarization.selectionRange.end_s - state.diarization.selectionRange.start_s) > 0.15) {
        if (el.diarSelectionActionsBar) el.diarSelectionActionsBar.classList.remove('hidden');
      } else {
        clearTimelineSelection();
      }
    }
  });

  // Track area mousedown for scrubbing / seeking or region selecting
  tracksArea.addEventListener('mousedown', (e) => {
    if (e.target.closest('.diar-turn-segment') || e.target.closest('.diar-selection-actions-bar')) {
      return;
    }

    const clickSec = getTrackTimeFromClientX(e.clientX);

    if (e.shiftKey) {
      // Shift+Click drag initiates range selection
      isSelectingRegion = true;
      selectionStartSec = clickSec;
    } else {
      // Direct click or scrub
      state.diarization.isScrubbing = true;
      seekTo(clickSec);
    }
  });

  // Playhead scrubber handle mousedown
  if (el.diarPlayheadHandle) {
    el.diarPlayheadHandle.addEventListener('mousedown', (e) => {
      e.stopPropagation();
      e.preventDefault();
      state.diarization.isScrubbing = true;
    });
  }
}

function clearTimelineSelection() {
  state.diarization.selectionRange = null;
  if (el.diarSelectionOverlay) el.diarSelectionOverlay.classList.add('hidden');
}

function showTurnTooltip(e, turn) {
  const tooltip = el.diarTurnTooltip;
  if (!tooltip) return;
  const spkName = getSpeakerName(turn.speaker_id);
  const dur = (turn.end_s - turn.start_s).toFixed(2);

  tooltip.innerHTML = `
    <strong>${escapeHtml(spkName)}</strong> (${turn.speaker_id})<br>
    Start: <code>${turn.start_s.toFixed(2)}s</code> • End: <code>${turn.end_s.toFixed(2)}s</code><br>
    Duration: <strong>${dur}s</strong> ${turn.has_overlap ? '• ⚠️ Overlap' : ''}
  `;

  tooltip.classList.remove('hidden');
  tooltip.style.left = `${e.clientX + 14}px`;
  tooltip.style.top = `${e.clientY + 14}px`;
}

function hideTurnTooltip() {
  if (el.diarTurnTooltip) el.diarTurnTooltip.classList.add('hidden');
}

function highlightActiveTurn(idx) {
  document.querySelectorAll('.diar-turn-segment').forEach(s => {
    s.classList.toggle('active-turn', parseInt(s.dataset.index) === idx);
  });
}

function updateDiarizationPlayhead(currentTime, totalDuration) {
  if (!el.diarPlayheadLine) return;
  const dur = totalDuration || state.diarization.duration || 1;
  const pct = Math.min(100, Math.max(0, (currentTime / dur) * 100));

  el.diarPlayheadLine.style.left = `${pct}%`;
  if (el.diarMinimapPlayhead) el.diarMinimapPlayhead.style.left = `${pct}%`;

  if (el.diarTimeCurrent) el.diarTimeCurrent.textContent = formatTimePrecise(currentTime);
  if (el.diarTimeTotal) el.diarTimeTotal.textContent = formatTimePrecise(dur);

  // Turn looping handler
  if (state.diarization.loopTurn && state.diarization.activeTurnIndex !== null) {
    const turn = state.diarization.turns[state.diarization.activeTurnIndex];
    if (turn && currentTime >= turn.end_s) {
      seekTo(turn.start_s);
    }
  }

  // Active turn segment and table row tracking
  const activeTurn = state.diarization.turns.find(t => currentTime >= t.start_s && currentTime <= t.end_s);
  if (activeTurn) {
    const idx = state.diarization.turns.indexOf(activeTurn);
    highlightActiveTurn(idx);

    document.querySelectorAll('.diar-turns-table tr').forEach(r => r.classList.remove('playing-row'));
    const activeRow = document.getElementById(`turn-row-${idx}`);
    if (activeRow) activeRow.classList.add('playing-row');
  }

  // Speaker Solo / Mute Audio Handling
  applyDiarSoloMuteAudio(currentTime);

  // Follow Playhead Auto-Scroll
  if (state.diarization.followPlayhead && !el.audio.paused && el.diarMultitrackViewport) {
    const viewport = el.diarMultitrackViewport;
    const totalWidth = viewport.scrollWidth;
    const playheadPx = (pct / 100) * totalWidth;
    const scrollMin = viewport.scrollLeft;
    const scrollMax = viewport.scrollLeft + viewport.clientWidth;

    if (playheadPx > scrollMax - 80 || playheadPx < scrollMin) {
      viewport.scrollLeft = Math.max(0, playheadPx - viewport.clientWidth / 3);
    }
  }
}

function applyDiarSoloMuteAudio(currentTime) {
  if (state.activeTab !== 'tab-diarization') return;

  const isSoloActive = Boolean(state.diarization.soloSpeaker);
  const hasMuted = state.diarization.mutedSpeakers && state.diarization.mutedSpeakers.size > 0;
  const allSpeakers = state.diarization.speakers || [];
  const allMuted = allSpeakers.length > 0 && allSpeakers.every(s => state.diarization.mutedSpeakers.has(s.speaker_id));

  if (allMuted) {
    el.audio.muted = true;
    return;
  }

  if (isSoloActive || hasMuted) {
    const activeTurns = (state.diarization.turns || []).filter(t => currentTime >= t.start_s && currentTime <= t.end_s);
    if (activeTurns.length > 0) {
      if (isSoloActive) {
        const hasSoloMatch = activeTurns.some(t => t.speaker_id === state.diarization.soloSpeaker);
        el.audio.muted = !hasSoloMatch;
      } else {
        const hasUnmutedTurn = activeTurns.some(t => !state.diarization.mutedSpeakers.has(t.speaker_id));
        el.audio.muted = !hasUnmutedTurn;
      }
    } else {
      // In silence gaps: if solo is active, silence gaps so only the solo speaker's voice is heard
      if (isSoloActive) {
        el.audio.muted = true;
      } else {
        el.audio.muted = false;
      }
    }
  } else {
    el.audio.muted = false;
  }
}

function toggleSpeakerSolo(speakerId) {
  if (state.diarization.soloSpeaker === speakerId) {
    state.diarization.soloSpeaker = null;
    showToast(`Solo disabled for ${getSpeakerName(speakerId)}`, 'info');
  } else {
    state.diarization.soloSpeaker = speakerId;
    if (state.diarization.mutedSpeakers) {
      state.diarization.mutedSpeakers.delete(speakerId);
    }
    showToast(`Soloing ${getSpeakerName(speakerId)} [S]`, 'success');
  }
  applyDiarSoloMuteAudio(el.audio.currentTime || 0);
  renderSpeakerSwimlanes();
  renderCompositeLane();
  renderSpeakerProfiles();
}

function toggleSpeakerMute(speakerId) {
  if (state.diarization.mutedSpeakers.has(speakerId)) {
    state.diarization.mutedSpeakers.delete(speakerId);
    showToast(`Unmuted ${getSpeakerName(speakerId)}`, 'info');
  } else {
    state.diarization.mutedSpeakers.add(speakerId);
    if (state.diarization.soloSpeaker === speakerId) {
      state.diarization.soloSpeaker = null;
    }
    showToast(`Muted ${getSpeakerName(speakerId)} [M]`, 'info');
  }
  applyDiarSoloMuteAudio(el.audio.currentTime || 0);
  renderSpeakerSwimlanes();
  renderCompositeLane();
  renderSpeakerProfiles();
}

function toggleMuteAllSpeakers() {
  const allSpeakers = state.diarization.speakers || [];
  if (allSpeakers.length === 0) return;

  const allMuted = allSpeakers.length > 0 && allSpeakers.every(s => state.diarization.mutedSpeakers.has(s.speaker_id));

  if (allMuted) {
    state.diarization.mutedSpeakers.clear();
    if (el.iconDiarMuteAll) el.iconDiarMuteAll.textContent = "🔇";
    if (el.diarMuteAllLabel) el.diarMuteAllLabel.textContent = "Mute All";
    if (el.btnDiarMuteAll) el.btnDiarMuteAll.classList.remove('active');
    showToast("Unmuted all speakers", "info");
  } else {
    allSpeakers.forEach(s => state.diarization.mutedSpeakers.add(s.speaker_id));
    state.diarization.soloSpeaker = null;
    if (el.iconDiarMuteAll) el.iconDiarMuteAll.textContent = "🔊";
    if (el.diarMuteAllLabel) el.diarMuteAllLabel.textContent = "Unmute All";
    if (el.btnDiarMuteAll) el.btnDiarMuteAll.classList.add('active');
    showToast("Muted all speakers", "info");
  }

  applyDiarSoloMuteAudio(el.audio.currentTime || 0);
  renderSpeakerSwimlanes();
  renderCompositeLane();
  renderSpeakerProfiles();
}

function renderSpeakerProfiles() {
  if (!el.diarSpeakersGrid) return;
  el.diarSpeakersGrid.innerHTML = "";
  const totalDur = state.diarization.duration || 1;

  state.diarization.speakers.forEach(spk => {
    const spkId = spk.speaker_id;
    const spkName = getSpeakerName(spkId);
    const color = getSpeakerColor(spkId);
    const spkTurns = state.diarization.turns.filter(t => t.speaker_id === spkId);

    const isSolo = state.diarization.soloSpeaker === spkId;
    const isMuted = state.diarization.mutedSpeakers.has(spkId);

    const totalSpeechS = spkTurns.reduce((acc, t) => acc + Math.max(0, t.end_s - t.start_s), 0);
    const turnsCount = spkTurns.length;
    const talkPct = ((totalSpeechS / totalDur) * 100).toFixed(1);
    const avgDurS = turnsCount > 0 ? (totalSpeechS / turnsCount).toFixed(2) : "0.00";
    const overlapDurS = spkTurns.filter(t => t.has_overlap).reduce((acc, t) => acc + (t.end_s - t.start_s), 0).toFixed(1);

    const card = document.createElement("div");
    card.className = "diar-speaker-card";
    card.innerHTML = `
      <div class="diar-spk-header">
        <span class="diar-spk-avatar" style="background-color: ${color};"></span>
        <input type="text" class="diar-spk-name-input" value="${escapeHtml(spkName)}" title="Click to rename speaker" data-speaker="${spkId}">
        <input type="color" class="diar-spk-color-picker" value="${hslToHex(color)}" title="Change speaker color" data-speaker="${spkId}">
      </div>

      <div class="diar-spk-share-bar-track">
        <div class="diar-spk-share-bar-fill" style="width: ${talkPct}%; background-color: ${color};"></div>
      </div>

      <div class="diar-spk-stats-grid">
        <div class="diar-spk-stat-item">
          <span class="diar-spk-stat-label">Speech Time:</span>
          <span class="diar-spk-stat-val">${totalSpeechS.toFixed(1)}s (${talkPct}%)</span>
        </div>
        <div class="diar-spk-stat-item">
          <span class="diar-spk-stat-label">Turn Count:</span>
          <span class="diar-spk-stat-val">${turnsCount} turns</span>
        </div>
        <div class="diar-spk-stat-item">
          <span class="diar-spk-stat-label">Avg Turn:</span>
          <span class="diar-spk-stat-val">${avgDurS}s</span>
        </div>
        <div class="diar-spk-stat-item">
          <span class="diar-spk-stat-label">Overlaps:</span>
          <span class="diar-spk-stat-val">${overlapDurS}s</span>
        </div>
      </div>

      <div class="diar-spk-actions-row">
        <button class="btn btn-xs btn-secondary btn-audition-spk" data-speaker="${spkId}" title="Play all turns of this speaker sequentially">▶ Audition</button>
        <button class="btn btn-xs ${isSolo ? 'btn-warning' : 'btn-ghost'} btn-solo-spk" data-speaker="${spkId}" title="Solo Speaker">Solo</button>
        <button class="btn btn-xs ${isMuted ? 'btn-destructive' : 'btn-ghost'} btn-mute-spk" data-speaker="${spkId}" title="Mute Speaker">Mute</button>
        <button class="btn btn-xs btn-ghost btn-filter-spk" data-speaker="${spkId}" title="Filter turns table">🔍 Filter</button>
        <button class="btn btn-xs btn-ghost btn-merge-spk" data-speaker="${spkId}" title="Merge into another speaker">🔀 Merge</button>
        <button class="btn btn-xs btn-primary btn-extract-spk" data-speaker="${spkId}" title="Extract and save speaker audio to workspace">✂ Extract</button>
      </div>
    `;

    // Event listeners
    const nameInput = card.querySelector('.diar-spk-name-input');
    nameInput.addEventListener('change', (e) => {
      const val = e.target.value.trim() || spkId;
      state.diarization.customNames[spkId] = val;
      renderSpeakerSwimlanes();
      renderCompositeLane();
      renderTurnsTable();
      updateExportPreview();
      showToast(`Speaker ${spkId} renamed to "${val}"`, "success");
    });

    const colorPicker = card.querySelector('.diar-spk-color-picker');
    colorPicker.addEventListener('input', (e) => {
      state.diarization.colors[spkId] = e.target.value;
      card.querySelector('.diar-spk-avatar').style.backgroundColor = e.target.value;
      card.querySelector('.diar-spk-share-bar-fill').style.backgroundColor = e.target.value;
      renderSpeakerSwimlanes();
      renderCompositeLane();
      renderTurnsTable();
      renderDiarMinimap();
    });

    card.querySelector('.btn-audition-spk').addEventListener('click', () => auditionSpeakerTurns(spkId));
    card.querySelector('.btn-solo-spk').addEventListener('click', () => toggleSpeakerSolo(spkId));
    card.querySelector('.btn-mute-spk').addEventListener('click', () => toggleSpeakerMute(spkId));
    card.querySelector('.btn-filter-spk').addEventListener('click', () => {
      if (el.diarFilterSpeakerSelect) {
        el.diarFilterSpeakerSelect.value = spkId;
        state.diarization.activeSpeakerFilter = spkId;
        renderTurnsTable();
      }
    });
    card.querySelector('.btn-merge-spk').addEventListener('click', () => openMergeModal(spkId));
    card.querySelector('.btn-extract-spk').addEventListener('click', () => extractSpeakerAudio(spkId, spkName));

    el.diarSpeakersGrid.appendChild(card);
  });
}

function getFilteredAndSortedTurns() {
  let turns = state.diarization.turns.map((t, idx) => ({ ...t, originalIndex: idx }));

  // Filter by Speaker
  if (state.diarization.activeSpeakerFilter && state.diarization.activeSpeakerFilter !== 'all') {
    turns = turns.filter(t => t.speaker_id === state.diarization.activeSpeakerFilter);
  }

  // Filter by Search Query
  if (state.diarization.searchQuery) {
    const q = state.diarization.searchQuery;
    turns = turns.filter(t => {
      const spkName = getSpeakerName(t.speaker_id).toLowerCase();
      return spkName.includes(q) || t.speaker_id.toLowerCase().includes(q) || `#${t.originalIndex + 1}`.includes(q) || t.start_s.toString().includes(q);
    });
  }

  // Filter by Min Duration
  if (state.diarization.minDurFilter > 0) {
    turns = turns.filter(t => (t.end_s - t.start_s) >= state.diarization.minDurFilter);
  }

  // Sort
  if (state.diarization.sortMode === 'time-desc') {
    turns.sort((a, b) => b.start_s - a.start_s);
  } else if (state.diarization.sortMode === 'dur-desc') {
    turns.sort((a, b) => (b.end_s - b.start_s) - (a.end_s - a.start_s));
  } else if (state.diarization.sortMode === 'speaker') {
    turns.sort((a, b) => a.speaker_id.localeCompare(b.speaker_id));
  } else {
    turns.sort((a, b) => a.start_s - b.start_s);
  }

  return turns;
}

function updateBatchBar() {
  const count = state.diarization.selectedTurnIndices.size;
  if (el.diarBatchCount) el.diarBatchCount.textContent = count;
  if (el.diarBatchBar) {
    el.diarBatchBar.classList.toggle('hidden', count === 0);
  }
}

function renderTurnsTable() {
  if (!el.turnsTableBody) return;
  el.turnsTableBody.innerHTML = "";

  const turns = getFilteredAndSortedTurns();

  if (el.diarFilteredTurnsCount) {
    el.diarFilteredTurnsCount.textContent = `${turns.length} of ${state.diarization.turns.length} turns`;
  }

  if (turns.length === 0) {
    el.turnsTableBody.innerHTML = `<tr><td colspan="8" class="text-center text-muted" style="padding: 24px;">No turns match the active filter criteria.</td></tr>`;
    return;
  }

  turns.forEach(turn => {
    const idx = turn.originalIndex;
    const color = getSpeakerColor(turn.speaker_id);
    const duration = Math.max(0, turn.end_s - turn.start_s).toFixed(2);
    const isSelected = state.diarization.selectedTurnIndices.has(idx);

    const tr = document.createElement("tr");
    tr.id = `turn-row-${idx}`;
    if (isSelected) tr.classList.add('selected-row');

    tr.innerHTML = `
      <td style="text-align: center;">
        <input type="checkbox" class="turn-select-checkbox" data-index="${idx}" ${isSelected ? 'checked' : ''}>
      </td>
      <td><span class="text-muted font-mono">#${idx + 1}</span></td>
      <td>
        <select class="turn-speaker-select" data-index="${idx}" style="color: ${color}; font-weight: 700;">
          ${state.diarization.speakers.map(s => `
            <option value="${s.speaker_id}" ${s.speaker_id === turn.speaker_id ? 'selected' : ''}>
              ${escapeHtml(getSpeakerName(s.speaker_id))}
            </option>
          `).join('')}
        </select>
      </td>
      <td>
        <div class="flex-row items-center gap-1">
          <button class="btn btn-ghost turn-nudge-btn" data-index="${idx}" data-field="start" data-delta="-0.1">-0.1</button>
          <input type="number" step="0.05" class="turn-time-input turn-start-input" data-index="${idx}" value="${turn.start_s.toFixed(2)}">
          <button class="btn btn-ghost turn-nudge-btn" data-index="${idx}" data-field="start" data-delta="0.1">+0.1</button>
        </div>
      </td>
      <td>
        <div class="flex-row items-center gap-1">
          <button class="btn btn-ghost turn-nudge-btn" data-index="${idx}" data-field="end" data-delta="-0.1">-0.1</button>
          <input type="number" step="0.05" class="turn-time-input turn-end-input" data-index="${idx}" value="${turn.end_s.toFixed(2)}">
          <button class="btn btn-ghost turn-nudge-btn" data-index="${idx}" data-field="end" data-delta="0.1">+0.1</button>
        </div>
      </td>
      <td><span class="badge badge-ghost">${duration}s</span></td>
      <td>${turn.has_overlap ? '<span class="badge badge-warning">⚠️ Overlap</span>' : '<span class="text-muted text-xs">Clean</span>'}</td>
      <td class="table-actions">
        <button class="btn btn-sm btn-ghost btn-play-turn" data-index="${idx}" title="Play turn segment">▶ Play</button>
        <button class="btn btn-sm btn-ghost btn-loop-turn" data-index="${idx}" title="Loop turn segment">🔁 Loop</button>
        <button class="btn btn-sm btn-secondary btn-cut-turn" data-index="${idx}" title="Send segment to Clips library">✂ Cut</button>
        <button class="btn btn-sm btn-ghost btn-split-turn" data-index="${idx}" title="Split turn into two">✂ Split</button>
        <button class="btn btn-sm btn-ghost btn-merge-next-turn" data-index="${idx}" title="Merge with next chronological turn">🔗 Merge</button>
        <button class="btn btn-sm btn-ghost btn-delete-turn text-destructive" data-index="${idx}" title="Delete turn">🗑</button>
      </td>
    `;

    // Row selection checkbox
    tr.querySelector('.turn-select-checkbox').addEventListener('change', (e) => {
      if (e.target.checked) {
        state.diarization.selectedTurnIndices.add(idx);
        tr.classList.add('selected-row');
      } else {
        state.diarization.selectedTurnIndices.delete(idx);
        tr.classList.remove('selected-row');
      }
      updateBatchBar();
    });

    // Reassign Speaker
    tr.querySelector('.turn-speaker-select').addEventListener('change', (e) => {
      state.diarization.turns[idx].speaker_id = e.target.value;
      renderSpeakerSwimlanes();
      renderCompositeLane();
      renderSpeakerProfiles();
      renderDiarMinimap();
      updateExportPreview();
    });

    // Start / End inputs
    const startInput = tr.querySelector('.turn-start-input');
    startInput.addEventListener('change', (e) => {
      const val = Math.max(0, parseFloat(e.target.value) || 0);
      state.diarization.turns[idx].start_s = val;
      if (state.diarization.turns[idx].end_s <= val) {
        state.diarization.turns[idx].end_s = val + 0.1;
      }
      detectTurnOverlaps();
      renderSpeakerSwimlanes();
      renderCompositeLane();
      renderSpeakerProfiles();
      renderDiarMinimap();
      renderTurnsTable();
      updateExportPreview();
    });

    const endInput = tr.querySelector('.turn-end-input');
    endInput.addEventListener('change', (e) => {
      const val = Math.max(state.diarization.turns[idx].start_s + 0.05, parseFloat(e.target.value) || 0);
      state.diarization.turns[idx].end_s = val;
      detectTurnOverlaps();
      renderSpeakerSwimlanes();
      renderCompositeLane();
      renderSpeakerProfiles();
      renderDiarMinimap();
      renderTurnsTable();
      updateExportPreview();
    });

    // Nudge buttons
    tr.querySelectorAll('.turn-nudge-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const field = btn.dataset.field;
        const delta = parseFloat(btn.dataset.delta);
        if (field === 'start') {
          state.diarization.turns[idx].start_s = Math.max(0, roundNum(state.diarization.turns[idx].start_s + delta, 2));
        } else {
          state.diarization.turns[idx].end_s = Math.max(state.diarization.turns[idx].start_s + 0.05, roundNum(state.diarization.turns[idx].end_s + delta, 2));
        }
        detectTurnOverlaps();
        renderTurnsTable();
        renderSpeakerSwimlanes();
        renderCompositeLane();
        renderSpeakerProfiles();
        renderDiarMinimap();
        updateExportPreview();
      });
    });

    // Action buttons
    tr.querySelector('.btn-play-turn').addEventListener('click', () => {
      const audioId = state.diarization.audioId || el.diarInputSelect.value;
      loadAudioIntoPlayer(audioId);
      seekTo(turn.start_s);
      state.player.previewEnd = turn.end_s;
      el.audio.play();
    });

    tr.querySelector('.btn-loop-turn').addEventListener('click', () => {
      state.diarization.activeTurnIndex = idx;
      state.diarization.loopTurn = true;
      if (el.btnDiarLoopTurn) el.btnDiarLoopTurn.classList.add('active');
      const audioId = state.diarization.audioId || el.diarInputSelect.value;
      loadAudioIntoPlayer(audioId);
      seekTo(turn.start_s);
      el.audio.play();
      showToast(`Looping turn #${idx + 1} (${turn.start_s.toFixed(2)}s - ${turn.end_s.toFixed(2)}s)`, 'info');
    });

    tr.querySelector('.btn-cut-turn').addEventListener('click', () => {
      const audioId = state.diarization.audioId || el.diarInputSelect.value;
      addCutToRegistry(audioId, turn.start_s, turn.end_s, 'seconds');
      showToast(`Added turn #${idx + 1} to Clips registry!`, 'success');
    });

    tr.querySelector('.btn-split-turn').addEventListener('click', () => {
      const mid = roundNum(turn.start_s + (turn.end_s - turn.start_s) / 2, 2);
      const turn1 = { ...turn, end_s: mid };
      const turn2 = { ...turn, start_s: mid };
      state.diarization.turns.splice(idx, 1, turn1, turn2);
      detectTurnOverlaps();
      renderDiarizationWorkspace(state.diarization.data, state.diarization.audioId, true);
      showToast(`Turn #${idx + 1} split into two segments`, 'success');
    });

    tr.querySelector('.btn-merge-next-turn').addEventListener('click', () => {
      // Find next chronological turn
      const nextIdx = state.diarization.turns.findIndex(t => t.start_s >= turn.start_s && t !== state.diarization.turns[idx]);
      if (nextIdx === -1) {
        showToast("No subsequent chronological turn found to merge with", "info");
        return;
      }
      const nextTurn = state.diarization.turns[nextIdx];
      turn.end_s = Math.max(turn.end_s, nextTurn.end_s);
      state.diarization.turns.splice(nextIdx, 1);
      detectTurnOverlaps();
      renderDiarizationWorkspace(state.diarization.data, state.diarization.audioId, true);
      showToast(`Merged turn #${idx + 1} with subsequent turn`, 'success');
    });

    tr.querySelector('.btn-delete-turn').addEventListener('click', () => {
      state.diarization.turns.splice(idx, 1);
      state.diarization.selectedTurnIndices.delete(idx);
      detectTurnOverlaps();
      renderDiarizationWorkspace(state.diarization.data, state.diarization.audioId, true);
      showToast(`Turn #${idx + 1} removed`, 'info');
    });

    el.turnsTableBody.appendChild(tr);
  });
}

function addTurnAtCursor() {
  const curTime = state.player.currentTime || 0;
  const dur = state.diarization.duration || 10;
  const start = Math.max(0, Math.min(curTime, dur - 1));
  const end = Math.min(dur, start + 2.0);
  const spkId = state.diarization.speakers[0]?.speaker_id || "spk_00";

  const newTurn = {
    speaker_id: spkId,
    start_s: roundNum(start, 2),
    end_s: roundNum(end, 2),
    confidence: 1.0,
  };

  state.diarization.turns.push(newTurn);
  state.diarization.turns.sort((a, b) => a.start_s - b.start_s);
  detectTurnOverlaps();
  renderDiarizationWorkspace(state.diarization.data, state.diarization.audioId, true);
  showToast(`Added new turn at ${start.toFixed(2)}s`, 'success');
}

function navigateToAdjacentTurn(delta) {
  const turns = state.diarization.turns;
  if (!turns || turns.length === 0) return;
  let nextIdx = 0;
  if (state.diarization.activeTurnIndex !== null) {
    nextIdx = (state.diarization.activeTurnIndex + delta + turns.length) % turns.length;
  } else {
    nextIdx = delta > 0 ? 0 : turns.length - 1;
  }
  state.diarization.activeTurnIndex = nextIdx;
  const turn = turns[nextIdx];
  highlightActiveTurn(nextIdx);
  seekTo(turn.start_s);
  const row = document.getElementById(`turn-row-${nextIdx}`);
  if (row) {
    row.scrollIntoView({ behavior: 'smooth', block: 'center' });
    row.classList.add('selected-row');
    setTimeout(() => row.classList.remove('selected-row'), 1000);
  }
  showToast(`Selected Turn #${nextIdx + 1} (${getSpeakerName(turn.speaker_id)})`, 'info');
}

function splitActiveTurnAtPlayhead() {
  const curTime = state.player.currentTime || 0;
  const turns = state.diarization.turns;
  let targetIdx = state.diarization.activeTurnIndex;
  if (targetIdx === null) {
    targetIdx = turns.findIndex(t => curTime > t.start_s + 0.1 && curTime < t.end_s - 0.1);
  }
  if (targetIdx === -1 || targetIdx === null || !turns[targetIdx]) {
    showToast("Place playhead inside a turn segment to split it", "info");
    return;
  }
  const turn = turns[targetIdx];
  if (curTime <= turn.start_s + 0.05 || curTime >= turn.end_s - 0.05) {
    showToast("Playhead is at boundary of turn", "info");
    return;
  }
  const turn1 = { ...turn, end_s: roundNum(curTime, 2) };
  const turn2 = { ...turn, start_s: roundNum(curTime, 2) };
  turns.splice(targetIdx, 1, turn1, turn2);
  detectTurnOverlaps();
  renderDiarizationWorkspace(state.diarization.data, state.diarization.audioId, true);
  showToast(`Split turn #${targetIdx + 1} at ${curTime.toFixed(2)}s`, 'success');
}

function deleteActiveTurn() {
  if (state.diarization.activeTurnIndex === null) {
    showToast("No turn selected to delete", "info");
    return;
  }
  const idx = state.diarization.activeTurnIndex;
  state.diarization.turns.splice(idx, 1);
  state.diarization.selectedTurnIndices.delete(idx);
  state.diarization.activeTurnIndex = null;
  detectTurnOverlaps();
  renderDiarizationWorkspace(state.diarization.data, state.diarization.audioId, true);
  showToast(`Turn #${idx + 1} deleted`, 'info');
}

function runAutoMergeMicroGaps(thresholdSec) {
  const turns = state.diarization.turns;
  if (turns.length <= 1) {
    showToast("Not enough turns to merge", "info");
    return;
  }

  turns.sort((a, b) => a.start_s - b.start_s);
  let mergedCount = 0;
  const merged = [];

  for (let i = 0; i < turns.length; i++) {
    const cur = { ...turns[i] };
    if (merged.length > 0) {
      const prev = merged[merged.length - 1];
      if (prev.speaker_id === cur.speaker_id && (cur.start_s - prev.end_s) <= thresholdSec && cur.start_s >= prev.end_s) {
        prev.end_s = Math.max(prev.end_s, cur.end_s);
        mergedCount++;
        continue;
      }
    }
    merged.push(cur);
  }

  state.diarization.turns = merged;
  detectTurnOverlaps();
  renderDiarizationWorkspace(state.diarization.data, state.diarization.audioId, true);
  showToast(`Merged ${mergedCount} micro-gaps (≤ ${thresholdSec}s)!`, 'success');
}

function auditionSpeakerTurns(speakerId) {
  const turns = state.diarization.turns.filter(t => t.speaker_id === speakerId);
  if (turns.length === 0) {
    showToast("No turns found for this speaker", "info");
    return;
  }

  diarAuditionQueue = [...turns];
  diarAuditionIndex = 0;
  diarAuditionActive = true;

  playNextAuditionTurn();
  showToast(`Auditioning ${turns.length} turns for ${getSpeakerName(speakerId)}...`, 'info');
}

function playNextAuditionTurn() {
  if (diarAuditionIndex >= diarAuditionQueue.length) {
    diarAuditionActive = false;
    showToast("Speaker audition finished", "success");
    return;
  }

  const turn = diarAuditionQueue[diarAuditionIndex];
  const audioId = state.diarization.audioId || el.diarInputSelect.value;
  loadAudioIntoPlayer(audioId);
  seekTo(turn.start_s);
  state.player.previewEnd = turn.end_s;
  el.audio.play();

  // Monitor end of turn
  const checkTurnEnd = () => {
    if (el.audio.currentTime >= turn.end_s - 0.05 || el.audio.paused) {
      el.audio.removeEventListener('timeupdate', checkTurnEnd);
      diarAuditionIndex++;
      setTimeout(playNextAuditionTurn, 400);
    }
  };
  el.audio.addEventListener('timeupdate', checkTurnEnd);
}

// Modal Merge Speaker
let activeMergeSourceSpeaker = null;

function openMergeModal(sourceSpeakerId) {
  activeMergeSourceSpeaker = sourceSpeakerId;
  if (!el.modalMergeSpeaker) return;

  const spkName = getSpeakerName(sourceSpeakerId);
  if (el.mergeSourceSpkName) el.mergeSourceSpkName.textContent = `${spkName} (${sourceSpeakerId})`;

  if (el.mergeTargetSpkSelect) {
    const targets = state.diarization.speakers.filter(s => s.speaker_id !== sourceSpeakerId);
    el.mergeTargetSpkSelect.innerHTML = targets.map(s => `
      <option value="${s.speaker_id}">${escapeHtml(getSpeakerName(s.speaker_id))} (${s.speaker_id})</option>
    `).join('');
  }

  el.modalMergeSpeaker.classList.remove('hidden');
}

function closeMergeModal() {
  if (el.modalMergeSpeaker) el.modalMergeSpeaker.classList.add('hidden');
  activeMergeSourceSpeaker = null;
}

function confirmMergeSpeakers() {
  const targetSpkId = el.mergeTargetSpkSelect?.value;
  if (!activeMergeSourceSpeaker || !targetSpkId) return;

  const srcName = getSpeakerName(activeMergeSourceSpeaker);
  const tgtName = getSpeakerName(targetSpkId);

  state.diarization.turns.forEach(t => {
    if (t.speaker_id === activeMergeSourceSpeaker) {
      t.speaker_id = targetSpkId;
    }
  });

  // Remove source speaker from list
  state.diarization.speakers = state.diarization.speakers.filter(s => s.speaker_id !== activeMergeSourceSpeaker);

  closeMergeModal();
  detectTurnOverlaps();
  renderDiarizationWorkspace(state.diarization.data, state.diarization.audioId, true);
  showToast(`Merged all turns of ${srcName} into ${tgtName}!`, 'success');
}

// Export Studio Generator
function updateExportPreview() {
  if (!el.exportPreviewTextarea) return;
  const format = state.diarization.activeExportTab || 'rttm';
  const audioId = state.diarization.audioId || 'audio_sample';
  const turns = state.diarization.turns || [];
  const totalDur = state.diarization.duration || 60;

  let content = "";
  let filename = `diarization_${audioId}.${format}`;

  if (format === 'rttm') {
    filename = `diarization_${audioId}.rttm`;
    content = turns.map(t => {
      const dur = (t.end_s - t.start_s).toFixed(3);
      const spkName = getSpeakerName(t.speaker_id);
      return `SPEAKER ${audioId} 1 ${t.start_s.toFixed(3)} ${dur} <NA> <NA> ${spkName} <NA> <NA>`;
    }).join('\n');
  } else if (format === 'json') {
    filename = `diarization_${audioId}.json`;
    const jsonOutput = {
      schema_version: "1.0",
      audio_id: audioId,
      model: state.diarization.data?.model || { backend: "pyannote", model_id: "pyannote/speaker-diarization-community-1" },
      speaker_count: state.diarization.speakers.length,
      speakers: state.diarization.speakers.map(s => ({
        speaker_id: s.speaker_id,
        display_name: getSpeakerName(s.speaker_id),
      })),
      turns: turns.map(t => ({
        speaker_id: t.speaker_id,
        speaker_name: getSpeakerName(t.speaker_id),
        start_s: roundNum(t.start_s, 3),
        end_s: roundNum(t.end_s, 3),
        duration_s: roundNum(t.end_s - t.start_s, 3),
        confidence: t.confidence || null,
        has_overlap: Boolean(t.has_overlap),
      })),
    };
    content = JSON.stringify(jsonOutput, null, 2);
  } else if (format === 'csv') {
    filename = `diarization_${audioId}.csv`;
    content = "turn_index,speaker_id,speaker_name,start_seconds,end_seconds,duration_seconds,has_overlap\n" +
      turns.map((t, idx) => `${idx + 1},${t.speaker_id},"${getSpeakerName(t.speaker_id)}",${t.start_s.toFixed(3)},${t.end_s.toFixed(3)},${(t.end_s - t.start_s).toFixed(3)},${t.has_overlap ? 1 : 0}`).join('\n');
  } else if (format === 'srt') {
    filename = `diarization_${audioId}.srt`;
    content = turns.map((t, idx) => {
      const srtStart = secondsToSrtTime(t.start_s);
      const srtEnd = secondsToSrtTime(t.end_s);
      return `${idx + 1}\n${srtStart} --> ${srtEnd}\n[${getSpeakerName(t.speaker_id)}]: Speech turn\n`;
    }).join('\n');
  } else if (format === 'vtt') {
    filename = `diarization_${audioId}.vtt`;
    content = "WEBVTT\n\n" + turns.map((t, idx) => {
      const vttStart = secondsToVttTime(t.start_s);
      const vttEnd = secondsToVttTime(t.end_s);
      return `${idx + 1}\n${vttStart} --> ${vttEnd}\n<v ${getSpeakerName(t.speaker_id)}>Speech turn</v>\n`;
    }).join('\n');
  } else if (format === 'audacity') {
    filename = `diarization_${audioId}_labels.txt`;
    content = turns.map(t => `${t.start_s.toFixed(6)}\t${t.end_s.toFixed(6)}\t${getSpeakerName(t.speaker_id)}`).join('\n');
  } else if (format === 'praat') {
    filename = `diarization_${audioId}.TextGrid`;
    content = `File type = "ooTextFile"\nObject class = "TextGrid"\n\nxmin = 0.0\nxmax = ${totalDur.toFixed(3)}\ntiers? <exists>\nsize = 1\nitem []:\n    item [1]:\n        class = "IntervalTier"\n        name = "speakers"\n        xmin = 0.0\n        xmax = ${totalDur.toFixed(3)}\n        intervals: size = ${turns.length}\n` +
      turns.map((t, idx) => `        intervals [${idx + 1}]:\n            xmin = ${t.start_s.toFixed(3)}\n            xmax = ${t.end_s.toFixed(3)}\n            text = "${getSpeakerName(t.speaker_id)}"\n`).join('');
  }

  el.exportPreviewTextarea.value = content;
  if (el.exportFilenameLabel) el.exportFilenameLabel.textContent = filename;
}

function downloadActiveExport() {
  const content = el.exportPreviewTextarea?.value;
  const filename = el.exportFilenameLabel?.textContent || "diarization.rttm";
  if (!content) return;

  const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  showToast(`Downloaded ${filename}`, 'success');
}

function secondsToSrtTime(seconds) {
  const hrs = Math.floor(seconds / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);
  const millis = Math.floor((seconds % 1) * 1000);
  return `${padZero(hrs, 2)}:${padZero(mins, 2)}:${padZero(secs, 2)},${padZero(millis, 3)}`;
}

function secondsToVttTime(seconds) {
  const hrs = Math.floor(seconds / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);
  const millis = Math.floor((seconds % 1) * 1000);
  return `${padZero(hrs, 2)}:${padZero(mins, 2)}:${padZero(secs, 2)}.${padZero(millis, 3)}`;
}

function padZero(num, size) {
  let s = String(num);
  while (s.length < size) s = "0" + s;
  return s;
}

function roundNum(num, decimals) {
  return Number(Math.round(num + "e" + decimals) + "e-" + decimals);
}

function hslToHex(hslStr) {
  if (!hslStr) return "#00e5ff";
  if (hslStr.startsWith('#')) return hslStr;
  if (!hslStr.startsWith('hsl')) return "#00e5ff";
  const match = hslStr.match(/\d+(\.\d+)?/g);
  if (!match || match.length < 3) return "#00e5ff";
  let h = parseFloat(match[0]) / 360;
  let s = parseFloat(match[1]) / 100;
  let l = parseFloat(match[2]) / 100;
  let r, g, b;
  if (s === 0) {
    r = g = b = l;
  } else {
    const hue2rgb = (p, q, t) => {
      if (t < 0) t += 1;
      if (t > 1) t -= 1;
      if (t < 1/6) return p + (q - p) * 6 * t;
      if (t < 1/2) return q;
      if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
      return p;
    };
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    r = hue2rgb(p, q, h + 1/3);
    g = hue2rgb(p, q, h);
    b = hue2rgb(p, q, h - 1/3);
  }
  const toHex = x => {
    const hex = Math.round(x * 255).toString(16);
    return hex.length === 1 ? '0' + hex : hex;
  };
  return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
}

// Annotation File Parser
function importAnnotationFile(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    try {
      const text = e.target.result;
      const audioId = state.activeAudio?.id || (state.audioList[0] ? state.audioList[0].id : "imported_track");
      let turns = [];
      let speakers = new Set();

      if (file.name.endsWith('.json')) {
        const json = JSON.parse(text);
        turns = json.turns || [];
        if (json.speakers) {
          json.speakers.forEach(s => speakers.add(s.speaker_id || s));
        }
      } else if (file.name.endsWith('.rttm') || text.includes('SPEAKER')) {
        const lines = text.split('\n');
        lines.forEach(line => {
          const parts = line.trim().split(/\s+/);
          if (parts[0] === 'SPEAKER' && parts.length >= 8) {
            const start = parseFloat(parts[3]);
            const dur = parseFloat(parts[4]);
            const spkId = parts[7];
            if (!isNaN(start) && !isNaN(dur) && spkId) {
              speakers.add(spkId);
              turns.push({
                speaker_id: spkId,
                start_s: start,
                end_s: start + dur,
              });
            }
          }
        });
      } else if (file.name.endsWith('.csv')) {
        const lines = text.split('\n');
        lines.slice(1).forEach(line => {
          const parts = line.split(',');
          if (parts.length >= 5) {
            const spkId = parts[1].trim();
            const start = parseFloat(parts[3]);
            const end = parseFloat(parts[4]);
            if (spkId && !isNaN(start) && !isNaN(end)) {
              speakers.add(spkId);
              turns.push({ speaker_id: spkId, start_s: start, end_s: end });
            }
          }
        });
      } else if (file.name.endsWith('.txt') || file.name.endsWith('.labels')) {
        // Audacity Tab-delimited label format: start \t end \t label
        const lines = text.split('\n');
        lines.forEach(line => {
          const parts = line.trim().split('\t');
          if (parts.length >= 3) {
            const start = parseFloat(parts[0]);
            const end = parseFloat(parts[1]);
            const spkId = parts[2].trim();
            if (!isNaN(start) && !isNaN(end) && spkId) {
              speakers.add(spkId);
              turns.push({ speaker_id: spkId, start_s: start, end_s: end });
            }
          }
        });
      } else if (file.name.endsWith('.TextGrid')) {
        // Praat TextGrid parsing
        const intervalMatches = [...text.matchAll(/xmin\s*=\s*([\d\.]+)\s*xmax\s*=\s*([\d\.]+)\s*text\s*=\s*"([^"]+)"/g)];
        intervalMatches.forEach(m => {
          const start = parseFloat(m[1]);
          const end = parseFloat(m[2]);
          const spkId = m[3].trim();
          if (!isNaN(start) && !isNaN(end) && spkId) {
            speakers.add(spkId);
            turns.push({ speaker_id: spkId, start_s: start, end_s: end });
          }
        });
      }

      if (turns.length === 0) {
        throw new Error("No valid speaker turns could be parsed from this file");
      }

      turns.sort((a, b) => a.start_s - b.start_s);
      const diarResult = {
        schema_version: "1.0",
        audio_id: audioId,
        speakers: Array.from(speakers).map(id => ({ speaker_id: id })),
        turns: turns,
        model: { backend: "import", model_id: file.name },
      };

      renderDiarizationWorkspace(diarResult, audioId, true);
      showToast(`Imported ${turns.length} turns across ${speakers.size} speakers!`, 'success');
    } catch (err) {
      showToast(`Failed to parse annotation: ${err.message}`, 'error');
    }
  };
  reader.readAsText(file);
}

// Extract Speaker Audio Handlers
async function extractSpeakerAudio(speakerId, speakerName) {
  const audioId = state.diarization.audioId || el.diarInputSelect.value;
  if (!audioId) {
    showToast("No active audio track selected", "error");
    return;
  }

  const turns = state.diarization.turns.filter(t => t.speaker_id === speakerId);
  if (turns.length === 0) {
    showToast("No turns found for speaker extraction", "error");
    return;
  }

  const mode = el.diarExtractModeSelect ? el.diarExtractModeSelect.value : "concatenated";
  showToast(`Extracting audio for ${speakerName || speakerId} (${mode})...`, "info");

  try {
    const res = await fetch("/api/diarization/extract-speaker", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        audio_id: audioId,
        speaker_id: speakerId,
        speaker_name: speakerName || speakerId,
        mode: mode,
        turns: turns,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Failed to extract speaker audio");

    await fetchAudioList();
    showToast(`Speaker track extracted: "${data.metadata?.title || speakerId}" (${data.duration_s?.toFixed(2)}s)`, "success");
  } catch (err) {
    showToast(`Extraction failed: ${err.message}`, "error");
  }
}

async function extractAllSpeakers() {
  const audioId = state.diarization.audioId || el.diarInputSelect.value;
  if (!audioId) {
    showToast("No active audio track selected", "error");
    return;
  }

  const mode = el.diarExtractModeSelect ? el.diarExtractModeSelect.value : "concatenated";
  showToast(`Extracting all speaker stems (${mode})...`, "info");

  try {
    const res = await fetch("/api/diarization/extract-all-speakers", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        audio_id: audioId,
        mode: mode,
        turns: state.diarization.turns,
        speaker_names: state.diarization.customNames,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Failed to extract all speakers");

    await fetchAudioList();
    showToast(`Successfully extracted ${data.total_speakers} speaker stems into workspace!`, "success");
  } catch (err) {
    showToast(`Extraction failed: ${err.message}`, "error");
  }
}

function generateDemoDiarization(audioId) {
  const audioItem = state.audioList.find(a => a.id === audioId) || state.activeAudio;
  const dur = (audioItem ? audioItem.duration_s : 0) || 60;

  const demoTurns = [
    { speaker_id: "spk_00", start_s: 0.5, end_s: Math.min(dur * 0.15, 6.2), confidence: 0.95 },
    { speaker_id: "spk_01", start_s: Math.min(dur * 0.16, 6.5), end_s: Math.min(dur * 0.35, 14.8), confidence: 0.92 },
    { speaker_id: "spk_00", start_s: Math.min(dur * 0.36, 15.2), end_s: Math.min(dur * 0.55, 23.5), confidence: 0.96 },
    { speaker_id: "spk_02", start_s: Math.min(dur * 0.52, 22.0), end_s: Math.min(dur * 0.72, 31.0), confidence: 0.89 },
    { speaker_id: "spk_01", start_s: Math.min(dur * 0.73, 31.5), end_s: Math.min(dur * 0.88, 38.0), confidence: 0.94 },
    { speaker_id: "spk_00", start_s: Math.min(dur * 0.89, 38.5), end_s: Math.min(dur * 0.98, 42.0), confidence: 0.97 },
  ];

  state.diarization.customNames = {
    "spk_00": "Host / Interviewer",
    "spk_01": "Guest 1 (Alice)",
    "spk_02": "Guest 2 (Bob)",
  };

  const demoData = {
    schema_version: "1.0",
    audio_id: audioId,
    speakers: [
      { speaker_id: "spk_00" },
      { speaker_id: "spk_01" },
      { speaker_id: "spk_02" },
    ],
    turns: demoTurns,
    model: { backend: "pyannote", model_id: "pyannote/speaker-diarization-community-1" },
  };

  renderDiarizationWorkspace(demoData, audioId, true);
}

// ==================== DIARIZATION HISTORY & SESSION MANAGER ====================

function saveDiarizationToHistory(diarization, audioId) {
  if (!diarization || !diarization.turns || diarization.turns.length === 0) return;

  const audioItem = state.audioList.find(a => a.id === audioId) || state.activeAudio;
  const totalDur = state.diarization.duration || 60;
  const totalSpeechS = state.diarization.turns.reduce((acc, t) => acc + Math.max(0, t.end_s - t.start_s), 0);
  const speechRatioPct = ((totalSpeechS / totalDur) * 100).toFixed(1);

  const historyItem = {
    id: `diar_hist_${Date.now()}_${Math.random().toString(36).substring(2, 6)}`,
    timestamp: Date.now(),
    audio_id: audioId,
    audio_title: audioItem?.title || audioItem?.source_id || audioId,
    duration_s: totalDur,
    model_backend: diarization.model?.backend || diarization.model?.model_id || 'Pyannote',
    speaker_count: state.diarization.speakers.length,
    turn_count: state.diarization.turns.length,
    speech_ratio_pct: speechRatioPct,
    total_speech_s: totalSpeechS,
    speakers: JSON.parse(JSON.stringify(state.diarization.speakers)),
    turns: JSON.parse(JSON.stringify(state.diarization.turns)),
    customNames: { ...state.diarization.customNames },
    colors: { ...state.diarization.colors },
  };

  if (!state.diarization.history) state.diarization.history = [];
  // Avoid duplicate immediately identical entries
  const existingIdx = state.diarization.history.findIndex(h => h.audio_id === audioId && h.turn_count === historyItem.turn_count);
  if (existingIdx !== -1) {
    state.diarization.history.splice(existingIdx, 1);
  }

  state.diarization.history.unshift(historyItem);
  if (state.diarization.history.length > 30) {
    state.diarization.history = state.diarization.history.slice(0, 30);
  }

  try {
    localStorage.setItem('sonic_diarization_history', JSON.stringify(state.diarization.history));
  } catch (err) {
    console.warn("Failed to persist diarization history in localStorage:", err);
  }

  renderDiarizationHistory();
}

function loadDiarizationHistory() {
  try {
    const raw = localStorage.getItem('sonic_diarization_history');
    if (raw) {
      state.diarization.history = JSON.parse(raw);
    }
  } catch (err) {
    console.warn("Could not load diarization history from localStorage:", err);
    state.diarization.history = [];
  }
  renderDiarizationHistory();
}

function renderDiarizationHistory() {
  const container = el.diarHistoryList;
  if (!container) return;
  container.innerHTML = "";

  let items = state.diarization.history || [];

  // Filter if search query active
  if (state.diarization.historySearch) {
    const q = state.diarization.historySearch;
    items = items.filter(item => {
      const title = (item.audio_title || '').toLowerCase();
      const model = (item.model_backend || '').toLowerCase();
      const spkNames = Object.values(item.customNames || {}).join(' ').toLowerCase();
      const dateStr = new Date(item.timestamp).toLocaleString().toLowerCase();
      return title.includes(q) || model.includes(q) || spkNames.includes(q) || dateStr.includes(q);
    });
  }

  if (el.diarHistoryCountBadge) {
    el.diarHistoryCountBadge.textContent = `${items.length} saved run${items.length === 1 ? '' : 's'}`;
  }

  if (items.length === 0) {
    container.innerHTML = `
      <div class="empty-placeholder">
        <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>
        <span>${state.diarization.historySearch ? 'No saved sessions match your search query.' : 'No diarization sessions saved yet. Run diarization or import annotations above.'}</span>
      </div>
    `;
    return;
  }

  items.forEach(item => {
    const isCurrentActive = state.diarization.audioId === item.audio_id && state.diarization.turns.length === item.turn_count;
    const dateStr = new Date(item.timestamp).toLocaleString();
    const timeAgo = formatTimeAgo(item.timestamp);
    const modelTag = String(item.model_backend).toLowerCase().includes("sortformer") ? "NeMo Sortformer" : (String(item.model_backend).includes("community") ? "Pyannote Community-1" : (item.model_backend || "Pyannote Community-1"));

    const card = document.createElement("div");
    card.className = `diar-history-item ${isCurrentActive ? 'active-session' : ''}`;
    card.dataset.historyId = item.id;

    card.innerHTML = `
      <div class="diar-hist-top">
        <div class="diar-hist-title-wrap">
          <span class="badge ${isCurrentActive ? 'badge-accent' : 'badge-info'}">${escapeHtml(modelTag)}</span>
          <span class="diar-hist-title" title="${escapeHtml(item.audio_title)}">${escapeHtml(item.audio_title)}</span>
          ${isCurrentActive ? '<span class="badge badge-success badge-sm">Active Session</span>' : ''}
        </div>
        <span class="diar-hist-time" title="${dateStr}">🕒 ${timeAgo}</span>
      </div>

      <div class="diar-hist-meta-row">
        <span><strong>${item.speaker_count}</strong> Speakers</span>
        <span>•</span>
        <span><strong>${item.turn_count}</strong> Turns</span>
        <span>•</span>
        <span><strong>${item.speech_ratio_pct}%</strong> Speech (${(item.total_speech_s || 0).toFixed(1)}s)</span>
        <span>•</span>
        <span>Duration: <strong>${formatTime(item.duration_s || 0)}</strong></span>
      </div>

      <div class="diar-hist-spk-chips">
        ${(item.speakers || []).map(s => {
          const name = (item.customNames && item.customNames[s.speaker_id]) || s.speaker_id;
          const color = (item.colors && item.colors[s.speaker_id]) || getSpeakerColor(s.speaker_id);
          return `
            <span class="diar-hist-spk-chip" style="border-left: 3px solid ${color};">
              <span style="color:${color}; font-weight:700;">●</span>
              <span>${escapeHtml(name)}</span>
            </span>
          `;
        }).join('')}
      </div>

      <div class="diar-hist-actions">
        <button class="btn btn-xs btn-primary btn-load-hist" data-id="${item.id}" title="Restore this diarization into the interactive workspace">⚡ Load Session</button>
        <button class="btn btn-xs btn-secondary btn-export-rttm-hist" data-id="${item.id}" title="Download NIST RTTM file">RTTM</button>
        <button class="btn btn-xs btn-secondary btn-export-json-hist" data-id="${item.id}" title="Download Pipeline JSON file">JSON</button>
        <button class="btn btn-xs btn-ghost btn-delete-hist text-destructive" data-id="${item.id}" title="Delete this history entry">🗑</button>
      </div>
    `;

    // Event Listeners
    card.querySelector('.btn-load-hist').addEventListener('click', () => loadHistorySession(item.id));
    card.querySelector('.btn-export-rttm-hist').addEventListener('click', () => downloadHistoryExport(item, 'rttm'));
    card.querySelector('.btn-export-json-hist').addEventListener('click', () => downloadHistoryExport(item, 'json'));
    card.querySelector('.btn-delete-hist').addEventListener('click', () => deleteHistoryItem(item.id));

    container.appendChild(card);
  });
}

function loadHistorySession(historyId) {
  const item = state.diarization.history?.find(h => h.id === historyId);
  if (!item) {
    showToast("Session not found in history", "error");
    return;
  }

  // Restore custom names & colors
  if (item.customNames) state.diarization.customNames = { ...item.customNames };
  if (item.colors) state.diarization.colors = { ...item.colors };

  const restoredData = {
    schema_version: "1.0",
    audio_id: item.audio_id,
    speakers: item.speakers || [],
    turns: item.turns || [],
    model: { backend: item.model_backend, model_id: item.model_id },
  };

  // Sync Input Select Dropdown
  if (el.diarInputSelect) {
    el.diarInputSelect.value = item.audio_id;
    updateDiarInputMeta(item.audio_id);
    renderDiarizationChildren(item.audio_id);
  }

  // Render without pushing duplicate to history
  renderDiarizationWorkspace(restoredData, item.audio_id, false);

  // Load into master audio player
  loadAudioIntoPlayer(item.audio_id);

  // Scroll to workspace
  if (el.diarResultsWrapper) {
    el.diarResultsWrapper.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  showToast(`Restored diarization session for "${item.audio_title}" (${item.turn_count} turns)!`, "success");
}

function deleteHistoryItem(historyId) {
  state.diarization.history = (state.diarization.history || []).filter(h => h.id !== historyId);
  try {
    localStorage.setItem('sonic_diarization_history', JSON.stringify(state.diarization.history));
  } catch (_) {}
  renderDiarizationHistory();
  showToast("History entry deleted", "info");
}

function clearDiarizationHistory() {
  if (!state.diarization.history || state.diarization.history.length === 0) return;
  state.diarization.history = [];
  try {
    localStorage.removeItem('sonic_diarization_history');
  } catch (_) {}
  renderDiarizationHistory();
  showToast("Diarization history cleared", "info");
}

function downloadHistoryExport(item, format) {
  const turns = item.turns || [];
  const audioId = item.audio_id || "audio";
  let content = "";
  let filename = `diarization_${audioId}.${format}`;

  if (format === 'rttm') {
    filename = `diarization_${audioId}.rttm`;
    content = turns.map(t => {
      const dur = (t.end_s - t.start_s).toFixed(3);
      const spkName = (item.customNames && item.customNames[t.speaker_id]) || t.speaker_id;
      return `SPEAKER ${audioId} 1 ${t.start_s.toFixed(3)} ${dur} <NA> <NA> ${spkName} <NA> <NA>`;
    }).join('\n');
  } else {
    filename = `diarization_${audioId}.json`;
    const jsonOutput = {
      schema_version: "1.0",
      audio_id: audioId,
      model: { backend: item.model_backend, model_id: item.model_id },
      speaker_count: item.speaker_count,
      speakers: item.speakers || [],
      turns: turns.map(t => ({
        speaker_id: t.speaker_id,
        speaker_name: (item.customNames && item.customNames[t.speaker_id]) || t.speaker_id,
        start_s: roundNum(t.start_s, 3),
        end_s: roundNum(t.end_s, 3),
        duration_s: roundNum(t.end_s - t.start_s, 3),
        has_overlap: Boolean(t.has_overlap),
      })),
    };
    content = JSON.stringify(jsonOutput, null, 2);
  }

  const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  showToast(`Downloaded ${filename}`, 'success');
}

function exportAllDiarizationHistory() {
  const history = state.diarization.history || [];
  if (history.length === 0) {
    showToast("No history entries to export", "info");
    return;
  }
  const content = JSON.stringify({
    schema_version: "1.0",
    exported_at: new Date().toISOString(),
    total_sessions: history.length,
    sessions: history,
  }, null, 2);

  const filename = `sonic_diarization_history_${new Date().toISOString().slice(0, 10)}.json`;
  const blob = new Blob([content], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  showToast(`Exported all history to ${filename}`, 'success');
}

function formatTimeAgo(timestamp) {
  const sec = Math.floor((Date.now() - timestamp) / 1000);
  if (sec < 10) return "Just now";
  if (sec < 60) return `${sec}s ago`;
  const min = Math.floor(sec / 60);
  if (min < 60) return `${min}m ago`;
  const hr = Math.floor(min / 60);
  if (hr < 24) return `${hr}h ago`;
  const days = Math.floor(hr / 24);
  return `${days}d ago`;
}

// ==================== CUTS MANAGER ====================

function initCutsManager() {
  renderCutsTable();
}

function addCutToRegistry(audioId, start, end, unit = state.cutUnit) {
  const audioItem = state.audioList.find(a => a.id === audioId) || state.activeAudio;
  const sourceDuration = audioItem?.duration_s || state.diarization?.duration || 0;
  const startSeconds = cutValueToSeconds(start, unit, sourceDuration);
  const endSeconds = sourceDuration > 0 ? Math.min(cutValueToSeconds(end, unit, sourceDuration), sourceDuration) : cutValueToSeconds(end, unit, sourceDuration);
  const audio = audioItem || {
    id: audioId,
    title: `${audioId}_cut_${start}_${end}`,
    duration_s: Math.max(0, endSeconds - startSeconds),
  };
  if (!state.cuts) state.cuts = [];
  state.cuts.unshift({
    id: audioId,
    title: audio.title || audioId,
    parentId: audioItem ? audioItem.id : null,
    start: startSeconds,
    end: endSeconds,
    duration: Math.max(0, endSeconds - startSeconds),
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
    el.cutsTableBody.innerHTML = `<tr><td colspan="5" class="empty-table-msg">No clips yet. Select a waveform range, confirm the boundaries, then choose where to open the new clip.</td></tr>`;
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
  auditionAudio.playbackRate = state.player.playbackRate;

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

  if (el.btnAuditionSkipBack) {
    el.btnAuditionSkipBack.addEventListener('click', () => seekRelative(-5));
  }
  if (el.btnAuditionStart) {
    el.btnAuditionStart.addEventListener('click', () => seekTo(0));
  }
  if (el.btnAuditionSkipFwd) {
    el.btnAuditionSkipFwd.addEventListener('click', () => seekRelative(5));
  }
  if (el.auditionSpeedSelect) {
    el.auditionSpeedSelect.addEventListener('change', (e) => setPlaybackRate(e.target.value));
  }

  if (el.btnAuditionLoop) {
    el.btnAuditionLoop.addEventListener('click', togglePlaybackLoop);
  }

  if (el.auditionVolumeSlider) {
    el.auditionVolumeSlider.addEventListener('input', (e) => {
      state.player.volume = parseFloat(e.target.value);
      el.audio.volume = state.player.volume;
      auditionAudio.volume = state.player.volume;
      syncVolumeControls(state.player.volume);
    });
  }

  if (el.auditionScrubber) {
    el.auditionScrubber.addEventListener('input', (e) => {
      const pct = parseFloat(e.target.value);
      if (getPlaybackDuration(auditionAudio)) {
        seekTo((pct / 100) * getPlaybackDuration(auditionAudio));
      }
    });
  }

  auditionAudio.addEventListener('timeupdate', () => {
    updateAuditionTimeDisplays();
  });

  auditionAudio.addEventListener('loadedmetadata', () => {
    state.player.duration = auditionAudio.duration || getPlaybackDuration(auditionAudio);
    updateAuditionTimeDisplays();
    syncActivePlaybackControls();
  });

  auditionAudio.addEventListener('play', () => {
    if (isAuditionPlaybackActive()) setPlayingUI(true);
  });

  auditionAudio.addEventListener('pause', () => {
    if (isAuditionPlaybackActive()) setPlayingUI(false);
  });

  auditionAudio.addEventListener('ended', () => {
    if (isAuditionPlaybackActive()) {
      setPlayingUI(false);
      seekTo(0);
    }
  });

  syncActivePlaybackControls();

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

function updateAuditionTimeDisplays() {
  const duration = getPlaybackDuration(auditionAudio);
  const currentTime = Math.min(auditionAudio.currentTime || 0, duration || Infinity);

  if (el.auditionTimeCurrent) {
    el.auditionTimeCurrent.textContent = formatTimePrecise(currentTime);
  }
  if (el.auditionTimeTotal) {
    el.auditionTimeTotal.textContent = formatTimePrecise(duration);
  }
  if (el.auditionScrubber && duration) {
    el.auditionScrubber.value = (currentTime / duration) * 100;
  }

  if (isAuditionPlaybackActive()) {
    state.player.duration = duration;
    state.player.currentTime = currentTime;
    if (el.timeCurrent) el.timeCurrent.textContent = formatTime(currentTime);
    if (el.timeTotal) {
      el.timeTotal.textContent = state.player.showRemainingTime
        ? `-${formatTime(Math.max(0, duration - currentTime))}`
        : formatTime(duration);
    }
    const pct = duration ? Math.min(100, Math.max(0, (currentTime / duration) * 100)) : 0;
    if (el.scrubProgress) el.scrubProgress.style.width = `${pct}%`;
  }
}

function toggleAuditionPlay() {
  if (auditionAudio.paused) {
    if (!auditionAudio.src || auditionAudio.src === window.location.href) {
      if (auditionTracks.length > 0) {
        switchAuditionTrack(activeAuditionIndex, true);
      }
      return;
    }
    el.audio.pause();
    syncActivePlaybackControls();
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

function switchAuditionTrack(idx, autoplay = false) {
  if (idx < 0 || idx >= auditionTracks.length) return;
  activeAuditionIndex = idx;
  const track = auditionTracks[idx];

  const wasPlaying = !auditionAudio.paused || autoplay;
  const curTime = auditionAudio.currentTime;

  auditionAudio.pause();
  auditionAudio.playbackRate = state.player.playbackRate;
  auditionAudio.volume = state.player.volume;
  auditionAudio.src = `/api/audio/${track.id}/stream`;

  const restorePositionAndPlay = () => {
    const duration = getPlaybackDuration(auditionAudio);
    auditionAudio.currentTime = Math.max(0, Math.min(curTime || 0, duration || curTime || 0));
    updateAuditionTimeDisplays();
    if (wasPlaying) {
      el.audio.pause();
      auditionAudio.play().catch(e => console.error("Playback switch error:", e));
    }
  };
  if (auditionAudio.readyState >= 1) {
    restorePositionAndPlay();
  } else {
    auditionAudio.addEventListener('loadedmetadata', restorePositionAndPlay, { once: true });
    auditionAudio.load();
  }

  if (el.activeAuditionTrackName) {
    el.activeAuditionTrackName.textContent = track.label;
  }
  if (isAuditionPlaybackActive()) {
    if (el.playerTitle) el.playerTitle.textContent = track.label;
    if (el.playerSub) el.playerSub.textContent = `Audition track • ${track.stem || 'reference mix'} • ${track.id}`;
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

  syncActivePlaybackControls();
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
    const targetDevice = state.selectedGpu || (el.sepDeviceSelect ? el.sepDeviceSelect.value : 'auto');
    const res = await fetch("/api/separation/batch-compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        audio_id: audioId,
        models: models,
        device: targetDevice,
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

function getFileModelBadge(file) {
  const cat = (file.category || "").toLowerCase();
  const path = (file.path || "").toLowerCase();
  const name = (file.name || "").toLowerCase();

  if (cat.includes("speech") || path.includes("speech")) {
    return { label: "Speech Source", class: "badge-success" };
  }
  if (cat.includes("music") || path.includes("music")) {
    return { label: "Music BGM", class: "badge-accent" };
  }
  if (cat.includes("cuts") || path.includes("cut") || name.includes("_cut_")) {
    return { label: "Audio Cut", class: "badge-warning" };
  }
  if (path.includes("bs_roformer") || name.includes("bs_roformer")) {
    return { label: "BS-RoFormer", class: "badge-primary" };
  }
  if (path.includes("mel_roformer") || name.includes("mel_roformer")) {
    return { label: "Mel-RoFormer", class: "badge-primary" };
  }
  if (path.includes("htdemucs") || name.includes("demucs")) {
    return { label: "HTDemucs", class: "badge-primary" };
  }
  if (path.includes("mvsep") || name.includes("mdx")) {
    return { label: "MVSep MDX", class: "badge-primary" };
  }
  if (cat.includes("stem") || cat.includes("separated")) {
    return { label: "Separated Stem", class: "badge-primary" };
  }
  if (cat.includes("youtube") || path.includes("yt_crawler") || path.includes("download")) {
    return { label: "YouTube Ingest", class: "badge-secondary" };
  }
  if (cat.includes("pipeline") || path.includes("pipeline")) {
    return { label: "Pipeline Asset", class: "badge-info" };
  }
  if (cat.includes("temp") || path.includes("temp") || name.includes("quick_save")) {
    return { label: "Quick Save", class: "badge-ghost" };
  }
  if (cat.includes("upload") || path.includes("upload")) {
    return { label: "Upload", class: "badge-secondary" };
  }
  return { label: (file.format || "WAV").toUpperCase(), class: "badge-ghost" };
}

let previewAudioEl = null;
let currentPreviewingPath = null;

function toggleFilePreview(filePath, btn) {
  if (!previewAudioEl) {
    previewAudioEl = new Audio();
    previewAudioEl.addEventListener('ended', () => {
      currentPreviewingPath = null;
      document.querySelectorAll('.btn-preview-file').forEach(b => {
        b.classList.remove('playing');
        b.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>';
      });
    });
    previewAudioEl.addEventListener('pause', () => {
      if (previewAudioEl.ended || previewAudioEl.paused) {
        document.querySelectorAll('.btn-preview-file').forEach(b => {
          b.classList.remove('playing');
          b.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>';
        });
      }
    });
  }

  if (currentPreviewingPath === filePath && !previewAudioEl.paused) {
    previewAudioEl.pause();
    currentPreviewingPath = null;
    if (btn) {
      btn.classList.remove('playing');
      btn.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>';
    }
  } else {
    // Pause main workspace player if playing
    if (el.audio && !el.audio.paused) {
      el.audio.pause();
      state.isPlaying = false;
      updatePlayPauseButton();
    }

    previewAudioEl.pause();
    currentPreviewingPath = filePath;
    previewAudioEl.src = `/api/library/stream?path=${encodeURIComponent(filePath)}`;
    previewAudioEl.play().catch(e => {
      console.warn("Audio preview failed:", e);
      showToast("Audio preview error", "error");
    });

    document.querySelectorAll('.btn-preview-file').forEach(b => {
      b.classList.remove('playing');
      b.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>';
    });

    if (btn) {
      btn.classList.add('playing');
      btn.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect></svg>';
    }
  }
}

async function fetchAudioList() {
  try {
    const res = await fetch("/api/audio");
    const data = await res.json();
    state.audioList = data.audios || [];
    if (el.sessionCountBadge) el.sessionCountBadge.textContent = `${state.audioList.length} items`;
    renderSessionHistory();
    populateAllAudioSelects();
  } catch (err) {
    console.error("Failed to fetch audio list:", err);
  }
}

function renderSessionHistory() {
  const container = el.sessionHistoryList;
  if (!container) return;
  container.innerHTML = "";

  if (state.audioList.length === 0) {
    container.innerHTML = `<div class="empty-placeholder">No active audio in this session.</div>`;
    return;
  }

  state.audioList.forEach(item => {
    const card = document.createElement("div");
    card.className = "file-item-card";
    card.innerHTML = `
      <div class="file-left-group">
        <div class="file-details">
          <div class="file-title-row">
            <span class="badge badge-accent" style="font-size: 0.72rem; font-weight: 700;">${item.source_type.toUpperCase()}</span>
            <span class="file-name" title="${escapeHtml(item.title)}">${escapeHtml(item.title)}</span>
          </div>
          <div class="file-meta-row">
            <span class="file-path">${escapeHtml(item.path)}</span>
            <span class="meta-chip">${(item.duration_s || 0).toFixed(1)}s</span>
            <span class="meta-chip">${item.sample_rate.toLocaleString()} Hz</span>
            <span class="meta-chip">${item.channels === 1 ? 'Mono' : 'Stereo'}</span>
            <span class="meta-chip">${item.format.toUpperCase()}</span>
          </div>
        </div>
      </div>
      <div class="file-actions">
        <button class="btn btn-sm btn-primary btn-load-session" data-id="${item.id}" title="Load into Studio Workspace & Player">Play</button>
        <div class="dropdown-actions-wrap" style="position: relative; display: inline-block;">
          <button class="btn btn-sm btn-secondary btn-more-session" title="More routing actions">
            <span>⋯</span>
          </button>
          <div class="actions-popup-menu hidden" style="position: absolute; right: 0; top: 100%; margin-top: 4px; z-index: 50; background: var(--bg-surface-elevated); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); box-shadow: 0 8px 24px rgba(0,0,0,0.5); padding: 4px; min-width: 170px; display: flex; flex-direction: column; gap: 2px;">
            <button class="menu-item-btn btn-session-cutter" style="text-align: left; padding: 6px 10px; font-size: 0.78rem; background: none; border: none; color: var(--text-primary); cursor: pointer; border-radius: 4px; display: flex; align-items: center; gap: 6px;">✂️ Open in Cutter</button>
            <button class="menu-item-btn btn-session-sep" style="text-align: left; padding: 6px 10px; font-size: 0.78rem; background: none; border: none; color: var(--text-primary); cursor: pointer; border-radius: 4px; display: flex; align-items: center; gap: 6px;">🎛️ Send to Separation</button>
            <button class="menu-item-btn btn-session-diar" style="text-align: left; padding: 6px 10px; font-size: 0.78rem; background: none; border: none; color: var(--text-primary); cursor: pointer; border-radius: 4px; display: flex; align-items: center; gap: 6px;">👥 Send to Diarization</button>
            <button class="menu-item-btn btn-session-speech" style="text-align: left; padding: 6px 10px; font-size: 0.78rem; background: none; border: none; color: var(--text-primary); cursor: pointer; border-radius: 4px; display: flex; align-items: center; gap: 6px;">🗣️ Set as Speech (Mixer)</button>
            <button class="menu-item-btn btn-session-music" style="text-align: left; padding: 6px 10px; font-size: 0.78rem; background: none; border: none; color: var(--text-primary); cursor: pointer; border-radius: 4px; display: flex; align-items: center; gap: 6px;">🎵 Set as Music (Mixer)</button>
            <a href="/api/audio/${item.id}/stream" download="${escapeHtml(item.title)}.${item.format}" class="menu-item-btn" style="text-align: left; padding: 6px 10px; font-size: 0.78rem; background: none; border: none; color: var(--accent-primary-hover); text-decoration: none; cursor: pointer; border-radius: 4px; display: flex; align-items: center; gap: 6px;">⬇️ Download Audio</a>
          </div>
        </div>
        <button class="btn btn-sm btn-ghost btn-delete-session" data-id="${item.id}" title="Remove from active session">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path><line x1="10" y1="11" x2="10" y2="17"></line><line x1="14" y1="11" x2="14" y2="17"></line></svg>
        </button>
      </div>
    `;

    card.querySelector('.btn-load-session').addEventListener('click', () => {
      switchTab('tab-workspace');
      setActiveAudio(item.id, { play: true });
    });

    const btnMore = card.querySelector('.btn-more-session');
    const popupMenu = card.querySelector('.actions-popup-menu');
    if (btnMore && popupMenu) {
      btnMore.addEventListener('click', (e) => {
        e.stopPropagation();
        document.querySelectorAll('.actions-popup-menu').forEach(m => {
          if (m !== popupMenu) m.classList.add('hidden');
        });
        popupMenu.classList.toggle('hidden');
      });
    }

    card.querySelector('.btn-session-cutter')?.addEventListener('click', () => {
      popupMenu.classList.add('hidden');
      switchTab('tab-cutter');
      setActiveAudio(item.id, { play: false });
    });
    card.querySelector('.btn-session-sep')?.addEventListener('click', () => {
      popupMenu.classList.add('hidden');
      switchTab('tab-separation');
      if (el.sepInputSelect) el.sepInputSelect.value = item.id;
      showToast(`Selected "${item.title}" for Separation!`, "success");
    });
    card.querySelector('.btn-session-diar')?.addEventListener('click', () => {
      popupMenu.classList.add('hidden');
      switchTab('tab-diarization');
      if (el.diarInputSelect) {
        el.diarInputSelect.value = item.id;
        el.diarInputSelect.dispatchEvent(new Event('change'));
      }
      showToast(`Selected "${item.title}" for Diarization!`, "success");
    });
    card.querySelector('.btn-session-speech')?.addEventListener('click', () => {
      popupMenu.classList.add('hidden');
      switchTab('tab-bench-mixer');
      if (el.mixSpeechSelect) el.mixSpeechSelect.value = item.id;
      showToast(`Set "${item.title}" as Speech source!`, "success");
    });
    card.querySelector('.btn-session-music')?.addEventListener('click', () => {
      popupMenu.classList.add('hidden');
      switchTab('tab-bench-mixer');
      if (el.mixMusicSelect) el.mixMusicSelect.value = item.id;
      showToast(`Set "${item.title}" as Music source!`, "success");
    });

    card.querySelector('.btn-delete-session').addEventListener('click', async () => {
      if (confirm(`Remove "${item.title}" from active session?`)) {
        try {
          const res = await fetch(`/api/audio/${item.id}`, { method: 'DELETE' });
          if (!res.ok) throw new Error("Failed to delete audio from session");
          showToast(`Removed "${item.title}" from session`, "info");
          await fetchAudioList();
          if (state.activeAudio && state.activeAudio.id === item.id) {
            state.activeAudio = null;
            el.activeSection.classList.add('hidden');
          }
        } catch (err) {
          showToast(err.message, "error");
        }
      }
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
    renderLibraryModalItems();
  } catch (err) {
    console.error("Failed to fetch library:", err);
  }
}

function filterServerFiles(files, query, category) {
  const q = (query || "").toLowerCase().trim();
  const cat = (category || "all").toLowerCase().trim();

  return (files || []).filter(file => {
    const fileCat = (file.category || "").toLowerCase();
    const filePath = (file.path || "").toLowerCase();
    const fileName = (file.name || "").toLowerCase();
    const fileTitle = (file.title || "").toLowerCase();

    let matchesCat = cat === "all";
    if (!matchesCat) {
      if (cat === "speech") matchesCat = fileCat.includes("speech") || filePath.includes("speech");
      else if (cat === "music") matchesCat = fileCat.includes("music") || filePath.includes("music");
      else if (cat === "separated" || cat === "stems") matchesCat = fileCat.includes("separated") || fileCat.includes("stem") || filePath.includes("demucs") || filePath.includes("roformer") || filePath.includes("mvsep") || filePath.includes("stems");
      else if (cat === "downloads") matchesCat = fileCat.includes("download") || filePath.includes("yt_crawler") || filePath.includes("downloads");
      else if (cat === "cuts") matchesCat = fileCat.includes("cuts") || filePath.includes("cuts") || fileName.includes("_cut_");
      else if (cat === "temp") matchesCat = fileCat.includes("temp") || filePath.includes("temp") || fileName.includes("quick_save");
      else matchesCat = fileCat.includes(cat) || filePath.includes(cat);
    }

    const matchesQ = !q ||
      fileName.includes(q) ||
      fileTitle.includes(q) ||
      filePath.includes(q) ||
      fileCat.includes(q);

    return matchesCat && matchesQ;
  });
}

function buildFileItemCard(file, { isModal = false } = {}) {
  const badgeInfo = getFileModelBadge(file);
  const isPlaying = currentPreviewingPath === file.path && previewAudioEl && !previewAudioEl.paused;
  const durStr = (file.duration_s || 0) > 0 ? `${(file.duration_s).toFixed(1)}s` : '';
  const srStr = file.sample_rate ? `${(file.sample_rate / 1000).toFixed(1)} kHz` : '';
  const chStr = file.channels === 1 ? 'Mono' : (file.channels === 2 ? 'Stereo' : '');
  const fmtStr = (file.format || 'wav').toUpperCase();

  const card = document.createElement("div");
  card.className = "file-item-card";
  card.innerHTML = `
    <div class="file-left-group">
      <button class="btn-preview-file ${isPlaying ? 'playing' : ''}" data-path="${escapeHtml(file.path)}" title="${isPlaying ? 'Pause preview' : 'Play & Audition preview'}" aria-label="Preview track">
        ${isPlaying
          ? '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect></svg>'
          : '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>'
        }
      </button>
      <div class="file-details">
        <div class="file-title-row">
          <span class="badge ${badgeInfo.class}" style="font-size: 0.72rem; font-weight: 700;">${escapeHtml(badgeInfo.label)}</span>
          <span class="file-name" title="${escapeHtml(file.name)}">${escapeHtml(file.title || file.name)}</span>
        </div>
        <div class="file-meta-row">
          <span class="file-path" title="${escapeHtml(file.path)}">${escapeHtml(file.path)}</span>
          ${durStr ? `<span class="meta-chip">${durStr}</span>` : ''}
          ${srStr ? `<span class="meta-chip">${srStr}</span>` : ''}
          ${chStr ? `<span class="meta-chip">${chStr}</span>` : ''}
          <span class="meta-chip">${fmtStr}</span>
          <span class="meta-chip">${formatBytes(file.size || 0)}</span>
        </div>
      </div>
    </div>
    <div class="file-actions">
      <button class="btn btn-sm btn-primary btn-load-target" data-path="${escapeHtml(file.path)}" data-target="workspace" title="Load into Studio Workspace">
        <span>Load</span>
      </button>
      <div class="dropdown-actions-wrap" style="position: relative; display: inline-block;">
        <button class="btn btn-sm btn-secondary btn-more-actions" title="More routing actions">
          <span>⋯</span>
        </button>
        <div class="actions-popup-menu hidden" style="position: absolute; right: 0; top: 100%; margin-top: 4px; z-index: 50; background: var(--bg-surface-elevated); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); box-shadow: 0 8px 24px rgba(0,0,0,0.5); padding: 4px; min-width: 170px; display: flex; flex-direction: column; gap: 2px;">
          <button class="menu-item-btn btn-send-cutter" data-path="${escapeHtml(file.path)}" style="text-align: left; padding: 6px 10px; font-size: 0.78rem; background: none; border: none; color: var(--text-primary); cursor: pointer; border-radius: 4px; display: flex; align-items: center; gap: 6px;">✂️ Open in Cutter</button>
          <button class="menu-item-btn btn-send-sep" data-path="${escapeHtml(file.path)}" style="text-align: left; padding: 6px 10px; font-size: 0.78rem; background: none; border: none; color: var(--text-primary); cursor: pointer; border-radius: 4px; display: flex; align-items: center; gap: 6px;">🎛️ Send to Separation</button>
          <button class="menu-item-btn btn-send-diar" data-path="${escapeHtml(file.path)}" style="text-align: left; padding: 6px 10px; font-size: 0.78rem; background: none; border: none; color: var(--text-primary); cursor: pointer; border-radius: 4px; display: flex; align-items: center; gap: 6px;">👥 Send to Diarization</button>
          <button class="menu-item-btn btn-send-speech" data-path="${escapeHtml(file.path)}" style="text-align: left; padding: 6px 10px; font-size: 0.78rem; background: none; border: none; color: var(--text-primary); cursor: pointer; border-radius: 4px; display: flex; align-items: center; gap: 6px;">🗣️ Set as Speech (Mixer)</button>
          <button class="menu-item-btn btn-send-music" data-path="${escapeHtml(file.path)}" style="text-align: left; padding: 6px 10px; font-size: 0.78rem; background: none; border: none; color: var(--text-primary); cursor: pointer; border-radius: 4px; display: flex; align-items: center; gap: 6px;">🎵 Set as Music (Mixer)</button>
          <a href="/api/library/download?path=${encodeURIComponent(file.path)}" download="${escapeHtml(file.name)}" class="menu-item-btn" style="text-align: left; padding: 6px 10px; font-size: 0.78rem; background: none; border: none; color: var(--accent-primary-hover); text-decoration: none; cursor: pointer; border-radius: 4px; display: flex; align-items: center; gap: 6px;">⬇️ Download File</a>
        </div>
      </div>
      <button class="btn btn-sm btn-ghost btn-delete-file text-danger" data-path="${escapeHtml(file.path)}" title="Permanently delete from disk">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path><line x1="10" y1="11" x2="10" y2="17"></line><line x1="14" y1="11" x2="14" y2="17"></line></svg>
      </button>
    </div>
  `;

  // Attach event listeners
  const btnPrev = card.querySelector('.btn-preview-file');
  btnPrev.addEventListener('click', () => toggleFilePreview(file.path, btnPrev));

  card.querySelector('.btn-load-target').addEventListener('click', () => loadLibraryFileTo(file.path, 'workspace'));

  const btnMore = card.querySelector('.btn-more-actions');
  const popupMenu = card.querySelector('.actions-popup-menu');
  if (btnMore && popupMenu) {
    btnMore.addEventListener('click', (e) => {
      e.stopPropagation();
      document.querySelectorAll('.actions-popup-menu').forEach(m => {
        if (m !== popupMenu) m.classList.add('hidden');
      });
      popupMenu.classList.toggle('hidden');
    });
  }

  card.querySelector('.btn-send-cutter')?.addEventListener('click', () => { popupMenu.classList.add('hidden'); loadLibraryFileTo(file.path, 'cutter'); });
  card.querySelector('.btn-send-sep')?.addEventListener('click', () => { popupMenu.classList.add('hidden'); loadLibraryFileTo(file.path, 'separation'); });
  card.querySelector('.btn-send-diar')?.addEventListener('click', () => { popupMenu.classList.add('hidden'); loadLibraryFileTo(file.path, 'diarization'); });
  card.querySelector('.btn-send-speech')?.addEventListener('click', () => { popupMenu.classList.add('hidden'); loadLibraryFileTo(file.path, 'speech'); });
  card.querySelector('.btn-send-music')?.addEventListener('click', () => { popupMenu.classList.add('hidden'); loadLibraryFileTo(file.path, 'music'); });

  card.querySelector('.btn-delete-file').addEventListener('click', () => deleteServerFile(file.path, file.name));

  return card;
}

function renderServerFiles() {
  const container = el.serverFilesList;
  if (!container) return;
  container.innerHTML = "";

  const query = state.tabLibrarySearch || "";
  const category = state.tabLibraryCategory || "all";
  const filtered = filterServerFiles(state.serverFiles, query, category);

  if (filtered.length === 0) {
    container.innerHTML = `<div class="empty-placeholder" style="padding: 2.5rem 1rem; text-align: center;">No project audio files found matching filter.</div>`;
    return;
  }

  filtered.forEach(file => {
    const card = buildFileItemCard(file, { isModal: false });
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
  } catch (err) {
    showToast(`Delete failed: ${err.message}`, "error");
  }
}

async function loadLibraryFileTo(filePath, target = 'workspace') {
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
      closeAllModals();

      if (target === 'workspace') {
        switchTab('tab-workspace');
        await setActiveAudio(data.audio_id, { play: true });
        showToast(`Loaded "${data.metadata?.title || filePath}" into Studio Workspace!`, "success");
      } else if (target === 'cutter') {
        switchTab('tab-cutter');
        await setActiveAudio(data.audio_id, { play: false });
        showToast(`Loaded "${data.metadata?.title || filePath}" into Audio Cutter!`, "success");
      } else if (target === 'separation') {
        switchTab('tab-separation');
        if (el.sepInputSelect) el.sepInputSelect.value = data.audio_id;
        showToast(`Selected "${data.metadata?.title || filePath}" for Separation!`, "success");
      } else if (target === 'diarization') {
        switchTab('tab-diarization');
        if (el.diarInputSelect) el.diarInputSelect.value = data.audio_id;
        showToast(`Selected "${data.metadata?.title || filePath}" for Diarization!`, "success");
      } else if (target === 'speech') {
        switchTab('tab-bench-mixer');
        if (el.mixSpeechSelect) el.mixSpeechSelect.value = data.audio_id;
        showToast(`Set "${data.metadata?.title || filePath}" as Speech source!`, "success");
      } else if (target === 'music') {
        switchTab('tab-bench-mixer');
        if (el.mixMusicSelect) el.mixMusicSelect.value = data.audio_id;
        showToast(`Set "${data.metadata?.title || filePath}" as Music source!`, "success");
      }
    }
  } catch (err) {
    showToast(`Failed to load file: ${err.message}`, "error");
  }
}

async function loadServerFile(filePath) {
  return loadLibraryFileTo(filePath, 'workspace');
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

  const query = state.libraryModalSearch || "";
  const category = state.libraryModalCategory || "all";
  const filtered = filterServerFiles(state.serverFiles, query, category);

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
    const card = buildFileItemCard(file, { isModal: true });
    el.modalLibraryItems.appendChild(card);
  });
}

function toggleShortcutsModal() {
  if (el.modalShortcuts) {
    el.modalShortcuts.classList.toggle('hidden');
  }
}

// ==================== STUDIO TASK QUEUE MODAL LOGIC ====================

function openQueueModal() {
  if (!el.modalTaskQueue) return;
  closeAllModals();
  el.modalTaskQueue.classList.remove('hidden');
  loadAndRenderQueueModal();

  if (state.queuePollingInterval) clearInterval(state.queuePollingInterval);
  state.queuePollingInterval = setInterval(() => {
    if (el.modalTaskQueue && !el.modalTaskQueue.classList.contains('hidden')) {
      loadAndRenderQueueModal();
    } else {
      clearInterval(state.queuePollingInterval);
      state.queuePollingInterval = null;
    }
  }, 1200);
}

function closeQueueModal() {
  if (el.modalTaskQueue) {
    el.modalTaskQueue.classList.add('hidden');
  }
  if (state.queuePollingInterval) {
    clearInterval(state.queuePollingInterval);
    state.queuePollingInterval = null;
  }
}

function toggleQueueModal() {
  if (el.modalTaskQueue && !el.modalTaskQueue.classList.contains('hidden')) {
    closeQueueModal();
  } else {
    openQueueModal();
  }
}

function formatTaskType(type) {
  switch (type) {
    case 'youtube_crawl': return 'YouTube Ingest';
    case 'separation': return 'Stem Separation';
    case 'multi_model_separation': return 'Multi-Model Separation';
    case 'diarization': return 'Speaker Diarization';
    case 'benchmark_mix': return 'Benchmark Mix';
    case 'batch_process': return 'Batch Processor';
    default: return (type || 'Task').replace(/_/g, ' ').toUpperCase();
  }
}

function formatTaskDuration(sec) {
  if (sec == null || isNaN(sec) || sec < 0) return '0s';
  if (sec < 60) return `${sec.toFixed(1)}s`;
  const mins = Math.floor(sec / 60);
  const remSec = Math.floor(sec % 60);
  return `${mins}m ${remSec}s`;
}

function formatTaskTime(task) {
  const now = Date.now() / 1000;
  if (task.status === 'running' && task.start_time) {
    return `Running for ${formatTaskDuration(now - task.start_time)}`;
  }
  if (task.end_time && task.start_time) {
    return `Finished in ${formatTaskDuration(task.end_time - task.start_time)}`;
  }
  if (task.created_at) {
    const elapsed = Math.max(0, now - task.created_at);
    return `Queued ${formatTaskDuration(elapsed)} ago`;
  }
  return '';
}

async function loadAndRenderQueueModal() {
  if (!el.studioQueueTaskList) return;
  try {
    let data;
    try {
      const res = await fetch('/api/queue/shared');
      if (res.ok) {
        data = await res.json();
      }
    } catch (_) {}

    if (!data) {
      const res = await fetch('/api/tasks');
      if (!res.ok) throw new Error("Failed to load tasks");
      const localData = await res.json();
      data = {
        device: { name: (state.systemStatus?.device_name || "Compute Node").split(':')[0], gpu_load_pct: null, vram_used_mb: null, vram_total_mb: null, vram_pct: null },
        summary: {
          total_running: (localData.tasks || []).filter(t => t.status === 'running').length,
          total_queued: (localData.tasks || []).filter(t => t.status === 'pending').length,
          studio_running: (localData.tasks || []).filter(t => t.status === 'running').length,
          studio_queued: (localData.tasks || []).filter(t => t.status === 'pending').length,
          pipeline_running: 0,
          pipeline_queued: 0,
        },
        items: (localData.tasks || []).map(t => ({
          id: t.id,
          source: 'studio',
          source_label: 'SonicStudio',
          title: t.metadata?.title || t.metadata?.model || t.type,
          type: t.type,
          status: t.status,
          progress: t.progress || 0.0,
          message: t.message || "",
          error: t.error,
          created_at: t.created_at,
          start_time: t.start_time,
          end_time: t.end_time,
          queue_position: t.queue_position,
          metadata: t.metadata || {},
        })),
      };
    }

    const items = data.items || [];
    const summary = data.summary || { total_running: 0, total_queued: 0, studio_running: 0, studio_queued: 0, pipeline_running: 0, pipeline_queued: 0 };
    const device = data.device || {};

    // Update GPU Ribbon
    if (el.queueGpuName) el.queueGpuName.textContent = device.name || "Compute Device";
    if (el.queueGpuLoad) el.queueGpuLoad.textContent = Number.isFinite(device.gpu_load_pct) ? `${Math.round(device.gpu_load_pct)}%` : 'Active';
    if (el.queueGpuVram) {
      if (device.vram_used_mb != null && device.vram_total_mb != null) {
        el.queueGpuVram.textContent = `${device.vram_used_mb} / ${device.vram_total_mb} MB (${Math.round(device.vram_pct || 0)}%)`;
      } else {
        el.queueGpuVram.textContent = "Shared Memory";
      }
    }
    if (el.queueGpuPower) {
      const curPowerW = device.power_w ?? data.telemetry?.gpu?.aggregate?.total_power_w ?? data.telemetry?.gpu?.power_w;
      const powerLimitW = device.power_limit_w ?? data.telemetry?.gpu?.aggregate?.total_power_limit_w ?? data.telemetry?.gpu?.power_limit_w;
      el.queueGpuPower.textContent = curPowerW != null
        ? `${curPowerW} / ${powerLimitW ?? '--'} W`
        : '-- / -- W';
    }
    if (el.queueActiveSplit) {
      el.queueActiveSplit.textContent = `Studio: ${summary.studio_running} active • Pipeline: ${summary.pipeline_running} active`;
    }

    // Render multi-GPU cards if 2+ GPUs detected
    if (el.queueGpuDevicesGrid) {
      const devList = device.devices || (data.telemetry && data.telemetry.gpu && data.telemetry.gpu.devices) || [];
      if (devList.length > 1) {
        el.queueGpuDevicesGrid.style.display = 'grid';
        el.queueGpuDevicesGrid.innerHTML = devList.map((d, i) => {
          const dLoad = Number.isFinite(d.load_percent) ? Math.round(d.load_percent) : (Number.isFinite(d.utilization_percent) ? Math.round(d.utilization_percent) : 0);
          const dVram = (d.used_vram_mb != null && d.total_vram_mb != null) ? `${d.used_vram_mb} / ${d.total_vram_mb} MB (${Math.round(d.vram_percent || 0)}%)` : '-- MB';
          const dTemp = d.temperature_c != null ? `${Math.round(d.temperature_c)}°C` : '';
          const dPower = d.power_w != null ? `${d.power_w} / ${d.power_limit_w ?? '--'} W` : '';
          return `
            <div class="gpu-device-card">
              <div class="gpu-device-card-header">
                <span class="gpu-device-card-title">⚡ GPU ${i}: ${escapeHtml(d.name)}</span>
                <span class="gpu-device-card-temp font-mono">${dTemp}</span>
              </div>
              <div class="gpu-device-card-bar-wrap">
                <div class="gpu-device-card-bar-fill" style="width: ${dLoad}%;"></div>
              </div>
              <div class="gpu-device-card-sub font-mono">
                <span>Load: <strong>${dLoad}%</strong></span>
                <span>VRAM: ${dVram}</span>
                ${dPower ? `<span>Power: <strong>${dPower}</strong></span>` : ''}
              </div>
            </div>
          `;
        }).join('');
      } else {
        el.queueGpuDevicesGrid.style.display = 'none';
      }
    }

    if (data.telemetry) {
      updateTelemetryDisplay(data.telemetry);
    }

    // Update stat numbers
    const completedCount = items.filter(t => t.status === 'completed').length;
    const failedCount = items.filter(t => t.status === 'failed' || t.status === 'cancelled').length;

    if (el.queueStatRunning) el.queueStatRunning.textContent = summary.total_running;
    if (el.queueStatQueued) el.queueStatQueued.textContent = summary.total_queued;
    if (el.queueStatCompleted) el.queueStatCompleted.textContent = completedCount;
    if (el.queueStatFailed) el.queueStatFailed.textContent = failedCount;

    if (el.queueModalSubtitle) {
      el.queueModalSubtitle.textContent = `Shared GPU Queue: ${summary.total_running} running, ${summary.total_queued} queued across Studio & Pipeline`;
    }

    // Filter items according to state.queueModalFilter
    const filter = state.queueModalFilter || 'all';
    let filteredItems = items;
    if (filter === 'active') {
      filteredItems = items.filter(t => t.status === 'running' || t.status === 'pending');
    } else if (filter === 'studio') {
      filteredItems = items.filter(t => t.source === 'studio');
    } else if (filter === 'pipeline') {
      filteredItems = items.filter(t => t.source === 'pipeline');
    } else if (filter === 'completed') {
      filteredItems = items.filter(t => t.status === 'completed');
    } else if (filter === 'failed') {
      filteredItems = items.filter(t => t.status === 'failed' || t.status === 'cancelled');
    }

    if (filteredItems.length === 0) {
      el.studioQueueTaskList.innerHTML = `
        <div class="queue-empty-state">
          <div class="queue-empty-icon">☕</div>
          <div class="queue-empty-title">${filter === 'all' ? 'Shared GPU Queue is Idle' : 'No workloads in this view'}</div>
          <div class="queue-empty-sub">${filter === 'all' ? 'All model separations, YouTube downloads, diarizations, and batch jobs across Studio & Pipeline will appear here in real-time.' : 'No workloads match the selected filter.'}</div>
        </div>
      `;
      return;
    }

    el.studioQueueTaskList.innerHTML = '';
    filteredItems.forEach(item => {
      const card = document.createElement('div');
      const isRunning = item.status === 'running';
      const isPending = item.status === 'pending';
      const isFailed = item.status === 'failed';
      const isCancelled = item.status === 'cancelled';
      const isCompleted = item.status === 'completed';
      const isStudio = item.source === 'studio';

      let cardClass = 'queue-task-card';
      if (isRunning) cardClass += ' task-running';
      if (isFailed) cardClass += ' task-failed';
      card.className = cardClass;

      const progressKnown = item.progress_known === true || isCompleted;
      let progressPct = Number(item.progress || 0);
      progressPct = Math.round(Math.min(100, Math.max(0, progressPct)));
      if (isCompleted) progressPct = 100;
      const typeLabel = formatTaskType(item.type);
      const timeStr = formatTaskTime(item);

      let statusBadgeHtml = '';
      if (isRunning) {
        statusBadgeHtml = progressKnown
          ? `<span class="task-status-pill task-status-running"><span class="dot dot-pulse"></span> Running (${progressPct}%)</span>`
          : `<span class="task-status-pill task-status-running"><span class="dot dot-pulse"></span> Running</span>`;
      } else if (isPending) {
        const posText = item.queue_position ? ` #${item.queue_position}` : '';
        statusBadgeHtml = `<span class="task-status-pill task-status-pending">⏳ Queued${posText}</span>`;
      } else if (isCompleted) {
        statusBadgeHtml = `<span class="task-status-pill task-status-completed">✓ Completed</span>`;
      } else if (isFailed) {
        statusBadgeHtml = `<span class="task-status-pill task-status-failed">⚠ Failed</span>`;
      } else if (isCancelled) {
        statusBadgeHtml = `<span class="task-status-pill task-status-cancelled">⊘ Cancelled</span>`;
      }

      const sourceBadgeHtml = isStudio
        ? `<span class="workload-source-badge source-studio">🎙️ Studio</span>`
        : `<span class="workload-source-badge source-pipeline">⚡ Pipeline</span>`;

      let errorHtml = '';
      if (item.error) {
        errorHtml = `<div class="task-card-error">${escapeHtml(item.error)}</div>`;
      }

      let cancelBtnHtml = '';
      if (isRunning) {
        cancelBtnHtml = `<button class="btn btn-sm btn-danger btn-cancel-task" data-item-id="${item.id}" title="Stop running workload">Stop</button>`;
      } else if (isPending) {
        cancelBtnHtml = `<button class="btn btn-sm btn-danger btn-cancel-task" data-item-id="${item.id}" title="Cancel queued workload">Cancel</button>`;
      }

      // Metadata summary
      const meta = item.metadata || {};
      let metaSummary = [];
      if (item.title && item.title !== typeLabel) metaSummary.push(item.title);
      if (meta.model) metaSummary.push(`Model: ${meta.model}`);
      if (meta.backend) metaSummary.push(`Backend: ${meta.backend}`);
      const itemDevice = meta.device || item.params?.device || (item.result && item.result.device);
      const itemPower = meta.power_w ?? (item.result && item.result.power_w);
      if (itemDevice) metaSummary.push(`🖥️ ${itemDevice}`);
      if (itemPower != null) metaSummary.push(`⚡ ${itemPower}W`);
      if (meta.url) metaSummary.push(`URL: ${meta.url.length > 30 ? meta.url.substring(0, 30) + '...' : meta.url}`);
      if (item.total_items && item.total_items > 1) metaSummary.push(`${item.processed_items || 0}/${item.total_items} items`);
      const metaLine = metaSummary.length > 0 ? metaSummary.join(' • ') : '';

      card.innerHTML = `
        <div class="task-card-header">
          <div class="task-card-meta">
            ${sourceBadgeHtml}
            <span class="task-type-badge">${typeLabel}</span>
            <span class="task-id-label">${item.id}</span>
          </div>
          <div class="task-card-status-wrap">
            <span class="task-time-label">${timeStr}</span>
            ${statusBadgeHtml}
          </div>
        </div>

        <div class="task-card-msg">${escapeHtml(item.message || (isRunning ? 'Processing on compute device...' : ''))}</div>
        ${errorHtml}

        ${isRunning || isPending ? `
          <div class="task-card-progress ${isRunning && !progressKnown ? 'indeterminate' : ''}">
            <div class="task-card-progress-fill ${isRunning && !progressKnown ? 'indeterminate' : ''}" style="${isRunning && progressKnown ? `width: ${progressPct}%;` : (isPending ? 'width: 100%; opacity: 0.5;' : '')}"></div>
          </div>
          ${isRunning && !progressKnown ? `<div class="task-progress-unknown">No numeric progress from this backend</div>` : ''}
        ` : ''}

        <div class="task-card-footer">
          <span class="task-details-hint font-mono">${escapeHtml(metaLine)}</span>
          ${cancelBtnHtml}
        </div>
      `;

      el.studioQueueTaskList.appendChild(card);
    });

    // Wire cancel buttons
    el.studioQueueTaskList.querySelectorAll('.btn-cancel-task').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        cancelSharedQueueItem(btn.dataset.itemId);
      });
    });

  } catch (err) {
    if (el.studioQueueTaskList) {
      el.studioQueueTaskList.innerHTML = `
        <div class="queue-empty-state">
          <div class="queue-empty-icon">⚠</div>
          <div class="queue-empty-title">Could not reach Shared Queue</div>
          <div class="queue-empty-sub">${escapeHtml(err.message)}</div>
        </div>
      `;
    }
  }
}

async function cancelSharedQueueItem(itemId) {
  try {
    let res = await fetch(`/api/queue/shared/${itemId}/cancel`, { method: 'POST' });
    if (!res.ok) {
      res = await fetch(`/api/queue/shared/${itemId}`, { method: 'DELETE' });
    }
    if (!res.ok) {
      res = await fetch(`/api/tasks/${itemId}`, { method: 'DELETE' });
    }
    const data = await res.json();
    if (res.ok) {
      showToast("Stopping workload...", "info");
      loadAndRenderQueueModal();
      fetchSystemStatus();
    } else {
      showToast(data.error || "Failed to cancel workload", "error");
    }
  } catch (err) {
    showToast(`Error: ${err.message}`, "error");
  }
}

async function clearQueueFinished() {
  try {
    const res = await fetch('/api/tasks/clear', { method: 'POST' });
    const data = await res.json();
    if (res.ok) {
      showToast(`Cleared ${data.cleared || 0} finished Studio task(s)`, "success");
      loadAndRenderQueueModal();
    } else {
      showToast("Failed to clear tasks", "error");
    }
  } catch (err) {
    showToast(`Error: ${err.message}`, "error");
  }
}

function closeAllModals() {
  if (el.modalSaveTo) el.modalSaveTo.classList.add('hidden');
  if (el.modalLibrary) el.modalLibrary.classList.add('hidden');
  if (el.modalShortcuts) el.modalShortcuts.classList.add('hidden');
  if (el.modalTaskQueue) el.modalTaskQueue.classList.add('hidden');
  if (state.queuePollingInterval) {
    clearInterval(state.queuePollingInterval);
    state.queuePollingInterval = null;
  }
}

function initModals() {
  // Queue & Telemetry Modal triggers
  if (el.queueBadge) {
    el.queueBadge.addEventListener('click', (e) => {
      e.preventDefault();
      toggleQueueModal();
    });
  }
  if (el.gpuLoadBadge) {
    el.gpuLoadBadge.addEventListener('click', (e) => {
      e.preventDefault();
      toggleQueueModal();
    });
  }
  if (el.btnCloseQueueModal) {
    el.btnCloseQueueModal.addEventListener('click', closeQueueModal);
  }
  if (el.btnCancelQueueModal) {
    el.btnCancelQueueModal.addEventListener('click', closeQueueModal);
  }
  if (el.btnRefreshQueueModal) {
    el.btnRefreshQueueModal.addEventListener('click', loadAndRenderQueueModal);
  }
  if (el.btnClearQueueFinished) {
    el.btnClearQueueFinished.addEventListener('click', clearQueueFinished);
  }
  if (el.queueModalFilters) {
    el.queueModalFilters.addEventListener('click', (e) => {
      const btn = e.target.closest('.pill-btn');
      if (!btn) return;
      el.queueModalFilters.querySelectorAll('.pill-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.queueModalFilter = btn.dataset.filter || 'all';
      loadAndRenderQueueModal();
    });
  }

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
        closeAllModals();
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

  // Update derivative lineage displays
  if (el.sepInputSelect) renderSeparationChildren(el.sepInputSelect.value);
  if (el.diarInputSelect) {
    renderDiarizationChildren(el.diarInputSelect.value);
    updateDiarInputMeta(el.diarInputSelect.value);
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

    // Space is handled by the shared player shortcut above. Keeping one
    // listener avoids toggling the audition player twice.
  });
}

// ==================== TASK POLLING HELPER ====================

function updateTaskProgressUI(task) {
  let statusText = null;
  let progressBar = null;
  if (task.type === "youtube_crawl") {
    statusText = el.ytTaskStatusText;
    progressBar = el.ytProgressBar;
  } else if (task.type === "separation" || task.type === "multi_model_separation") {
    statusText = el.sepTaskStatusText;
    progressBar = el.sepProgressBar;
  } else if (task.type === "diarization") {
    statusText = el.diarTaskStatusText;
    progressBar = el.diarProgressBar;
  }

  if (statusText) statusText.textContent = task.message || task.status;
  if (progressBar) {
    const known = task.progress_known === true || task.status === "completed";
    if (task.status === "pending" || (task.status === "running" && !known)) {
      progressBar.classList.add("progress-animated");
      progressBar.style.removeProperty("width");
      progressBar.style.removeProperty("transform");
    } else if (known) {
      const pct = task.progress > 1 ? task.progress : (task.progress || 0) * 100;
      progressBar.classList.remove("progress-animated");
      progressBar.style.width = `${Math.min(100, pct)}%`;
      progressBar.style.transform = "none";
    }
  }
}

function pollTask(taskId, onComplete, onError) {
  const poll = async () => {
    try {
      const res = await fetch(`/api/tasks/${taskId}`);
      if (!res.ok) throw new Error("Task polling error");
      const task = await res.json();
      updateTaskProgressUI(task);

      if (task.status === "completed") {
        if (onComplete) onComplete(task.result);
      } else if (task.status === "failed" || task.status === "cancelled") {
        if (onError) onError(task.error || task.message || "Task cancelled");
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

// ==================== SYSTEM STATUS & TELEMETRY ====================

function setTargetGpu(gpuId, notify = true) {
  if (!gpuId) return;
  state.selectedGpu = gpuId;
  try {
    localStorage.setItem('sonic_selected_gpu', gpuId);
  } catch (_) {}

  // Update sidebar active badge
  const pill = document.getElementById('sidebar-selected-gpu-pill');
  if (pill) {
    if (gpuId.startsWith('cuda:')) {
      const idx = gpuId.split(':')[1];
      pill.textContent = `GPU ${idx}`;
    } else if (gpuId === 'cuda') {
      pill.textContent = 'Auto GPU';
    } else if (gpuId === 'mps') {
      pill.textContent = 'MPS';
    } else {
      pill.textContent = 'CPU';
    }
  }

  // Update visual state of all cards in the sidebar
  const cards = document.querySelectorAll('#sidebar-gpu-cards-list .sidebar-gpu-card');
  let selectedDeviceName = gpuId;
  cards.forEach(card => {
    const cardId = card.getAttribute('data-gpu-id');
    const isSelected = (cardId === gpuId) || (gpuId.startsWith('cuda:') && cardId === gpuId);
    card.classList.toggle('selected', isSelected);
    const radioDot = card.querySelector('.sidebar-gpu-radio-dot');
    if (radioDot) radioDot.classList.toggle('checked', isSelected);
    const badge = card.querySelector('.sidebar-gpu-status-badge');
    if (badge) {
      badge.textContent = isSelected ? 'ACTIVE' : 'READY';
      badge.className = `sidebar-gpu-status-badge ${isSelected ? 'active' : 'standby'}`;
    }
    if (isSelected) {
      const nameEl = card.querySelector('.sidebar-gpu-card-name');
      if (nameEl) selectedDeviceName = nameEl.textContent;
    }
  });

  // Synchronize dropdown selects in tabs
  const syncSelect = (sel) => {
    if (!sel) return;
    const hasOpt = Array.from(sel.options).some(o => o.value === gpuId);
    if (hasOpt) {
      sel.value = gpuId;
    } else if (gpuId.startsWith('cuda:') && Array.from(sel.options).some(o => o.value === 'cuda')) {
      sel.value = 'cuda';
    }
  };

  syncSelect(el.sepDeviceSelect);
  syncSelect(el.diarDeviceSelect);
  
  if (notify) {
    showToast(`⚡ Target compute device set to ${selectedDeviceName}`, 'info');
  }
}

function renderSidebarGpuCards(devices, gpuInfo) {
  const container = document.getElementById('sidebar-gpu-cards-list');
  if (!container) return;

  const currentSelected = state.selectedGpu || localStorage.getItem('sonic_selected_gpu') || (devices && devices[0] ? devices[0].id : 'cuda:0');
  state.selectedGpu = currentSelected;

  if (!devices || devices.length === 0) {
    if (gpuInfo && gpuInfo.type === 'mps') {
      container.innerHTML = `
        <div class="sidebar-gpu-card selected" data-gpu-id="mps" role="button" tabindex="0" title="Apple Silicon MPS Active">
          <div class="sidebar-gpu-card-header">
            <div class="sidebar-gpu-title-group">
              <span class="sidebar-gpu-radio-dot checked"></span>
              <span class="sidebar-gpu-card-name">Apple Silicon (MPS)</span>
            </div>
            <span class="sidebar-gpu-status-badge active">ACTIVE</span>
          </div>
          <div class="sidebar-gpu-metrics">
            <span class="gpu-metric-pill">Unified Memory</span>
          </div>
        </div>
      `;
    } else {
      container.innerHTML = `
        <div class="sidebar-gpu-card selected" data-gpu-id="cpu" role="button" tabindex="0" title="CPU Compute Mode">
          <div class="sidebar-gpu-card-header">
            <div class="sidebar-gpu-title-group">
              <span class="sidebar-gpu-radio-dot checked"></span>
              <span class="sidebar-gpu-card-name">CPU Multi-Core</span>
            </div>
            <span class="sidebar-gpu-status-badge active">ACTIVE</span>
          </div>
          <div class="sidebar-gpu-metrics">
            <span class="gpu-metric-pill">Host RAM</span>
          </div>
        </div>
      `;
    }
    const pill = document.getElementById('sidebar-selected-gpu-pill');
    if (pill) pill.textContent = (gpuInfo && gpuInfo.type === 'mps') ? 'MPS' : 'CPU';
    return;
  }

  // Check if we can do an in-place update to preserve DOM & listeners
  const existingCards = container.querySelectorAll('.sidebar-gpu-card');
  if (existingCards.length === devices.length) {
    devices.forEach((d, i) => {
      const card = existingCards[i];
      if (!card) return;
      const isSelected = (d.id === state.selectedGpu) || (!state.selectedGpu && i === 0);
      card.classList.toggle('selected', isSelected);

      const radioDot = card.querySelector('.sidebar-gpu-radio-dot');
      if (radioDot) radioDot.classList.toggle('checked', isSelected);

      const badge = card.querySelector('.sidebar-gpu-status-badge');
      if (badge) {
        badge.textContent = isSelected ? 'ACTIVE' : 'READY';
        badge.className = `sidebar-gpu-status-badge ${isSelected ? 'active' : 'standby'}`;
      }

      const loadVal = Number.isFinite(d.load_percent) ? Math.round(d.load_percent) : (Number.isFinite(d.utilization_percent) ? Math.round(d.utilization_percent) : 0);
      const vramUsed = d.used_vram_mb != null ? (d.used_vram_mb >= 1024 ? `${(d.used_vram_mb / 1024).toFixed(1)}G` : `${Math.round(d.used_vram_mb)}M`) : '--';
      const vramTotal = d.total_vram_mb != null ? (d.total_vram_mb >= 1024 ? `${Math.round(d.total_vram_mb / 1024)}GB` : `${Math.round(d.total_vram_mb)}MB`) : '--';
      const dTemp = d.temperature_c != null ? `${Math.round(d.temperature_c)}°C` : '';
      const dPower = d.power_w != null ? `${Math.round(d.power_w)} / ${d.power_limit_w != null ? Math.round(d.power_limit_w) : '--'} W` : '';

      const loadEl = card.querySelector('.metric-load');
      if (loadEl) loadEl.textContent = `⚡ ${loadVal}%`;
      const vramEl = card.querySelector('.metric-vram');
      if (vramEl) vramEl.textContent = `💾 ${vramUsed}/${vramTotal}`;
      const pwrEl = card.querySelector('.metric-pwr');
      if (pwrEl) pwrEl.textContent = dPower;
      const tempEl = card.querySelector('.metric-temp');
      if (tempEl) tempEl.textContent = dTemp;
      const barEl = card.querySelector('.sidebar-gpu-meter-fill');
      if (barEl) barEl.style.width = `${loadVal}%`;
    });
    return;
  }

  // Initial full render of cards
  container.innerHTML = devices.map((d, i) => {
    const isSelected = (d.id === state.selectedGpu) || (!state.selectedGpu && i === 0);
    const loadVal = Number.isFinite(d.load_percent) ? Math.round(d.load_percent) : (Number.isFinite(d.utilization_percent) ? Math.round(d.utilization_percent) : 0);
    const vramUsed = d.used_vram_mb != null ? (d.used_vram_mb >= 1024 ? `${(d.used_vram_mb / 1024).toFixed(1)}G` : `${Math.round(d.used_vram_mb)}M`) : '--';
    const vramTotal = d.total_vram_mb != null ? (d.total_vram_mb >= 1024 ? `${Math.round(d.total_vram_mb / 1024)}GB` : `${Math.round(d.total_vram_mb)}MB`) : '--';
    const dTemp = d.temperature_c != null ? `${Math.round(d.temperature_c)}°C` : '';
    const dPower = d.power_w != null ? `${Math.round(d.power_w)} / ${d.power_limit_w != null ? Math.round(d.power_limit_w) : '--'} W` : '';

    let cleanName = (d.name || `GPU ${i}`).replace(/^NVIDIA\s+/i, '').replace(/^GeForce\s+/i, '');

    return `
      <div class="sidebar-gpu-card ${isSelected ? 'selected' : ''}" data-gpu-id="${d.id}" data-gpu-index="${i}" role="button" tabindex="0" title="Click to route all AI jobs to GPU ${i} (${escapeHtml(d.name)})">
        <div class="sidebar-gpu-card-header">
          <div class="sidebar-gpu-title-group">
            <span class="sidebar-gpu-radio-dot ${isSelected ? 'checked' : ''}"></span>
            <span class="sidebar-gpu-card-name">GPU ${i}: ${escapeHtml(cleanName)}</span>
          </div>
          <span class="sidebar-gpu-status-badge ${isSelected ? 'active' : 'standby'}">${isSelected ? 'ACTIVE' : 'READY'}</span>
        </div>
        <div class="sidebar-gpu-metrics">
          <span class="gpu-metric-pill metric-load" title="GPU Core Utilization">⚡ ${loadVal}%</span>
          <span class="gpu-metric-pill metric-vram" title="VRAM Memory Used / Total">💾 ${vramUsed}/${vramTotal}</span>
          ${dPower ? `<span class="gpu-metric-pill metric-pwr font-mono" title="Current power draw / configured power limit">${dPower}</span>` : ''}
          ${dTemp ? `<span class="gpu-metric-pill metric-temp" title="GPU Temperature">${dTemp}</span>` : ''}
        </div>
        <div class="sidebar-gpu-meter-track">
          <div class="sidebar-gpu-meter-fill" style="width: ${loadVal}%;"></div>
        </div>
      </div>
    `;
  }).join('');

  // Attach click listeners to cards
  container.querySelectorAll('.sidebar-gpu-card').forEach(card => {
    card.addEventListener('click', () => {
      const gpuId = card.getAttribute('data-gpu-id');
      setTargetGpu(gpuId, true);
    });
  });

  setTargetGpu(state.selectedGpu || devices[0].id, false);
}

function updateTelemetryDisplay(telemetry) {
  if (!telemetry) return;
  const gpu = telemetry.gpu;
  if (!gpu) return;

  renderSidebarGpuCards(gpu.devices || [], gpu);

  if (gpu.available && gpu.type === 'cuda') {
    const isMultiGpu = gpu.device_count > 1 && gpu.devices && gpu.devices.length > 1;
    const load = isMultiGpu
      ? (gpu.aggregate?.avg_load_percent ?? gpu.load_percent ?? gpu.utilization_percent)
      : (gpu.load_percent ?? gpu.utilization_percent);
    const vramPct = gpu.vram_percent;
    const loadPct = Number.isFinite(load) ? Math.max(0, Math.min(100, load)) : 0;
    const powerW = isMultiGpu
      ? (gpu.aggregate?.total_power_w ?? gpu.power_w)
      : gpu.power_w;
    const powerLimitW = isMultiGpu
      ? (gpu.aggregate?.total_power_limit_w ?? gpu.power_limit_w)
      : gpu.power_limit_w;

    if (el.gpuLoadLabel) {
      if (isMultiGpu) {
        const devLoads = gpu.devices.map(d => {
          const l = d.load_percent ?? d.utilization_percent;
          return Number.isFinite(l) ? `${Math.round(l)}%` : '--';
        }).join(' | ');
        el.gpuLoadLabel.textContent = `GPU: ${devLoads}`;
      } else {
        el.gpuLoadLabel.textContent = Number.isFinite(load) ? `GPU: ${Math.round(load)}%` : 'GPU: Active';
      }
    }
    if (el.headerGpuMeter) {
      el.headerGpuMeter.style.width = `${loadPct}%`;
    }
    if (el.gpuLoadBadge) {
      const pwr = powerW != null ? ` · ${powerW}/${powerLimitW ?? '--'}W` : '';
      const tip = isMultiGpu
        ? `${gpu.device_count} GPUs Active\n${gpu.devices.map((d, i) => `GPU ${i} (${d.name}): ${d.load_percent ?? 0}% load · ${d.used_vram_mb}/${d.total_vram_mb} MB · ${d.power_w ?? '--'}/${d.power_limit_w ?? '--'}W · ${d.temperature_c ?? '--'}°C`).join('\n')}`
        : `${gpu.name}: ${Math.round(loadPct)}% load · ${gpu.used_vram_mb}/${gpu.total_vram_mb} MB (${Math.round(vramPct || 0)}%)${pwr} · ${gpu.temperature_c ?? '--'}°C`;
      el.gpuLoadBadge.title = `${tip}\n(Click to view full Telemetry & Queue)`;
    }
  } else if (gpu.type === 'mps') {
    if (el.gpuLoadLabel) el.gpuLoadLabel.textContent = 'MPS';
    if (el.headerGpuMeter) el.headerGpuMeter.style.width = '0%';
    if (el.gpuLoadBadge) el.gpuLoadBadge.title = 'Apple Silicon (MPS Accelerator)';
  } else {
    if (el.gpuLoadLabel) el.gpuLoadLabel.textContent = 'CPU';
    if (el.headerGpuMeter) el.headerGpuMeter.style.width = '0%';
    if (el.gpuLoadBadge) el.gpuLoadBadge.title = 'CPU Mode (No GPU accelerator detected)';
  }
}

async function fetchSystemStatus() {
  try {
    const res = await fetch("/api/system/status");
    const data = await res.json();
    state.systemStatus = data;
    if (el.deviceLabel) {
      el.deviceLabel.textContent = `${data.device_name.split(':')[0]}`;
    }

    // Render accelerator cards, including CPU/MPS fallback states.
    renderSidebarGpuCards(data.devices || [], data.telemetry?.gpu || { type: data.device_type, available: data.cuda_available });

    // Dynamically populate device selection dropdowns if multiple GPUs exist
    if (data.devices && data.devices.length > 0 && !state._devicesPopulated) {
      state._devicesPopulated = true;
      const populateSelect = (selectEl) => {
        if (!selectEl) return;
        const currentVal = selectEl.value;
        let html = '<option value="auto">Auto (Best Available)</option>';
        data.devices.forEach(d => {
          const pwr = d.power_w != null ? ` · ${d.power_w}/${d.power_limit_w ?? '--'}W` : '';
          html += `<option value="${d.id}">${d.id} (${d.name}${pwr})</option>`;
        });
        html += '<option value="cpu">CPU</option>';
        selectEl.innerHTML = html;
        if (state.selectedGpu && Array.from(selectEl.options).some(o => o.value === state.selectedGpu)) {
          selectEl.value = state.selectedGpu;
        } else if (currentVal && Array.from(selectEl.options).some(o => o.value === currentVal)) {
          selectEl.value = currentVal;
        }
      };
      populateSelect(el.sepDeviceSelect);
      populateSelect(el.diarDeviceSelect);

      if (el.sepDeviceSelect) {
        el.sepDeviceSelect.addEventListener('change', (e) => setTargetGpu(e.target.value, false));
      }
      if (el.diarDeviceSelect) {
        el.diarDeviceSelect.addEventListener('change', (e) => setTargetGpu(e.target.value, false));
      }
    }

    if (el.queueLabel) {
      if (data.shared_queue) {
        const totalActive = data.shared_queue.total_running;
        const totalQueued = data.shared_queue.total_queued;
        el.queueLabel.textContent = totalActive > 0 ? `GPU Active ${totalActive} • Queued ${totalQueued}` : `GPU Queue ${totalQueued}`;
        if (el.queueDot) el.queueDot.classList.toggle('dot-pulse', totalActive > 0);
      } else if (data.task_queue) {
        const active = data.task_queue.running;
        const queued = data.task_queue.queued;
        el.queueLabel.textContent = active ? `Running ${active} • Queued ${queued}` : `Queue ${queued}`;
        if (el.queueDot) el.queueDot.classList.toggle('dot-pulse', active > 0);
      }
    }

    if (data.telemetry) {
      updateTelemetryDisplay(data.telemetry);
    }
  } catch (err) {
    if (el.deviceLabel) el.deviceLabel.textContent = "Offline";
    if (el.queueLabel) el.queueLabel.textContent = "Queue offline";
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
  state.activeTab = tabId;

  if (tabId === 'tab-comparison') {
    // Only one audio context should be audible at a time. The bottom player
    // will now operate on the audition element while this tab is active.
    el.audio.pause();
  } else if (isAuditionPlaybackActive()) {
    auditionAudio.pause();
  }

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
    syncActivePlaybackControls();
  } else if (tabId === 'tab-diarization') {
    if (el.diarInputSelect) {
      if (!el.diarInputSelect.value && state.activeAudio) {
        el.diarInputSelect.value = state.activeAudio.id;
        renderDiarizationChildren(state.activeAudio.id);
        updateDiarInputMeta(state.activeAudio.id);
        loadDiarWaveform(state.activeAudio.id);
      } else if (el.diarInputSelect.value) {
        updateDiarInputMeta(el.diarInputSelect.value);
        renderDiarizationChildren(el.diarInputSelect.value);
      }
    }
    setDiarZoom(state.diarization.zoom || 1.0);
    renderDiarWaveform();
    renderDiarRuler();
    updateMinimapViewport();
  } else if (tabId === 'tab-matrix') {
    fetchEvaluations();
  }

  if (tabId !== 'tab-diarization' && el.audio) {
    el.audio.muted = false;
  }

  if (tabId !== 'tab-comparison') syncActivePlaybackControls();
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

  if (el.btnClearSession) {
    el.btnClearSession.addEventListener('click', async () => {
      if (confirm("Are you sure you want to clear all active audio objects from the current session?")) {
        try {
          const res = await fetch("/api/audio/clear-all", { method: "POST" });
          const data = await parseJsonResponse(res);
          state.audioList = [];
          state.activeAudio = null;
          if (el.activeSection) el.activeSection.classList.add('hidden');
          showToast(`Cleared ${data.cleared_count || 0} session items`, "info");
          await fetchAudioList();
        } catch (err) {
          showToast(`Failed to clear session: ${err.message}`, "error");
        }
      }
    });
  }

  window.addEventListener('click', () => {
    document.querySelectorAll('.actions-popup-menu').forEach(m => m.classList.add('hidden'));
  });

  try { await fetchSystemStatus(); } catch (e) { console.error("fetchSystemStatus error:", e); }
  window.setInterval(fetchSystemStatus, 2000);
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
