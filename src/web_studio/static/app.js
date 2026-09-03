/**
 * SonicStudio — Audio Preparation & Separation Studio Web Application
 * Modular ES6 Vanilla JavaScript Frontend Architecture
 */

// ==================== STATE MANAGEMENT ====================

const state = {
  activeTab: 'tab-workspace',
  activeAudio: null,       // Currently selected Audio metadata
  selection: { start: 0, end: 0, active: false },
  cutUnit: 'seconds',
  zoom: 1.0,
  waveform: {
    audioId: null,
    start: 0,
    end: 0,
    data: null,
    controller: null,
    requestId: 0,
    loading: false,
    error: '',
    editing: false,
    panning: false,
  },
  audioList: [],           // All registered Audio items
  serverFiles: [],         // Files on disk from /api/library
  systemStatus: null,
  selectedGpu: localStorage.getItem('sonic_selected_gpu') || 'cuda:0',

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
  libraryLoadTarget: "workspace",
  librarySort: "newest",
  librarySelectedPaths: new Set(),

  // Tab 7 Project Explorer Filter State
  tabLibrarySearch: "",
  tabLibraryCategory: "all",
  tabLibraryFilters: { dataset: 'all', channel: 'all', speaker: 'all', verification: 'all', format: 'all' },

  // Diarization Studio State
  diarization: {
    audioId: null,
    data: null,
    speakers: [],
    rawTurns: [],
    turns: [],
    cleanTurnsEnabled: false,
    cleanTurnsSummary: null,
    cleanTurnsSettings: {
      min_turn_duration_s: 0.5,
      merge_same_speaker_gap_s: 1.0,
      boundary_collar_s: 0.0,
      jitter_max_duration_s: 3.0,
    },
    extractionSettings: {
      add_extra: false,
      stop_at_other_speakers: false,
      pre_roll_s: 0.12,
      post_roll_s: 0.20,
    },
    customNames: {},
    colors: {},
    zoom: 1.0,
    duration: 0,
    activeTurnIndex: null,
    activeSpeakerFilter: 'all',
    minDurFilter: 0,
    maxDurFilter: 0,
    overlapFilter: false,
    targetMatchFilter: 'all',
    reviewFilter: 'all',
    searchQuery: '',
    sortMode: 'time-asc',
    histogramBinWidth: localStorage.getItem('sonic_diar_hist_bin') || 'auto',
    histogramCustomBinWidth: parseFloat(localStorage.getItem('sonic_diar_hist_bin_custom') || '1.0') || 1.0,
    isScrubbing: false,
    soloSpeaker: null,
    mutedSpeakers: new Set(),
    autoAdvance: false,
    history: [],
    historySearch: '',
    activeHistoryId: null,
    waveform: { data: null, controller: null, requestId: 0, loading: false, error: '' },
  },

  // Manual ground-truth diarization annotation and model evaluation
  annotation: {
    audioId: null,
    current: null,
    catalog: [],
    speakers: [],
    turns: [],
    activeSpeakerId: null,
    selectedTurnId: null,
    markIn: null,
    markOut: null,
    zoom: 1,
    snapS: 0.05,
    stepS: 0.1,
    undo: [],
    redo: [],
    saveTimer: null,
    savePromise: null,
    dirty: false,
    editVersion: 0,
    resultCatalog: [],
    seedResultId: null,
    evaluation: null,
    drag: null,
    rangeDrag: null,
    suppressTimelineClick: false,
    loopTurnId: null,
    waveform: { data: null, controller: null, requestId: 0, loading: false, error: '' },
  },

  // Post-diarization target-speaker review state
  targetSpeaker: {
    scored: null,
    audioId: null,
    profileName: '',
    assignedSpeakerId: '',
    threshold: 0.60,
    minDur: 1.5,
    excludeOverlap: true,
    labels: {},
  },

  // Speaker Purity Workbench state
  purity: {
    audioId: null,
    diarizationResults: [],
    selectedResultIds: new Set(),
    profileName: '',
    results: null,
    metrics: null,
    settings: {
      minDuration: 1.5,
    },
    overlap: {
      enabled: true,
      backend: 'gemma4',
      model: '',
      endpoint: '',
      timeout: 120,
      maxOutputTokens: 128,
      prompt: 'Does this audio contain overlapping speech from two or more speakers at the same time?',
      failurePolicy: 'fail_closed',
      minSecondarySpeech: 0.25,
      batchSize: 1,
    },
    serverConfig: null,
    verifierStatus: null,
    runSettings: null,
    filterStatus: 'all',
    filterSpeaker: 'all',
    filterReason: 'all',
    sortMode: 'time-asc',
    searchQuery: '',
  },
};

const puritySegmentAudio = new Audio();
let activePuritySegmentKey = null;
let puritySegmentPlayRaf = 0;
let workspacePreviewRaf = 0;
let rangePreviewSeeking = false;
let rangePreviewGeneration = 0;
const turnPreviewAudio = new Audio();
let turnPreviewUrl = null;
let turnPreviewRaf = 0;
let activeTurnPreviewKey = null;
let turnPreviewRange = null;
let turnPreviewGeneration = 0;
puritySegmentAudio.addEventListener('ended', stopPuritySegmentPreview);
turnPreviewAudio.addEventListener('ended', finishTurnPreview);
turnPreviewAudio.addEventListener('error', () => {
  if (!activeTurnPreviewKey) return;
  showToast('Unable to play this turn', 'error');
  stopTurnPreview();
});

const TIMELINE_MAX_ZOOM = 1000;
const TIMELINE_ZOOM_SLIDER_MAX = 1000;

function clampTimelineZoom(zoom, minZoom) {
  const parsed = Number(zoom);
  const fallback = Number.isFinite(minZoom) ? minZoom : 1;
  return Math.min(
    TIMELINE_MAX_ZOOM,
    Math.max(fallback, Number.isFinite(parsed) && parsed > 0 ? parsed : fallback),
  );
}

function timelineZoomToSlider(zoom, minZoom) {
  const clamped = clampTimelineZoom(zoom, minZoom);
  const t = Math.log(clamped / minZoom) / Math.log(TIMELINE_MAX_ZOOM / minZoom);
  return Math.round(Math.max(0, Math.min(1, t)) * TIMELINE_ZOOM_SLIDER_MAX);
}

function timelineSliderToZoom(sliderValue, minZoom) {
  const t = Math.max(0, Math.min(1, Number(sliderValue) / TIMELINE_ZOOM_SLIDER_MAX));
  return minZoom * Math.pow(TIMELINE_MAX_ZOOM / minZoom, t);
}

function formatZoomMultiplier(zoom) {
  if (!Number.isFinite(zoom)) return '1';
  if (Number.isInteger(zoom) || Math.abs(zoom - Math.round(zoom)) < 1e-6) return String(Math.round(zoom));
  return zoom.toFixed(1);
}

// DOM Elements Cache
const el = {
  // Navigation
  tabs: document.querySelectorAll('.nav-tab'),
  tabPanes: document.querySelectorAll('.tab-pane'),
  deviceLabel: document.getElementById('device-label'),
  gpuLoadBadge: document.getElementById('gpu-load-badge'),
  gpuLoadLabel: document.getElementById('gpu-load-label'),
  headerGpuMeter: document.getElementById('header-gpu-meter'),
  queueLabel: document.getElementById('queue-label'),
  queueDot: document.getElementById('queue-dot'),
  btnThemeToggle: document.getElementById('btn-theme-toggle'),
  iconThemeSun: document.getElementById('icon-theme-sun'),
  iconThemeMoon: document.getElementById('icon-theme-moon'),

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
  timeCurrent: document.getElementById('player-time-current'),
  timeTotal: document.getElementById('player-time-total'),
  playerTitle: document.getElementById('player-track-title'),
  playerSub: document.getElementById('player-track-sub'),

  // Workspace
  dropzone: document.getElementById('audio-dropzone'),
  fileInput: document.getElementById('file-input'),
  ytUrlInput: document.getElementById('yt-url-input'),
  ytSampleRate: document.getElementById('yt-sample-rate'),
  ytIngestHint: document.getElementById('yt-ingest-hint'),
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
  metaChannel: document.getElementById('meta-channel'),
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
  timeRuler: document.getElementById('time-ruler'),
  waveformScrollbar: document.getElementById('waveform-scrollbar'),
  btnZoomIn: document.getElementById('btn-zoom-in'),
  btnZoomOut: document.getElementById('btn-zoom-out'),
  btnResetZoom: document.getElementById('btn-reset-zoom'),
  zoomLabel: document.getElementById('zoom-level-label'),
  wsZoomInput: document.getElementById('ws-zoom-input'),
  btnToggleSpec: document.getElementById('btn-toggle-spectrogram'),
  spectrogramPanel: document.getElementById('spectrogram-panel'),
  specImageWrapper: document.getElementById('spec-img-wrapper'),
  specImage: document.getElementById('spec-image'),
  specLoader: document.getElementById('spec-loader'),
  btnRefreshSpec: document.getElementById('btn-refresh-spec'),
  specSelectionOverlay: document.getElementById('spec-selection-overlay'),
  specPlayheadLine: document.getElementById('spec-playhead-line'),

  // Audio Cutter
  cutStartInput: document.getElementById('cut-start-input'),
  cutEndInput: document.getElementById('cut-end-input'),
  cutDurationDisplay: document.getElementById('cut-duration-display'),
  cutValidation: document.getElementById('cut-validation'),
  btnSetStartPlayhead: document.getElementById('btn-set-start-playhead'),
  btnSetEndPlayhead: document.getElementById('btn-set-end-playhead'),
  rangePresets: document.querySelectorAll('.range-preset'),
  btnPreviewCut: document.getElementById('btn-preview-cut'),
  btnApplyCut: document.getElementById('btn-apply-cut'),
  btnCutAndAudition: document.getElementById('btn-cut-and-audition'),
  btnCutAndRunModels: document.getElementById('btn-cut-and-run-models'),
  cutsTableBody: document.getElementById('cuts-table-body'),
  cutsCounterBadge: document.getElementById('cuts-counter-badge'),
  cutUnitRadios: document.querySelectorAll('input[name="cut_unit"]'),

  // Separation Studio
  sepInputSelect: document.getElementById('sep-input-select'),
  btnSepBrowseLibrary: document.getElementById('btn-sep-browse-library'),
  sepChildrenBox: document.getElementById('sep-children-box'),
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
  btnToggleDiarSetup: document.getElementById('btn-toggle-diar-setup'),
  diarSetupBody: document.getElementById('diar-setup-body'),
  diarSetupToggleIcon: document.getElementById('diar-setup-toggle-icon'),
  diarSetupToggleText: document.getElementById('diar-setup-toggle-text'),
  diarInputSelect: document.getElementById('diar-input-select'),
  btnDiarBrowseLibrary: document.getElementById('btn-diar-browse-library'),
  diarSavedNoticePill: document.getElementById('diar-saved-notice-pill'),
  diarSavedNoticeText: document.getElementById('diar-saved-notice-text'),
  btnLoadSavedForTrack: document.getElementById('btn-load-saved-for-track'),
  diarAudioMetaChip: document.getElementById('diar-audio-meta-chip'),
  diarChildrenBox: document.getElementById('diar-children-box'),
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
  diarBatchSize: document.getElementById('diar-batch-size'),
  diarSortformerOnset: document.getElementById('diar-sortformer-onset'),
  diarSortformerOffset: document.getElementById('diar-sortformer-offset'),
  diarSortformerPadOnset: document.getElementById('diar-sortformer-pad-onset'),
  diarSortformerPadOffset: document.getElementById('diar-sortformer-pad-offset'),
  btnRunDiarization: document.getElementById('btn-run-diarization'),
  btnDiarReset: document.getElementById('btn-diar-reset'),
  diarTaskProgressBox: document.getElementById('diar-task-progress-box'),
  diarTaskTimer: document.getElementById('diar-task-timer'),
  diarProgressBar: document.getElementById('diar-progress-bar'),
  diarTaskStatusText: document.getElementById('diar-task-status-text'),
  diarEmptyPlaceholder: document.getElementById('diar-empty-placeholder'),
  diarResultsWrapper: document.getElementById('diar-results-wrapper'),
  diarModelBadge: document.getElementById('diar-model-badge'),
  diarSpeakerCountBadge: document.getElementById('diar-speaker-count-badge'),
  diarTurnsCountBadge: document.getElementById('diar-turns-count-badge'),
  diarSpeechRatioBadge: document.getElementById('diar-speech-ratio-badge'),
  btnDiarSkipBack: document.getElementById('btn-diar-skip-back'),
  btnDiarPlayToggle: document.getElementById('btn-diar-play-toggle'),
  iconDiarPlay: document.getElementById('icon-diar-play'),
  iconDiarPause: document.getElementById('icon-diar-pause'),
  btnDiarSkipFwd: document.getElementById('btn-diar-skip-fwd'),
  diarTimeCurrent: document.getElementById('diar-time-current'),
  diarTimeTotal: document.getElementById('diar-time-total'),
  btnDiarZoomOut: document.getElementById('btn-diar-zoom-out'),
  btnDiarZoomIn: document.getElementById('btn-diar-zoom-in'),
  btnDiarZoomFit: document.getElementById('btn-diar-zoom-fit'),
  diarZoomLevel: document.getElementById('diar-zoom-level'),
  diarZoomInput: document.getElementById('diar-zoom-input'),
  diarSpeedSelect: document.getElementById('diar-speed-select'),
  diarAutoNext: document.getElementById('diar-auto-next'),
  diarMultitrackViewport: document.getElementById('diar-multitrack-viewport'),
  diarLaneLabelsCol: document.getElementById('diar-lane-labels-col'),
  diarSpkLabelsWrap: document.getElementById('diar-spk-labels-wrap'),
  diarTracksArea: document.getElementById('diar-tracks-area'),
  diarRulerTrack: document.getElementById('diar-ruler-track'),
  diarWaveformTrack: document.getElementById('diar-waveform-track'),
  diarWaveformCanvas: document.getElementById('diar-waveform-canvas'),
  diarSpeakerLanesWrap: document.getElementById('diar-speaker-lanes-wrap'),
  diarPlayheadLine: document.getElementById('diar-playhead-line'),
  diarPlayheadHandle: document.getElementById('diar-playhead-handle'),
  diarTurnTooltip: document.getElementById('diar-turn-tooltip'),
  diarSpeakersGrid: document.getElementById('diar-speakers-grid'),
  diarExtractModeSelect: document.getElementById('diar-extract-mode-select'),
  diarExtractAddExtra: document.getElementById('diar-extract-add-extra'),
  diarExtractStopOther: document.getElementById('diar-extract-stop-other'),
  diarExtractAmounts: document.getElementById('diar-extract-amounts'),
  diarExtractPreRoll: document.getElementById('diar-extract-pre-roll'),
  diarExtractPostRoll: document.getElementById('diar-extract-post-roll'),
  purityExtractAddExtra: document.getElementById('purity-extract-add-extra'),
  purityExtractStopOther: document.getElementById('purity-extract-stop-other'),
  purityExtractAmounts: document.getElementById('purity-extract-amounts'),
  purityExtractPreRoll: document.getElementById('purity-extract-pre-roll'),
  purityExtractPostRoll: document.getElementById('purity-extract-post-roll'),
  btnExtractAllSpeakers: document.getElementById('btn-extract-all-speakers'),
  diarFilterSpeakerSelect: document.getElementById('diar-filter-speaker-select'),
  diarTurnsSearchInput: document.getElementById('diar-turns-search-input'),
  diarFilterMinDur: document.getElementById('diar-filter-min-dur'),
  diarFilterMaxDur: document.getElementById('diar-filter-max-dur'),
  btnDiarFilterOverlaps: document.getElementById('btn-diar-filter-overlaps'),
  diarFilterTargetSelect: document.getElementById('diar-filter-target-select'),
  diarFilterReviewSelect: document.getElementById('diar-filter-review-select'),
  btnDiarClearFilters: document.getElementById('btn-diar-clear-filters'),
  btnDiarCleanTurns: document.getElementById('btn-diar-clean-turns'),
  diarCleanJitterMax: document.getElementById('diar-clean-jitter-max'),
  diarCleanBoundaryCollar: document.getElementById('diar-clean-boundary-collar'),
  diarCleanMergeGap: document.getElementById('diar-clean-merge-gap'),
  diarCleanMinDuration: document.getElementById('diar-clean-min-duration'),
  diarCleanTurnsSummary: document.getElementById('diar-clean-turns-summary'),
  diarSortTurnsSelect: document.getElementById('diar-sort-turns-select'),
  diarFilteredTurnsCount: document.getElementById('diar-filtered-turns-count'),
  diarDurationHistogramPlot: document.getElementById('diar-duration-histogram-plot'),
  diarDurationHistogramSummary: document.getElementById('diar-duration-histogram-summary'),
  diarHistogramBinSelect: document.getElementById('diar-histogram-bin-select'),
  diarHistogramBinCustom: document.getElementById('diar-histogram-bin-custom'),
  diarReviewedCount: document.getElementById('diar-reviewed-count'),
  diarAcceptedCount: document.getElementById('diar-accepted-count'),
  diarRejectedCount: document.getElementById('diar-rejected-count'),
  btnDownloadFilteredTurns: document.getElementById('btn-download-filtered-turns'),
  btnDownloadFilteredTurnsLabel: document.getElementById('btn-download-filtered-turns-label'),
  turnsTableBody: document.getElementById('turns-table-body'),
  btnDownloadExport: document.getElementById('btn-download-export'),
  diarHistoryCountBadge: document.getElementById('diar-history-count-badge'),
  btnClearDiarHistory: document.getElementById('btn-clear-diar-history'),
  diarHistorySearchInput: document.getElementById('diar-history-search-input'),
  diarHistoryList: document.getElementById('diar-history-list'),

  // Manual annotation and diarization evaluation
  annSaveState: document.getElementById('ann-save-state'),
  annImportInput: document.getElementById('ann-import-input'),
  btnAnnImport: document.getElementById('btn-ann-import'),
  annExportFormat: document.getElementById('ann-export-format'),
  btnAnnExport: document.getElementById('btn-ann-export'),
  annAudioSelect: document.getElementById('ann-audio-select'),
  btnAnnBrowseLibrary: document.getElementById('btn-ann-browse-library'),
  annAudioMeta: document.getElementById('ann-audio-meta'),
  annSavedSelect: document.getElementById('ann-saved-select'),
  btnAnnNew: document.getElementById('btn-ann-new'),
  btnAnnLoad: document.getElementById('btn-ann-load'),
  btnAnnDelete: document.getElementById('btn-ann-delete'),
  annNameInput: document.getElementById('ann-name-input'),
  annRevisionLabel: document.getElementById('ann-revision-label'),
  annSeedResultSelect: document.getElementById('ann-seed-result-select'),
  annSeedResultMeta: document.getElementById('ann-seed-result-meta'),
  btnAnnCreateSeed: document.getElementById('btn-ann-create-seed'),
  annSeedNotice: document.getElementById('ann-seed-notice'),
  annSeedNoticeDetail: document.getElementById('ann-seed-notice-detail'),
  annEmptyState: document.getElementById('ann-empty-state'),
  annWorkspace: document.getElementById('ann-workspace'),
  btnAnnStart: document.getElementById('btn-ann-start'),
  btnAnnBack1: document.getElementById('btn-ann-back-1'),
  btnAnnBackFrame: document.getElementById('btn-ann-back-frame'),
  btnAnnPlay: document.getElementById('btn-ann-play'),
  btnAnnForwardFrame: document.getElementById('btn-ann-forward-frame'),
  btnAnnForward1: document.getElementById('btn-ann-forward-1'),
  annTimecode: document.getElementById('ann-timecode'),
  annStepSelect: document.getElementById('ann-step-select'),
  annSnapSelect: document.getElementById('ann-snap-select'),
  annSpeedSelect: document.getElementById('ann-speed-select'),
  btnAnnUndo: document.getElementById('btn-ann-undo'),
  btnAnnRedo: document.getElementById('btn-ann-redo'),
  annSpeakerChips: document.getElementById('ann-speaker-chips'),
  btnAnnAddSpeaker: document.getElementById('btn-ann-add-speaker'),
  btnAnnRenameSpeaker: document.getElementById('btn-ann-rename-speaker'),
  btnAnnLinkSpeaker: document.getElementById('btn-ann-link-speaker'),
  btnAnnMergeSpeaker: document.getElementById('btn-ann-merge-speaker'),
  btnAnnRemoveSpeaker: document.getElementById('btn-ann-remove-speaker'),
  annMarkIn: document.getElementById('ann-mark-in'),
  annMarkOut: document.getElementById('ann-mark-out'),
  btnAnnSetIn: document.getElementById('btn-ann-set-in'),
  btnAnnSetOut: document.getElementById('btn-ann-set-out'),
  annMarkDuration: document.getElementById('ann-mark-duration'),
  btnAnnCreateTurn: document.getElementById('btn-ann-create-turn'),
  btnAnnClearMarks: document.getElementById('btn-ann-clear-marks'),
  annTurnCount: document.getElementById('ann-turn-count'),
  annOverlapCount: document.getElementById('ann-overlap-count'),
  btnAnnZoomOut: document.getElementById('btn-ann-zoom-out'),
  annZoomRange: document.getElementById('ann-zoom-range'),
  annZoomInput: document.getElementById('ann-zoom-input'),
  btnAnnZoomIn: document.getElementById('btn-ann-zoom-in'),
  btnAnnZoomFit: document.getElementById('btn-ann-zoom-fit'),
  annZoomLabel: document.getElementById('ann-zoom-label'),
  annLaneLabels: document.getElementById('ann-lane-labels'),
  annTimelineScroll: document.getElementById('ann-timeline-scroll'),
  annTimelineStage: document.getElementById('ann-timeline-stage'),
  annWaveformCanvas: document.getElementById('ann-waveform-canvas'),
  annRuler: document.getElementById('ann-ruler'),
  annLanes: document.getElementById('ann-lanes'),
  annPlayhead: document.getElementById('ann-playhead'),
  annMarkRegion: document.getElementById('ann-mark-region'),
  annTurnSearch: document.getElementById('ann-turn-search'),
  btnAnnLoopSelected: document.getElementById('btn-ann-loop-selected'),
  btnAnnSplit: document.getElementById('btn-ann-split'),
  btnAnnReassign: document.getElementById('btn-ann-reassign'),
  btnAnnDeleteTurn: document.getElementById('btn-ann-delete-turn'),
  annTurnsBody: document.getElementById('ann-turns-body'),
  btnAnnRefreshResults: document.getElementById('btn-ann-refresh-results'),
  annResultList: document.getElementById('ann-result-list'),
  annCollarInput: document.getElementById('ann-collar-input'),
  annSkipOverlap: document.getElementById('ann-skip-overlap'),
  btnAnnEvaluate: document.getElementById('btn-ann-evaluate'),
  btnAnnDownloadReport: document.getElementById('btn-ann-download-report'),
  annEvalResults: document.getElementById('ann-eval-results'),

  // Speaker Purity Workbench
  purityResultList: document.getElementById('purity-result-list'),
  purityResultSelectionSummary: document.getElementById('purity-result-selection-summary'),
  btnPurityRefreshResults: document.getElementById('btn-purity-refresh-results'),
  purityCandidateSpeaker: document.getElementById('purity-candidate-speaker'),
  purityCandidateMinDuration: document.getElementById('purity-candidate-min-duration'),
  purityCandidateMaxDuration: document.getElementById('purity-candidate-max-duration'),
  purityCandidateOverlap: document.getElementById('purity-candidate-overlap'),
  purityCandidateVerification: document.getElementById('purity-candidate-verification'),
  purityInputSelect: document.getElementById('purity-input-select'),
  btnPurityBrowseLibrary: document.getElementById('btn-purity-browse-library'),
  purityAudioMetaChip: document.getElementById('purity-audio-meta-chip'),
  purityInputPreviewPill: document.getElementById('purity-input-preview-pill'),
  btnPurityPreviewInput: document.getElementById('btn-purity-preview-input'),
  purityTrackTitleText: document.getElementById('purity-track-title-text'),
  purityTrackSpecChip: document.getElementById('purity-track-spec-chip'),
  purityDiarStatusBox: document.getElementById('purity-diar-status-box'),
  purityDiarTurnsChip: document.getElementById('purity-diar-turns-chip'),
  purityDiarDesc: document.getElementById('purity-diar-desc'),
  purityProfileSelect: document.getElementById('purity-profile-select'),
  btnPurityRefreshProfiles: document.getElementById('btn-purity-refresh-profiles'),
  purityOverlapConfig: document.getElementById('purity-overlap-config'),
  purityOverlapStatusBadge: document.getElementById('purity-overlap-status-badge'),
  purityOverlapBackend: document.getElementById('purity-overlap-backend'),
  purityOverlapModel: document.getElementById('purity-overlap-model'),
  purityOverlapModelLabel: document.getElementById('purity-overlap-model-label'),
  purityOverlapVibevoiceModel: document.getElementById('purity-overlap-vibevoice-model'),
  purityVibevoiceModelHint: document.getElementById('purity-vibevoice-model-hint'),
  purityOverlapEndpointField: document.getElementById('purity-overlap-endpoint-field'),
  purityOverlapEndpoint: document.getElementById('purity-overlap-endpoint'),
  purityOverlapApiKey: document.getElementById('purity-overlap-api-key'),
  purityOverlapKeyStatus: document.getElementById('purity-overlap-key-status'),
  btnTogglePurityOverlapKey: document.getElementById('btn-toggle-purity-overlap-key'),
  purityOverlapTimeoutField: document.getElementById('purity-overlap-timeout-field'),
  purityOverlapTimeout: document.getElementById('purity-overlap-timeout'),
  purityOverlapMaxTokensLabel: document.getElementById('purity-overlap-max-tokens-label'),
  purityOverlapMaxTokens: document.getElementById('purity-overlap-max-tokens'),
  purityOverlapFailurePolicy: document.getElementById('purity-overlap-failure-policy'),
  purityOverlapPromptLabel: document.getElementById('purity-overlap-prompt-label'),
  purityOverlapPromptStatus: document.getElementById('purity-overlap-prompt-status'),
  purityOverlapPrompt: document.getElementById('purity-overlap-prompt'),
  purityOverlapPromptField: document.getElementById('purity-overlap-prompt-field'),
  purityOverlapKeyField: document.getElementById('purity-overlap-key-field'),
  purityVibevoiceSecondaryField: document.getElementById('purity-vibevoice-secondary-field'),
  purityVibevoiceSecondary: document.getElementById('purity-vibevoice-secondary'),
  purityVibevoiceBatchField: document.getElementById('purity-vibevoice-batch-field'),
  purityVibevoiceBatchSize: document.getElementById('purity-vibevoice-batch-size'),
  btnResetPurityOverlap: document.getElementById('btn-reset-purity-overlap'),
  btnPurityCheckVerifier: document.getElementById('btn-purity-check-verifier'),
  purityVerifierStatusMsg: document.getElementById('purity-verifier-status-msg'),
  purityVibevoiceDeviceField: document.getElementById('purity-vibevoice-device-field'),
  purityVibevoiceHfField: document.getElementById('purity-vibevoice-hf-field'),
  purityDeviceSelect: document.getElementById('purity-device-select'),
  purityHfTokenInput: document.getElementById('purity-hf-token-input'),
  btnTogglePurityHfVis: document.getElementById('btn-toggle-purity-hf-vis'),
  btnRunPurity: document.getElementById('btn-run-purity'),
  btnRunPurityManual: document.getElementById('btn-run-purity-manual'),
  btnPurityReset: document.getElementById('btn-purity-reset'),
  purityTaskProgressBox: document.getElementById('purity-task-progress-box'),
  purityTaskTimer: document.getElementById('purity-task-timer'),
  purityProgressBar: document.getElementById('purity-progress-bar'),
  purityTaskStatusText: document.getElementById('purity-task-status-text'),
  purityEmptyPlaceholder: document.getElementById('purity-empty-placeholder'),
  purityResultsWrapper: document.getElementById('purity-results-wrapper'),
  purityProfileBadge: document.getElementById('purity-profile-badge'),
  purityResultsTitle: document.getElementById('purity-results-title'),
  purityResultsMeta: document.getElementById('purity-results-meta'),
  purityMetricPassCount: document.getElementById('purity-metric-pass-count'),
  purityMetricPassPct: document.getElementById('purity-metric-pass-pct'),
  purityMetricPassDuration: document.getElementById('purity-metric-pass-duration'),
  purityMetricTotalDuration: document.getElementById('purity-metric-total-duration'),
  purityReasonsPills: document.getElementById('purity-reasons-pills'),
  purityMetricLlmChecked: document.getElementById('purity-metric-llm-checked'),
  purityMetricLlmDetail: document.getElementById('purity-metric-llm-detail'),
  purityErrorBanner: document.getElementById('purity-error-banner'),
  purityTableBody: document.getElementById('purity-table-body'),
  purityCountAll: document.getElementById('purity-count-all'),
  purityCountPass: document.getElementById('purity-count-pass'),
  purityCountReject: document.getElementById('purity-count-reject'),
  purityCountUncertain: document.getElementById('purity-count-uncertain'),
  purityCountError: document.getElementById('purity-count-error'),
  purityFooterSelectionInfo: document.getElementById('purity-footer-selection-info'),

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
  evalNotesInput: document.getElementById('eval-notes-input'),
  btnSaveEvaluation: document.getElementById('btn-save-evaluation'),
  btnCopyEvalNote: document.getElementById('btn-copy-eval-note'),

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
  libraryModalTitle: document.getElementById('library-modal-title'),
  libraryModalSubtitle: document.getElementById('library-modal-subtitle'),
  libraryModalSort: document.getElementById('library-modal-sort'),
  libraryModalSelectAll: document.getElementById('library-modal-select-all'),
  btnBulkDeleteLibraryModal: document.getElementById('btn-bulk-delete-library-modal'),
  btnRefreshLibraryModal: document.getElementById('btn-refresh-library-modal'),
  tabLibrarySearch: document.getElementById('tab-library-search'),
  tabLibraryCategories: document.getElementById('tab-library-categories'),
  tabLibraryDataset: document.getElementById('tab-library-dataset'),
  tabLibraryChannel: document.getElementById('tab-library-channel'),
  tabLibrarySpeaker: document.getElementById('tab-library-speaker'),
  tabLibraryVerification: document.getElementById('tab-library-verification'),
  tabLibraryFormat: document.getElementById('tab-library-format'),
  tabLibrarySort: document.getElementById('tab-library-sort'),
  tabLibrarySelectAll: document.getElementById('tab-library-select-all'),
  tabLibraryCount: document.getElementById('tab-library-count'),
  btnBulkDeleteLibrary: document.getElementById('btn-bulk-delete-library'),
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
  if (isNaN(seconds) || seconds < 0) return "00:00.000";
  const milliseconds = Math.round(seconds * 1000);
  const m = Math.floor(milliseconds / 60000);
  const s = Math.floor((milliseconds % 60000) / 1000);
  const fraction = milliseconds % 1000;
  return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}.${fraction.toString().padStart(3, '0')}`;
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
  return seconds.toFixed(3);
}

function formatBytes(bytes) {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
}

function parseJsonText(text) {
  let cleaned = String(text || '').replace(/^\uFEFF/, '').trim();
  if (/^\d{3}\s*[\[{]/.test(cleaned)) {
    cleaned = cleaned.slice(3).trim();
  }
  try {
    return JSON.parse(cleaned);
  } catch (err) {
    const match = /position\s+(\d+)/i.exec(err.message || '');
    if (match) {
      const pos = Number(match[1]);
      if (pos > 0) {
        try {
          const parsed = JSON.parse(cleaned.slice(0, pos));
          if (parsed && typeof parsed === 'object') return parsed;
        } catch (_) {}
      }
    }
    const objStart = cleaned.indexOf('{');
    const arrStart = cleaned.indexOf('[');
    const start = [objStart, arrStart].filter(index => index >= 0).sort((a, b) => a - b)[0];
    if (start > 0) {
      try {
        const parsed = JSON.parse(cleaned.slice(start));
        if (parsed && typeof parsed === 'object') return parsed;
      } catch (_) {}
    }
    throw err;
  }
}

async function parseJsonResponse(res) {
  const text = await res.text();
  let data;
  try {
    data = parseJsonText(text);
  } catch (_) {
    if (res.status === 404) {
      throw new Error(`Endpoint not found (HTTP 404). Please ensure the backend server was restarted with the latest routes!`);
    }
    throw new Error(`Server returned HTTP ${res.status}: ${text.substring(0, 120) || res.statusText}`);
  }
  if (!res.ok) {
    const msg = (data && typeof data === 'object' && (data.error || data.message || data.detail)) || `Request failed with status ${res.status}`;
    throw new Error(msg);
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
  setSelectValue(el.annSpeedSelect);
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
  turnPreviewAudio.playbackRate = parsedRate;
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
      turnPreviewAudio.volume = state.player.volume;
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
        turnPreviewAudio.volume = 0;
        if (el.volumeSlider) el.volumeSlider.value = 0;
      } else {
        const restoredVolume = state.player.volume || 1.0;
        if (el.audio) el.audio.volume = restoredVolume;
        if (auditionAudio) auditionAudio.volume = restoredVolume;
        turnPreviewAudio.volume = restoredVolume;
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
      updateAnnotationPlayhead(el.audio.currentTime || 0);
      startDiarPlaybackWatch();
    });
    el.audio.addEventListener('pause', () => {
      if (activeTurnPreviewKey) return;
      if (!isAuditionPlaybackActive()) setPlayingUI(false);
      updateAnnotationPlayhead(el.audio.currentTime || 0);
      stopDiarPlaybackWatch();
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

  if (isModalOpen()) {
    if (e.code === 'Escape') {
      e.preventDefault();
      closeAllModals();
    }
    return;
  }

  if (state.activeTab === 'tab-annotation' && state.annotation.current) {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'z') {
      e.preventDefault();
      if (e.shiftKey) redoAnnotation();
      else undoAnnotation();
      return;
    }
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'y') {
      e.preventDefault();
      redoAnnotation();
      return;
    }
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 's') {
      e.preventDefault();
      saveAnnotationNow();
      return;
    }
    if (e.code === 'ArrowLeft' || e.code === 'ArrowRight') {
      e.preventDefault();
      const direction = e.code === 'ArrowLeft' ? -1 : 1;
      const step = e.altKey ? 0.01 : (e.shiftKey ? 1 : 0.1);
      seekRelative(direction * step);
      return;
    }
    if (/^[1-9]$/.test(e.key)) {
      const speaker = state.annotation.speakers[Number(e.key) - 1];
      if (speaker) {
        e.preventDefault();
        state.annotation.activeSpeakerId = speaker.speaker_id;
        renderAnnotationSpeakers();
        updateAnnotationMarks();
      }
      return;
    }
    if (e.key.toLowerCase() === 'i') {
      e.preventDefault();
      state.annotation.markIn = snapAnnotationTime(el.audio?.currentTime || 0);
      state.annotation.markOut = null;
      updateAnnotationMarks();
      return;
    }
    if (e.key.toLowerCase() === 'o') {
      e.preventDefault();
      state.annotation.markOut = snapAnnotationTime(el.audio?.currentTime || 0);
      updateAnnotationMarks();
      createAnnotationTurn();
      return;
    }
    if (e.key.toLowerCase() === 's' && !e.metaKey && !e.ctrlKey) {
      e.preventDefault();
      splitSelectedAnnotationTurn();
      return;
    }
    if (e.key === 'Delete' || e.key === 'Backspace') {
      e.preventDefault();
      deleteSelectedAnnotationTurn();
      return;
    }
  }

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

  // Diarization Studio: zoom hotkeys only
  if (state.activeTab === 'tab-diarization') {
    if (e.key === 'z' || e.key === 'Z') {
      e.preventDefault();
      if (e.shiftKey) {
        setDiarZoom(state.diarization.zoom / 1.5);
      } else {
        setDiarZoom(state.diarization.zoom * 1.5);
      }
      return;
    }
    if (e.key === '0') {
      e.preventDefault();
      setDiarZoom(1.0);
      return;
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

  const diarPlayText = el.btnDiarPlayToggle?.querySelector('span');
  if (diarPlayText) {
    diarPlayText.textContent = isPlaying ? 'Pause' : 'Play All';
  }
}

function togglePlayPause() {
  if (activeTurnPreviewKey) {
    stopTurnPreview();
    return;
  }
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
  stopTurnPreview();
  clearRangePreview();
  if (prepareDiarPlaybackGate() === false) {
    showToast("No more speaker segments", "info");
    return Promise.resolve();
  }
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
  if (!rangePreviewSeeking) {
    stopTurnPreview();
    clearRangePreview();
  } else {
    stopRangePreviewMonitor();
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

function stopRangePreviewMonitor() {
  if (workspacePreviewRaf) {
    cancelAnimationFrame(workspacePreviewRaf);
    workspacePreviewRaf = 0;
  }
}

function clearRangePreview() {
  stopRangePreviewMonitor();
  rangePreviewGeneration += 1;
  state.player.previewEnd = null;
}

function previewWorkspaceRangeOnce(start, end) {
  stopRangePreviewMonitor();
  const startSec = Number(start) || 0;
  const endSec = Number(end) || 0;
  if (!el.audio || !(endSec > startSec)) return;
  const generation = ++rangePreviewGeneration;

  const begin = () => {
    if (generation !== rangePreviewGeneration) return;
    rangePreviewSeeking = true;
    seekTo(startSec);
    rangePreviewSeeking = false;
    state.player.previewEnd = endSec;
    const stopAt = Math.max(startSec, endSec - 0.01);
    const monitor = () => {
      workspacePreviewRaf = 0;
      if (generation !== rangePreviewGeneration) return;
      if (state.player.previewEnd === null || !el.audio || el.audio.paused) return;
      if (el.audio.currentTime >= stopAt) {
        el.audio.pause();
        el.audio.currentTime = endSec;
        state.player.previewEnd = null;
        updatePlayheadPosition(endSec);
        return;
      }
      workspacePreviewRaf = requestAnimationFrame(monitor);
    };
    el.audio.play()
      .then(() => {
        if (generation !== rangePreviewGeneration) return;
        workspacePreviewRaf = requestAnimationFrame(monitor);
      })
      .catch(error => {
        if (generation !== rangePreviewGeneration) return;
        clearRangePreview();
        showToast(`Could not preview range: ${error.message}`, 'error');
      });
  };

  if (el.audio.readyState < 1) {
    el.audio.addEventListener('loadedmetadata', begin, { once: true });
    return;
  }
  begin();
}

function onLoadedMetadata() {
  if (isAuditionPlaybackActive()) return;
  state.player.duration = el.audio.duration || (state.activeAudio ? state.activeAudio.duration_s : 0);
  el.timeTotal.textContent = formatTime(state.player.duration);
  renderWorkspaceRuler();
}

function onTimeUpdate() {
  if (isAuditionPlaybackActive()) return;
  let cur = el.audio.currentTime;
  const dur = state.player.duration || 1;

  // Backup stop if the animation-frame monitor is not running
  if (state.player.previewEnd !== null && cur >= state.player.previewEnd) {
    const end = state.player.previewEnd;
    el.audio.pause();
    cur = end;
    el.audio.currentTime = end;
  }
  state.player.currentTime = cur;

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
  updateAnnotationPlayhead(cur);
}

function onEnded() {
  if (isAuditionPlaybackActive()) return;
  if (state.activeTab === 'tab-diarization' && state.diarization.autoAdvance && state.diarization.turns.length && state.player.loop) {
    const first = findNextAudibleTurn(-1, { wrap: true });
    if (first) {
      jumpToDiarTurn(first);
      playCurrentAudio();
      return;
    }
  }
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
    state.waveform.controller?.abort();
    state.waveform.requestId += 1;
    state.waveform.specController?.abort();
    state.waveform.specRequestId = (state.waveform.specRequestId || 0) + 1;
    if (state.waveform.specUrl) {
      URL.revokeObjectURL(state.waveform.specUrl);
      state.waveform.specUrl = null;
    }
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
    if (el.metaChannel) el.metaChannel.textContent = meta.channel_name || fullItem.channel_name || 'Unassigned';
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
  const duration = state.activeAudio?.duration_s || 0;
  state.waveform.audioId = audioId;
  state.waveform.start = 0;
  state.waveform.end = duration;
  state.waveform.data = null;
  state.waveform.error = '';
  state.zoom = 1;
  setWorkspaceViewport(0, duration, { request: false });
  return requestWaveformWindow({
    audioId,
    canvas: el.waveformCanvas,
    start: state.waveform.start,
    end: state.waveform.end,
    view: state.waveform,
    draw: renderWaveform,
  });
}

// ==================== WAVEFORM CANVAS RENDERER ====================

function drawWaveformEnvelope(canvas, view, fallbackMessage = 'No waveform data loaded') {
  if (!canvas) return;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, rect.width || canvas.parentElement?.clientWidth || 1);
  const height = Math.max(1, rect.height || canvas.parentElement?.clientHeight || 1);
  const dpr = window.devicePixelRatio || 1;
  const pixelWidth = Math.max(1, Math.round(width * dpr));
  const pixelHeight = Math.max(1, Math.round(height * dpr));
  if (canvas.width !== pixelWidth) canvas.width = pixelWidth;
  if (canvas.height !== pixelHeight) canvas.height = pixelHeight;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  const data = view.data;
  const channels = data?.channels || [];
  if (!channels.length || !channels[0]?.min?.length) {
    ctx.fillStyle = "rgba(148, 163, 184, 0.6)";
    ctx.font = "12px JetBrains Mono";
    ctx.textAlign = "center";
    ctx.fillText(view.loading ? 'Loading waveform…' : (view.error || fallbackMessage), width / 2, height / 2);
    return;
  }
  const isLight = document.documentElement.getAttribute('data-theme') === 'light';
  ctx.strokeStyle = isLight ? "rgba(148, 163, 184, 0.4)" : "rgba(255, 255, 255, 0.1)";
  ctx.lineWidth = 1;
  const laneHeight = height / channels.length;
  channels.forEach((channel, channelIndex) => {
    const laneTop = laneHeight * channelIndex;
    const centerY = laneTop + laneHeight / 2;
    const amplitudeHeight = Math.max(1, laneHeight / 2 - 5);
    ctx.beginPath();
    ctx.moveTo(0, centerY);
    ctx.lineTo(width, centerY);
    ctx.stroke();
    if (channelIndex) {
      ctx.beginPath();
      ctx.moveTo(0, laneTop);
      ctx.lineTo(width, laneTop);
      ctx.stroke();
    }
    const minima = channel.min;
    const maxima = channel.max;
    const count = Math.min(minima.length, maxima.length);
    const isSampleTrace = count === data.frame_count;
    ctx.strokeStyle = isLight ? 'hsl(205, 88%, 43%)' : 'hsl(188, 86%, 58%)';
    ctx.lineWidth = isSampleTrace ? 1.25 : Math.max(1, width / count);
    ctx.beginPath();
    if (isSampleTrace) {
      for (let index = 0; index < count; index += 1) {
        const x = count > 1 ? index / (count - 1) * width : width / 2;
        const y = centerY - Math.max(-1, Math.min(1, maxima[index] || 0)) * amplitudeHeight;
        if (index) ctx.lineTo(x, y); else ctx.moveTo(x, y);
      }
    } else {
      for (let index = 0; index < count; index += 1) {
        const x = (index + 0.5) / count * width;
        ctx.moveTo(x, centerY - Math.max(-1, Math.min(1, maxima[index] || 0)) * amplitudeHeight);
        ctx.lineTo(x, centerY - Math.max(-1, Math.min(1, minima[index] || 0)) * amplitudeHeight);
      }
    }
    ctx.stroke();
    if (channels.length > 1) {
      ctx.fillStyle = 'rgba(148,163,184,.75)';
      ctx.font = '9px JetBrains Mono';
      ctx.textAlign = 'left';
      ctx.fillText(`CH ${channelIndex + 1}`, 5, laneTop + 11);
    }
  });
}

async function requestWaveformWindow({ audioId, canvas, start, end, view, draw }) {
  if (!audioId || !canvas || !(end > start)) return;
  view.controller?.abort();
  const controller = new AbortController();
  const requestId = ++view.requestId;
  view.controller = controller;
  view.loading = true;
  view.error = '';
  view.data = null;
  draw();
  const width = Math.max(1, canvas.getBoundingClientRect().width || canvas.parentElement?.clientWidth || 1);
  const bins = Math.min(8192, Math.max(1, Math.ceil(width * (window.devicePixelRatio || 1))));
  const query = new URLSearchParams({ start_s: String(start), end_s: String(end), bins: String(bins) });
  try {
    const response = await fetch(`/api/audio/${encodeURIComponent(audioId)}/waveform?${query}`, { signal: controller.signal });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Failed to load waveform');
    if (requestId !== view.requestId) return;
    view.data = data;
  } catch (error) {
    if (error.name === 'AbortError' || requestId !== view.requestId) return;
    view.error = error.message || 'Waveform unavailable';
  } finally {
    if (requestId === view.requestId) {
      view.loading = false;
      draw();
    }
  }
}

function renderWaveform() {
  drawWaveformEnvelope(el.waveformCanvas, state.waveform);
}

function workspaceDuration() {
  return Math.max(0, state.activeAudio?.duration_s || state.player.duration || 0);
}

function workspaceTimeToX(time) {
  const span = state.waveform.end - state.waveform.start;
  const width = el.waveformViewport?.clientWidth || 1;
  return span > 0 ? (time - state.waveform.start) / span * width : 0;
}

function workspaceXToTime(x) {
  const width = el.waveformViewport?.clientWidth || 1;
  return state.waveform.start + Math.max(0, Math.min(width, x)) / width * (state.waveform.end - state.waveform.start);
}

function workspaceMaxZoom() {
  const duration = workspaceDuration();
  const sampleRate = state.waveform.data?.sample_rate || state.activeAudio?.sample_rate || 44100;
  const width = Math.max(1, el.waveformViewport?.clientWidth || 1);
  return Math.max(1, duration * sampleRate / width);
}

function renderWorkspaceRuler() {
  if (!el.timeRuler) return;
  const start = state.waveform.start;
  const end = state.waveform.end;
  const span = end - start;
  const width = el.timeRuler.clientWidth || 600;
  if (!(span > 0)) {
    el.timeRuler.innerHTML = '';
    return;
  }
  const target = span / Math.max(2, width / 90);
  const steps = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1800];
  const step = steps.find(value => value >= target) || Math.ceil(target / 1800) * 1800;
  const first = Math.ceil(start / step) * step;
  const ticks = [];
  for (let time = first; time <= end + step * 1e-6; time += step) {
    const left = (time - start) / span * 100;
    const label = step < 1 ? `${time.toFixed(step < 0.01 ? 3 : 2)}s` : formatTime(time);
    ticks.push(`<span class="ruler-tick" style="left:${left}%">${label}</span>`);
  }
  el.timeRuler.innerHTML = ticks.join('');
}

function updateWorkspaceScrollbar() {
  if (!el.waveformScrollbar) return;
  const duration = workspaceDuration();
  const span = state.waveform.end - state.waveform.start;
  const travel = Math.max(0, duration - span);
  el.waveformScrollbar.disabled = travel <= 1e-12;
  el.waveformScrollbar.value = travel > 0 ? String(Math.round(state.waveform.start / travel * 1000)) : '0';
}

function requestWorkspaceWaveform() {
  clearTimeout(state.waveform.requestTimer);
  state.waveform.requestTimer = setTimeout(() => {
    requestWaveformWindow({
      audioId: state.waveform.audioId,
      canvas: el.waveformCanvas,
      start: state.waveform.start,
      end: state.waveform.end,
      view: state.waveform,
      draw: renderWaveform,
    });
  }, 45);
}

function setWorkspaceViewport(start, end, { request = true } = {}) {
  const duration = workspaceDuration();
  if (!(duration > 0)) return;
  const sampleRate = Math.max(1, state.waveform.data?.sample_rate || state.activeAudio?.sample_rate || 44100);
  const frameS = 1 / sampleRate;
  const span = Math.max(frameS, Math.min(duration, end - start));
  const boundedStart = Math.max(0, Math.min(start, duration - span));
  const totalFrames = Math.max(1, Math.round(duration * sampleRate));
  const startFrame = Math.max(0, Math.min(totalFrames - 1, Math.floor(boundedStart * sampleRate)));
  const endFrame = Math.max(startFrame + 1, Math.min(totalFrames, Math.ceil((boundedStart + span) * sampleRate)));
  state.waveform.start = startFrame / sampleRate;
  state.waveform.end = endFrame / sampleRate;
  if (state.waveform.data && (
    Math.abs(state.waveform.data.start_s - state.waveform.start) > frameS / 2
    || Math.abs(state.waveform.data.end_s - state.waveform.end) > frameS / 2
  )) state.waveform.data = null;
  state.zoom = duration / (state.waveform.end - state.waveform.start);
  if (el.zoomLabel) el.zoomLabel.textContent = `${Math.round(state.zoom * 100)}%`;
  if (el.wsZoomInput && document.activeElement !== el.wsZoomInput) {
    el.wsZoomInput.value = state.zoom < 10 ? state.zoom.toFixed(1) : state.zoom.toFixed(0);
  }
  renderWaveform();
  renderWorkspaceRuler();
  updateWorkspaceScrollbar();
  updatePlayheadPosition(state.player.currentTime);
  updateSelectionOverlay();
  if (request) requestWorkspaceWaveform();
  if (!el.spectrogramPanel?.classList.contains('hidden')) scheduleSpectrogramImage();
}

function updatePlayheadPosition(currentTime) {
  if (!workspaceDuration() || !el.waveformViewport) return;
  if (!el.audio?.paused && !state.waveform.editing && !state.waveform.panning) {
    const span = state.waveform.end - state.waveform.start;
    if (currentTime > state.waveform.end) setWorkspaceViewport(currentTime - span * 0.1, currentTime + span * 0.9);
    else if (currentTime < state.waveform.start) setWorkspaceViewport(currentTime - span * 0.9, currentTime + span * 0.1);
  }
  const pos = workspaceTimeToX(currentTime);
  const visible = pos >= 0 && pos <= el.waveformViewport.clientWidth;
  el.playheadLine.style.opacity = visible ? '1' : '0';
  el.playheadLine.style.transform = `translateX(${pos}px)`;
  if (el.specPlayheadLine) {
    const specWidth = el.specImageWrapper?.clientWidth || el.waveformViewport.clientWidth;
    const specPos = (currentTime - state.waveform.start) / (state.waveform.end - state.waveform.start) * specWidth;
    el.specPlayheadLine.style.opacity = visible ? '1' : '0';
    el.specPlayheadLine.style.left = `${specPos}px`;
  }
}

function clearSelection() {
  state.selection.active = false;
  state.selection.start = 0;
  state.selection.end = 0;
  if (el.selectionOverlay) el.selectionOverlay.classList.add('hidden');
  if (el.selectionActionsBar) el.selectionActionsBar.style.display = 'none';
  if (el.selectionHelper) {
    el.selectionHelper.classList.remove('has-selection');
    el.selectionHelper.innerHTML = '<span class="selection-helper-icon">↔</span><span><strong>Click to seek or drag to select.</strong> Zoom with the controls or Ctrl/Cmd-wheel; pan horizontally with the scrollbar, trackpad, or Shift-wheel.</span>';
  }
  updateSelectionOverlay();
}

function initWaveformInteractions() {
  const viewport = el.waveformViewport;
  if (viewport) {
    let drag = null;
    viewport.addEventListener('pointerdown', event => {
      if (!state.activeAudio || event.button !== 0) return;
      const rect = viewport.getBoundingClientRect();
      const x = Math.max(0, Math.min(event.clientX - rect.left, rect.width));
      drag = { pointerId: event.pointerId, mode: event.target.closest('.selection-handle')?.dataset.handle || 'new', startX: x, startTime: workspaceXToTime(x), moved: false };
      state.waveform.editing = true;
      viewport.setPointerCapture(event.pointerId);
      event.preventDefault();
    });
    viewport.addEventListener('pointermove', event => {
      const rect = viewport.getBoundingClientRect();
      const x = Math.max(0, Math.min(event.clientX - rect.left, rect.width));
      if (el.timeTooltip) {
        el.timeTooltip.classList.remove('hidden');
        el.timeTooltip.textContent = formatTimePrecise(workspaceXToTime(x));
        el.timeTooltip.style.left = `${Math.min(x, rect.width - 60)}px`;
      }
      if (!drag || event.pointerId !== drag.pointerId) return;
      if (Math.abs(x - drag.startX) >= 3) drag.moved = true;
      if (!drag.moved && drag.mode === 'new') return;
      const time = workspaceXToTime(x);
      const frameS = 1 / Math.max(1, state.waveform.data?.sample_rate || state.activeAudio.sample_rate || 44100);
      if (drag.mode === 'start') state.selection.start = Math.min(time, state.selection.end - frameS);
      else if (drag.mode === 'end') state.selection.end = Math.max(time, state.selection.start + frameS);
      else {
        state.selection.start = Math.min(drag.startTime, time);
        state.selection.end = Math.max(drag.startTime, time);
        state.selection.active = true;
      }
      state.selection.start = Math.max(0, state.selection.start);
      state.selection.end = Math.min(workspaceDuration(), state.selection.end);
      updateSelectionOverlay();
    });
    const finishPointer = event => {
      if (!drag || event.pointerId !== drag.pointerId) return;
      if (!drag.moved && drag.mode === 'new') seekTo(drag.startTime);
      else if (state.selection.active) {
        if (el.selectionActionsBar) el.selectionActionsBar.style.display = 'flex';
        populateCutBoundsFromSelection();
        if (el.selectionHelper) {
          el.selectionHelper.classList.add('has-selection');
          el.selectionHelper.innerHTML = `<span class="selection-helper-icon">✓</span><span><strong>${formatTimePrecise(state.selection.end - state.selection.start)} selected.</strong> Drag either edge or edit millisecond-precision fields to refine it.</span>`;
        }
      }
      state.waveform.editing = false;
      if (viewport.hasPointerCapture(event.pointerId)) viewport.releasePointerCapture(event.pointerId);
      drag = null;
    };
    viewport.addEventListener('pointerup', finishPointer);
    viewport.addEventListener('pointercancel', finishPointer);
    viewport.addEventListener('pointerleave', () => { if (!drag) el.timeTooltip?.classList.add('hidden'); });
    viewport.addEventListener('wheel', event => {
      if (!state.activeAudio) return;
      if (event.ctrlKey || event.metaKey) {
        event.preventDefault();
        const rect = viewport.getBoundingClientRect();
        const fraction = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
        setZoom(state.zoom * Math.exp(-event.deltaY * 0.002), workspaceXToTime(fraction * rect.width), fraction);
      } else if (event.shiftKey || Math.abs(event.deltaX) > Math.abs(event.deltaY)) {
        event.preventDefault();
        const delta = event.shiftKey ? event.deltaY : event.deltaX;
        const span = state.waveform.end - state.waveform.start;
        state.waveform.panning = true;
        setWorkspaceViewport(state.waveform.start + delta / viewport.clientWidth * span, state.waveform.end + delta / viewport.clientWidth * span);
        clearTimeout(state.waveform.panTimer);
        state.waveform.panTimer = setTimeout(() => { state.waveform.panning = false; }, 140);
      }
    }, { passive: false });
  }

  // Audition Selection button
  if (el.btnAuditionSelection) {
    el.btnAuditionSelection.addEventListener('click', () => {
      if (!state.activeAudio || !state.selection.active) return;
      previewWorkspaceRangeOnce(state.selection.start, state.selection.end);
      showToast(`Previewing once: ${state.selection.start.toFixed(3)}s to ${state.selection.end.toFixed(3)}s`, "info");
    });
  }

  // Clear Selection button
  if (el.btnClearSelection) {
    el.btnClearSelection.addEventListener('click', clearSelection);
  }

  // Zoom controls
  if (el.btnZoomIn) el.btnZoomIn.addEventListener('click', () => setZoom(state.zoom * 1.5));
  if (el.btnZoomOut) el.btnZoomOut.addEventListener('click', () => setZoom(state.zoom / 1.5));
  if (el.btnResetZoom) el.btnResetZoom.addEventListener('click', () => setZoom(1.0));
  if (el.wsZoomInput) {
    const handleWsZoom = (e) => {
      const val = parseFloat(e.target.value);
      if (!isNaN(val) && val > 0) setZoom(val);
    };
    el.wsZoomInput.addEventListener('input', handleWsZoom);
    el.wsZoomInput.addEventListener('change', handleWsZoom);
    el.wsZoomInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        handleWsZoom(e);
        el.wsZoomInput.blur();
      }
    });
  }

  // Spectrogram Toggle
  if (el.btnToggleSpec) el.btnToggleSpec.addEventListener('click', toggleSpectrogramPanel);
  if (el.btnRefreshSpec) el.btnRefreshSpec.addEventListener('click', loadSpectrogramImage);
  el.waveformScrollbar?.addEventListener('input', () => {
    const duration = workspaceDuration();
    const span = state.waveform.end - state.waveform.start;
    const start = (duration - span) * Number(el.waveformScrollbar.value) / 1000;
    state.waveform.panning = true;
    setWorkspaceViewport(start, start + span);
    clearTimeout(state.waveform.panTimer);
    state.waveform.panTimer = setTimeout(() => { state.waveform.panning = false; }, 140);
  });

  window.addEventListener('resize', () => {
    setWorkspaceViewport(state.waveform.start, state.waveform.end);
    if (state.activeTab === 'tab-diarization') {
      setDiarZoom(state.diarization.zoom || 1.0);
    }
  });
}

function setZoom(newZoom, anchorTime = null, anchorFraction = null) {
  const parsedZoom = parseFloat(newZoom);
  const zoom = Math.min(workspaceMaxZoom(), Math.max(1, Number.isFinite(parsedZoom) ? parsedZoom : 1));
  const duration = workspaceDuration();
  if (!(duration > 0)) return;
  const oldSpan = state.waveform.end - state.waveform.start || duration;
  const playhead = state.player.currentTime;
  const defaultAnchor = playhead >= state.waveform.start && playhead <= state.waveform.end
    ? playhead
    : state.waveform.start + oldSpan / 2;
  const anchor = Number.isFinite(anchorTime) ? anchorTime : defaultAnchor;
  const fraction = Number.isFinite(anchorFraction) ? anchorFraction : (anchor - state.waveform.start) / oldSpan;
  const span = duration / zoom;
  setWorkspaceViewport(anchor - fraction * span, anchor + (1 - fraction) * span);
}

function updateSelectionOverlay() {
  if (!el.selectionOverlay) return;
  const visibleStart = Math.max(state.selection.start, state.waveform.start);
  const visibleEnd = Math.min(state.selection.end, state.waveform.end);
  const visible = state.selection.active && visibleEnd > visibleStart;
  el.selectionOverlay.classList.toggle('hidden', !visible);
  if (el.specSelectionOverlay) el.specSelectionOverlay.classList.toggle('hidden', !visible);
  if (!visible) return;
  const left = workspaceTimeToX(visibleStart);
  const right = workspaceTimeToX(visibleEnd);
  el.selectionOverlay.style.left = `${left}px`;
  el.selectionOverlay.style.width = `${right - left}px`;
  const leftHandle = el.selectionOverlay.querySelector('.handle-left');
  const rightHandle = el.selectionOverlay.querySelector('.handle-right');
  if (leftHandle) leftHandle.style.display = state.selection.start >= state.waveform.start ? '' : 'none';
  if (rightHandle) rightHandle.style.display = state.selection.end <= state.waveform.end ? '' : 'none';
  el.selectionRangeLabel.textContent = `${state.selection.start.toFixed(3)}s – ${state.selection.end.toFixed(3)}s (${(state.selection.end - state.selection.start).toFixed(3)}s)`;
  if (el.specSelectionOverlay) {
    const specWidth = el.specImageWrapper?.clientWidth || el.waveformViewport.clientWidth;
    const span = state.waveform.end - state.waveform.start;
    const specLeft = (visibleStart - state.waveform.start) / span * specWidth;
    const specRight = (visibleEnd - state.waveform.start) / span * specWidth;
    el.specSelectionOverlay.style.left = `${specLeft}px`;
    el.specSelectionOverlay.style.width = `${specRight - specLeft}px`;
  }
}

async function toggleSpectrogramPanel() {
  const isHidden = el.spectrogramPanel.classList.toggle('hidden');
  if (!isHidden && el.specImage.classList.contains('hidden')) {
    await loadSpectrogramImage();
  }
}

async function loadSpectrogramImage() {
  if (!state.activeAudio) return;
  state.waveform.specController?.abort();
  const controller = new AbortController();
  const requestId = (state.waveform.specRequestId || 0) + 1;
  state.waveform.specRequestId = requestId;
  state.waveform.specController = controller;
  el.specLoader.classList.remove('hidden');
  el.specLoader.textContent = 'Generating visible spectrogram…';
  el.specImage.classList.add('hidden');
  try {
    const width = Math.max(32, Math.round(el.specImage.parentElement.clientWidth * (window.devicePixelRatio || 1)));
    const height = Math.max(32, Math.round(180 * (window.devicePixelRatio || 1)));
    const query = new URLSearchParams({ start_s: String(state.waveform.start), end_s: String(state.waveform.end), width: String(Math.min(4096, width)), height: String(Math.min(2048, height)) });
    const response = await fetch(`/api/audio/${encodeURIComponent(state.activeAudio.id)}/spectrogram?${query}`, { signal: controller.signal });
    if (!response.ok) throw new Error(await response.text() || 'Failed to load spectrogram');
    const blob = await response.blob();
    if (requestId !== state.waveform.specRequestId) return;
    if (state.waveform.specUrl) URL.revokeObjectURL(state.waveform.specUrl);
    state.waveform.specUrl = URL.createObjectURL(blob);
    el.specImage.src = state.waveform.specUrl;
    el.specLoader.classList.add('hidden');
    el.specImage.classList.remove('hidden');
    updateSelectionOverlay();
    updatePlayheadPosition(state.player.currentTime);
  } catch (err) {
    if (err.name === 'AbortError' || requestId !== state.waveform.specRequestId) return;
    el.specLoader.textContent = "Failed to load spectrogram.";
    el.specLoader.classList.remove('hidden');
  }
}

function scheduleSpectrogramImage() {
  clearTimeout(state.waveform.specTimer);
  state.waveform.specTimer = setTimeout(loadSpectrogramImage, 240);
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
    updateSelectionOverlay();
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

    previewWorkspaceRangeOnce(range.start, range.effectiveEnd);
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
  function ytCrawlSampleRatePayload() {
    const raw = el.ytSampleRate?.value || "44100";
    return raw === "native" ? "native" : parseInt(raw, 10);
  }

  function updateYtIngestHint() {
    if (!el.ytIngestHint || !el.ytSampleRate) return;
    const raw = el.ytSampleRate.value;
    if (raw === "native") {
      el.ytIngestHint.textContent = "Downloads audio to mono WAV at the source/native sample rate.";
    } else if (raw === "16000") {
      el.ytIngestHint.textContent = "Downloads audio to mono WAV at 16 kHz (speech / ASR).";
    } else {
      el.ytIngestHint.textContent = "Downloads audio to mono WAV at 44.1 kHz.";
    }
  }

  if (el.ytSampleRate) {
    el.ytSampleRate.addEventListener("change", updateYtIngestHint);
    updateYtIngestHint();
  }

  el.btnYtDownload.addEventListener('click', async () => {
    const url = el.ytUrlInput.value.trim();
    if (!url) {
      showToast("Please enter a YouTube video URL", "error");
      return;
    }
    const sampleRate = ytCrawlSampleRatePayload();
    el.btnYtDownload.disabled = true;
    el.btnYtDownload.innerHTML = `<span class="dot dot-pulse"></span> Fetching...`;

    try {
      const res = await fetch("/api/audio/youtube", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url, sample_rate: sampleRate }),
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
      el.btnYtDownload.textContent = "Fetch Audio";
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
  el.btnBrowseLibrary.addEventListener('click', () => openLibraryModal('workspace'));
  const btnBrowseTop = document.getElementById('btn-browse-library-top');
  if (btnBrowseTop) btnBrowseTop.addEventListener('click', () => openLibraryModal('workspace'));
  el.btnCloseLibraryModal.addEventListener('click', () => {
    stopFilePreview();
    el.modalLibrary.classList.add('hidden');
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
    el.sepInputSelect.addEventListener('change', async () => {
      const audioId = el.sepInputSelect.value;
      if (audioId.startsWith('lib:')) {
        await loadLibraryFileTo(audioId.slice(4), 'separation');
        return;
      }
      renderSeparationChildren(audioId);
    });
  }

  if (el.btnSepBrowseLibrary) {
    el.btnSepBrowseLibrary.addEventListener('click', () => openLibraryModal('separation'));
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
  "hsl(188, 86%, 53%)",
  "hsl(158, 64%, 52%)",
  "hsl(38, 92%, 50%)",
  "hsl(348, 83%, 60%)",
  "hsl(270, 75%, 65%)",
  "hsl(205, 90%, 55%)",
  "hsl(84, 80%, 50%)",
  "hsl(22, 90%, 55%)",
];

function roundNum(value, decimals) {
  const factor = Math.pow(10, decimals);
  return Math.round(value * factor) / factor;
}

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
  if (key.includes("diarizen")) return "DiariZen Large s80-v2";
  if (key.includes("clustering") || key.includes("cluster")) return "NeMo Clustering";
  if (key.includes("3d") || key.includes("speakerlab") || key.includes("threed")) return "3D-Speaker";
  if (key.includes("3.1") || key.includes("pyannote_31") || key.includes("pyannote_3")) return "Pyannote 3.1";
  if (key.includes("community") || key.includes("pyannote_community")) return "Pyannote Community-1";
  if (key.includes("pyannote")) return "Pyannote Community-1";
  return modelTypeOrBackend || "Pyannote";
}

function syncDiarModelOptions(modelType) {
  const isPyannote = modelType && modelType.startsWith("pyannote");
  const isDiariZen = modelType === "diarizen" || modelType === "diarizen_large_s80_v2";
  const is3d = modelType === "3d_speaker" || modelType === "3d-speaker" || modelType === "threed_speaker";
  const isClustering = modelType === "clustering" || modelType === "nemo-clustering" || modelType === "nemo_clustering";
  const isSortformer = modelType === "sortformer";
  const activeCard = document.querySelector(`.model-card[data-diar-model="${modelType}"]`);
  if (el.diarBatchSize && el.diarBatchSize.dataset.modelType !== modelType) {
    el.diarBatchSize.value = activeCard?.dataset.defaultBatchSize || '1';
    el.diarBatchSize.dataset.modelType = modelType;
  }

  const hfGroup = document.getElementById("hf-token-group");
  const overlapCheck = document.getElementById("diar-3d-overlap");
  const needsHf = isPyannote || isDiariZen || (is3d && overlapCheck && overlapCheck.checked);
  if (hfGroup) {
    hfGroup.style.display = needsHf ? "" : "none";
  }
  const chunkGroup = document.getElementById("diar-3d-chunk-group");
  if (chunkGroup) {
    chunkGroup.style.display = is3d ? "" : "none";
  }
  const clusteringGroup = document.getElementById("diar-clustering-params-group");
  if (clusteringGroup) {
    clusteringGroup.style.display = isClustering ? "" : "none";
  }
  const sortformerGroup = document.getElementById("diar-sortformer-params-group");
  if (sortformerGroup) {
    sortformerGroup.style.display = isSortformer ? "" : "none";
  }

  const enrollmentSelect = document.getElementById('diar-enrollment-profile-select');
  const enrollmentSupport = document.getElementById('diar-enrollment-support');
  const supportsEnrollment = isSortformer;
  if (enrollmentSelect) {
    enrollmentSelect.disabled = !supportsEnrollment;
    if (!supportsEnrollment) enrollmentSelect.value = '';
  }
  if (enrollmentSupport) {
    enrollmentSupport.textContent = supportsEnrollment
      ? 'The selected clips are embedded with Sortformer’s TitaNet encoder before target-audio inference and anchor speaker assignment.'
      : 'This pipeline does not expose genuine pre-inference enrollment; choose NeMo Sortformer or run ordinary diarization.';
  }
}

function initDiarizationStudio() {
  if (el.btnToggleDiarSetup) {
    el.btnToggleDiarSetup.addEventListener('click', () => {
      const body = el.diarSetupBody;
      if (!body) return;
      const isCollapsed = body.style.display === 'none';
      if (isCollapsed) {
        body.style.display = '';
        if (el.diarSetupToggleIcon) el.diarSetupToggleIcon.textContent = '▲';
        if (el.diarSetupToggleText) el.diarSetupToggleText.textContent = 'Collapse';
      } else {
        body.style.display = 'none';
        if (el.diarSetupToggleIcon) el.diarSetupToggleIcon.textContent = '▼';
        if (el.diarSetupToggleText) el.diarSetupToggleText.textContent = 'Expand';
      }
    });
  }

  if (el.diarInputSelect) {
    el.diarInputSelect.addEventListener('change', async () => {
      const audioId = el.diarInputSelect.value;
      if (!audioId) {
        updateDiarInputMeta(null);
        renderDiarizationChildren(null);
        hideSavedDiarizationNotice();
        return;
      }
      if (audioId.startsWith('lib:')) {
        const filePath = audioId.slice(4);
        await loadLibraryFileTo(filePath, 'diarization');
        return;
      }
      openDiarizationAudio(audioId, { restoreHistory: true });
      loadSpeakerProfiles();
    });
  }

  if (el.btnDiarBrowseLibrary) {
    el.btnDiarBrowseLibrary.addEventListener('click', () => openLibraryModal('diarization'));
  }

  if (el.btnDiarPreviewInput) {
    el.btnDiarPreviewInput.addEventListener('click', () => {
      const audioId = el.diarInputSelect.value || state.activeAudio?.id;
      if (audioId) {
        loadAudioIntoPlayer(audioId, true);
      }
    });
  }

  el.diarModelCards.forEach(card => {
    card.addEventListener('click', () => {
      el.diarModelCards.forEach(c => c.classList.remove('active'));
      card.classList.add('active');
      syncDiarModelOptions(card.dataset.diarModel);
    });
  });
  const initiallyActive = document.querySelector('.model-card[data-diar-model].active');
  syncDiarModelOptions(initiallyActive ? initiallyActive.dataset.diarModel : "pyannote_community");

  const overlapCheck = document.getElementById("diar-3d-overlap");
  if (overlapCheck) {
    overlapCheck.addEventListener('change', () => {
      const activeCard = document.querySelector('.model-card[data-diar-model].active');
      syncDiarModelOptions(activeCard ? activeCard.dataset.diarModel : "3d_speaker");
    });
  }

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
      const batchSize = el.diarBatchSize ? parseInt(el.diarBatchSize.value, 10) : 1;
      const token = el.hfTokenInput.value.trim() || undefined;
      const minSpkEl = document.getElementById('diar-min-speakers');
      const maxSpkEl = document.getElementById('diar-max-speakers');
      const numSpkEl = document.getElementById('diar-num-speakers');
      const minSpeakers = minSpkEl && minSpkEl.value.trim() ? parseInt(minSpkEl.value, 10) : undefined;
      const maxSpeakers = maxSpkEl && maxSpkEl.value.trim() ? parseInt(maxSpkEl.value, 10) : undefined;
      const numSpeakers = numSpkEl && numSpkEl.value.trim() ? parseInt(numSpkEl.value, 10) : undefined;

      const overlapEl = document.getElementById('diar-3d-overlap');
      const includeOverlap = overlapEl ? overlapEl.checked : false;
      const enrollmentSelect = document.getElementById('diar-enrollment-profile-select');
      const enrollmentProfile = enrollmentSelect && !enrollmentSelect.disabled
        ? enrollmentSelect.value || undefined
        : undefined;

      const vadOnsetEl = document.getElementById('diar-vad-onset');
      const vadOffsetEl = document.getElementById('diar-vad-offset');
      const vadOnset = vadOnsetEl && vadOnsetEl.value.trim() ? parseFloat(vadOnsetEl.value) : 0.5;
      const vadOffset = vadOffsetEl && vadOffsetEl.value.trim() ? parseFloat(vadOffsetEl.value) : 0.3;

      const chunkDurationEl = document.getElementById('diar-chunk-duration');
      const chunkStepEl = document.getElementById('diar-chunk-step');
      const chunkDuration = chunkDurationEl ? parseFloat(chunkDurationEl.value) : 1.5;
      const chunkStep = chunkStepEl ? parseFloat(chunkStepEl.value) : 0.75;
      const sortformerOnset = el.diarSortformerOnset ? parseFloat(el.diarSortformerOnset.value) : 0.74;
      const sortformerOffset = el.diarSortformerOffset ? parseFloat(el.diarSortformerOffset.value) : 0.64;
      const sortformerPadOnset = el.diarSortformerPadOnset ? parseFloat(el.diarSortformerPadOnset.value) : 0.12;
      const sortformerPadOffset = el.diarSortformerPadOffset ? parseFloat(el.diarSortformerPadOffset.value) : 0.20;

      if (
        modelType === 'sortformer'
        && Number.isFinite(sortformerOnset)
        && Number.isFinite(sortformerOffset)
        && sortformerOnset < sortformerOffset
      ) {
        showToast('Sortformer Boundary Onset must be greater than or equal to Boundary Offset', 'error');
        return;
      }

      el.btnRunDiarization.disabled = true;
      el.diarTaskProgressBox.classList.remove('hidden');
      if (el.diarTaskStatusText) {
        el.diarTaskStatusText.textContent = enrollmentProfile
          ? `Building ${enrollmentProfile} enrollment, then running ${diarizationModelLabel(modelType)}...`
          : `Running ${diarizationModelLabel(modelType)} diarization...`;
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
            batch_size: Number.isInteger(batchSize) && batchSize > 0 ? batchSize : 1,
            token: token,
            min_speakers: Number.isFinite(minSpeakers) ? minSpeakers : undefined,
            max_speakers: Number.isFinite(maxSpeakers) ? maxSpeakers : undefined,
            num_speakers: Number.isFinite(numSpeakers) ? numSpeakers : undefined,
            include_overlap: includeOverlap,
            vad_onset: Number.isFinite(vadOnset) ? vadOnset : 0.5,
            vad_offset: Number.isFinite(vadOffset) ? vadOffset : 0.3,
            chunk_duration_s: Number.isFinite(chunkDuration) ? chunkDuration : 1.5,
            chunk_step_s: Number.isFinite(chunkStep) ? chunkStep : 0.75,
            sortformer_onset: Number.isFinite(sortformerOnset) ? sortformerOnset : 0.74,
            sortformer_offset: Number.isFinite(sortformerOffset) ? sortformerOffset : 0.64,
            sortformer_pad_onset_s: Number.isFinite(sortformerPadOnset) ? sortformerPadOnset : 0.12,
            sortformer_pad_offset_s: Number.isFinite(sortformerPadOffset) ? sortformerPadOffset : 0.20,
            enrollment_profile: enrollmentProfile,
          }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Diarization failed to start");

        pollTask(data.task_id, (result) => {
          clearInterval(timerInterval);
          el.diarTaskProgressBox.classList.add('hidden');
          el.btnRunDiarization.disabled = false;
          showToast(
            result.enrollment_profile
              ? `Diarization completed with enrolled speaker "${result.enrollment_profile}" in ${result.elapsed_s}s!`
              : `Speaker Diarization completed in ${result.elapsed_s}s!`,
            "success",
          );
          state.diarization.customNames = {};
          state.diarization.colors = {};
          state.diarization.activeHistoryId = null;
          renderDiarizationWorkspace(result.diarization, audioId);
          saveDiarizationToHistory(result.diarization, audioId, result);
          loadDiarizationResultsForVerification();
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

  if (el.btnDiarReset) {
    el.btnDiarReset.addEventListener('click', () => {
      clearDiarizationWorkspace();
      showToast("Diarization workspace reset", "info");
    });
  }

  if (el.diarHistorySearchInput) {
    el.diarHistorySearchInput.addEventListener('input', (event) => {
      state.diarization.historySearch = event.target.value.toLowerCase().trim();
      renderDiarizationHistory();
    });
  }

  if (el.btnClearDiarHistory) {
    el.btnClearDiarHistory.addEventListener('click', clearDiarizationHistory);
  }

  loadDiarizationHistory();

  if (el.btnDiarPlayToggle) {
    el.btnDiarPlayToggle.addEventListener('click', () => {
      const audioId = state.diarization.audioId || (el.diarInputSelect ? el.diarInputSelect.value : null);
      if (audioId) {
        if (!el.audio.src || !el.audio.src.includes(audioId)) {
          loadAudioIntoPlayer(audioId);
        }
      }
      togglePlayPause();
    });
  }

  if (el.btnDiarSkipBack) el.btnDiarSkipBack.addEventListener('click', () => seekRelative(-5));
  if (el.btnDiarSkipFwd) el.btnDiarSkipFwd.addEventListener('click', () => seekRelative(5));

  if (el.diarAutoNext) {
    el.diarAutoNext.checked = Boolean(state.diarization.autoAdvance);
    el.diarAutoNext.addEventListener('change', () => {
      state.diarization.autoAdvance = el.diarAutoNext.checked;
      const t = el.audio?.currentTime || 0;
      applySpeakerSoloMuteAudio(t);
      if (state.diarization.autoAdvance) {
        maybeAutoAdvanceSegment(t);
        startDiarPlaybackWatch();
      }
    });
  }

  if (el.diarSpeedSelect) {
    el.diarSpeedSelect.addEventListener('change', (e) => {
      setPlaybackRate(parseFloat(e.target.value) || 1.0);
    });
  }

  if (el.btnDiarZoomIn) {
    el.btnDiarZoomIn.addEventListener('click', () => setDiarZoom((state.diarization.zoom || 1.0) * 1.5));
  }
  if (el.btnDiarZoomOut) {
    el.btnDiarZoomOut.addEventListener('click', () => setDiarZoom((state.diarization.zoom || 1.0) / 1.5));
  }
  if (el.btnDiarZoomFit) {
    el.btnDiarZoomFit.addEventListener('click', () => setDiarZoom(1.0));
  }
  if (el.diarZoomInput) {
    const handleZoomInput = (e) => {
      const val = parseFloat(e.target.value);
      if (!isNaN(val) && val > 0) setDiarZoom(val);
    };
    el.diarZoomInput.addEventListener('input', handleZoomInput);
    el.diarZoomInput.addEventListener('change', handleZoomInput);
    el.diarZoomInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        handleZoomInput(e);
        el.diarZoomInput.blur();
      }
    });
  }

  if (el.diarFilterSpeakerSelect) {
    el.diarFilterSpeakerSelect.addEventListener('change', (e) => {
      state.diarization.activeSpeakerFilter = e.target.value;
      renderDiarizationFilteredViews();
    });
  }

  if (el.diarTurnsSearchInput) {
    el.diarTurnsSearchInput.addEventListener('input', (e) => {
      state.diarization.searchQuery = e.target.value.toLowerCase().trim();
      renderDiarizationFilteredViews();
    });
  }

  if (el.diarFilterMinDur) {
    const handleMinDur = (e) => {
      const val = parseFloat(e.target.value);
      state.diarization.minDurFilter = (!isNaN(val) && val > 0) ? val : 0;
      renderDiarizationFilteredViews();
    };
    el.diarFilterMinDur.addEventListener('input', handleMinDur);
    el.diarFilterMinDur.addEventListener('change', handleMinDur);
  }

  if (el.diarFilterMaxDur) {
    const handleMaxDur = (e) => {
      const val = parseFloat(e.target.value);
      state.diarization.maxDurFilter = (!isNaN(val) && val > 0) ? val : 0;
      renderDiarizationFilteredViews();
    };
    el.diarFilterMaxDur.addEventListener('input', handleMaxDur);
    el.diarFilterMaxDur.addEventListener('change', handleMaxDur);
  }

  if (el.btnDiarFilterOverlaps) {
    el.btnDiarFilterOverlaps.addEventListener('click', () => {
      state.diarization.overlapFilter = !state.diarization.overlapFilter;
      el.btnDiarFilterOverlaps.classList.toggle('active', state.diarization.overlapFilter);
      el.btnDiarFilterOverlaps.setAttribute(
        'aria-pressed',
        state.diarization.overlapFilter ? 'true' : 'false',
      );
      renderDiarizationFilteredViews();
    });
  }

  if (el.diarFilterTargetSelect) {
    el.diarFilterTargetSelect.addEventListener('change', (e) => {
      state.diarization.targetMatchFilter = e.target.value;
      renderDiarizationFilteredViews();
    });
  }

  if (el.diarFilterReviewSelect) {
    el.diarFilterReviewSelect.addEventListener('change', (e) => {
      state.diarization.reviewFilter = e.target.value;
      renderDiarizationFilteredViews();
    });
  }

  if (el.btnDiarClearFilters) {
    el.btnDiarClearFilters.addEventListener('click', clearDiarizationTurnFilters);
  }

  if (el.btnDiarCleanTurns) {
    el.btnDiarCleanTurns.addEventListener('click', toggleDiarizationCleanTurns);
  }

  [
    el.diarCleanJitterMax,
    el.diarCleanBoundaryCollar,
    el.diarCleanMergeGap,
    el.diarCleanMinDuration,
  ].filter(Boolean).forEach(input => {
    input.addEventListener('change', () => {
      readDiarizationCleanTurnsSettings();
      if (state.diarization.cleanTurnsEnabled) applyDiarizationCleanTurns();
    });
  });

  [
    el.diarExtractAddExtra,
    el.diarExtractStopOther,
    el.purityExtractAddExtra,
    el.purityExtractStopOther,
    el.diarExtractPreRoll,
    el.diarExtractPostRoll,
    el.purityExtractPreRoll,
    el.purityExtractPostRoll,
  ]
    .filter(Boolean)
    .forEach(input => {
      input.addEventListener('change', () => readDiarizationExtractionSettings(input));
    });
  readDiarizationExtractionSettings();

  if (el.diarSortTurnsSelect) {
    el.diarSortTurnsSelect.addEventListener('change', (e) => {
      state.diarization.sortMode = e.target.value;
      renderTurnsTable();
    });
  }

  if (el.diarHistogramBinSelect) {
    const rawSaved = state.diarization.histogramBinWidth;
    const isCustom = rawSaved === 'custom' || (!['auto', '0.1', '0.25', '0.5', '1', '2', '3', '5', '10'].includes(String(rawSaved)) && rawSaved !== 'auto');
    if (isCustom) {
      el.diarHistogramBinSelect.value = 'custom';
      if (el.diarHistogramBinCustom) {
        el.diarHistogramBinCustom.classList.remove('hidden');
        el.diarHistogramBinCustom.value = state.diarization.histogramCustomBinWidth || 1.0;
      }
    } else {
      el.diarHistogramBinSelect.value = String(rawSaved || 'auto');
      if (el.diarHistogramBinCustom) {
        el.diarHistogramBinCustom.classList.add('hidden');
      }
    }

    el.diarHistogramBinSelect.addEventListener('change', (e) => {
      const val = e.target.value;
      if (val === 'custom') {
        if (el.diarHistogramBinCustom) {
          el.diarHistogramBinCustom.classList.remove('hidden');
          if (!el.diarHistogramBinCustom.value) {
            el.diarHistogramBinCustom.value = state.diarization.histogramCustomBinWidth || 1.0;
          }
          el.diarHistogramBinCustom.focus();
        }
        state.diarization.histogramBinWidth = 'custom';
        try { localStorage.setItem('sonic_diar_hist_bin', 'custom'); } catch {}
      } else {
        if (el.diarHistogramBinCustom) {
          el.diarHistogramBinCustom.classList.add('hidden');
        }
        state.diarization.histogramBinWidth = val === 'auto' ? 'auto' : parseFloat(val);
        try { localStorage.setItem('sonic_diar_hist_bin', String(val)); } catch {}
      }
      renderTurnsHistogramOnly();
    });
  }

  if (el.diarHistogramBinCustom) {
    const handleCustomBin = () => {
      const val = parseFloat(el.diarHistogramBinCustom.value);
      if (Number.isFinite(val) && val > 0) {
        state.diarization.histogramCustomBinWidth = val;
        try { localStorage.setItem('sonic_diar_hist_bin_custom', String(val)); } catch {}
        renderTurnsHistogramOnly();
      }
    };
    el.diarHistogramBinCustom.addEventListener('input', handleCustomBin);
    el.diarHistogramBinCustom.addEventListener('change', handleCustomBin);
  }

  if (el.btnExtractAllSpeakers) {
    el.btnExtractAllSpeakers.addEventListener('click', extractAllSpeakers);
  }

  if (el.btnDownloadFilteredTurns) {
    el.btnDownloadFilteredTurns.addEventListener('click', downloadFilteredTurns);
  }

  if (el.btnDownloadExport) {
    el.btnDownloadExport.addEventListener('click', downloadDiarizationRttm);
  }

  setupDiarTimelineSeek();
  el.diarMultitrackViewport?.addEventListener('scroll', () => {
    state.diarization.waveform.data = null;
    renderDiarWaveform();
    renderDiarRuler();
    scheduleDiarWaveform();
  }, { passive: true });

  const btnScrollTop = document.getElementById('btn-scroll-to-top');
  const btnScrollTimeline = document.getElementById('btn-scroll-to-timeline');
  const btnScrollTurns = document.getElementById('btn-scroll-to-turns');

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
}

async function loadDiarWaveform(audioId) {
  if (!audioId) return;
  if (state.diarization.waveform.audioId !== audioId) {
    state.diarization.waveform.data = null;
    state.diarization.waveform.error = '';
  }
  state.diarization.waveform.audioId = audioId;
  scheduleDiarWaveform();
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

function clearDiarizationWorkspace() {
  state.diarization.data = null;
  state.diarization.rawTurns = [];
  state.diarization.turns = [];
  state.diarization.cleanTurnsEnabled = false;
  state.diarization.cleanTurnsSummary = null;
  state.diarization.speakers = [];
  state.diarization.customNames = {};
  state.diarization.colors = {};
  state.diarization.activeTurnIndex = null;
  state.diarization.activeHistoryId = null;
  state.diarization.soloSpeaker = null;
  state.diarization.mutedSpeakers.clear();
  state.diarization.overlapFilter = false;
  state.diarization.targetMatchFilter = 'all';
  state.diarization.reviewFilter = 'all';
  if (el.btnDiarFilterOverlaps) {
    el.btnDiarFilterOverlaps.classList.remove('active');
    el.btnDiarFilterOverlaps.setAttribute('aria-pressed', 'false');
  }
  updateDiarizationCleanTurnsControl();
  if (el.audio) el.audio.muted = false;
  if (el.diarResultsWrapper) el.diarResultsWrapper.classList.add('hidden');
  if (el.diarEmptyPlaceholder) el.diarEmptyPlaceholder.classList.remove('hidden');
  resetTargetSpeakerEvaluation({ preserveSelection: true });
  state.targetSpeaker.assignedSpeakerId = '';
  renderTargetSpeakerAssignmentOptions();
  renderTargetSpeakerContext();
  renderDiarizationHistory();
}

function diarizationHistoryMatch(item, audio) {
  if (!item || !audio) return false;
  if (item.session_audio_id && item.session_audio_id === audio.id) return true;
  if (item.audio_path && audio.path && normalizedAudioPath(item.audio_path) === normalizedAudioPath(audio.path)) return true;
  if (item.audio_fingerprint && audio.fingerprint && item.audio_fingerprint === audio.fingerprint) return true;
  const sameSource = Boolean(
    (item.audio_source_id && item.audio_source_id === audio.source_id)
    || (item.audio_id && (item.audio_id === audio.id || item.audio_id === audio.source_id))
  );
  if (!sameSource) return false;
  if (item.duration_s && audio.duration_s) {
    return Math.abs(Number(item.duration_s) - Number(audio.duration_s)) < 0.5;
  }
  return true;
}

function findDiarizationHistoryForAudio(audioId) {
  const audio = state.audioList.find(item => item.id === audioId);
  if (!audio) return null;
  const matches = (state.diarization.history || []).filter(item => diarizationHistoryMatch(item, audio));
  if (!matches.length) return null;
  const ranked = matches.map(item => {
    let score = 0;
    if (item.session_audio_id && item.session_audio_id === audio.id) score += 8;
    if (item.audio_path && audio.path && normalizedAudioPath(item.audio_path) === normalizedAudioPath(audio.path)) score += 4;
    if (item.audio_fingerprint && audio.fingerprint && item.audio_fingerprint === audio.fingerprint) score += 2;
    return { item, score };
  });
  ranked.sort((a, b) => b.score - a.score || (b.item.timestamp || 0) - (a.item.timestamp || 0));
  return ranked[0].item;
}

function resolveDiarizationHistoryAudio(item) {
  if (!item) return null;
  if (item.session_audio_id) {
    const bySession = state.audioList.find(audio => audio.id === item.session_audio_id);
    if (bySession) return bySession;
  }
  if (item.audio_path) {
    const byPath = state.audioList.find(audio => (
      audio.path && normalizedAudioPath(audio.path) === normalizedAudioPath(item.audio_path)
    ));
    if (byPath) return byPath;
  }
  if (item.audio_fingerprint) {
    const byFingerprint = state.audioList.find(audio => (
      audio.fingerprint && audio.fingerprint === item.audio_fingerprint
    ));
    if (byFingerprint) return byFingerprint;
  }
  return state.audioList.find(audio => diarizationHistoryMatch(item, audio)) || null;
}

function hideSavedDiarizationNotice() {
  if (el.diarSavedNoticePill) el.diarSavedNoticePill.classList.add('hidden');
}

function showSavedDiarizationNotice(item, audioId) {
  if (!item || !el.diarSavedNoticePill) return;
  el.diarSavedNoticePill.classList.remove('hidden');
  if (el.diarSavedNoticeText) {
    el.diarSavedNoticeText.textContent = `Restored ${item.turn_count || 0} turns across ${item.speaker_count || 0} speakers from history`;
  }
  if (el.btnLoadSavedForTrack) {
    el.btnLoadSavedForTrack.textContent = 'Reload Session';
    el.btnLoadSavedForTrack.onclick = () => {
      restoreDiarizationHistoryItem(item, audioId, { notify: true, scroll: true });
    };
  }
}

async function openDiarizationAudio(audioId, { restoreHistory = false, waitForHistory = true } = {}) {
  if (restoreHistory && waitForHistory) {
    await diarizationHistoryReady;
  }
  const previousAudioId = state.diarization.audioId;
  renderDiarizationChildren(audioId);
  updateDiarInputMeta(audioId);
  loadDiarWaveform(audioId);
  loadAudioIntoPlayer(audioId, false);

  const historyItem = findDiarizationHistoryForAudio(audioId);
  if (restoreHistory && historyItem) {
    await restoreDiarizationHistoryItem(historyItem, audioId, { notify: true, scroll: false });
    return;
  }

  if (previousAudioId && previousAudioId !== audioId && state.diarization.data) {
    clearDiarizationWorkspace();
  }
  state.diarization.audioId = audioId;
  if (historyItem) showSavedDiarizationNotice(historyItem, audioId);
  else hideSavedDiarizationNotice();
}

function cloneDiarizationData(data) {
  return JSON.parse(JSON.stringify(data || {}));
}

function saveDiarizationToHistory(diarization, audioId, runResult = {}) {
  if (!diarization) return;
  if (!Array.isArray(state.diarization.turns)) state.diarization.turns = [];
  if (!Array.isArray(state.diarization.rawTurns)) state.diarization.rawTurns = [];
  if (!Array.isArray(state.diarization.history)) state.diarization.history = [];

  const audio = state.audioList.find(item => item.id === audioId) || {};
  const completeData = cloneDiarizationData(state.diarization.data || diarization);
  completeData.turns = cloneDiarizationData(state.diarization.rawTurns);
  completeData.speakers = cloneDiarizationData(state.diarization.speakers);
  completeData.duration_s = state.diarization.duration;

  const totalSpeechS = state.diarization.rawTurns.reduce(
    (total, turn) => total + Math.max(0, turn.end_s - turn.start_s),
    0,
  );
  const modelBackend = completeData.model?.backend || completeData.model?.model_id || 'Pyannote';
  const historyItem = {
    id: completeData.result_id || `diar_hist_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
    timestamp: Date.now(),
    audio_id: audioId,
    audio_title: audio.title || audio.source_id || audioId,
    audio_path: audio.path || null,
    audio_fingerprint: audio.fingerprint || null,
    audio_source_id: audio.source_id || null,
    duration_s: state.diarization.duration,
    model_backend: modelBackend,
    model_id: completeData.model?.model_id || null,
    speaker_count: state.diarization.speakers.length,
    turn_count: state.diarization.rawTurns.length,
    total_speech_s: totalSpeechS,
    speech_ratio_pct: state.diarization.duration > 0
      ? Number(((totalSpeechS / state.diarization.duration) * 100).toFixed(1))
      : 0,
    diarization: completeData,
    custom_names: { ...state.diarization.customNames },
    colors: { ...state.diarization.colors },
    segment_labels: { ...state.targetSpeaker.labels },
    run: {
      elapsed_s: runResult.elapsed_s ?? null,
      device: runResult.device ?? null,
      power_w: runResult.power_w ?? null,
      enrollment_profile: runResult.enrollment_profile ?? null,
    },
  };

  state.diarization.history.unshift(historyItem);
  state.diarization.history = state.diarization.history.slice(0, 30);
  state.diarization.activeHistoryId = historyItem.id;
  persistDiarizationHistory();
  renderDiarizationHistory();
  showSavedDiarizationNotice(historyItem, audioId);
  loadDiarizationHistory();
}

function persistDiarizationHistory() {
  try {
    const uiState = Object.fromEntries((state.diarization.history || []).map(item => [
      item.diarization?.result_id || item.id,
      {
        custom_names: item.custom_names || {},
        colors: item.colors || {},
        segment_labels: item.segment_labels || {},
      },
    ]));
    localStorage.setItem('sonic_diarization_ui_state', JSON.stringify(uiState));
    localStorage.removeItem('sonic_diarization_history');
  } catch (err) {
    console.warn('Could not persist diarization viewer preferences:', err);
  }
}

function updateActiveDiarizationHistory() {
  const item = state.diarization.history.find(entry => entry.id === state.diarization.activeHistoryId);
  if (!item || !state.diarization.data) return;
  item.diarization = cloneDiarizationData(state.diarization.data);
  item.diarization.turns = cloneDiarizationData(state.diarization.rawTurns);
  item.diarization.speakers = cloneDiarizationData(state.diarization.speakers);
  item.custom_names = { ...state.diarization.customNames };
  item.colors = { ...state.diarization.colors };
  item.segment_labels = { ...state.targetSpeaker.labels };
  persistDiarizationHistory();
  renderDiarizationHistory();
}

function normalizeDiarizationHistoryItem(item) {
  if (!item || typeof item !== 'object') return null;
  const legacyNames = item.customNames || item.custom_names || {};
  const diarization = item.diarization || {
    schema_version: '1.0',
    audio_id: item.audio_id,
    duration_s: item.duration_s || 0,
    speakers: item.speakers || [],
    turns: item.turns || [],
    model: { backend: item.model_backend, model_id: item.model_id },
  };
  const speakers = Array.isArray(diarization.speakers) ? diarization.speakers : [];
  const turns = Array.isArray(diarization.turns) ? diarization.turns : [];
  const uniqueTurnSpeakers = new Set(turns.map(t => t.speaker_id).filter(Boolean));
  const speakerCount = uniqueTurnSpeakers.size > 0 ? uniqueTurnSpeakers.size : (Number(item.speaker_count) || speakers.length);
  return {
    ...item,
    id: item.id || `diar_hist_${item.timestamp || Date.now()}`,
    timestamp: Number(item.timestamp) || Date.now(),
    speaker_count: speakerCount,
    turn_count: Number(item.turn_count) || turns.length,
    diarization,
    custom_names: legacyNames,
    colors: item.colors || {},
    segment_labels: item.segment_labels || {},
  };
}

let diarizationHistoryLoadSeq = 0;
let diarizationHistoryReady = Promise.resolve();

async function loadDiarizationHistory() {
  const seq = ++diarizationHistoryLoadSeq;
  const pending = (async () => {
    try {
      const payload = await parseJsonResponse(await fetch('/api/diarization/results'));
      if (seq !== diarizationHistoryLoadSeq) return;
      await fetchAudioList();
      if (seq !== diarizationHistoryLoadSeq) return;
      const uiState = JSON.parse(localStorage.getItem('sonic_diarization_ui_state') || '{}');
      const loaded = (payload.results || []).map(result => {
        const source = result.source_audio || {};
        const summary = result.summary || {};
        const ui = uiState[result.result_id] || {};
        return normalizeDiarizationHistoryItem({
          id: result.result_id,
          timestamp: Number(result.created_at || 0) * 1000,
          audio_id: result.session_audio_id || result.audio_id,
          session_audio_id: result.session_audio_id || null,
          audio_title: source.title || result.audio_id,
          audio_path: source.path,
          audio_source_id: source.source_id || result.audio_id,
          audio_fingerprint: source.fingerprint || null,
          duration_s: source.duration_s || 0,
          model_backend: result.model?.backend,
          model_id: result.model?.model_id,
          speaker_count: summary.speaker_count,
          turn_count: summary.turn_count,
          total_speech_s: summary.total_speech_duration_s,
          speech_ratio_pct: source.duration_s > 0 ? Number((100 * summary.total_speech_duration_s / source.duration_s).toFixed(1)) : 0,
          source_available: Boolean(result.source_available),
          diarization: result,
          custom_names: ui.custom_names || {},
          colors: ui.colors || {},
          segment_labels: ui.segment_labels || {},
        });
      }).filter(Boolean);
      const loadedIds = new Set(loaded.map(item => item.id));
      const localOnly = (state.diarization.history || []).filter(item => (
        item?.id
        && !loadedIds.has(item.id)
        && Array.isArray(item.diarization?.turns)
        && item.diarization.turns.length > 0
      ));
      const activeId = state.diarization.activeHistoryId;
      state.diarization.history = [...localOnly, ...loaded];
      if (activeId && state.diarization.history.some(item => item.id === activeId)) {
        state.diarization.activeHistoryId = activeId;
      }
    } catch (err) {
      if (seq !== diarizationHistoryLoadSeq) return;
      console.warn('Could not load durable diarization history:', err);
      if (!Array.isArray(state.diarization.history)) state.diarization.history = [];
    }
    if (seq !== diarizationHistoryLoadSeq) return;
    renderDiarizationHistory();
    const selectedId = el.diarInputSelect?.value;
    if (selectedId && !selectedId.startsWith('lib:')) {
      const historyItem = findDiarizationHistoryForAudio(selectedId);
      if (historyItem) showSavedDiarizationNotice(historyItem, selectedId);
      else hideSavedDiarizationNotice();
    }
  })();
  diarizationHistoryReady = pending.catch(() => {});
  await pending;
}

function historyDiarizationData(item) {
  return cloneDiarizationData(normalizeDiarizationHistoryItem(item)?.diarization || {});
}

async function restoreDiarizationHistoryItem(item, targetAudioId = null, { notify = false, scroll = false } = {}) {
  const normalized = normalizeDiarizationHistoryItem(item);
  if (!normalized) return;
  const resultId = normalized.diarization?.result_id || normalized.id;
  let diarization = historyDiarizationData(normalized);
  if (!Array.isArray(diarization.turns) || diarization.turns.length === 0) {
    try {
      diarization = await parseJsonResponse(
        await fetch(`/api/diarization/results/${encodeURIComponent(resultId)}`)
      );
      normalized.diarization = diarization;
      normalized.session_audio_id = diarization.session_audio_id || normalized.session_audio_id;
      normalized.audio_path = diarization.source_audio?.path || normalized.audio_path;
      normalized.audio_fingerprint = diarization.source_audio?.fingerprint || normalized.audio_fingerprint;
      normalized.turn_count = diarization.turns?.length || normalized.turn_count;
      normalized.speaker_count = diarization.speakers?.length || normalized.speaker_count;
      const historyIndex = (state.diarization.history || []).findIndex(entry => entry.id === normalized.id);
      if (historyIndex >= 0) state.diarization.history[historyIndex] = normalized;
      await fetchAudioList();
    } catch (err) {
      showToast(`Could not load diarization result: ${err.message}`, 'error');
      return;
    }
  }

  const matchedAudio = targetAudioId
    ? state.audioList.find(audio => audio.id === targetAudioId)
    : resolveDiarizationHistoryAudio(normalized);
  const audioId = matchedAudio?.id || diarization.session_audio_id || targetAudioId || normalized.session_audio_id || normalized.audio_id;

  state.diarization.customNames = { ...normalized.custom_names };
  state.diarization.colors = { ...normalized.colors };
  state.diarization.activeHistoryId = normalized.id;
  renderDiarizationWorkspace(diarization, audioId);
  state.targetSpeaker.labels = { ...normalized.segment_labels };
  renderDiarizationFilteredViews();

  if (matchedAudio || state.audioList.some(audio => audio.id === audioId)) {
    const resolvedId = matchedAudio?.id || audioId;
    if (el.diarInputSelect) el.diarInputSelect.value = resolvedId;
    renderDiarizationChildren(resolvedId);
    updateDiarInputMeta(resolvedId);
    loadAudioIntoPlayer(resolvedId, false);
  }
  showSavedDiarizationNotice(normalized, audioId);
  renderDiarizationHistory();

  if (scroll && el.diarResultsWrapper) {
    el.diarResultsWrapper.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
  if (notify) {
    showToast(`Restored diarization for "${normalized.audio_title || audioId}"`, 'success');
  }
}

function formatDiarizationHistoryAge(timestamp) {
  const seconds = Math.max(0, Math.floor((Date.now() - timestamp) / 1000));
  if (seconds < 10) return 'Just now';
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

function renderDiarizationHistory() {
  if (!el.diarHistoryList) return;
  const query = state.diarization.historySearch;
  const items = (state.diarization.history || []).filter(item => {
    if (!query) return true;
    const names = Object.values(item.custom_names || {}).join(' ');
    const searchable = `${item.audio_title || ''} ${item.model_backend || ''} ${names} ${new Date(item.timestamp).toLocaleString()}`.toLowerCase();
    return searchable.includes(query);
  });

  if (el.diarHistoryCountBadge) {
    const total = state.diarization.history.length;
    el.diarHistoryCountBadge.textContent = `${total} session${total === 1 ? '' : 's'}`;
  }
  if (items.length === 0) {
    el.diarHistoryList.innerHTML = `<div class="empty-placeholder">${query ? 'No saved sessions match this filter.' : 'No diarization sessions yet.'}</div>`;
    return;
  }

  el.diarHistoryList.innerHTML = '';
  items.forEach(item => {
    const active = state.diarization.activeHistoryId === item.id;
    const sourceAvailable = Boolean(resolveDiarizationHistoryAudio(item));
    const card = document.createElement('div');
    card.className = `diar-history-item ${active ? 'active-session' : ''}`;
    card.innerHTML = `
      <div class="diar-hist-top">
        <div class="diar-hist-title-wrap">
          <span class="badge ${active ? 'badge-accent' : 'badge-info'}">${escapeHtml(diarizationModelLabel(item.model_backend))}</span>
          <span class="diar-hist-title" title="${escapeHtml(item.audio_title || item.audio_id)}">${escapeHtml(item.audio_title || item.audio_id)}</span>
          ${active ? '<span class="badge badge-success badge-sm">Open</span>' : ''}
        </div>
        <span class="diar-hist-time" title="${escapeHtml(new Date(item.timestamp).toLocaleString())}">${formatDiarizationHistoryAge(item.timestamp)}</span>
      </div>
      <div class="diar-hist-meta-row">
        <span><strong>${item.speaker_count}</strong> speakers</span>
        <span><strong>${item.turn_count}</strong> turns</span>
        <span><strong>${Number(item.speech_ratio_pct || 0).toFixed(1)}%</strong> speech</span>
        <span>${formatTime(item.duration_s || 0)}</span>
        <span class="badge badge-sm ${sourceAvailable || item.source_available ? 'badge-success' : 'badge-ghost'}">${sourceAvailable || item.source_available ? 'Audio available' : 'Annotations only'}</span>
      </div>
      <div class="diar-hist-actions">
        <button class="btn btn-xs btn-primary btn-load-diar-history">Open Viewer</button>
        <button class="btn btn-xs btn-ghost text-destructive btn-delete-diar-history" title="Delete this history entry">Delete</button>
      </div>
    `;
    card.querySelector('.btn-load-diar-history').addEventListener('click', () => {
      restoreDiarizationHistoryItem(item, null, { notify: true, scroll: true });
    });
    card.querySelector('.btn-delete-diar-history').addEventListener('click', () => {
      deleteDiarizationHistoryItem(item);
    });
    el.diarHistoryList.appendChild(card);
  });
}

async function deleteDiarizationHistoryItem(item) {
  const resultId = item?.diarization?.result_id || item?.id;
  if (!resultId) return;
  try {
    await parseJsonResponse(await fetch(`/api/diarization/results/${encodeURIComponent(resultId)}`, {
      method: 'DELETE',
    }));
    if (state.diarization.activeHistoryId === resultId) {
      state.diarization.activeHistoryId = null;
      hideSavedDiarizationNotice();
    }
    persistDiarizationHistory();
    await loadDiarizationHistory();
    loadDiarizationResultsForVerification();
    showToast('Deleted diarization history entry', 'info');
  } catch (err) {
    showToast(`Could not delete history entry: ${err.message}`, 'error');
  }
}

async function clearDiarizationHistory() {
  const count = state.diarization.history.length;
  if (!count || !confirm(`Clear all ${count} saved diarization session${count === 1 ? '' : 's'}?`)) return;
  try {
    await parseJsonResponse(await fetch('/api/diarization/results/clear', { method: 'POST' }));
    state.diarization.history = [];
    state.diarization.activeHistoryId = null;
    localStorage.removeItem('sonic_diarization_ui_state');
    localStorage.removeItem('sonic_diarization_history');
    hideSavedDiarizationNotice();
    renderDiarizationHistory();
    loadDiarizationResultsForVerification();
    showToast('Diarization history cleared', 'info');
  } catch (err) {
    showToast(`Could not clear history: ${err.message}`, 'error');
  }
}

function setDiarZoom(zoom) {
  const clampedZoom = clampTimelineZoom(zoom, 0.2);
  state.diarization.zoom = clampedZoom;
  if (el.diarZoomInput && document.activeElement !== el.diarZoomInput) {
    el.diarZoomInput.value = formatZoomMultiplier(clampedZoom);
  }
  if (el.diarZoomLevel) el.diarZoomLevel.textContent = `${formatZoomMultiplier(clampedZoom)}x`;

  const viewportWidth = el.diarMultitrackViewport ? el.diarMultitrackViewport.clientWidth : 1000;
  const labelColWidth = el.diarLaneLabelsCol ? el.diarLaneLabelsCol.offsetWidth : 200;
  const visibleTrackWidth = Math.max(300, viewportWidth - labelColWidth);
  const targetWidth = Math.round(visibleTrackWidth * clampedZoom);
  if (el.diarTracksArea) {
    el.diarTracksArea.style.width = `${targetWidth}px`;
    el.diarTracksArea.style.minWidth = `${targetWidth}px`;
  }

  state.diarization.waveform.data = null;
  renderDiarWaveform();
  renderDiarRuler();
  scheduleDiarWaveform();
}

function normalizedDiarizationTurns(turns, fallbackSpeakerId = "spk_00") {
  return (turns || []).map(t => ({
    ...t,
    speaker_id: t.speaker_id || fallbackSpeakerId,
    start_s: roundNum(Math.max(0, Number(t.start_s) || 0), 2),
    end_s: roundNum(Math.max(0.05, Number(t.end_s) || 0), 2),
  }));
}

function canonicalDiarizationTurns(turns, fallbackSpeakerId = "spk_00") {
  return (turns || []).map(t => ({
    ...t,
    speaker_id: t.speaker_id || fallbackSpeakerId,
    start_s: Math.max(0, Number(t.start_s) || 0),
    end_s: Math.max(0.05, Number(t.end_s) || 0),
  }));
}

function readDiarizationCleanTurnsSettings() {
  const readNonNegative = (input, fallback) => {
    const value = input ? parseFloat(input.value) : fallback;
    return Number.isFinite(value) && value >= 0 ? value : fallback;
  };
  state.diarization.cleanTurnsSettings = {
    jitter_max_duration_s: readNonNegative(el.diarCleanJitterMax, 3.0),
    boundary_collar_s: readNonNegative(el.diarCleanBoundaryCollar, 0.0),
    merge_same_speaker_gap_s: readNonNegative(el.diarCleanMergeGap, 1.0),
    min_turn_duration_s: readNonNegative(el.diarCleanMinDuration, 0.5),
  };
  return state.diarization.cleanTurnsSettings;
}

function readDiarizationExtractionSettings(sourceInput) {
  const readNonNegative = (input, fallback) => {
    const value = input ? parseFloat(input.value) : fallback;
    return Number.isFinite(value) && value >= 0 ? value : fallback;
  };
  const preInput = sourceInput && sourceInput.id && sourceInput.id.includes('pre-roll')
    ? sourceInput
    : (el.diarExtractPreRoll || el.purityExtractPreRoll);
  const postInput = sourceInput && sourceInput.id && sourceInput.id.includes('post-roll')
    ? sourceInput
    : (el.diarExtractPostRoll || el.purityExtractPostRoll);
  const addExtraInput = sourceInput && sourceInput.id && sourceInput.id.includes('add-extra')
    ? sourceInput
    : (el.diarExtractAddExtra || el.purityExtractAddExtra);
  const stopOtherInput = sourceInput && sourceInput.id && sourceInput.id.includes('stop-other')
    ? sourceInput
    : (el.diarExtractStopOther || el.purityExtractStopOther);
  const stopOtherChecked = Boolean(stopOtherInput?.checked);
  state.diarization.extractionSettings = {
    add_extra: addExtra,
    stop_at_other_speakers: addExtra && stopOtherChecked,
    pre_roll_s: readNonNegative(preInput, 0.12),
    post_roll_s: readNonNegative(postInput, 0.20),
  };
  const preValue = String(state.diarization.extractionSettings.pre_roll_s);
  const postValue = String(state.diarization.extractionSettings.post_roll_s);
  [el.diarExtractPreRoll, el.purityExtractPreRoll].filter(Boolean).forEach(input => {
    if (input !== preInput) input.value = preValue;
  });
  [el.diarExtractPostRoll, el.purityExtractPostRoll].filter(Boolean).forEach(input => {
    if (input !== postInput) input.value = postValue;
  });
  [el.diarExtractAddExtra, el.purityExtractAddExtra].filter(Boolean).forEach(input => {
    input.checked = addExtra;
  });
  const stopOther = stopOtherChecked;
  [el.diarExtractStopOther, el.purityExtractStopOther].filter(Boolean).forEach(input => {
    input.checked = stopOther;
    input.disabled = !addExtra;
    input.closest('.extract-postprocess-option')?.classList.toggle('is-disabled', !addExtra);
  });
  [el.diarExtractAmounts, el.purityExtractAmounts].filter(Boolean).forEach(row => {
    row.classList.toggle('is-disabled', !addExtra);
  });
  [el.diarExtractPreRoll, el.diarExtractPostRoll, el.purityExtractPreRoll, el.purityExtractPostRoll]
    .filter(Boolean)
    .forEach(input => {
      input.disabled = !addExtra;
    });
  return state.diarization.extractionSettings;
}

function extractionBlockerTurns() {
  const settings = state.diarization.extractionSettings;
  if (!settings.add_extra || !settings.stop_at_other_speakers) return [];
  const turns = state.diarization.turns.length
    ? state.diarization.turns
    : (state.diarization.rawTurns || []);
  return turns.map(turn => ({
    speaker_id: turn.speaker_id,
    start_s: turn.start_s,
    end_s: turn.end_s,
  }));
}

function updateDiarizationCleanTurnsControl() {
  const enabled = state.diarization.cleanTurnsEnabled;
  if (el.btnDiarCleanTurns) {
    el.btnDiarCleanTurns.disabled = !state.diarization.rawTurns.length;
    el.btnDiarCleanTurns.classList.toggle('active', enabled);
    el.btnDiarCleanTurns.setAttribute('aria-pressed', enabled ? 'true' : 'false');
    el.btnDiarCleanTurns.textContent = enabled ? 'Clean Turns: On' : 'Clean Turns';
  }
  if (!el.diarCleanTurnsSummary) return;
  const summary = state.diarization.cleanTurnsSummary;
  el.diarCleanTurnsSummary.classList.toggle('hidden', !enabled || !summary);
  if (enabled && summary) {
    el.diarCleanTurnsSummary.textContent = `${summary.raw_turn_count} raw → ${summary.clean_turn_count} clean`;
    el.diarCleanTurnsSummary.title = `Non-destructive view; canonical raw result is unchanged. ${Number(summary.raw_duration_s || 0).toFixed(1)}s raw → ${Number(summary.clean_duration_s || 0).toFixed(1)}s clean.`;
  }
}

async function applyDiarizationCleanTurns() {
  el.btnDiarCleanTurns.disabled = true;
  try {
    const resultId = state.diarization.data.result_id;
    const response = await fetch('/api/diarization/clean-turns', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        result_id: resultId || undefined,
        turns: resultId ? undefined : state.diarization.rawTurns,
        settings: readDiarizationCleanTurnsSettings(),
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || 'Could not clean diarization turns');
    renderDiarizationWorkspace(state.diarization.data, state.diarization.audioId, {
      rawTurns: state.diarization.rawTurns,
      viewTurns: payload.turns,
      cleanTurnsEnabled: true,
      cleanTurnsSummary: payload.summary,
    });
    showToast(`Clean view: ${payload.summary.raw_turn_count} raw → ${payload.summary.clean_turn_count} turns`, 'success');
  } catch (err) {
    showToast(`Clean-turn cleanup failed: ${err.message}`, 'error');
  } finally {
    updateDiarizationCleanTurnsControl();
  }
}

async function toggleDiarizationCleanTurns() {
  if (!state.diarization.data || !state.diarization.rawTurns.length) return;

  if (state.diarization.cleanTurnsEnabled) {
    renderDiarizationWorkspace(state.diarization.data, state.diarization.audioId, {
      rawTurns: state.diarization.rawTurns,
    });
    showToast('Showing canonical raw diarization turns', 'info');
    return;
  }

  await applyDiarizationCleanTurns();
}

function renderDiarizationWorkspace(diarization, audioId, options = {}) {
  const preserveLabels = state.diarization.audioId === audioId;
  resetTargetSpeakerEvaluation({ preserveSelection: true, preserveLabels });
  state.targetSpeaker.assignedSpeakerId = '';
  state.diarization.audioId = audioId;
  state.diarization.data = cloneDiarizationData(diarization || state.diarization.data || {});
  state.diarization.soloSpeaker = null;
  state.diarization.mutedSpeakers.clear();
  state.diarization.activeSpeakerFilter = 'all';
  state.diarization.searchQuery = '';
  state.diarization.minDurFilter = 0;
  state.diarization.maxDurFilter = 0;
  state.diarization.overlapFilter = false;
  state.diarization.targetMatchFilter = 'all';
  state.diarization.reviewFilter = 'all';
  if (el.diarTurnsSearchInput) el.diarTurnsSearchInput.value = '';
  if (el.diarFilterMinDur) el.diarFilterMinDur.value = '';
  if (el.diarFilterMaxDur) el.diarFilterMaxDur.value = '';
  if (el.diarFilterTargetSelect) el.diarFilterTargetSelect.value = 'all';
  if (el.diarFilterReviewSelect) el.diarFilterReviewSelect.value = 'all';
  if (el.btnDiarFilterOverlaps) {
    el.btnDiarFilterOverlaps.classList.remove('active');
    el.btnDiarFilterOverlaps.setAttribute('aria-pressed', 'false');
  }
  if (el.audio) el.audio.muted = false;

  const rawSpeakers = state.diarization.data.speakers || state.diarization.speakers || [];
  const initialSpeakers = rawSpeakers.map(s => {
    if (typeof s === 'string') return { speaker_id: s };
    if (s && s.speaker_id) return s;
    return { speaker_id: s?.id || String(s) };
  });

  const fallbackSpeakerId = initialSpeakers[0]?.speaker_id || "spk_00";
  const canonicalTurns = options.rawTurns || state.diarization.data.turns || [];
  state.diarization.rawTurns = canonicalDiarizationTurns(canonicalTurns, fallbackSpeakerId);
  state.diarization.turns = normalizedDiarizationTurns(
    options.viewTurns || state.diarization.rawTurns,
    fallbackSpeakerId,
  );
  state.diarization.cleanTurnsEnabled = options.cleanTurnsEnabled === true;
  state.diarization.cleanTurnsSummary = options.cleanTurnsSummary || null;

  if (el.diarEmptyPlaceholder) el.diarEmptyPlaceholder.classList.add('hidden');
  if (el.diarResultsWrapper) el.diarResultsWrapper.classList.remove('hidden');

  const audioItem = state.audioList.find(a => a.id === audioId);
  const maxTurnEnd = state.diarization.turns.length > 0 ? Math.max(...state.diarization.turns.map(t => t.end_s)) : 10;
  const totalAudioDuration = (audioItem ? audioItem.duration_s : 0) || state.diarization.data.duration_s || maxTurnEnd || 10;
  state.diarization.duration = totalAudioDuration;

  const uniqueSpkIds = Array.from(new Set(state.diarization.turns.map(t => t.speaker_id).filter(Boolean)));
  if (uniqueSpkIds.length > 0) {
    const rawMap = new Map();
    initialSpeakers.forEach(s => {
      if (s.speaker_id && !rawMap.has(s.speaker_id)) {
        rawMap.set(s.speaker_id, s);
      }
    });
    const syncedSpeakers = [];
    rawMap.forEach((obj, id) => {
      if (uniqueSpkIds.includes(id)) {
        syncedSpeakers.push(obj);
      }
    });
    uniqueSpkIds.forEach(id => {
      if (!rawMap.has(id)) {
        syncedSpeakers.push({ speaker_id: id });
      }
    });
    state.diarization.speakers = syncedSpeakers;
  } else {
    state.diarization.speakers = initialSpeakers;
  }

  state.diarization.data.turns = cloneDiarizationData(state.diarization.rawTurns);
  state.diarization.data.speakers = state.diarization.speakers;
  state.diarization.data.duration_s = totalAudioDuration;

  state.diarization.speakers.forEach((spk, idx) => {
    if (!state.diarization.colors[spk.speaker_id]) {
      state.diarization.colors[spk.speaker_id] = DIAR_PALETTE[idx % DIAR_PALETTE.length];
    }
    if (!state.diarization.customNames[spk.speaker_id]) {
      state.diarization.customNames[spk.speaker_id] = spk.global_speaker_id || spk.speaker_id;
    }
  });

  detectTurnOverlaps();

  const totalSpeechS = state.diarization.turns.reduce((acc, t) => acc + Math.max(0, t.end_s - t.start_s), 0);
  const speechRatioPct = totalAudioDuration > 0 ? ((totalSpeechS / totalAudioDuration) * 100).toFixed(1) : "0.0";

  if (el.diarModelBadge) {
    const backend = state.diarization.data.model?.backend || state.diarization.data.model?.model_id || "Pyannote";
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

  updateDiarizationCleanTurnsControl();

  if (el.diarFilterSpeakerSelect) {
    el.diarFilterSpeakerSelect.innerHTML = `<option value="all">All Speakers (${state.diarization.speakers.length})</option>` +
      state.diarization.speakers.map(s => `<option value="${s.speaker_id}">${escapeHtml(getSpeakerName(s.speaker_id))}</option>`).join('');
    if (Array.from(el.diarFilterSpeakerSelect.options).some(option => option.value === state.diarization.activeSpeakerFilter)) {
      el.diarFilterSpeakerSelect.value = state.diarization.activeSpeakerFilter;
    } else {
      state.diarization.activeSpeakerFilter = 'all';
      el.diarFilterSpeakerSelect.value = 'all';
    }
  }

  loadDiarWaveform(audioId);
  renderDiarRuler();
  renderSpeakerSwimlanes();
  renderSpeakerProfiles();
  renderTurnsTable();
  renderTargetSpeakerAssignmentOptions({ autoMatch: true });
  renderTargetSpeakerContext();
  updateDiarizationPlayhead(state.player.currentTime || 0, totalAudioDuration);
}

function detectTurnOverlaps() {
  const turns = state.diarization.turns;
  turns.forEach(t => { t.has_overlap = Boolean(t.overlaps_other_speaker); });

  for (let i = 0; i < turns.length; i++) {
    for (let j = i + 1; j < turns.length; j++) {
      const a = turns[i];
      const b = turns[j];
      const overlapStart = Math.max(a.start_s, b.start_s);
      const overlapEnd = Math.min(a.end_s, b.end_s);
      if (overlapEnd > overlapStart + 0.01) {
        a.has_overlap = true;
        b.has_overlap = true;
      }
    }
  }
}

function seekDiarAudio(timeSec, andPlay = false) {
  const audioId = state.diarization.audioId || (el.diarInputSelect ? el.diarInputSelect.value : null);
  if (audioId) {
    if (!el.audio.src || !el.audio.src.includes(audioId)) {
      loadAudioIntoPlayer(audioId);
    }
  }
  const dur = state.diarization.duration || 1;
  const targetTime = Math.max(0, Math.min(Number(timeSec) || 0, dur));
  seekTo(targetTime);
  updateDiarizationPlayhead(targetTime, dur);
  if (andPlay) {
    el.audio.play().catch(() => {});
  }
}

function getTrackTimeFromClientX(clientX) {
  const tracksArea = el.diarTracksArea;
  if (!tracksArea) return 0;
  const rect = tracksArea.getBoundingClientRect();
  const dur = state.diarization.duration || 1;
  const relX = Math.max(0, Math.min(rect.width, clientX - rect.left));
  return (relX / rect.width) * dur;
}

function setupDiarTimelineSeek() {
  const tracksArea = el.diarTracksArea;
  if (!tracksArea) return;

  window.addEventListener('mousemove', (e) => {
    if (!state.diarization.isScrubbing) return;
    const dur = state.diarization.duration || 1;
    const scrubSec = getTrackTimeFromClientX(e.clientX);
    updateDiarizationPlayhead(scrubSec, dur);
    if (el.audio && !isNaN(scrubSec)) {
      el.audio.currentTime = Math.max(0, Math.min(scrubSec, dur));
    }
  });

  window.addEventListener('mouseup', () => {
    if (state.diarization.isScrubbing) {
      state.diarization.isScrubbing = false;
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    }
  });

  tracksArea.addEventListener('mousedown', (e) => {
    if (e.target.closest('.diar-turn-segment')) return;
    const clickSec = getTrackTimeFromClientX(e.clientX);
    state.diarization.isScrubbing = true;
    document.body.style.cursor = 'ew-resize';
    document.body.style.userSelect = 'none';
    seekDiarAudio(clickSec);
  });

  if (el.diarPlayheadHandle) {
    el.diarPlayheadHandle.addEventListener('mousedown', (e) => {
      e.stopPropagation();
      e.preventDefault();
      state.diarization.isScrubbing = true;
      document.body.style.cursor = 'ew-resize';
      document.body.style.userSelect = 'none';
    });
  }
}

function renderDiarRuler() {
  if (!el.diarRulerTrack) return;
  const dur = state.diarization.duration || 1;
  el.diarRulerTrack.innerHTML = "";

  const windowRange = diarWaveformWindow();
  const visibleSpan = Math.max(0, windowRange.end - windowRange.start);
  const labelWidth = el.diarLaneLabelsCol?.offsetWidth || 0;
  const visibleWidth = Math.max(1, (el.diarMultitrackViewport?.clientWidth || 800) - labelWidth);
  const useWindow = visibleSpan > 0 && visibleSpan < dur;
  const viewStart = useWindow ? windowRange.start : 0;
  const viewEnd = useWindow ? windowRange.end : dur;
  const span = Math.max(1e-6, viewEnd - viewStart);
  const pad = useWindow ? span * 0.25 : 0;
  const start = Math.max(0, viewStart - pad);
  const end = Math.min(dur, viewEnd + pad);
  const pixelsPerSec = visibleWidth / span;

  const candidateSteps = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600];
  const stepSec = candidateSteps.find(s => s * pixelsPerSec >= 65) || 60;
  const first = Math.ceil(start / stepSec - 1e-9) * stepSec;
  const tickLabel = roundedT => {
    if (stepSec < 0.01) return `${roundedT.toFixed(3)}s`;
    if (stepSec < 1) return `${roundedT.toFixed(2)}s`;
    return formatTime(roundedT);
  };

  for (let i = 0; i < 80; i += 1) {
    const t = first + i * stepSec;
    if (t > end + stepSec * 0.25) break;
    const roundedT = Math.round(t * 1000) / 1000;
    const pct = Math.min(100, (roundedT / dur) * 100);
    const tick = document.createElement("div");
    tick.className = "diar-ruler-tick";
    tick.style.left = `${pct}%`;
    if (pct > 96) {
      tick.style.transform = "translateX(-100%)";
      tick.style.borderLeft = "none";
      tick.style.borderRight = "1px solid var(--border-subtle)";
      tick.style.paddingRight = "4px";
      tick.style.paddingLeft = "0";
    }
    tick.textContent = tickLabel(roundedT);
    el.diarRulerTrack.appendChild(tick);

    if (stepSec >= 1 && stepSec <= 10) {
      const subT = roundedT + stepSec / 2;
      if (subT < dur && subT <= end) {
        const subTick = document.createElement("div");
        subTick.className = "diar-ruler-subtick";
        subTick.style.left = `${(subT / dur) * 100}%`;
        el.diarRulerTrack.appendChild(subTick);
      }
    }
  }
}

function renderDiarWaveform() {
  const canvas = el.diarWaveformCanvas;
  if (!canvas || !el.diarWaveformTrack) return;
  const viewport = el.diarMultitrackViewport;
  const labelWidth = el.diarLaneLabelsCol?.offsetWidth || 0;
  const visibleWidth = Math.max(1, (viewport?.clientWidth || el.diarWaveformTrack.clientWidth) - labelWidth);
  const left = Math.max(0, viewport?.scrollLeft || 0);
  canvas.style.left = `${left}px`;
  canvas.style.width = `${visibleWidth}px`;
  drawWaveformEnvelope(canvas, state.diarization.waveform, 'Waveform unavailable');
}

function diarWaveformWindow() {
  const duration = state.diarization.duration || 0;
  const trackWidth = Math.max(1, el.diarTracksArea?.clientWidth || 1);
  const labelWidth = el.diarLaneLabelsCol?.offsetWidth || 0;
  const visibleWidth = Math.max(1, (el.diarMultitrackViewport?.clientWidth || trackWidth) - labelWidth);
  const left = Math.max(0, el.diarMultitrackViewport?.scrollLeft || 0);
  const start = Math.min(duration, left / trackWidth * duration);
  return { start, end: Math.min(duration, start + visibleWidth / trackWidth * duration) };
}

function scheduleDiarWaveform() {
  clearTimeout(state.diarization.waveform.timer);
  state.diarization.waveform.timer = setTimeout(() => {
    const audioId = state.diarization.waveform.audioId || state.diarization.audioId;
    const windowRange = diarWaveformWindow();
    renderDiarWaveform();
    if (!audioId || !(windowRange.end > windowRange.start)) return;
    requestWaveformWindow({
      audioId,
      canvas: el.diarWaveformCanvas,
      start: windowRange.start,
      end: windowRange.end,
      view: state.diarization.waveform,
      draw: renderDiarWaveform,
    });
  }, 80);
}

function renderSpeakerSwimlanes() {
  if (!el.diarSpeakerLanesWrap || !el.diarSpkLabelsWrap) return;
  el.diarSpeakerLanesWrap.innerHTML = "";
  el.diarSpkLabelsWrap.innerHTML = "";
  const dur = state.diarization.duration || 1;
  const isSoloActive = Boolean(state.diarization.soloSpeaker);
  const visibleTurns = getFilteredAndSortedTurns();
  const visibleSpeakerIds = new Set(visibleTurns.map(turn => turn.speaker_id));

  state.diarization.speakers.filter(spk => visibleSpeakerIds.has(spk.speaker_id)).forEach(spk => {
    const spkId = spk.speaker_id;
    const color = getSpeakerColor(spkId);
    const spkName = getSpeakerName(spkId);
    const spkTurns = visibleTurns.filter(t => t.speaker_id === spkId);
    const spkTotalSpeech = spkTurns.reduce((acc, t) => acc + Math.max(0, t.end_s - t.start_s), 0);
    const isSolo = state.diarization.soloSpeaker === spkId;
    const isMuted = state.diarization.mutedSpeakers.has(spkId);
    const isDimmed = (isSoloActive && !isSolo) || isMuted;

    const labelRow = document.createElement("div");
    labelRow.className = `diar-spk-label-row ${isDimmed ? 'lane-dimmed' : ''}`;
    labelRow.dataset.speakerId = spkId;
    labelRow.innerHTML = `
      <div class="spk-label-left" title="${escapeHtml(spkName)}">
        <span class="spk-color-indicator" style="background-color: ${color}; width: 12px; height: 12px; border-radius: 50%; box-shadow: 0 0 6px ${color}; flex-shrink: 0;"></span>
        <div class="spk-name-wrap">
          <span class="lane-spk-name" style="color: ${color};" title="${escapeHtml(spkName)}">${escapeHtml(spkName)}</span>
          <span class="spk-stats-sub">${spkTurns.length} turns • ${spkTotalSpeech.toFixed(1)}s</span>
        </div>
      </div>
      <div class="spk-label-controls">
        <button type="button" class="spk-ctrl-btn btn-solo ${isSolo ? 'active' : ''}" data-speaker="${spkId}" title="Solo speaker">S</button>
        <button type="button" class="spk-ctrl-btn btn-mute ${isMuted ? 'active' : ''}" data-speaker="${spkId}" title="Mute speaker">M</button>
      </div>
    `;
    labelRow.querySelector('.btn-solo').addEventListener('click', (e) => {
      e.stopPropagation();
      toggleSpeakerSolo(spkId);
    });
    labelRow.querySelector('.btn-mute').addEventListener('click', (e) => {
      e.stopPropagation();
      toggleSpeakerMute(spkId);
    });
    el.diarSpkLabelsWrap.appendChild(labelRow);

    const track = document.createElement("div");
    track.className = `diar-speaker-lane-track ${isDimmed ? 'lane-dimmed' : ''}`;
    track.dataset.speaker = spkId;

    spkTurns.forEach(turn => {
      const idx = turn.originalIndex;
      const leftPct = (turn.start_s / dur) * 100;
      const widthPct = (Math.max(0, turn.end_s - turn.start_s) / dur) * 100;

      const seg = document.createElement("div");
      seg.className = `diar-turn-segment ${turn.has_overlap ? 'has-overlap' : ''} ${state.diarization.activeTurnIndex === idx ? 'active-turn' : ''} ${isDimmed ? 'turn-dimmed' : ''}`;
      seg.style.left = `${leftPct}%`;
      seg.style.width = `${widthPct}%`;
      seg.style.borderTop = `3px solid ${color}`;
      seg.style.borderBottom = `3px solid ${color}`;
      seg.style.borderLeft = `2px solid ${color}`;
      seg.style.borderRight = `2px solid ${color}`;
      seg.style.backgroundColor = `${color}28`;
      seg.style.boxShadow = `0 0 12px ${color}45`;
      seg.dataset.index = idx;
      seg.innerHTML = `<span class="turn-label-text">${(turn.end_s - turn.start_s).toFixed(1)}s</span>`;

      attachTurnSegmentEvents(seg, turn, idx);
      track.appendChild(seg);
    });

    el.diarSpeakerLanesWrap.appendChild(track);
  });
}

function attachTurnSegmentEvents(segEl, turn, idx) {
  segEl.addEventListener('mouseenter', (e) => {
    showTurnTooltip(e, turn);
  });

  segEl.addEventListener('mouseleave', () => {
    hideTurnTooltip();
  });

  segEl.addEventListener('click', (e) => {
    e.stopPropagation();
    state.diarization.activeTurnIndex = idx;
    highlightActiveTurn(idx);

    const row = document.getElementById(`turn-row-${idx}`);
    if (row) {
      document.querySelectorAll('.diar-turns-table tr').forEach(r => r.classList.remove('selected-row'));
      row.classList.add('selected-row');
      row.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    const audioId = state.diarization.audioId || el.diarInputSelect.value;
    if (audioId) {
      if (!el.audio.src || el.audio.src.indexOf(audioId) === -1) loadAudioIntoPlayer(audioId);
      seekTo(turn.start_s);
      el.audio.play().catch(() => {});
    }
  });
}

function showTurnTooltip(e, turn) {
  const tooltip = el.diarTurnTooltip;
  if (!tooltip) return;
  const spkName = getSpeakerName(turn.speaker_id);
  const dur = (turn.end_s - turn.start_s).toFixed(2);

  tooltip.innerHTML = `
    <strong>${escapeHtml(spkName)}</strong> <span style="opacity:0.7;">(${turn.speaker_id})</span><br>
    Start: <code>${turn.start_s.toFixed(2)}s</code> • End: <code>${turn.end_s.toFixed(2)}s</code><br>
    Duration: <strong>${dur}s</strong> ${turn.has_overlap ? '• <span style="color:var(--accent-amber);">⚠️ Overlap</span>' : ''}
  `;

  tooltip.classList.remove('hidden');
  const tooltipWidth = tooltip.offsetWidth || 180;
  const tooltipHeight = tooltip.offsetHeight || 60;
  let posX = e.clientX + 14;
  let posY = e.clientY + 14;

  if (posX + tooltipWidth > window.innerWidth - 10) {
    posX = Math.max(10, e.clientX - tooltipWidth - 14);
  }
  if (posY + tooltipHeight > window.innerHeight - 10) {
    posY = Math.max(10, e.clientY - tooltipHeight - 14);
  }

  tooltip.style.left = `${posX}px`;
  tooltip.style.top = `${posY}px`;
}

function hideTurnTooltip() {
  if (el.diarTurnTooltip) el.diarTurnTooltip.classList.add('hidden');
}

function highlightActiveTurn(idx) {
  document.querySelectorAll('.diar-turn-segment').forEach(s => {
    s.classList.toggle('active-turn', parseInt(s.dataset.index, 10) === idx);
  });
}

function isSpeakerLaneAudible(speakerId) {
  if (state.diarization.soloSpeaker && state.diarization.soloSpeaker !== speakerId) {
    return false;
  }
  if (state.diarization.mutedSpeakers.has(speakerId)) {
    return false;
  }
  return true;
}

function turnContainsTime(turn, timeSec) {
  return timeSec >= turn.start_s && timeSec < turn.end_s;
}

function audibleTurnsAtTime(timeSec) {
  return state.diarization.turns.filter(
    (turn) => turnContainsTime(turn, timeSec) && isSpeakerLaneAudible(turn.speaker_id),
  );
}

function isDiarTimeAudible(timeSec) {
  return audibleTurnsAtTime(timeSec).length > 0;
}

function diarIsolationActive() {
  return Boolean(state.diarization.soloSpeaker)
    || state.diarization.mutedSpeakers.size > 0
    || Boolean(state.diarization.autoAdvance);
}

function findNextAudibleTurn(afterTime, { wrap = false } = {}) {
  const audible = state.diarization.turns
    .filter((turn) => isSpeakerLaneAudible(turn.speaker_id) && turn.end_s - turn.start_s > 0.02)
    .sort((a, b) => a.start_s - b.start_s || a.end_s - b.end_s);
  if (audible.length === 0) return null;
  const next = audible.find((turn) => turn.start_s > afterTime + 0.02);
  if (next) return next;
  return wrap ? audible[0] : null;
}

let diarPlayRaf = 0;
let diarSegmentJumping = false;

function stopDiarPlaybackWatch() {
  if (diarPlayRaf) {
    cancelAnimationFrame(diarPlayRaf);
    diarPlayRaf = 0;
  }
}

function startDiarPlaybackWatch() {
  if (diarPlayRaf || !el.audio || el.audio.paused) return;
  if (state.activeTab !== 'tab-diarization') return;
  if (!state.diarization.turns.length || !diarIsolationActive()) return;

  const tick = () => {
    diarPlayRaf = 0;
    if (!el.audio || el.audio.paused) return;
    if (!state.diarization.turns.length || !diarIsolationActive()) return;
    const t = el.audio.currentTime || 0;
    const dur = state.diarization.duration || state.player.duration || 1;
    updateDiarizationPlayhead(t, dur);
    diarPlayRaf = requestAnimationFrame(tick);
  };
  diarPlayRaf = requestAnimationFrame(tick);
}

function prepareDiarPlaybackGate() {
  if (state.activeTab !== 'tab-diarization') return true;
  if (!el.audio || !state.diarization.turns.length) return true;
  const t = el.audio.currentTime || 0;
  applySpeakerSoloMuteAudio(t);
  return maybeAutoAdvanceSegment(t, { evenIfPaused: true });
}

function jumpToDiarTurn(turn) {
  if (!el.audio || !turn) return;
  diarSegmentJumping = true;
  el.audio.muted = true;
  seekTo(turn.start_s);
  applySpeakerSoloMuteAudio(turn.start_s);
  const idx = state.diarization.turns.indexOf(turn);
  if (idx >= 0) {
    state.diarization.activeTurnIndex = idx;
    highlightActiveTurn(idx);
  }
  requestAnimationFrame(() => {
    diarSegmentJumping = false;
  });
}

function maybeAutoAdvanceSegment(currentTime, { evenIfPaused = false } = {}) {
  if (!state.diarization.autoAdvance) return true;
  if (!el.audio || state.diarization.isScrubbing || diarSegmentJumping) return true;
  if (!evenIfPaused && el.audio.paused) return true;
  if (isDiarTimeAudible(currentTime)) return true;

  const next = findNextAudibleTurn(currentTime, { wrap: Boolean(state.player.loop) });
  if (next) {
    jumpToDiarTurn(next);
    return true;
  }
  if (!el.audio.paused) el.audio.pause();
  applySpeakerSoloMuteAudio(currentTime);
  return false;
}

function updateDiarizationPlayhead(currentTime, totalDuration) {
  if (!el.diarPlayheadLine) return;
  const dur = totalDuration || state.diarization.duration || 1;
  const pct = Math.min(100, Math.max(0, (currentTime / dur) * 100));

  el.diarPlayheadLine.style.left = `${pct}%`;

  if (el.diarTimeCurrent) el.diarTimeCurrent.textContent = formatTimePrecise(currentTime);
  if (el.diarTimeTotal) el.diarTimeTotal.textContent = formatTimePrecise(dur);
  if (state.activeTab !== 'tab-diarization') return;

  const activeTurn = audibleTurnsAtTime(currentTime)[0]
    || state.diarization.turns.find((t) => turnContainsTime(t, currentTime));
  if (activeTurn) {
    const idx = state.diarization.turns.indexOf(activeTurn);
    highlightActiveTurn(idx);

    document.querySelectorAll('.diar-turns-table tr').forEach(r => r.classList.remove('playing-row'));
    const activeRow = document.getElementById(`turn-row-${idx}`);
    if (activeRow) activeRow.classList.add('playing-row');
  }

  applySpeakerSoloMuteAudio(currentTime);
  if (!activeTurnPreviewKey) maybeAutoAdvanceSegment(currentTime);
}

function applySpeakerSoloMuteAudio(currentTime) {
  if (!el.audio) return;
  const isolationOn = Boolean(state.diarization.soloSpeaker) || state.diarization.mutedSpeakers.size > 0;
  if (!isolationOn) {
    if (!state.diarization.autoAdvance) {
      el.audio.muted = false;
    } else if (!isDiarTimeAudible(currentTime)) {
      el.audio.muted = true;
    } else {
      el.audio.muted = false;
    }
    return;
  }

  // Solo: mute unless that speaker is talking right now, including gaps.
  // Mute-only: mute when every active turn belongs to a muted speaker, and
  // also mute gaps if every speaker is muted.
  el.audio.muted = !isDiarTimeAudible(currentTime);
}

function toggleSpeakerSolo(speakerId) {
  if (state.diarization.soloSpeaker === speakerId) {
    state.diarization.soloSpeaker = null;
    showToast(`Solo disabled for ${getSpeakerName(speakerId)}`, 'info');
  } else {
    state.diarization.soloSpeaker = speakerId;
    showToast(`Soloing ${getSpeakerName(speakerId)}`, 'success');
  }
  renderSpeakerSwimlanes();
  renderSpeakerProfiles();
  applySpeakerSoloMuteAudio(el.audio?.currentTime || 0);
  maybeAutoAdvanceSegment(el.audio?.currentTime || 0);
  startDiarPlaybackWatch();
}

function toggleSpeakerMute(speakerId) {
  if (state.diarization.mutedSpeakers.has(speakerId)) {
    state.diarization.mutedSpeakers.delete(speakerId);
    showToast(`Unmuted ${getSpeakerName(speakerId)}`, 'info');
  } else {
    state.diarization.mutedSpeakers.add(speakerId);
    showToast(`Muted ${getSpeakerName(speakerId)}`, 'info');
  }
  renderSpeakerSwimlanes();
  renderSpeakerProfiles();
  applySpeakerSoloMuteAudio(el.audio?.currentTime || 0);
  maybeAutoAdvanceSegment(el.audio?.currentTime || 0);
  startDiarPlaybackWatch();
}

function renderSpeakerProfiles() {
  if (!el.diarSpeakersGrid) return;
  el.diarSpeakersGrid.innerHTML = "";
  const totalDur = state.diarization.duration || 1;

  state.diarization.speakers.forEach(spk => {
    const spkId = spk.speaker_id;
    const spkName = getSpeakerName(spkId);
    const isEnrolled = Boolean(spk.global_speaker_id);
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
        <input type="text" class="diar-spk-name-input" value="${escapeHtml(spkName)}" title="${isEnrolled ? 'Injected enrolled identity' : 'Display name for export'}" data-speaker="${spkId}" ${isEnrolled ? 'readonly' : ''}>
        ${isEnrolled ? '<span class="badge badge-accent">Enrolled</span>' : ''}
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
        <button type="button" class="btn btn-xs ${isSolo ? 'btn-solo-spk active' : 'btn-ghost'} btn-solo-spk" data-speaker="${spkId}" title="Solo speaker">Solo</button>
        <button type="button" class="btn btn-xs ${isMuted ? 'btn-mute-spk active' : 'btn-ghost'} btn-mute-spk" data-speaker="${spkId}" title="Mute speaker">Mute</button>
        <button type="button" class="btn btn-xs btn-ghost btn-filter-spk" data-speaker="${spkId}" title="Filter timeline and turns">🔍 Filter</button>
        <button type="button" class="btn btn-xs btn-primary btn-extract-spk" data-speaker="${spkId}" title="Extract and save speaker audio to workspace">✂ Extract</button>
      </div>
    `;

    const nameInput = card.querySelector('.diar-spk-name-input');
    nameInput.addEventListener('change', (e) => {
      const val = e.target.value.trim() || spkId;
      state.diarization.customNames[spkId] = val;
      renderSpeakerSwimlanes();
      renderTurnsTable();
      renderTargetSpeakerAssignmentOptions({ autoMatch: !state.targetSpeaker.assignedSpeakerId });
      renderTargetSpeakerContext();
      renderTargetSpeakerResults();
      updateActiveDiarizationHistory();
      showToast(`Speaker ${spkId} renamed to "${val}"`, "success");
    });

    card.querySelector('.btn-solo-spk').addEventListener('click', () => toggleSpeakerSolo(spkId));
    card.querySelector('.btn-mute-spk').addEventListener('click', () => toggleSpeakerMute(spkId));
    card.querySelector('.btn-filter-spk').addEventListener('click', () => {
      if (el.diarFilterSpeakerSelect) {
        el.diarFilterSpeakerSelect.value = spkId;
        state.diarization.activeSpeakerFilter = spkId;
        renderDiarizationFilteredViews();
      }
    });
    card.querySelector('.btn-extract-spk').addEventListener('click', () => extractSpeakerAudio(spkId, spkName));

    el.diarSpeakersGrid.appendChild(card);
  });
}

function getFilteredAndSortedTurns() {
  let turns = state.diarization.turns.map((t, idx) => ({ ...t, originalIndex: idx }));

  if (state.diarization.activeSpeakerFilter && state.diarization.activeSpeakerFilter !== 'all') {
    turns = turns.filter(t => t.speaker_id === state.diarization.activeSpeakerFilter);
  }

  if (state.diarization.searchQuery) {
    const q = state.diarization.searchQuery;
    turns = turns.filter(t => {
      const spkName = getSpeakerName(t.speaker_id).toLowerCase();
      return spkName.includes(q) || t.speaker_id.toLowerCase().includes(q) || `#${t.originalIndex + 1}`.includes(q) || t.start_s.toString().includes(q);
    });
  }

  if (state.diarization.minDurFilter > 0) {
    turns = turns.filter(t => (t.end_s - t.start_s) >= state.diarization.minDurFilter);
  }

  if (state.diarization.maxDurFilter > 0) {
    turns = turns.filter(t => (t.end_s - t.start_s) <= state.diarization.maxDurFilter);
  }

  if (state.diarization.overlapFilter) {
    turns = turns.filter(t => !t.has_overlap);
  }

  if (state.diarization.targetMatchFilter !== 'all') {
    turns = turns.filter(turn => {
      const segment = findTargetScoredSegment(turn);
      if (!segment) return false;
      const proposed = isTargetSegmentProposed(segment);
      return state.diarization.targetMatchFilter === 'proposed' ? proposed : !proposed;
    });
  }

  if (state.diarization.reviewFilter !== 'all') {
    turns = turns.filter(turn => turnReviewLabel(turn) === state.diarization.reviewFilter);
  }

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

function clearDiarizationTurnFilters() {
  state.diarization.activeSpeakerFilter = 'all';
  state.diarization.searchQuery = '';
  state.diarization.minDurFilter = 0;
  state.diarization.maxDurFilter = 0;
  state.diarization.overlapFilter = false;
  state.diarization.targetMatchFilter = 'all';
  state.diarization.reviewFilter = 'all';

  if (el.diarFilterSpeakerSelect) el.diarFilterSpeakerSelect.value = 'all';
  if (el.diarTurnsSearchInput) el.diarTurnsSearchInput.value = '';
  if (el.diarFilterMinDur) el.diarFilterMinDur.value = '';
  if (el.diarFilterMaxDur) el.diarFilterMaxDur.value = '';
  if (el.diarFilterTargetSelect) el.diarFilterTargetSelect.value = 'all';
  if (el.diarFilterReviewSelect) el.diarFilterReviewSelect.value = 'all';
  if (el.btnDiarFilterOverlaps) {
    el.btnDiarFilterOverlaps.classList.remove('active');
    el.btnDiarFilterOverlaps.setAttribute('aria-pressed', 'false');
  }
  renderDiarizationFilteredViews();
}

function renderDiarizationFilteredViews() {
  renderSpeakerSwimlanes();
  renderTurnsTable();
}

function turnReviewLabel(turn) {
  const scoredSegment = findTargetScoredSegment(turn);
  return state.targetSpeaker.labels[targetSegmentKey(scoredSegment || turn)] || 'unreviewed';
}

function getTurnReviewStats() {
  let accepted = 0;
  let rejected = 0;
  for (const turn of state.diarization.turns) {
    const label = turnReviewLabel(turn);
    if (label === 'qualified') accepted += 1;
    else if (label === 'rejected') rejected += 1;
  }
  return { accepted, rejected, reviewed: accepted + rejected, total: state.diarization.turns.length };
}

function renderTurnReviewStats(visibleCount) {
  const stats = getTurnReviewStats();
  if (el.diarFilteredTurnsCount) {
    el.diarFilteredTurnsCount.textContent = `${visibleCount} of ${stats.total} turns`;
  }
  if (el.diarReviewedCount) {
    el.diarReviewedCount.textContent = `${stats.reviewed} reviewed by me`;
  }
  if (el.diarAcceptedCount) {
    el.diarAcceptedCount.textContent = `${stats.accepted} accepted`;
  }
  if (el.diarRejectedCount) {
    el.diarRejectedCount.textContent = `${stats.rejected} rejected`;
  }
  if (el.btnDownloadFilteredTurns) {
    el.btnDownloadFilteredTurns.disabled = visibleCount === 0 || el.btnDownloadFilteredTurns.dataset.busy === '1';
  }
  if (el.btnDownloadFilteredTurnsLabel && el.btnDownloadFilteredTurns?.dataset.busy !== '1') {
    el.btnDownloadFilteredTurnsLabel.textContent = visibleCount === 1
      ? 'Download 1 Filtered Turn'
      : `Download ${visibleCount} Filtered Turns`;
  }
}

function diarizationDurationHistogram(turns, requestedBinWidth = 'auto') {
  const durations = turns.map(turn => {
    const duration = Number(turn.end_s) - Number(turn.start_s);
    return Number.isFinite(duration) ? Math.max(0, duration) : 0;
  });
  if (durations.length === 0) return null;

  const minDuration = Math.min(...durations);
  const maxDuration = Math.max(...durations);
  let binWidth = 0.5;
  let firstBinStart = 0;
  let binCount = 1;

  const isAuto = !requestedBinWidth || requestedBinWidth === 'auto';

  if (isAuto) {
    while (true) {
      firstBinStart = Math.floor(minDuration / binWidth) * binWidth;
      const coveredWidths = (maxDuration - firstBinStart) / binWidth;
      binCount = Math.max(1, Math.ceil(coveredWidths - 1e-10));
      if (binCount <= 24) break;

      const exponent = Math.floor(Math.log10(binWidth));
      const magnitude = 10 ** exponent;
      const leading = binWidth / magnitude;
      if (leading < 1) binWidth = magnitude;
      else if (leading < 2) binWidth = 2 * magnitude;
      else if (leading < 5) binWidth = 5 * magnitude;
      else binWidth = 10 * magnitude;
    }
  } else {
    binWidth = Math.max(0.01, Number(requestedBinWidth) || 1);
    firstBinStart = Math.max(0, Math.floor(minDuration / binWidth) * binWidth);
    const coveredWidths = (maxDuration - firstBinStart) / binWidth;
    binCount = Math.max(1, Math.ceil(coveredWidths - 1e-10));
    const MAX_BINS = 250;
    if (binCount > MAX_BINS) {
      binWidth = Math.ceil(((maxDuration - firstBinStart) / MAX_BINS) * 10) / 10;
      firstBinStart = Math.max(0, Math.floor(minDuration / binWidth) * binWidth);
      binCount = Math.max(1, Math.ceil(((maxDuration - firstBinStart) / binWidth) - 1e-10));
    }
  }

  firstBinStart = Math.max(0, Number(firstBinStart.toFixed(6)));

  const decimalsCount = (binWidth.toString().split('.')[1] || '').length;
  const factor = Math.pow(10, Math.min(4, Math.max(2, decimalsCount + 1)));

  const bins = Array.from({ length: binCount }, (_, index) => {
    const rawStart = firstBinStart + (index * binWidth);
    const rawEnd = firstBinStart + ((index + 1) * binWidth);
    return {
      start: Math.round(rawStart * factor) / factor,
      end: Math.round(rawEnd * factor) / factor,
      count: 0,
    };
  });

  durations.forEach(duration => {
    const index = Math.min(
      bins.length - 1,
      Math.max(0, Math.floor(((duration - firstBinStart) / binWidth) + 1e-10)),
    );
    bins[index].count += 1;
  });

  return { bins, binWidth, minDuration, maxDuration, isAuto };
}

function formatHistogramSeconds(value, binWidth) {
  let decimals = 0;
  if (binWidth < 0.1) {
    decimals = 2;
  } else if (binWidth < 1 || !Number.isInteger(binWidth)) {
    const str = binWidth.toString();
    const dot = str.indexOf('.');
    decimals = dot >= 0 ? Math.min(2, str.length - dot - 1) : 1;
  }
  return Number(value.toFixed(decimals)).toString();
}

function renderTurnsHistogramOnly() {
  const turns = getFilteredAndSortedTurns();
  renderDiarizationDurationHistogram(turns);
}

function renderDiarizationDurationHistogram(turns) {
  if (!el.diarDurationHistogramPlot) return;

  let targetBinWidth = state.diarization.histogramBinWidth;
  if (targetBinWidth === 'custom') {
    targetBinWidth = state.diarization.histogramCustomBinWidth || 1.0;
  }

  const histogram = diarizationDurationHistogram(turns, targetBinWidth);
  if (!histogram) {
    el.diarDurationHistogramPlot.innerHTML = '<div class="diar-duration-histogram-empty">No turns match the active filters.</div>';
    if (el.diarDurationHistogramSummary) el.diarDurationHistogramSummary.textContent = '0 segments';
    if (el.diarHistogramBinSelect) {
      const autoOption = el.diarHistogramBinSelect.querySelector('option[value="auto"]');
      if (autoOption) autoOption.textContent = 'Auto';
    }
    return;
  }

  const { bins, binWidth, isAuto } = histogram;
  const minSlotWidth = 18;
  const idealPlotWidth = bins.length * minSlotWidth;
  const margin = { top: 24, right: 18, bottom: 58, left: 52 };
  const minSvgWidth = 900;
  const width = Math.max(minSvgWidth, margin.left + margin.right + idealPlotWidth);
  const height = 220;
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const maxCount = Math.max(...bins.map(bin => bin.count), 1);
  const slotWidth = plotWidth / bins.length;
  const barGap = Math.min(5, Math.max(1, slotWidth * 0.18));
  const maxLabels = Math.max(8, Math.min(20, Math.floor(plotWidth / 65)));
  const labelEvery = Math.max(1, Math.ceil(bins.length / maxLabels));
  const yTicks = [...new Set([0, Math.ceil(maxCount / 2), maxCount])].sort((a, b) => a - b);
  const rangeText = bin => `${formatHistogramSeconds(bin.start, binWidth)}–${formatHistogramSeconds(bin.end, binWidth)}`;

  const grid = yTicks.map(tick => {
    const y = margin.top + plotHeight - ((tick / maxCount) * plotHeight);
    return `<line class="diar-histogram-grid" x1="${margin.left}" y1="${y}" x2="${width - margin.right}" y2="${y}"></line>
      <text class="diar-histogram-axis-tick" x="${margin.left - 8}" y="${y + 3}" text-anchor="end">${tick}</text>`;
  }).join('');

  const bars = bins.map((bin, index) => {
    const x = margin.left + (index * slotWidth) + (barGap / 2);
    const barWidth = Math.max(1, slotWidth - barGap);
    const barHeight = (bin.count / maxCount) * plotHeight;
    const y = margin.top + plotHeight - barHeight;
    const upperRule = index === bins.length - 1 ? 'inclusive' : 'exclusive';
    const tooltip = `${rangeText(bin)} seconds (lower inclusive, upper ${upperRule}): ${bin.count} segment${bin.count === 1 ? '' : 's'}`;
    const showLabel = (index % labelEvery === 0 && (bins.length - 1 - index) >= Math.floor(labelEvery * 0.5)) || index === bins.length - 1;
    const showCount = bin.count > 0 && (slotWidth >= 16 || index % Math.ceil(16 / slotWidth) === 0);
    return `<g class="diar-histogram-bin">
        <title>${tooltip}</title>
        <rect class="diar-histogram-hit-area" x="${margin.left + (index * slotWidth)}" y="${margin.top}" width="${slotWidth}" height="${plotHeight}"></rect>
        <rect class="diar-histogram-bar" x="${x}" y="${y}" width="${barWidth}" height="${barHeight}" rx="2"></rect>
        ${showCount ? `<text class="diar-histogram-count" x="${x + (barWidth / 2)}" y="${Math.max(margin.top + 9, y - 5)}" text-anchor="middle">${bin.count}</text>` : ''}
        ${showLabel ? `<text class="diar-histogram-range" x="${x + (barWidth / 2)}" y="${margin.top + plotHeight + 17}" text-anchor="middle">${rangeText(bin)}</text>` : ''}
      </g>`;
  }).join('');

  const total = bins.reduce((sum, bin) => sum + bin.count, 0);
  const formattedBinWidth = formatHistogramSeconds(binWidth, binWidth);
  const unit = binWidth === 1 ? 'second' : 'seconds';
  if (el.diarDurationHistogramSummary) {
    el.diarDurationHistogramSummary.textContent = `${total} segment${total === 1 ? '' : 's'} · ${formattedBinWidth} ${unit}/bin`;
  }
  if (el.diarHistogramBinSelect) {
    const autoOption = el.diarHistogramBinSelect.querySelector('option[value="auto"]');
    if (autoOption) {
      autoOption.textContent = isAuto
        ? `Auto (${formattedBinWidth}s)`
        : 'Auto';
    }
  }
  const minWidthAttr = width > minSvgWidth ? ` style="min-width: ${width}px;"` : '';
  el.diarDurationHistogramPlot.innerHTML = `<svg viewBox="0 0 ${width} ${height}"${minWidthAttr} role="img" aria-label="Segment duration histogram with ${total} filtered turns in ${bins.length} bins of ${formattedBinWidth} seconds">
      <title>Filtered segment length distribution</title>
      ${grid}
      <line class="diar-histogram-axis" x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${margin.top + plotHeight}"></line>
      <line class="diar-histogram-axis" x1="${margin.left}" y1="${margin.top + plotHeight}" x2="${width - margin.right}" y2="${margin.top + plotHeight}"></line>
      ${bars}
      <text class="diar-histogram-axis-title" x="${margin.left + (plotWidth / 2)}" y="${height - 7}" text-anchor="middle">Duration range (seconds)</text>
      <text class="diar-histogram-axis-title" x="14" y="${margin.top + (plotHeight / 2)}" text-anchor="middle" transform="rotate(-90 14 ${margin.top + (plotHeight / 2)})">Segment count</text>
    </svg>`;
}

function renderTurnsTable() {
  if (!el.turnsTableBody) return;
  el.turnsTableBody.innerHTML = "";

  const turns = getFilteredAndSortedTurns();
  renderTurnReviewStats(turns.length);
  renderDiarizationDurationHistogram(turns);

  if (turns.length === 0) {
    el.turnsTableBody.innerHTML = `<tr><td colspan="7" class="text-center text-muted" style="padding: 24px;">No turns match the active filter criteria.</td></tr>`;
    return;
  }

  turns.forEach(turn => {
    const idx = turn.originalIndex;
    const color = getSpeakerColor(turn.speaker_id);
    const duration = Math.max(0, turn.end_s - turn.start_s).toFixed(2);
    const scoredSegment = findTargetScoredSegment(turn);
    const segmentKey = targetSegmentKey(scoredSegment || turn);
    const proposed = scoredSegment ? isTargetSegmentProposed(scoredSegment) : false;
    const label = turnReviewLabel(turn);
    const targetScore = scoredSegment
      ? `<strong>${Number(scoredSegment.similarity).toFixed(3)}</strong><small class="turn-target-status ${proposed ? 'is-proposed' : ''}">${proposed ? 'Proposed' : 'Filtered'}</small>`
      : '<span class="text-xs text-muted">Not scored</span>';
    const evaluationActions = `<div class="turn-review-actions">
          <button type="button" class="btn btn-xs ${label === 'qualified' ? 'btn-primary' : 'btn-ghost'} ts-turn-label" data-key="${escapeHtml(segmentKey)}" data-label="qualified" aria-pressed="${label === 'qualified'}" title="Accept this turn">✓ Accept</button>
          <button type="button" class="btn btn-xs ${label === 'rejected' ? 'target-label-rejected' : 'btn-ghost'} ts-turn-label" data-key="${escapeHtml(segmentKey)}" data-label="rejected" aria-pressed="${label === 'rejected'}" title="Reject this turn">✕ Reject</button>
        </div>`;

    const tr = document.createElement("tr");
    tr.id = `turn-row-${idx}`;
    if (state.diarization.activeTurnIndex === idx) tr.classList.add('selected-row');
    if (label === 'qualified') tr.classList.add('turn-row-accepted');
    if (label === 'rejected') tr.classList.add('turn-row-rejected');

    tr.innerHTML = `
      <td><span class="text-muted font-mono">#${idx + 1}</span></td>
      <td><span style="color: ${color}; font-weight: 700;">${escapeHtml(getSpeakerName(turn.speaker_id))}</span></td>
      <td class="turn-time-range"><code>${turn.start_s.toFixed(2)}<span class="turn-time-sep">–</span>${turn.end_s.toFixed(2)}</code></td>
      <td><span class="badge badge-ghost">${duration}s</span></td>
      <td>${turn.has_overlap ? '<span class="badge badge-warning">Yes</span>' : '<span class="text-xs text-muted">No</span>'}</td>
      <td class="turn-target-score">${targetScore}</td>
      <td class="table-actions">
        <div class="turn-row-actions">
          <button class="btn btn-sm btn-ghost btn-play-turn" data-index="${idx}" data-preview-key="${escapeHtml(turnPreviewKey(turn))}" title="Play the labeled turn WAV (same file as Download)">▶ Play</button>
          <button type="button" class="btn btn-sm btn-ghost btn-download-turn" data-index="${idx}" title="Download this turn WAV onto this computer">⬇ Download</button>
          <button type="button" class="btn btn-sm btn-ghost btn-save-turn-cut" data-index="${idx}" title="Save this turn as a session audio cut">Save Cut</button>
          ${evaluationActions}
        </div>
      </td>
    `;

    tr.addEventListener('click', (e) => {
      if (e.target.closest('button')) return;
      state.diarization.activeTurnIndex = idx;
      highlightActiveTurn(idx);
      document.querySelectorAll('.diar-turns-table tr').forEach(r => r.classList.remove('selected-row'));
      tr.classList.add('selected-row');
      seekDiarAudio(turn.start_s, true);
    });

    tr.querySelector('.btn-play-turn').addEventListener('click', (e) => {
      e.stopPropagation();
      playTurnExact(turn, e.currentTarget);
    });

    tr.querySelector('.btn-download-turn').addEventListener('click', (e) => {
      e.stopPropagation();
      downloadTurnAudio(turn, e.currentTarget);
    });

    tr.querySelector('.btn-save-turn-cut').addEventListener('click', (e) => {
      e.stopPropagation();
      saveTurnAsCut(turn, e.currentTarget);
    });

    tr.querySelectorAll('.ts-turn-label').forEach(button => {
      button.addEventListener('click', (e) => {
        e.stopPropagation();
        toggleTargetSegmentLabel(button.dataset.key, button.dataset.label);
      });
    });

    el.turnsTableBody.appendChild(tr);
  });
  updateTurnPreviewButtons();
}

function turnPreviewKey(turn) {
  const audioId = state.diarization.audioId || el.diarInputSelect?.value || '';
  return `${audioId}:${Number(turn.start_s).toFixed(3)}-${Number(turn.end_s).toFixed(3)}:${turn.speaker_id || ''}`;
}

function stopTurnPreviewMonitor() {
  if (turnPreviewRaf) {
    cancelAnimationFrame(turnPreviewRaf);
    turnPreviewRaf = 0;
  }
}

function stopTurnPreview() {
  stopTurnPreviewMonitor();
  turnPreviewGeneration += 1;
  activeTurnPreviewKey = null;
  turnPreviewRange = null;
  turnPreviewAudio.pause();
  turnPreviewAudio.removeAttribute('src');
  turnPreviewAudio.load();
  if (turnPreviewUrl) {
    URL.revokeObjectURL(turnPreviewUrl);
    turnPreviewUrl = null;
  }
  updateTurnPreviewButtons();
  if (!isAuditionPlaybackActive() && el.audio?.paused) setPlayingUI(false);
}

function finishTurnPreview() {
  if (turnPreviewRange) {
    const end = turnPreviewRange.end_s;
    const dur = state.diarization.duration || state.player.duration || 1;
    state.player.currentTime = end;
    if (el.audio && el.audio.readyState > 0) el.audio.currentTime = end;
    updatePlayheadPosition(end);
    updateDiarizationPlayhead(end, dur);
  }
  stopTurnPreview();
}

function updateTurnPreviewButtons() {
  document.querySelectorAll('.btn-play-turn').forEach(btn => {
    const isActive = Boolean(activeTurnPreviewKey) && btn.dataset.previewKey === activeTurnPreviewKey;
    btn.textContent = isActive ? '■ Stop' : '▶ Play';
    btn.disabled = false;
    btn.setAttribute('aria-pressed', String(isActive));
  });
}

function watchTurnPreviewPlayhead() {
  const tick = () => {
    turnPreviewRaf = 0;
    if (!turnPreviewRange || turnPreviewAudio.paused) return;
    const t = Math.min(
      turnPreviewRange.end_s,
      turnPreviewRange.start_s + (turnPreviewAudio.currentTime || 0),
    );
    const dur = state.diarization.duration || state.player.duration || 1;
    state.player.currentTime = t;
    if (el.timeCurrent) el.timeCurrent.textContent = formatTime(t);
    updatePlayheadPosition(t);
    updateDiarizationPlayhead(t, dur);
    turnPreviewRaf = requestAnimationFrame(tick);
  };
  turnPreviewRaf = requestAnimationFrame(tick);
}

async function playTurnExact(turn, button) {
  const { audioId } = diarizationSourceAudio();
  if (!audioId) {
    showToast('No active audio track selected', 'error');
    return;
  }
  const start = turn.start_s;
  const end = turn.end_s;
  if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) {
    showToast('This turn has an invalid time range', 'error');
    return;
  }

  const key = turnPreviewKey(turn);
  if (activeTurnPreviewKey === key) {
    stopTurnPreview();
    return;
  }

  stopTurnPreview();
  if (el.audio && !el.audio.paused) el.audio.pause();
  if (puritySegmentAudio && !puritySegmentAudio.paused) stopPuritySegmentPreview();

  const generation = ++turnPreviewGeneration;
  activeTurnPreviewKey = key;
  turnPreviewRange = { start_s: start, end_s: end };
  if (button) {
    button.disabled = true;
    button.textContent = 'Loading...';
  }

  try {
    const params = new URLSearchParams({
      start: String(start),
      end: String(end),
      filename: turnDownloadFilename(turn),
    });
    const res = await fetch(`/api/audio/${encodeURIComponent(audioId)}/segment?${params.toString()}`);
    if (!res.ok) await parseJsonResponse(res);
    const blob = await res.blob();
    if (generation !== turnPreviewGeneration) return;
    turnPreviewUrl = URL.createObjectURL(blob);
    turnPreviewAudio.volume = state.player.volume;
    turnPreviewAudio.playbackRate = state.player.playbackRate;
    turnPreviewAudio.src = turnPreviewUrl;
    await turnPreviewAudio.play();
    if (generation !== turnPreviewGeneration) return;
    setPlayingUI(true);
    updateTurnPreviewButtons();
    watchTurnPreviewPlayhead();
  } catch (err) {
    if (generation === turnPreviewGeneration) {
      stopTurnPreview();
      showToast(err.message || 'Unable to play this turn', 'error');
    }
  }
}

async function saveTurnAsCut(turn, button) {
  const audioId = state.diarization.audioId || el.diarInputSelect.value;
  if (!audioId) {
    showToast("No active audio track selected", "error");
    return;
  }

  const start = turn.start_s;
  const end = turn.end_s;
  if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) {
    showToast("This turn has an invalid time range", "error");
    return;
  }

  const originalLabel = button ? button.innerHTML : "Save Cut";
  if (button) {
    button.disabled = true;
    button.textContent = "Saving...";
  }

  try {
    const res = await fetch(`/api/audio/${audioId}/cut`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ start, end, unit: "seconds" }),
    });
    const data = await parseJsonResponse(res);
    await fetchAudioList();
    addCutToRegistry(data.audio_id, start, end, "seconds");
    const title = data.metadata?.title || data.audio_id;
    showToast(`Saved turn as clip ${title}`, "success");
  } catch (err) {
    showToast(err.message, "error");
  } finally {
    if (button) {
      button.disabled = false;
      button.innerHTML = originalLabel;
    }
  }
}

function diarizationSourceAudio() {
  const audioId = state.diarization.audioId || el.diarInputSelect?.value;
  const item = state.audioList.find(entry => entry.id === audioId);
  return { audioId, item };
}

function sanitizeDownloadStem(value, fallback = 'audio') {
  const cleaned = String(value || '').replace(/[^\w.-]+/g, '_').replace(/^_+|_+$/g, '');
  return cleaned.slice(0, 60) || fallback;
}

function turnDownloadFilename(turn) {
  const { item } = diarizationSourceAudio();
  const title = sanitizeDownloadStem(item?.title, 'audio');
  const speaker = sanitizeDownloadStem(getSpeakerName(turn.speaker_id), turn.speaker_id || 'spk');
  const n = String((turn.originalIndex ?? 0) + 1).padStart(3, '0');
  return `${title}_turn${n}_${speaker}_${Number(turn.start_s).toFixed(2)}-${Number(turn.end_s).toFixed(2)}.wav`;
}

function filenameFromContentDisposition(header, fallback) {
  if (!header) return fallback;
  const star = /filename\*=(?:UTF-8'')?([^;]+)/i.exec(header);
  if (star) {
    try {
      return decodeURIComponent(star[1].trim().replace(/^"+|"+$/g, ''));
    } catch (_) { /* fall through */ }
  }
  const quoted = /filename="([^"]+)"/i.exec(header);
  if (quoted) return quoted[1];
  const plain = /filename=([^;]+)/i.exec(header);
  return plain ? plain[1].trim() : fallback;
}

async function downloadResponseBlob(res, fallbackName) {
  if (!res.ok) {
    await parseJsonResponse(res);
  }
  const blob = await res.blob();
  const name = filenameFromContentDisposition(res.headers.get('Content-Disposition'), fallbackName);
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = name;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
  return name;
}

async function downloadTurnAudio(turn, button) {
  const { audioId } = diarizationSourceAudio();
  if (!audioId) {
    showToast("No active audio track selected", "error");
    return;
  }
  const start = turn.start_s;
  const end = turn.end_s;
  if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) {
    showToast("This turn has an invalid time range", "error");
    return;
  }

  const filename = turnDownloadFilename(turn);
  const originalLabel = button ? button.innerHTML : "⬇ Download";
  if (button) {
    button.disabled = true;
    button.textContent = "Downloading...";
  }

  try {
    const params = new URLSearchParams({
      start: String(start),
      end: String(end),
      filename,
    });
    const res = await fetch(`/api/audio/${encodeURIComponent(audioId)}/segment?${params.toString()}`);
    const savedAs = await downloadResponseBlob(res, filename);
    showToast(`Downloaded ${savedAs}`, "success");
  } catch (err) {
    showToast(err.message, "error");
  } finally {
    if (button) {
      button.disabled = false;
      button.innerHTML = originalLabel;
    }
  }
}

async function downloadFilteredTurns() {
  const { audioId, item } = diarizationSourceAudio();
  if (!audioId) {
    showToast("No active audio track selected", "error");
    return;
  }
  const turns = getFilteredAndSortedTurns();
  if (turns.length === 0) {
    showToast("No filtered turns to download", "info");
    return;
  }

  const zipName = `${sanitizeDownloadStem(item?.title, 'audio')}_${turns.length}_turns.zip`;
  const button = el.btnDownloadFilteredTurns;
  const label = el.btnDownloadFilteredTurnsLabel;
  const originalLabel = label ? label.textContent : "Download Filtered Turns";
  if (button) {
    button.disabled = true;
    button.dataset.busy = '1';
  }
  if (label) label.textContent = turns.length === 1 ? "Preparing 1 turn…" : `Preparing ${turns.length} turns…`;

  try {
    const res = await fetch(`/api/audio/${encodeURIComponent(audioId)}/segments.zip`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        filename: zipName,
        segments: turns.map(turn => ({
          start: turn.start_s,
          end: turn.end_s,
          filename: turnDownloadFilename(turn),
        })),
      }),
    });
    const savedAs = await downloadResponseBlob(res, zipName);
    showToast(`Downloaded ${savedAs}`, "success");
  } catch (err) {
    showToast(err.message, "error");
  } finally {
    if (button) {
      button.dataset.busy = '0';
      button.disabled = getFilteredAndSortedTurns().length === 0;
    }
    if (label) {
      const visibleCount = getFilteredAndSortedTurns().length;
      label.textContent = visibleCount === 1
        ? 'Download 1 Filtered Turn'
        : `Download ${visibleCount} Filtered Turns`;
    } else if (button) {
      button.textContent = originalLabel;
    }
  }
}

function downloadDiarizationRttm() {
  const audioId = state.diarization.audioId || 'audio_sample';
  const turns = state.diarization.rawTurns || [];
  if (turns.length === 0) {
    showToast("No diarization turns to export", "info");
    return;
  }
  const content = turns.map(t => {
    const dur = (t.end_s - t.start_s).toFixed(3);
    const spkName = getSpeakerName(t.speaker_id);
    return `SPEAKER ${audioId} 1 ${t.start_s.toFixed(3)} ${dur} <NA> <NA> ${spkName} <NA> <NA>`;
  }).join('\n');
  const filename = `diarization_${audioId}.rttm`;
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

async function extractSpeakerAudio(speakerId, speakerName) {
  const audioId = state.diarization.audioId || el.diarInputSelect.value;
  if (!audioId) {
    showToast("No active audio track selected", "error");
    return;
  }

  const previewTurns = state.diarization.turns.filter(t => t.speaker_id === speakerId);
  const rawTurns = state.diarization.rawTurns || [];
  if (previewTurns.length === 0) {
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
        turns: rawTurns,
        clean_turns: state.diarization.cleanTurnsEnabled,
        settings: state.diarization.cleanTurnsSettings,
        extraction_settings: readDiarizationExtractionSettings(),
        blocker_turns: extractionBlockerTurns(),
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Failed to extract speaker audio");

    await fetchAudioList();
    showToast(`Speaker track extracted from ${data.turn_policy || 'raw'} turns: "${data.metadata?.title || speakerId}" (${data.duration_s?.toFixed(2)}s)`, "success");
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
        turns: state.diarization.rawTurns,
        clean_turns: state.diarization.cleanTurnsEnabled,
        settings: state.diarization.cleanTurnsSettings,
        extraction_settings: readDiarizationExtractionSettings(),
        speaker_names: state.diarization.customNames,
        blocker_turns: extractionBlockerTurns(),
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Failed to extract all speakers");

    await fetchAudioList();
    showToast(`Extracted ${data.total_speakers} speaker stems from ${data.turn_policy || 'raw'} turns`, "success");
  } catch (err) {
    showToast(`Extraction failed: ${err.message}`, "error");
  }
}
// ==================== KNOWN SPEAKER MANAGEMENT ====================

function initKnownSpeakerManager() {
  state.knownSpeakers = { profiles: [] };

  const profileSelect = document.getElementById('ts-profile-select');
  const refreshBtn = document.getElementById('btn-ts-refresh-profiles');
  const deleteBtn = document.getElementById('btn-ts-delete-profile');
  const createBtn = document.getElementById('btn-ts-create-profile');
  const addBtn = document.getElementById('btn-ts-add-clips');
  const enrollDetails = document.getElementById('ts-enroll-details');

  if (refreshBtn) refreshBtn.addEventListener('click', loadSpeakerProfiles);
  if (deleteBtn) deleteBtn.addEventListener('click', deleteSelectedSpeakerProfile);
  if (createBtn) createBtn.addEventListener('click', createSpeakerProfile);
  if (addBtn) addBtn.addEventListener('click', addClipsToSelectedSpeaker);
  if (profileSelect) {
    profileSelect.addEventListener('change', renderSelectedSpeakerClips);
    loadSpeakerProfiles();
  }
  if (enrollDetails) {
    enrollDetails.addEventListener('toggle', () => {
      if (enrollDetails.open) populateTargetClipSelect();
    });
  }

  document.getElementById('btn-ts-go-source')?.addEventListener('click', () => switchTab('tab-workspace'));
  document.getElementById('btn-ts-go-separation')?.addEventListener('click', () => {
    const audioId = state.diarization.audioId || el.diarInputSelect?.value || state.activeAudio?.id;
    switchTab('tab-separation');
    if (audioId && el.sepInputSelect) {
      el.sepInputSelect.value = audioId;
      el.sepInputSelect.dispatchEvent(new Event('change'));
    }
  });
  document.getElementById('btn-ts-go-enroll')?.addEventListener('click', () => {
    if (enrollDetails) {
      enrollDetails.open = true;
      populateTargetClipSelect();
      enrollDetails.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  });
  document.getElementById('btn-ts-open-cutter')?.addEventListener('click', () => {
    switchTab('tab-workspace');
    showToast('Cut clean single-speaker references, then return here to create or update the identity.', 'info');
  });
  document.getElementById('btn-ts-go-results')?.addEventListener('click', () => {
    document.getElementById('diar-enrollment-group')?.scrollIntoView({ behavior: 'smooth', block: 'center' });
  });

  populateTargetClipSelect();
}

function populateTargetClipSelect() {
  const clipSelect = document.getElementById('ts-clip-select');
  if (!clipSelect) return;
  const previous = new Set(Array.from(clipSelect.selectedOptions).map(option => option.value));
  clipSelect.innerHTML = '';
  (state.audioList || []).forEach(item => {
    const option = document.createElement('option');
    option.value = item.id;
    const duration = item.duration_s ? ` (${item.duration_s.toFixed(1)}s)` : '';
    option.textContent = `${item.title || item.id}${duration}`;
    option.selected = previous.has(item.id);
    clipSelect.appendChild(option);
  });
  if (clipSelect.options.length === 0) {
    const option = document.createElement('option');
    option.disabled = true;
    option.textContent = 'No session audio — cut reference clips in Workspace first';
    clipSelect.appendChild(option);
  }
}

async function loadSpeakerProfiles() {
  const profileSelect = document.getElementById('ts-profile-select');
  const enrollmentSelect = document.getElementById('diar-enrollment-profile-select');
  const evaluationSelect = document.getElementById('ts-eval-profile-select');
  const previousProfile = profileSelect?.value || '';
  const previousEnrollment = enrollmentSelect?.value || '';
  const previousEvaluation = evaluationSelect?.value || state.targetSpeaker.profileName || '';

  try {
    const res = await fetch('/api/speaker-profiles');
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Failed to load speakers');
    const profiles = data.profiles || [];
    state.knownSpeakers.profiles = profiles;

    if (profileSelect) {
      profileSelect.innerHTML = profiles.length
        ? profiles.map(profile => `<option value="${escapeHtml(profile.name)}">${escapeHtml(profile.name)} (${profile.num_clips} clips)</option>`).join('')
        : '<option value="">No speakers enrolled</option>';
      if (profiles.some(profile => profile.name === previousProfile)) {
        profileSelect.value = previousProfile;
      }
    }

    if (enrollmentSelect) {
      enrollmentSelect.innerHTML = '<option value="">No known speaker (ordinary diarization)</option>' +
        profiles.map(profile => `<option value="${escapeHtml(profile.name)}">${escapeHtml(profile.name)} · ${profile.num_clips} clips</option>`).join('');
      if (profiles.some(profile => profile.name === previousEnrollment)) {
        enrollmentSelect.value = previousEnrollment;
      }
    }

    if (evaluationSelect) {
      evaluationSelect.innerHTML = profiles.length
        ? '<option value="">Select a target speaker</option>' + profiles.map(profile => `<option value="${escapeHtml(profile.name)}">${escapeHtml(profile.name)} · ${profile.num_clips} clips</option>`).join('')
        : '<option value="">No known speakers enrolled</option>';
      const nextEvaluation = profiles.some(profile => profile.name === previousEvaluation)
        ? previousEvaluation
        : (profiles.some(profile => profile.name === previousProfile) ? previousProfile : '');
      evaluationSelect.value = nextEvaluation;
      if (nextEvaluation !== previousEvaluation) state.targetSpeaker.assignedSpeakerId = '';
      state.targetSpeaker.profileName = nextEvaluation;
    }

    renderSelectedSpeakerClips();
    renderTargetSpeakerAssignmentOptions({ autoMatch: true });
    renderTargetSpeakerContext();
    const activeCard = document.querySelector('.model-card[data-diar-model].active');
    syncDiarModelOptions(activeCard?.dataset.diarModel || 'pyannote_community');
  } catch (err) {
    showToast(`Failed to load known speakers: ${err.message}`, 'error');
  }
}

function renderSelectedSpeakerClips() {
  const profileSelect = document.getElementById('ts-profile-select');
  const container = document.getElementById('ts-profile-clips');
  if (!container) return;
  const profile = (state.knownSpeakers?.profiles || []).find(item => item.name === profileSelect?.value);
  if (!profile) {
    container.innerHTML = '<span class="text-xs text-muted">Select a speaker to inspect their clips.</span>';
    return;
  }

  container.innerHTML = (profile.clips || []).map((clip, index) => `
    <div class="speaker-profile-clip">
      <div><strong>Clip ${index + 1}</strong><small>${escapeHtml(clip.name)}</small></div>
      <audio controls preload="none" src="${clip.stream_url}"></audio>
      <button type="button" class="btn btn-xs btn-ghost text-destructive ts-remove-clip" data-clip="${escapeHtml(clip.name)}">Remove</button>
    </div>
  `).join('');
  container.querySelectorAll('.ts-remove-clip').forEach(button => {
    button.addEventListener('click', () => removeSpeakerClip(profile.name, button.dataset.clip));
  });
}

function selectedReferenceClipIds() {
  const clipSelect = document.getElementById('ts-clip-select');
  return clipSelect
    ? Array.from(clipSelect.selectedOptions).map(option => option.value).filter(Boolean)
    : [];
}

async function createSpeakerProfile() {
  const nameInput = document.getElementById('ts-new-profile-name');
  const name = nameInput?.value.trim() || '';
  const clipIds = selectedReferenceClipIds();
  if (!name) { showToast('Enter a speaker name', 'error'); return; }
  if (clipIds.length === 0) { showToast('Select at least one clean reference clip', 'error'); return; }

  try {
    const res = await fetch('/api/speaker-profiles', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, clip_audio_ids: clipIds }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Speaker creation failed');
    showToast(`Speaker "${data.name}" created with ${data.num_clips} clips`, 'success');
    if (nameInput) nameInput.value = '';
    await loadSpeakerProfiles();
    const profileSelect = document.getElementById('ts-profile-select');
    if (profileSelect) profileSelect.value = data.name;
    renderSelectedSpeakerClips();
  } catch (err) {
    showToast(`Speaker creation failed: ${err.message}`, 'error');
  }
}

async function addClipsToSelectedSpeaker() {
  const profileSelect = document.getElementById('ts-profile-select');
  const name = profileSelect?.value || '';
  const clipIds = selectedReferenceClipIds();
  if (!name) { showToast('Select a speaker first', 'error'); return; }
  if (clipIds.length === 0) { showToast('Select at least one clean reference clip', 'error'); return; }

  try {
    const res = await fetch(`/api/speaker-profiles/${encodeURIComponent(name)}/clips`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ clip_audio_ids: clipIds }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Adding clips failed');
    showToast(`Added clips to "${name}" · ${data.num_clips} total`, 'success');
    await loadSpeakerProfiles();
    if (profileSelect) profileSelect.value = name;
    renderSelectedSpeakerClips();
  } catch (err) {
    showToast(`Adding clips failed: ${err.message}`, 'error');
  }
}

async function removeSpeakerClip(name, clipName) {
  if (!confirm(`Remove "${clipName}" from speaker "${name}"?`)) return;
  try {
    const res = await fetch(
      `/api/speaker-profiles/${encodeURIComponent(name)}/clips/${encodeURIComponent(clipName)}`,
      { method: 'DELETE' },
    );
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Clip removal failed');
    showToast(`Removed clip from "${name}"`, 'success');
    await loadSpeakerProfiles();
    const profileSelect = document.getElementById('ts-profile-select');
    if (profileSelect) profileSelect.value = name;
    renderSelectedSpeakerClips();
  } catch (err) {
    showToast(`Clip removal failed: ${err.message}`, 'error');
  }
}

async function deleteSelectedSpeakerProfile() {
  const profileSelect = document.getElementById('ts-profile-select');
  const name = profileSelect?.value || '';
  if (!name) { showToast('Select a speaker to delete', 'error'); return; }
  if (!confirm(`Delete speaker "${name}" and all reference clips?`)) return;
  try {
    const res = await fetch(`/api/speaker-profiles/${encodeURIComponent(name)}`, { method: 'DELETE' });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Delete failed');
    showToast(`Speaker "${name}" deleted`, 'success');
    await loadSpeakerProfiles();
  } catch (err) {
    showToast(`Delete failed: ${err.message}`, 'error');
  }
}

// ==================== TARGET SPEAKER EVALUATION ====================

function initTargetSpeakerEvaluation() {
  const profileSelect = document.getElementById('ts-eval-profile-select');
  const assignmentSelect = document.getElementById('ts-eval-diar-speaker-select');
  const scoreButton = document.getElementById('btn-ts-score');

  profileSelect?.addEventListener('change', () => {
    state.targetSpeaker.profileName = profileSelect.value;
    resetTargetSpeakerEvaluation({ preserveSelection: true, preserveLabels: true });
    state.targetSpeaker.assignedSpeakerId = '';
    renderTargetSpeakerAssignmentOptions({ autoMatch: true });
    renderTargetSpeakerContext();
  });
  assignmentSelect?.addEventListener('change', () => {
    state.targetSpeaker.assignedSpeakerId = assignmentSelect.value;
    renderTargetSpeakerContext();
    renderTargetSpeakerResults();
  });
  scoreButton?.addEventListener('click', runTargetSpeakerScore);

  renderTargetSpeakerAssignmentOptions({ autoMatch: true });
  renderTargetSpeakerContext();
}

function normalizedSpeakerName(value) {
  return String(value || '').trim().toLocaleLowerCase();
}

function findNamedDiarizedSpeaker(profileName) {
  const targetName = normalizedSpeakerName(profileName);
  if (!targetName) return null;
  const speakers = state.diarization.speakers || [];
  return speakers.find(speaker => normalizedSpeakerName(speaker.global_speaker_id) === targetName)
    || speakers.find(speaker => normalizedSpeakerName(getSpeakerName(speaker.speaker_id)) === targetName)
    || speakers.find(speaker => normalizedSpeakerName(speaker.speaker_id) === targetName)
    || null;
}

function renderTargetSpeakerAssignmentOptions({ autoMatch = false } = {}) {
  const select = document.getElementById('ts-eval-diar-speaker-select');
  if (!select) return;
  const speakers = state.diarization.speakers || [];
  const current = state.targetSpeaker.assignedSpeakerId;
  const profileName = document.getElementById('ts-eval-profile-select')?.value || state.targetSpeaker.profileName;
  const namedMatch = findNamedDiarizedSpeaker(profileName);
  const currentIsValid = speakers.some(speaker => speaker.speaker_id === current);
  const next = currentIsValid
    ? current
    : (autoMatch && namedMatch ? namedMatch.speaker_id : '');

  select.innerHTML = '<option value="">No named match — score all turns independently</option>' + speakers.map(speaker => {
    const displayName = getSpeakerName(speaker.speaker_id);
    const identity = speaker.global_speaker_id ? ' · enrolled identity' : '';
    return `<option value="${escapeHtml(speaker.speaker_id)}">${escapeHtml(displayName)} · local ${escapeHtml(speaker.speaker_id)}${identity}</option>`;
  }).join('');
  select.value = next;
  state.targetSpeaker.assignedSpeakerId = next;

  const help = document.getElementById('ts-eval-assignment-help');
  if (help) {
    help.textContent = namedMatch
      ? `Matched “${profileName}” to local speaker ${namedMatch.speaker_id}. Verification still scores every turn independently.`
      : 'Optional: identify which local diarized speaker represents the target. Display names never replace local speaker IDs.';
  }
}

function renderTargetSpeakerContext() {
  const title = document.getElementById('ts-eval-context-title');
  const detail = document.getElementById('ts-eval-context-detail');
  const status = document.getElementById('ts-score-status');
  if (!title || !detail) return;
  const turns = state.diarization.turns || [];
  const speakers = state.diarization.speakers || [];
  const profileName = document.getElementById('ts-eval-profile-select')?.value || state.targetSpeaker.profileName;
  const assignedId = state.targetSpeaker.assignedSpeakerId;

  if (!turns.length) {
    title.textContent = 'Run diarization first';
    detail.textContent = 'A completed speaker timeline is required before target-speaker scoring.';
    if (status && !state.targetSpeaker.scored) status.textContent = 'Choose a target speaker after diarization.';
    return;
  }
  if (!profileName) {
    title.textContent = `${speakers.length} speakers · ${turns.length} turns ready`;
    detail.textContent = 'Choose a known target speaker to begin verification.';
    if (status && !state.targetSpeaker.scored) status.textContent = 'Choose a target speaker to score these turns.';
    return;
  }

  title.textContent = `${profileName} · ${turns.length} turns ready`;
  detail.textContent = assignedId
    ? `Optional diarization reference: ${getSpeakerName(assignedId)} (local ${assignedId}).`
    : 'No same-named diarized speaker is required; all turns will be verified against the reference clips.';
  if (status && !state.targetSpeaker.scored) status.textContent = `Ready to score ${turns.length} turns against “${profileName}”.`;
}

function resetTargetSpeakerEvaluation({ preserveSelection = true, preserveLabels = false } = {}) {
  state.targetSpeaker.scored = null;
  state.targetSpeaker.audioId = null;
  if (!preserveLabels) state.targetSpeaker.labels = {};
  state.targetSpeaker.threshold = 0.60;
  state.targetSpeaker.minDur = 1.5;
  state.targetSpeaker.excludeOverlap = true;
  if (!preserveSelection) {
    state.targetSpeaker.profileName = '';
    state.targetSpeaker.assignedSpeakerId = '';
  }
  const summary = document.getElementById('ts-evaluation-summary');
  const results = document.getElementById('ts-evaluation-results');
  const chip = document.getElementById('ts-kept-chip');
  if (summary) summary.hidden = true;
  if (results) results.innerHTML = '';
  if (chip) chip.textContent = 'Not scored';
  state.diarization.targetMatchFilter = 'all';
  if (el.diarFilterTargetSelect) {
    el.diarFilterTargetSelect.value = 'all';
    el.diarFilterTargetSelect.disabled = true;
  }
  renderTurnsTable();
}

async function runTargetSpeakerScore() {
  const profileSelect = document.getElementById('ts-eval-profile-select');
  const status = document.getElementById('ts-score-status');
  const button = document.getElementById('btn-ts-score');
  const profileName = profileSelect?.value || '';
  const audioId = state.diarization.audioId || el.diarInputSelect?.value || '';
  const turns = state.diarization.turns || [];

  if (!profileName) { showToast('Choose a target speaker', 'error'); return; }
  if (!audioId || !turns.length) { showToast('Run diarization first — scoring needs speaker turns', 'error'); return; }

  state.targetSpeaker.profileName = profileName;
  if (button) button.disabled = true;
  if (status) status.textContent = `Scoring ${turns.length} turns against “${profileName}”…`;

  try {
    const response = await fetch('/api/diarization/target-speaker-score', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        audio_id: audioId,
        profile: profileName,
        turns: turns.map(turn => ({
          speaker_id: turn.speaker_id,
          start_s: turn.start_s,
          end_s: turn.end_s,
        })),
        device: state.selectedGpu || el.diarDeviceSelect?.value || 'auto',
        token: localStorage.getItem('sonic_hf_token') || undefined,
      }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Target-speaker scoring could not start');
    const result = await new Promise((resolve, reject) => pollTask(data.task_id, resolve, reject));

    state.targetSpeaker.scored = result.scored;
    state.targetSpeaker.audioId = result.audio_id;
    const manualLabels = { ...state.targetSpeaker.labels };
    const saved = (state.evaluations || []).find(evaluation =>
      evaluation.evaluation_type === 'target_speaker'
      && evaluation.clip_id === result.audio_id
      && evaluation.profile_name === profileName
    );
    if (saved) {
      state.targetSpeaker.threshold = Number(saved.threshold ?? 0.60);
      state.targetSpeaker.minDur = Number(saved.min_duration_s ?? 1.5);
      state.targetSpeaker.excludeOverlap = saved.exclude_overlap !== false;
      state.targetSpeaker.labels = { ...manualLabels, ...(saved.segment_labels || {}) };
      const assignedTag = (saved.tags || []).find(tag => String(tag).startsWith('diarized_speaker:'));
      const savedSpeakerId = assignedTag ? assignedTag.slice('diarized_speaker:'.length) : '';
      if ((state.diarization.speakers || []).some(speaker => speaker.speaker_id === savedSpeakerId)) {
        state.targetSpeaker.assignedSpeakerId = savedSpeakerId;
        renderTargetSpeakerAssignmentOptions();
      }
    }
    renderTargetSpeakerResults();
    if (el.diarFilterTargetSelect) el.diarFilterTargetSelect.disabled = false;
    renderTurnsTable();
    renderTargetSpeakerContext();
    if (status) status.textContent = `Scored ${result.scored.segments.length} turns in ${result.elapsed_s}s. Review each result directly in the turn table below.`;
    showToast(`Target-speaker scoring completed for “${profileName}”`, 'success');
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    if (status) status.textContent = `Scoring failed: ${message}`;
    showToast(`Scoring failed: ${message}`, 'error');
  } finally {
    if (button) button.disabled = false;
  }
}

function targetSegmentKey(segment) {
  return `${Number(segment.start_s).toFixed(3)}-${Number(segment.end_s).toFixed(3)}-${segment.speaker_id}`;
}

function findTargetScoredSegment(turn) {
  const segments = state.targetSpeaker.scored?.segments || [];
  const exactKey = targetSegmentKey(turn);
  return segments.find(segment => targetSegmentKey(segment) === exactKey)
    || segments.find(segment =>
      segment.speaker_id === turn.speaker_id
      && Math.abs(Number(segment.start_s) - Number(turn.start_s)) < 0.01
      && Math.abs(Number(segment.end_s) - Number(turn.end_s)) < 0.01
    )
    || null;
}

function isTargetSegmentProposed(segment) {
  const target = state.targetSpeaker;
  return Number(segment.similarity) >= target.threshold
    && (Number(segment.end_s) - Number(segment.start_s)) >= target.minDur
    && !(target.excludeOverlap && segment.overlaps_other_speaker);
}

function toggleTargetSegmentLabel(key, nextLabel) {
  if (!key) return;
  const currentLabel = state.targetSpeaker.labels[key];
  state.targetSpeaker.labels[key] = currentLabel === nextLabel ? 'unreviewed' : nextLabel;
  updateActiveDiarizationHistory();
  renderTargetSpeakerResults();
  renderDiarizationFilteredViews();
}

function targetSpeakerKeptSegments() {
  return (state.targetSpeaker.scored?.segments || []).filter(isTargetSegmentProposed);
}

function renderTargetSpeakerResults() {
  const target = state.targetSpeaker;
  const scored = target.scored;
  const results = document.getElementById('ts-evaluation-results');
  const summary = document.getElementById('ts-evaluation-summary');
  if (!results || !summary) return;
  if (!scored) {
    results.innerHTML = '';
    summary.hidden = true;
    return;
  }

  const kept = targetSpeakerKeptSegments();
  const keptKeys = new Set(kept.map(targetSegmentKey));
  const labels = target.labels || {};
  const currentLabels = scored.segments.map(segment => labels[targetSegmentKey(segment)] || 'unreviewed');
  const reviewed = currentLabels.filter(label => label === 'qualified' || label === 'rejected').length;
  const qualified = currentLabels.filter(label => label === 'qualified').length;
  const totalDuration = scored.segments.reduce((sum, segment) => sum + segment.end_s - segment.start_s, 0);
  const keptDuration = kept.reduce((sum, segment) => sum + segment.end_s - segment.start_s, 0);
  const assignedId = target.assignedSpeakerId;
  const assignedSegments = assignedId ? scored.segments.filter(segment => segment.speaker_id === assignedId) : [];
  const agreementCount = assignedId ? scored.segments.filter(segment =>
    keptKeys.has(targetSegmentKey(segment)) === (segment.speaker_id === assignedId)
  ).length : 0;
  const agreementPercent = assignedId && scored.segments.length ? (agreementCount / scored.segments.length) * 100 : null;
  const chip = document.getElementById('ts-kept-chip');
  if (chip) chip.textContent = `${qualified} accepted · ${reviewed} reviewed`;

  summary.hidden = false;
  summary.innerHTML = `<strong>${escapeHtml(scored.profile_name)}</strong><span>${kept.length}/${scored.segments.length} proposed turns · ${keptDuration.toFixed(1)}s/${totalDuration.toFixed(1)}s diarized speech</span>`;

  results.innerHTML = `
    <article class="target-evaluation-card">
      <div class="target-settings-grid">
        <label>Similarity ≥ <strong id="ts-threshold-value">${target.threshold.toFixed(2)}</strong><input id="ts-threshold" type="range" min="-1" max="1" step="0.01" value="${target.threshold}"></label>
        <label>Minimum seconds<input id="ts-min-duration" class="text-input" type="number" min="0" step="0.25" value="${target.minDur}"></label>
        <label class="checkbox-pill"><input id="ts-exclude-overlap" type="checkbox" ${target.excludeOverlap ? 'checked' : ''}> Exclude overlaps</label>
      </div>
      <div class="target-metrics-grid">
        <div><strong>${kept.length}/${scored.segments.length}</strong><span>proposed turns</span></div>
        <div><strong>${keptDuration.toFixed(1)}s</strong><span>proposed duration</span></div>
        <div><strong>${qualified}/${reviewed || 0}</strong><span>accepted / reviewed</span></div>
        <div><strong>${agreementPercent === null ? '—' : `${agreementPercent.toFixed(1)}%`}</strong><span>${assignedId ? `agreement with ${getSpeakerName(assignedId)} (${assignedSegments.length} turns)` : 'optional named-speaker agreement'}</span></div>
      </div>
      <div class="target-segments-toolbar">
        <span>Review all ${scored.segments.length} scored turns with the Accept / Reject controls in the inspector table.</span>
      </div>
      <footer class="target-evaluation-actions">
        <button type="button" class="btn btn-secondary btn-sm" id="btn-ts-export-qualified">Export accepted audio</button>
        <button type="button" class="btn btn-primary btn-sm" id="btn-ts-save-evaluation">Save filter + evaluation</button>
      </footer>
    </article>`;

  document.getElementById('ts-threshold')?.addEventListener('input', event => {
    target.threshold = parseFloat(event.target.value);
    renderTargetSpeakerResults();
    renderDiarizationFilteredViews();
  });
  document.getElementById('ts-min-duration')?.addEventListener('change', event => {
    target.minDur = Math.max(0, parseFloat(event.target.value) || 0);
    renderTargetSpeakerResults();
    renderDiarizationFilteredViews();
  });
  document.getElementById('ts-exclude-overlap')?.addEventListener('change', event => {
    target.excludeOverlap = event.target.checked;
    renderTargetSpeakerResults();
    renderDiarizationFilteredViews();
  });
  document.getElementById('btn-ts-save-evaluation')?.addEventListener('click', saveTargetSpeakerEvaluation);
  document.getElementById('btn-ts-export-qualified')?.addEventListener('click', exportTargetSpeakerSegments);
}

async function saveTargetSpeakerEvaluation() {
  const target = state.targetSpeaker;
  const scored = target.scored;
  if (!scored) { showToast('Score a target speaker first', 'error'); return; }
  const audio = state.audioList.find(item => item.id === target.audioId) || {};
  const labels = target.labels || {};
  const reviewed = scored.segments.filter(segment => {
    const label = labels[targetSegmentKey(segment)];
    return label === 'qualified' || label === 'rejected';
  }).length;
  const qualifiedSegments = scored.segments.filter(segment => labels[targetSegmentKey(segment)] === 'qualified');
  const totalDuration = scored.segments.reduce((sum, segment) => sum + segment.end_s - segment.start_s, 0);
  const qualifiedDuration = qualifiedSegments.reduce((sum, segment) => sum + segment.end_s - segment.start_s, 0);
  const evalId = `target-${target.audioId}-${scored.profile_name}`.replace(/[^A-Za-z0-9_.-]/g, '_');
  const tags = ['target_speaker', `profile:${scored.profile_name}`];
  if (target.assignedSpeakerId) tags.push(`diarized_speaker:${target.assignedSpeakerId}`);

  try {
    const response = await fetch('/api/evaluations', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        id: evalId,
        evaluation_type: 'target_speaker',
        clip_id: target.audioId,
        clip_title: audio.title || target.audioId,
        clip_path: audio.path || '',
        model_id: scored.model?.model_id || 'speaker_verifier',
        model_name: scored.model?.backend || 'Target speaker verifier',
        profile_name: scored.profile_name,
        channel_id: scored.channel_id || audio.channel_id || null,
        channel_name: scored.channel_name || audio.channel_name || null,
        threshold: target.threshold,
        min_duration_s: target.minDur,
        exclude_overlap: target.excludeOverlap,
        qualified_segments: qualifiedSegments.length,
        reviewed_segments: reviewed,
        total_segments: scored.segments.length,
        qualified_duration_s: qualifiedDuration,
        total_duration_s: totalDuration,
        qualified_percent: scored.segments.length ? (qualifiedSegments.length / scored.segments.length) * 100 : 0,
        segment_labels: labels,
        score_overall: reviewed ? (qualifiedSegments.length / reviewed) * 5 : 0,
        tags,
      }),
    });
    if (!response.ok) throw new Error('Could not save target-speaker evaluation');
    await fetchEvaluations();
    showToast(`Saved evaluation for “${scored.profile_name}”`, 'success');
  } catch (err) {
    showToast(err.message, 'error');
  }
}

async function exportTargetSpeakerSegments() {
  const target = state.targetSpeaker;
  const scored = target.scored;
  if (!scored) { showToast('Score a target speaker first', 'error'); return; }
  const qualified = scored.segments.filter(segment => target.labels[targetSegmentKey(segment)] === 'qualified');
  if (!qualified.length) { showToast('Accept at least one segment before export', 'error'); return; }
  const mode = el.diarExtractModeSelect?.value || 'concatenated';

  try {
    const response = await fetch('/api/diarization/extract-speaker', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        audio_id: target.audioId,
        speaker_id: 'target',
        speaker_name: scored.profile_name,
        mode,
        turns: qualified.map(segment => ({
          speaker_id: 'target',
          start_s: segment.start_s,
          end_s: segment.end_s,
        })),
        extraction_settings: readDiarizationExtractionSettings(),
        blocker_turns: extractionBlockerTurns(),
      }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Export failed');
    await fetchAudioList();
    showToast(`Exported ${qualified.length} accepted target-speaker segments`, 'success');
  } catch (err) {
    showToast(`Export failed: ${err.message}`, 'error');
  }
}

// ==================== SPEAKER PURITY WORKBENCH ====================

function purityBackendDefaults(backend = state.purity.overlap.backend) {
  return state.purity.serverConfig?.[backend] || {};
}

function vibevoiceModelChoices() {
  const models = state.purity.serverConfig?.vibevoice?.models;
  if (Array.isArray(models) && models.length) return models;
  return [
    { id: 'microsoft/VibeVoice-ASR-HF', label: 'Full BF16 (~17 GB VRAM)' },
    { id: 'Dubedo/VibeVoice-ASR-HF-INT8', label: 'INT8 (~10–11 GB VRAM)' },
    { id: 'Dubedo/VibeVoice-ASR-HF-NF4', label: 'NF4 4-bit (~7–8 GB VRAM)' },
  ];
}

function resolveVibevoiceModelId(modelId) {
  const choices = vibevoiceModelChoices();
  const requested = String(modelId || '').trim();
  if (choices.some(choice => choice.id === requested)) return requested;
  const fallback = state.purity.serverConfig?.vibevoice?.model || choices[0]?.id || '';
  if (choices.some(choice => choice.id === fallback)) return fallback;
  return choices[0]?.id || '';
}

function populateVibevoiceModelSelect(selectedId) {
  const select = el.purityOverlapVibevoiceModel;
  if (!select) return '';
  const choices = vibevoiceModelChoices();
  const resolved = resolveVibevoiceModelId(selectedId);
  select.innerHTML = '';
  for (const choice of choices) {
    const option = document.createElement('option');
    option.value = choice.id;
    option.textContent = choice.label;
    select.appendChild(option);
  }
  select.value = resolved;
  return select.value;
}

function purityOverlapBackendLabel(backend = state.purity.overlap.backend) {
  if (backend === 'vibevoice') return 'VibeVoice-ASR';
  if (backend === 'gemini') return 'Gemini 3.1 Pro';
  if (backend === 'gemini-flash-lite') return 'Gemini 3.1 Flash-Lite';
  return 'Gemma 4';
}

async function loadDiarizationResultsForVerification() {
  if (!el.purityResultList) return;
  try {
    const payload = await parseJsonResponse(await fetch('/api/diarization/results'));
    state.purity.diarizationResults = payload.results || [];
    const available = new Set(state.purity.diarizationResults.map(item => item.result_id));
    state.purity.selectedResultIds = new Set(
      [...state.purity.selectedResultIds].filter(resultId => available.has(resultId))
    );
    renderDiarizationResultCandidates();
  } catch (err) {
    el.purityResultList.innerHTML = `<div class="empty-placeholder">${escapeHtml(err.message || String(err))}</div>`;
  }
}

function renderDiarizationResultCandidates() {
  if (!el.purityResultList) return;
  const results = state.purity.diarizationResults || [];
  if (!results.length) {
    el.purityResultList.innerHTML = '<div class="empty-placeholder">No persisted diarization results yet. Completing diarization will add one automatically.</div>';
    return;
  }
  el.purityResultList.innerHTML = results.map(result => {
    const source = result.source_audio || {};
    const summary = result.summary || {};
    const model = result.model?.model_id || result.model?.backend || 'Unknown model';
    const stateLabel = result.verification?.state || 'unverified';
    const selected = state.purity.selectedResultIds.has(result.result_id);
    return `<label class="purity-result-card ${selected ? 'selected' : ''}">
      <input type="checkbox" data-result-id="${escapeHtml(result.result_id)}" ${selected ? 'checked' : ''} ${result.source_available ? '' : 'disabled'}>
      <span class="purity-result-card-body">
        <strong>${escapeHtml(source.title || result.audio_id)}</strong>
        <span class="text-xs text-muted">${escapeHtml(model)}</span>
        <span class="purity-result-meta">
          <span>${summary.speaker_count ?? result.speakers?.length ?? 0} speakers</span>
          <span>${summary.turn_count ?? result.turns?.length ?? 0} turns</span>
          <span>${Number(summary.total_speech_duration_s || 0).toFixed(1)}s speech</span>
          <span class="badge badge-sm ${stateLabel === 'passed' ? 'badge-success' : stateLabel === 'rejected' || stateLabel === 'error' ? 'badge-danger' : 'badge-ghost'}">${escapeHtml(stateLabel)}</span>
          ${result.source_available ? '' : '<span class="badge badge-sm badge-warning">source unavailable</span>'}
        </span>
      </span>
    </label>`;
  }).join('');
  el.purityResultList.querySelectorAll('input[data-result-id]').forEach(input => {
    input.addEventListener('change', () => {
      if (input.checked) state.purity.selectedResultIds.add(input.dataset.resultId);
      else state.purity.selectedResultIds.delete(input.dataset.resultId);
      renderDiarizationResultCandidates();
      syncDiarizationCandidateFilters();
    });
  });
  syncDiarizationCandidateFilters();
}

function syncDiarizationCandidateFilters() {
  const selected = (state.purity.diarizationResults || []).filter(result => state.purity.selectedResultIds.has(result.result_id));
  const speakers = [...new Set(selected.flatMap(result => (result.speakers || []).map(speaker => speaker.speaker_id)))].sort();
  if (el.purityCandidateSpeaker) {
    const current = el.purityCandidateSpeaker.value;
    el.purityCandidateSpeaker.innerHTML = '<option value="">All speakers</option>' + speakers.map(speaker => `<option value="${escapeHtml(speaker)}">${escapeHtml(speaker)}</option>`).join('');
    if (speakers.includes(current)) el.purityCandidateSpeaker.value = current;
  }
  const turns = selected.reduce((sum, result) => sum + diarizationResultTurnCount(result), 0);
  if (el.purityResultSelectionSummary) {
    el.purityResultSelectionSummary.textContent = `${selected.length} result(s) selected • ${turns} turns before filters • all eligible turns will run as one batch`;
  }
}

function diarizationResultTurnCount(result) {
  const fromSummary = Number(result?.summary?.turn_count);
  if (Number.isFinite(fromSummary)) return fromSummary;
  return Array.isArray(result?.turns) ? result.turns.length : 0;
}

function purityTurnsForSelectedAudio(audioId) {
  if (audioId && state.diarization.audioId === audioId && state.diarization.turns?.length > 0) {
    return state.diarization.turns.map(turn => ({
      speaker_id: turn.speaker_id,
      start_s: turn.start_s,
      end_s: turn.end_s,
    }));
  }
  return [];
}

// ==================== MANUAL DIARIZATION ANNOTATION ====================

const ANNOTATION_SPEAKER_COLORS = [
  '#168aad', '#2f9e6f', '#c98200', '#dc3656', '#805ad5',
  '#2574c8', '#65a30d', '#d95f20', '#0891b2', '#be185d',
];

function formatAnnotationTime(seconds) {
  const safe = Math.max(0, Number(seconds) || 0);
  const totalMs = Math.round(safe * 1000);
  const hours = Math.floor(totalMs / 3600000);
  const minutes = Math.floor((totalMs % 3600000) / 60000);
  const wholeSeconds = Math.floor((totalMs % 60000) / 1000);
  const milliseconds = totalMs % 1000;
  return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(wholeSeconds).padStart(2, '0')}.${String(milliseconds).padStart(3, '0')}`;
}

function parseAnnotationTime(value) {
  const text = String(value ?? '').trim();
  if (!text) return NaN;
  if (!text.includes(':')) {
    const seconds = Number(text);
    return Number.isFinite(seconds) && seconds >= 0 ? seconds : NaN;
  }
  const parts = text.split(':').map(Number);
  if (parts.length < 2 || parts.length > 3 || parts.some(part => !Number.isFinite(part) || part < 0)) return NaN;
  if (parts.slice(1).some(part => part >= 60)) return NaN;
  return parts.length === 2
    ? parts[0] * 60 + parts[1]
    : parts[0] * 3600 + parts[1] * 60 + parts[2];
}

function annotationDuration() {
  return Number(state.annotation.current?.source_audio?.duration_s)
    || Number(state.audioList.find(item => item.id === state.annotation.audioId)?.duration_s)
    || Number(state.activeAudio?.duration_s)
    || 0;
}

function snapAnnotationTime(seconds) {
  const duration = annotationDuration();
  const snap = Number(state.annotation.snapS) || 0;
  const clamped = Math.max(0, Math.min(Number(seconds) || 0, duration));
  return Number((snap ? Math.round(clamped / snap) * snap : clamped).toFixed(3));
}

function annotationSpeaker(speakerId) {
  return state.annotation.speakers.find(speaker => speaker.speaker_id === speakerId) || null;
}

function annotationSelectedTurn() {
  return state.annotation.turns.find(turn => turn.turn_id === state.annotation.selectedTurnId) || null;
}

function setAnnotationSaveState(label, status = 'idle') {
  if (!el.annSaveState) return;
  el.annSaveState.textContent = label;
  el.annSaveState.dataset.state = status;
}

function annotationSnapshot() {
  return {
    name: state.annotation.current?.name || el.annNameInput?.value || 'Ground truth',
    speakers: structuredClone(state.annotation.speakers),
    turns: structuredClone(state.annotation.turns),
    activeSpeakerId: state.annotation.activeSpeakerId,
    selectedTurnId: state.annotation.selectedTurnId,
    markIn: state.annotation.markIn,
    markOut: state.annotation.markOut,
  };
}

function restoreAnnotationSnapshot(snapshot) {
  if (!snapshot) return;
  state.annotation.speakers = structuredClone(snapshot.speakers || []);
  state.annotation.turns = structuredClone(snapshot.turns || []);
  state.annotation.activeSpeakerId = snapshot.activeSpeakerId || state.annotation.speakers[0]?.speaker_id || null;
  state.annotation.selectedTurnId = snapshot.selectedTurnId || null;
  state.annotation.markIn = snapshot.markIn;
  state.annotation.markOut = snapshot.markOut;
  if (el.annNameInput) el.annNameInput.value = snapshot.name || 'Ground truth';
}

function annotationChanged(mutator) {
  if (!state.annotation.current) return false;
  const before = annotationSnapshot();
  const changed = mutator();
  if (changed === false) return false;
  state.annotation.undo.push(before);
  state.annotation.undo = state.annotation.undo.slice(-100);
  state.annotation.redo = [];
  state.annotation.editVersion += 1;
  state.annotation.dirty = true;
  renderAnnotationEditor();
  scheduleAnnotationSave();
  return true;
}

function undoAnnotation() {
  const snapshot = state.annotation.undo.pop();
  if (!snapshot) return;
  state.annotation.redo.push(annotationSnapshot());
  restoreAnnotationSnapshot(snapshot);
  state.annotation.editVersion += 1;
  state.annotation.dirty = true;
  renderAnnotationEditor();
  scheduleAnnotationSave();
}

function redoAnnotation() {
  const snapshot = state.annotation.redo.pop();
  if (!snapshot) return;
  state.annotation.undo.push(annotationSnapshot());
  restoreAnnotationSnapshot(snapshot);
  state.annotation.editVersion += 1;
  state.annotation.dirty = true;
  renderAnnotationEditor();
  scheduleAnnotationSave();
}

function scheduleAnnotationSave(delay = 550) {
  clearTimeout(state.annotation.saveTimer);
  state.annotation.dirty = true;
  setAnnotationSaveState('Unsaved changes', 'saving');
  state.annotation.saveTimer = setTimeout(() => saveAnnotationNow(), delay);
}

async function saveAnnotationNow() {
  clearTimeout(state.annotation.saveTimer);
  state.annotation.saveTimer = null;
  if (!state.annotation.current || !state.annotation.audioId || !state.annotation.dirty) return state.annotation.current;
  if (state.annotation.savePromise) {
    await state.annotation.savePromise.catch(() => {});
    if (!state.annotation.dirty) return state.annotation.current;
  }
  const current = state.annotation.current;
  const capturedVersion = state.annotation.editVersion;
  const payload = {
    annotation_id: current.annotation_id || undefined,
    revision: current.revision || 0,
    session_audio_id: state.annotation.audioId,
    name: el.annNameInput?.value.trim() || current.name || 'Ground truth',
    speakers: structuredClone(state.annotation.speakers),
    turns: structuredClone(state.annotation.turns),
  };
  setAnnotationSaveState('Saving…', 'saving');
  const request = (async () => {
    const response = await fetch('/api/diarization/annotations', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const saved = await parseJsonResponse(response);
    if (state.annotation.current !== current) return saved;
    current.annotation_id = saved.annotation_id;
    current.revision = saved.revision;
    current.created_at = saved.created_at;
    current.updated_at = saved.updated_at;
    current.source_audio = saved.source_audio;
    current.audio_id = saved.audio_id;
    current.name = saved.name;
    current.seed = saved.seed || null;
    state.annotation.turns = structuredClone(saved.turns || []);
    state.annotation.speakers = structuredClone(saved.speakers || []);
    if (state.annotation.editVersion === capturedVersion) {
      state.annotation.dirty = false;
      setAnnotationSaveState(`Saved · r${saved.revision}`, 'saved');
    } else {
      scheduleAnnotationSave(100);
    }
    if (el.annRevisionLabel) {
      el.annRevisionLabel.textContent = `Server revision ${saved.revision} · autosave enabled`;
    }
    if (el.btnAnnExport) el.btnAnnExport.disabled = false;
    await loadAnnotationCatalog(saved.annotation_id);
    return saved;
  })();
  state.annotation.savePromise = request;
  try {
    return await request;
  } catch (error) {
    setAnnotationSaveState(error.message || 'Save failed', 'error');
    showToast(`Annotation save failed: ${error.message}`, 'error');
    throw error;
  } finally {
    if (state.annotation.savePromise === request) state.annotation.savePromise = null;
  }
}

async function loadAnnotationCatalog(selectedId = null) {
  if (!el.annSavedSelect) return;
  try {
    const payload = await parseJsonResponse(await fetch('/api/diarization/annotations'));
    state.annotation.catalog = payload.annotations || [];
    const currentValue = selectedId || state.annotation.current?.annotation_id || el.annSavedSelect.value;
    el.annSavedSelect.innerHTML = '<option value="">New annotation</option>' + state.annotation.catalog.map(item => {
      const title = item.name || item.source_audio?.title || item.annotation_id;
      const details = `${item.speaker_count} spk · ${item.turn_count} turns · r${item.revision}`;
      return `<option value="${escapeHtml(item.annotation_id)}">${escapeHtml(title)} — ${details}</option>`;
    }).join('');
    if (currentValue && state.annotation.catalog.some(item => item.annotation_id === currentValue)) {
      el.annSavedSelect.value = currentValue;
    }
  } catch (error) {
    console.warn('Could not load annotation catalog:', error);
  }
}

function clearAnnotationEditor() {
  clearTimeout(state.annotation.saveTimer);
  state.annotation.audioId = null;
  state.annotation.current = null;
  state.annotation.speakers = [];
  state.annotation.turns = [];
  state.annotation.activeSpeakerId = null;
  state.annotation.selectedTurnId = null;
  state.annotation.markIn = null;
  state.annotation.markOut = null;
  state.annotation.undo = [];
  state.annotation.redo = [];
  state.annotation.drag = null;
  state.annotation.rangeDrag = null;
  state.annotation.suppressTimelineClick = false;
  state.annotation.loopTurnId = null;
  state.annotation.dirty = false;
  state.annotation.evaluation = null;
  state.annotation.seedResultId = null;
  el.annTimelineStage?.classList.remove('selecting');
  if (el.annNameInput) {
    el.annNameInput.value = '';
    el.annNameInput.disabled = true;
  }
  el.annWorkspace?.classList.add('hidden');
  el.annSeedNotice?.classList.add('hidden');
  el.annEmptyState?.classList.remove('hidden');
  if (el.btnAnnExport) el.btnAnnExport.disabled = true;
  setAnnotationSaveState('No annotation loaded', 'idle');
}

async function selectAnnotationAudio(audioId) {
  if (!audioId) return;
  if (audioId.startsWith('lib:')) {
    await loadLibraryFileTo(audioId.slice(4), 'annotation');
    return;
  }
  const item = state.audioList.find(audio => audio.id === audioId);
  if (!item) {
    showToast('The selected audio is no longer available', 'error');
    return;
  }
  if (state.annotation.current && state.annotation.audioId !== audioId) {
    await saveAnnotationNow().catch(() => {});
    clearAnnotationEditor();
  }
  state.annotation.audioId = audioId;
  if (el.annAudioSelect) el.annAudioSelect.value = audioId;
  if (el.annAudioMeta) {
    el.annAudioMeta.textContent = `${item.title || item.source_id} · ${formatAnnotationTime(item.duration_s)} · ${(item.sample_rate || 0).toLocaleString()} Hz · ${item.fingerprint || 'fingerprint pending'}`;
  }
  await setActiveAudio(audioId, { play: false });
  if (state.annotation.waveform.audioId !== audioId) {
    state.annotation.waveform.data = null;
    state.annotation.waveform.error = '';
  }
  state.annotation.waveform.audioId = audioId;
  renderAnnotationTimeline();
  scheduleAnnotationWaveform();
  await loadCompatibleDiarizationResults();
}

async function createAnnotationForSelectedAudio() {
  let audioId = el.annAudioSelect?.value || state.annotation.audioId || state.activeAudio?.id;
  if (audioId?.startsWith('lib:')) {
    await loadLibraryFileTo(audioId.slice(4), 'annotation');
    return;
  }
  if (!audioId || !state.audioList.some(item => item.id === audioId)) {
    showToast('Select source audio first', 'warning');
    return;
  }
  await saveAnnotationNow().catch(() => {});
  clearAnnotationEditor();
  await selectAnnotationAudio(audioId);
  const item = state.audioList.find(audio => audio.id === audioId);
  const speakerId = 'spk_1';
  state.annotation.current = {
    annotation_id: null,
    revision: 0,
    name: `${item?.title || 'Audio'} ground truth`,
    source_audio: { duration_s: item?.duration_s, title: item?.title, fingerprint: item?.fingerprint },
  };
  state.annotation.speakers = [{ speaker_id: speakerId, name: 'Speaker 1', color: ANNOTATION_SPEAKER_COLORS[0], global_speaker_id: null }];
  state.annotation.turns = [];
  state.annotation.activeSpeakerId = speakerId;
  state.annotation.markIn = 0;
  state.annotation.editVersion += 1;
  state.annotation.dirty = true;
  el.annEmptyState?.classList.add('hidden');
  el.annWorkspace?.classList.remove('hidden');
  if (el.annNameInput) {
    el.annNameInput.disabled = false;
    el.annNameInput.value = state.annotation.current.name;
  }
  renderAnnotationEditor();
  await saveAnnotationNow();
  await loadCompatibleDiarizationResults();
}

async function loadAnnotation(annotationId) {
  if (!annotationId) return;
  await saveAnnotationNow().catch(() => {});
  const payload = await parseJsonResponse(
    await fetch(`/api/diarization/annotations/${encodeURIComponent(annotationId)}`)
  );
  clearAnnotationEditor();
  state.annotation.current = payload;
  state.annotation.seedResultId = payload.seed?.result_id || null;
  state.annotation.audioId = payload.session_audio_id || null;
  state.annotation.speakers = structuredClone(payload.speakers || []);
  state.annotation.turns = structuredClone(payload.turns || []);
  state.annotation.activeSpeakerId = state.annotation.speakers[0]?.speaker_id || null;
  state.annotation.markIn = 0;
  state.annotation.editVersion += 1;
  el.annEmptyState?.classList.add('hidden');
  el.annWorkspace?.classList.remove('hidden');
  if (el.annNameInput) {
    el.annNameInput.disabled = false;
    el.annNameInput.value = payload.name || 'Ground truth';
  }
  if (el.annSavedSelect) el.annSavedSelect.value = annotationId;
  if (payload.session_audio_id) {
    await fetchAudioList();
    await selectAnnotationAudio(payload.session_audio_id);
  } else {
    state.annotation.waveform.data = null;
    state.annotation.waveform.audioId = null;
    el.audio?.pause();
    el.audio?.removeAttribute('src');
    el.audio?.load();
    if (el.annAudioMeta) el.annAudioMeta.textContent = 'Source file is unavailable on this machine; playback is disabled but timestamp editing remains available.';
    showToast('Annotation loaded, but its source audio file is unavailable', 'warning');
  }
  if (el.annRevisionLabel) el.annRevisionLabel.textContent = `Server revision ${payload.revision} · autosave enabled`;
  setAnnotationSaveState(`Saved · r${payload.revision}`, 'saved');
  if (el.btnAnnExport) el.btnAnnExport.disabled = false;
  renderAnnotationEditor();
  renderAnnotationSeedNotice();
  await loadCompatibleDiarizationResults();
}

function annotationTurnOverlap(candidate, excludeTurnId = null) {
  return state.annotation.turns.find(turn => (
    turn.turn_id !== excludeTurnId
    && turn.speaker_id === candidate.speaker_id
    && candidate.start_s < turn.end_s - 0.000001
    && turn.start_s < candidate.end_s - 0.000001
  )) || null;
}

function createAnnotationTurn(startValue = state.annotation.markIn, endValue = state.annotation.markOut) {
  if (!state.annotation.current || !state.annotation.activeSpeakerId) return false;
  if (!Number.isFinite(Number(startValue)) || !Number.isFinite(Number(endValue))) {
    showToast('Enter valid mark-in and mark-out timestamps', 'warning');
    return false;
  }
  const start = snapAnnotationTime(startValue);
  const end = snapAnnotationTime(endValue);
  if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) {
    showToast('Mark out must be after mark in', 'warning');
    return false;
  }
  const turn = {
    turn_id: `turn_${crypto.randomUUID().replaceAll('-', '')}`,
    speaker_id: state.annotation.activeSpeakerId,
    start_s: start,
    end_s: end,
  };
  const overlap = annotationTurnOverlap(turn);
  if (overlap) {
    showToast(`This overlaps another ${annotationSpeaker(turn.speaker_id)?.name || 'speaker'} turn. Use a separate speaker lane for simultaneous speech.`, 'warning');
    return false;
  }
  return annotationChanged(() => {
    state.annotation.turns.push(turn);
    state.annotation.turns.sort((a, b) => a.start_s - b.start_s || a.end_s - b.end_s);
    state.annotation.selectedTurnId = turn.turn_id;
    state.annotation.markIn = end;
    state.annotation.markOut = null;
    seekTo(start);
  });
}

function updateAnnotationMarks() {
  const markIn = state.annotation.markIn;
  const markOut = state.annotation.markOut;
  const hasRange = Number.isFinite(markIn) && Number.isFinite(markOut) && markOut > markIn;
  const activeSpeaker = annotationSpeaker(state.annotation.activeSpeakerId);
  if (el.annMarkIn && document.activeElement !== el.annMarkIn) {
    el.annMarkIn.value = Number.isFinite(markIn) ? formatAnnotationTime(markIn) : '';
  }
  if (el.annMarkOut && document.activeElement !== el.annMarkOut) {
    el.annMarkOut.value = Number.isFinite(markOut) ? formatAnnotationTime(markOut) : '';
  }
  if (el.annMarkDuration) {
    el.annMarkDuration.textContent = hasRange
      ? `${formatAnnotationTime(markOut - markIn)} selected for ${activeSpeaker?.name || 'active speaker'}`
      : 'No complete range marked';
  }
  if (el.btnAnnCreateTurn) {
    el.btnAnnCreateTurn.disabled = !hasRange || !activeSpeaker;
    el.btnAnnCreateTurn.textContent = activeSpeaker ? `Create ${activeSpeaker.name} turn` : 'Create speaker turn';
  }
  if (el.annMarkRegion) {
    const duration = annotationDuration();
    const valid = duration > 0 && hasRange;
    el.annMarkRegion.classList.toggle('hidden', !valid);
    if (valid) {
      el.annMarkRegion.style.left = `${markIn / duration * 100}%`;
      el.annMarkRegion.style.width = `${(markOut - markIn) / duration * 100}%`;
    }
  }
}

function renderAnnotationSpeakers() {
  if (!el.annSpeakerChips || !el.annLaneLabels) return;
  el.annSpeakerChips.innerHTML = state.annotation.speakers.map((speaker, index) => `
    <button class="ann-speaker-chip ${speaker.speaker_id === state.annotation.activeSpeakerId ? 'active' : ''}"
      data-speaker-id="${escapeHtml(speaker.speaker_id)}" style="--speaker-color:${escapeHtml(speaker.color)}"
      title="${speaker.global_speaker_id ? `Linked to ${escapeHtml(speaker.global_speaker_id)}` : 'Local anonymous speaker'}">
      <span class="ann-speaker-chip-key">${index < 9 ? index + 1 : '•'}</span>
      <span>${escapeHtml(speaker.name)}</span>
      ${speaker.global_speaker_id ? '<span aria-label="Known profile">🔗</span>' : ''}
    </button>
  `).join('');
  el.annSpeakerChips.querySelectorAll('[data-speaker-id]').forEach(button => {
    button.addEventListener('click', () => {
      state.annotation.activeSpeakerId = button.dataset.speakerId;
      renderAnnotationSpeakers();
      updateAnnotationMarks();
    });
  });
  el.annLaneLabels.innerHTML = state.annotation.speakers.map(speaker => {
    const count = state.annotation.turns.filter(turn => turn.speaker_id === speaker.speaker_id).length;
    const speechS = state.annotation.turns.filter(turn => turn.speaker_id === speaker.speaker_id).reduce((sum, turn) => sum + turn.end_s - turn.start_s, 0);
    return `<div class="ann-lane-label" style="--speaker-color:${escapeHtml(speaker.color)}">
      <span class="ann-lane-label-dot"></span><span title="${escapeHtml(speaker.speaker_id)}">${escapeHtml(speaker.name)}</span><small>${count} · ${speechS.toFixed(1)}s</small>
    </div>`;
  }).join('');
}

function renderAnnotationWaveform() {
  const canvas = el.annWaveformCanvas;
  const stage = el.annTimelineStage;
  if (!canvas || !stage) return;
  const left = Math.max(0, el.annTimelineScroll?.scrollLeft || 0);
  const width = Math.max(1, el.annTimelineScroll?.clientWidth || stage.clientWidth);
  canvas.style.left = `${left}px`;
  canvas.style.width = `${width}px`;
  drawWaveformEnvelope(canvas, state.annotation.waveform, 'Waveform unavailable');
}

function annotationWaveformWindow() {
  const duration = annotationDuration();
  const stageWidth = Math.max(1, el.annTimelineStage?.clientWidth || 1);
  const visibleWidth = Math.max(1, el.annTimelineScroll?.clientWidth || stageWidth);
  const left = Math.max(0, el.annTimelineScroll?.scrollLeft || 0);
  const start = Math.min(duration, left / stageWidth * duration);
  return { start, end: Math.min(duration, start + visibleWidth / stageWidth * duration) };
}

function scheduleAnnotationWaveform() {
  clearTimeout(state.annotation.waveform.timer);
  state.annotation.waveform.timer = setTimeout(() => {
    const range = annotationWaveformWindow();
    renderAnnotationWaveform();
    if (!state.annotation.waveform.audioId || !(range.end > range.start)) return;
    requestWaveformWindow({
      audioId: state.annotation.waveform.audioId,
      canvas: el.annWaveformCanvas,
      start: range.start,
      end: range.end,
      view: state.annotation.waveform,
      draw: renderAnnotationWaveform,
    });
  }, 80);
}

function niceAnnotationTickStep(duration, width) {
  const target = duration / Math.max(2, width / 110);
  const steps = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600];
  return steps.find(step => step >= target) || Math.ceil(target / 600) * 600;
}

function renderAnnotationRuler() {
  if (!el.annRuler) return;
  const duration = annotationDuration() || 1;
  const range = annotationWaveformWindow();
  const visibleWidth = Math.max(1, el.annTimelineScroll?.clientWidth || 600);
  const visibleSpan = Math.max(0, range.end - range.start);
  const useWindow = visibleSpan > 0 && visibleSpan < duration;
  const viewStart = useWindow ? range.start : 0;
  const viewEnd = useWindow ? range.end : duration;
  const span = Math.max(1e-6, viewEnd - viewStart);
  const pad = useWindow ? span * 0.25 : 0;
  const start = Math.max(0, viewStart - pad);
  const end = Math.min(duration, viewEnd + pad);
  const step = niceAnnotationTickStep(span, visibleWidth);
  const first = Math.ceil(start / step - 1e-9) * step;
  const ticks = [];
  for (let i = 0; i < 80; i += 1) {
    const seconds = first + i * step;
    if (seconds > end + step * 0.25) break;
    const pct = duration ? Math.min(100, seconds / duration * 100) : 0;
    ticks.push(`<div class="ann-ruler-tick" style="left:${pct}%"><span>${formatAnnotationTime(seconds).replace(/^00:/, '')}</span></div>`);
  }
  el.annRuler.innerHTML = ticks.join('');
}

function syncAnnotationZoomControls() {
  const zoom = state.annotation.zoom;
  if (el.annZoomRange && document.activeElement !== el.annZoomRange) {
    el.annZoomRange.value = timelineZoomToSlider(zoom, 1);
  }
  if (el.annZoomInput && document.activeElement !== el.annZoomInput) {
    el.annZoomInput.value = formatZoomMultiplier(zoom);
  }
  if (el.annZoomLabel) el.annZoomLabel.textContent = `${Math.round(zoom * 100)}%`;
}

function renderAnnotationTimeline() {
  if (!el.annTimelineStage || !state.annotation.current) return;
  const duration = annotationDuration();
  const viewportWidth = Math.max(600, el.annTimelineScroll?.clientWidth || 600);
  const width = Math.max(viewportWidth, Math.round(viewportWidth * state.annotation.zoom));
  const height = 112 + Math.max(1, state.annotation.speakers.length) * 52;
  el.annTimelineStage.style.width = `${width}px`;
  el.annTimelineStage.style.height = `${height}px`;
  if (el.annLaneLabels) el.annLaneLabels.style.minHeight = `${height}px`;
  syncAnnotationZoomControls();

  renderAnnotationWaveform();
  scheduleAnnotationWaveform();
  renderAnnotationRuler();
  if (el.annLanes) {
    el.annLanes.innerHTML = state.annotation.speakers.map(speaker => {
      const turns = state.annotation.turns.filter(turn => turn.speaker_id === speaker.speaker_id);
      return `<div class="ann-lane" data-lane-speaker="${escapeHtml(speaker.speaker_id)}">${turns.map(turn => {
        const left = duration ? turn.start_s / duration * 100 : 0;
        const widthPct = duration ? (turn.end_s - turn.start_s) / duration * 100 : 0;
        return `<div class="ann-segment ${turn.turn_id === state.annotation.selectedTurnId ? 'selected' : ''}"
          data-turn-id="${escapeHtml(turn.turn_id)}" style="left:${left}%;width:${widthPct}%;--speaker-color:${escapeHtml(speaker.color)}"
          title="${escapeHtml(speaker.name)} · ${formatAnnotationTime(turn.start_s)} – ${formatAnnotationTime(turn.end_s)}">
          <span class="ann-segment-handle start" data-edge="start"></span>
          <span class="ann-segment-label">${escapeHtml(speaker.name)} · ${(turn.end_s - turn.start_s).toFixed(3)}s</span>
          <span class="ann-segment-handle end" data-edge="end"></span>
        </div>`;
      }).join('')}</div>`;
    }).join('');
  }
  updateAnnotationMarks();
  updateAnnotationPlayhead(el.audio?.currentTime || 0);
}

function countAnnotationOverlaps() {
  let count = 0;
  const turns = state.annotation.turns;
  for (let left = 0; left < turns.length; left += 1) {
    for (let right = left + 1; right < turns.length; right += 1) {
      if (turns[left].speaker_id !== turns[right].speaker_id && turns[left].start_s < turns[right].end_s && turns[right].start_s < turns[left].end_s) count += 1;
    }
  }
  return count;
}

function renderAnnotationTurnsTable() {
  if (!el.annTurnsBody) return;
  const query = (el.annTurnSearch?.value || '').toLowerCase().trim();
  const turns = state.annotation.turns.filter(turn => {
    const speaker = annotationSpeaker(turn.speaker_id);
    return !query || `${speaker?.name || ''} ${turn.speaker_id} ${formatAnnotationTime(turn.start_s)} ${formatAnnotationTime(turn.end_s)}`.toLowerCase().includes(query);
  });
  if (!turns.length) {
    el.annTurnsBody.innerHTML = '<tr><td colspan="6"><div class="empty-placeholder">No turns yet. Drag a range on the timeline and create the active speaker turn, or use I/O while playing.</div></td></tr>';
  } else {
    el.annTurnsBody.innerHTML = turns.map((turn, index) => `
      <tr data-turn-id="${escapeHtml(turn.turn_id)}" class="${turn.turn_id === state.annotation.selectedTurnId ? 'selected' : ''}">
        <td>${index + 1}</td>
        <td><select class="select-input select-sm ann-row-speaker" aria-label="Turn speaker">${state.annotation.speakers.map(speaker => `<option value="${escapeHtml(speaker.speaker_id)}" ${speaker.speaker_id === turn.speaker_id ? 'selected' : ''}>${escapeHtml(speaker.name)}</option>`).join('')}</select></td>
        <td><input class="text-input ann-time-input ann-row-start" value="${formatAnnotationTime(turn.start_s)}" aria-label="Turn start"></td>
        <td><input class="text-input ann-time-input ann-row-end" value="${formatAnnotationTime(turn.end_s)}" aria-label="Turn end"></td>
        <td class="font-mono">${(turn.end_s - turn.start_s).toFixed(3)}s</td>
        <td><button class="btn btn-xs btn-ghost ann-row-play">▶ Play</button> <button class="btn btn-xs btn-ghost text-destructive ann-row-delete">Delete</button></td>
      </tr>
    `).join('');
  }
  const selected = annotationSelectedTurn();
  [el.btnAnnLoopSelected, el.btnAnnSplit, el.btnAnnReassign, el.btnAnnDeleteTurn].forEach(button => { if (button) button.disabled = !selected; });
}

function renderAnnotationEditor() {
  if (!state.annotation.current) return;
  renderAnnotationSeedNotice();
  renderAnnotationSpeakers();
  renderAnnotationTimeline();
  renderAnnotationTurnsTable();
  updateAnnotationMarks();
  if (el.annTurnCount) el.annTurnCount.textContent = `${state.annotation.turns.length} turn${state.annotation.turns.length === 1 ? '' : 's'}`;
  if (el.annOverlapCount) {
    const overlaps = countAnnotationOverlaps();
    el.annOverlapCount.textContent = `${overlaps} overlap${overlaps === 1 ? '' : 's'}`;
  }
  if (el.btnAnnUndo) el.btnAnnUndo.disabled = state.annotation.undo.length === 0;
  if (el.btnAnnRedo) el.btnAnnRedo.disabled = state.annotation.redo.length === 0;
}

function updateAnnotationPlayhead(seconds) {
  if (!state.annotation.current) return;
  const duration = annotationDuration();
  const current = Math.max(0, Math.min(Number(seconds) || 0, duration));
  if (el.annTimecode) el.annTimecode.textContent = formatAnnotationTime(current);
  if (el.annPlayhead) el.annPlayhead.style.left = `${duration ? current / duration * 100 : 0}%`;
  if (el.btnAnnPlay) el.btnAnnPlay.textContent = el.audio && !el.audio.paused ? '❚❚ Pause' : '▶ Play';
  const loopTurn = state.annotation.loopTurnId
    ? state.annotation.turns.find(turn => turn.turn_id === state.annotation.loopTurnId)
    : null;
  if (loopTurn && el.audio && !el.audio.paused && current >= loopTurn.end_s) {
    el.audio.currentTime = loopTurn.start_s;
  }
}

function selectAnnotationTurn(turnId, { seek = false } = {}) {
  state.annotation.selectedTurnId = turnId || null;
  const turn = annotationSelectedTurn();
  if (turn) {
    state.annotation.activeSpeakerId = turn.speaker_id;
    if (seek) seekTo(turn.start_s);
  }
  renderAnnotationEditor();
}

function playAnnotationTurn(turn, { loop = false } = {}) {
  if (!turn || !el.audio?.src) return;
  state.annotation.loopTurnId = loop ? turn.turn_id : null;
  if (el.btnAnnLoopSelected) el.btnAnnLoopSelected.classList.toggle('active', loop);
  if (loop) {
    clearRangePreview();
    seekTo(turn.start_s);
    el.audio.play().catch(error => showToast(`Could not play turn: ${error.message}`, 'error'));
    return;
  }
  previewWorkspaceRangeOnce(turn.start_s, turn.end_s);
}

function deleteSelectedAnnotationTurn() {
  const turnId = state.annotation.selectedTurnId;
  if (!turnId) return;
  annotationChanged(() => {
    state.annotation.turns = state.annotation.turns.filter(turn => turn.turn_id !== turnId);
    state.annotation.selectedTurnId = null;
    if (state.annotation.loopTurnId === turnId) state.annotation.loopTurnId = null;
  });
}

function splitSelectedAnnotationTurn() {
  const turn = annotationSelectedTurn();
  const split = snapAnnotationTime(el.audio?.currentTime || 0);
  if (!turn || split <= turn.start_s + 0.001 || split >= turn.end_s - 0.001) {
    showToast('Place the playhead inside the selected turn before splitting', 'warning');
    return;
  }
  annotationChanged(() => {
    const originalEnd = turn.end_s;
    turn.end_s = split;
    const second = {
      turn_id: `turn_${crypto.randomUUID().replaceAll('-', '')}`,
      speaker_id: turn.speaker_id,
      start_s: split,
      end_s: originalEnd,
    };
    state.annotation.turns.push(second);
    state.annotation.turns.sort((a, b) => a.start_s - b.start_s || a.end_s - b.end_s);
    state.annotation.selectedTurnId = second.turn_id;
  });
}

function addAnnotationSpeaker() {
  if (!state.annotation.current) return;
  const suggested = `Speaker ${state.annotation.speakers.length + 1}`;
  const name = prompt('Speaker display name:', suggested);
  if (name === null || !name.trim()) return;
  annotationChanged(() => {
    let number = state.annotation.speakers.length + 1;
    let speakerId = `spk_${number}`;
    while (annotationSpeaker(speakerId)) speakerId = `spk_${++number}`;
    state.annotation.speakers.push({
      speaker_id: speakerId,
      name: name.trim(),
      color: ANNOTATION_SPEAKER_COLORS[(number - 1) % ANNOTATION_SPEAKER_COLORS.length],
      global_speaker_id: null,
    });
    state.annotation.activeSpeakerId = speakerId;
  });
}

function renameActiveAnnotationSpeaker() {
  const speaker = annotationSpeaker(state.annotation.activeSpeakerId);
  if (!speaker) return;
  const name = prompt('Rename speaker:', speaker.name);
  if (name === null || !name.trim() || name.trim() === speaker.name) return;
  annotationChanged(() => { speaker.name = name.trim(); });
}

async function linkActiveAnnotationSpeaker() {
  const speaker = annotationSpeaker(state.annotation.activeSpeakerId);
  if (!speaker) return;
  if (!state.knownSpeakers?.profiles?.length) await loadSpeakerProfiles();
  const names = (state.knownSpeakers?.profiles || []).map(profile => profile.name);
  if (!names.length) {
    showToast('Create a known speaker profile in the Diarization tab first', 'info');
    return;
  }
  const entered = prompt(`Known profile to link (blank unlinks):\n${names.join(', ')}`, speaker.global_speaker_id || '');
  if (entered === null) return;
  const profile = entered.trim();
  if (profile && !names.includes(profile)) {
    showToast('That known speaker profile does not exist', 'warning');
    return;
  }
  annotationChanged(() => { speaker.global_speaker_id = profile || null; });
}

function mergeActiveAnnotationSpeaker() {
  const source = annotationSpeaker(state.annotation.activeSpeakerId);
  const targets = state.annotation.speakers.filter(speaker => speaker.speaker_id !== source?.speaker_id);
  if (!source || !targets.length) return;
  const entered = prompt(`Merge “${source.name}” into which speaker?\n${targets.map(speaker => `${speaker.name} (${speaker.speaker_id})`).join(', ')}`);
  if (entered === null) return;
  const key = entered.trim();
  const target = targets.find(speaker => speaker.speaker_id === key || speaker.name === key);
  if (!target) {
    showToast('Target speaker was not found', 'warning');
    return;
  }
  const prospective = state.annotation.turns.map(turn => ({
    ...turn,
    speaker_id: turn.speaker_id === source.speaker_id ? target.speaker_id : turn.speaker_id,
  })).sort((a, b) => a.start_s - b.start_s || a.end_s - b.end_s);
  const targetTurns = prospective.filter(turn => turn.speaker_id === target.speaker_id);
  if (targetTurns.some((turn, index) => index > 0 && turn.start_s < targetTurns[index - 1].end_s - 0.000001)) {
    showToast('Merge would create overlapping turns on one speaker lane. Reassign those turns first.', 'warning');
    return;
  }
  annotationChanged(() => {
    state.annotation.turns = prospective;
    state.annotation.speakers = state.annotation.speakers.filter(speaker => speaker.speaker_id !== source.speaker_id);
    state.annotation.activeSpeakerId = target.speaker_id;
  });
}

function removeActiveAnnotationSpeaker() {
  const speaker = annotationSpeaker(state.annotation.activeSpeakerId);
  if (!speaker) return;
  const count = state.annotation.turns.filter(turn => turn.speaker_id === speaker.speaker_id).length;
  if (count) {
    showToast(`Reassign or delete ${count} turn${count === 1 ? '' : 's'} before removing this speaker`, 'warning');
    return;
  }
  if (state.annotation.speakers.length === 1) {
    showToast('An annotation must keep at least one speaker', 'warning');
    return;
  }
  annotationChanged(() => {
    state.annotation.speakers = state.annotation.speakers.filter(item => item.speaker_id !== speaker.speaker_id);
    state.annotation.activeSpeakerId = state.annotation.speakers[0]?.speaker_id || null;
  });
}

function reassignSelectedAnnotationTurn() {
  const turn = annotationSelectedTurn();
  if (!turn) return;
  const entered = prompt(`Reassign to:\n${state.annotation.speakers.map(speaker => `${speaker.name} (${speaker.speaker_id})`).join(', ')}`);
  if (entered === null) return;
  const key = entered.trim();
  const target = state.annotation.speakers.find(speaker => speaker.speaker_id === key || speaker.name === key);
  if (!target || target.speaker_id === turn.speaker_id) return;
  const candidate = { ...turn, speaker_id: target.speaker_id };
  if (annotationTurnOverlap(candidate, turn.turn_id)) {
    showToast('Reassignment would overlap another turn on that speaker lane', 'warning');
    return;
  }
  annotationChanged(() => {
    turn.speaker_id = target.speaker_id;
    state.annotation.activeSpeakerId = target.speaker_id;
  });
}

function beginAnnotationSegmentDrag(event) {
  const segment = event.target.closest('.ann-segment');
  if (!segment || event.button !== 0) return;
  event.preventDefault();
  const turn = state.annotation.turns.find(item => item.turn_id === segment.dataset.turnId);
  if (!turn) return;
  state.annotation.selectedTurnId = turn.turn_id;
  state.annotation.activeSpeakerId = turn.speaker_id;
  const mode = event.target.dataset.edge || 'move';
  state.annotation.drag = {
    turnId: turn.turn_id,
    mode,
    startX: event.clientX,
    originalStart: turn.start_s,
    originalEnd: turn.end_s,
    before: annotationSnapshot(),
    changed: false,
  };
  segment.classList.add('selected', 'dragging');
  renderAnnotationTurnsTable();
}

function moveAnnotationSegmentDrag(event) {
  const drag = state.annotation.drag;
  if (!drag || !el.annTimelineStage) return;
  const turn = state.annotation.turns.find(item => item.turn_id === drag.turnId);
  if (!turn) return;
  const delta = (event.clientX - drag.startX) / el.annTimelineStage.clientWidth * annotationDuration();
  const length = drag.originalEnd - drag.originalStart;
  if (drag.mode === 'start') {
    turn.start_s = Math.min(snapAnnotationTime(drag.originalStart + delta), turn.end_s - 0.001);
  } else if (drag.mode === 'end') {
    turn.end_s = Math.max(snapAnnotationTime(drag.originalEnd + delta), turn.start_s + 0.001);
  } else {
    const start = Math.min(
      snapAnnotationTime(Math.max(0, Math.min(drag.originalStart + delta, annotationDuration() - length))),
      Math.max(0, annotationDuration() - length),
    );
    turn.start_s = start;
    turn.end_s = Number(Math.min(annotationDuration(), start + length).toFixed(3));
  }
  drag.changed = turn.start_s !== drag.originalStart || turn.end_s !== drag.originalEnd;
  renderAnnotationTimeline();
}

function endAnnotationSegmentDrag() {
  const drag = state.annotation.drag;
  if (!drag) return;
  state.annotation.drag = null;
  const turn = state.annotation.turns.find(item => item.turn_id === drag.turnId);
  if (!turn) return;
  const overlap = annotationTurnOverlap(turn, turn.turn_id);
  if (overlap || turn.end_s <= turn.start_s) {
    restoreAnnotationSnapshot(drag.before);
    showToast('Boundary change would create an invalid same-speaker overlap', 'warning');
  } else if (drag.changed) {
    state.annotation.undo.push(drag.before);
    state.annotation.undo = state.annotation.undo.slice(-100);
    state.annotation.redo = [];
    state.annotation.editVersion += 1;
    state.annotation.dirty = true;
    scheduleAnnotationSave();
  }
  renderAnnotationEditor();
}

function annotationTimeFromPointer(event) {
  if (!el.annTimelineStage) return 0;
  const rect = el.annTimelineStage.getBoundingClientRect();
  const x = Math.max(0, Math.min(event.clientX - rect.left, rect.width));
  return snapAnnotationTime(rect.width ? x / rect.width * annotationDuration() : 0);
}

function beginAnnotationRangeDrag(event) {
  if (!state.annotation.current || event.button !== 0 || event.target.closest('.ann-segment')) return;
  const duration = annotationDuration();
  if (!duration || !el.annTimelineStage) return;
  const start = annotationTimeFromPointer(event);
  const lane = event.target.closest('.ann-lane');
  if (lane?.dataset.laneSpeaker) {
    state.annotation.activeSpeakerId = lane.dataset.laneSpeaker;
    renderAnnotationSpeakers();
  }
  state.annotation.rangeDrag = {
    pointerId: event.pointerId,
    startX: event.clientX,
    start,
    previousMarkIn: state.annotation.markIn,
    previousMarkOut: state.annotation.markOut,
    moved: false,
  };
  state.annotation.selectedTurnId = null;
  el.annLanes?.querySelectorAll('.ann-segment.selected').forEach(segment => segment.classList.remove('selected'));
  state.annotation.markIn = start;
  state.annotation.markOut = start;
  el.annTimelineStage.classList.add('selecting');
  updateAnnotationMarks();
}

function moveAnnotationRangeDrag(event) {
  const drag = state.annotation.rangeDrag;
  if (!drag || event.pointerId !== drag.pointerId) return;
  if (Math.abs(event.clientX - drag.startX) >= 3) drag.moved = true;
  if (!drag.moved) return;
  event.preventDefault();
  const current = annotationTimeFromPointer(event);
  state.annotation.markIn = Math.min(drag.start, current);
  state.annotation.markOut = Math.max(drag.start, current);
  updateAnnotationMarks();
}

function endAnnotationRangeDrag(event) {
  const drag = state.annotation.rangeDrag;
  if (!drag || event.pointerId !== drag.pointerId) return;
  state.annotation.rangeDrag = null;
  el.annTimelineStage?.classList.remove('selecting');
  const hasRange = drag.moved && state.annotation.markOut - state.annotation.markIn >= 0.001;
  if (!hasRange) {
    state.annotation.markIn = drag.previousMarkIn;
    state.annotation.markOut = drag.previousMarkOut;
    updateAnnotationMarks();
    return;
  }
  state.annotation.suppressTimelineClick = true;
  setTimeout(() => { state.annotation.suppressTimelineClick = false; }, 0);
  seekTo(state.annotation.markIn);
  renderAnnotationTurnsTable();
  updateAnnotationMarks();
}

function annotationTimelineSeek(event) {
  if (event.target.closest('.ann-segment') || !el.annTimelineStage) return;
  if (state.annotation.suppressTimelineClick) {
    state.annotation.suppressTimelineClick = false;
    return;
  }
  const rect = el.annTimelineStage.getBoundingClientRect();
  seekTo((event.clientX - rect.left) / rect.width * annotationDuration());
}

function updateAnnotationTurnFromRow(row) {
  const turn = state.annotation.turns.find(item => item.turn_id === row?.dataset.turnId);
  if (!turn) return;
  const start = parseAnnotationTime(row.querySelector('.ann-row-start')?.value);
  const end = parseAnnotationTime(row.querySelector('.ann-row-end')?.value);
  const speakerId = row.querySelector('.ann-row-speaker')?.value;
  const candidate = {
    ...turn,
    speaker_id: speakerId,
    start_s: Number(start.toFixed(3)),
    end_s: Number(end.toFixed(3)),
  };
  if (!Number.isFinite(start) || !Number.isFinite(end) || start < 0 || end <= start || end > annotationDuration() + 0.001) {
    showToast(`Turn must satisfy 0 ≤ start < end ≤ ${formatAnnotationTime(annotationDuration())}`, 'warning');
    renderAnnotationTurnsTable();
    return;
  }
  if (annotationTurnOverlap(candidate, turn.turn_id)) {
    showToast('That edit overlaps another turn on the same speaker lane', 'warning');
    renderAnnotationTurnsTable();
    return;
  }
  annotationChanged(() => {
    turn.speaker_id = candidate.speaker_id;
    turn.start_s = candidate.start_s;
    turn.end_s = candidate.end_s;
    state.annotation.activeSpeakerId = candidate.speaker_id;
    state.annotation.turns.sort((left, right) => left.start_s - right.start_s || left.end_s - right.end_s);
  });
}

const CUT_SOURCE_ID_SUFFIX = /_\d+\.\d{3}-\d+\.\d{3}$/;

function annotationAudioIsCut(audioId, history, fingerprint) {
  if (Array.isArray(history) && history.some(step => String(step).startsWith('cut_'))) return true;
  if (fingerprint && String(fingerprint).includes('__cut_')) return true;
  return Boolean(audioId && CUT_SOURCE_ID_SUFFIX.test(String(audioId)));
}

function annotationSourceMatchKind(result) {
  const selectedAudio = state.audioList.find(item => item.id === state.annotation.audioId);
  const annotation = state.annotation.current || (selectedAudio ? {
    audio_id: selectedAudio.source_id,
    source_audio: selectedAudio,
  } : null);
  const source = annotation?.source_audio || {};
  const resultSource = result.source_audio || {};
  const annotationFingerprint = source.fingerprint || '';
  const resultFingerprint = resultSource.fingerprint || '';
  if (annotationFingerprint && resultFingerprint && annotationFingerprint === resultFingerprint) {
    return 'fingerprint';
  }
  if (source.path && resultSource.path && normalizedAudioPath(source.path) === normalizedAudioPath(resultSource.path)) {
    return 'path';
  }
  const annotationDuration = Number(source.duration_s);
  const resultDuration = Number(resultSource.duration_s);
  const annotationIsCut = annotationAudioIsCut(
    annotation?.audio_id,
    source.history,
    annotationFingerprint,
  );
  const resultIsCut = annotationAudioIsCut(
    result.audio_id || resultSource.source_id,
    resultSource.history,
    resultFingerprint,
  );
  if (
    annotation?.audio_id
    && annotation.audio_id === result.audio_id
    && Number.isFinite(annotationDuration)
    && Number.isFinite(resultDuration)
    && Math.abs(annotationDuration - resultDuration) <= 0.05
    && !annotationIsCut
    && !resultIsCut
  ) {
    return 'timeline';
  }
  return null;
}

function annotationSourceMatchesResult(result) {
  return annotationSourceMatchKind(result) != null;
}

function annotationResultTimelineLabel(result) {
  const source = result?.source_audio || {};
  const history = Array.isArray(source.history) ? source.history.filter(Boolean) : [];
  if (history.length) return history.join(' · ');
  const fingerprint = String(source.fingerprint || '');
  if (fingerprint) {
    const parts = fingerprint.split('__').filter(Boolean);
    if (parts.length >= 3) return parts.slice(1, -1).join(' · ');
  }
  return 'source mix';
}

async function loadCompatibleDiarizationResults() {
  if (!state.annotation.current && !state.annotation.audioId) return;
  try {
    const payload = await parseJsonResponse(await fetch('/api/diarization/results'));
    state.annotation.resultCatalog = (payload.results || []).filter(annotationSourceMatchesResult);
    renderAnnotationResultList();
    renderAnnotationSeedSelector();
  } catch (error) {
    if (el.annResultList) el.annResultList.innerHTML = `<div class="empty-placeholder">${escapeHtml(error.message)}</div>`;
    if (el.annSeedResultMeta) el.annSeedResultMeta.textContent = error.message;
  }
}

function renderAnnotationSeedSelector() {
  if (!el.annSeedResultSelect) return;
  const results = state.annotation.resultCatalog || [];
  el.annSeedResultSelect.innerHTML = '<option value="">Select compatible saved result…</option>' + results.map(result => {
    const model = result.model?.model_id || result.model?.backend || 'Unknown model';
    const summary = result.summary || {};
    const exact = ['fingerprint', 'path'].includes(annotationSourceMatchKind(result));
    return `<option value="${escapeHtml(result.result_id)}">${escapeHtml(model)} — ${summary.speaker_count || 0} speakers · ${summary.turn_count || 0} turns · ${exact ? 'exact file' : 'same timeline'}</option>`;
  }).join('');
  el.annSeedResultSelect.disabled = !results.length;
  el.btnAnnCreateSeed.disabled = true;
  if (el.annSeedResultMeta) {
    el.annSeedResultMeta.textContent = results.length
      ? `${results.length} compatible saved result${results.length === 1 ? '' : 's'}.`
      : 'No exact-file or same-timeline results available.';
  }
}

function renderAnnotationSeedNotice() {
  const seed = state.annotation.current?.seed;
  el.annSeedNotice?.classList.toggle('hidden', !seed);
  if (!seed || !el.annSeedNoticeDetail) return;
  const model = seed.model?.model_id || seed.model?.backend || 'saved model result';
  el.annSeedNoticeDetail.textContent = `Seeded from ${model} (${seed.result_id}). Verify speaker identities and every boundary before treating this reference as ground truth.`;
}

async function createMachineSeededAnnotation() {
  const seedResultId = el.annSeedResultSelect?.value;
  const audioId = state.annotation.audioId || el.annAudioSelect?.value;
  if (!audioId || !seedResultId) return;
  await saveAnnotationNow();
  el.btnAnnCreateSeed.disabled = true;
  el.btnAnnCreateSeed.textContent = 'Creating…';
  try {
    const saved = await parseJsonResponse(await fetch('/api/diarization/annotations', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_audio_id: audioId, seed_result_id: seedResultId }),
    }));
    await loadAnnotation(saved.annotation_id);
    showToast('Editable machine-seeded reference created', 'success');
  } finally {
    el.btnAnnCreateSeed.textContent = 'Create editable draft';
    el.btnAnnCreateSeed.disabled = !el.annSeedResultSelect?.value;
  }
}

function renderAnnotationResultList() {
  if (!el.annResultList) return;
  const results = state.annotation.resultCatalog || [];
  if (!results.length) {
    el.annResultList.innerHTML = '<div class="empty-placeholder">No compatible model results. Run diarization on this clip or a same-timeline stem, then refresh.</div>';
    return;
  }
  el.annResultList.innerHTML = results.map(result => {
    const model = result.model?.model_id || result.model?.backend || 'Unknown model';
    const summary = result.summary || {};
    const matchKind = annotationSourceMatchKind(result);
    const exact = matchKind === 'fingerprint' || matchKind === 'path';
    const badge = exact ? 'exact file' : 'same timeline';
    const audioLabel = annotationResultTimelineLabel(result);
    return `<label class="ann-result-option">
      <input type="checkbox" data-ann-result-id="${escapeHtml(result.result_id)}" ${result.result_id === state.annotation.seedResultId ? 'checked' : ''}>
      <span><strong>${escapeHtml(model)}</strong><small>${escapeHtml(result.result_id)} · ${summary.speaker_count || 0} speakers · ${summary.turn_count || 0} turns</small><small>${escapeHtml(audioLabel)}</small></span>
      <span class="badge ${exact ? 'badge-success' : 'badge-accent'}">${escapeHtml(badge)}</span>
    </label>`;
  }).join('');
}

async function evaluateSelectedAnnotationResults() {
  if (!state.annotation.current?.annotation_id) {
    showToast('Wait for the reference annotation to finish saving', 'warning');
    return;
  }
  await saveAnnotationNow();
  const resultIds = [...el.annResultList.querySelectorAll('input[data-ann-result-id]:checked')].map(input => input.dataset.annResultId);
  if (!resultIds.length) {
    showToast('Select at least one compatible model result', 'warning');
    return;
  }
  const collarS = Number(el.annCollarInput?.value);
  if (!Number.isFinite(collarS) || collarS < 0 || collarS > 10) {
    showToast('Boundary collar must be between 0 and 10 seconds', 'warning');
    return;
  }
  el.btnAnnEvaluate.disabled = true;
  el.btnAnnEvaluate.textContent = 'Evaluating…';
  try {
    const response = await fetch('/api/diarization/evaluate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        annotation_id: state.annotation.current.annotation_id,
        result_ids: resultIds,
        collar_s: collarS,
        skip_overlap: Boolean(el.annSkipOverlap?.checked),
      }),
    });
    state.annotation.evaluation = await parseJsonResponse(response);
    renderAnnotationEvaluation();
    showToast(`Evaluated ${resultIds.length} model result${resultIds.length === 1 ? '' : 's'}`, 'success');
  } catch (error) {
    showToast(`Evaluation failed: ${error.message}`, 'error');
  } finally {
    el.btnAnnEvaluate.disabled = false;
    el.btnAnnEvaluate.textContent = 'Evaluate selected models';
  }
}

function renderAnnotationEvaluation() {
  if (!el.annEvalResults) return;
  const reports = state.annotation.evaluation?.results || [];
  el.annEvalResults.classList.toggle('hidden', reports.length === 0);
  if (el.btnAnnDownloadReport) el.btnAnnDownloadReport.disabled = reports.length === 0;
  el.annEvalResults.innerHTML = reports.map((report, index) => {
    const model = report.model?.model_id || report.model?.backend || report.result_id;
    const catalogItem = (state.annotation.resultCatalog || []).find(item => item.result_id === report.result_id);
    const audioLabel = catalogItem ? annotationResultTimelineLabel(catalogItem) : '';
    const mapping = (report.speaker_mapping || []).map(item => `${item.hypothesis_speaker_id} → ${annotationSpeaker(item.reference_speaker_id)?.name || item.reference_speaker_id}`).join(' · ');
    return `<section class="ann-eval-model">
      <div class="flex-row items-center gap-2 flex-wrap"><span class="badge ${index === 0 ? 'badge-success' : 'badge-ghost'}">#${index + 1}</span><h4>${escapeHtml(model)}</h4><span class="text-xs text-muted">${escapeHtml(report.result_id)}${audioLabel ? ` · ${escapeHtml(audioLabel)}` : ''}</span></div>
      <div class="ann-score-grid">
        <div class="ann-score-card"><span>DER</span><strong>${Number(report.der_pct).toFixed(2)}%</strong></div>
        <div class="ann-score-card"><span>JER</span><strong>${Number(report.jer_pct).toFixed(2)}%</strong></div>
        <div class="ann-score-card"><span>Missed speech</span><strong>${Number(report.missed_speech_s).toFixed(3)}s</strong></div>
        <div class="ann-score-card"><span>False alarm</span><strong>${Number(report.false_alarm_s).toFixed(3)}s</strong></div>
        <div class="ann-score-card"><span>Confusion</span><strong>${Number(report.speaker_confusion_s).toFixed(3)}s</strong></div>
        <div class="ann-score-card"><span>Reference speech</span><strong>${Number(report.reference_speaker_s).toFixed(3)}s</strong></div>
        <div class="ann-score-card"><span>Scored audio</span><strong>${Number(report.scored_audio_s).toFixed(3)}s</strong></div>
        <div class="ann-score-card"><span>Source</span><strong style="font-size:12px">${escapeHtml(report.source_match)}</strong></div>
      </div>
      <div class="text-xs"><strong>Optimal speaker mapping:</strong> ${escapeHtml(mapping || 'No temporal speaker mapping')}</div>
      <div class="turns-table-wrapper" style="margin-top:10px"><table class="turns-table"><thead><tr><th>Reference</th><th>Model speaker</th><th>Reference</th><th>Intersection</th><th>Coverage</th><th>JER</th></tr></thead><tbody>${(report.per_speaker || []).map(item => `<tr><td>${escapeHtml(annotationSpeaker(item.reference_speaker_id)?.name || item.reference_speaker_id)}</td><td>${escapeHtml(item.hypothesis_speaker_id || 'Unmapped')}</td><td>${Number(item.reference_s).toFixed(3)}s</td><td>${Number(item.intersection_s).toFixed(3)}s</td><td>${Number(item.coverage_pct).toFixed(2)}%</td><td>${Number(item.jer_pct).toFixed(2)}%</td></tr>`).join('')}</tbody></table></div>
    </section>`;
  }).join('');
}

function downloadTextFile(content, filename, type = 'application/json') {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function exportAnnotation() {
  if (!state.annotation.current) return;
  await saveAnnotationNow();
  const format = el.annExportFormat?.value || 'json';
  const base = (state.annotation.current.name || 'diarization_reference').replace(/[^a-z0-9_-]+/gi, '_');
  if (format === 'rttm') {
    const fileId = (state.annotation.current.audio_id || base).replace(/\s+/g, '_');
    const lines = state.annotation.turns.map(turn => {
      const speaker = annotationSpeaker(turn.speaker_id);
      const label = (speaker?.name || turn.speaker_id).replace(/\s+/g, '_');
      return `SPEAKER ${fileId} 1 ${turn.start_s.toFixed(6)} ${(turn.end_s - turn.start_s).toFixed(6)} <NA> <NA> ${label} <NA> <NA>`;
    });
    downloadTextFile(`${lines.join('\n')}\n`, `${base}.rttm`, 'text/plain');
  } else {
    downloadTextFile(`${JSON.stringify({ ...state.annotation.current, speakers: state.annotation.speakers, turns: state.annotation.turns }, null, 2)}\n`, `${base}.json`);
  }
}

function parseImportedAnnotation(text, filename) {
  if (filename.toLowerCase().endsWith('.rttm')) {
    const turns = [];
    const names = new Set();
    text.split(/\r?\n/).forEach((line, lineIndex) => {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith('#')) return;
      const fields = trimmed.split(/\s+/);
      if (fields.length < 8 || fields[0] !== 'SPEAKER') throw new Error(`Invalid RTTM line ${lineIndex + 1}`);
      const start = Number(fields[3]);
      const duration = Number(fields[4]);
      const name = fields[7];
      if (!Number.isFinite(start) || !Number.isFinite(duration) || duration <= 0) throw new Error(`Invalid RTTM timestamps on line ${lineIndex + 1}`);
      names.add(name);
      turns.push({ importedName: name, start_s: start, end_s: start + duration });
    });
    const speakers = [...names].map((name, index) => ({ speaker_id: `spk_${index + 1}`, name, color: ANNOTATION_SPEAKER_COLORS[index % ANNOTATION_SPEAKER_COLORS.length], global_speaker_id: null }));
    const idByName = new Map(speakers.map(speaker => [speaker.name, speaker.speaker_id]));
    return {
      name: filename.replace(/\.rttm$/i, ''),
      speakers,
      turns: turns.map(turn => ({ turn_id: `turn_${crypto.randomUUID().replaceAll('-', '')}`, speaker_id: idByName.get(turn.importedName), start_s: turn.start_s, end_s: turn.end_s })),
    };
  }
  const payload = parseJsonText(text);
  const rawTurns = payload.turns || payload.diarization?.turns;
  if (!Array.isArray(rawTurns)) throw new Error('JSON does not contain a turns array');
  const rawSpeakers = Array.isArray(payload.speakers) ? payload.speakers : [];
  const ids = [...new Set([...rawSpeakers.map(speaker => speaker.speaker_id), ...rawTurns.map(turn => turn.speaker_id)].filter(Boolean))];
  const speakers = ids.map((speakerId, index) => {
    const source = rawSpeakers.find(speaker => speaker.speaker_id === speakerId) || {};
    return { speaker_id: String(speakerId), name: String(source.name || source.global_speaker_id || speakerId), color: source.color || ANNOTATION_SPEAKER_COLORS[index % ANNOTATION_SPEAKER_COLORS.length], global_speaker_id: source.global_speaker_id || null };
  });
  return {
    name: payload.name || filename.replace(/\.json$/i, ''),
    speakers,
    turns: rawTurns.map(turn => ({ turn_id: turn.turn_id || `turn_${crypto.randomUUID().replaceAll('-', '')}`, speaker_id: String(turn.speaker_id), start_s: Number(turn.start_s), end_s: Number(turn.end_s) })),
  };
}

function validateImportedAnnotation(imported) {
  if (!imported.speakers.length) throw new Error('Annotation must contain at least one speaker');
  const speakerIds = new Set(imported.speakers.map(speaker => speaker.speaker_id));
  const duration = annotationDuration();
  imported.turns.forEach((turn, index) => {
    if (!speakerIds.has(turn.speaker_id)) throw new Error(`Turn ${index + 1} references an unknown speaker`);
    if (!Number.isFinite(turn.start_s) || !Number.isFinite(turn.end_s) || turn.start_s < 0 || turn.end_s <= turn.start_s || turn.end_s > duration + 0.001) {
      throw new Error(`Turn ${index + 1} is outside the ${formatAnnotationTime(duration)} source duration`);
    }
    turn.start_s = Number(turn.start_s.toFixed(3));
    turn.end_s = Number(Math.min(duration, turn.end_s).toFixed(3));
  });
  imported.turns.sort((left, right) => left.start_s - right.start_s || left.end_s - right.end_s);
  imported.speakers.forEach(speaker => {
    const turns = imported.turns.filter(turn => turn.speaker_id === speaker.speaker_id);
    if (turns.some((turn, index) => index > 0 && turn.start_s < turns[index - 1].end_s - 0.000001)) {
      throw new Error(`${speaker.name} has overlapping turns; simultaneous speech must use separate speaker lanes`);
    }
  });
  return imported;
}

function initAnnotationTab() {
  loadAnnotationCatalog();
  el.annAudioSelect?.addEventListener('change', async () => {
    const value = el.annAudioSelect.value;
    if (!value) return;
    await selectAnnotationAudio(value);
  });
  el.btnAnnBrowseLibrary?.addEventListener('click', () => openLibraryModal('annotation'));
  el.btnAnnNew?.addEventListener('click', createAnnotationForSelectedAudio);
  el.annSeedResultSelect?.addEventListener('change', () => {
    const result = state.annotation.resultCatalog.find(item => item.result_id === el.annSeedResultSelect.value);
    el.btnAnnCreateSeed.disabled = !result;
    if (!result || !el.annSeedResultMeta) return;
    const model = result.model?.model_id || result.model?.backend || 'Unknown model';
    const summary = result.summary || {};
    const exact = ['fingerprint', 'path'].includes(annotationSourceMatchKind(result));
    el.annSeedResultMeta.textContent = `${model} · ${summary.speaker_count || 0} speakers · ${summary.turn_count || 0} turns · ${exact ? 'exact file' : 'same timeline'}`;
  });
  el.btnAnnCreateSeed?.addEventListener('click', () => {
    createMachineSeededAnnotation().catch(error => showToast(`Could not create draft: ${error.message}`, 'error'));
  });
  el.btnAnnLoad?.addEventListener('click', () => {
    const annotationId = el.annSavedSelect?.value;
    if (!annotationId) showToast('Choose a saved annotation first', 'info');
    else loadAnnotation(annotationId).catch(error => showToast(error.message, 'error'));
  });
  el.annSavedSelect?.addEventListener('dblclick', () => el.btnAnnLoad?.click());
  el.btnAnnDelete?.addEventListener('click', async () => {
    const annotationId = el.annSavedSelect?.value || state.annotation.current?.annotation_id;
    if (!annotationId || !confirm('Delete this saved ground-truth annotation? The audio file is not removed.')) return;
    try {
      clearTimeout(state.annotation.saveTimer);
      state.annotation.saveTimer = null;
      if (state.annotation.savePromise) await state.annotation.savePromise;
      await parseJsonResponse(await fetch(`/api/diarization/annotations/${encodeURIComponent(annotationId)}`, { method: 'DELETE' }));
      if (state.annotation.current?.annotation_id === annotationId) clearAnnotationEditor();
      await loadAnnotationCatalog();
      showToast('Annotation deleted; source audio was kept', 'success');
    } catch (error) {
      showToast(`Delete failed: ${error.message}`, 'error');
    }
  });
  el.annNameInput?.addEventListener('change', () => {
    if (!state.annotation.current) return;
    const value = el.annNameInput.value.trim();
    if (!value) {
      el.annNameInput.value = state.annotation.current.name || 'Ground truth';
      return;
    }
    annotationChanged(() => { state.annotation.current.name = value; });
  });

  el.btnAnnStart?.addEventListener('click', () => seekTo(0));
  el.btnAnnBack1?.addEventListener('click', () => seekRelative(-1));
  el.btnAnnForward1?.addEventListener('click', () => seekRelative(1));
  el.btnAnnBackFrame?.addEventListener('click', () => seekRelative(-state.annotation.stepS));
  el.btnAnnForwardFrame?.addEventListener('click', () => seekRelative(state.annotation.stepS));
  el.btnAnnPlay?.addEventListener('click', togglePlayPause);
  el.annStepSelect?.addEventListener('change', () => { state.annotation.stepS = Number(el.annStepSelect.value) || 0.1; });
  el.annSnapSelect?.addEventListener('change', () => { state.annotation.snapS = Number(el.annSnapSelect.value) || 0; });
  el.annSpeedSelect?.addEventListener('change', () => setPlaybackRate(el.annSpeedSelect.value));
  el.btnAnnUndo?.addEventListener('click', undoAnnotation);
  el.btnAnnRedo?.addEventListener('click', redoAnnotation);

  el.btnAnnAddSpeaker?.addEventListener('click', addAnnotationSpeaker);
  el.btnAnnRenameSpeaker?.addEventListener('click', renameActiveAnnotationSpeaker);
  el.btnAnnLinkSpeaker?.addEventListener('click', linkActiveAnnotationSpeaker);
  el.btnAnnMergeSpeaker?.addEventListener('click', mergeActiveAnnotationSpeaker);
  el.btnAnnRemoveSpeaker?.addEventListener('click', removeActiveAnnotationSpeaker);

  const setMarkIn = () => {
    if (!state.annotation.current) return;
    state.annotation.markIn = snapAnnotationTime(el.audio?.currentTime || 0);
    state.annotation.markOut = null;
    updateAnnotationMarks();
  };
  const setMarkOutAndCreate = () => {
    if (!state.annotation.current) return;
    state.annotation.markOut = snapAnnotationTime(el.audio?.currentTime || 0);
    updateAnnotationMarks();
    createAnnotationTurn();
  };
  el.btnAnnSetIn?.addEventListener('click', setMarkIn);
  el.btnAnnSetOut?.addEventListener('click', setMarkOutAndCreate);
  el.btnAnnCreateTurn?.addEventListener('click', () => {
    const start = parseAnnotationTime(el.annMarkIn?.value);
    const end = parseAnnotationTime(el.annMarkOut?.value);
    createAnnotationTurn(start, end);
  });
  el.btnAnnClearMarks?.addEventListener('click', () => {
    state.annotation.markIn = null;
    state.annotation.markOut = null;
    updateAnnotationMarks();
  });
  el.annMarkIn?.addEventListener('change', () => {
    const value = parseAnnotationTime(el.annMarkIn.value);
    if (!Number.isFinite(value) || value > annotationDuration()) return updateAnnotationMarks();
    state.annotation.markIn = snapAnnotationTime(value);
    updateAnnotationMarks();
  });
  el.annMarkOut?.addEventListener('change', () => {
    const value = parseAnnotationTime(el.annMarkOut.value);
    if (!Number.isFinite(value) || value > annotationDuration()) return updateAnnotationMarks();
    state.annotation.markOut = snapAnnotationTime(value);
    updateAnnotationMarks();
  });

  const setAnnotationZoom = value => {
    state.annotation.zoom = clampTimelineZoom(value, 1);
    state.annotation.waveform.data = null;
    renderAnnotationTimeline();
  };
  el.btnAnnZoomOut?.addEventListener('click', () => setAnnotationZoom(state.annotation.zoom / 1.5));
  el.btnAnnZoomIn?.addEventListener('click', () => setAnnotationZoom(state.annotation.zoom * 1.5));
  el.btnAnnZoomFit?.addEventListener('click', () => setAnnotationZoom(1));
  el.annZoomRange?.addEventListener('input', () => setAnnotationZoom(timelineSliderToZoom(el.annZoomRange.value, 1)));
  if (el.annZoomInput) {
    const handleAnnZoomInput = event => {
      const val = parseFloat(event.target.value);
      if (!isNaN(val) && val > 0) setAnnotationZoom(val);
    };
    el.annZoomInput.addEventListener('input', handleAnnZoomInput);
    el.annZoomInput.addEventListener('change', handleAnnZoomInput);
    el.annZoomInput.addEventListener('keydown', event => {
      if (event.key === 'Enter') {
        handleAnnZoomInput(event);
        el.annZoomInput.blur();
      }
    });
  }
  el.annTimelineScroll?.addEventListener('scroll', () => {
    state.annotation.waveform.data = null;
    renderAnnotationWaveform();
    renderAnnotationRuler();
    scheduleAnnotationWaveform();
  }, { passive: true });
  el.annTimelineStage?.addEventListener('click', annotationTimelineSeek);
  el.annTimelineStage?.addEventListener('pointerdown', beginAnnotationRangeDrag);
  el.annLanes?.addEventListener('pointerdown', beginAnnotationSegmentDrag);
  document.addEventListener('pointermove', moveAnnotationSegmentDrag);
  document.addEventListener('pointermove', moveAnnotationRangeDrag);
  document.addEventListener('pointerup', endAnnotationSegmentDrag);
  document.addEventListener('pointerup', endAnnotationRangeDrag);
  document.addEventListener('pointercancel', endAnnotationSegmentDrag);
  document.addEventListener('pointercancel', endAnnotationRangeDrag);
  el.annLanes?.addEventListener('dblclick', event => {
    const segment = event.target.closest('.ann-segment');
    if (segment) selectAnnotationTurn(segment.dataset.turnId, { seek: true });
  });

  el.annTurnSearch?.addEventListener('input', renderAnnotationTurnsTable);
  el.annTurnsBody?.addEventListener('click', event => {
    const row = event.target.closest('tr[data-turn-id]');
    if (!row) return;
    const turn = state.annotation.turns.find(item => item.turn_id === row.dataset.turnId);
    if (event.target.closest('.ann-row-delete')) {
      state.annotation.selectedTurnId = row.dataset.turnId;
      deleteSelectedAnnotationTurn();
    } else if (event.target.closest('.ann-row-play')) {
      selectAnnotationTurn(row.dataset.turnId);
      playAnnotationTurn(turn);
    } else if (!event.target.closest('input,select,button')) {
      selectAnnotationTurn(row.dataset.turnId, { seek: true });
    }
  });
  el.annTurnsBody?.addEventListener('change', event => {
    if (event.target.matches('.ann-row-start,.ann-row-end,.ann-row-speaker')) updateAnnotationTurnFromRow(event.target.closest('tr'));
  });
  el.btnAnnDeleteTurn?.addEventListener('click', deleteSelectedAnnotationTurn);
  el.btnAnnSplit?.addEventListener('click', splitSelectedAnnotationTurn);
  el.btnAnnReassign?.addEventListener('click', reassignSelectedAnnotationTurn);
  el.btnAnnLoopSelected?.addEventListener('click', () => {
    const turn = annotationSelectedTurn();
    if (!turn) return;
    const enable = state.annotation.loopTurnId !== turn.turn_id;
    state.annotation.loopTurnId = enable ? turn.turn_id : null;
    el.btnAnnLoopSelected.classList.toggle('active', enable);
    if (enable) playAnnotationTurn(turn, { loop: true });
    else {
      el.audio?.pause();
      clearRangePreview();
    }
  });

  el.btnAnnRefreshResults?.addEventListener('click', loadCompatibleDiarizationResults);
  el.btnAnnEvaluate?.addEventListener('click', evaluateSelectedAnnotationResults);
  el.btnAnnDownloadReport?.addEventListener('click', () => {
    if (!state.annotation.evaluation) return;
    downloadTextFile(`${JSON.stringify(state.annotation.evaluation, null, 2)}\n`, `${state.annotation.current?.name || 'diarization'}_evaluation.json`);
  });
  el.btnAnnExport?.addEventListener('click', exportAnnotation);
  el.btnAnnImport?.addEventListener('click', () => el.annImportInput?.click());
  el.annImportInput?.addEventListener('change', async () => {
    const file = el.annImportInput.files?.[0];
    if (!file) return;
    try {
      if (!state.annotation.current) await createAnnotationForSelectedAudio();
      if (!state.annotation.current) return;
      const imported = validateImportedAnnotation(parseImportedAnnotation(await file.text(), file.name));
      annotationChanged(() => {
        state.annotation.current.name = imported.name;
        state.annotation.speakers = imported.speakers;
        state.annotation.turns = imported.turns;
        state.annotation.activeSpeakerId = imported.speakers[0]?.speaker_id || null;
        state.annotation.selectedTurnId = null;
        if (el.annNameInput) el.annNameInput.value = imported.name;
      });
      showToast(`Imported ${imported.turns.length} turns from ${file.name}`, 'success');
    } catch (error) {
      showToast(`Import failed: ${error.message}`, 'error');
    } finally {
      el.annImportInput.value = '';
    }
  });

  window.addEventListener('beforeunload', event => {
    if (!state.annotation.dirty) return;
    event.preventDefault();
    event.returnValue = '';
  });
  window.addEventListener('resize', () => {
    if (state.activeTab === 'tab-annotation') renderAnnotationTimeline();
  });
}

function savePurityPreferences() {
  try {
    localStorage.setItem('sonic_purity_preferences', JSON.stringify({
      settings: state.purity.settings,
      overlap: state.purity.overlap,
    }));
  } catch (_) {}
}

function purityOverlapVerifierPayload() {
  const overlap = state.purity.overlap;
  overlap.prompt = el.purityOverlapPrompt?.value.trim() || overlap.prompt;
  overlap.model = overlap.backend === 'vibevoice'
    ? (el.purityOverlapVibevoiceModel?.value || overlap.model)
    : (el.purityOverlapModel?.value.trim() || overlap.model);
  overlap.endpoint = el.purityOverlapEndpoint?.value.trim() || overlap.endpoint;
  if (el.purityVibevoiceSecondary) {
    overlap.minSecondarySpeech = Math.max(0, parseFloat(el.purityVibevoiceSecondary.value) || 0.25);
  }
  if (el.purityVibevoiceBatchSize) {
    overlap.batchSize = Math.max(1, parseInt(el.purityVibevoiceBatchSize.value, 10) || 1);
  }
  return {
    enabled: true,
    backend: overlap.backend,
    model: overlap.model,
    endpoint: overlap.backend === 'gemma4' ? overlap.endpoint : undefined,
    api_key: overlap.backend === 'vibevoice' ? undefined : (el.purityOverlapApiKey?.value || undefined),
    timeout_s: overlap.timeout,
    max_output_tokens: overlap.backend === 'vibevoice' ? undefined : overlap.maxOutputTokens,
    max_new_tokens: overlap.backend === 'vibevoice' ? overlap.maxOutputTokens : undefined,
    min_secondary_speech_s: overlap.backend === 'vibevoice' ? overlap.minSecondarySpeech : undefined,
    batch_size: overlap.backend === 'vibevoice' ? overlap.batchSize : undefined,
    prompt: overlap.backend === 'vibevoice' ? undefined : overlap.prompt,
    failure_policy: overlap.failurePolicy,
  };
}

function applyPurityControls() {
  const overlap = state.purity.overlap;
  overlap.enabled = true;
  if (el.purityOverlapBackend) el.purityOverlapBackend.value = overlap.backend;
  if (el.purityOverlapModel) el.purityOverlapModel.value = overlap.model;
  if (overlap.backend === 'vibevoice') {
    overlap.model = populateVibevoiceModelSelect(overlap.model);
  }
  if (el.purityOverlapEndpoint) el.purityOverlapEndpoint.value = overlap.endpoint;
  if (el.purityOverlapTimeout) el.purityOverlapTimeout.value = overlap.timeout;
  if (el.purityOverlapMaxTokens) el.purityOverlapMaxTokens.value = overlap.maxOutputTokens;
  if (el.purityOverlapFailurePolicy) el.purityOverlapFailurePolicy.value = overlap.failurePolicy;
  if (el.purityOverlapPrompt) el.purityOverlapPrompt.value = overlap.prompt;
  if (el.purityVibevoiceSecondary) el.purityVibevoiceSecondary.value = overlap.minSecondarySpeech ?? 0.25;
  if (el.purityVibevoiceBatchSize) el.purityVibevoiceBatchSize.value = overlap.batchSize ?? 1;
  syncPurityOverlapUi();
}

function applyPurityVerifierStatus(status) {
  state.purity.verifierStatus = status || null;
  const ready = Boolean(status?.ready);
  const message = status?.message || 'Verifier status unknown.';
  if (el.purityOverlapStatusBadge) {
    if (!status) {
      el.purityOverlapStatusBadge.textContent = 'Checking…';
      el.purityOverlapStatusBadge.className = 'badge badge-sm badge-ghost';
    } else if (ready) {
      el.purityOverlapStatusBadge.textContent = `${purityOverlapBackendLabel()} ready`;
      el.purityOverlapStatusBadge.className = 'badge badge-sm badge-success';
    } else {
      el.purityOverlapStatusBadge.textContent = `${purityOverlapBackendLabel()} not ready`;
      el.purityOverlapStatusBadge.className = 'badge badge-sm badge-danger';
    }
  }
  if (el.purityVerifierStatusMsg) {
    el.purityVerifierStatusMsg.textContent = message;
    el.purityVerifierStatusMsg.className = `purity-verifier-status-msg ${ready ? 'is-ready' : 'is-unready'}`;
  }
}

async function refreshPurityVerifierStatus() {
  const overlap = state.purity.overlap;
  applyPurityVerifierStatus({ ready: false, message: `Checking ${purityOverlapBackendLabel()}…` });
  try {
    const params = new URLSearchParams({ backend: overlap.backend || 'gemma4' });
    if (overlap.model) params.set('model', overlap.model);
    if (overlap.backend === 'vibevoice') params.set('device', el.purityDeviceSelect?.value || 'auto');
    if (overlap.backend === 'gemma4' && overlap.endpoint) params.set('endpoint', overlap.endpoint);
    const response = await fetch(`/api/purity/verifier-status?${params}`);
    const text = await response.text();
    let data;
    try {
      data = parseJsonText(text);
    } catch (_) {
      if (response.status === 404) {
        data = { ready: false, message: 'Verifier status endpoint not found (HTTP 404). Please ensure the backend server was restarted with the latest routes!' };
      } else {
        data = { ready: false, message: `Server returned HTTP ${response.status}: ${text.substring(0, 150) || response.statusText}` };
      }
    }
    const status = (data && typeof data === 'object') ? data : {
      ready: false,
      message: text ? text.substring(0, 150) : `HTTP ${response.status}`,
    };
    applyPurityVerifierStatus(status);
    return status;
  } catch (err) {
    const status = {
      ready: false,
      message: err instanceof Error ? err.message : String(err),
    };
    applyPurityVerifierStatus(status);
    return status;
  }
}

function syncPurityOverlapUi() {
  const overlap = state.purity.overlap;
  overlap.enabled = true;
  const isVibevoice = overlap.backend === 'vibevoice';
  const isGemma = overlap.backend === 'gemma4';
  if (el.purityOverlapConfig) el.purityOverlapConfig.classList.remove('hidden');
  if (el.purityOverlapEndpointField) el.purityOverlapEndpointField.classList.toggle('hidden', !isGemma);
  if (el.purityOverlapKeyField) el.purityOverlapKeyField.classList.toggle('hidden', isVibevoice);
  if (el.purityOverlapPromptField) el.purityOverlapPromptField.classList.toggle('hidden', isVibevoice);
  if (el.purityOverlapTimeoutField) el.purityOverlapTimeoutField.classList.toggle('hidden', isVibevoice);
  if (el.purityVibevoiceSecondaryField) el.purityVibevoiceSecondaryField.classList.toggle('hidden', !isVibevoice);
  if (el.purityVibevoiceBatchField) el.purityVibevoiceBatchField.classList.toggle('hidden', !isVibevoice);
  if (el.purityVibevoiceDeviceField) el.purityVibevoiceDeviceField.classList.toggle('hidden', !isVibevoice);
  if (el.purityVibevoiceHfField) el.purityVibevoiceHfField.classList.toggle('hidden', !isVibevoice);
  if (el.purityOverlapModel) el.purityOverlapModel.classList.toggle('hidden', isVibevoice);
  if (el.purityOverlapVibevoiceModel) el.purityOverlapVibevoiceModel.classList.toggle('hidden', !isVibevoice);
  if (el.purityVibevoiceModelHint) el.purityVibevoiceModelHint.classList.toggle('hidden', !isVibevoice);
  if (el.purityOverlapModelLabel) {
    el.purityOverlapModelLabel.textContent = isVibevoice ? 'VibeVoice checkpoint' : 'Model ID';
    el.purityOverlapModelLabel.setAttribute(
      'for',
      isVibevoice ? 'purity-overlap-vibevoice-model' : 'purity-overlap-model'
    );
  }
  if (el.purityOverlapMaxTokensLabel) {
    el.purityOverlapMaxTokensLabel.textContent = isVibevoice ? 'Max new tokens' : 'Max output tokens';
  }
  if (el.btnPurityCheckVerifier) {
    el.btnPurityCheckVerifier.textContent = isGemma ? 'Check Unsloth' : (isVibevoice ? 'Check VibeVoice' : 'Check Gemini');
  }
  if (el.purityOverlapKeyStatus) {
    const configured = purityBackendDefaults(overlap.backend).api_key_configured;
    el.purityOverlapKeyStatus.textContent = configured ? '(server .env configured)' : '(optional)';
    el.purityOverlapKeyStatus.className = configured ? 'text-success' : 'text-muted';
  }
  syncPurityPromptUi();
  refreshPurityVerifierStatus();
}

function syncPurityPromptUi() {
  if (state.purity.overlap.backend === 'vibevoice') return;
  const backendName = purityOverlapBackendLabel();
  const prompt = (el.purityOverlapPrompt?.value ?? state.purity.overlap.prompt ?? '').trim();
  const defaultPrompt = (
    state.purity.serverConfig?.overlap_prompt
    || 'Does this audio contain overlapping speech from two or more speakers at the same time?'
  ).trim();
  if (el.purityOverlapPromptLabel) el.purityOverlapPromptLabel.textContent = `${backendName} prompt`;
  if (el.purityOverlapPromptStatus) {
    el.purityOverlapPromptStatus.textContent = !prompt ? 'Required' : (prompt === defaultPrompt ? 'Server default' : 'Custom prompt');
    el.purityOverlapPromptStatus.className = `badge badge-sm ${!prompt ? 'badge-danger' : (prompt === defaultPrompt ? 'badge-ghost' : 'badge-info')}`;
  }
}

async function loadSpeakerPurityConfig() {
  if (state.purity.serverConfig) return;
  try {
    const config = await parseJsonResponse(await fetch('/api/purity/config'));
    state.purity.serverConfig = config;
    const backend = config.overlap_backend || 'gemma4';
    const defaults = config[backend] || {};
    const vibevoiceDefaults = config.vibevoice || {};
    state.purity.overlap = {
      enabled: true,
      backend,
      model: defaults.model || '',
      endpoint: config.gemma4?.endpoint || '',
      timeout: config.overlap_timeout_s || 120,
      maxOutputTokens: backend === 'vibevoice'
        ? (vibevoiceDefaults.max_new_tokens || 2048)
        : (config.overlap_max_output_tokens || 128),
      prompt: config.overlap_prompt || state.purity.overlap.prompt,
      failurePolicy: 'fail_closed',
      minSecondarySpeech: vibevoiceDefaults.min_secondary_speech_s ?? 0.25,
      batchSize: vibevoiceDefaults.batch_size ?? 1,
    };
    try {
      const saved = JSON.parse(localStorage.getItem('sonic_purity_preferences') || 'null');
      if (saved?.settings) state.purity.settings = { ...state.purity.settings, ...saved.settings };
      if (saved?.overlap) state.purity.overlap = { ...state.purity.overlap, ...saved.overlap };
    } catch (_) {}
    // The staged embedding→verifier flow is gone; map its retired policy value.
    if (state.purity.overlap.failurePolicy === 'keep_embedding_decision') {
      state.purity.overlap.failurePolicy = 'fail_open';
    }
    state.purity.overlap.enabled = true;
    if (state.purity.overlap.backend === 'vibevoice' && state.purity.overlap.maxOutputTokens === 128) {
      state.purity.overlap.maxOutputTokens = vibevoiceDefaults.max_new_tokens || 2048;
    }
    applyPurityControls();
  } catch (err) {
    console.warn('Speaker purity defaults unavailable:', err);
    applyPurityControls();
  }
}

function restorePurityOverlapDefaults() {
  const config = state.purity.serverConfig || {};
  const backend = config.overlap_backend || 'gemma4';
  const defaults = config[backend] || {};
  const vibevoiceDefaults = config.vibevoice || {};
  state.purity.overlap = {
    enabled: true,
    backend,
    model: defaults.model || '',
    endpoint: config.gemma4?.endpoint || '',
    timeout: config.overlap_timeout_s || 120,
    maxOutputTokens: backend === 'vibevoice'
      ? (vibevoiceDefaults.max_new_tokens || 2048)
      : (config.overlap_max_output_tokens || 128),
    prompt: config.overlap_prompt || 'Does this audio contain overlapping speech from two or more speakers at the same time?',
    failurePolicy: 'fail_closed',
    minSecondarySpeech: vibevoiceDefaults.min_secondary_speech_s ?? 0.25,
    batchSize: vibevoiceDefaults.batch_size ?? 1,
  };
  if (el.purityOverlapApiKey) el.purityOverlapApiKey.value = '';
  applyPurityControls();
  savePurityPreferences();
  showToast('Restored overlap verifier defaults', 'info');
}

function initPurityTab() {
  loadDiarizationResultsForVerification();
  el.btnPurityRefreshResults?.addEventListener('click', async () => {
    await loadDiarizationResultsForVerification();
    showToast('Diarization results refreshed', 'info');
  });
  loadSpeakerPurityConfig();

  el.purityOverlapBackend?.addEventListener('change', e => {
    const previousBackend = state.purity.overlap.backend;
    state.purity.overlap.backend = e.target.value;
    const defaults = purityBackendDefaults(e.target.value);
    state.purity.overlap.model = defaults.model || '';
    if (e.target.value === 'gemma4') {
      state.purity.overlap.endpoint = defaults.endpoint || state.purity.serverConfig?.gemma4?.endpoint || '';
    }
    if (e.target.value === 'vibevoice') {
      state.purity.overlap.model = resolveVibevoiceModelId(defaults.model);
      if (previousBackend !== 'vibevoice' && state.purity.overlap.maxOutputTokens === 128) {
        state.purity.overlap.maxOutputTokens = defaults.max_new_tokens || 2048;
      }
      state.purity.overlap.minSecondarySpeech = defaults.min_secondary_speech_s ?? 0.25;
    } else if (previousBackend === 'vibevoice' && state.purity.overlap.maxOutputTokens === 2048) {
      state.purity.overlap.maxOutputTokens = state.purity.serverConfig?.overlap_max_output_tokens || 128;
    }
    applyPurityControls();
    savePurityPreferences();
  });

  el.purityOverlapModel?.addEventListener('change', e => {
    state.purity.overlap.model = e.target.value.trim();
    savePurityPreferences();
  });
  el.purityOverlapVibevoiceModel?.addEventListener('change', e => {
    state.purity.overlap.model = e.target.value;
    savePurityPreferences();
    refreshPurityVerifierStatus();
  });
  el.purityOverlapEndpoint?.addEventListener('change', e => {
    state.purity.overlap.endpoint = e.target.value.trim();
    savePurityPreferences();
    refreshPurityVerifierStatus();
  });
  el.purityOverlapTimeout?.addEventListener('change', e => {
    state.purity.overlap.timeout = Math.max(1, parseFloat(e.target.value) || 120);
    e.target.value = state.purity.overlap.timeout;
    savePurityPreferences();
  });
  el.purityOverlapMaxTokens?.addEventListener('change', e => {
    state.purity.overlap.maxOutputTokens = Math.max(1, parseInt(e.target.value, 10) || 128);
    e.target.value = state.purity.overlap.maxOutputTokens;
    savePurityPreferences();
  });
  el.purityVibevoiceBatchSize?.addEventListener('change', e => {
    state.purity.overlap.batchSize = Math.max(1, parseInt(e.target.value, 10) || 1);
    e.target.value = state.purity.overlap.batchSize;
    savePurityPreferences();
  });
  el.purityOverlapFailurePolicy?.addEventListener('change', e => {
    state.purity.overlap.failurePolicy = e.target.value;
    savePurityPreferences();
  });
  el.purityOverlapPrompt?.addEventListener('input', e => {
    state.purity.overlap.prompt = e.target.value;
    syncPurityPromptUi();
    savePurityPreferences();
  });
  el.purityVibevoiceSecondary?.addEventListener('change', e => {
    state.purity.overlap.minSecondarySpeech = Math.max(0, parseFloat(e.target.value) || 0.25);
    e.target.value = state.purity.overlap.minSecondarySpeech;
    savePurityPreferences();
  });
  el.btnResetPurityOverlap?.addEventListener('click', restorePurityOverlapDefaults);
  el.btnPurityCheckVerifier?.addEventListener('click', async () => {
    const status = await refreshPurityVerifierStatus();
    if (status?.ready) showToast(status.message || 'Verifier is ready', 'success');
    else showToast(status?.message || 'Verifier is not ready', 'error');
  });
  el.btnTogglePurityOverlapKey?.addEventListener('click', () => {
    if (!el.purityOverlapApiKey) return;
    const reveal = el.purityOverlapApiKey.type === 'password';
    el.purityOverlapApiKey.type = reveal ? 'text' : 'password';
    el.btnTogglePurityOverlapKey.textContent = reveal ? 'Hide' : 'Show';
  });

  if (el.purityInputSelect) {
    el.purityInputSelect.addEventListener('change', async () => {
      const audioId = el.purityInputSelect.value;
      if (audioId && audioId.startsWith('lib:')) {
        await loadLibraryFileTo(audioId.slice(4), 'purity');
        return;
      }
      state.purity.audioId = audioId || null;
      updatePurityInputMeta(audioId);
      syncPurityDiarizationStatus();
    });
  }

  if (el.btnPurityBrowseLibrary) {
    el.btnPurityBrowseLibrary.addEventListener('click', () => openLibraryModal('purity'));
  }

  if (el.btnPurityPreviewInput) {
    el.btnPurityPreviewInput.addEventListener('click', () => {
      const audioId = el.purityInputSelect?.value;
      if (audioId) playAudioItem(audioId);
    });
  }

  if (el.btnPurityRefreshProfiles) {
    el.btnPurityRefreshProfiles.addEventListener('click', async () => {
      await loadSpeakerProfiles();
      syncPurityProfileSelect();
      showToast('Speaker profiles refreshed', 'info');
    });
  }

  if (el.btnTogglePurityHfVis && el.purityHfTokenInput) {
    el.btnTogglePurityHfVis.addEventListener('click', () => {
      const isPass = el.purityHfTokenInput.type === 'password';
      el.purityHfTokenInput.type = isPass ? 'text' : 'password';
      el.btnTogglePurityHfVis.textContent = isPass ? 'Hide' : 'Show';
    });
  }

  if (el.btnRunPurity) {
    el.btnRunPurity.addEventListener('click', () => runSpeakerPurityVerification(false));
  }
  el.btnRunPurityManual?.addEventListener('click', () => runSpeakerPurityVerification(true));

  if (el.btnPurityReset) {
    el.btnPurityReset.addEventListener('click', resetPurityTab);
  }

  document.querySelectorAll('.purity-tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.purity-tab-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.purity.filterStatus = btn.dataset.filter || 'all';
      renderPurityResults();
    });
  });

  const speakerFilter = document.getElementById('purity-speaker-filter');
  if (speakerFilter) {
    speakerFilter.addEventListener('change', e => {
      state.purity.filterSpeaker = e.target.value;
      renderPurityResults();
    });
  }

  const sortSelect = document.getElementById('purity-sort-select');
  if (sortSelect) {
    sortSelect.addEventListener('change', e => {
      state.purity.sortMode = e.target.value;
      renderPurityResults();
    });
  }

  document.getElementById('btn-purity-export-audio')?.addEventListener('click', () => exportPurityAudio('concat'));
  document.getElementById('btn-purity-download-json')?.addEventListener('click', downloadPurityReportJSON);
  document.getElementById('btn-purity-download-csv')?.addEventListener('click', downloadPurityReportCSV);
  document.getElementById('btn-purity-save-eval')?.addEventListener('click', savePurityEvaluation);
}

function syncPurityProfileSelect() {
  if (!el.purityProfileSelect) return;
  const profiles = state.knownSpeakers?.profiles || [];
  const current = el.purityProfileSelect.value;
  el.purityProfileSelect.innerHTML = '<option value="">-- No label --</option>';
  profiles.forEach(p => {
    const opt = document.createElement('option');
    opt.value = p.name;
    opt.textContent = `${p.name} (${p.num_clips} clip${p.num_clips === 1 ? '' : 's'})`;
    el.purityProfileSelect.appendChild(opt);
  });
  if (current && profiles.some(p => p.name === current)) {
    el.purityProfileSelect.value = current;
  }
}

function updatePurityInputMeta(audioId) {
  if (!audioId) {
    if (el.purityAudioMetaChip) el.purityAudioMetaChip.textContent = 'No track selected';
    if (el.purityInputPreviewPill) el.purityInputPreviewPill.classList.add('hidden');
    return;
  }
  const item = state.audioList.find(a => a.id === audioId);
  if (!item) {
    if (el.purityAudioMetaChip) el.purityAudioMetaChip.textContent = audioId.startsWith('lib:') ? 'Library track' : 'Session track';
    if (el.purityInputPreviewPill) el.purityInputPreviewPill.classList.add('hidden');
    return;
  }
  const sr = item.sample_rate ? `${(item.sample_rate / 1000).toFixed(1)}kHz` : '44.1kHz';
  const ch = item.channels === 1 ? 'Mono' : item.channels === 2 ? 'Stereo' : `${item.channels || 1}ch`;
  const dur = formatTime(item.duration_s || 0);
  if (el.purityAudioMetaChip) el.purityAudioMetaChip.textContent = `${sr} • ${ch} • ${dur}`;
  if (el.purityTrackTitleText) el.purityTrackTitleText.textContent = item.title || item.id;
  if (el.purityTrackSpecChip) el.purityTrackSpecChip.textContent = `${sr} • ${ch} • ${dur}`;
  if (el.purityInputPreviewPill) el.purityInputPreviewPill.classList.remove('hidden');
}

function syncPurityDiarizationStatus() {
  const audioId = el.purityInputSelect?.value;
  if (!el.purityDiarTurnsChip) return;
  if (!audioId) {
    el.purityDiarTurnsChip.textContent = 'No track selected';
    el.purityDiarTurnsChip.className = 'badge badge-sm badge-ghost';
    if (el.purityDiarDesc) {
      el.purityDiarDesc.textContent = 'Pick any session or library track. Diarization turns are optional.';
    }
    return;
  }
  const turns = purityTurnsForSelectedAudio(audioId);
  if (turns.length > 0) {
    const dur = turns.reduce((sum, turn) => sum + (turn.end_s - turn.start_s), 0);
    el.purityDiarTurnsChip.textContent = `${turns.length} turns ready (${dur.toFixed(1)}s)`;
    el.purityDiarTurnsChip.className = 'badge badge-sm badge-success';
    if (el.purityDiarDesc) {
      el.purityDiarDesc.textContent = `Using ${turns.length} diarized speaker turns from the active timeline.`;
    }
    return;
  }
  const item = state.audioList.find(audio => audio.id === audioId);
  const dur = Number(item?.duration_s) || 0;
  el.purityDiarTurnsChip.textContent = dur
    ? `Whole file · 1 candidate (${dur.toFixed(1)}s)`
    : 'Whole file · 1 candidate';
  el.purityDiarTurnsChip.className = 'badge badge-sm badge-success';
  if (el.purityDiarDesc) {
    el.purityDiarDesc.textContent = 'No diarization turns in memory for this track. The verifier will listen to the whole file as one candidate.';
  }
}

async function runSpeakerPurityVerification(forceManual = false) {
  if (typeof forceManual !== 'boolean') forceManual = false;
  if (!forceManual && state.purity.selectedResultIds.size > 0) {
    await runDiarizationResultBatchVerification();
    return;
  }
  const audioId = el.purityInputSelect?.value;
  const profileName = el.purityProfileSelect?.value || 'unlabeled';

  if (!audioId) {
    showToast('Select at least one diarization result, or choose a session/library track to verify.', 'error');
    return;
  }
  if (audioId.startsWith('lib:')) {
    showToast('Wait for the library track to finish loading, then try again.', 'error');
    return;
  }

  const turns = purityTurnsForSelectedAudio(audioId);

  const minDuration = Math.max(0.01, parseFloat(el.purityCandidateMinDuration?.value) || state.purity.settings.minDuration || 1.5);
  const device = el.purityDeviceSelect?.value || 'auto';
  const token = el.purityHfTokenInput?.value || localStorage.getItem('sonic_hf_token') || undefined;
  const overlap = state.purity.overlap;
  overlap.enabled = true;
  const overlapPrompt = el.purityOverlapPrompt?.value.trim() || '';
  if (overlap.backend !== 'vibevoice' && !overlapPrompt) {
    showToast('The LLM verifier prompt cannot be empty', 'error');
    el.purityOverlapPrompt?.focus();
    return;
  }
  const status = await refreshPurityVerifierStatus();
  if (status && status.ready === false) {
    showToast(status.message || `${purityOverlapBackendLabel()} is not ready`, 'error');
    return;
  }
  const overlapVerifier = purityOverlapVerifierPayload();
  savePurityPreferences();

  state.purity.audioId = audioId;
  state.purity.profileName = profileName;

  if (el.btnRunPurity) el.btnRunPurity.disabled = true;
  if (el.btnRunPurityManual) el.btnRunPurityManual.disabled = true;
  if (el.purityTaskProgressBox) el.purityTaskProgressBox.classList.remove('hidden');
  if (el.purityTaskStatusText) {
    el.purityTaskStatusText.textContent = turns.length
      ? `Checking ${turns.length} candidates with ${purityOverlapBackendLabel()}...`
      : `Checking whole file with ${purityOverlapBackendLabel()}...`;
  }

  let timerInterval = null;
  const startTime = Date.now();
  if (el.purityTaskTimer) {
    el.purityTaskTimer.textContent = '0.0s';
    timerInterval = setInterval(() => {
      const elapsed = (Date.now() - startTime) / 1000;
      el.purityTaskTimer.textContent = `${elapsed.toFixed(1)}s`;
    }, 100);
  }

  try {
    const data = await parseJsonResponse(await fetch('/api/purity/verify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        audio_id: audioId,
        profile: profileName,
        turns: turns.map(t => ({
          speaker_id: t.speaker_id,
          start_s: t.start_s,
          end_s: t.end_s,
        })),
        min_candidate_duration_s: minDuration,
        device: device,
        token: token,
        overlap_verifier: overlapVerifier,
      }),
    }));

    const result = await new Promise((resolve, reject) => pollTask(data.task_id, resolve, reject));
    if (timerInterval) clearInterval(timerInterval);

    state.purity.results = result.purity_results || [];
    state.purity.metrics = result.metrics || {};
    state.purity.audioId = result.audio_id;
    state.purity.profileName = result.profile;
    state.purity.runSettings = result.settings || null;
    state.purity.verifierStatus = result.verifier_status || state.purity.verifierStatus;

    if (el.purityTaskProgressBox) el.purityTaskProgressBox.classList.add('hidden');
    if (el.purityEmptyPlaceholder) el.purityEmptyPlaceholder.classList.add('hidden');
    if (el.purityResultsWrapper) el.purityResultsWrapper.classList.remove('hidden');

    renderPurityResults();
    const errorCount = result.verifier_status?.error_count || result.metrics?.direct_overlap_errors || 0;
    if (errorCount) {
      showToast(`Purity finished with ${errorCount} verifier error(s). See the Errors tab.`, 'warning');
    } else {
      showToast(`Speaker purity verification complete: ${result.metrics?.passed_candidates || 0}/${result.purity_results?.length || 0} passed via ${purityOverlapBackendLabel()}`, 'success');
    }
  } catch (err) {
    if (timerInterval) clearInterval(timerInterval);
    if (el.purityTaskProgressBox) el.purityTaskProgressBox.classList.add('hidden');
    const msg = err instanceof Error ? err.message : String(err);
    showToast(`Purity verification failed: ${msg}`, 'error');
    await refreshPurityVerifierStatus();
  } finally {
    if (el.btnRunPurity) el.btnRunPurity.disabled = false;
    if (el.btnRunPurityManual) el.btnRunPurityManual.disabled = false;
  }
}

async function runDiarizationResultBatchVerification() {
  const resultIds = [...state.purity.selectedResultIds];
  const profileName = el.purityProfileSelect?.value || 'unlabeled';
  const minDuration = Math.max(0.01, parseFloat(el.purityCandidateMinDuration?.value) || 1.5);
  const maxDurationRaw = el.purityCandidateMaxDuration?.value;
  const device = el.purityDeviceSelect?.value || 'auto';
  const token = el.purityHfTokenInput?.value || localStorage.getItem('sonic_hf_token') || undefined;
  const overlap = state.purity.overlap;
  overlap.enabled = true;
  const overlapPrompt = el.purityOverlapPrompt?.value.trim() || '';
  if (overlap.backend !== 'vibevoice' && !overlapPrompt) {
    showToast('The LLM verifier prompt cannot be empty', 'error');
    el.purityOverlapPrompt?.focus();
    return;
  }
  const status = await refreshPurityVerifierStatus();
  if (status && status.ready === false) {
    showToast(status.message || `${purityOverlapBackendLabel()} is not ready`, 'error');
    return;
  }
  const overlapVerifier = purityOverlapVerifierPayload();
  savePurityPreferences();
  if (el.btnRunPurity) el.btnRunPurity.disabled = true;
  if (el.btnRunPurityManual) el.btnRunPurityManual.disabled = true;
  if (el.purityTaskProgressBox) el.purityTaskProgressBox.classList.remove('hidden');
  if (el.purityTaskStatusText) {
    el.purityTaskStatusText.textContent = `Checking eligible turns from ${resultIds.length} result(s) with ${purityOverlapBackendLabel()}...`;
  }
  try {
    const payload = await parseJsonResponse(await fetch('/api/diarization/results/verify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        result_ids: resultIds,
        profile: profileName,
        speaker_ids: el.purityCandidateSpeaker?.value ? [el.purityCandidateSpeaker.value] : [],
        min_duration_s: minDuration,
        max_duration_s: maxDurationRaw ? parseFloat(maxDurationRaw) : null,
        overlap_state: el.purityCandidateOverlap?.value || 'any',
        verification_state: el.purityCandidateVerification?.value || 'all',
        device,
        token,
        overlap_verifier: overlapVerifier,
      }),
    }));
    const report = await new Promise((resolve, reject) => pollTask(payload.task_id, resolve, reject));
    state.purity.results = report.results || [];
    state.purity.metrics = {
      total_candidates: report.results?.length || 0,
      passed_candidates: report.counts?.pass || 0,
      rejected_candidates: report.counts?.reject || 0,
      uncertain_candidates: report.counts?.uncertain || 0,
      error_candidates: report.counts?.error || 0,
      direct_overlap_checked: (report.results || []).filter(row => row.direct_overlap).length,
      direct_overlap_errors: (report.results || []).filter(row => row.direct_overlap?.error).length,
      vibevoice_checked: (report.results || []).filter(row => row.vibevoice).length,
    };
    state.purity.profileName = report.profile;
    state.purity.audioId = null;
    state.purity.runSettings = report.settings || null;
    state.purity.verifierStatus = report.verifier_status || state.purity.verifierStatus;
    if (el.purityEmptyPlaceholder) el.purityEmptyPlaceholder.classList.add('hidden');
    if (el.purityResultsWrapper) el.purityResultsWrapper.classList.remove('hidden');
    renderPurityResults();
    await loadDiarizationResultsForVerification();
    const errorCount = report.verifier_status?.error_count || state.purity.metrics.direct_overlap_errors || 0;
    if (errorCount) {
      showToast(`Batch finished with ${errorCount} verifier error(s). See the Errors tab.`, 'warning');
    } else {
      showToast(`Batch complete: ${report.counts?.pass || 0} passed, ${report.counts?.reject || 0} rejected, ${report.counts?.error || 0} errors via ${purityOverlapBackendLabel()}`, 'success');
    }
  } catch (err) {
    showToast(`Batch verification failed: ${err.message || String(err)}`, 'error');
    await refreshPurityVerifierStatus();
  } finally {
    if (el.btnRunPurity) el.btnRunPurity.disabled = false;
    if (el.btnRunPurityManual) el.btnRunPurityManual.disabled = false;
    if (el.purityTaskProgressBox) el.purityTaskProgressBox.classList.add('hidden');
  }
}

function purityTurnKey(turn) {
  if (turn.result_id && Number.isInteger(turn.turn_index)) {
    return `${turn.result_id}:${turn.turn_index}`;
  }
  return `${Number(turn.start_s).toFixed(3)}-${Number(turn.end_s).toFixed(3)}-${turn.speaker_id}`;
}

function stopPuritySegmentPreview() {
  if (puritySegmentPlayRaf) {
    cancelAnimationFrame(puritySegmentPlayRaf);
    puritySegmentPlayRaf = 0;
  }
  puritySegmentAudio.pause();
  activePuritySegmentKey = null;
  updatePuritySegmentPlaybackButtons();
}

function updatePuritySegmentPlaybackButtons() {
  document.querySelectorAll('.purity-play-segment').forEach(btn => {
    const isActive = btn.dataset.key === activePuritySegmentKey;
    btn.innerHTML = `${isActive ? '■' : '▶'} ${escapeHtml(btn.dataset.range || '')}`;
    btn.setAttribute('aria-pressed', String(isActive));
  });
}

function watchPuritySegmentEnd(endSec, key) {
  const tick = () => {
    puritySegmentPlayRaf = 0;
    if (activePuritySegmentKey !== key || puritySegmentAudio.paused) return;
    if (puritySegmentAudio.currentTime >= endSec - 0.01) {
      stopPuritySegmentPreview();
      return;
    }
    puritySegmentPlayRaf = requestAnimationFrame(tick);
  };
  puritySegmentPlayRaf = requestAnimationFrame(tick);
}

function togglePuritySegmentPreview(turn) {
  if (turn.result_id && Number.isInteger(turn.turn_index)) {
    const key = `${turn.result_id}:${turn.turn_index}`;
    if (activePuritySegmentKey === key) {
      stopPuritySegmentPreview();
      return;
    }
    stopPuritySegmentPreview();
    activePuritySegmentKey = key;
    puritySegmentAudio.src = `/api/diarization/results/${encodeURIComponent(turn.result_id)}/turns/${turn.turn_index}/audio`;
    puritySegmentAudio.currentTime = 0;
    puritySegmentAudio.play().catch(err => {
      console.error('Segment preview error:', err);
      stopPuritySegmentPreview();
      showToast('Unable to play this segment', 'error');
    });
    updatePuritySegmentPlaybackButtons();
    return;
  }
  const audioId = state.purity.audioId || state.diarization.audioId;
  if (!audioId) return;

  const key = purityTurnKey(turn);
  if (activePuritySegmentKey === key) {
    stopPuritySegmentPreview();
    return;
  }

  stopPuritySegmentPreview();
  if (el.audio && !el.audio.paused) el.audio.pause();

  activePuritySegmentKey = key;
  puritySegmentAudio.volume = state.player.volume;
  puritySegmentAudio.playbackRate = state.player.playbackRate;
  updatePuritySegmentPlaybackButtons();

  const beginPlayback = () => {
    if (activePuritySegmentKey !== key) return;
    puritySegmentAudio.currentTime = turn.start_s;
    puritySegmentAudio.play()
      .then(() => watchPuritySegmentEnd(turn.end_s, key))
      .catch(err => {
        console.error('Segment preview error:', err);
        stopPuritySegmentPreview();
        showToast('Unable to play this segment', 'error');
      });
  };

  const streamUrl = `/api/audio/${audioId}/stream`;
  if (!puritySegmentAudio.src.endsWith(streamUrl)) {
    puritySegmentAudio.src = streamUrl;
    puritySegmentAudio.addEventListener('loadedmetadata', beginPlayback, { once: true });
    puritySegmentAudio.load();
  } else if (puritySegmentAudio.readyState >= 1) {
    beginPlayback();
  } else {
    puritySegmentAudio.addEventListener('loadedmetadata', beginPlayback, { once: true });
  }
}

function formatPurityReason(reason) {
  if (!reason) return '';
  switch (reason) {
    case 'candidate_too_short': return 'Candidate Too Short';
    case 'overlap_detected': return 'Overlap Detected';
    case 'direct_overlap_detected': return 'Overlap Detected';
    case 'direct_overlap_verification_failed': return 'Overlap Verifier Failed';
    case 'single_speaker': return 'Single Speaker';
    case 'multiple_speakers': return 'Multiple Speakers';
    case 'tiny_secondary_speaker': return 'Tiny Secondary Speaker';
    case 'empty_output': return 'Empty VibeVoice Output';
    case 'no_speaker_labels': return 'No Speaker Labels';
    case 'inference_error': return 'VibeVoice Inference Error';
    case 'vibevoice_verification_failed': return 'VibeVoice Verifier Failed';
    default: return reason.replace(/_/g, ' ');
  }
}

function renderPurityResults() {
  const results = state.purity.results;
  if (!results || results.length === 0) {
    if (el.purityResultsWrapper) el.purityResultsWrapper.classList.add('hidden');
    if (el.purityEmptyPlaceholder) el.purityEmptyPlaceholder.classList.remove('hidden');
    return;
  }

  const runSettings = state.purity.runSettings || {};
  const runMaxOverlap = 0.05;
  const passed = results.filter(r => r.decision === 'pass');
  const rejected = results.filter(r => r.decision === 'reject');
  const uncertain = results.filter(r => r.decision === 'uncertain');
  const errors = results.filter(r => r.decision === 'error' || r.direct_overlap?.error);

  const totalDur = results.reduce((sum, r) => sum + (r.duration_s || (r.end_s - r.start_s)), 0);
  const passDur = passed.reduce((sum, r) => sum + (r.duration_s || (r.end_s - r.start_s)), 0);
  const passPct = results.length ? (passed.length / results.length * 100) : 0;
  const durPct = totalDur > 0 ? (passDur / totalDur * 100) : 0;
  const verifierBackend = runSettings.overlap_verifier?.backend || state.purity.overlap.backend;
  const checked = verifierBackend === 'vibevoice'
    ? (state.purity.metrics?.vibevoice_checked || results.filter(r => r.vibevoice).length)
    : (state.purity.metrics?.direct_overlap_checked || results.filter(r => r.direct_overlap).length);
  const overlapDetected = state.purity.metrics?.direct_overlap_detected
    ?? results.filter(r => r.direct_overlap?.overlap === true).length;
  const requestErrors = state.purity.metrics?.direct_overlap_errors
    ?? results.filter(r => r.direct_overlap?.error).length;
  const errorSamples = state.purity.verifierStatus?.error_samples
    || results.map(r => r.error || r.direct_overlap?.error).filter(Boolean);

  if (el.purityProfileBadge) {
    el.purityProfileBadge.textContent = verifierBackend === 'vibevoice'
      ? 'VibeVoice-ASR'
      : (verifierBackend === 'gemini' ? 'Gemini' : 'Gemma 4 / Unsloth');
  }
  if (el.purityResultsTitle) {
    const label = state.purity.profileName && state.purity.profileName !== 'unlabeled'
      ? state.purity.profileName
      : 'LLM verifier';
    el.purityResultsTitle.textContent = `Purity Verification — ${label}`;
  }
  if (el.purityResultsMeta) {
    const errorNote = requestErrors ? ` • ${requestErrors} request error(s)` : '';
    const uncertainNote = uncertain.length ? ` • ${uncertain.length} uncertain` : '';
    el.purityResultsMeta.textContent = `${results.length} candidates evaluated • ${checked} checked by ${verifierBackend}${errorNote}${uncertainNote}`;
  }

  if (el.purityErrorBanner) {
    if (requestErrors) {
      const sample = errorSamples[0] || 'The LLM verifier returned request errors.';
      el.purityErrorBanner.classList.remove('hidden');
      el.purityErrorBanner.innerHTML = `<strong>${requestErrors} Unsloth/LLM request error(s).</strong> ${escapeHtml(sample)} Open the Errors tab for every failed candidate.`;
    } else {
      el.purityErrorBanner.classList.add('hidden');
      el.purityErrorBanner.textContent = '';
    }
  }

  if (el.purityMetricPassCount) el.purityMetricPassCount.textContent = `${passed.length} / ${results.length}`;
  if (el.purityMetricPassPct) el.purityMetricPassPct.textContent = `${passPct.toFixed(1)}% candidate pass rate`;
  if (el.purityMetricPassDuration) el.purityMetricPassDuration.textContent = `${passDur.toFixed(1)}s`;
  if (el.purityMetricTotalDuration) el.purityMetricTotalDuration.textContent = `of ${totalDur.toFixed(1)}s total candidate speech (${durPct.toFixed(1)}%)`;
  if (el.purityMetricLlmChecked) el.purityMetricLlmChecked.textContent = String(checked);
  if (el.purityMetricLlmDetail) {
    el.purityMetricLlmDetail.textContent = requestErrors
      ? `${overlapDetected} overlap • ${requestErrors} error(s)`
      : `${overlapDetected} overlap detected`;
  }

  // Rejection Breakdown Pills
  if (el.purityReasonsPills) {
    const reasonsMap = {};
    results.forEach(r => {
      if (r.reason) reasonsMap[r.reason] = (reasonsMap[r.reason] || 0) + 1;
    });
    if (Object.keys(reasonsMap).length === 0) {
      el.purityReasonsPills.innerHTML = '<span class="badge badge-xs badge-success">Zero rejections</span>';
    } else {
      el.purityReasonsPills.innerHTML = Object.entries(reasonsMap).map(([reason, count]) => {
        const label = formatPurityReason(reason);
        const isUncertain = ['tiny_secondary_speaker', 'empty_output', 'no_speaker_labels', 'inference_error'].includes(reason);
        const badgeClass = isUncertain ? 'badge-warning' : 'badge-danger';
        return `<span class="badge badge-xs ${badgeClass} font-mono">${label}: ${count}</span>`;
      }).join('');
    }
  }

  // Filter counters
  if (el.purityCountAll) el.purityCountAll.textContent = results.length;
  if (el.purityCountPass) el.purityCountPass.textContent = passed.length;
  if (el.purityCountReject) el.purityCountReject.textContent = rejected.length;
  if (el.purityCountUncertain) el.purityCountUncertain.textContent = uncertain.length;
  if (el.purityCountError) el.purityCountError.textContent = errors.length;
  if (el.purityFooterSelectionInfo) el.purityFooterSelectionInfo.textContent = `${passed.length} passed pure candidates (${passDur.toFixed(1)}s) ready for export`;

  // Speaker filter options update
  const speakerFilter = document.getElementById('purity-speaker-filter');
  if (speakerFilter) {
    const currentSpk = speakerFilter.value;
    const uniqueSpks = [...new Set(results.map(r => r.speaker_id))].sort();
    speakerFilter.innerHTML = '<option value="all">All Diarized Speakers</option>' +
      uniqueSpks.map(spk => `<option value="${escapeHtml(spk)}">${escapeHtml(getSpeakerName(spk))} (${spk})</option>`).join('');
    if (currentSpk && (currentSpk === 'all' || uniqueSpks.includes(currentSpk))) {
      speakerFilter.value = currentSpk;
    }
  }

  // Filter and sort items
  let filtered = [...results];
  if (state.purity.filterStatus === 'pass') filtered = filtered.filter(r => r.decision === 'pass');
  else if (state.purity.filterStatus === 'reject') filtered = filtered.filter(r => r.decision === 'reject');
  else if (state.purity.filterStatus === 'uncertain') filtered = filtered.filter(r => r.decision === 'uncertain');
  else if (state.purity.filterStatus === 'error') {
    filtered = filtered.filter(r => r.decision === 'error' || r.direct_overlap?.error);
  }

  if (state.purity.filterSpeaker && state.purity.filterSpeaker !== 'all') {
    filtered = filtered.filter(r => r.speaker_id === state.purity.filterSpeaker);
  }

  switch (state.purity.sortMode) {
    case 'time-desc': filtered.sort((a, b) => b.start_s - a.start_s); break;
    case 'dur-desc': filtered.sort((a, b) => (b.end_s - b.start_s) - (a.end_s - a.start_s)); break;
    case 'overlap-desc': filtered.sort((a, b) => b.overlap_duration_s - a.overlap_duration_s); break;
    default: filtered.sort((a, b) => a.start_s - b.start_s); break;
  }

  if (!el.purityTableBody) return;
  if (filtered.length === 0) {
    el.purityTableBody.innerHTML = `<tr><td colspan="7" class="empty-table-msg">No candidates match current filter criteria.</td></tr>`;
    return;
  }

  el.purityTableBody.innerHTML = filtered.map((r, index) => {
    const key = purityTurnKey(r);
    const dur = r.duration_s || (r.end_s - r.start_s);
    const rangeLabel = `${r.start_s.toFixed(2)}s – ${r.end_s.toFixed(2)}s`;
    const isPlaying = activePuritySegmentKey === key;
    const spkName = getSpeakerName(r.speaker_id);
    const overlapStr = `${r.overlap_duration_s.toFixed(2)}s (${(r.overlap_ratio * 100).toFixed(0)}%)`;
    const hasOverlap = r.overlap_duration_s > runMaxOverlap;
    const direct = r.direct_overlap;

    let directOverlapHtml = '<span class="badge badge-xs badge-ghost">Not checked</span>';
    if (r.vibevoice) {
      const vv = r.vibevoice;
      const speakerLabel = vv.num_speakers === 1 ? '1 speaker' : `${vv.num_speakers} speakers`;
      const secondary = vv.secondary_speech_s != null
        ? `${Number(vv.secondary_speech_s).toFixed(2)}s secondary`
        : '';
      if (r.decision === 'pass') {
        directOverlapHtml = `<span class="badge badge-xs badge-success">${escapeHtml(speakerLabel)}</span>`;
      } else if (r.decision === 'reject') {
        directOverlapHtml = `<span class="badge badge-xs badge-danger">${escapeHtml(speakerLabel)}</span><small class="purity-direct-reason" title="${escapeHtml(secondary)}">${escapeHtml(secondary)}</small>`;
      } else {
        const detail = formatPurityReason(vv.reason || r.reason || '');
        directOverlapHtml = `<span class="badge badge-xs badge-warning">${escapeHtml(speakerLabel)}</span><small class="purity-direct-reason" title="${escapeHtml(detail)}">${escapeHtml(detail)}</small>`;
      }
    } else if (direct?.error) {
      directOverlapHtml = `<span class="badge badge-xs badge-warning">Request error</span><small class="purity-direct-reason" title="${escapeHtml(direct.error)}">${escapeHtml(direct.error)}</small>`;
    } else if (direct?.overlap === true) {
      directOverlapHtml = `<span class="badge badge-xs badge-danger">Overlap detected</span><small class="purity-direct-reason" title="${escapeHtml(direct.reason || '')}">${escapeHtml(direct.reason || '')}</small>`;
    } else if (direct?.overlap === false) {
      directOverlapHtml = `<span class="badge badge-xs badge-success">No overlap</span><small class="purity-direct-reason" title="${escapeHtml(direct.reason || '')}">${escapeHtml(direct.reason || '')}</small>`;
    }

    let decisionHtml = '';
    let rowClass = '';
    if (r.decision === 'pass') {
      decisionHtml = `<span class="badge badge-success font-mono font-bold">✓ PASS</span>`;
    } else if (r.decision === 'reject') {
      rowClass = 'purity-row-rejected';
      decisionHtml = `<span class="badge badge-danger font-mono font-bold">✕ REJECT</span> <small class="text-xs text-muted block" style="margin-top:2px;">${formatPurityReason(r.reason)}</small>`;
    } else if (r.decision === 'uncertain') {
      rowClass = 'purity-row-uncertain';
      decisionHtml = `<span class="badge badge-warning font-mono font-bold">? UNCERTAIN</span> <small class="text-xs text-muted block" style="margin-top:2px;">${formatPurityReason(r.reason)}</small>`;
    } else {
      rowClass = 'purity-row-error';
      decisionHtml = `<span class="badge badge-warning font-mono font-bold">⚠ ERROR</span> <small class="text-xs text-muted block" style="margin-top:2px;">${escapeHtml(r.error || r.reason || '')}</small>`;
    }

    return `
      <tr class="${rowClass}">
        <td>${index + 1}</td>
        <td>
          <button class="btn btn-xs btn-ghost purity-play-segment" data-key="${key}" data-range="${rangeLabel}">
            ${isPlaying ? '■' : '▶'} ${rangeLabel}
          </button>
        </td>
        <td>
          <strong>${escapeHtml(spkName)}</strong>
          <small class="target-local-id">${escapeHtml(r.speaker_id)}</small>
          ${r.source_title ? `<small class="target-local-id" title="${escapeHtml(r.source_title)}">${escapeHtml(r.source_title)}</small>` : ''}
        </td>
        <td class="font-mono">${dur.toFixed(2)}s</td>
        <td class="font-mono ${hasOverlap ? 'text-danger font-bold' : 'text-muted'}">${overlapStr}</td>
        <td>${directOverlapHtml}</td>
        <td>${decisionHtml}</td>
      </tr>
    `;
  }).join('');

  el.purityTableBody.querySelectorAll('.purity-play-segment').forEach(btn => {
    btn.addEventListener('click', () => {
      const turn = results.find(item => purityTurnKey(item) === btn.dataset.key);
      if (turn) togglePuritySegmentPreview(turn);
    });
  });
}

async function exportPurityAudio(mode = 'concat') {
  const results = state.purity.results;
  const audioId = state.purity.audioId;
  const profileName = state.purity.profileName || 'pure_speaker';

  if (!results || results.length === 0 || !audioId) {
    showToast('No purity results to export', 'error');
    return;
  }

  const passed = results.filter(r => r.decision === 'pass');
  if (passed.length === 0) {
    showToast('No candidates passed purity verification to export', 'warning');
    return;
  }

  try {
    const data = await parseJsonResponse(await fetch('/api/purity/export-audio', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        audio_id: audioId,
        profile_name: profileName,
        mode: mode,
        segments: passed.map(r => ({
          start_s: r.start_s,
          end_s: r.end_s,
        })),
        extraction_settings: readDiarizationExtractionSettings(),
        blocker_turns: extractionBlockerTurns(),
      }),
    }));

    await fetchAudioList();
    showToast(`Exported ${passed.length} pure segments (${data.duration_s?.toFixed(1)}s) to workspace audio!`, 'success');
  } catch (err) {
    showToast(`Export failed: ${err.message}`, 'error');
  }
}

function downloadPurityReportJSON() {
  const results = state.purity.results;
  if (!results || results.length === 0) {
    showToast('No purity results to download', 'error');
    return;
  }
  const payload = {
    schema_version: '1.0',
    export_type: 'speaker_purity_report',
    exported_at: new Date().toISOString(),
    audio_id: state.purity.audioId,
    profile_name: state.purity.profileName,
    settings: state.purity.runSettings || state.purity.settings,
    metrics: state.purity.metrics,
    candidates: results,
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `purity_${state.purity.audioId}_${state.purity.profileName}.json`;
  a.click();
  URL.revokeObjectURL(url);
  showToast('Downloaded Purity Report JSON', 'success');
}

function downloadPurityReportCSV() {
  const results = state.purity.results;
  if (!results || results.length === 0) {
    showToast('No purity results to download', 'error');
    return;
  }
  const headers = ['index', 'audio_id', 'profile_name', 'speaker_id', 'start_s', 'end_s', 'duration_s', 'decision', 'reason', 'diarization_overlap_duration_s', 'diarization_overlap_ratio', 'direct_overlap', 'direct_overlap_reason', 'direct_overlap_error', 'vibevoice_num_speakers', 'vibevoice_secondary_speech_s', 'error'];
  const rows = results.map((r, i) => [
    i + 1,
    r.audio_id,
    r.profile_name,
    r.speaker_id,
    r.start_s.toFixed(3),
    r.end_s.toFixed(3),
    (r.duration_s || (r.end_s - r.start_s)).toFixed(3),
    r.decision,
    r.reason || '',
    r.overlap_duration_s.toFixed(3),
    r.overlap_ratio.toFixed(4),
    r.direct_overlap?.overlap ?? '',
    r.direct_overlap?.reason ? `"${r.direct_overlap.reason.replace(/"/g, '""')}"` : '',
    r.direct_overlap?.error ? `"${r.direct_overlap.error.replace(/"/g, '""')}"` : '',
    r.vibevoice?.num_speakers ?? '',
    r.vibevoice?.secondary_speech_s != null ? Number(r.vibevoice.secondary_speech_s).toFixed(3) : '',
    r.error ? `"${r.error.replace(/"/g, '""')}"` : ''
  ]);
  const csvContent = [headers.join(','), ...rows.map(row => row.join(','))].join('\n');
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `purity_${state.purity.audioId}_${state.purity.profileName}.csv`;
  a.click();
  URL.revokeObjectURL(url);
  showToast('Downloaded Purity Report CSV', 'success');
}

async function savePurityEvaluation() {
  const results = state.purity.results;
  if (!results || results.length === 0) {
    showToast('Run purity verification first', 'error');
    return;
  }
  const audio = state.audioList.find(a => a.id === state.purity.audioId) || {};
  const passed = results.filter(r => r.decision === 'pass');
  const totalDur = results.reduce((sum, r) => sum + (r.duration_s || (r.end_s - r.start_s)), 0);
  const passDur = passed.reduce((sum, r) => sum + (r.duration_s || (r.end_s - r.start_s)), 0);
  const evalId = `purity-${state.purity.audioId}-${state.purity.profileName}`.replace(/[^A-Za-z0-9_.-]/g, '_');
  const used = state.purity.runSettings || {};

  try {
    const response = await fetch('/api/evaluations', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        id: evalId,
        evaluation_type: 'speaker_purity',
        clip_id: state.purity.audioId,
        clip_title: audio.title || state.purity.audioId,
        clip_path: audio.path || '',
        profile_name: state.purity.profileName,
        model_id: used.overlap_verifier?.model || used.overlap_verifier?.backend || 'llm-verifier',
        model_name: `${purityOverlapBackendLabel(used.overlap_verifier?.backend)} Direct Audio Verifier`,
        min_duration_s: used.min_candidate_duration_s ?? used.min_duration_s ?? state.purity.settings.minDuration,
        overlap_verifier: used.overlap_verifier || { enabled: true },
        qualified_segments: passed.length,
        total_segments: results.length,
        qualified_duration_s: passDur,
        total_duration_s: totalDur,
        qualified_percent: results.length ? (passed.length / results.length * 100) : 0,
        score_overall: results.length ? (passed.length / results.length * 5) : 0,
        tags: ['speaker_purity', `profile:${state.purity.profileName}`],
      }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Failed to save evaluation');
    showToast('Saved Speaker Purity evaluation to vault', 'success');
  } catch (err) {
    showToast(`Failed to save evaluation: ${err.message}`, 'error');
  }
}

function resetPurityTab() {
  state.purity.results = null;
  state.purity.metrics = null;
  state.purity.runSettings = null;
  stopPuritySegmentPreview();
  if (el.purityResultsWrapper) el.purityResultsWrapper.classList.add('hidden');
  if (el.purityEmptyPlaceholder) el.purityEmptyPlaceholder.classList.remove('hidden');
  showToast('Purity analysis reset', 'info');
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

      <button class="sbs-solo-btn ${idx === activeAuditionIndex ? 'active' : ''}" data-index="${idx}" title="Listen solo at current position">
        <span>${idx === activeAuditionIndex ? '🔊 Active / Auditioning' : '▶ Solo / Audition'}</span>
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
          tags: [],
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
    if (el.evalNotesInput) el.evalNotesInput.value = existing.notes || "";
    if (el.currentEvalScoreBadge) {
      el.currentEvalScoreBadge.textContent = `★ ${existing.score_overall.toFixed(1)} Saved`;
      el.currentEvalScoreBadge.style.color = "#4ade80";
    }
  } else {
    setStarRating(5.0);
    if (el.evalNotesInput) el.evalNotesInput.value = "";
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
    score_vocal_clarity: 5,
    score_bleed: 5,
    score_artifacts: 5,
    notes: el.evalNotesInput ? el.evalNotesInput.value.trim() : "",
    tags: [],
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

async function fetchEvaluations() {
  try {
    const res = await fetch("/api/evaluations");
    const data = await parseJsonResponse(res);
    state.evaluations = data.evaluations || [];
  } catch (err) {
    console.error("Failed to fetch evaluations:", err);
  }
}

// ==================== LIBRARY & HISTORY ====================

function getFileModelBadge(file) {
  const cat = (file.category || "").toLowerCase();
  const categoryId = fileCategoryId(file);
  const systemTags = new Set((file.system_tags || []).map(tag => String(tag).toLowerCase()));
  const customTags = new Set((file.custom_tags || []).map(tag => String(tag).toLowerCase()));
  const history = (file.history || []).join(' ').toLowerCase();

  if (systemTags.has('stage:verified') || categoryId === 'verified') {
    return { label: "Verified Speech", class: "badge-success" };
  }
  if (systemTags.has('type:cut') || categoryId === 'cuts') {
    return { label: "Audio Cut", class: "badge-warning" };
  }
  if (history.includes('bs_roformer')) {
    return { label: "BS-RoFormer", class: "badge-primary" };
  }
  if (history.includes('mel_roformer')) {
    return { label: "Mel-RoFormer", class: "badge-primary" };
  }
  if (history.includes('htdemucs')) {
    return { label: "HTDemucs", class: "badge-primary" };
  }
  if (history.includes('mvsep') || history.includes('mdx')) {
    return { label: "MVSep MDX", class: "badge-primary" };
  }
  if (systemTags.has('type:stem') || systemTags.has('stage:separated') || categoryId === 'stems') {
    return { label: "Separated Stem", class: "badge-primary" };
  }
  if (systemTags.has('stage:diarized') || categoryId === 'diarized') {
    return { label: "Diarized Source", class: "badge-info" };
  }
  if (categoryId === 'speech' || cat.includes("speech") || customTags.has('speech')) {
    return { label: "Speech Source", class: "badge-success" };
  }
  if (categoryId === 'music' || cat.includes("music") || customTags.has('music')) {
    return { label: "Music BGM", class: "badge-accent" };
  }
  if (systemTags.has('stage:ingested') || categoryId === 'ingest') {
    return { label: "Ingested Source", class: "badge-secondary" };
  }
  if (file.registry_item_id || categoryId === 'pipeline') {
    return { label: "Pipeline Asset", class: "badge-info" };
  }
  if (categoryId === 'temp' || cat.includes("temp")) {
    return { label: "Quick Save", class: "badge-ghost" };
  }
  if (categoryId === 'uploads' || cat.includes("upload")) {
    return { label: "Upload", class: "badge-secondary" };
  }
  return { label: (file.format || "WAV").toUpperCase(), class: "badge-ghost" };
}

let previewAudioEl = null;
let currentPreviewingPath = null;

const LIBRARY_CATEGORY_ORDER = ['speech', 'music', 'cuts', 'stems', 'verified', 'diarized', 'ingest', 'pipeline', 'uploads', 'temp', 'data', 'other'];
const LIBRARY_LOAD_TARGETS = {
  workspace: {
    title: 'Project Sample Library',
    subtitle: 'Browse project audio and load it into the Studio workspace',
    button: 'Load into Workspace',
  },
  cutter: {
    title: 'Open from Sample Library',
    subtitle: 'Choose a file to inspect and cut',
    button: 'Open in Cutter',
  },
  separation: {
    title: 'Load into Separation',
    subtitle: 'Choose a source track for vocal / stem separation',
    button: 'Send to Separation',
  },
  diarization: {
    title: 'Load into Diarization',
    subtitle: 'Choose a source track for speaker diarization',
    button: 'Send to Diarization',
  },
  annotation: {
    title: 'Load into Manual Annotation',
    subtitle: 'Choose the exact source track for ground-truth speaker annotation',
    button: 'Annotate this audio',
  },
  purity: {
    title: 'Load into Speaker Purity',
    subtitle: 'Choose a session or library track to verify as one clip, or with its diarization turns',
    button: 'Send to Speaker Purity',
  },
};
const PREVIEW_PLAY_ICON = '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>';
const PREVIEW_PAUSE_ICON = '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect></svg>';

function isModalOpen() {
  return [el.modalLibrary, el.modalSaveTo, el.modalTaskQueue].some(modal => modal && !modal.classList.contains('hidden'));
}

function fileCategoryId(file) {
  return file?.category_id || 'other';
}

function resetPreviewButtons() {
  document.querySelectorAll('.btn-preview-file').forEach(btn => {
    btn.classList.remove('playing');
    btn.innerHTML = PREVIEW_PLAY_ICON;
  });
}

function stopFilePreview() {
  if (previewAudioEl && !previewAudioEl.paused) {
    previewAudioEl.pause();
  }
  currentPreviewingPath = null;
  resetPreviewButtons();
}

function toggleFilePreview(filePath, btn) {
  if (!previewAudioEl) {
    previewAudioEl = new Audio();
    previewAudioEl.addEventListener('ended', stopFilePreview);
    previewAudioEl.addEventListener('pause', () => {
      if (previewAudioEl.ended || previewAudioEl.paused) resetPreviewButtons();
    });
  }

  if (currentPreviewingPath === filePath && !previewAudioEl.paused) {
    previewAudioEl.pause();
    currentPreviewingPath = null;
    if (btn) {
      btn.classList.remove('playing');
      btn.innerHTML = PREVIEW_PLAY_ICON;
    }
    return;
  }

  if (el.audio && !el.audio.paused) {
    el.audio.pause();
    setPlayingUI(false);
  }

  previewAudioEl.pause();
  currentPreviewingPath = filePath;
  previewAudioEl.src = `/api/library/stream?path=${encodeURIComponent(filePath)}`;
  previewAudioEl.play().catch(err => {
    console.warn("Audio preview failed:", err);
    showToast("Audio preview error", "error");
  });

  resetPreviewButtons();
  if (btn) {
    btn.classList.add('playing');
    btn.innerHTML = PREVIEW_PAUSE_ICON;
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
            <button class="menu-item-btn btn-session-purity" style="text-align: left; padding: 6px 10px; font-size: 0.78rem; background: none; border: none; color: var(--text-primary); cursor: pointer; border-radius: 4px; display: flex; align-items: center; gap: 6px;">🛡️ Send to Speaker Purity</button>
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
      switchTab('tab-workspace');
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
    card.querySelector('.btn-session-purity')?.addEventListener('click', () => {
      popupMenu.classList.add('hidden');
      switchTab('tab-purity');
      if (el.purityInputSelect) {
        el.purityInputSelect.value = item.id;
        el.purityInputSelect.dispatchEvent(new Event('change'));
      }
      showToast(`Selected "${item.title}" for Speaker Purity!`, "success");
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
    const selected = state.librarySelectedPaths;
    const livePaths = new Set(state.serverFiles.map(file => file.path));
    state.librarySelectedPaths = new Set([...selected].filter(path => livePaths.has(path)));
    populateLibraryMetadataFilters();
    renderServerFiles();
    renderLibraryModalItems();
    populateAllAudioSelects();
  } catch (err) {
    console.error("Failed to fetch library:", err);
    showToast("Failed to scan the sample library", "error");
  }
}

function filterServerFiles(files, query, category, metadataFilters = null) {
  const q = (query || "").toLowerCase().trim();
  const cat = (category || "all").toLowerCase().trim();

  return (files || []).filter(file => {
    const categoryId = fileCategoryId(file);
    const fileCat = (file.category || "").toLowerCase();
    const filePath = (file.path || "").toLowerCase();
    const fileName = (file.name || "").toLowerCase();
    const fileTitle = (file.title || "").toLowerCase();
    const systemTags = (file.system_tags || []).map(tag => String(tag).toLowerCase());
    const customTags = (file.custom_tags || []).map(tag => String(tag).toLowerCase());

    const matchesCat = cat === "all" || categoryId === cat ||
      (cat === "downloads" && categoryId === "ingest") ||
      (cat === "separated" && categoryId === "stems");

    const matchesQ = !q ||
      fileName.includes(q) ||
      fileTitle.includes(q) ||
      filePath.includes(q) ||
      fileCat.includes(q) ||
      categoryId.includes(q) ||
      systemTags.some(tag => tag.includes(q)) ||
      customTags.some(tag => tag.includes(q)) ||
      String(file.dataset || '').toLowerCase().includes(q) ||
      String(file.channel_name || '').toLowerCase().includes(q);

    const filters = metadataFilters || {};
    const matchesMetadata =
      (!filters.dataset || filters.dataset === 'all' || file.dataset === filters.dataset) &&
      (!filters.channel || filters.channel === 'all' || file.channel_id === filters.channel) &&
      (!filters.speaker || filters.speaker === 'all' || systemTags.includes(filters.speaker)) &&
      (!filters.verification || filters.verification === 'all' || systemTags.includes(`verification:${filters.verification}`)) &&
      (!filters.format || filters.format === 'all' || String(file.format).toLowerCase() === filters.format);

    return matchesCat && matchesQ && matchesMetadata;
  });
}

function sortLibraryFiles(files, mode = state.librarySort) {
  const sorted = [...(files || [])];
  const byName = (left, right) => String(left.title || left.name || '').localeCompare(String(right.title || right.name || ''), undefined, { sensitivity: 'base' });
  if (mode === 'oldest') sorted.sort((left, right) => (left.modified || 0) - (right.modified || 0));
  else if (mode === 'name') sorted.sort(byName);
  else if (mode === 'duration') sorted.sort((left, right) => (right.duration_s || 0) - (left.duration_s || 0) || byName(left, right));
  else if (mode === 'size') sorted.sort((left, right) => (right.size || 0) - (left.size || 0) || byName(left, right));
  else sorted.sort((left, right) => (right.modified || 0) - (left.modified || 0));
  return sorted;
}

function updateCategoryPills(container, files, activeCategory) {
  if (!container) return;
  const counts = { all: (files || []).length };
  (files || []).forEach(file => {
    const id = fileCategoryId(file);
    counts[id] = (counts[id] || 0) + 1;
  });
  container.querySelectorAll('.pill-btn').forEach(btn => {
    const cat = btn.dataset.category || 'all';
    const base = btn.dataset.label || btn.textContent.replace(/\s*\(\d+\)\s*$/, '').trim();
    btn.dataset.label = base;
    const count = counts[cat] || 0;
    btn.textContent = `${base} (${cat === 'all' ? counts.all : count})`;
    btn.style.display = (cat === 'all' || count > 0 || cat === activeCategory) ? '' : 'none';
  });
}

function populateLibraryMetadataFilters() {
  const fill = (select, values, label) => {
    if (!select) return;
    const current = select.value || 'all';
    select.innerHTML = `<option value="all">${label}</option>` + values.map(value => `<option value="${escapeHtml(value.value)}">${escapeHtml(value.label)}</option>`).join('');
    if ([...select.options].some(option => option.value === current)) select.value = current;
  };
  const unique = values => [...new Set(values.filter(Boolean))].sort();
  fill(el.tabLibraryDataset, unique(state.serverFiles.map(file => file.dataset)).map(value => ({ value, label: value })), 'All datasets');
  const channels = new Map(state.serverFiles.filter(file => file.channel_id).map(file => [file.channel_id, file.channel_name || file.channel_id]));
  fill(el.tabLibraryChannel, [...channels].map(([value, label]) => ({ value, label })), 'All channels');
  const identities = unique(state.serverFiles.flatMap(file => (file.system_tags || []).filter(tag => tag.startsWith('speaker:') || tag.startsWith('profile:'))));
  fill(el.tabLibrarySpeaker, identities.map(value => ({ value, label: value })), 'All speakers/profiles');
  fill(el.tabLibraryFormat, unique(state.serverFiles.map(file => String(file.format || '').toLowerCase())).map(value => ({ value, label: value.toUpperCase() })), 'All formats');
}

function visibleTagChips(file) {
  const preferred = (file.system_tags || []).filter(tag =>
    tag.startsWith('speaker:') || tag.startsWith('profile:') || tag.startsWith('verification:')
  );
  const custom = file.custom_tags || [];
  return [...preferred, ...custom]
    .slice(0, 4)
    .map(tag => `<span class="meta-chip ${String(tag).includes(':') ? 'system-tag-chip' : 'custom-tag-chip'}">${escapeHtml(tag)}</span>`)
    .join('');
}

let lastCheckedLibraryCheck = null;

function updateLibrarySelectionUi() {
  const count = state.librarySelectedPaths.size;
  if (el.btnBulkDeleteLibrary) {
    el.btnBulkDeleteLibrary.disabled = count === 0;
    el.btnBulkDeleteLibrary.textContent = count ? `Delete selected (${count})` : 'Delete selected';
  }
  if (el.btnBulkDeleteLibraryModal) {
    el.btnBulkDeleteLibraryModal.disabled = count === 0;
    el.btnBulkDeleteLibraryModal.textContent = count ? `Delete selected (${count})` : 'Delete selected';
  }

  if (el.tabLibrarySelectAll) {
    const tabFiles = currentTabLibraryFiles();
    el.tabLibrarySelectAll.checked = tabFiles.length > 0 && tabFiles.every(file => state.librarySelectedPaths.has(file.path));
    el.tabLibrarySelectAll.indeterminate = tabFiles.some(file => state.librarySelectedPaths.has(file.path)) && !el.tabLibrarySelectAll.checked;
  }

  if (el.libraryModalSelectAll) {
    const modalFiles = currentModalLibraryFiles();
    el.libraryModalSelectAll.checked = modalFiles.length > 0 && modalFiles.every(file => state.librarySelectedPaths.has(file.path));
    el.libraryModalSelectAll.indeterminate = modalFiles.some(file => state.librarySelectedPaths.has(file.path)) && !el.libraryModalSelectAll.checked;
  }
}

function renderGroupedFileList(container, files, { isModal = false, selectable = false } = {}) {
  container.innerHTML = "";
  const grouped = new Map();
  files.forEach(file => {
    const key = fileCategoryId(file);
    if (!grouped.has(key)) grouped.set(key, []);
    grouped.get(key).push(file);
  });
  const keys = [
    ...LIBRARY_CATEGORY_ORDER.filter(key => grouped.has(key)),
    ...[...grouped.keys()].filter(key => !LIBRARY_CATEGORY_ORDER.includes(key)),
  ];
  keys.forEach(key => {
    const groupFiles = grouped.get(key) || [];
    const header = document.createElement('div');
    header.className = 'library-group-header';
    header.innerHTML = `<span>${escapeHtml(groupFiles[0]?.category || key)}</span><span>${groupFiles.length}</span>`;
    container.appendChild(header);
    groupFiles.forEach(file => container.appendChild(buildFileItemCard(file, { isModal, selectable })));
  });
}

function buildFileItemCard(file, { isModal = false, selectable = false } = {}) {
  const badgeInfo = getFileModelBadge(file);
  const isPlaying = currentPreviewingPath === file.path && previewAudioEl && !previewAudioEl.paused;
  const selected = state.librarySelectedPaths.has(file.path);
  const durStr = (file.duration_s || 0) > 0 ? `${(file.duration_s).toFixed(1)}s` : '';
  const srStr = file.sample_rate ? `${(file.sample_rate / 1000).toFixed(1)} kHz` : '';
  const chStr = file.channels === 1 ? 'Mono' : (file.channels === 2 ? 'Stereo' : '');
  const fmtStr = (file.format || 'wav').toUpperCase();
  const loadTarget = isModal ? (state.libraryLoadTarget || 'workspace') : 'workspace';
  const loadLabel = LIBRARY_LOAD_TARGETS[loadTarget]?.button || 'Load';

  const card = document.createElement("div");
  card.className = `file-item-card${selected ? ' selected' : ''}`;
  card.innerHTML = `
    <div class="file-left-group">
      ${selectable ? `<input type="checkbox" class="file-select-check" data-path="${escapeHtml(file.path)}" ${selected ? 'checked' : ''} aria-label="Select ${escapeHtml(file.title || file.name)}">` : ''}
      <button class="btn-preview-file ${isPlaying ? 'playing' : ''}" data-path="${escapeHtml(file.path)}" title="${isPlaying ? 'Pause preview' : 'Preview this file'}" aria-label="Preview track">
        ${isPlaying ? PREVIEW_PAUSE_ICON : PREVIEW_PLAY_ICON}
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
          ${file.dataset ? `<span class="meta-chip">dataset:${escapeHtml(file.dataset)}</span>` : ''}
          ${visibleTagChips(file)}
        </div>
      </div>
    </div>
    <div class="file-actions">
      ${file.registry_item_id ? `<button class="btn btn-sm btn-ghost btn-edit-file-tags" title="Edit custom tags">Tags</button>` : ''}
      <button class="btn btn-sm btn-primary btn-load-target" data-path="${escapeHtml(file.path)}" title="${escapeHtml(loadLabel)}">
        <span>${escapeHtml(isModal ? loadLabel.replace(/^Send to |^Load into |^Open in /, '') : 'Load')}</span>
      </button>
      <div class="dropdown-actions-wrap" style="position: relative; display: inline-block;">
        <button class="btn btn-sm btn-secondary btn-more-actions" title="More routing actions">
          <span>⋯</span>
        </button>
        <div class="actions-popup-menu hidden" style="position: absolute; right: 0; top: 100%; margin-top: 4px; z-index: 50; background: var(--bg-surface-elevated); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); box-shadow: 0 8px 24px rgba(0,0,0,0.5); padding: 4px; min-width: 180px; display: flex; flex-direction: column; gap: 2px;">
          <button class="menu-item-btn btn-send-workspace" style="text-align: left; padding: 6px 10px; font-size: 0.78rem; background: none; border: none; color: var(--text-primary); cursor: pointer; border-radius: 4px; display: flex; align-items: center; gap: 6px;">🎵 Load into Workspace</button>
          <button class="menu-item-btn btn-send-cutter" style="text-align: left; padding: 6px 10px; font-size: 0.78rem; background: none; border: none; color: var(--text-primary); cursor: pointer; border-radius: 4px; display: flex; align-items: center; gap: 6px;">✂️ Open in Cutter</button>
          <button class="menu-item-btn btn-send-sep" style="text-align: left; padding: 6px 10px; font-size: 0.78rem; background: none; border: none; color: var(--text-primary); cursor: pointer; border-radius: 4px; display: flex; align-items: center; gap: 6px;">🎛️ Send to Separation</button>
          <button class="menu-item-btn btn-send-diar" style="text-align: left; padding: 6px 10px; font-size: 0.78rem; background: none; border: none; color: var(--text-primary); cursor: pointer; border-radius: 4px; display: flex; align-items: center; gap: 6px;">👥 Send to Diarization</button>
          <button class="menu-item-btn btn-send-purity" style="text-align: left; padding: 6px 10px; font-size: 0.78rem; background: none; border: none; color: var(--text-primary); cursor: pointer; border-radius: 4px; display: flex; align-items: center; gap: 6px;">🛡️ Send to Speaker Purity</button>
          <a href="/api/library/download?path=${encodeURIComponent(file.path)}" download="${escapeHtml(file.name)}" class="menu-item-btn" style="text-align: left; padding: 6px 10px; font-size: 0.78rem; background: none; border: none; color: var(--accent-primary-hover); text-decoration: none; cursor: pointer; border-radius: 4px; display: flex; align-items: center; gap: 6px;">⬇️ Download File</a>
        </div>
      </div>
      <button class="btn btn-sm btn-ghost btn-delete-file text-danger" data-path="${escapeHtml(file.path)}" title="Permanently delete from disk">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path><line x1="10" y1="11" x2="10" y2="17"></line><line x1="14" y1="11" x2="14" y2="17"></line></svg>
      </button>
    </div>
  `;

  const btnPrev = card.querySelector('.btn-preview-file');
  btnPrev.addEventListener('click', () => toggleFilePreview(file.path, btnPrev));

  const chk = card.querySelector('.file-select-check');
  if (chk) {
    chk.addEventListener('click', (event) => {
      const container = chk.closest('.library-list-wrapper, .library-modal-list');
      if (event.shiftKey && lastCheckedLibraryCheck && container && container.contains(lastCheckedLibraryCheck)) {
        const checks = Array.from(container.querySelectorAll('.file-select-check'));
        const start = checks.indexOf(lastCheckedLibraryCheck);
        const end = checks.indexOf(chk);
        if (start !== -1 && end !== -1) {
          const [low, high] = start < end ? [start, end] : [end, start];
          for (let i = low; i <= high; i++) {
            checks[i].checked = chk.checked;
            const p = checks[i].dataset.path;
            if (p) {
              if (chk.checked) state.librarySelectedPaths.add(p);
              else state.librarySelectedPaths.delete(p);
            }
            checks[i].closest('.file-item-card')?.classList.toggle('selected', chk.checked);
          }
        }
      } else {
        if (chk.checked) state.librarySelectedPaths.add(file.path);
        else state.librarySelectedPaths.delete(file.path);
        card.classList.toggle('selected', chk.checked);
      }
      lastCheckedLibraryCheck = chk;
      updateLibrarySelectionUi();
    });
  }

  card.querySelector('.btn-load-target').addEventListener('click', () => {
    loadLibraryFileTo(file.path, loadTarget);
  });
  card.querySelector('.btn-edit-file-tags')?.addEventListener('click', async () => {
    const entered = prompt('Custom tags (comma-separated). System tags remain read-only.', (file.custom_tags || []).join(', '));
    if (entered === null) return;
    const customTags = entered.split(',').map(tag => tag.trim()).filter(Boolean);
    try {
      const response = await fetch(`/api/items/${encodeURIComponent(file.registry_item_id)}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ custom_tags: customTags }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || 'Unable to update tags');
      await fetchServerFiles();
      showToast('Custom tags updated', 'success');
    } catch (err) {
      showToast(`Tag update failed: ${err.message || String(err)}`, 'error');
    }
  });

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

  card.querySelector('.btn-send-workspace')?.addEventListener('click', () => { popupMenu.classList.add('hidden'); loadLibraryFileTo(file.path, 'workspace'); });
  card.querySelector('.btn-send-cutter')?.addEventListener('click', () => { popupMenu.classList.add('hidden'); loadLibraryFileTo(file.path, 'cutter'); });
  card.querySelector('.btn-send-sep')?.addEventListener('click', () => { popupMenu.classList.add('hidden'); loadLibraryFileTo(file.path, 'separation'); });
  card.querySelector('.btn-send-diar')?.addEventListener('click', () => { popupMenu.classList.add('hidden'); loadLibraryFileTo(file.path, 'diarization'); });
  card.querySelector('.btn-send-purity')?.addEventListener('click', () => { popupMenu.classList.add('hidden'); loadLibraryFileTo(file.path, 'purity'); });
  card.querySelector('.btn-delete-file').addEventListener('click', () => deleteServerFile(file.path, file.name));

  return card;
}

function currentTabLibraryFiles() {
  return sortLibraryFiles(
    filterServerFiles(state.serverFiles, state.tabLibrarySearch, state.tabLibraryCategory, state.tabLibraryFilters)
  );
}

function currentModalLibraryFiles() {
  return sortLibraryFiles(
    filterServerFiles(state.serverFiles, state.libraryModalSearch, state.libraryModalCategory)
  );
}

function renderServerFiles() {
  const container = el.serverFilesList;
  if (!container) return;

  const searchMatches = filterServerFiles(state.serverFiles, state.tabLibrarySearch, "all", state.tabLibraryFilters);
  updateCategoryPills(el.tabLibraryCategories, searchMatches, state.tabLibraryCategory);
  const filtered = currentTabLibraryFiles();

  if (el.tabLibraryCount) {
    el.tabLibraryCount.textContent = `${filtered.length} of ${(state.serverFiles || []).length} files`;
  }
  updateLibrarySelectionUi();

  if (filtered.length === 0) {
    container.innerHTML = `<div class="empty-placeholder" style="padding: 2.5rem 1rem; text-align: center;">No project audio files found matching filter.</div>`;
    return;
  }

  renderGroupedFileList(container, filtered, { isModal: false, selectable: true });
}

async function deleteServerFile(filePath, fileName) {
  const displayName = fileName || filePath.split('/').pop();
  if (!confirm(`Permanently delete "${displayName}" from disk?\nThe matching .json sidecar will also be removed.`)) {
    return;
  }
  try {
    showToast(`Deleting ${displayName}...`, "info");
    const res = await fetch("/api/library/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: filePath }),
    });
    await parseJsonResponse(res);
    state.librarySelectedPaths.delete(filePath);
    if (currentPreviewingPath === filePath) {
      stopFilePreview();
    }
    showToast(`Deleted ${displayName}`, "success");
    await fetchServerFiles();
    await fetchAudioList();
  } catch (err) {
    showToast(`Delete failed: ${err.message}`, "error");
  }
}

async function bulkDeleteSelectedLibraryFiles() {
  const paths = [...state.librarySelectedPaths];
  if (!paths.length) return;
  if (!confirm(`Permanently delete ${paths.length} selected file${paths.length === 1 ? '' : 's'} from disk?\nMatching .json sidecars will also be removed.`)) {
    return;
  }
  try {
    showToast(`Deleting ${paths.length} files...`, "info");
    const res = await fetch("/api/library/bulk-delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ paths }),
    });
    const data = await parseJsonResponse(res);
    state.librarySelectedPaths = new Set();
    if (paths.includes(currentPreviewingPath)) {
      stopFilePreview();
    }
    const failed = (data.errors || []).length;
    showToast(`Deleted ${data.deleted_count || 0} file${(data.deleted_count || 0) === 1 ? '' : 's'}${failed ? ` (${failed} failed)` : ''}`, failed ? "warning" : "success");
    await fetchServerFiles();
    await fetchAudioList();
  } catch (err) {
    showToast(`Bulk delete failed: ${err.message}`, "error");
  }
}

async function loadLibraryFileTo(filePath, target = 'workspace') {
  try {
    showToast(`Loading ${filePath.split('/').pop()}...`, "info");
    const res = await fetch("/api/library/load", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: filePath }),
    });
    const data = await parseJsonResponse(res);

    if (data.audio_id) {
      await fetchAudioList();
      closeAllModals();
      const title = data.metadata?.title || filePath.split('/').pop();

      if (target === 'workspace') {
        switchTab('tab-workspace');
        await setActiveAudio(data.audio_id, { play: true });
        showToast(`Loaded "${title}" into Workspace`, "success");
      } else if (target === 'cutter') {
        switchTab('tab-workspace');
        await setActiveAudio(data.audio_id, { play: false });
        showToast(`Loaded "${title}" into Audio Cutter`, "success");
      } else if (target === 'separation') {
        switchTab('tab-separation');
        if (el.sepInputSelect) {
          el.sepInputSelect.value = data.audio_id;
          el.sepInputSelect.dispatchEvent(new Event('change'));
        }
        showToast(`Selected "${title}" for Separation`, "success");
      } else if (target === 'diarization') {
        switchTab('tab-diarization');
        if (el.diarInputSelect) {
          el.diarInputSelect.value = data.audio_id;
          el.diarInputSelect.dispatchEvent(new Event('change'));
        }
        showToast(`Selected "${title}" for Diarization`, "success");
      } else if (target === 'annotation') {
        switchTab('tab-annotation');
        if (el.annAudioSelect) {
          el.annAudioSelect.value = data.audio_id;
          await selectAnnotationAudio(data.audio_id);
        }
        showToast(`Selected "${title}" for manual annotation`, "success");
      } else if (target === 'purity') {
        switchTab('tab-purity');
        if (el.purityInputSelect) {
          el.purityInputSelect.value = data.audio_id;
          state.purity.audioId = data.audio_id;
          el.purityInputSelect.dispatchEvent(new Event('change'));
        }
        showToast(`Selected "${title}" for Speaker Purity`, "success");
      } else {
        switchTab('tab-workspace');
        await setActiveAudio(data.audio_id, { play: true });
        showToast(`Loaded "${title}" into Workspace`, "success");
      }
    }
  } catch (err) {
    showToast(`Failed to load file: ${err.message}`, "error");
  }
}

function applyLibraryModalContext(target) {
  const meta = LIBRARY_LOAD_TARGETS[target] || LIBRARY_LOAD_TARGETS.workspace;
  if (el.libraryModalTitle) el.libraryModalTitle.textContent = meta.title;
  if (el.libraryModalSubtitle) el.libraryModalSubtitle.textContent = meta.subtitle;
}

async function openLibraryModal(loadTarget = 'workspace') {
  const target = (typeof loadTarget === 'string' && loadTarget && LIBRARY_LOAD_TARGETS[loadTarget]) ? loadTarget : 'workspace';
  state.libraryLoadTarget = target;
  applyLibraryModalContext(target);
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
  if (el.libraryModalSort) el.libraryModalSort.value = state.librarySort;
  if (el.modalLibraryItems) {
    el.modalLibraryItems.innerHTML = '<div class="empty-placeholder">Scanning project directories…</div>';
  }
  await fetchServerFiles();
  renderLibraryModalItems();
}

function renderLibraryModalItems() {
  if (!el.modalLibraryItems) return;

  const searchMatches = filterServerFiles(state.serverFiles, state.libraryModalSearch, "all");
  updateCategoryPills(el.libraryModalCategories, searchMatches, state.libraryModalCategory);
  const filtered = currentModalLibraryFiles();

  if (el.libraryModalCount) {
    el.libraryModalCount.textContent = `${filtered.length} of ${(state.serverFiles || []).length} sample files`;
  }
  updateLibrarySelectionUi();

  if (filtered.length === 0) {
    el.modalLibraryItems.innerHTML = `
      <div class="empty-placeholder" style="padding: 2.5rem 1rem; text-align: center;">
        <span style="font-size: 2rem; display: block; margin-bottom: 0.5rem;">📂</span>
        <span>No sample audio files match your search filter.</span>
      </div>
    `;
    return;
  }

  renderGroupedFileList(el.modalLibraryItems, filtered, { isModal: true, selectable: true });
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
    case 'speaker_purity_verify': return 'Speaker Purity';
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
      const lanes = data.device_queues || {};
      const laneText = Object.keys(lanes).length
        ? Object.entries(lanes)
            .map(([dev, lane]) => `${dev}: ${lane.running || 0} run / ${lane.queued || 0} q`)
            .join(' · ')
        : `Studio: ${summary.studio_running} active • Pipeline: ${summary.pipeline_running} active`;
      el.queueActiveSplit.textContent = laneText;
    }

    // Render multi-GPU cards if 2+ GPUs detected
    if (el.queueGpuDevicesGrid) {
      const devList = device.devices || (data.telemetry && data.telemetry.gpu && data.telemetry.gpu.devices) || [];
      const lanes = data.device_queues || {};
      if (devList.length > 1) {
        el.queueGpuDevicesGrid.style.display = 'grid';
        el.queueGpuDevicesGrid.innerHTML = devList.map((d, i) => {
          const dLoad = Number.isFinite(d.load_percent) ? Math.round(d.load_percent) : (Number.isFinite(d.utilization_percent) ? Math.round(d.utilization_percent) : 0);
          const dVram = (d.used_vram_mb != null && d.total_vram_mb != null) ? `${d.used_vram_mb} / ${d.total_vram_mb} MB (${Math.round(d.vram_percent || 0)}%)` : '-- MB';
          const dTemp = d.temperature_c != null ? `${Math.round(d.temperature_c)}°C` : '';
          const dPower = d.power_w != null ? `${d.power_w} / ${d.power_limit_w ?? '--'} W` : '';
          const lane = lanes[d.id] || lanes[`cuda:${i}`] || {};
          const qRunning = lane.running || 0;
          const qQueued = lane.queued || 0;
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
                <span>Queue: <strong>${qRunning} run / ${qQueued} wait</strong></span>
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
      el.queueModalSubtitle.textContent = `Per-GPU queues: ${summary.total_running} running, ${summary.total_queued} queued (Studio + Pipeline)`;
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

    if (el.studioQueueTaskList) {
      filteredItems = filteredItems.slice().sort((a, b) => {
        const da = a.device || a.metadata?.queue_device || a.metadata?.device || '';
        const db = b.device || b.metadata?.queue_device || b.metadata?.device || '';
        if (da !== db) return String(da).localeCompare(String(db));
        return (b.created_at || 0) - (a.created_at || 0);
      });
    }
    const errorsByItemId = new Map(
      filteredItems.filter(item => item.error).map(item => [String(item.id), String(item.error)])
    );

    if (filteredItems.length === 0) {
      el.studioQueueTaskList.innerHTML = `
        <div class="queue-empty-state">
          <div class="queue-empty-icon">☕</div>
          <div class="queue-empty-title">${filter === 'all' ? 'All GPU queues are idle' : 'No workloads in this view'}</div>
          <div class="queue-empty-sub">${filter === 'all' ? 'Each GPU has its own queue. Studio and Pipeline jobs appear here when routed to a device.' : 'No workloads match the selected filter.'}</div>
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
        errorHtml = `
          <div class="task-card-error">
            <button class="btn-copy-task-error" data-item-id="${escapeHtml(item.id)}" title="Copy error message">Copy</button>
            <div class="task-card-error-text">${escapeHtml(item.error)}</div>
          </div>`;
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
      const itemDevice = item.device || meta.queue_device || meta.device || item.params?.device || (item.result && item.result.device);
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
            ${itemDevice ? `<span class="workload-source-badge" style="background:hsla(200,80%,45%,0.15);color:var(--text-secondary);">${escapeHtml(itemDevice)}</span>` : ''}
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
    el.studioQueueTaskList.querySelectorAll('.btn-copy-task-error').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        e.stopPropagation();
        const error = errorsByItemId.get(btn.dataset.itemId);
        if (!error) return;
        try {
          if (navigator.clipboard?.writeText) {
            await navigator.clipboard.writeText(error);
          } else {
            const textarea = document.createElement('textarea');
            textarea.value = error;
            textarea.style.position = 'fixed';
            textarea.style.opacity = '0';
            document.body.appendChild(textarea);
            textarea.select();
            const copied = document.execCommand('copy');
            textarea.remove();
            if (!copied) throw new Error('Clipboard copy was rejected');
          }
          showToast('Error message copied', 'success');
        } catch (_) {
          showToast('Could not copy error message', 'error');
        }
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
  stopFilePreview();
  if (el.modalSaveTo) el.modalSaveTo.classList.add('hidden');
  if (el.modalLibrary) el.modalLibrary.classList.add('hidden');
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
      stopFilePreview();
      if (el.modalLibrary) el.modalLibrary.classList.add('hidden');
    });
  }
  if (el.btnCancelLibraryModal) {
    el.btnCancelLibraryModal.addEventListener('click', () => {
      stopFilePreview();
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

  if (el.libraryModalSort) {
    el.libraryModalSort.addEventListener('change', (e) => {
      state.librarySort = e.target.value || 'newest';
      if (el.tabLibrarySort) el.tabLibrarySort.value = state.librarySort;
      renderLibraryModalItems();
      renderServerFiles();
    });
  }

  if (el.btnRefreshLibraryModal) {
    el.btnRefreshLibraryModal.addEventListener('click', async () => {
      el.btnRefreshLibraryModal.disabled = true;
      try {
        await fetchServerFiles();
      } finally {
        el.btnRefreshLibraryModal.disabled = false;
      }
    });
  }

  if (el.libraryModalSelectAll) {
    el.libraryModalSelectAll.addEventListener('change', () => {
      const visible = currentModalLibraryFiles();
      if (el.libraryModalSelectAll.checked) {
        visible.forEach(file => state.librarySelectedPaths.add(file.path));
      } else {
        visible.forEach(file => state.librarySelectedPaths.delete(file.path));
      }
      renderLibraryModalItems();
      renderServerFiles();
    });
  }

  if (el.btnBulkDeleteLibraryModal) {
    el.btnBulkDeleteLibraryModal.addEventListener('click', bulkDeleteSelectedLibraryFiles);
  }
}

function audioOptionLabel(item, { includeFormat = true } = {}) {
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
  const title = item.title || item.name || item.id || "Untitled";
  const channel = item.channel_name ? `[${item.channel_name}] ` : '';
  const dur = Number(item.duration_s) || 0;
  const fmt = (item.format || "wav").toString().toUpperCase();
  if (includeFormat) {
    return `${channel}${prefix}${title} (${dur.toFixed(1)}s, ${fmt})`;
  }
  return `${channel}${prefix}${title} (${dur.toFixed(1)}s)`;
}

function normalizedAudioPath(path) {
  return String(path || "").replaceAll('\\', '/').replace(/\/{2,}/g, '/');
}

function populateAllAudioSelects() {
  const standardSelects = [
    el.sepInputSelect,
    el.diarInputSelect,
    el.annAudioSelect,
    el.purityInputSelect,
  ];

  standardSelects.forEach(select => {
    if (!select) return;
    const currentVal = select.value;
    const isDiar = select === el.diarInputSelect;
    const isPurity = select === el.purityInputSelect;
    select.innerHTML = '<option value="">-- Select session or library track --</option>';

    const sessionGroup = document.createElement("optgroup");
    sessionGroup.label = `🎛️ Session Audio (${state.audioList.length})`;

    state.audioList.forEach(item => {
      const opt = document.createElement("option");
      opt.value = item.id;
      opt.textContent = audioOptionLabel(item);
      sessionGroup.appendChild(opt);
    });

    if (sessionGroup.children.length > 0) {
      select.appendChild(sessionGroup);
    }

    // All processing selectors can load a library file into the active registry.
    if (Array.isArray(state.serverFiles) && state.serverFiles.length > 0) {
      const sessionPaths = new Set(
        state.audioList.map(a => normalizedAudioPath(a.path)).filter(Boolean)
      );
      const grouped = new Map();
      sortLibraryFiles(state.serverFiles).forEach(file => {
        const canonicalPath = normalizedAudioPath(file?.absolute_path || file?.path);
        if (!file?.path || sessionPaths.has(canonicalPath)) return;
        const key = fileCategoryId(file);
        if (!grouped.has(key)) grouped.set(key, []);
        grouped.get(key).push(file);
      });
      const keys = [
        ...LIBRARY_CATEGORY_ORDER.filter(key => grouped.has(key)),
        ...[...grouped.keys()].filter(key => !LIBRARY_CATEGORY_ORDER.includes(key)),
      ];
      keys.forEach(key => {
        const groupFiles = grouped.get(key) || [];
        const libraryGroup = document.createElement("optgroup");
        libraryGroup.label = `📁 ${groupFiles[0]?.category || key} (${groupFiles.length})`;
        groupFiles.forEach(file => {
          const opt = document.createElement("option");
          opt.value = `lib:${file.path}`;
          const title = file.title || file.name || file.path;
          const dur = Number(file.duration_s) || 0;
          const fmt = (file.format || "wav").toString().toUpperCase();
          opt.textContent = `${title} (${dur.toFixed(1)}s, ${fmt})`;
          libraryGroup.appendChild(opt);
        });
        select.appendChild(libraryGroup);
      });
    }

    if (currentVal && (
      state.audioList.some(a => a.id === currentVal) ||
      (typeof currentVal === "string" && currentVal.startsWith("lib:"))
    )) {
      // Keep lib: selection only briefly; load flow replaces it with session id
      if (!currentVal.startsWith("lib:") || Array.from(select.options).some(o => o.value === currentVal)) {
        select.value = currentVal;
      } else if (state.activeAudio) {
        select.value = state.activeAudio.id;
      }
    } else if (state.diarization?.audioId && isPurity && state.diarization.turns?.length > 0 && state.audioList.some(a => a.id === state.diarization.audioId)) {
      select.value = state.diarization.audioId;
    } else if (state.activeAudio && state.audioList.some(a => a.id === state.activeAudio.id)) {
      select.value = state.activeAudio.id;
    } else if (state.diarization?.audioId && isDiar && state.audioList.some(a => a.id === state.diarization.audioId)) {
      select.value = state.diarization.audioId;
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

      const title = item.title || item.id || "Untitled";
      opt.textContent = `${prefix}${title} (${(Number(item.duration_s) || 0).toFixed(1)}s)`;

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
    const selected = el.diarInputSelect.value;
    if (selected && !selected.startsWith('lib:')) {
      renderDiarizationChildren(selected);
      updateDiarInputMeta(selected);
    }
  }
  populateTargetClipSelect();
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
  } else if (task.type === "speaker_purity_verify" || task.type === "diarization_batch_verify") {
    statusText = el.purityTaskStatusText;
    progressBar = el.purityProgressBar;
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
  syncSelect(el.purityDeviceSelect);
  
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
      populateSelect(el.purityDeviceSelect);

      if (el.sepDeviceSelect) {
        el.sepDeviceSelect.addEventListener('change', (e) => setTargetGpu(e.target.value, false));
      }
      if (el.diarDeviceSelect) {
        el.diarDeviceSelect.addEventListener('change', (e) => setTargetGpu(e.target.value, false));
      }
      if (el.purityDeviceSelect) {
        el.purityDeviceSelect.addEventListener('change', (e) => setTargetGpu(e.target.value, false));
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
  [
    ['dataset', el.tabLibraryDataset],
    ['channel', el.tabLibraryChannel],
    ['speaker', el.tabLibrarySpeaker],
    ['verification', el.tabLibraryVerification],
    ['format', el.tabLibraryFormat],
  ].forEach(([name, select]) => {
    select?.addEventListener('change', () => {
      state.tabLibraryFilters[name] = select.value;
      renderServerFiles();
    });
  });

  if (el.tabLibrarySort) {
    el.tabLibrarySort.addEventListener('change', (e) => {
      state.librarySort = e.target.value || 'newest';
      if (el.libraryModalSort) el.libraryModalSort.value = state.librarySort;
      renderServerFiles();
      renderLibraryModalItems();
    });
  }

  if (el.tabLibrarySelectAll) {
    el.tabLibrarySelectAll.addEventListener('change', () => {
      const visible = currentTabLibraryFiles();
      if (el.tabLibrarySelectAll.checked) {
        visible.forEach(file => state.librarySelectedPaths.add(file.path));
      } else {
        visible.forEach(file => state.librarySelectedPaths.delete(file.path));
      }
      renderServerFiles();
      renderLibraryModalItems();
    });
  }

  if (el.btnBulkDeleteLibrary) {
    el.btnBulkDeleteLibrary.addEventListener('click', bulkDeleteSelectedLibraryFiles);
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
    loadSpeakerProfiles();
    if (el.diarInputSelect) {
      if (!el.diarInputSelect.value && state.activeAudio) {
        el.diarInputSelect.value = state.activeAudio.id;
      }
      const audioId = el.diarInputSelect.value;
      if (audioId && !audioId.startsWith('lib:') && (!state.diarization.data || state.diarization.audioId !== audioId)) {
        openDiarizationAudio(audioId, { restoreHistory: true });
      } else if (audioId) {
        updateDiarInputMeta(audioId);
        renderDiarizationChildren(audioId);
      }
    }
    setDiarZoom(state.diarization.zoom || 1.0);
    renderDiarWaveform();
    renderDiarRuler();
    startDiarPlaybackWatch();
  } else if (tabId === 'tab-annotation') {
    loadAnnotationCatalog();
    if (!state.annotation.current && el.annAudioSelect && !el.annAudioSelect.value && state.activeAudio) {
      el.annAudioSelect.value = state.activeAudio.id;
      state.annotation.audioId = state.activeAudio.id;
      const item = state.audioList.find(audio => audio.id === state.activeAudio.id);
      if (item && el.annAudioMeta) {
        el.annAudioMeta.textContent = `${item.title || item.source_id} · ${formatAnnotationTime(item.duration_s)} · ${(item.sample_rate || 0).toLocaleString()} Hz`;
      }
    }
    renderAnnotationEditor();
  } else if (tabId === 'tab-purity') {
    loadSpeakerProfiles().then(() => syncPurityProfileSelect());
    if (el.purityInputSelect) {
      if (!state.purity.audioId && state.diarization.audioId && state.diarization.turns?.length > 0) {
        el.purityInputSelect.value = state.diarization.audioId;
      } else if (!el.purityInputSelect.value && state.activeAudio) {
        el.purityInputSelect.value = state.activeAudio.id;
      }
      const audioId = el.purityInputSelect.value;
      if (audioId) {
        updatePurityInputMeta(audioId);
        syncPurityDiarizationStatus();
      }
    }
  }

  if (tabId !== 'tab-diarization' && el.audio) {
    el.audio.muted = false;
    stopDiarPlaybackWatch();
  }

  if (tabId !== 'tab-annotation') {
    state.annotation.loopTurnId = null;
    if (el.btnAnnLoopSelected) el.btnAnnLoopSelected.classList.remove('active');
  }

  if (tabId !== 'tab-purity') {
    stopPuritySegmentPreview();
  }

  if (tabId !== 'tab-comparison') syncActivePlaybackControls();
}

// ==================== THEME MANAGEMENT ====================

function initTheme() {
  applyTheme('light');

  if (el.btnThemeToggle) {
    el.btnThemeToggle.addEventListener('click', () => {
      applyTheme('light');
    });
  }
}

function applyTheme(theme = 'light') {
  document.documentElement.setAttribute('data-theme', 'light');
  try {
    localStorage.setItem('sonic_theme', 'light');
  } catch (_) {}

  if (el.iconThemeSun) el.iconThemeSun.classList.remove('hidden');
  if (el.iconThemeMoon) el.iconThemeMoon.classList.add('hidden');
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
  try { initSeparationStudio(); } catch (e) { console.error("initSeparationStudio error:", e); }
  try { initDiarizationStudio(); } catch (e) { console.error("initDiarizationStudio error:", e); }
  try { initAnnotationTab(); } catch (e) { console.error("initAnnotationTab error:", e); }
  try { initKnownSpeakerManager(); } catch (e) { console.error("initKnownSpeakerManager error:", e); }
  try { initTargetSpeakerEvaluation(); } catch (e) { console.error("initTargetSpeakerEvaluation error:", e); }
  try { initPurityTab(); } catch (e) { console.error("initPurityTab error:", e); }
  try { initAuditionHub(); } catch (e) { console.error("initAuditionHub error:", e); }
  try { initKeyboardShortcuts(); } catch (e) { console.error("initKeyboardShortcuts error:", e); }
  try { initNavigation(); } catch (e) { console.error("initNavigation error:", e); }
  try { initModals(); } catch (e) { console.error("initModals error:", e); }

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
  try { await fetchEvaluations(); } catch (e) { console.error("fetchEvaluations error:", e); }

  // Restore saved active tab
  try {
    const savedTab = localStorage.getItem('sonic_active_tab');
    if (savedTab && document.getElementById(savedTab)) {
      switchTab(savedTab);
    }
  } catch (_) {}

  // Restore saved active audio. An empty session stays empty until the user
  // explicitly chooses a library file or creates a new audio object.
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
    }
  } catch (e) {
    console.error("Audio auto-load error:", e);
  }

  if (state.activeTab === 'tab-diarization' && el.diarInputSelect?.value && !state.diarization.data) {
    openDiarizationAudio(el.diarInputSelect.value, { restoreHistory: true });
  }

  console.log("SonicStudio initialized successfully!");
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initApp);
} else {
  initApp();
}
