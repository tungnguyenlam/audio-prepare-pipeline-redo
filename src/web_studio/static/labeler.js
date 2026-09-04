/**
 * SonicStudio — Sample Quality Labeler
 * Fast keyboard-driven labeling of DiarizationResult turns into:
 * Accept | Contain background noise | Contain more than 1 speaker | Word being chopped off
 * and export to self-contained audio datasets with train/val/test splits.
 */

(function () {
  'use strict';

  const LABELS = {
    accept: { name: 'Accept', short: 'Accept', key: '1', color: 'accept', hint: 'Clean single-speaker' },
    noise: { name: 'Contain background noise', short: 'Noise', key: '2', color: 'noise', hint: 'Music, hum, ambient' },
    multi_speaker: { name: 'Contain more than 1 speaker', short: 'Multi-Spk', key: '3', color: 'multi_speaker', hint: 'Overlap, crosstalk' },
    chopped: { name: 'Word being chopped off', short: 'Chopped', key: '4', color: 'chopped', hint: 'VAD edge cut' },
  };

  const state = {
    activeResultId: null,
    resultData: null,
    labels: {}, // { [turnIndex]: { label, tags, notes } }
    activeTurnIndex: 0,
    filterLabel: 'all',
    filterSpeaker: 'all',
    searchQuery: '',
    autoPlayNext: true,
    autoAdvance: true,
    isDirty: false,
    saveTimer: null,
    audioPlayer: new Audio(),
    playingTurnIndex: null,
    // Trainer state
    activeTrainTaskId: null,
    trainPollTimer: null,
    trainStartTime: null,
    trainElapsedTimer: null,
    trainerExpanded: true,
    chartRenderer: null,
  };

  // DOM Cache
  let dom = {};

  function initDom() {
    dom = {
      tabPane: document.getElementById('tab-labeler'),
      resultSelect: document.getElementById('lbl-result-select'),
      btnReloadResults: document.getElementById('btn-lbl-reload-results'),
      btnSaveDraft: document.getElementById('btn-lbl-save-draft'),
      saveIndicator: document.getElementById('lbl-save-indicator'),
      btnExportModal: document.getElementById('btn-lbl-open-export'),
      btnShortcutsModal: document.getElementById('btn-lbl-shortcuts'),
      
      // Header & Stats
      sourceInfoBar: document.getElementById('lbl-source-info-bar'),
      statTotal: document.getElementById('lbl-stat-total'),
      statLabeled: document.getElementById('lbl-stat-labeled'),
      statAccept: document.getElementById('lbl-stat-accept'),
      statNoise: document.getElementById('lbl-stat-noise'),
      statMulti: document.getElementById('lbl-stat-multi'),
      statChopped: document.getElementById('lbl-stat-chopped'),

      // Trainer Card & Controls
      btnToggleTrainer: document.getElementById('btn-lbl-toggle-trainer'),
      trainerCard: document.getElementById('lbl-trainer-card'),
      trainerHeader: document.getElementById('lbl-trainer-header'),
      trainerBody: document.getElementById('lbl-trainer-body'),
      btnTrainerCollapse: document.getElementById('btn-lbl-trainer-collapse'),
      trainStatusBadge: document.getElementById('lbl-train-status-badge'),
      trainDataset: document.getElementById('lbl-train-dataset'),
      btnRefreshDatasets: document.getElementById('btn-lbl-refresh-datasets'),
      trainBackbone: document.getElementById('lbl-train-backbone'),
      trainMode: document.getElementById('lbl-train-mode'),
      trainDevice: document.getElementById('lbl-train-device'),
      trainEpochs: document.getElementById('lbl-train-epochs'),
      trainBatchSize: document.getElementById('lbl-train-batch-size'),
      trainLrBackbone: document.getElementById('lbl-train-lr-backbone'),
      trainLrHead: document.getElementById('lbl-train-lr-head'),
      btnStartTrain: document.getElementById('btn-lbl-start-train'),
      btnStopTrain: document.getElementById('btn-lbl-stop-train'),
      trainElapsedTime: document.getElementById('lbl-train-elapsed-time'),

      // Weights & Biases Controls
      chkUseWandb: document.getElementById('lbl-train-use-wandb'),
      wandbEnvStatus: document.getElementById('lbl-wandb-env-status'),
      wandbEnvHint: document.getElementById('lbl-wandb-env-hint'),
      wandbFields: document.getElementById('lbl-wandb-fields'),
      trainWandbProject: document.getElementById('lbl-train-wandb-project'),
      trainWandbRunName: document.getElementById('lbl-train-wandb-run-name'),
      wandbRunLink: document.getElementById('lbl-wandb-run-link'),

      // Live Telemetry & Metrics
      trainTelemetry: document.getElementById('lbl-train-telemetry'),
      trainProgressBar: document.getElementById('lbl-train-progress-bar'),
      trainStepText: document.getElementById('lbl-train-step-text'),
      trainPctText: document.getElementById('lbl-train-pct-text'),
      valTrainLoss: document.getElementById('lbl-val-train-loss'),
      valValLoss: document.getElementById('lbl-val-val-loss'),
      valAcceptAcc: document.getElementById('lbl-val-accept-acc'),
      valNoiseF1: document.getElementById('lbl-val-noise-f1'),
      valMultiF1: document.getElementById('lbl-val-multi-f1'),
      valChoppedF1: document.getElementById('lbl-val-chopped-f1'),
      bestEpochText: document.getElementById('lbl-best-epoch-text'),
      historyTbody: document.getElementById('lbl-history-tbody'),
      trainTerminal: document.getElementById('lbl-train-terminal'),
      btnClearLogs: document.getElementById('btn-lbl-clear-logs'),

      // Interactive Charts Viewport
      chartCanvas: document.getElementById('lbl-canvas-charts'),
      chartTooltip: document.getElementById('lbl-chart-tooltip'),
      chartLegends: document.getElementById('lbl-chart-legends'),
      chartHoverInfo: document.getElementById('lbl-chart-hover-info'),
      chartViewport: document.getElementById('lbl-chart-viewport'),
      chartTabs: document.querySelectorAll('.lbl-chart-tab-btn'),

      // Models Modal
      btnOpenModels: document.getElementById('btn-lbl-open-models'),
      modelsModal: document.getElementById('lbl-models-modal'),
      btnModelsClose: document.getElementById('btn-lbl-models-close'),
      modelsListContainer: document.getElementById('lbl-models-list-container'),
      
      // Filters
      speakerSelect: document.getElementById('lbl-speaker-filter'),
      searchInput: document.getElementById('lbl-search-input'),
      chkAutoPlay: document.getElementById('lbl-chk-autoplay'),
      chkAutoAdvance: document.getElementById('lbl-chk-autoadvance'),
      filterButtons: document.querySelectorAll('.lbl-filter-btn'),
      
      // Cards container & empty state
      turnsContainer: document.getElementById('lbl-turns-list'),
      emptyPlaceholder: document.getElementById('lbl-empty-state'),
      
      // Modals
      shortcutsModal: document.getElementById('lbl-shortcuts-modal'),
      btnShortcutsClose: document.getElementById('btn-lbl-shortcuts-close'),
      exportModal: document.getElementById('lbl-export-modal'),
      btnExportClose: document.getElementById('btn-lbl-export-close'),
      btnConfirmExport: document.getElementById('btn-lbl-confirm-export'),
      exportDatasetName: document.getElementById('lbl-export-name'),
      exportStrategy: document.getElementById('lbl-export-strategy'),
      exportTrainPct: document.getElementById('lbl-export-train-pct'),
      exportValPct: document.getElementById('lbl-export-val-pct'),
      exportTestPct: document.getElementById('lbl-export-test-pct'),
      exportSampleRate: document.getElementById('lbl-export-sr'),
      exportIncludeUnlabeled: document.getElementById('lbl-export-unlabeled'),
      exportProgress: document.getElementById('lbl-export-progress'),
      exportResultBox: document.getElementById('lbl-export-result-box'),
    };
  }

  // Lifecycle hooks exposed to studio
  window.LabelerTab = {
    onTabActivated: async function () {
      if (!dom.tabPane) initDom();
      if (!state.chartRenderer && dom.chartCanvas) {
        state.chartRenderer = new WandbChartRenderer(dom.chartCanvas, dom.chartTooltip, dom.chartHoverInfo, dom.chartLegends);
      }
      await loadResultsList();
      await loadDatasetsForTraining();
      await checkWandbStatus();
      bindGlobalKeyboard();
    },
    onTabDeactivated: function () {
      stopAudio();
      if (state.isDirty) {
        saveDraftImmediate();
      }
      unbindGlobalKeyboard();
    },
  };

  async function loadResultsList() {
    try {
      const res = await fetch('/api/labeler/results');
      if (!res.ok) throw new Error(`Failed loading results: ${res.statusText}`);
      const data = await res.json();
      
      dom.resultSelect.innerHTML = '<option value="">-- Select Diarization Result --</option>';
      (data.results || []).forEach(r => {
        const opt = document.createElement('option');
        opt.value = r.result_id;
        const progressStr = r.labeled_count > 0 ? ` [${r.labeled_count}/${r.turn_count} labeled]` : ` [${r.turn_count} turns]`;
        opt.textContent = `${r.title || r.audio_id || r.result_id}${progressStr}`;
        if (!r.source_available) {
          opt.textContent += ' (Audio missing)';
          opt.disabled = true;
        }
        dom.resultSelect.appendChild(opt);
      });

      // Restore active if available
      if (state.activeResultId) {
        dom.resultSelect.value = state.activeResultId;
      }
    } catch (err) {
      console.error('Error fetching diarization results list:', err);
    }
  }

  async function loadSession(resultId) {
    if (!resultId) {
      state.activeResultId = null;
      state.resultData = null;
      state.labels = {};
      renderTurns();
      updateStats();
      return;
    }

    try {
      dom.saveIndicator.textContent = 'Loading...';
      const res = await fetch(`/api/labeler/session/${encodeURIComponent(resultId)}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      const data = await res.json();

      state.activeResultId = resultId;
      state.resultData = data;
      state.labels = data.labels || {};
      state.activeTurnIndex = 0;

      // Populate speaker filter
      dom.speakerSelect.innerHTML = '<option value="all">All Speakers</option>';
      (data.speakers || []).forEach(spk => {
        const opt = document.createElement('option');
        opt.value = spk;
        opt.textContent = spk;
        dom.speakerSelect.appendChild(opt);
      });

      // Update source info bar
      const src = data.source_audio || {};
      dom.sourceInfoBar.innerHTML = `
        <div class="flex-row items-center gap-3 flex-wrap">
          <span class="badge badge-sm font-mono">${data.audio_id}</span>
          <span class="text-xs text-secondary">${data.turn_count} turns · ${formatDuration(src.duration_s || 0)} · ${src.sample_rate || 44100} Hz</span>
          ${data.source_available ? '<span class="badge badge-sm badge-success">Audio Ready</span>' : '<span class="badge badge-sm badge-destructive">Audio Missing on Disk</span>'}
        </div>
      `;

      dom.saveIndicator.textContent = data.updated_at ? 'Draft loaded' : 'Ready';
      renderTurns();
      updateStats();
    } catch (err) {
      console.error('Failed to load diarization session:', err);
      dom.saveIndicator.textContent = 'Error loading session';
      alert(`Could not load result: ${err.message}`);
    }
  }

  function updateStats() {
    if (!state.resultData) {
      dom.statTotal.textContent = '0';
      dom.statLabeled.textContent = '0';
      dom.statAccept.textContent = '0';
      dom.statNoise.textContent = '0';
      dom.statMulti.textContent = '0';
      dom.statChopped.textContent = '0';
      return;
    }

    const total = state.resultData.turn_count || 0;
    let labeled = 0;
    let accept = 0;
    let noise = 0;
    let multi = 0;
    let chopped = 0;

    Object.values(state.labels).forEach(entry => {
      if (!entry) return;
      const primary = typeof entry === 'string' ? entry : entry.label;
      const tags = (entry && entry.tags) || (primary ? [primary] : []);
      if (primary) labeled++;
      if (primary === 'accept') accept++;
      if (tags.includes('noise')) noise++;
      if (tags.includes('multi_speaker')) multi++;
      if (tags.includes('chopped')) chopped++;
    });

    dom.statTotal.textContent = total;
    dom.statLabeled.textContent = `${labeled} (${total ? Math.round((labeled / total) * 100) : 0}%)`;
    dom.statAccept.textContent = accept;
    dom.statNoise.textContent = noise;
    dom.statMulti.textContent = multi;
    dom.statChopped.textContent = chopped;

    // Update filter badge counts
    document.querySelectorAll('.lbl-filter-btn').forEach(btn => {
      const f = btn.dataset.filter;
      const countEl = btn.querySelector('.badge-count');
      if (!countEl) return;
      if (f === 'all') countEl.textContent = total;
      else if (f === 'unlabeled') countEl.textContent = total - labeled;
      else if (f === 'accept') countEl.textContent = accept;
      else if (f === 'noise') countEl.textContent = noise;
      else if (f === 'multi_speaker') countEl.textContent = multi;
      else if (f === 'chopped') countEl.textContent = chopped;
    });
  }

  function renderTurns() {
    if (!state.resultData || !state.resultData.turns || state.resultData.turns.length === 0) {
      dom.turnsContainer.innerHTML = '';
      dom.emptyPlaceholder.style.display = 'flex';
      return;
    }

    dom.emptyPlaceholder.style.display = 'none';
    const fragment = document.createDocumentFragment();

    state.resultData.turns.forEach(turn => {
      const isVisible = checkTurnVisibility(turn);
      const card = createTurnCard(turn, isVisible);
      fragment.appendChild(card);
    });

    dom.turnsContainer.innerHTML = '';
    dom.turnsContainer.appendChild(fragment);
  }

  function checkTurnVisibility(turn) {
    const entry = state.labels[String(turn.index)];
    const primary = entry ? (typeof entry === 'string' ? entry : entry.label) : null;
    const tags = entry && entry.tags ? entry.tags : (primary ? [primary] : []);

    // Filter by speaker
    if (state.filterSpeaker !== 'all' && turn.speaker_id !== state.filterSpeaker) {
      return false;
    }

    // Filter by label
    if (state.filterLabel === 'unlabeled') {
      if (primary) return false;
    } else if (state.filterLabel === 'accept') {
      if (primary !== 'accept') return false;
    } else if (state.filterLabel === 'noise') {
      if (!tags.includes('noise')) return false;
    } else if (state.filterLabel === 'multi_speaker') {
      if (!tags.includes('multi_speaker')) return false;
    } else if (state.filterLabel === 'chopped') {
      if (!tags.includes('chopped')) return false;
    }

    // Search query
    if (state.searchQuery) {
      const q = state.searchQuery.toLowerCase();
      const matchIdx = String(turn.index) === q;
      const matchSpk = turn.speaker_id.toLowerCase().includes(q);
      const matchNote = entry && entry.notes && entry.notes.toLowerCase().includes(q);
      if (!matchIdx && !matchSpk && !matchNote) return false;
    }

    return true;
  }

  function createTurnCard(turn, isVisible) {
    const card = document.createElement('div');
    card.className = 'lbl-turn-card';
    card.id = `lbl-card-${turn.index}`;
    card.dataset.index = turn.index;
    if (!isVisible) card.style.display = 'none';

    const entry = state.labels[String(turn.index)];
    const primary = entry ? (typeof entry === 'string' ? entry : entry.label) : null;
    const tags = entry && entry.tags ? entry.tags : (primary ? [primary] : []);

    if (primary) {
      card.classList.add(`labeled-${primary}`);
    }
    if (turn.index === state.activeTurnIndex) {
      card.classList.add('active');
    }

    const speakerColor = getSpeakerColor(turn.speaker_id);

    card.innerHTML = `
      <div class="lbl-turn-header">
        <div class="lbl-turn-meta">
          <span class="lbl-turn-idx">#${turn.index + 1}</span>
          <span class="badge badge-sm font-mono" style="background: ${speakerColor}; color: #fff;">${turn.speaker_id}</span>
          <span class="lbl-turn-time">${formatTime(turn.start_s)} → ${formatTime(turn.end_s)}</span>
          <span class="lbl-turn-dur">${turn.duration_s.toFixed(2)}s</span>
          ${turn.overlaps_other_speaker ? '<span class="badge badge-sm badge-destructive" title="Diarizer marked overlap">Overlap</span>' : ''}
        </div>
        <div class="lbl-card-actions">
          <button type="button" class="lbl-clear-btn" title="Clear label (0)" data-turn="${turn.index}">✕ Clear</button>
        </div>
      </div>

      <div class="lbl-turn-audio">
        <button type="button" class="lbl-play-btn" data-turn="${turn.index}" title="Play/Pause (Space)">
          <svg class="icon-play" width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
        </button>
        <div class="lbl-progress-track" data-turn="${turn.index}">
          <div class="lbl-progress-fill" id="lbl-progress-${turn.index}"></div>
        </div>
        <span class="lbl-time-display" id="lbl-time-${turn.index}">0:00 / ${formatTime(turn.duration_s)}</span>
      </div>

      <div class="lbl-actions-row">
        <div class="lbl-buttons-group">
          <button type="button" class="lbl-tag-btn accept ${tags.includes('accept') ? 'selected' : ''}" data-turn="${turn.index}" data-label="accept">
            <span>Accept</span>
            <span class="key-hint">1</span>
          </button>
          <button type="button" class="lbl-tag-btn noise ${tags.includes('noise') ? 'selected' : ''}" data-turn="${turn.index}" data-label="noise">
            <span>Background Noise</span>
            <span class="key-hint">2</span>
          </button>
          <button type="button" class="lbl-tag-btn multi_speaker ${tags.includes('multi_speaker') ? 'selected' : ''}" data-turn="${turn.index}" data-label="multi_speaker">
            <span>&gt;1 Speaker</span>
            <span class="key-hint">3</span>
          </button>
          <button type="button" class="lbl-tag-btn chopped ${tags.includes('chopped') ? 'selected' : ''}" data-turn="${turn.index}" data-label="chopped">
            <span>Word Chopped</span>
            <span class="key-hint">4</span>
          </button>
        </div>
      </div>
    `;

    // Click to select/focus
    card.addEventListener('click', (e) => {
      if (!e.target.closest('button') && !e.target.closest('.lbl-progress-track')) {
        setActiveTurn(turn.index, false);
      }
    });

    // Tag button click
    card.querySelectorAll('.lbl-tag-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const lbl = btn.dataset.label;
        assignLabel(turn.index, lbl, e.shiftKey); // Shift-click allows multi-label toggling
      });
    });

    // Clear button
    const clearBtn = card.querySelector('.lbl-clear-btn');
    if (clearBtn) {
      clearBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        assignLabel(turn.index, null);
      });
    }

    // Play button
    const playBtn = card.querySelector('.lbl-play-btn');
    if (playBtn) {
      playBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        togglePlayTurn(turn.index);
      });
    }

    // Progress track click
    const track = card.querySelector('.lbl-progress-track');
    if (track) {
      track.addEventListener('click', (e) => {
        e.stopPropagation();
        if (state.playingTurnIndex === turn.index) {
          const rect = track.getBoundingClientRect();
          const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
          state.audioPlayer.currentTime = ratio * (state.audioPlayer.duration || turn.duration_s);
        } else {
          togglePlayTurn(turn.index);
        }
      });
    }

    return card;
  }

  function setActiveTurn(index, scroll = true) {
    if (!state.resultData || index < 0 || index >= state.resultData.turns.length) return;
    
    // Remove old active class
    const oldCard = document.getElementById(`lbl-card-${state.activeTurnIndex}`);
    if (oldCard) oldCard.classList.remove('active');

    state.activeTurnIndex = index;
    const newCard = document.getElementById(`lbl-card-${index}`);
    if (newCard) {
      newCard.classList.add('active');
      if (scroll) {
        newCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
    }
  }

  function assignLabel(turnIndex, labelKey, toggle = false) {
    if (!state.resultData) return;
    const key = String(turnIndex);
    const existing = state.labels[key] || {};
    let currentTags = existing.tags ? [...existing.tags] : (existing.label ? [existing.label] : []);

    if (labelKey === null) {
      // Clear
      delete state.labels[key];
    } else if (labelKey === 'accept') {
      // Accept is mutually exclusive with defects
      state.labels[key] = {
        label: 'accept',
        tags: ['accept'],
        notes: existing.notes || '',
        updated_at: Date.now(),
      };
    } else {
      // Defect label (noise, multi_speaker, chopped)
      if (toggle) {
        // Toggle tag
        currentTags = currentTags.filter(t => t !== 'accept');
        if (currentTags.includes(labelKey)) {
          currentTags = currentTags.filter(t => t !== labelKey);
        } else {
          currentTags.push(labelKey);
        }
        if (currentTags.length === 0) {
          delete state.labels[key];
        } else {
          state.labels[key] = {
            label: currentTags[0],
            tags: currentTags,
            notes: existing.notes || '',
            updated_at: Date.now(),
          };
        }
      } else {
        // Direct select single defect
        state.labels[key] = {
          label: labelKey,
          tags: [labelKey],
          notes: existing.notes || '',
          updated_at: Date.now(),
        };
      }
    }

    state.isDirty = true;
    scheduleAutosave();

    // Fast DOM update for current card
    updateCardUi(turnIndex);
    updateStats();

    // Auto-advance and auto-play
    if (labelKey !== null && dom.chkAutoAdvance.checked) {
      const nextIdx = getNextVisibleTurnIndex(turnIndex);
      if (nextIdx !== null) {
        setActiveTurn(nextIdx, true);
        if (dom.chkAutoPlay.checked) {
          playTurn(nextIdx);
        }
      }
    }
  }

  function updateCardUi(turnIndex) {
    const card = document.getElementById(`lbl-card-${turnIndex}`);
    if (!card) return;

    // Reset labeled-* classes
    card.classList.remove('labeled-accept', 'labeled-noise', 'labeled-multi_speaker', 'labeled-chopped');
    card.querySelectorAll('.lbl-tag-btn').forEach(b => b.classList.remove('selected'));

    const entry = state.labels[String(turnIndex)];
    if (!entry) return;

    const primary = typeof entry === 'string' ? entry : entry.label;
    const tags = entry.tags || (primary ? [primary] : []);

    if (primary) {
      card.classList.add(`labeled-${primary}`);
    }
    tags.forEach(tag => {
      const btn = card.querySelector(`.lbl-tag-btn.${tag}`);
      if (btn) btn.classList.add('selected');
    });
  }

  function getNextVisibleTurnIndex(fromIndex) {
    if (!state.resultData) return null;
    const total = state.resultData.turns.length;
    for (let i = fromIndex + 1; i < total; i++) {
      const card = document.getElementById(`lbl-card-${i}`);
      if (card && card.style.display !== 'none') {
        return i;
      }
    }
    return null;
  }

  function getPrevVisibleTurnIndex(fromIndex) {
    if (!state.resultData) return null;
    for (let i = fromIndex - 1; i >= 0; i--) {
      const card = document.getElementById(`lbl-card-${i}`);
      if (card && card.style.display !== 'none') {
        return i;
      }
    }
    return null;
  }

  // Audio Playback
  function togglePlayTurn(turnIndex) {
    if (state.playingTurnIndex === turnIndex && !state.audioPlayer.paused) {
      pauseAudio();
    } else {
      playTurn(turnIndex);
    }
  }

  function playTurn(turnIndex) {
    if (!state.activeResultId) return;
    stopAudio();

    const turn = state.resultData.turns[turnIndex];
    if (!turn) return;

    setActiveTurn(turnIndex, false);

    const url = `/api/labeler/results/${encodeURIComponent(state.activeResultId)}/turns/${turnIndex}/audio`;
    state.playingTurnIndex = turnIndex;
    state.audioPlayer.src = url;
    state.audioPlayer.play().catch(err => {
      console.warn('Playback interrupted or failed:', err);
    });

    const card = document.getElementById(`lbl-card-${turnIndex}`);
    if (card) {
      const playBtn = card.querySelector('.lbl-play-btn');
      if (playBtn) {
        playBtn.classList.add('playing');
        playBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect></svg>';
      }
    }
  }

  function pauseAudio() {
    state.audioPlayer.pause();
    if (state.playingTurnIndex !== null) {
      const card = document.getElementById(`lbl-card-${state.playingTurnIndex}`);
      if (card) {
        const playBtn = card.querySelector('.lbl-play-btn');
        if (playBtn) {
          playBtn.classList.remove('playing');
          playBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>';
        }
      }
    }
  }

  function stopAudio() {
    pauseAudio();
    if (state.playingTurnIndex !== null) {
      const prog = document.getElementById(`lbl-progress-${state.playingTurnIndex}`);
      if (prog) prog.style.width = '0%';
    }
    state.playingTurnIndex = null;
  }

  // Audio Player Event Listeners
  state.audioPlayer.addEventListener('timeupdate', () => {
    if (state.playingTurnIndex === null) return;
    const dur = state.audioPlayer.duration || (state.resultData.turns[state.playingTurnIndex]?.duration_s) || 1;
    const cur = state.audioPlayer.currentTime;
    const pct = Math.min(100, Math.max(0, (cur / dur) * 100));

    const prog = document.getElementById(`lbl-progress-${state.playingTurnIndex}`);
    if (prog) prog.style.width = `${pct}%`;

    const timeEl = document.getElementById(`lbl-time-${state.playingTurnIndex}`);
    if (timeEl) timeEl.textContent = `${formatTime(cur)} / ${formatTime(dur)}`;
  });

  state.audioPlayer.addEventListener('ended', () => {
    stopAudio();
  });

  // Autosave Draft
  function scheduleAutosave() {
    dom.saveIndicator.textContent = 'Unsaved edits...';
    clearTimeout(state.saveTimer);
    state.saveTimer = setTimeout(() => {
      saveDraftImmediate();
    }, 1500);
  }

  async function saveDraftImmediate() {
    if (!state.activeResultId || !state.isDirty) return;
    try {
      dom.saveIndicator.textContent = 'Saving...';
      const res = await fetch(`/api/labeler/session/${encodeURIComponent(state.activeResultId)}/labels`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ labels: state.labels }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      state.isDirty = false;
      dom.saveIndicator.textContent = `Saved at ${new Date().toLocaleTimeString()}`;
    } catch (err) {
      console.error('Failed saving label draft:', err);
      dom.saveIndicator.textContent = 'Error saving draft';
    }
  }

  // Global Keyboard Navigation
  function handleKeyDown(e) {
    if (!dom.tabPane || dom.tabPane.classList.contains('active') === false) return;
    
    // Ignore keystrokes inside text inputs / textareas / modals
    const tag = e.target.tagName.toLowerCase();
    if (tag === 'input' || tag === 'textarea' || tag === 'select' || e.target.isContentEditable) {
      return;
    }

    const cur = state.activeTurnIndex;

    switch (e.key) {
      case '1':
        e.preventDefault();
        assignLabel(cur, 'accept', e.shiftKey);
        break;
      case '2':
        e.preventDefault();
        assignLabel(cur, 'noise', e.shiftKey);
        break;
      case '3':
        e.preventDefault();
        assignLabel(cur, 'multi_speaker', e.shiftKey);
        break;
      case '4':
        e.preventDefault();
        assignLabel(cur, 'chopped', e.shiftKey);
        break;
      case '0':
      case 'Backspace':
      case 'Delete':
        e.preventDefault();
        assignLabel(cur, null);
        break;
      case ' ':
        e.preventDefault();
        togglePlayTurn(cur);
        break;
      case 'j':
      case 'ArrowDown':
        e.preventDefault();
        const next = getNextVisibleTurnIndex(cur);
        if (next !== null) {
          setActiveTurn(next, true);
          if (dom.chkAutoPlay.checked) playTurn(next);
        }
        break;
      case 'k':
      case 'ArrowUp':
        e.preventDefault();
        const prev = getPrevVisibleTurnIndex(cur);
        if (prev !== null) {
          setActiveTurn(prev, true);
          if (dom.chkAutoPlay.checked) playTurn(prev);
        }
        break;
      case 'r':
        e.preventDefault();
        playTurn(cur);
        break;
      case '?':
        e.preventDefault();
        openShortcutsModal();
        break;
    }
  }

  function bindGlobalKeyboard() {
    window.addEventListener('keydown', handleKeyDown);
  }

  function unbindGlobalKeyboard() {
    window.removeEventListener('keydown', handleKeyDown);
  }

  // Modals & Export
  function openShortcutsModal() {
    dom.shortcutsModal.classList.add('open');
  }

  function closeShortcutsModal() {
    dom.shortcutsModal.classList.remove('open');
  }

  function openExportModal() {
    if (!state.activeResultId) {
      alert('Please load a DiarizationResult first.');
      return;
    }
    const defaultName = `tts_quality_${state.resultData.audio_id.replace(/[^\w]/g, '_').toLowerCase()}`;
    dom.exportDatasetName.value = defaultName;
    dom.exportResultBox.style.display = 'none';
    dom.exportProgress.style.display = 'none';
    dom.exportModal.classList.add('open');
  }

  function closeExportModal() {
    dom.exportModal.classList.remove('open');
  }

  async function executeExport() {
    const datasetName = dom.exportDatasetName.value.trim();
    if (!datasetName) {
      alert('Please specify a dataset name.');
      return;
    }

    const payload = {
      result_id: state.activeResultId,
      dataset_name: datasetName,
      split_strategy: dom.exportStrategy.value,
      split_ratios: {
        train: parseFloat(dom.exportTrainPct.value) / 100 || 0.8,
        val: parseFloat(dom.exportValPct.value) / 100 || 0.1,
        test: parseFloat(dom.exportTestPct.value) / 100 || 0.1,
      },
      target_sample_rate: dom.exportSampleRate.value ? parseInt(dom.exportSampleRate.value, 10) : null,
      include_unlabeled: dom.exportIncludeUnlabeled.checked,
      labels_override: state.labels,
    };

    dom.btnConfirmExport.disabled = true;
    dom.exportProgress.style.display = 'block';
    dom.exportProgress.textContent = 'Extracting audio segments and building splits...';

    try {
      const res = await fetch('/api/labeler/export-dataset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || res.statusText);

      dom.exportProgress.style.display = 'none';
      dom.exportResultBox.style.display = 'block';
      dom.exportResultBox.innerHTML = `
        <div class="card p-3" style="background: rgba(16, 185, 129, 0.1); border-color: rgba(16, 185, 129, 0.3);">
          <h4 class="font-bold text-sm text-success">✓ Export Complete!</h4>
          <p class="text-xs text-secondary mt-1">Exported <strong>${data.total_exported}</strong> audio samples into <code>${data.dataset_path}</code></p>
          <div class="flex-row items-center gap-2 mt-3">
            <a href="/api/labeler/datasets/${encodeURIComponent(data.dataset_name)}/download" class="btn btn-sm btn-primary" download>
              ⬇ Download ZIP Bundle
            </a>
            <button type="button" class="btn btn-sm btn-ghost" onclick="document.getElementById('lbl-export-modal').classList.remove('open')">Close</button>
          </div>
        </div>
      `;

      // Update trainer dropdown and select the newly exported dataset
      await loadDatasetsForTraining();
      if (dom.trainDataset) {
        dom.trainDataset.value = data.dataset_name;
      }
    } catch (err) {
      console.error('Export failed:', err);
      dom.exportProgress.textContent = `Export failed: ${err.message}`;
    } finally {
      dom.btnConfirmExport.disabled = false;
    }
  }

  // Utilities
  function formatTime(s) {
    if (!s || isNaN(s)) return '0:00';
    const m = Math.floor(s / 60);
    const sec = (s % 60).toFixed(2);
    const secPad = sec < 10 ? '0' + sec : sec;
    return `${m}:${secPad}`;
  }

  function formatDuration(s) {
    if (!s || isNaN(s)) return '0s';
    const m = Math.floor(s / 60);
    const rem = Math.round(s % 60);
    return m > 0 ? `${m}m ${rem}s` : `${rem}s`;
  }

  const SPEAKER_COLORS = [
    'hsl(190, 90%, 38%)',
    'hsl(158, 70%, 34%)',
    'hsl(38, 92%, 40%)',
    'hsl(348, 83%, 50%)',
    'hsl(270, 75%, 52%)',
    'hsl(205, 90%, 42%)',
    'hsl(84, 80%, 38%)',
    'hsl(22, 90%, 45%)',
  ];

  function getSpeakerColor(spkId) {
    if (!spkId) return SPEAKER_COLORS[0];
    const match = spkId.match(/\d+/);
    const idx = match ? parseInt(match[0], 10) : 0;
    return SPEAKER_COLORS[idx % SPEAKER_COLORS.length];
  }

  // =========================================================================
  // Weights & Biases (W&B) Chart Renderer & Live Telemetry
  // =========================================================================

  async function checkWandbStatus() {
    if (!dom.wandbEnvStatus) return;
    try {
      const res = await fetch('/api/labeler/wandb/status');
      if (!res.ok) return;
      const data = await res.json();
      if (data.has_api_key) {
        dom.wandbEnvStatus.textContent = `🟢 Key Active (${data.api_key_masked})`;
        dom.wandbEnvStatus.className = 'badge badge-sm badge-success';
        if (dom.chkUseWandb) dom.chkUseWandb.checked = true;
      } else {
        dom.wandbEnvStatus.textContent = '🟡 No Key in .env';
        dom.wandbEnvStatus.className = 'badge badge-sm badge-warning';
        if (dom.wandbEnvHint) {
          dom.wandbEnvHint.innerHTML = 'Add <code>WANDB_API_KEY</code> to <code>.env</code> on server for cloud logging';
        }
      }
      if (data.default_project && dom.trainWandbProject) {
        dom.trainWandbProject.value = data.default_project;
      }
    } catch (err) {
      console.warn('Failed checking WandB status:', err);
    }
  }

  class WandbChartRenderer {
    constructor(canvas, tooltipEl, hoverInfoEl, legendsEl) {
      this.canvas = canvas;
      this.ctx = canvas ? canvas.getContext('2d') : null;
      this.tooltip = tooltipEl;
      this.hoverInfo = hoverInfoEl;
      this.legendsEl = legendsEl;
      this.currentMode = 'loss'; // 'loss', 'metrics', or 'steps'
      this.history = [];
      this.stepHistory = [];
      this.hiddenSeries = new Set();
      this.hoverIndex = null;
      this.dpr = window.devicePixelRatio || 1;

      if (this.canvas) {
        this.bindEvents();
        this.renderLegends();
      }
    }

    setMode(mode) {
      this.currentMode = mode;
      this.renderLegends();
      this.draw();
    }

    updateData(history, stepHistory) {
      this.history = history || [];
      this.stepHistory = stepHistory || [];
      this.draw();
    }

    bindEvents() {
      this.canvas.addEventListener('mousemove', (e) => this.handleMouseMove(e));
      this.canvas.addEventListener('mouseleave', () => this.handleMouseLeave());
      window.addEventListener('resize', () => this.draw());
    }

    getSeriesConfig() {
      if (this.currentMode === 'loss') {
        return [
          { key: 'train_loss', label: 'train/loss', color: '#06b6d4' },
          { key: 'val_loss', label: 'val/loss', color: '#f59e0b' },
        ];
      } else if (this.currentMode === 'metrics') {
        return [
          { key: 'clean_accept_acc', label: 'val/clean_accept_accuracy', color: '#10b981' },
          { key: 'noise_f1', label: 'val/noise_f1', color: '#f97316' },
          { key: 'multi_f1', label: 'val/multi_speaker_f1', color: '#a855f7' },
          { key: 'chopped_f1', label: 'val/chopped_f1', color: '#f43f5e' },
        ];
      } else {
        return [
          { key: 'loss', label: 'train/step_loss', color: '#3b82f6' },
        ];
      }
    }

    renderLegends() {
      if (!this.legendsEl) return;
      const seriesList = this.getSeriesConfig();
      let html = '';
      seriesList.forEach(s => {
        const isMuted = this.hiddenSeries.has(s.key);
        html += `
          <div class="lbl-legend-chip ${isMuted ? 'muted' : ''}" data-key="${s.key}">
            <span class="lbl-legend-dot" style="background: ${s.color};"></span>
            <span>${s.label}</span>
          </div>
        `;
      });
      this.legendsEl.innerHTML = html;
      this.legendsEl.querySelectorAll('.lbl-legend-chip').forEach(chip => {
        chip.addEventListener('click', () => {
          const key = chip.dataset.key;
          if (this.hiddenSeries.has(key)) {
            this.hiddenSeries.delete(key);
            chip.classList.remove('muted');
          } else {
            this.hiddenSeries.add(key);
            chip.classList.add('muted');
          }
          this.draw();
        });
      });
    }

    draw() {
      if (!this.canvas || !this.ctx) return;
      const rect = this.canvas.getBoundingClientRect();
      const width = rect.width;
      const height = rect.height;

      // Handle HiDPI
      this.canvas.width = Math.floor(width * this.dpr);
      this.canvas.height = Math.floor(height * this.dpr);
      this.ctx.resetTransform();
      this.ctx.scale(this.dpr, this.dpr);

      const ctx = this.ctx;
      ctx.clearRect(0, 0, width, height);

      const data = this.currentMode === 'steps' ? this.stepHistory : this.history;
      if (!data || data.length === 0) {
        ctx.fillStyle = '#475569';
        ctx.font = '12px var(--font-mono, monospace)';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('Telemetry live curves will appear as training runs...', width / 2, height / 2);
        return;
      }

      const padding = { top: 20, right: 30, bottom: 30, left: 45 };
      const plotW = width - padding.left - padding.right;
      const plotH = height - padding.top - padding.bottom;

      const seriesList = this.getSeriesConfig().filter(s => !this.hiddenSeries.has(s.key));

      // Calculate Min & Max Y
      let minY = Infinity;
      let maxY = -Infinity;

      data.forEach(item => {
        seriesList.forEach(s => {
          const val = item[s.key];
          if (val != null && !isNaN(val)) {
            if (val < minY) minY = val;
            if (val > maxY) maxY = val;
          }
        });
      });

      if (!isFinite(minY) || !isFinite(maxY)) {
        minY = 0;
        maxY = 1;
      }
      if (minY === maxY) {
        minY = Math.max(0, minY - 0.5);
        maxY = maxY + 0.5;
      }
      const yRange = maxY - minY;
      minY = Math.max(0, minY - yRange * 0.05);
      maxY = maxY + yRange * 0.05;

      // Draw Grid Lines & Y Ticks
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
      ctx.lineWidth = 1;
      ctx.fillStyle = '#64748b';
      ctx.font = '10px var(--font-mono, monospace)';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';

      const yTicks = 4;
      for (let i = 0; i <= yTicks; i++) {
        const yFrac = i / yTicks;
        const yVal = minY + (1 - yFrac) * (maxY - minY);
        const yPos = padding.top + yFrac * plotH;

        ctx.beginPath();
        ctx.moveTo(padding.left, yPos);
        ctx.lineTo(width - padding.right, yPos);
        ctx.stroke();

        ctx.fillText(yVal < 1 ? yVal.toFixed(3) : yVal.toFixed(2), padding.left - 8, yPos);
      }

      // Draw X Ticks
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      const nPoints = data.length;
      const xStep = nPoints > 1 ? plotW / (nPoints - 1) : plotW / 2;

      const xTickInterval = Math.max(1, Math.floor(nPoints / 6));
      for (let i = 0; i < nPoints; i += xTickInterval) {
        const xPos = nPoints > 1 ? padding.left + i * xStep : padding.left + plotW / 2;
        const label = this.currentMode === 'steps' ? `#${data[i].step}` : `Ep ${data[i].epoch}`;
        ctx.fillText(label, xPos, height - padding.bottom + 8);
      }

      // Plot Series Lines
      seriesList.forEach(series => {
        ctx.strokeStyle = series.color;
        ctx.lineWidth = 2;
        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';
        ctx.beginPath();

        let hasMoved = false;
        data.forEach((item, idx) => {
          const val = item[series.key];
          if (val == null || isNaN(val)) return;
          const x = nPoints > 1 ? padding.left + idx * xStep : padding.left + plotW / 2;
          const y = padding.top + (1 - (val - minY) / (maxY - minY)) * plotH;

          if (!hasMoved) {
            ctx.moveTo(x, y);
            hasMoved = true;
          } else {
            ctx.lineTo(x, y);
          }
        });
        ctx.stroke();

        // Draw point dots
        ctx.fillStyle = series.color;
        data.forEach((item, idx) => {
          const val = item[series.key];
          if (val == null || isNaN(val)) return;
          const x = nPoints > 1 ? padding.left + idx * xStep : padding.left + plotW / 2;
          const y = padding.top + (1 - (val - minY) / (maxY - minY)) * plotH;

          ctx.beginPath();
          ctx.arc(x, y, 3, 0, Math.PI * 2);
          ctx.fill();
        });
      });

      // Hover Crosshair & Highlights
      if (this.hoverIndex != null && this.hoverIndex >= 0 && this.hoverIndex < nPoints) {
        const hoveredItem = data[this.hoverIndex];
        const hX = nPoints > 1 ? padding.left + this.hoverIndex * xStep : padding.left + plotW / 2;

        // Vertical Guide Line
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.35)';
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(hX, padding.top);
        ctx.lineTo(hX, height - padding.bottom);
        ctx.stroke();
        ctx.setLineDash([]);

        // Highlight Active Series Dots
        seriesList.forEach(series => {
          const val = hoveredItem[series.key];
          if (val == null || isNaN(val)) return;
          const hY = padding.top + (1 - (val - minY) / (maxY - minY)) * plotH;

          ctx.beginPath();
          ctx.arc(hX, hY, 5, 0, Math.PI * 2);
          ctx.fillStyle = series.color;
          ctx.fill();
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = 1.5;
          ctx.stroke();
        });

        // Update Tooltip
        this.renderTooltip(hoveredItem, hX, padding);
      }
    }

    handleMouseMove(e) {
      const data = this.currentMode === 'steps' ? this.stepHistory : this.history;
      if (!data || data.length === 0) return;
      const rect = this.canvas.getBoundingClientRect();
      const padding = { top: 20, right: 30, bottom: 30, left: 45 };
      const plotW = rect.width - padding.left - padding.right;
      const mouseX = e.clientX - rect.left;

      if (mouseX < padding.left || mouseX > rect.width - padding.right) {
        this.handleMouseLeave();
        return;
      }

      const frac = (mouseX - padding.left) / plotW;
      const idx = Math.min(data.length - 1, Math.max(0, Math.round(frac * (data.length - 1))));
      this.hoverIndex = idx;
      this.draw();
    }

    handleMouseLeave() {
      this.hoverIndex = null;
      if (this.tooltip) this.tooltip.style.display = 'none';
      if (this.hoverInfo) this.hoverInfo.textContent = 'Hover chart to inspect epoch metrics';
      this.draw();
    }

    renderTooltip(item, hX, padding) {
      if (!this.tooltip) return;
      const seriesList = this.getSeriesConfig().filter(s => !this.hiddenSeries.has(s.key));
      const title = this.currentMode === 'steps' ? `Step #${item.step} (Epoch ${item.epoch || '?'})` : `Epoch ${item.epoch}`;

      let rows = `<div class="lbl-chart-tooltip-header">${title}</div>`;
      seriesList.forEach(s => {
        const val = item[s.key];
        const displayVal = val != null ? (typeof val === 'number' ? (val < 1 ? val.toFixed(4) : val.toFixed(3)) : val) : '—';
        rows += `
          <div class="lbl-chart-tooltip-row">
            <span class="metric-label"><span class="lbl-legend-dot" style="background:${s.color};"></span>${s.label}</span>
            <span class="metric-val" style="color:${s.color};">${displayVal}</span>
          </div>
        `;
      });

      this.tooltip.innerHTML = rows;
      this.tooltip.style.display = 'block';

      const rect = this.canvas.getBoundingClientRect();
      let left = hX + 12;
      if (left + 190 > rect.width) {
        left = hX - 200;
      }
      this.tooltip.style.left = `${Math.max(10, left)}px`;
      this.tooltip.style.top = `25px`;

      if (this.hoverInfo) {
        this.hoverInfo.textContent = `${title}: ` + seriesList.map(s => `${s.label.split('/').pop()}=${item[s.key] != null ? Number(item[s.key]).toFixed(3) : '—'}`).join(' · ');
      }
    }
  }

  async function loadDatasetsForTraining() {
    if (!dom.trainDataset) return;
    try {
      const res = await fetch('/api/labeler/datasets');
      if (!res.ok) return;
      const data = await res.json();
      const currentVal = dom.trainDataset.value;
      dom.trainDataset.innerHTML = '<option value="__current__">Current Labeled Session (Auto-export split)</option>';
      (data.datasets || []).forEach(d => {
        const opt = document.createElement('option');
        opt.value = d.dataset_name;
        opt.textContent = `${d.dataset_name} (${d.total_samples} samples)`;
        dom.trainDataset.appendChild(opt);
      });
      if (currentVal && Array.from(dom.trainDataset.options).some(o => o.value === currentVal)) {
        dom.trainDataset.value = currentVal;
      }
    } catch (err) {
      console.warn('Failed to load datasets for trainer:', err);
    }
  }

  function toggleTrainer(forceOpen) {
    if (!dom.trainerBody) return;
    if (typeof forceOpen === 'boolean') {
      state.trainerExpanded = forceOpen;
    } else {
      state.trainerExpanded = !state.trainerExpanded;
    }
    dom.trainerBody.style.display = state.trainerExpanded ? 'block' : 'none';
    if (dom.btnTrainerCollapse) {
      dom.btnTrainerCollapse.textContent = state.trainerExpanded ? '▲' : '▼';
    }
  }

  async function startTraining() {
    const datasetVal = dom.trainDataset.value;
    if (datasetVal === '__current__' && !state.activeResultId) {
      alert('Please load a DiarizationResult first or select an exported dataset.');
      return;
    }

    const useWandb = dom.chkUseWandb ? dom.chkUseWandb.checked : false;
    const wandbProject = dom.trainWandbProject ? dom.trainWandbProject.value.trim() : 'tts-quality-classifier';
    const wandbRunName = dom.trainWandbRunName ? dom.trainWandbRunName.value.trim() : '';

    const payload = {
      dataset_name: datasetVal,
      result_id: state.activeResultId,
      backbone: dom.trainBackbone.value,
      finetune_mode: dom.trainMode.value,
      device: dom.trainDevice.value,
      epochs: parseInt(dom.trainEpochs.value, 10) || 15,
      batch_size: parseInt(dom.trainBatchSize.value, 10) || 8,
      lr_backbone: parseFloat(dom.trainLrBackbone.value) || 1e-5,
      lr_head: parseFloat(dom.trainLrHead.value) || 5e-4,
      use_wandb: useWandb,
      wandb_project: wandbProject,
      wandb_run_name: wandbRunName,
      labels_override: state.labels,
    };

    dom.btnStartTrain.disabled = true;
    dom.btnStopTrain.style.display = 'inline-block';
    dom.btnStopTrain.disabled = false;
    if (dom.wandbRunLink) dom.wandbRunLink.style.display = 'none';

    dom.trainStatusBadge.textContent = 'Launching...';
    dom.trainStatusBadge.className = 'badge badge-warning';
    dom.trainTelemetry.style.display = 'block';
    dom.trainTerminal.textContent = '[00:00:00] Submitting training job to compute worker...';
    dom.trainProgressBar.style.width = '0%';
    dom.trainPctText.textContent = '0%';
    dom.trainStepText.textContent = 'Initializing model backbone & feature extractor...';

    // Reset scorecards & charts
    dom.valTrainLoss.textContent = '—';
    dom.valValLoss.textContent = '—';
    dom.valAcceptAcc.textContent = '—';
    dom.valNoiseF1.textContent = '—';
    dom.valMultiF1.textContent = '—';
    dom.valChoppedF1.textContent = '—';
    dom.bestEpochText.textContent = 'Best: —';
    dom.historyTbody.innerHTML = '<tr><td colspan="7" class="text-center text-muted text-xs py-2">Training started...</td></tr>';

    if (state.chartRenderer) {
      state.chartRenderer.updateData([], []);
    }

    state.trainStartTime = Date.now();
    if (state.trainElapsedTimer) clearInterval(state.trainElapsedTimer);
    state.trainElapsedTimer = setInterval(() => {
      if (!state.trainStartTime) return;
      const sec = Math.floor((Date.now() - state.trainStartTime) / 1000);
      dom.trainElapsedTime.textContent = `Elapsed: ${Math.floor(sec / 60)}m ${sec % 60}s`;
    }, 1000);

    try {
      const res = await fetch('/api/labeler/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || res.statusText);

      state.activeTrainTaskId = data.task_id;
      dom.trainStatusBadge.textContent = 'Running';
      dom.trainStatusBadge.className = 'badge badge-accent';

      if (state.trainPollTimer) clearInterval(state.trainPollTimer);
      state.trainPollTimer = setInterval(() => pollTrainingStatus(state.activeTrainTaskId), 1200);
      pollTrainingStatus(state.activeTrainTaskId);
    } catch (err) {
      console.error('Failed to launch training:', err);
      alert(`Training failed to start: ${err.message}`);
      resetTrainingUiState();
    }
  }

  async function pollTrainingStatus(taskId) {
    if (!taskId) return;
    try {
      const res = await fetch(`/api/labeler/train/status/${encodeURIComponent(taskId)}`);
      if (!res.ok) return;
      const resp = await res.json();
      const task = resp.task;
      if (!task) return;

      // Update progress
      const pct = Math.round((task.progress || 0) * 100);
      dom.trainProgressBar.style.width = `${pct}%`;
      dom.trainPctText.textContent = `${pct}%`;
      dom.trainStepText.textContent = task.message || 'Training...';

      // Update status badge
      if (task.status === 'running') {
        const ep = task.data?.epoch || 0;
        const tot = task.params?.epochs || 15;
        dom.trainStatusBadge.textContent = `Epoch ${ep}/${tot}`;
        dom.trainStatusBadge.className = 'badge badge-accent';
      } else if (task.status === 'completed') {
        dom.trainStatusBadge.textContent = 'Completed';
        dom.trainStatusBadge.className = 'badge badge-success';
      } else if (task.status === 'failed') {
        dom.trainStatusBadge.textContent = 'Failed';
        dom.trainStatusBadge.className = 'badge badge-destructive';
      } else if (task.status === 'cancelled') {
        dom.trainStatusBadge.textContent = 'Cancelled';
        dom.trainStatusBadge.className = 'badge badge-muted';
      }

      // Update metrics & history if present
      const d = task.data || {};
      if (d.train_loss != null) dom.valTrainLoss.textContent = Number(d.train_loss).toFixed(4);
      if (d.val_loss != null) dom.valValLoss.textContent = Number(d.val_loss).toFixed(4);
      if (d.clean_accept_acc != null) {
        dom.valAcceptAcc.textContent = `${(Number(d.clean_accept_acc) * 100).toFixed(1)}%`;
      }
      if (d.metrics) {
        if (d.metrics.has_noise) dom.valNoiseF1.textContent = Number(d.metrics.has_noise.f1).toFixed(3);
        if (d.metrics.has_multi_speaker) dom.valMultiF1.textContent = Number(d.metrics.has_multi_speaker.f1).toFixed(3);
        if (d.metrics.is_chopped) dom.valChoppedF1.textContent = Number(d.metrics.is_chopped.f1).toFixed(3);
      }
      if (d.best_epoch) {
        dom.bestEpochText.textContent = `Best: Ep ${d.best_epoch}`;
      }

      // Update interactive canvas charts
      if (state.chartRenderer) {
        state.chartRenderer.updateData(d.history || [], d.step_history || []);
      }

      // Update WandB link if available
      if (d.wandb_url && dom.wandbRunLink) {
        dom.wandbRunLink.href = d.wandb_url;
        dom.wandbRunLink.style.display = 'inline-flex';
      }

      // Update history table
      if (d.history && d.history.length > 0) {
        let rowsHtml = '';
        d.history.forEach(h => {
          const isBest = h.epoch === d.best_epoch;
          rowsHtml += `
            <tr class="${isBest ? 'best-epoch-row' : ''}">
              <td>${h.epoch}${isBest ? ' ⭐' : ''}</td>
              <td>${h.train_loss.toFixed(4)}</td>
              <td>${h.val_loss.toFixed(4)}</td>
              <td class="text-success">${(h.clean_accept_acc * 100).toFixed(1)}%</td>
              <td>${h.noise_f1.toFixed(3)}</td>
              <td>${h.multi_f1.toFixed(3)}</td>
              <td>${h.chopped_f1.toFixed(3)}</td>
            </tr>
          `;
        });
        dom.historyTbody.innerHTML = rowsHtml;
      }

      // Update terminal logs
      if (d.logs && d.logs.length > 0) {
        dom.trainTerminal.textContent = d.logs.join('\n');
        dom.trainTerminal.scrollTop = dom.trainTerminal.scrollHeight;
      }

      // Termination check
      if (['completed', 'failed', 'cancelled'].includes(task.status)) {
        if (state.trainPollTimer) clearInterval(state.trainPollTimer);
        state.trainPollTimer = null;
        if (state.trainElapsedTimer) clearInterval(state.trainElapsedTimer);
        state.trainElapsedTimer = null;
        dom.btnStartTrain.disabled = false;
        dom.btnStopTrain.style.display = 'none';

        if (task.status === 'completed') {
          dom.trainStepText.textContent = '✓ Best model saved with tri-scale boundary pooling head!';
        } else if (task.status === 'failed') {
          dom.trainStepText.textContent = `❌ ${task.error || 'Training failed'}`;
        }
      }
    } catch (err) {
      console.warn('Error polling train status:', err);
    }
  }

  async function cancelTraining() {
    if (!state.activeTrainTaskId) return;
    try {
      dom.btnStopTrain.disabled = true;
      dom.trainStatusBadge.textContent = 'Cancelling...';
      await fetch(`/api/labeler/train/cancel/${encodeURIComponent(state.activeTrainTaskId)}`, {
        method: 'POST',
      });
    } catch (err) {
      console.error('Cancel request failed:', err);
    }
  }

  function resetTrainingUiState() {
    if (state.trainPollTimer) clearInterval(state.trainPollTimer);
    if (state.trainElapsedTimer) clearInterval(state.trainElapsedTimer);
    state.activeTrainTaskId = null;
    state.trainPollTimer = null;
    state.trainElapsedTimer = null;
    dom.btnStartTrain.disabled = false;
    dom.btnStopTrain.style.display = 'none';
    dom.btnStopTrain.disabled = false;
    dom.trainStatusBadge.textContent = 'Idle';
    dom.trainStatusBadge.className = 'badge badge-accent';
  }

  // Trained Models Checkpoints Modal
  async function openModelsModal() {
    dom.modelsModal.classList.add('open');
    dom.modelsListContainer.innerHTML = '<p class="text-xs text-muted">Scanning saved checkpoints...</p>';
    try {
      const res = await fetch('/api/labeler/models');
      if (!res.ok) throw new Error(res.statusText);
      const data = await res.json();
      const models = data.models || [];
      if (models.length === 0) {
        dom.modelsListContainer.innerHTML = `
          <div class="empty-placeholder p-4 text-center">
            <p class="text-xs text-muted">No trained checkpoints found yet. Start training a model above!</p>
          </div>
        `;
        return;
      }

      let html = '';
      models.forEach(m => {
        const cfg = m.config || {};
        const met = m.metrics || {};
        const bestEp = met.best_epoch || 1;
        const cleanAcc = met.clean_accept ? (met.clean_accept.accuracy * 100).toFixed(1) + '%' : 'N/A';
        const noiseF1 = met.has_noise ? met.has_noise.f1.toFixed(3) : 'N/A';
        const multiF1 = met.has_multi_speaker ? met.has_multi_speaker.f1.toFixed(3) : 'N/A';
        const chopF1 = met.is_chopped ? met.is_chopped.f1.toFixed(3) : 'N/A';
        const dateStr = m.created_at ? new Date(m.created_at * 1000).toLocaleString() : '';

        html += `
          <div class="lbl-model-card">
            <div class="lbl-model-header">
              <div>
                <span class="lbl-model-name">${m.model_name}</span>
                <span class="text-xs text-muted ml-2">${dateStr}</span>
              </div>
              <span class="badge badge-sm badge-success">Best Ep: ${bestEp}</span>
            </div>
            <div class="text-xs text-secondary">
              Backbone: <code>${cfg.backbone_id || 'wavlm'}</code> · Mode: <code>${cfg.finetune_mode || 'full'}</code> · Pooling: <code>onset+global+offset (tri-scale)</code>
            </div>
            <div class="lbl-model-metrics-row mt-1">
              <span class="lbl-metric-chip" title="Clean Speech Accept Accuracy">🎯 Clean Acc: <strong>${cleanAcc}</strong></span>
              <span class="lbl-metric-chip" title="Noise Defect F1">🔊 Noise F1: <strong>${noiseF1}</strong></span>
              <span class="lbl-metric-chip" title="Multi-Speaker Defect F1">👥 Multi F1: <strong>${multiF1}</strong></span>
              <span class="lbl-metric-chip" title="Chopped Syllable Defect F1">✂️ Chop F1: <strong>${chopF1}</strong></span>
            </div>
            <div class="flex-row items-center justify-between text-xs text-muted mt-2 pt-2 border-t border-subtle">
              <code class="text-xs" style="max-width: 80%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${m.path}</code>
              <button type="button" class="btn btn-ghost btn-xs" onclick="navigator.clipboard.writeText('${m.path}')" title="Copy path to clipboard">Copy Path</button>
            </div>
          </div>
        `;
      });
      dom.modelsListContainer.innerHTML = html;
    } catch (err) {
      dom.modelsListContainer.innerHTML = `<p class="text-xs text-danger">Failed to load models: ${err.message}</p>`;
    }
  }

  function closeModelsModal() {
    dom.modelsModal.classList.remove('open');
  }

  // Setup Listeners once DOM loads
  document.addEventListener('DOMContentLoaded', () => {
    initDom();

    dom.resultSelect?.addEventListener('change', (e) => {
      loadSession(e.target.value);
    });

    dom.btnReloadResults?.addEventListener('click', () => {
      loadResultsList();
    });

    dom.btnSaveDraft?.addEventListener('click', () => {
      saveDraftImmediate();
    });

    dom.btnExportModal?.addEventListener('click', () => {
      openExportModal();
    });

    dom.btnExportClose?.addEventListener('click', () => {
      closeExportModal();
    });

    dom.btnConfirmExport?.addEventListener('click', () => {
      executeExport();
    });

    dom.btnShortcutsModal?.addEventListener('click', () => {
      openShortcutsModal();
    });

    dom.btnShortcutsClose?.addEventListener('click', () => {
      closeShortcutsModal();
    });

    // Trainer & Models events
    dom.btnToggleTrainer?.addEventListener('click', () => {
      toggleTrainer();
    });
    dom.trainerHeader?.addEventListener('click', (e) => {
      if (e.target.closest('button')) return;
      toggleTrainer();
    });
    dom.btnTrainerCollapse?.addEventListener('click', (e) => {
      e.stopPropagation();
      toggleTrainer();
    });
    dom.btnRefreshDatasets?.addEventListener('click', () => {
      loadDatasetsForTraining();
    });
    dom.btnStartTrain?.addEventListener('click', () => {
      startTraining();
    });
    dom.btnStopTrain?.addEventListener('click', () => {
      cancelTraining();
    });
    dom.btnClearLogs?.addEventListener('click', () => {
      dom.trainTerminal.textContent = '';
    });
    dom.btnOpenModels?.addEventListener('click', () => {
      openModelsModal();
    });
    dom.btnModelsClose?.addEventListener('click', () => {
      closeModelsModal();
    });
    dom.modelsModal?.addEventListener('click', (e) => {
      if (e.target === dom.modelsModal) closeModelsModal();
    });

    // WandB and Charts events
    dom.chkUseWandb?.addEventListener('change', (e) => {
      if (dom.wandbFields) {
        dom.wandbFields.style.opacity = e.target.checked ? '1' : '0.4';
        dom.wandbFields.style.pointerEvents = e.target.checked ? 'auto' : 'none';
      }
    });

    dom.chartTabs?.forEach(btn => {
      btn.addEventListener('click', () => {
        dom.chartTabs.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const mode = btn.dataset.chart;
        if (state.chartRenderer) {
          state.chartRenderer.setMode(mode);
        }
      });
    });

    // Filters
    dom.speakerSelect?.addEventListener('change', (e) => {
      state.filterSpeaker = e.target.value;
      renderTurns();
    });

    dom.searchInput?.addEventListener('input', (e) => {
      state.searchQuery = e.target.value.trim();
      renderTurns();
    });

    dom.filterButtons?.forEach(btn => {
      btn.addEventListener('click', () => {
        dom.filterButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        state.filterLabel = btn.dataset.filter;
        renderTurns();
      });
    });
  });

})();
