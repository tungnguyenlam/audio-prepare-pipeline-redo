/**
 * SonicStudio — Experiment Tab: Zero-Contamination Single-Speaker Diarization
 * Modular ES6 Frontend Controller with Syllable & Boundary Integrity Gate
 */

(function () {
  'use strict';

  const DEFAULT_GEMMA_PROMPT = `Listen to the supplied audio directly. Do not transcribe it.
Judge speaker purity: exactly one human speaker must be audible throughout; reject overlap, sequential second speakers, whispers, distant speech, and intelligible background speech.
Judge word completeness (lẹm chữ): speech must begin and end on complete acoustic word boundaries; reject a clipped initial or final phoneme or syllable. Do not reject a grammatical fragment when its audible words are intact.
Return only the requested structured result and never include a transcript.`;

  const ExperimentTab = {
    currentAudioId: null,
    lastAudioId: null,
    activeTaskId: null,
    taskPollInterval: null,
    previewAudio: new Audio(),
    activePlayingBtn: null,
    lastResult: null,
    turnPreviewGeneration: 0,
    currentBlobUrl: null,

    init() {
      this.bindDOMElements();
      this.bindEvents();
      this.loadDefaults();
      this.probeGemma();
    },

    bindDOMElements() {
      this.el = {
        audioSelect: document.getElementById('exp-audio-select'),
        btnBrowseLibrary: document.getElementById('btn-exp-browse-library'),
        previewPill: document.getElementById('exp-input-preview-pill'),
        previewBtn: document.getElementById('btn-exp-preview-input'),
        trackTitleText: document.getElementById('exp-track-title-text'),
        trackSpecChip: document.getElementById('exp-track-spec-chip'),
        deviceSelect: document.getElementById('exp-device-select'),
        btnReset: document.getElementById('btn-exp-reset'),

        // Stage 1: Asymmetric Sensitivity & Competitor Tripwire
        primaryBackend: document.getElementById('exp-primary-backend'),
        targetOnset: document.getElementById('exp-target-onset'),
        targetOnsetNum: document.getElementById('exp-target-onset-num'),
        targetOnsetValue: document.getElementById('exp-target-onset-val'),
        targetOffset: document.getElementById('exp-target-offset'),
        targetOffsetNum: document.getElementById('exp-target-offset-num'),
        targetOffsetValue: document.getElementById('exp-target-offset-val'),
        competitorOnset: document.getElementById('exp-competitor-onset'),
        competitorOnsetNum: document.getElementById('exp-competitor-onset-num'),
        competitorOnsetValue: document.getElementById('exp-competitor-onset-val'),

        // Stage 2: Dual-Engine Consensus
        enableConsensus: document.getElementById('exp-enable-consensus'),
        secondaryBackend: document.getElementById('exp-secondary-backend'),
        secondaryDevice: document.getElementById('exp-secondary-device'),
        consensusFields: document.getElementById('exp-consensus-fields'),

        // Stage 3: Boundary & Syllable Integrity Gate
        enableCollar: document.getElementById('exp-enable-collar'),
        boundaryCollar: document.getElementById('exp-boundary-collar'),
        boundaryCollarNum: document.getElementById('exp-boundary-collar-num'),
        boundaryCollarValue: document.getElementById('exp-boundary-collar-val'),
        minDuration: document.getElementById('exp-min-duration'),
        minDurationNum: document.getElementById('exp-min-duration-num'),
        minDurationValue: document.getElementById('exp-min-duration-val'),
        transitionExclusion: document.getElementById('exp-transition-exclusion'),
        transitionExclusionNum: document.getElementById('exp-transition-exclusion-num'),
        transitionExclusionValue: document.getElementById('exp-transition-exclusion-val'),
        collarFields: document.getElementById('exp-collar-fields'),

        // Stage 3a: Option A - Context-Aware Handoff Guard
        enableContextCollar: document.getElementById('exp-enable-context-collar'),
        contextCollarFields: document.getElementById('exp-context-collar-fields'),
        handoffRisk: document.getElementById('exp-handoff-risk'),
        handoffRiskNum: document.getElementById('exp-handoff-risk-num'),
        handoffRiskValue: document.getElementById('exp-handoff-risk-val'),
        silenceTail: document.getElementById('exp-silence-tail'),
        silenceTailNum: document.getElementById('exp-silence-tail-num'),
        silenceTailValue: document.getElementById('exp-silence-tail-val'),

        // Stage 3b: Option B - Syllable & Word Forced Alignment Lock
        enableSyllableAlign: document.getElementById('exp-enable-syllable-align'),
        syllableAlignFields: document.getElementById('exp-syllable-align-fields'),
        alignerEngine: document.getElementById('exp-aligner-engine'),
        alignerDevice: document.getElementById('exp-aligner-device'),
        alignerModel: document.getElementById('exp-aligner-model'),
        alignerLang: document.getElementById('exp-aligner-lang'),
        whisperOptions: document.getElementById('exp-whisper-options'),
        alignerEndpoint: document.getElementById('exp-aligner-endpoint'),
        alignerEndpointWrap: document.getElementById('exp-aligner-endpoint-wrap'),

        // Stage 3c: Option C - Micro-Acoustic Energy & RMS Valley Snapping
        enableEnergySnapping: document.getElementById('exp-enable-energy-snapping'),
        energySnappingFields: document.getElementById('exp-energy-snapping-fields'),
        energyWindow: document.getElementById('exp-energy-window'),
        energyWindowNum: document.getElementById('exp-energy-window-num'),
        energyWindowValue: document.getElementById('exp-energy-window-val'),
        energyFloor: document.getElementById('exp-energy-floor'),
        energyFloorNum: document.getElementById('exp-energy-floor-num'),
        energyFloorValue: document.getElementById('exp-energy-floor-val'),

        // Stage 4: Dense WeSpeaker Homogeneity
        enableHomo: document.getElementById('exp-enable-homo'),
        homoDevice: document.getElementById('exp-homo-device'),
        homoSim: document.getElementById('exp-homo-sim'),
        homoSimNum: document.getElementById('exp-homo-sim-num'),
        homoSimValue: document.getElementById('exp-homo-sim-val'),
        homoWin: document.getElementById('exp-homo-win'),
        homoWinNum: document.getElementById('exp-homo-win-num'),
        homoWinValue: document.getElementById('exp-homo-win-val'),
        homoHop: document.getElementById('exp-homo-hop'),
        homoHopNum: document.getElementById('exp-homo-hop-num'),
        homoHopValue: document.getElementById('exp-homo-hop-val'),
        homoFields: document.getElementById('exp-homo-fields'),

        // Stage 5a: Direct-audio quality verifier
        enableGemma: document.getElementById('exp-enable-gemma'),
        gemmaFields: document.getElementById('exp-gemma-fields'),
        gemmaBackend: document.getElementById('exp-gemma-backend'),
        gemmaModelGroup: document.getElementById('exp-gemma-model-group'),
        gemmaEndpointControls: document.getElementById('exp-gemma-endpoint-controls'),
        gemmaEndpoint: document.getElementById('exp-gemma-endpoint'),
        gemmaModel: document.getElementById('exp-gemma-model'),
        gemmaMaxTokens: document.getElementById('exp-gemma-max-tokens'),
        gemmaTimeout: document.getElementById('exp-gemma-timeout'),
        gemmaTimeoutSlider: document.getElementById('exp-gemma-timeout-slider'),
        gemmaTimeoutValue: document.getElementById('exp-gemma-timeout-val'),
        gemmaPrompt: document.getElementById('exp-gemma-prompt'),
        btnGemmaPromptReset: document.getElementById('btn-exp-gemma-prompt-reset'),
        btnGemmaProbe: document.getElementById('btn-exp-gemma-probe'),
        gemmaBadge: document.getElementById('exp-gemma-badge'),
        btnGemmaTest: document.getElementById('btn-exp-gemma-test'),
        gemmaTestOut: document.getElementById('exp-gemma-test-output'),

        // Stage 5b: VibeVoice-ASR
        enableVibeVoice: document.getElementById('exp-enable-vibevoice'),
        vibevoiceModel: document.getElementById('exp-vibevoice-model'),
        vibevoiceDevice: document.getElementById('exp-vibevoice-device'),
        vibevoiceEndpoint: document.getElementById('exp-vibevoice-endpoint'),
        vibevoiceFields: document.getElementById('exp-vibevoice-fields'),
        vibevoiceMaxSec: document.getElementById('exp-vibevoice-max-sec'),
        vibevoiceMaxSecNum: document.getElementById('exp-vibevoice-max-sec-num'),
        vibevoiceMaxSecValue: document.getElementById('exp-vibevoice-max-sec-val'),

        // Action & Progress
        btnRun: document.getElementById('btn-run-experiment'),
        btnCancel: document.getElementById('btn-cancel-experiment'),
        progressWrap: document.getElementById('exp-progress-wrap'),
        progressFill: document.getElementById('exp-progress-fill'),
        progressMsg: document.getElementById('exp-progress-msg'),
        progressPct: document.getElementById('exp-progress-pct'),

        // Results
        resultsCard: document.getElementById('exp-results-card'),
        funnelContainer: document.getElementById('exp-funnel-stages'),
        directAudioAuditCard: document.getElementById('exp-direct-audio-audit-card'),
        directAudioAuditBody: document.getElementById('exp-direct-audio-audit-body'),
        directAudioTotalCost: document.getElementById('exp-direct-audio-total-cost'),
        turnsBody: document.getElementById('exp-turns-body'),
        turnsCountPill: document.getElementById('exp-turns-count-pill'),
        turnsDurationPill: document.getElementById('exp-turns-duration-pill'),
        turnsAvgPill: document.getElementById('exp-turns-avg-pill'),
        turnsSearch: document.getElementById('exp-turns-search'),
        turnsSpeakerFilter: document.getElementById('exp-turns-speaker-filter'),
        tableCard: document.getElementById('exp-table-card'),
        btnExportRttm: document.getElementById('btn-exp-export-rttm'),
        btnExportManifest: document.getElementById('btn-exp-export-manifest'),
      };
    },

    bindEvents() {
      const self = this;

      // Two-way synchronized granular sliders and numeric inputs
      this.bindDualControl({
        slider: this.el.targetOnset,
        numInput: this.el.targetOnsetNum,
        badge: this.el.targetOnsetValue,
        unit: '%',
        multiplier: 100,
        decimals: 3,
        isPercent: true,
      });
      this.bindDualControl({
        slider: this.el.targetOffset,
        numInput: this.el.targetOffsetNum,
        badge: this.el.targetOffsetValue,
        unit: '%',
        multiplier: 100,
        decimals: 3,
        isPercent: true,
      });
      this.bindDualControl({
        slider: this.el.competitorOnset,
        numInput: this.el.competitorOnsetNum,
        badge: this.el.competitorOnsetValue,
        unit: '%',
        multiplier: 100,
        decimals: 3,
        isPercent: true,
      });
      this.bindDualControl({
        slider: this.el.boundaryCollar,
        numInput: this.el.boundaryCollarNum,
        badge: this.el.boundaryCollarValue,
        unit: 's',
        decimals: 2,
      });
      this.bindDualControl({
        slider: this.el.minDuration,
        numInput: this.el.minDurationNum,
        badge: this.el.minDurationValue,
        unit: 's',
        decimals: 2,
      });
      this.bindDualControl({
        slider: this.el.transitionExclusion,
        numInput: this.el.transitionExclusionNum,
        badge: this.el.transitionExclusionValue,
        unit: 's',
        decimals: 2,
      });
      this.bindDualControl({
        slider: this.el.handoffRisk,
        numInput: this.el.handoffRiskNum,
        badge: this.el.handoffRiskValue,
        unit: 's',
        decimals: 2,
      });
      this.bindDualControl({
        slider: this.el.silenceTail,
        numInput: this.el.silenceTailNum,
        badge: this.el.silenceTailValue,
        unit: 's',
        prefix: '+',
        decimals: 2,
      });
      this.bindDualControl({
        slider: this.el.energyWindow,
        numInput: this.el.energyWindowNum,
        badge: this.el.energyWindowValue,
        unit: 's',
        isMs: true,
        decimals: 3,
      });
      this.bindDualControl({
        slider: this.el.energyFloor,
        numInput: this.el.energyFloorNum,
        badge: this.el.energyFloorValue,
        unit: ' dB',
        decimals: 1,
      });
      this.bindDualControl({
        slider: this.el.homoSim,
        numInput: this.el.homoSimNum,
        badge: this.el.homoSimValue,
        unit: '',
        decimals: 3,
      });
      this.bindDualControl({
        slider: this.el.homoWin,
        numInput: this.el.homoWinNum,
        badge: this.el.homoWinValue,
        unit: 's',
        decimals: 2,
      });
      this.bindDualControl({
        slider: this.el.homoHop,
        numInput: this.el.homoHopNum,
        badge: this.el.homoHopValue,
        unit: 's',
        decimals: 2,
      });
      this.bindDualControl({
        slider: this.el.gemmaTimeoutSlider,
        numInput: this.el.gemmaTimeout,
        badge: this.el.gemmaTimeoutValue,
        unit: 's',
        decimals: 0,
      });
      this.bindDualControl({
        slider: this.el.vibevoiceMaxSec,
        numInput: this.el.vibevoiceMaxSecNum,
        badge: this.el.vibevoiceMaxSecValue,
        unit: 's',
        decimals: 2,
      });

      // Toggles
      this.el.enableConsensus?.addEventListener('change', e => {
        if (self.el.consensusFields) self.el.consensusFields.style.display = e.target.checked ? 'block' : 'none';
      });
      this.el.enableCollar?.addEventListener('change', e => {
        if (self.el.collarFields) self.el.collarFields.style.display = e.target.checked ? 'block' : 'none';
      });
      this.el.enableContextCollar?.addEventListener('change', e => {
        if (self.el.contextCollarFields) self.el.contextCollarFields.style.display = e.target.checked ? 'block' : 'none';
      });
      this.el.enableEnergySnapping?.addEventListener('change', e => {
        if (self.el.energySnappingFields) self.el.energySnappingFields.style.display = e.target.checked ? 'block' : 'none';
      });
      this.el.enableSyllableAlign?.addEventListener('change', e => {
        if (self.el.syllableAlignFields) self.el.syllableAlignFields.style.display = e.target.checked ? 'block' : 'none';
      });
      this.el.alignerEngine?.addEventListener('change', e => {
        const isWhisper = e.target.value === 'whisper_timestamped';
        const isRemote = e.target.value === 'remote_whisper';
        if (self.el.whisperOptions) self.el.whisperOptions.style.display = isWhisper ? 'flex' : 'none';
        if (self.el.alignerEndpointWrap) self.el.alignerEndpointWrap.style.display = isRemote ? 'block' : 'none';
      });
      this.el.enableHomo?.addEventListener('change', e => {
        if (self.el.homoFields) self.el.homoFields.style.display = e.target.checked ? 'block' : 'none';
      });
      this.el.enableGemma?.addEventListener('change', e => {
        if (self.el.gemmaFields) self.el.gemmaFields.style.display = e.target.checked ? 'block' : 'none';
        if (e.target.checked) self.syncDirectAudioProvider();
      });
      this.el.gemmaBackend?.addEventListener('change', () => {
        self.syncDirectAudioProvider();
        self.probeGemma();
      });
      this.el.enableVibeVoice?.addEventListener('change', e => {
        if (self.el.vibevoiceFields) self.el.vibevoiceFields.style.display = e.target.checked ? 'block' : 'none';
      });

      // Track selection & preview
      this.el.audioSelect?.addEventListener('change', () => self.onAudioSelected());
      this.el.btnBrowseLibrary?.addEventListener('click', () => {
        if (typeof window.openLibraryModal === 'function') {
          window.openLibraryModal('experiment');
        }
      });
      this.el.previewBtn?.addEventListener('click', () => self.toggleTrackPreview());

      // Gemma probe, test & reset
      this.el.btnGemmaProbe?.addEventListener('click', () => self.probeGemma());
      this.el.btnGemmaTest?.addEventListener('click', () => self.testGemmaLive());
      this.el.btnGemmaPromptReset?.addEventListener('click', () => {
        if (self.el.gemmaPrompt) self.el.gemmaPrompt.value = DEFAULT_GEMMA_PROMPT;
      });

      // Reset Defaults
      this.el.btnReset?.addEventListener('click', () => self.resetToDefaults());

      // Run / Cancel
      this.el.btnRun?.addEventListener('click', () => self.runExperiment());
      this.el.btnCancel?.addEventListener('click', () => self.cancelExperiment());

      // Audio Preview
      this.previewAudio.addEventListener('ended', () => self.stopTurnPreview());
      this.previewAudio.addEventListener('error', (e) => {
        self.stopTurnPreview();
        console.warn('Audio preview element error:', e);
        if (window.showToast) window.showToast('Could not play audio preview', 'error');
      });

      // Surviving turns toolbar filter & search
      this.el.turnsSearch?.addEventListener('input', () => self.filterAndRenderTurns());
      this.el.turnsSpeakerFilter?.addEventListener('change', () => self.filterAndRenderTurns());

      // Export
      this.el.btnExportRttm?.addEventListener('click', () => self.exportRttm());
      this.el.btnExportManifest?.addEventListener('click', () => self.exportManifest());
    },

    bindDualControl({
      slider,
      numInput,
      badge,
      unit = '',
      multiplier = 1,
      prefix = '',
      decimals = 2,
      isPercent = false,
      isMs = false,
      onUpdate = null,
    }) {
      if (!slider && !numInput) return;

      const formatNum = (val) => {
        if (Number.isInteger(val)) return String(val);
        return String(parseFloat(val.toFixed(decimals)));
      };

      const updateBadge = (val) => {
        if (!badge) return;
        if (isPercent) {
          badge.textContent = Math.round(val * 100) + '%';
        } else if (isMs) {
          badge.textContent = `±${Math.round(val * 1000)}ms`;
        } else if (prefix === '+') {
          badge.textContent = `+${val.toFixed(decimals)}s`;
        } else {
          badge.textContent = prefix + (multiplier !== 1 ? Math.round(val * multiplier) : val.toFixed(decimals)) + unit;
        }
      };

      if (slider) {
        slider.addEventListener('input', () => {
          const val = parseFloat(slider.value);
          if (numInput && document.activeElement !== numInput) {
            numInput.value = formatNum(val);
          }
          updateBadge(val);
          if (onUpdate) onUpdate(val);
        });
      }

      if (numInput) {
        numInput.addEventListener('input', () => {
          let val = parseFloat(numInput.value);
          if (isNaN(val)) return;
          if (slider) {
            const sMin = parseFloat(slider.min);
            const sMax = parseFloat(slider.max);
            slider.value = String(Math.min(Math.max(val, sMin), sMax));
          }
          updateBadge(val);
          if (onUpdate) onUpdate(val);
        });

        numInput.addEventListener('change', () => {
          let val = parseFloat(numInput.value);
          if (isNaN(val)) {
            if (slider) numInput.value = formatNum(parseFloat(slider.value));
            return;
          }
          if (isPercent && val > 1.0) {
            val = val / 100;
            numInput.value = formatNum(val);
          }
          if (isMs && val > 2.0) {
            val = val / 1000;
            numInput.value = formatNum(val);
          }
          if (slider) {
            const sMin = parseFloat(slider.min);
            const sMax = parseFloat(slider.max);
            slider.value = String(Math.min(Math.max(val, sMin), sMax));
          }
          updateBadge(val);
          if (onUpdate) onUpdate(val);
        });

        numInput.addEventListener('keydown', (e) => {
          if (e.key === 'Enter') numInput.blur();
        });
      }

      // Initialize badge and input value on binding
      const initialVal = slider ? parseFloat(slider.value) : (numInput ? parseFloat(numInput.value) : NaN);
      if (!isNaN(initialVal)) {
        if (numInput && !numInput.value) numInput.value = formatNum(initialVal);
        updateBadge(initialVal);
      }
    },

    setParamValue(slider, numInput, val) {
      if (slider) slider.value = String(val);
      if (numInput) numInput.value = String(val);
      if (slider) {
        slider.dispatchEvent(new Event('input'));
      } else if (numInput) {
        numInput.dispatchEvent(new Event('input'));
      }
    },

    async loadDefaults() {
      try {
        const res = await fetch('/api/experiment/status');
        if (!res.ok) return;
        const data = await res.json();
        const devices = data.devices || ['cpu', 'cuda:0', 'cuda:1'];

        // Populate per-stage device dropdowns
        this.populateDeviceSelect(this.el.deviceSelect, devices, data.device || 'cuda:0', false);
        this.populateDeviceSelect(this.el.secondaryDevice, devices, 'same', true);
        this.populateDeviceSelect(this.el.alignerDevice, devices, 'cpu', true);
        this.populateDeviceSelect(this.el.homoDevice, devices, 'same', true);
        this.populateDeviceSelect(this.el.vibevoiceDevice, devices, 'same', true);

        if (data.defaults) {
          if (data.defaults.aligner_engine && this.el.alignerEngine) {
            this.el.alignerEngine.value = data.defaults.aligner_engine;
            this.el.alignerEngine.dispatchEvent(new Event('change'));
          }
          if (data.defaults.aligner_model && this.el.alignerModel) {
            this.el.alignerModel.value = data.defaults.aligner_model;
          }
          if (data.defaults.aligner_language && this.el.alignerLang) {
            this.el.alignerLang.value = data.defaults.aligner_language;
          }
        }
      } catch (err) {
        console.warn('Could not load experiment status defaults:', err);
      }
    },

    populateDeviceSelect(selectEl, devices, defaultValue, includeSame = false) {
      if (!selectEl || !Array.isArray(devices)) return;
      const currentVal = selectEl.value || defaultValue;
      selectEl.innerHTML = '';
      if (includeSame) {
        const opt = document.createElement('option');
        opt.value = 'same';
        opt.textContent = 'Same as Primary';
        selectEl.appendChild(opt);
      }
      devices.forEach(dev => {
        const opt = document.createElement('option');
        opt.value = dev;
        if (dev === 'cpu') {
          opt.textContent = selectEl.id === 'exp-aligner-device' ? 'CPU (Recommended - No GPU VRAM risk)' : 'CPU';
        } else {
          opt.textContent = dev;
        }
        selectEl.appendChild(opt);
      });
      if (currentVal && Array.from(selectEl.options).some(o => o.value === currentVal)) {
        selectEl.value = currentVal;
      } else if (defaultValue && Array.from(selectEl.options).some(o => o.value === defaultValue)) {
        selectEl.value = defaultValue;
      }
    },

    onTabActivated() {
      if (window.state && window.state.activeAudio && this.el.audioSelect) {
        if (!this.el.audioSelect.value || this.el.audioSelect.value !== window.state.activeAudio.id) {
          this.el.audioSelect.value = window.state.activeAudio.id;
        }
        this.onAudioSelected();
      }
    },

    async onAudioSelected() {
      const audioId = this.el.audioSelect?.value;
      if (!audioId) {
        if (this.el.previewPill) this.el.previewPill.classList.add('hidden');
        if (this.el.trackSpecChip) this.el.trackSpecChip.textContent = '';
        if (this.el.trackTitleText) this.el.trackTitleText.textContent = '';
        this.currentAudioId = null;
        return;
      }
      if (audioId.startsWith('lib:')) {
        const filePath = audioId.slice(4);
        if (typeof window.loadLibraryFileTo === 'function') {
          await window.loadLibraryFileTo(filePath, 'experiment');
        }
        return;
      }
      this.currentAudioId = audioId;
      const audioItem = window.state && window.state.audioList ? window.state.audioList.find(a => a.id === audioId) : null;
      if (audioItem) {
        if (this.el.previewPill) this.el.previewPill.classList.remove('hidden');
        if (this.el.trackTitleText) this.el.trackTitleText.textContent = audioItem.title || audioItem.id;
        if (this.el.trackSpecChip) {
          const dur = audioItem.duration_s ? `${audioItem.duration_s.toFixed(1)}s` : '';
          const sr = audioItem.sample_rate ? `${audioItem.sample_rate}Hz` : '';
          this.el.trackSpecChip.textContent = [dur, sr].filter(Boolean).join(' • ');
        }
      } else {
        if (this.el.previewPill) this.el.previewPill.classList.remove('hidden');
        if (this.el.trackTitleText) this.el.trackTitleText.textContent = audioId;
        if (this.el.trackSpecChip) this.el.trackSpecChip.textContent = '';
      }
    },

    async toggleTrackPreview() {
      const audioId = this.el.audioSelect?.value || this.currentAudioId;
      if (!audioId) return;

      const isPlayingCurrent = !this.previewAudio.paused && (
        this.previewAudio.src.includes(`/api/audio/${encodeURIComponent(audioId)}/stream`) ||
        (audioId.startsWith('lib:') && this.previewAudio.src.includes(`/api/library/stream?path=${encodeURIComponent(audioId.slice(4))}`))
      );
      if (isPlayingCurrent) {
        this.stopTurnPreview();
        return;
      }

      this.stopTurnPreview();

      const streamUrl = audioId.startsWith('lib:')
        ? `/api/library/stream?path=${encodeURIComponent(audioId.slice(4))}`
        : `/api/audio/${encodeURIComponent(audioId)}/stream`;

      try {
        this.previewAudio.src = streamUrl;
        await this.previewAudio.play();
        if (this.el.previewBtn) this.el.previewBtn.textContent = '⏹ Stop';
      } catch (err) {
        console.warn('Track preview error:', err);
        this.stopTurnPreview();
        if (window.showToast) window.showToast(`Track preview error: ${err.message || 'Playback failed'}`, 'error');
      }
    },

    resetToDefaults() {
      if (this.el.primaryBackend) this.el.primaryBackend.value = 'sortformer';
      this.setParamValue(this.el.targetOnset, this.el.targetOnsetNum, 0.70);
      this.setParamValue(this.el.targetOffset, this.el.targetOffsetNum, 0.50);
      this.setParamValue(this.el.competitorOnset, this.el.competitorOnsetNum, 0.20);
      if (this.el.enableConsensus) { this.el.enableConsensus.checked = true; this.el.consensusFields.style.display = 'block'; }
      if (this.el.secondaryBackend) this.el.secondaryBackend.value = 'diarizen';
      if (this.el.secondaryDevice) this.el.secondaryDevice.value = 'same';
      if (this.el.enableCollar) { this.el.enableCollar.checked = true; this.el.collarFields.style.display = 'block'; }
      this.setParamValue(this.el.boundaryCollar, this.el.boundaryCollarNum, 0.20);
      this.setParamValue(this.el.minDuration, this.el.minDurationNum, 0.60);
      this.setParamValue(this.el.transitionExclusion, this.el.transitionExclusionNum, 0.50);
      // Stage 3a: Option A - Context-Aware Handoff Guard
      if (this.el.enableContextCollar) { this.el.enableContextCollar.checked = true; this.el.contextCollarFields.style.display = 'block'; }
      this.setParamValue(this.el.handoffRisk, this.el.handoffRiskNum, 0.85);
      this.setParamValue(this.el.silenceTail, this.el.silenceTailNum, 0.25);
      // Stage 3b: Option B - Syllable & Word Forced Alignment Lock
      if (this.el.enableSyllableAlign) { this.el.enableSyllableAlign.checked = true; this.el.syllableAlignFields.style.display = 'block'; }
      if (this.el.alignerEngine) {
        this.el.alignerEngine.value = 'whisper_timestamped';
        this.el.alignerEngine.dispatchEvent(new Event('change'));
      }
      if (this.el.alignerModel) this.el.alignerModel.value = 'vinai/PhoWhisper-large';
      if (this.el.alignerLang) this.el.alignerLang.value = 'vi';
      if (this.el.alignerDevice) this.el.alignerDevice.value = 'same';
      // Stage 3c: Option C - Micro-Acoustic Energy & RMS Valley Snapping
      if (this.el.enableEnergySnapping) { this.el.enableEnergySnapping.checked = false; this.el.energySnappingFields.style.display = 'none'; }
      this.setParamValue(this.el.energyWindow, this.el.energyWindowNum, 0.15);
      this.setParamValue(this.el.energyFloor, this.el.energyFloorNum, -30);
      // Stage 4: Dense WeSpeaker Homogeneity
      if (this.el.enableHomo) { this.el.enableHomo.checked = true; this.el.homoFields.style.display = 'block'; }
      if (this.el.homoDevice) this.el.homoDevice.value = 'same';
      this.setParamValue(this.el.homoSim, this.el.homoSimNum, 0.74);
      this.setParamValue(this.el.homoWin, this.el.homoWinNum, 0.80);
      this.setParamValue(this.el.homoHop, this.el.homoHopNum, 0.10);
      // Stage 5a: Direct-audio verifier
      this.setParamValue(this.el.gemmaTimeoutSlider, this.el.gemmaTimeout, 120);
      if (this.el.gemmaPrompt) this.el.gemmaPrompt.value = DEFAULT_GEMMA_PROMPT;
      if (this.el.enableGemma) { this.el.enableGemma.checked = true; this.el.gemmaFields.style.display = 'block'; }
      if (this.el.gemmaBackend) this.el.gemmaBackend.value = 'gemini:gemini-3.8-flash';
      if (this.el.gemmaMaxTokens) this.el.gemmaMaxTokens.value = '256';
      this.syncDirectAudioProvider();
      // Stage 5b: VibeVoice-ASR
      if (this.el.enableVibeVoice) { this.el.enableVibeVoice.checked = false; this.el.vibevoiceFields.style.display = 'none'; }
      if (this.el.vibevoiceModel) this.el.vibevoiceModel.value = 'Dubedo/VibeVoice-ASR-HF-INT8';
      if (this.el.vibevoiceDevice) this.el.vibevoiceDevice.value = 'same';
      this.setParamValue(this.el.vibevoiceMaxSec, this.el.vibevoiceMaxSecNum, 0.00);
      if (window.showToast) window.showToast('Experiment parameters reset to recommended defaults', 'info');
    },

    syncDirectAudioProvider() {
      const isGemini = this.selectedDirectAudioBackend() === 'gemini';
      if (this.el.gemmaModelGroup) this.el.gemmaModelGroup.style.display = isGemini ? 'none' : '';
      if (this.el.gemmaEndpointControls) this.el.gemmaEndpointControls.style.display = isGemini ? 'none' : 'block';
    },

    selectedDirectAudioBackend() {
      return this.el.gemmaBackend?.value?.startsWith('gemini:') ? 'gemini' : 'gemma4';
    },

    selectedDirectAudioModel() {
      const selection = this.el.gemmaBackend?.value || 'gemma4';
      return selection.startsWith('gemini:')
        ? selection.slice('gemini:'.length)
        : (this.el.gemmaModel?.value || 'unsloth/gemma-4-12b-it-GGUF');
    },

    async probeGemma() {
      if (!this.el.gemmaBadge) return;
      this.el.gemmaBadge.className = 'badge badge-sm badge-ghost';
      this.el.gemmaBadge.textContent = 'Pinging…';

      const endpoint = this.el.gemmaEndpoint?.value || 'http://localhost:8888/v1/chat/completions';
      const backend = this.selectedDirectAudioBackend();
      const model = this.selectedDirectAudioModel();

      const t0 = performance.now();
      try {
        const res = await fetch('/api/experiment/direct-audio/probe', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ backend, endpoint, model }),
        });
        const elapsed = Math.round(performance.now() - t0);
        const data = await res.json();
        if (data.ready) {
          this.el.gemmaBadge.className = 'badge badge-sm badge-success';
          this.el.gemmaBadge.textContent = `Connected (${elapsed}ms)`;
          if (this.el.btnGemmaTest) this.el.btnGemmaTest.disabled = false;
        } else {
          this.el.gemmaBadge.className = 'badge badge-sm badge-danger';
          this.el.gemmaBadge.textContent = data.error ? `Error: ${data.error.substring(0, 24)}` : 'Offline / Refused';
        }
      } catch (err) {
        this.el.gemmaBadge.className = 'badge badge-sm badge-danger';
        this.el.gemmaBadge.textContent = 'Unreachable';
      }
    },

    async testGemmaLive() {
      const audioId = this.el.audioSelect?.value || (window.state && window.state.activeAudio && window.state.activeAudio.id);
      if (!audioId) {
        if (window.showToast) window.showToast('Select an audio track first', 'warning');
        return;
      }
      if (this.el.gemmaTestOut) {
        this.el.gemmaTestOut.style.display = 'block';
        this.el.gemmaTestOut.textContent = 'Sending candidate slice to the direct-audio verifier…';
      }

      let start_s = 0.0;
      let end_s = 0.0;
      if (window.state && window.state.selection && window.state.selection.active) {
        start_s = window.state.selection.start;
        end_s = window.state.selection.end;
      }

      try {
        const res = await fetch('/api/experiment/direct-audio/test', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            audio_id: audioId,
            start_s,
            end_s,
            backend: this.selectedDirectAudioBackend(),
            endpoint: this.el.gemmaEndpoint?.value,
            model: this.selectedDirectAudioModel(),
            prompt: this.el.gemmaPrompt?.value,
            max_output_tokens: parseInt(this.el.gemmaMaxTokens?.value || '256', 10),
          }),
        });
        const data = await res.json();
        if (this.el.gemmaTestOut) {
          if (res.ok) {
            const out = data.result;
            const decisionText = out.decision === 'pass' ? '✓ PASS' : `⚠️ ${String(out.decision || 'uncertain').toUpperCase()}`;
            const costText = out.cost?.total_usd !== undefined ? ` • Estimated cost: $${out.cost.total_usd.toFixed(6)}` : '';
            this.el.gemmaTestOut.textContent = `Result: ${decisionText}\n` +
              `Speaker purity: ${out.speaker_purity} • Word completeness: ${out.word_completeness} • Boundary: ${out.boundary_issue}\n` +
              `Failure codes: ${(out.failure_codes || []).join(', ') || 'none'}\n` +
              `Reason: ${out.reason}\n` +
              `Duration: ${out.tested_duration_s}s • Latency: ${out.latency_s}s${costText}`;
          } else {
            this.el.gemmaTestOut.textContent = `Verifier Test Error: ${data.error || 'Server error'}`;
          }
        }
      } catch (err) {
        if (this.el.gemmaTestOut) {
          this.el.gemmaTestOut.textContent = `Network Error: ${err.message}`;
        }
      }
    },

    buildPayload() {
      const audioId = this.el.audioSelect?.value || (window.state && window.state.activeAudio && window.state.activeAudio.id);
      return {
        audio_id: audioId,
        device: this.el.deviceSelect?.value || 'auto',
        primary_device: this.el.deviceSelect?.value || 'auto',
        primary_backend: this.el.primaryBackend?.value || 'sortformer',
        target_onset: parseFloat(this.el.targetOnsetNum?.value || this.el.targetOnset?.value || '0.70'),
        target_offset: parseFloat(this.el.targetOffsetNum?.value || this.el.targetOffset?.value || '0.50'),
        competitor_onset: parseFloat(this.el.competitorOnsetNum?.value || this.el.competitorOnset?.value || '0.20'),
        enable_consensus: Boolean(this.el.enableConsensus?.checked),
        secondary_backend: this.el.secondaryBackend?.value || 'diarizen',
        secondary_device: this.el.secondaryDevice?.value || 'same',
        // Stage 3 Base
        enable_collar_erosion: Boolean(this.el.enableCollar?.checked),
        boundary_collar_s: parseFloat(this.el.boundaryCollarNum?.value || this.el.boundaryCollar?.value || '0.20'),
        min_turn_duration_s: parseFloat(this.el.minDurationNum?.value || this.el.minDuration?.value || '0.60'),
        transition_exclusion_s: parseFloat(this.el.transitionExclusionNum?.value || this.el.transitionExclusion?.value || '0.50'),
        // Stage 3a: Option A - Context-Aware Handoff Guard
        enable_context_collar: Boolean(this.el.enableContextCollar?.checked),
        handoff_risk_distance_s: parseFloat(this.el.handoffRiskNum?.value || this.el.handoffRisk?.value || '0.85'),
        silence_tail_buffer_s: parseFloat(this.el.silenceTailNum?.value || this.el.silenceTail?.value || '0.25'),
        // Stage 3b: Option B - Syllable & Word Forced Alignment Lock
        enable_syllable_alignment: Boolean(this.el.enableSyllableAlign?.checked),
        aligner_engine: this.el.alignerEngine?.value || 'whisper_timestamped',
        aligner_model: this.el.alignerModel?.value || 'vinai/PhoWhisper-large',
        aligner_language: this.el.alignerLang?.value || 'vi',
        aligner_device: this.el.alignerDevice?.value || 'same',
        aligner_endpoint: this.el.alignerEndpoint?.value || '',
        // Stage 3c: Option C - Micro-Acoustic Energy & RMS Valley Snapping
        enable_energy_snapping: Boolean(this.el.enableEnergySnapping?.checked),
        energy_search_window_s: parseFloat(this.el.energyWindowNum?.value || this.el.energyWindow?.value || '0.15'),
        energy_valley_floor_db: parseFloat(this.el.energyFloorNum?.value || this.el.energyFloor?.value || '-30'),
        // Stage 4: Dense WeSpeaker Homogeneity
        enable_homogeneity: Boolean(this.el.enableHomo?.checked),
        homogeneity_device: this.el.homoDevice?.value || 'same',
        min_homogeneity_similarity: parseFloat(this.el.homoSimNum?.value || this.el.homoSim?.value || '0.74'),
        homogeneity_window_s: parseFloat(this.el.homoWinNum?.value || this.el.homoWin?.value || '0.80'),
        homogeneity_hop_s: parseFloat(this.el.homoHopNum?.value || this.el.homoHop?.value || '0.10'),
        // Stage 5a: Direct-Audio Quality Verifier
        enable_gemma: Boolean(this.el.enableGemma?.checked),
        gemma_backend: this.selectedDirectAudioBackend(),
        gemma_endpoint: this.el.gemmaEndpoint?.value,
        gemma_model: this.selectedDirectAudioModel(),
        gemma_prompt: this.el.gemmaPrompt?.value,
        gemma_timeout_s: parseFloat(this.el.gemmaTimeout?.value || this.el.gemmaTimeoutSlider?.value || '120'),
        gemma_max_output_tokens: parseInt(this.el.gemmaMaxTokens?.value || '256', 10),
        // Stage 5b
        enable_vibevoice: Boolean(this.el.enableVibeVoice?.checked),
        vibevoice_model_id: this.el.vibevoiceModel?.value || 'Dubedo/VibeVoice-ASR-HF-INT8',
        vibevoice_device: this.el.vibevoiceDevice?.value || 'same',
        vibevoice_endpoint: this.el.vibevoiceEndpoint?.value || '',
        max_secondary_speech_s: parseFloat(this.el.vibevoiceMaxSecNum?.value || this.el.vibevoiceMaxSec?.value || '0.0'),
      };
    },

    async runExperiment() {
      const payload = this.buildPayload();
      if (!payload.audio_id) {
        if (window.showToast) window.showToast('Please select a target audio track first', 'warning');
        return;
      }
      if (payload.audio_id.startsWith('lib:')) {
        if (window.showToast) window.showToast('Wait for the library track to finish loading, then try again.', 'warning');
        return;
      }
      this.lastAudioId = payload.audio_id;

      this.el.btnRun.disabled = true;
      if (this.el.btnCancel) this.el.btnCancel.style.display = 'inline-block';
      if (this.el.progressWrap) this.el.progressWrap.style.display = 'block';
      this.updateProgress(0.02, 'Queueing zero-contamination experiment…');

      try {
        const res = await fetch('/api/experiment/run', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.error || 'Server error starting experiment');
        }
        const data = await res.json();
        this.activeTaskId = data.task_id;
        this.startPollingTask(data.task_id);
      } catch (err) {
        this.resetActionButtons();
        if (window.showToast) window.showToast(err.message, 'error');
      }
    },

    startPollingTask(taskId) {
      if (this.taskPollInterval) clearInterval(this.taskPollInterval);
      const self = this;

      this.taskPollInterval = setInterval(async () => {
        try {
          const res = await fetch(`/api/tasks/${taskId}`);
          if (!res.ok) return;
          const task = await res.json();

          if (task.progress !== undefined) {
            self.updateProgress(task.progress, task.message || 'Processing on compute device…');
          }

          if (task.status === 'completed') {
            clearInterval(self.taskPollInterval);
            self.finishExperiment(task.result);
          } else if (task.status === 'failed' || task.status === 'cancelled') {
            clearInterval(self.taskPollInterval);
            self.resetActionButtons();
            if (window.showToast) window.showToast(task.message || 'Experiment stopped', 'error');
          }
        } catch (err) {
          console.warn('Poll error:', err);
        }
      }, 750);
    },

    async cancelExperiment() {
      if (!this.activeTaskId) return;
      try {
        await fetch(`/api/tasks/${this.activeTaskId}`, { method: 'DELETE' });
        if (window.showToast) window.showToast('Cancelling experiment…', 'info');
      } catch (err) {
        console.warn('Cancel error:', err);
      }
    },

    updateProgress(pct, msg) {
      const percentage = Math.min(100, Math.round(pct * 100));
      if (this.el.progressFill) this.el.progressFill.style.width = `${percentage}%`;
      if (this.el.progressMsg) this.el.progressMsg.textContent = msg;
      if (this.el.progressPct) this.el.progressPct.textContent = `${percentage}%`;
    },

    resetActionButtons() {
      this.el.btnRun.disabled = false;
      if (this.el.btnCancel) this.el.btnCancel.style.display = 'none';
      if (this.el.progressWrap) this.el.progressWrap.style.display = 'none';
    },

    finishExperiment(result) {
      this.resetActionButtons();
      this.lastResult = result;
      this.lastAudioId = this.el.audioSelect?.value || result.session_audio_id || result.audio_id || this.lastAudioId;
      if (window.showToast) window.showToast('Zero-contamination pipeline completed successfully!', 'success');

      if (this.el.resultsCard) this.el.resultsCard.style.display = 'block';
      if (this.el.tableCard) this.el.tableCard.style.display = 'block';

      this.renderFunnel(result.funnel_stats);
      this.renderDirectAudioAudits(result.foundation_audits || [], result.funnel_stats || {});
      this.renderTurnsTable(result.diarization?.turns || []);
    },

    renderFunnel(funnel) {
      if (!this.el.funnelContainer || !funnel) return;
      const stages = [
        {
          name: '1. Primary Speech',
          val: `${(funnel.initial_speech_duration_s || 0).toFixed(1)}s`,
          sub: `${funnel.initial_turns_count || 0} raw turns`,
        },
      ];

      if (funnel.consensus_speech_duration_s !== undefined) {
        stages.push({
          name: '2. Consensus ∩',
          val: `${(funnel.consensus_speech_duration_s).toFixed(1)}s`,
          sub: `${funnel.consensus_turns_count} mutual turns`,
        });
      }

      if (funnel.eroded_speech_duration_s !== undefined) {
        stages.push({
          name: '3. Boundary Gate',
          val: `${(funnel.eroded_speech_duration_s).toFixed(1)}s`,
          sub: `${funnel.eroded_turns_count} turns`,
        });
      }

      if (funnel.syllables_rescued_count !== undefined && funnel.syllables_rescued_count > 0) {
        stages.push({
          name: 'Syllables Rescued',
          val: `${funnel.syllables_rescued_count} tails`,
          sub: `Avg +${funnel.avg_tail_preservation_ms || 0}ms preserved`,
        });
      }

      if (funnel.homogeneity_speech_duration_s !== undefined) {
        stages.push({
          name: '4. WeSpeaker Homo',
          val: `${(funnel.homogeneity_speech_duration_s).toFixed(1)}s`,
          sub: `${funnel.homogeneity_turns_count} verified turns`,
        });
      }

      if (funnel.foundation_speech_duration_s !== undefined) {
        const cost = funnel.direct_audio_cost?.total_usd;
        stages.push({
          name: '5. Foundation Gate',
          val: `${(funnel.foundation_speech_duration_s).toFixed(1)}s`,
          sub: `${funnel.foundation_turns_count} retained${cost !== undefined ? ` • est. $${cost.toFixed(6)}` : ''}`,
        });
      }

      stages.push({
        name: 'Guaranteed Pure',
        val: `${(funnel.final_pure_speech_duration_s || 0).toFixed(1)}s`,
        sub: `100% Single-Speaker (${funnel.final_pure_turns_count || 0} turns)`,
        final: true,
      });

      this.el.funnelContainer.innerHTML = stages
        .map(
          s => `
        <div class="exp-funnel-stage ${s.final ? 'final' : ''}">
          <div class="exp-funnel-name">${s.name}</div>
          <div class="exp-funnel-val">${s.val}</div>
          <div class="exp-funnel-sub">${s.sub}</div>
        </div>
      `
        )
        .join('');
    },

    escapeHtml(value) {
      return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
    },

    renderDirectAudioAudits(audits, funnel) {
      const rows = audits.filter(a => a.direct_audio || a.direct_audio_error);
      if (!this.el.directAudioAuditCard || !this.el.directAudioAuditBody) return;
      this.el.directAudioAuditCard.style.display = rows.length ? 'block' : 'none';
      if (!rows.length) return;
      const total = funnel.direct_audio_cost?.total_usd;
      if (this.el.directAudioTotalCost) {
        this.el.directAudioTotalCost.textContent = total !== undefined
          ? `Estimated total: $${total.toFixed(6)}`
          : 'Local compute — no API cost';
      }
      this.el.directAudioAuditBody.innerHTML = rows.map(row => {
        const result = row.direct_audio || {};
        const decision = row.direct_audio_error ? 'error' : (result.decision || (row.passed ? 'pass' : 'reject'));
        const badge = decision === 'pass' ? 'badge-success' : (decision === 'reject' ? 'badge-danger' : 'badge-warning');
        const codes = (result.failure_codes || []).join(', ');
        const detail = row.direct_audio_error || result.reason || row.reason || '';
        const reason = codes ? `${codes}: ${detail}` : detail;
        const tokens = result.usage?.total_tokens;
        const sampleCost = result.cost?.total_usd;
        return `<tr>
          <td><code>${Number(row.start_s).toFixed(2)}–${Number(row.end_s).toFixed(2)}s</code></td>
          <td><span class="badge badge-sm ${badge}">${this.escapeHtml(decision)}</span></td>
          <td>${this.escapeHtml(result.speaker_purity || '—')}</td>
          <td>${this.escapeHtml(result.word_completeness || '—')}<br><small>${this.escapeHtml(result.boundary_issue || '')}</small></td>
          <td><small>${this.escapeHtml(reason)}</small></td>
          <td>${tokens !== undefined ? Number(tokens).toLocaleString() : '—'}</td>
          <td>${sampleCost !== undefined ? `$${Number(sampleCost).toFixed(6)}` : '—'}</td>
        </tr>`;
      }).join('');
    },

    renderTurnsTable(turns) {
      this.allTurns = turns || [];
      this.updateSpeakerFilterOptions();
      this.filterAndRenderTurns();
    },

    updateSpeakerFilterOptions() {
      if (!this.el.turnsSpeakerFilter) return;
      const currentVal = this.el.turnsSpeakerFilter.value || 'all';
      const speakers = Array.from(new Set((this.allTurns || []).map(t => t.speaker_id).filter(Boolean))).sort();

      this.el.turnsSpeakerFilter.innerHTML = '<option value="all">All Speakers</option>';
      speakers.forEach(spk => {
        const opt = document.createElement('option');
        opt.value = spk;
        opt.textContent = spk;
        this.el.turnsSpeakerFilter.appendChild(opt);
      });

      if (speakers.includes(currentVal)) {
        this.el.turnsSpeakerFilter.value = currentVal;
      } else {
        this.el.turnsSpeakerFilter.value = 'all';
      }
    },

    filterAndRenderTurns() {
      if (!this.el.turnsBody) return;
      const all = this.allTurns || [];

      const query = (this.el.turnsSearch?.value || '').trim().toLowerCase();
      const speaker = this.el.turnsSpeakerFilter?.value || 'all';

      const filtered = all.filter(t => {
        if (speaker !== 'all' && t.speaker_id !== speaker) return false;
        if (query) {
          const text = (t.transcript || '').toLowerCase();
          const spk = (t.speaker_id || '').toLowerCase();
          const policy = (t.boundary_policy || '').toLowerCase();
          if (!text.includes(query) && !spk.includes(query) && !policy.includes(query)) {
            return false;
          }
        }
        return true;
      });

      // Update Summary Pills
      const count = filtered.length;
      const totalDur = filtered.reduce((acc, t) => acc + (t.end_s - t.start_s), 0);
      const avgDur = count > 0 ? (totalDur / count) : 0;

      if (this.el.turnsCountPill) {
        this.el.turnsCountPill.textContent = `${count} Pure Turn${count === 1 ? '' : 's'}`;
      }
      if (this.el.turnsDurationPill) {
        this.el.turnsDurationPill.textContent = `${totalDur.toFixed(1)}s Total Speech`;
      }
      if (this.el.turnsAvgPill) {
        this.el.turnsAvgPill.textContent = `Avg: ${avgDur.toFixed(1)}s`;
      }

      if (filtered.length === 0) {
        this.el.turnsBody.innerHTML = `
          <tr><td colspan="9" style="text-align: center; color: var(--text-muted); padding: 2.5rem 1rem;">
            ${all.length === 0
              ? 'No turns survived the zero-contamination constraints. Consider slightly relaxing boundary collar erosion or target onset threshold.'
              : 'No turns match the active speaker or transcript search query.'}
          </td></tr>`;
        return;
      }

      const policyMap = {
        context_aware_collar: 'Context Guard',
        acoustic_energy_valley: 'RMS Valley',
        syllable_word_lock: 'Syllable Lock',
        whisper_word_lock: 'Whisper Lock',
        remote_whisper_lock: 'Remote Whisper',
        standard: 'Standard Collar',
      };

      const self = this;
      this.el.turnsBody.innerHTML = filtered
        .map((t, idx) => {
          const dur = (t.end_s - t.start_s).toFixed(2);
          const rawStart = t.raw_start_s !== undefined ? t.raw_start_s : t.start_s;
          const rawEnd = t.raw_end_s !== undefined ? t.raw_end_s : t.end_s;
          const deltaEnd = t.delta_end_ms || 0;

          let deltaBadge = '';
          if (t.tail_rescued || deltaEnd > 0) {
            deltaBadge = `<span class="badge badge-sm badge-success" title="Tail preserved into silence">+${Math.round(deltaEnd)}ms (rescued)</span>`;
          } else if (deltaEnd < 0) {
            deltaBadge = `<span class="badge badge-sm badge-warning" title="Eroded due to neighboring competitor">${Math.round(deltaEnd)}ms (handoff)</span>`;
          } else {
            deltaBadge = `<span class="badge badge-sm badge-ghost">0ms</span>`;
          }

          let policyLabel = policyMap[t.boundary_policy] || t.boundary_policy || 'Standard';
          if (typeof t.boundary_policy === 'string' && t.boundary_policy.startsWith('whisper_lock_')) {
            policyLabel = `Whisper (${t.boundary_policy.replace('whisper_lock_', '')})`;
          }

          const transcriptHtml = t.transcript
            ? `<div class="text-xs text-muted" style="margin-top: 3px; font-style: italic; max-width: 240px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${this.escapeHtml(t.transcript)}">“${this.escapeHtml(t.transcript)}”</div>`
            : '';

          return `
          <tr data-index="${idx}">
            <td class="exp-turn-index">#${idx + 1}</td>
            <td>
              <div style="display: flex; gap: 4px; align-items: center;">
                <button type="button" class="btn btn-ghost btn-xs exp-table-play-blunt" data-start="${rawStart}" data-end="${rawEnd}" title="Play with blunt collar erosion (before syllable rescue)">
                  ▶ Blunt
                </button>
                <button type="button" class="btn btn-primary btn-xs exp-table-play-pure" data-start="${t.start_s}" data-end="${t.end_s}" title="Play with refined boundary (syllable & energy snapped)">
                  ▶ Refined
                </button>
              </div>
            </td>
            <td>
              <span class="exp-speaker-chip">${this.escapeHtml(t.speaker_id)}</span>
              ${transcriptHtml}
            </td>
            <td><code>${t.start_s.toFixed(2)}s</code></td>
            <td><code>${t.end_s.toFixed(2)}s</code></td>
            <td><strong>${dur}s</strong></td>
            <td>${deltaBadge}</td>
            <td><span class="badge badge-sm badge-ghost">${this.escapeHtml(policyLabel)}</span></td>
            <td><span class="badge badge-sm badge-success">Pure Single Speaker</span></td>
          </tr>
        `;
        })
        .join('');

      // Play listeners for both Blunt and Pure buttons
      this.el.turnsBody.querySelectorAll('.exp-table-play-blunt, .exp-table-play-pure').forEach(btn => {
        btn.addEventListener('click', () => {
          const start = parseFloat(btn.dataset.start);
          const end = parseFloat(btn.dataset.end);
          self.togglePlayTurn(start, end, btn);
        });
      });
    },

    async togglePlayTurn(start, end, btn) {
      if (this.activePlayingBtn === btn) {
        this.stopTurnPreview();
        return;
      }
      this.stopTurnPreview();

      const audioId =
        this.lastAudioId ||
        this.lastResult?.session_audio_id ||
        this.lastResult?.audio_id ||
        this.el.audioSelect?.value ||
        this.lastResult?.diarization?.audio_id ||
        (window.state && window.state.activeAudio && window.state.activeAudio.id);
      if (!audioId) {
        if (window.showToast) window.showToast('No audio track found for preview', 'error');
        return;
      }

      this.activePlayingBtn = btn;
      const row = btn.closest('tr');
      if (row) {
        this.activePlayingRow = row;
        row.classList.add('active-playing');
      }
      btn.dataset.prevHtml = btn.innerHTML;
      btn.classList.add('btn-danger', 'is-playing');
      btn.textContent = '⏹ Stop';

      const generation = ++this.turnPreviewGeneration;
      const streamUrl = `/api/audio/${encodeURIComponent(audioId)}/segment?start=${encodeURIComponent(start)}&end=${encodeURIComponent(end)}&inline=1`;

      try {
        const res = await fetch(streamUrl);
        if (!res.ok) {
          let errMsg = `Server returned ${res.status}`;
          try {
            const json = await res.json();
            if (json.error) errMsg = json.error;
          } catch (_) {}
          throw new Error(errMsg);
        }
        const blob = await res.blob();
        if (generation !== this.turnPreviewGeneration || this.activePlayingBtn !== btn) return;

        if (this.currentBlobUrl) {
          URL.revokeObjectURL(this.currentBlobUrl);
          this.currentBlobUrl = null;
        }
        this.currentBlobUrl = URL.createObjectURL(blob);
        this.previewAudio.src = this.currentBlobUrl;
        await this.previewAudio.play();
      } catch (err) {
        if (generation === this.turnPreviewGeneration) {
          console.warn('Turn preview error:', err);
          this.stopTurnPreview();
          if (window.showToast) window.showToast(`Turn preview error: ${err.message}`, 'error');
        }
      }
    },

    stopTurnPreview() {
      this.turnPreviewGeneration = (this.turnPreviewGeneration || 0) + 1;
      if (!this.previewAudio.paused) {
        this.previewAudio.pause();
      }
      this.previewAudio.currentTime = 0;
      if (this.currentBlobUrl) {
        URL.revokeObjectURL(this.currentBlobUrl);
        this.currentBlobUrl = null;
      }
      if (this.activePlayingRow) {
        this.activePlayingRow.classList.remove('active-playing');
        this.activePlayingRow = null;
      }
      if (this.activePlayingBtn) {
        this.activePlayingBtn.classList.remove('btn-danger', 'is-playing');
        if (this.activePlayingBtn.dataset.prevHtml) {
          this.activePlayingBtn.innerHTML = this.activePlayingBtn.dataset.prevHtml;
        }
        this.activePlayingBtn = null;
      }
      if (this.el.previewBtn) {
        this.el.previewBtn.textContent = '▶ Play Track';
      }
    },

    exportRttm() {
      if (!this.lastResult || !this.lastResult.diarization) {
        if (window.showToast) window.showToast('No experiment result available to export', 'warning');
        return;
      }
      const diar = this.lastResult.diarization;
      const audioId = diar.audio_id || 'experiment_audio';
      const lines = (diar.turns || []).map(t => {
        const dur = (t.end_s - t.start_s).toFixed(3);
        return `SPEAKER ${audioId} 1 ${t.start_s.toFixed(3)} ${dur} <NA> <NA> ${t.speaker_id} <NA> <NA>`;
      });
      const blob = new Blob([lines.join('\n') + '\n'], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${audioId}_pure_single_speaker.rttm`;
      a.click();
      URL.revokeObjectURL(url);
    },

    exportManifest() {
      if (!this.lastResult || !this.lastResult.diarization) {
        if (window.showToast) window.showToast('No experiment result available to export', 'warning');
        return;
      }
      const diar = this.lastResult.diarization;
      const audioId = diar.audio_id || 'experiment_audio';
      const lines = (diar.turns || []).map((t, idx) => {
        const dur = Number((t.end_s - t.start_s).toFixed(3));
        return JSON.stringify({
          audio_id: audioId,
          turn_id: t.turn_id || `pure_${String(idx).padStart(4, '0')}`,
          speaker_id: t.speaker_id,
          start: Number(t.start_s.toFixed(3)),
          end: Number(t.end_s.toFixed(3)),
          duration: dur,
          raw_start: t.raw_start_s !== undefined ? Number(t.raw_start_s.toFixed(3)) : Number(t.start_s.toFixed(3)),
          raw_end: t.raw_end_s !== undefined ? Number(t.raw_end_s.toFixed(3)) : Number(t.end_s.toFixed(3)),
          boundary_policy: t.boundary_policy || 'standard',
          tail_rescued: !!t.tail_rescued,
          delta_end_ms: t.delta_end_ms || 0,
          purity_status: 'pure_single_speaker',
        });
      });
      const blob = new Blob([lines.join('\n') + '\n'], { type: 'application/x-ndjson' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${audioId}_pure_tts_manifest.jsonl`;
      a.click();
      URL.revokeObjectURL(url);
    },
  };

  window.ExperimentTab = ExperimentTab;
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => ExperimentTab.init());
  } else {
    ExperimentTab.init();
  }
})();
