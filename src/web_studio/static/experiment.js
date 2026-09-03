/**
 * SonicStudio — Experiment Tab: Zero-Contamination Single-Speaker Diarization
 * Modular ES6 Frontend Controller with Syllable & Boundary Integrity Gate
 */

(function () {
  'use strict';

  const DEFAULT_GEMMA_PROMPT = 'Does this audio contain overlapping speech from two or more speakers at the same time?';

  const ExperimentTab = {
    currentAudioId: null,
    activeTaskId: null,
    taskPollInterval: null,
    previewAudio: new Audio(),
    activePlayingBtn: null,
    lastResult: null,

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

        // Stage 3b: Option B - Micro-Acoustic Energy & RMS Valley Snapping
        enableEnergySnapping: document.getElementById('exp-enable-energy-snapping'),
        energySnappingFields: document.getElementById('exp-energy-snapping-fields'),
        energyWindow: document.getElementById('exp-energy-window'),
        energyWindowNum: document.getElementById('exp-energy-window-num'),
        energyWindowValue: document.getElementById('exp-energy-window-val'),
        energyFloor: document.getElementById('exp-energy-floor'),
        energyFloorNum: document.getElementById('exp-energy-floor-num'),
        energyFloorValue: document.getElementById('exp-energy-floor-val'),

        // Stage 3c: Option C - Syllable & Word Forced Alignment Lock
        enableSyllableAlign: document.getElementById('exp-enable-syllable-align'),
        syllableAlignFields: document.getElementById('exp-syllable-align-fields'),
        alignerEngine: document.getElementById('exp-aligner-engine'),
        alignerDevice: document.getElementById('exp-aligner-device'),
        alignerEndpoint: document.getElementById('exp-aligner-endpoint'),
        alignerEndpointWrap: document.getElementById('exp-aligner-endpoint-wrap'),

        // Stage 4: Dense WeSpeaker Homogeneity
        enableHomo: document.getElementById('exp-enable-homo'),
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

        // Stage 5a: Gemma 4 Remote
        enableGemma: document.getElementById('exp-enable-gemma'),
        gemmaFields: document.getElementById('exp-gemma-fields'),
        gemmaEndpoint: document.getElementById('exp-gemma-endpoint'),
        gemmaModel: document.getElementById('exp-gemma-model'),
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
        turnsBody: document.getElementById('exp-turns-body'),
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
        if (self.el.alignerEndpointWrap) self.el.alignerEndpointWrap.style.display = e.target.value === 'remote_whisper' ? 'block' : 'none';
      });
      this.el.enableHomo?.addEventListener('change', e => {
        if (self.el.homoFields) self.el.homoFields.style.display = e.target.checked ? 'block' : 'none';
      });
      this.el.enableGemma?.addEventListener('change', e => {
        if (self.el.gemmaFields) self.el.gemmaFields.style.display = e.target.checked ? 'block' : 'none';
      });
      this.el.enableVibeVoice?.addEventListener('change', e => {
        if (self.el.vibevoiceFields) self.el.vibevoiceFields.style.display = e.target.checked ? 'block' : 'none';
      });

      // Track selection & preview
      this.el.audioSelect?.addEventListener('change', () => self.onAudioSelected());
      this.el.btnBrowseLibrary?.addEventListener('click', () => {
        if (typeof window.openLibraryModal === 'function') {
          window.openLibraryModal();
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
      this.previewAudio.addEventListener('error', () => {
        self.stopTurnPreview();
        if (window.showToast) window.showToast('Could not play audio preview', 'error');
      });

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
        if (data.device && this.el.deviceSelect) {
          this.el.deviceSelect.value = data.device;
        }
      } catch (err) {
        console.warn('Could not load experiment status defaults:', err);
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

    onAudioSelected() {
      const audioId = this.el.audioSelect?.value;
      if (!audioId) {
        if (this.el.previewPill) this.el.previewPill.classList.add('hidden');
        return;
      }
      const audioItem = window.state && window.state.audioList ? window.state.audioList.find(a => a.id === audioId) : null;
      if (audioItem) {
        if (this.el.previewPill) this.el.previewPill.classList.remove('hidden');
        if (this.el.trackTitleText) this.el.trackTitleText.textContent = audioItem.title || audioItem.id;
        if (this.el.trackSpecChip) {
          const dur = audioItem.duration_s ? `${audioItem.duration_s.toFixed(1)}s` : '';
          const sr = audioItem.sample_rate ? `${audioItem.sample_rate}Hz` : '';
          this.el.trackSpecChip.textContent = [dur, sr].filter(Boolean).join(' • ');
        }
      }
    },

    toggleTrackPreview() {
      const audioId = this.el.audioSelect?.value;
      if (!audioId) return;
      if (!this.previewAudio.paused && this.previewAudio.src.includes(`/api/audio/${audioId}/stream`)) {
        this.previewAudio.pause();
        if (this.el.previewBtn) this.el.previewBtn.textContent = '▶ Play Track';
        return;
      }
      this.stopTurnPreview();
      this.previewAudio.src = `/api/audio/${audioId}/stream`;
      this.previewAudio.play().then(() => {
        if (this.el.previewBtn) this.el.previewBtn.textContent = '⏹ Stop';
      }).catch(err => {
        console.warn('Playback error:', err);
        if (this.el.previewBtn) this.el.previewBtn.textContent = '▶ Play Track';
      });
    },

    resetToDefaults() {
      if (this.el.primaryBackend) this.el.primaryBackend.value = 'sortformer';
      this.setParamValue(this.el.targetOnset, this.el.targetOnsetNum, 0.80);
      this.setParamValue(this.el.targetOffset, this.el.targetOffsetNum, 0.65);
      this.setParamValue(this.el.competitorOnset, this.el.competitorOnsetNum, 0.20);
      if (this.el.enableConsensus) { this.el.enableConsensus.checked = true; this.el.consensusFields.style.display = 'block'; }
      if (this.el.secondaryBackend) this.el.secondaryBackend.value = 'diarizen';
      if (this.el.enableCollar) { this.el.enableCollar.checked = true; this.el.collarFields.style.display = 'block'; }
      this.setParamValue(this.el.boundaryCollar, this.el.boundaryCollarNum, 0.35);
      this.setParamValue(this.el.minDuration, this.el.minDurationNum, 0.80);
      this.setParamValue(this.el.transitionExclusion, this.el.transitionExclusionNum, 0.50);
      // Option A
      if (this.el.enableContextCollar) { this.el.enableContextCollar.checked = true; this.el.contextCollarFields.style.display = 'block'; }
      this.setParamValue(this.el.handoffRisk, this.el.handoffRiskNum, 0.80);
      this.setParamValue(this.el.silenceTail, this.el.silenceTailNum, 0.15);
      // Option B
      if (this.el.enableEnergySnapping) { this.el.enableEnergySnapping.checked = false; this.el.energySnappingFields.style.display = 'none'; }
      this.setParamValue(this.el.energyWindow, this.el.energyWindowNum, 0.15);
      this.setParamValue(this.el.energyFloor, this.el.energyFloorNum, -30);
      // Option C
      if (this.el.enableSyllableAlign) { this.el.enableSyllableAlign.checked = false; this.el.syllableAlignFields.style.display = 'none'; }
      if (this.el.alignerEngine) this.el.alignerEngine.value = 'mms_fa';
      // Stage 4
      if (this.el.enableHomo) { this.el.enableHomo.checked = false; this.el.homoFields.style.display = 'none'; }
      this.setParamValue(this.el.homoSim, this.el.homoSimNum, 0.75);
      this.setParamValue(this.el.homoWin, this.el.homoWinNum, 1.00);
      this.setParamValue(this.el.homoHop, this.el.homoHopNum, 0.25);
      // Stage 5a
      this.setParamValue(this.el.gemmaTimeoutSlider, this.el.gemmaTimeout, 120);
      if (this.el.gemmaPrompt) this.el.gemmaPrompt.value = DEFAULT_GEMMA_PROMPT;
      // Stage 5b
      if (this.el.enableVibeVoice) { this.el.enableVibeVoice.checked = false; this.el.vibevoiceFields.style.display = 'none'; }
      if (this.el.vibevoiceModel) this.el.vibevoiceModel.value = 'Dubedo/VibeVoice-ASR-HF-INT8';
      if (this.el.vibevoiceDevice) this.el.vibevoiceDevice.value = 'same';
      this.setParamValue(this.el.vibevoiceMaxSec, this.el.vibevoiceMaxSecNum, 0.00);
      if (window.showToast) window.showToast('Experiment parameters reset to recommended defaults', 'info');
    },

    async probeGemma() {
      if (!this.el.gemmaBadge) return;
      this.el.gemmaBadge.className = 'badge badge-sm badge-ghost';
      this.el.gemmaBadge.textContent = 'Pinging…';

      const endpoint = this.el.gemmaEndpoint?.value || 'http://localhost:8888/v1/chat/completions';
      const model = this.el.gemmaModel?.value || 'unsloth/gemma-4-12b-it-GGUF';

      const t0 = performance.now();
      try {
        const res = await fetch('/api/experiment/gemma/probe', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ endpoint, model }),
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
        this.el.gemmaTestOut.textContent = 'Sending candidate slice to remote Gemma 4 server…';
      }

      let start_s = 0.0;
      let end_s = 0.0;
      if (window.state && window.state.selection && window.state.selection.active) {
        start_s = window.state.selection.start;
        end_s = window.state.selection.end;
      }

      try {
        const res = await fetch('/api/experiment/gemma/test', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            audio_id: audioId,
            start_s,
            end_s,
            endpoint: this.el.gemmaEndpoint?.value,
            model: this.el.gemmaModel?.value,
            prompt: this.el.gemmaPrompt?.value,
          }),
        });
        const data = await res.json();
        if (this.el.gemmaTestOut) {
          if (res.ok) {
            const out = data.result;
            const overlapText = out.overlap ? '⚠️ OVERLAP DETECTED (Multiple Voices)' : '✓ PURE SINGLE SPEAKER';
            this.el.gemmaTestOut.textContent = `Result: ${overlapText}\n` +
              `Reason: ${out.reason}\n` +
              `Duration: ${out.tested_duration_s}s • Latency: ${out.latency_s}s`;
          } else {
            this.el.gemmaTestOut.textContent = `Gemma Test Error: ${data.error || 'Server error'}`;
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
        primary_backend: this.el.primaryBackend?.value || 'sortformer',
        target_onset: parseFloat(this.el.targetOnsetNum?.value || this.el.targetOnset?.value || '0.80'),
        target_offset: parseFloat(this.el.targetOffsetNum?.value || this.el.targetOffset?.value || '0.65'),
        competitor_onset: parseFloat(this.el.competitorOnsetNum?.value || this.el.competitorOnset?.value || '0.20'),
        enable_consensus: Boolean(this.el.enableConsensus?.checked),
        secondary_backend: this.el.secondaryBackend?.value || 'diarizen',
        // Stage 3 Base
        enable_collar_erosion: Boolean(this.el.enableCollar?.checked),
        boundary_collar_s: parseFloat(this.el.boundaryCollarNum?.value || this.el.boundaryCollar?.value || '0.35'),
        min_turn_duration_s: parseFloat(this.el.minDurationNum?.value || this.el.minDuration?.value || '0.80'),
        transition_exclusion_s: parseFloat(this.el.transitionExclusionNum?.value || this.el.transitionExclusion?.value || '0.50'),
        // Stage 3a: Option A
        enable_context_collar: Boolean(this.el.enableContextCollar?.checked),
        handoff_risk_distance_s: parseFloat(this.el.handoffRiskNum?.value || this.el.handoffRisk?.value || '0.80'),
        silence_tail_buffer_s: parseFloat(this.el.silenceTailNum?.value || this.el.silenceTail?.value || '0.15'),
        // Stage 3b: Option B
        enable_energy_snapping: Boolean(this.el.enableEnergySnapping?.checked),
        energy_search_window_s: parseFloat(this.el.energyWindowNum?.value || this.el.energyWindow?.value || '0.15'),
        energy_valley_floor_db: parseFloat(this.el.energyFloorNum?.value || this.el.energyFloor?.value || '-30'),
        // Stage 3c: Option C
        enable_syllable_alignment: Boolean(this.el.enableSyllableAlign?.checked),
        aligner_engine: this.el.alignerEngine?.value || 'mms_fa',
        aligner_device: this.el.alignerDevice?.value || 'auto',
        aligner_endpoint: this.el.alignerEndpoint?.value || '',
        // Stage 4
        enable_homogeneity: Boolean(this.el.enableHomo?.checked),
        min_homogeneity_similarity: parseFloat(this.el.homoSimNum?.value || this.el.homoSim?.value || '0.75'),
        homogeneity_window_s: parseFloat(this.el.homoWinNum?.value || this.el.homoWin?.value || '1.00'),
        homogeneity_hop_s: parseFloat(this.el.homoHopNum?.value || this.el.homoHop?.value || '0.25'),
        // Stage 5a
        enable_gemma: Boolean(this.el.enableGemma?.checked),
        gemma_endpoint: this.el.gemmaEndpoint?.value,
        gemma_model: this.el.gemmaModel?.value,
        gemma_prompt: this.el.gemmaPrompt?.value,
        gemma_timeout_s: parseFloat(this.el.gemmaTimeout?.value || this.el.gemmaTimeoutSlider?.value || '120'),
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
      if (window.showToast) window.showToast('Zero-contamination pipeline completed successfully!', 'success');

      if (this.el.resultsCard) this.el.resultsCard.style.display = 'block';
      if (this.el.tableCard) this.el.tableCard.style.display = 'block';

      this.renderFunnel(result.funnel_stats);
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
        stages.push({
          name: '5. Foundation Gate',
          val: `${(funnel.foundation_speech_duration_s).toFixed(1)}s`,
          sub: `${funnel.foundation_turns_count} audited turns`,
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

    renderTurnsTable(turns) {
      if (!this.el.turnsBody) return;
      if (!turns || turns.length === 0) {
        this.el.turnsBody.innerHTML = `
          <tr><td colspan="8" style="text-align: center; color: var(--text-muted); padding: 2.5rem 1rem;">
            No turns survived the zero-contamination constraints. Consider slightly relaxing boundary collar erosion or target onset threshold.
          </td></tr>`;
        return;
      }

      const policyMap = {
        context_aware_collar: 'Context Guard',
        acoustic_energy_valley: 'RMS Valley',
        syllable_word_lock: 'Syllable Lock',
        standard: 'Standard Collar',
      };

      const self = this;
      this.el.turnsBody.innerHTML = turns
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

          const policyLabel = policyMap[t.boundary_policy] || t.boundary_policy || 'Standard';

          return `
          <tr data-index="${idx}">
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
            <td><span class="exp-speaker-chip">${t.speaker_id}</span></td>
            <td><code>${t.start_s.toFixed(2)}s</code></td>
            <td><code>${t.end_s.toFixed(2)}s</code></td>
            <td><strong>${dur}s</strong></td>
            <td>${deltaBadge}</td>
            <td><span class="badge badge-sm badge-ghost">${policyLabel}</span></td>
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

    togglePlayTurn(start, end, btn) {
      if (this.activePlayingBtn === btn) {
        this.stopTurnPreview();
        return;
      }
      this.stopTurnPreview();

      const audioId =
        this.lastResult?.diarization?.audio_id ||
        this.el.audioSelect?.value ||
        (window.state && window.state.activeAudio && window.state.activeAudio.id);
      if (!audioId) return;

      this.activePlayingBtn = btn;
      btn.dataset.prevHtml = btn.innerHTML;
      btn.classList.add('btn-danger');
      btn.textContent = '⏹ Stop';

      const streamUrl = `/api/audio/${encodeURIComponent(audioId)}/segment?start=${start}&end=${end}`;
      this.previewAudio.src = streamUrl;
      this.previewAudio.play().catch(err => {
        console.warn('Playback error:', err);
        this.stopTurnPreview();
      });
    },

    stopTurnPreview() {
      this.previewAudio.pause();
      this.previewAudio.currentTime = 0;
      if (this.activePlayingBtn) {
        this.activePlayingBtn.classList.remove('btn-danger');
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
