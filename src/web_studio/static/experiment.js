/**
 * SonicStudio — Experiment Tab: Zero-Contamination Single-Speaker Diarization
 * Modular ES6 Frontend Controller
 */

(function () {
  'use strict';

  const DEFAULT_GEMMA_PROMPT = 'Does this audio contain overlapping speech from two or more speakers at the same time?';

  const ExperimentTab = {
    currentAudioId: null,
    activeTaskId: null,
    taskPollInterval: null,
    previewAudio: new Audio(),
    activePlayingTurnIndex: null,
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

        // Stage 1
        primaryBackend: document.getElementById('exp-primary-backend'),
        targetOnset: document.getElementById('exp-target-onset'),
        targetOnsetValue: document.getElementById('exp-target-onset-val'),
        competitorOnset: document.getElementById('exp-competitor-onset'),
        competitorOnsetValue: document.getElementById('exp-competitor-onset-val'),

        // Stage 2
        enableConsensus: document.getElementById('exp-enable-consensus'),
        secondaryBackend: document.getElementById('exp-secondary-backend'),
        consensusFields: document.getElementById('exp-consensus-fields'),

        // Stage 3
        enableCollar: document.getElementById('exp-enable-collar'),
        boundaryCollar: document.getElementById('exp-boundary-collar'),
        boundaryCollarValue: document.getElementById('exp-boundary-collar-val'),
        minDuration: document.getElementById('exp-min-duration'),
        minDurationValue: document.getElementById('exp-min-duration-val'),
        transExclusion: document.getElementById('exp-trans-exclusion'),
        transExclusionValue: document.getElementById('exp-trans-exclusion-val'),
        collarFields: document.getElementById('exp-collar-fields'),

        // Stage 4
        enableHomo: document.getElementById('exp-enable-homo'),
        homoSim: document.getElementById('exp-homo-sim'),
        homoSimValue: document.getElementById('exp-homo-sim-val'),
        homoWin: document.getElementById('exp-homo-win'),
        homoWinValue: document.getElementById('exp-homo-win-val'),
        homoFields: document.getElementById('exp-homo-fields'),

        // Stage 5a: Gemma 4 Remote
        enableGemma: document.getElementById('exp-enable-gemma'),
        gemmaFields: document.getElementById('exp-gemma-fields'),
        gemmaEndpoint: document.getElementById('exp-gemma-endpoint'),
        gemmaModel: document.getElementById('exp-gemma-model'),
        gemmaTimeout: document.getElementById('exp-gemma-timeout'),
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

        // Action
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
      };
    },

    bindEvents() {
      const self = this;

      // Sliders dynamic values
      this.bindSlider(this.el.targetOnset, this.el.targetOnsetValue, '%', 100);
      this.bindSlider(this.el.competitorOnset, this.el.competitorOnsetValue, '%', 100);
      this.bindSlider(this.el.boundaryCollar, this.el.boundaryCollarValue, 's');
      this.bindSlider(this.el.minDuration, this.el.minDurationValue, 's');
      this.bindSlider(this.el.transExclusion, this.el.transExclusionValue, 's');
      this.bindSlider(this.el.homoSim, this.el.homoSimValue, '');
      this.bindSlider(this.el.homoWin, this.el.homoWinValue, 's');

      // Toggles
      this.el.enableConsensus?.addEventListener('change', e => {
        if (self.el.consensusFields) self.el.consensusFields.style.display = e.target.checked ? 'block' : 'none';
      });
      this.el.enableCollar?.addEventListener('change', e => {
        if (self.el.collarFields) self.el.collarFields.style.display = e.target.checked ? 'block' : 'none';
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
    },

    bindSlider(slider, label, unit, multiplier = 1) {
      if (!slider || !label) return;
      slider.addEventListener('input', () => {
        const val = parseFloat(slider.value) * multiplier;
        label.textContent = (multiplier === 100 ? Math.round(val) : val.toFixed(2)) + unit;
      });
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
      if (this.el.targetOnset) { this.el.targetOnset.value = '0.80'; this.el.targetOnsetValue.textContent = '80%'; }
      if (this.el.competitorOnset) { this.el.competitorOnset.value = '0.20'; this.el.competitorOnsetValue.textContent = '20%'; }
      if (this.el.enableConsensus) { this.el.enableConsensus.checked = true; this.el.consensusFields.style.display = 'block'; }
      if (this.el.secondaryBackend) this.el.secondaryBackend.value = 'diarizen';
      if (this.el.enableCollar) { this.el.enableCollar.checked = true; this.el.collarFields.style.display = 'block'; }
      if (this.el.boundaryCollar) { this.el.boundaryCollar.value = '0.35'; this.el.boundaryCollarValue.textContent = '0.35s'; }
      if (this.el.minDuration) { this.el.minDuration.value = '0.80'; this.el.minDurationValue.textContent = '0.80s'; }
      if (this.el.transExclusion) { this.el.transExclusion.value = '0.50'; this.el.transExclusionValue.textContent = '0.50s'; }
      if (this.el.enableHomo) { this.el.enableHomo.checked = false; this.el.homoFields.style.display = 'none'; }
      if (this.el.homoSim) { this.el.homoSim.value = '0.75'; this.el.homoSimValue.textContent = '0.75'; }
      if (this.el.homoWin) { this.el.homoWin.value = '1.00'; this.el.homoWinValue.textContent = '1.00s'; }
      if (this.el.gemmaPrompt) this.el.gemmaPrompt.value = DEFAULT_GEMMA_PROMPT;
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
        target_onset: parseFloat(this.el.targetOnset?.value || '0.80'),
        target_offset: 0.65,
        competitor_onset: parseFloat(this.el.competitorOnset?.value || '0.20'),
        enable_consensus: Boolean(this.el.enableConsensus?.checked),
        secondary_backend: this.el.secondaryBackend?.value || 'diarizen',
        enable_collar_erosion: Boolean(this.el.enableCollar?.checked),
        boundary_collar_s: parseFloat(this.el.boundaryCollar?.value || '0.35'),
        min_turn_duration_s: parseFloat(this.el.minDuration?.value || '0.80'),
        transition_exclusion_s: parseFloat(this.el.transExclusion?.value || '0.50'),
        enable_homogeneity: Boolean(this.el.enableHomo?.checked),
        min_homogeneity_similarity: parseFloat(this.el.homoSim?.value || '0.75'),
        homogeneity_window_s: parseFloat(this.el.homoWin?.value || '1.00'),
        // Remote Gemma 4
        enable_gemma: Boolean(this.el.enableGemma?.checked),
        gemma_endpoint: this.el.gemmaEndpoint?.value,
        gemma_model: this.el.gemmaModel?.value,
        gemma_prompt: this.el.gemmaPrompt?.value,
        gemma_timeout_s: parseFloat(this.el.gemmaTimeout?.value || '120'),
        // Remote or Dedicated GPU VibeVoice
        enable_vibevoice: Boolean(this.el.enableVibeVoice?.checked),
        vibevoice_model_id: this.el.vibevoiceModel?.value || 'Dubedo/VibeVoice-ASR-HF-INT8',
        vibevoice_device: this.el.vibevoiceDevice?.value || 'same',
        vibevoice_endpoint: this.el.vibevoiceEndpoint?.value || '',
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
          name: '3. Collar Eroded',
          val: `${(funnel.eroded_speech_duration_s).toFixed(1)}s`,
          sub: `${funnel.eroded_turns_count} shaved turns`,
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
          <tr><td colspan="6" style="text-align: center; color: var(--text-muted); padding: 2.5rem 1rem;">
            No turns survived the zero-contamination constraints. Consider slightly relaxing boundary collar erosion or target onset threshold.
          </td></tr>`;
        return;
      }

      const self = this;
      this.el.turnsBody.innerHTML = turns
        .map((t, idx) => {
          const dur = (t.end_s - t.start_s).toFixed(2);
          return `
          <tr data-index="${idx}">
            <td>
              <button class="btn btn-secondary btn-xs exp-table-play-btn" data-index="${idx}" data-start="${t.start_s}" data-end="${t.end_s}">
                ▶ Play
              </button>
            </td>
            <td><span class="exp-speaker-chip">${t.speaker_id}</span></td>
            <td><code>${t.start_s.toFixed(2)}s</code></td>
            <td><code>${t.end_s.toFixed(2)}s</code></td>
            <td><strong>${dur}s</strong></td>
            <td><span class="badge badge-sm badge-success">Pure Single Speaker</span></td>
          </tr>
        `;
        })
        .join('');

      // Play listeners
      this.el.turnsBody.querySelectorAll('.exp-table-play-btn').forEach(btn => {
        btn.addEventListener('click', () => {
          const idx = parseInt(btn.dataset.index, 10);
          const start = parseFloat(btn.dataset.start);
          const end = parseFloat(btn.dataset.end);
          self.togglePlayTurn(idx, start, end, btn);
        });
      });
    },

    togglePlayTurn(index, start, end, btn) {
      if (this.activePlayingTurnIndex === index) {
        this.stopTurnPreview();
        return;
      }
      this.stopTurnPreview();

      const audioId = this.el.audioSelect?.value || (window.state && window.state.activeAudio && window.state.activeAudio.id);
      if (!audioId) return;

      this.activePlayingTurnIndex = index;
      btn.classList.remove('btn-secondary');
      btn.classList.add('btn-danger');
      btn.textContent = '⏹ Stop';

      const streamUrl = `/api/audio/${audioId}/segment?start=${start}&end=${end}`;
      this.previewAudio.src = streamUrl;
      this.previewAudio.play().catch(err => {
        console.warn('Playback error:', err);
        this.stopTurnPreview();
      });
    },

    stopTurnPreview() {
      this.previewAudio.pause();
      this.previewAudio.currentTime = 0;
      if (this.activePlayingTurnIndex !== null && this.el.turnsBody) {
        const btn = this.el.turnsBody.querySelector(`.exp-table-play-btn[data-index="${this.activePlayingTurnIndex}"]`);
        if (btn) {
          btn.classList.remove('btn-danger');
          btn.classList.add('btn-secondary');
          btn.textContent = '▶ Play';
        }
      }
      this.activePlayingTurnIndex = null;
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
  };

  window.ExperimentTab = ExperimentTab;
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => ExperimentTab.init());
  } else {
    ExperimentTab.init();
  }
})();
