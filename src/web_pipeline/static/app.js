/**
 * SonicPipeline — Frontend Application Logic
 * Large-Scale Audio Processing, Dataset Engineering & Batch Orchestration
 */

(function () {
  'use strict';

  // State
  const state = {
    activeTab: 'tab-dashboard',
    datasets: [],
    items: [],
    selectedItemIds: new Set(),
    jobs: [],
    jobFilter: 'all',
    telemetry: null,
    eventSource: null,
    isPaused: false,
    maxConcurrency: 2,
    batchUploadFiles: [],
    currentPlayingItemId: null,
    logEntries: [],
  };

  const TAB_TITLES = {
    'tab-dashboard': 'Overview & Metrics',
    'tab-queue': 'Batch Job Queue & Orchestration',
    'tab-ingest': 'Batch Audio Ingestion',
    'tab-separation': 'Bulk Stem Separation Studio',
    'tab-diarization': 'Batch Speaker Diarization Studio',
    'tab-benchmark': 'Separation Benchmark Matrix',
    'tab-datasets': 'Dataset Hub & Manifest Curation',
  };

  // DOM Elements
  const els = {
    // Nav & Theme
    navBtns: document.querySelectorAll('.sidebar-nav-item'),
    tabPanes: document.querySelectorAll('.tab-pane'),
    activeBreadcrumbTitle: document.getElementById('active-breadcrumb-title'),
    activeJobsCounter: document.getElementById('active-jobs-counter'),
    btnThemeToggle: document.getElementById('btn-theme-toggle'),
    iconThemeSun: document.getElementById('icon-theme-sun'),
    iconThemeMoon: document.getElementById('icon-theme-moon'),
    
    // Telemetry header & sidebar
    valGpuVram: document.getElementById('val-gpu-vram'),
    valCpuUtil: document.getElementById('val-cpu-util'),
    valRamUtil: document.getElementById('val-ram-util'),
    valSseStatus: document.getElementById('val-sse-status'),
    meterVram: document.getElementById('meter-vram'),
    meterCpu: document.getElementById('meter-cpu'),
    meterRam: document.getElementById('meter-ram'),
    
    // Dashboard
    dashTotalHours: document.getElementById('dash-total-hours'),
    dashTotalItems: document.getElementById('dash-total-items'),
    dashSpeedup: document.getElementById('dash-speedup'),
    dashActiveJobs: document.getElementById('dash-active-jobs'),
    dashQueuedJobs: document.getElementById('dash-queued-jobs'),
    dashDatasetItems: document.getElementById('dash-dataset-items'),
    dashDatasetSize: document.getElementById('dash-dataset-size'),
    dashDeviceName: document.getElementById('dash-device-name'),
    dashVramText: document.getElementById('dash-vram-text'),
    meterVramDash: document.getElementById('meter-vram-dash'),
    dashCpuText: document.getElementById('dash-cpu-text'),
    meterCpuDash: document.getElementById('meter-cpu-dash'),
    dashRamText: document.getElementById('dash-ram-text'),
    meterRamDash: document.getElementById('meter-ram-dash'),
    dashDiskText: document.getElementById('dash-disk-text'),
    meterDisk: document.getElementById('meter-disk'),
    activityFeed: document.getElementById('activity-feed'),
    logSearchInput: document.getElementById('log-search-input'),
    btnCopyLogs: document.getElementById('btn-copy-logs'),
    btnClearLogs: document.getElementById('btn-clear-logs'),
    checkAutoScroll: document.getElementById('check-auto-scroll'),
    btnRefreshTelemetry: document.getElementById('btn-refresh-telemetry'),

    // Queue
    queueConcurrency: document.getElementById('queue-concurrency'),
    btnPauseQueue: document.getElementById('btn-pause-queue'),
    labelPauseQueue: document.getElementById('label-pause-queue'),
    btnClearCompleted: document.getElementById('btn-clear-completed'),
    filterPills: document.querySelectorAll('.filter-pill'),
    jobsList: document.getElementById('jobs-list'),
    countAll: document.getElementById('count-all'),
    countRunning: document.getElementById('count-running'),
    countPending: document.getElementById('count-pending'),
    countCompleted: document.getElementById('count-completed'),
    countFailed: document.getElementById('count-failed'),

    // Ingest
    formYtIngest: document.getElementById('form-yt-ingest'),
    ytUrls: document.getElementById('yt-urls'),
    ytUrlCountBadge: document.getElementById('yt-url-count-badge'),
    ytDataset: document.getElementById('yt-dataset'),
    ytSampleRate: document.getElementById('yt-sample-rate'),
    ytTags: document.getElementById('yt-tags'),
    fileDropzone: document.getElementById('file-dropzone'),
    fileInput: document.getElementById('file-input'),
    batchUploadPreview: document.getElementById('batch-upload-files-preview'),
    batchFilesCount: document.getElementById('batch-files-count'),
    btnClearBatchFiles: document.getElementById('btn-clear-batch-files'),
    btnSubmitBatchUpload: document.getElementById('btn-submit-batch-upload'),
    formScanDir: document.getElementById('form-scan-dir'),
    scanPath: document.getElementById('scan-path'),
    localDataset: document.getElementById('local-dataset'),
    localTags: document.getElementById('local-tags'),

    // Separation
    formBatchSeparation: document.getElementById('form-batch-separation'),
    sepDatasetSelect: document.getElementById('sep-dataset-select'),
    sepModel: document.getElementById('sep-model'),
    sepDevice: document.getElementById('sep-device'),
    sepTargetCountHint: document.getElementById('sep-target-count-hint'),
    sepModelRadios: document.querySelectorAll('input[name="sep-model-radio"]'),

    // Diarization
    formBatchDiarization: document.getElementById('form-batch-diarization'),
    diarDatasetSelect: document.getElementById('diar-dataset-select'),
    diarBackend: document.getElementById('diar-backend'),
    diarDevice: document.getElementById('diar-device'),
    diarMinSpk: document.getElementById('diar-min-spk'),
    diarMaxSpk: document.getElementById('diar-max-spk'),
    diarHfToken: document.getElementById('diar-hf-token'),
    btnToggleTokenVis: document.getElementById('btn-toggle-token-vis'),
    diarBackendRadios: document.querySelectorAll('input[name="diar-backend-radio"]'),

    // Benchmark
    formBatchBenchmark: document.getElementById('form-batch-benchmark'),
    benchSpeechDs: document.getElementById('bench-speech-ds'),
    benchMusicDs: document.getElementById('bench-music-ds'),
    benchSnrs: document.getElementById('bench-snrs'),
    benchmarkHistorySelect: document.getElementById('benchmark-history-select'),
    leaderboardTbody: document.getElementById('leaderboard-tbody'),

    // Datasets & Manifests
    itemFilterDataset: document.getElementById('item-filter-dataset'),
    itemFilterQuery: document.getElementById('item-filter-query'),
    itemFilterStems: document.getElementById('item-filter-stems'),
    itemsTbody: document.getElementById('items-tbody'),
    checkSelectAll: document.getElementById('check-select-all'),
    bulkBar: document.getElementById('bulk-bar'),
    bulkSelectedCount: document.getElementById('bulk-selected-count'),
    btnBulkTag: document.getElementById('btn-bulk-tag'),
    btnBulkMove: document.getElementById('btn-bulk-move'),
    btnBulkSeparate: document.getElementById('btn-bulk-separate'),
    btnBulkDelete: document.getElementById('btn-bulk-delete'),
    btnExportManifest: document.getElementById('btn-export-manifest'),
    btnExportZip: document.getElementById('btn-export-zip'),

    // Inspector Modal
    modalInspector: document.getElementById('modal-inspector'),
    inspectorTitle: document.getElementById('inspector-title'),
    inspectorBody: document.getElementById('inspector-body'),
    btnCloseInspector: document.getElementById('btn-close-inspector'),

    // Auditioning Audio Element
    auditionAudio: document.getElementById('pipeline-audition-audio'),

    // Toast
    toastContainer: document.getElementById('toast-container'),
  };

  // -----------------------------------------------------------------------
  // Toast Notifications
  // -----------------------------------------------------------------------
  function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    let icon = 'ℹ️';
    if (type === 'success') icon = '✓';
    if (type === 'danger' || type === 'error') icon = '✕';
    if (type === 'warning') icon = '⚠️';

    toast.innerHTML = `<span style="font-weight: 700;">${icon}</span><span>${escapeHtml(message)}</span>`;
    els.toastContainer.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateX(100%)';
      setTimeout(() => toast.remove(), 250);
    }, 3800);
  }

  // -----------------------------------------------------------------------
  // Live Activity Log Feed Terminal
  // -----------------------------------------------------------------------
  function addActivityLog(message, level = 'info') {
    if (!els.activityFeed) return;
    const now = new Date().toLocaleTimeString();
    const entryData = { time: now, message, level };
    state.logEntries.push(entryData);
    if (state.logEntries.length > 300) state.logEntries.shift();

    renderActivityLogs();
  }

  function renderActivityLogs() {
    if (!els.activityFeed) return;
    const query = (els.logSearchInput?.value || '').toLowerCase();
    const filtered = state.logEntries.filter(e => !query || e.message.toLowerCase().includes(query));

    els.activityFeed.innerHTML = filtered.map(e => `
      <div class="terminal-line">
        <span class="term-time">${e.time}</span>
        <span class="term-level term-level-${e.level}">${e.level}</span>
        <span class="term-text">${escapeHtml(e.message)}</span>
      </div>
    `).join('');

    if (els.checkAutoScroll && els.checkAutoScroll.checked) {
      els.activityFeed.scrollTop = els.activityFeed.scrollHeight;
    }
  }

  function initLogFeedControls() {
    if (els.logSearchInput) {
      els.logSearchInput.addEventListener('input', renderActivityLogs);
    }
    if (els.btnCopyLogs) {
      els.btnCopyLogs.addEventListener('click', () => {
        const text = state.logEntries.map(e => `[${e.time}] [${e.level.toUpperCase()}] ${e.message}`).join('\n');
        navigator.clipboard.writeText(text);
        showToast('All logs copied to clipboard!', 'success');
      });
    }
    if (els.btnClearLogs) {
      els.btnClearLogs.addEventListener('click', () => {
        state.logEntries = [];
        renderActivityLogs();
        showToast('Log feed cleared', 'info');
      });
    }
  }

  function escapeHtml(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  // -----------------------------------------------------------------------
  // Theme Management
  // -----------------------------------------------------------------------
  function initTheme() {
    const savedTheme = localStorage.getItem('sonic_pipeline_theme') || (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
    applyTheme(savedTheme);

    if (els.btnThemeToggle) {
      els.btnThemeToggle.addEventListener('click', () => {
        const current = document.documentElement.getAttribute('data-theme') || 'dark';
        const next = current === 'dark' ? 'light' : 'dark';
        applyTheme(next);
        showToast(`Switched to ${next} theme`, 'info');
      });
    }
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    try {
      localStorage.setItem('sonic_pipeline_theme', theme);
    } catch (_) {}

    const isLight = theme === 'light';
    if (els.iconThemeSun) els.iconThemeSun.classList.toggle('hidden', !isLight);
    if (els.iconThemeMoon) els.iconThemeMoon.classList.toggle('hidden', isLight);
  }

  // -----------------------------------------------------------------------
  // Tab Switching
  // -----------------------------------------------------------------------
  function initTabs() {
    els.navBtns.forEach((btn) => {
      btn.addEventListener('click', () => {
        const tabId = btn.getAttribute('data-tab');
        switchTab(tabId);
      });
    });
  }

  function switchTab(tabId) {
    state.activeTab = tabId;
    els.navBtns.forEach((b) => {
      b.classList.toggle('active', b.getAttribute('data-tab') === tabId);
    });
    els.tabPanes.forEach((pane) => {
      pane.classList.toggle('active', pane.id === tabId);
    });

    if (els.activeBreadcrumbTitle) {
      els.activeBreadcrumbTitle.textContent = TAB_TITLES[tabId] || 'Workspace';
    }

    if (tabId === 'tab-queue') loadJobs();
    if (tabId === 'tab-datasets') loadItems();
    if (tabId === 'tab-benchmark') loadBenchmarkHistory();
  }

  // -----------------------------------------------------------------------
  // Server-Sent Events (SSE) Real-Time Stream
  // -----------------------------------------------------------------------
  function initEventSource() {
    if (state.eventSource) {
      state.eventSource.close();
    }

    state.eventSource = new EventSource('/api/events');

    state.eventSource.onopen = () => {
      if (els.valSseStatus) {
        els.valSseStatus.textContent = 'Live';
        els.valSseStatus.className = 'telemetry-value text-success';
      }
    };

    state.eventSource.onmessage = (e) => {
      try {
        const payload = JSON.parse(e.data);
        handleStreamEvent(payload);
      } catch (err) {
        console.error('Failed to parse SSE event:', err);
      }
    };

    state.eventSource.onerror = () => {
      if (els.valSseStatus) {
        els.valSseStatus.textContent = 'Reconnecting';
        els.valSseStatus.className = 'telemetry-value text-muted';
      }
      state.eventSource.close();
      setTimeout(initEventSource, 3000);
    };
  }

  function handleStreamEvent(eventPayload) {
    const { event, data } = eventPayload;
    if (event === 'reload') {
      console.log('Hot reload event received from server!');
      window.location.reload();
      return;
    }
    if (event === 'telemetry') {
      updateTelemetryUI(data);
    } else if (event === 'job_created' || event === 'job_updated') {
      upsertJobInState(data);
      renderJobs();
      updateDashboardCards();
      if (data.current_step) {
        addActivityLog(`[${data.type}] ${data.current_step}`, data.status === 'failed' ? 'error' : 'info');
      }
    } else if (event === 'job_progress') {
      updateJobProgressInState(data);
      if (data.last_log) {
        addActivityLog(`[Job] ${data.last_log.message}`, data.last_log.level || 'info');
      }
    } else if (event === 'job_deleted') {
      state.jobs = state.jobs.filter((j) => j.id !== data.id);
      renderJobs();
      updateDashboardCards();
    } else if (event === 'datasets_updated' || event === 'items_deleted' || event === 'items_tagged' || event === 'items_moved') {
      loadDatasets();
      if (state.activeTab === 'tab-datasets') loadItems();
    }
  }

  // -----------------------------------------------------------------------
  // Telemetry UI
  // -----------------------------------------------------------------------
  function updateTelemetryUI(telemetry) {
    state.telemetry = telemetry;
    if (!telemetry) return;

    // Header pills & sidebar meters
    if (telemetry.gpu && telemetry.gpu.available && telemetry.gpu.vram_percent !== null) {
      const vramPct = telemetry.gpu.vram_percent;
      if (els.valGpuVram) els.valGpuVram.textContent = `${vramPct}%`;
      if (els.meterVram) els.meterVram.style.width = `${vramPct}%`;
      if (els.meterVramDash) els.meterVramDash.style.width = `${vramPct}%`;
      if (els.dashVramText) els.dashVramText.textContent = `${telemetry.gpu.reserved_vram_mb} / ${telemetry.gpu.total_vram_mb} MB (${vramPct}%)`;
    } else if (telemetry.gpu && telemetry.gpu.type === 'mps') {
      if (els.valGpuVram) els.valGpuVram.textContent = 'MPS';
      if (els.dashVramText) els.dashVramText.textContent = 'Apple MPS Accelerator';
    } else {
      if (els.valGpuVram) els.valGpuVram.textContent = 'CPU';
      if (els.dashVramText) els.dashVramText.textContent = 'CPU Mode (No CUDA VRAM)';
    }

    if (telemetry.cpu) {
      const cpuPct = Math.round(telemetry.cpu.utilization_percent);
      if (els.valCpuUtil) els.valCpuUtil.textContent = `${cpuPct}%`;
      if (els.meterCpu) els.meterCpu.style.width = `${cpuPct}%`;
      if (els.meterCpuDash) els.meterCpuDash.style.width = `${cpuPct}%`;
      if (els.dashCpuText) els.dashCpuText.textContent = `${cpuPct}% (${telemetry.cpu.logical_cores} cores)`;
    }

    if (telemetry.ram) {
      const ramPct = Math.round(telemetry.ram.percent);
      if (els.valRamUtil) els.valRamUtil.textContent = `${ramPct}%`;
      if (els.meterRam) els.meterRam.style.width = `${ramPct}%`;
      if (els.meterRamDash) els.meterRamDash.style.width = `${ramPct}%`;
      if (els.dashRamText) els.dashRamText.textContent = `${telemetry.ram.used_gb} / ${telemetry.ram.total_gb} GB (${ramPct}%)`;
    }

    // Dashboard Overview
    if (telemetry.throughput) {
      if (els.dashTotalHours) els.dashTotalHours.innerHTML = `${telemetry.throughput.processed_audio_hours.toFixed(2)} <span class="kpi-unit">hrs</span>`;
      if (els.dashTotalItems) els.dashTotalItems.textContent = `${telemetry.throughput.processed_items} items processed`;
      if (els.dashSpeedup) els.dashSpeedup.innerHTML = `${telemetry.throughput.speedup_factor} <span class="kpi-unit">x Realtime</span>`;
    }

    if (telemetry.disk) {
      const diskPct = Math.round(telemetry.disk.percent);
      if (els.dashDatasetSize) els.dashDatasetSize.textContent = `${telemetry.disk.pipeline_data_mb} MB on disk`;
      if (els.dashDiskText) els.dashDiskText.textContent = `${telemetry.disk.used_gb} / ${telemetry.disk.total_gb} GB (${diskPct}%)`;
      if (els.meterDisk) els.meterDisk.style.width = `${telemetry.disk.percent}%`;
    }

    if (telemetry.gpu && els.dashDeviceName) {
      els.dashDeviceName.textContent = telemetry.gpu.name || 'CPU Node';
    }
  }

  // -----------------------------------------------------------------------
  // Datasets Management
  // -----------------------------------------------------------------------
  async function loadDatasets() {
    try {
      const res = await fetch('/api/datasets');
      const data = await res.json();
      state.datasets = data;
      populateDatasetDropdowns();
      updateDashboardCards();
    } catch (err) {
      console.error('Failed to load datasets:', err);
    }
  }

  function populateDatasetDropdowns() {
    const dropdowns = document.querySelectorAll('.select-dataset-list');
    dropdowns.forEach((dd) => {
      const currentVal = dd.value;
      const allowAll = dd.id === 'sep-dataset-select' || dd.id === 'item-filter-dataset' || dd.id === 'diar-dataset-select';
      
      let html = '';
      if (allowAll) {
        html += '<option value="all">All Datasets</option>';
      }
      state.datasets.forEach((ds) => {
        html += `<option value="${escapeHtml(ds.name)}">${escapeHtml(ds.name)} (${ds.item_count} items)</option>`;
      });

      dd.innerHTML = html;
      if (currentVal) dd.value = currentVal;
    });

    updateSeparationTargetCountHint();
  }

  function updateSeparationTargetCountHint() {
    if (!els.sepTargetCountHint || !els.sepDatasetSelect) return;
    const val = els.sepDatasetSelect.value;
    if (val === 'all') {
      const total = state.datasets.reduce((acc, d) => acc + (d.item_count || 0), 0);
      els.sepTargetCountHint.innerHTML = `<span class="dot-pulse-green"></span><span>Target Scope: Entire Library (${total} audio tracks ready for separation)</span>`;
    } else {
      const ds = state.datasets.find(d => d.name === val);
      const count = ds ? ds.item_count : 0;
      els.sepTargetCountHint.innerHTML = `<span class="dot-pulse-green"></span><span>Target Scope: '${escapeHtml(val)}' (${count} audio tracks)</span>`;
    }
  }

  function updateDashboardCards() {
    const totalItems = state.datasets.reduce((acc, d) => acc + (d.item_count || 0), 0);
    if (els.dashDatasetItems) els.dashDatasetItems.innerHTML = `${totalItems} <span class="kpi-unit">files</span>`;

    const activeJobs = state.jobs.filter((j) => j.status === 'running').length;
    const queuedJobs = state.jobs.filter((j) => j.status === 'pending').length;

    if (els.dashActiveJobs) els.dashActiveJobs.innerHTML = `${activeJobs} <span class="kpi-unit">running</span>`;
    if (els.dashQueuedJobs) els.dashQueuedJobs.textContent = `${queuedJobs} pending in queue`;
    if (els.activeJobsCounter) {
      els.activeJobsCounter.textContent = activeJobs + queuedJobs;
      els.activeJobsCounter.style.display = (activeJobs + queuedJobs) > 0 ? 'inline-flex' : 'none';
    }

    // Update filter counts
    if (els.countAll) els.countAll.textContent = state.jobs.length;
    if (els.countRunning) els.countRunning.textContent = state.jobs.filter(j => j.status === 'running').length;
    if (els.countPending) els.countPending.textContent = state.jobs.filter(j => j.status === 'pending').length;
    if (els.countCompleted) els.countCompleted.textContent = state.jobs.filter(j => j.status === 'completed').length;
    if (els.countFailed) els.countFailed.textContent = state.jobs.filter(j => j.status === 'failed').length;
  }

  // -----------------------------------------------------------------------
  // Batch Queue & Jobs Management
  // -----------------------------------------------------------------------
  async function loadJobs() {
    try {
      const res = await fetch('/api/jobs');
      state.jobs = await res.json();
      renderJobs();
      updateDashboardCards();
    } catch (err) {
      console.error('Failed to fetch jobs:', err);
    }
  }

  function upsertJobInState(jobData) {
    const idx = state.jobs.findIndex((j) => j.id === jobData.id);
    if (idx >= 0) {
      state.jobs[idx] = { ...state.jobs[idx], ...jobData };
    } else {
      state.jobs.unshift(jobData);
    }
  }

  function updateJobProgressInState(progressData) {
    const job = state.jobs.find((j) => j.id === progressData.id);
    if (job) {
      job.progress = progressData.progress;
      job.current_step = progressData.current_step;
      job.processed_items = progressData.processed_items;
      job.failed_items = progressData.failed_items;
      job.status = progressData.status;
      renderJobs();
    }
  }

  function renderJobs() {
    let filtered = state.jobs;
    if (state.jobFilter !== 'all') {
      filtered = state.jobs.filter((j) => j.status === state.jobFilter);
    }

    if (filtered.length === 0) {
      els.jobsList.innerHTML = `
        <div class="empty-state-box">
          <div class="empty-state-icon">⚡</div>
          <p class="empty-state-text">No jobs matching the current filter.</p>
          <p class="empty-state-sub">Launch an ingest, stem separation, or diarization batch job to populate.</p>
        </div>
      `;
      return;
    }

    els.jobsList.innerHTML = filtered.map((j) => createJobCardHTML(j)).join('');
    attachJobActionListeners();
  }

  function createJobCardHTML(job) {
    const statusClass = `job-status-${job.status}`;
    const percent = Math.round(job.progress || 0);
    const isRunning = job.status === 'running';

    let etaText = '';
    if (isRunning && percent > 0 && percent < 100) {
      const elapsed = (Date.now() / 1000) - (job.created_at || (Date.now() / 1000));
      const totalEstimated = elapsed / (percent / 100);
      const remaining = Math.max(0, totalEstimated - elapsed);
      etaText = ` • ETA: ${Math.round(remaining)}s`;
    }

    return `
      <div class="job-card" id="card-${job.id}">
        <div class="job-header">
          <div class="job-title-group">
            <span class="badge badge-accent">${escapeHtml(job.type.replace('batch_', '').toUpperCase())}</span>
            <span class="job-name">${escapeHtml(job.title)}</span>
            <span class="job-id">${escapeHtml(job.id)}</span>
          </div>
          <span class="job-status-badge ${statusClass}">${job.status}</span>
        </div>

        <div class="job-progress-row">
          <div class="job-progress-meta">
            <span>${escapeHtml(job.current_step || 'Processing...')}</span>
            <span><strong>${percent}%</strong> (${job.processed_items || 0}/${job.total_items || 0} items${etaText})</span>
          </div>
          <div class="job-progress-bar">
            <div class="job-progress-fill ${isRunning ? 'running' : ''}" style="width: ${percent}%;"></div>
          </div>
        </div>

        <div class="job-footer">
          <div class="job-stats-pills">
            <span>Started ${new Date(job.created_at * 1000).toLocaleTimeString()}</span>
            ${job.failed_items > 0 ? `<span style="color: var(--color-danger); font-weight: 700;">${job.failed_items} errors</span>` : ''}
          </div>
          <div class="job-actions">
            ${isRunning || job.status === 'pending' ? `<button class="btn btn-secondary btn-xs btn-cancel-job" data-id="${job.id}">Cancel</button>` : ''}
            <button class="btn btn-secondary btn-xs btn-delete-job" data-id="${job.id}">Delete</button>
          </div>
        </div>
      </div>
    `;
  }

  function attachJobActionListeners() {
    document.querySelectorAll('.btn-cancel-job').forEach((b) => {
      b.addEventListener('click', async () => {
        const id = b.getAttribute('data-id');
        try {
          await fetch(`/api/jobs/${id}/cancel`, { method: 'POST' });
          showToast('Job cancellation requested', 'warning');
        } catch (err) {
          showToast('Failed to cancel job', 'danger');
        }
      });
    });

    document.querySelectorAll('.btn-delete-job').forEach((b) => {
      b.addEventListener('click', async () => {
        const id = b.getAttribute('data-id');
        try {
          await fetch(`/api/jobs/${id}`, { method: 'DELETE' });
          showToast('Job record deleted', 'info');
        } catch (err) {
          showToast('Failed to delete job', 'danger');
        }
      });
    });
  }

  // -----------------------------------------------------------------------
  // Ingest Handlers
  // -----------------------------------------------------------------------
  function initIngestForms() {
    // URL Count Counter
    if (els.ytUrls && els.ytUrlCountBadge) {
      els.ytUrls.addEventListener('input', () => {
        const count = els.ytUrls.value.split('\n').filter(l => l.trim().length > 0).length;
        els.ytUrlCountBadge.textContent = `${count} URL${count === 1 ? '' : 's'}`;
      });
    }

    // YouTube Ingest
    if (els.formYtIngest) {
      els.formYtIngest.addEventListener('submit', async (e) => {
        e.preventDefault();
        const urls = els.ytUrls.value.trim();
        const dataset = els.ytDataset.value;
        const sampleRate = parseInt(els.ytSampleRate.value, 10);
        const tags = els.ytTags.value.split(',').map((t) => t.trim()).filter(Boolean);

        if (!urls) {
          showToast('Please enter YouTube URLs or playlist link', 'warning');
          return;
        }

        try {
          const res = await fetch('/api/jobs/batch_ingest_yt', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ urls, dataset, sample_rate: sampleRate, tags }),
          });
          const job = await res.json();
          if (res.ok) {
            showToast('YouTube batch ingest enqueued!', 'success');
            els.ytUrls.value = '';
            if (els.ytUrlCountBadge) els.ytUrlCountBadge.textContent = '0 URLs';
            switchTab('tab-queue');
          } else {
            showToast(job.error || 'Failed to submit ingest', 'danger');
          }
        } catch (err) {
          showToast('Network error submitting job', 'danger');
        }
      });
    }

    // Dropzone Upload
    if (els.fileDropzone) {
      els.fileDropzone.addEventListener('click', () => els.fileInput.click());
      els.fileDropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        els.fileDropzone.classList.add('dragover');
      });
      els.fileDropzone.addEventListener('dragleave', () => els.fileDropzone.classList.remove('dragover'));
      els.fileDropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        els.fileDropzone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
          handleFilesSelected(e.dataTransfer.files);
        }
      });
    }

    if (els.fileInput) {
      els.fileInput.addEventListener('change', () => {
        if (els.fileInput.files.length) {
          handleFilesSelected(els.fileInput.files);
        }
      });
    }

    if (els.btnClearBatchFiles) {
      els.btnClearBatchFiles.addEventListener('click', () => {
        state.batchUploadFiles = [];
        els.batchUploadPreview.classList.add('hidden');
      });
    }

    if (els.btnSubmitBatchUpload) {
      els.btnSubmitBatchUpload.addEventListener('click', async () => {
        if (state.batchUploadFiles.length === 0) return;
        await uploadFiles(state.batchUploadFiles);
        state.batchUploadFiles = [];
        els.batchUploadPreview.classList.add('hidden');
      });
    }

    // Server Directory Scan
    if (els.formScanDir) {
      els.formScanDir.addEventListener('submit', async (e) => {
        e.preventDefault();
        const scanDir = els.scanPath.value.trim();
        const dataset = els.localDataset.value;
        const tags = els.localTags.value.split(',').map((t) => t.trim()).filter(Boolean);

        if (!scanDir) {
          showToast('Please specify a directory path', 'warning');
          return;
        }

        try {
          const res = await fetch('/api/jobs/batch_ingest_files', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ scan_directory: scanDir, dataset, tags }),
          });
          const job = await res.json();
          if (res.ok) {
            showToast('Directory scan job enqueued!', 'success');
            els.scanPath.value = '';
            switchTab('tab-queue');
          } else {
            showToast(job.error || 'Scan submission failed', 'danger');
          }
        } catch (err) {
          showToast('Error connecting to server', 'danger');
        }
      });
    }
  }

  function handleFilesSelected(files) {
    state.batchUploadFiles = Array.from(files);
    const totalBytes = state.batchUploadFiles.reduce((acc, f) => acc + f.size, 0);
    const sizeMb = (totalBytes / (1024 * 1024)).toFixed(1);

    if (els.batchFilesCount) els.batchFilesCount.textContent = `${state.batchUploadFiles.length} file(s) selected (${sizeMb} MB)`;
    if (els.batchUploadPreview) els.batchUploadPreview.classList.remove('hidden');
  }

  async function uploadFiles(files) {
    const formData = new FormData();
    formData.append('dataset', els.localDataset.value || 'Default');
    formData.append('tags', els.localTags.value || 'upload');
    for (let i = 0; i < files.length; i++) {
      formData.append('files', files[i]);
    }

    showToast(`Uploading ${files.length} audio file(s)...`, 'info');
    try {
      const res = await fetch('/api/jobs/batch_upload', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (res.ok) {
        showToast('Files uploaded and batch ingest started!', 'success');
        switchTab('tab-queue');
      } else {
        showToast(data.error || 'Upload failed', 'danger');
      }
    } catch (err) {
      showToast('File upload failed', 'danger');
    }
  }

  // -----------------------------------------------------------------------
  // Separation Handlers
  // -----------------------------------------------------------------------
  function initSeparationForm() {
    if (els.sepDatasetSelect) {
      els.sepDatasetSelect.addEventListener('change', updateSeparationTargetCountHint);
    }

    // Separation model radio cards
    els.sepModelRadios.forEach(radio => {
      radio.addEventListener('change', () => {
        document.querySelectorAll('input[name="sep-model-radio"]').forEach(r => {
          r.closest('.model-pro-card')?.classList.remove('active');
        });
        radio.closest('.model-pro-card')?.classList.add('active');
        if (els.sepModel) els.sepModel.value = radio.value;
      });
    });

    if (els.formBatchSeparation) {
      els.formBatchSeparation.addEventListener('submit', async (e) => {
        e.preventDefault();
        const dataset = els.sepDatasetSelect.value;
        const model = els.sepModel.value;
        const device = els.sepDevice.value;

        try {
          const res = await fetch('/api/jobs/batch_separation', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              dataset: dataset === 'all' ? null : dataset,
              model,
              device,
            }),
          });
          const job = await res.json();
          if (res.ok) {
            showToast(`Separation job enqueued for model ${model}!`, 'success');
            switchTab('tab-queue');
          } else {
            showToast(job.error || 'Separation failed to start', 'danger');
          }
        } catch (err) {
          showToast('Error submitting separation job', 'danger');
        }
      });
    }
  }

  // -----------------------------------------------------------------------
  // Diarization Handlers
  // -----------------------------------------------------------------------
  function initDiarizationForm() {
    // Restore cached HF token
    try {
      const savedToken = localStorage.getItem('sonic_hf_token');
      if (savedToken && els.diarHfToken) {
        els.diarHfToken.value = savedToken;
      }
    } catch (_) {}

    // Toggle token visibility
    if (els.btnToggleTokenVis && els.diarHfToken) {
      els.btnToggleTokenVis.addEventListener('click', () => {
        const isPass = els.diarHfToken.type === 'password';
        els.diarHfToken.type = isPass ? 'text' : 'password';
        els.btnToggleTokenVis.textContent = isPass ? 'Hide' : 'Show';
      });

      els.diarHfToken.addEventListener('change', () => {
        try {
          localStorage.setItem('sonic_hf_token', els.diarHfToken.value.trim());
        } catch (_) {}
      });
    }

    // Diarization backend radio cards
    els.diarBackendRadios.forEach(radio => {
      radio.addEventListener('change', () => {
        document.querySelectorAll('input[name="diar-backend-radio"]').forEach(r => {
          r.closest('.model-pro-card')?.classList.remove('active');
        });
        radio.closest('.model-pro-card')?.classList.add('active');
        if (els.diarBackend) els.diarBackend.value = radio.value;
      });
    });

    if (els.formBatchDiarization) {
      els.formBatchDiarization.addEventListener('submit', async (e) => {
        e.preventDefault();
        const dataset = els.diarDatasetSelect.value;
        const backend = els.diarBackend.value;
        const device = els.diarDevice.value;
        const minSpk = els.diarMinSpk.value ? parseInt(els.diarMinSpk.value, 10) : null;
        const maxSpk = els.diarMaxSpk.value ? parseInt(els.diarMaxSpk.value, 10) : null;
        const token = els.diarHfToken.value.trim() || null;

        try {
          const res = await fetch('/api/jobs/batch_diarization', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              dataset: dataset === 'all' ? null : dataset,
              backend,
              device,
              min_speakers: minSpk,
              max_speakers: maxSpk,
              hf_token: token,
            }),
          });
          const job = await res.json();
          if (res.ok) {
            showToast(`Diarization job enqueued for ${backend}!`, 'success');
            switchTab('tab-queue');
          } else {
            showToast(job.error || 'Diarization failed to start', 'danger');
          }
        } catch (err) {
          showToast('Error submitting diarization job', 'danger');
        }
      });
    }
  }

  // -----------------------------------------------------------------------
  // Benchmark Handlers
  // -----------------------------------------------------------------------
  function initBenchmark() {
    if (els.formBatchBenchmark) {
      els.formBatchBenchmark.addEventListener('submit', async (e) => {
        e.preventDefault();
        const speechDs = els.benchSpeechDs.value;
        const musicDs = els.benchMusicDs.value;
        const snrs = els.benchSnrs.value.split(',').map((s) => parseFloat(s.trim())).filter((n) => !isNaN(n));
        const modelCheckboxes = document.querySelectorAll('input[name="bench-models"]:checked');
        const models = Array.from(modelCheckboxes).map((cb) => cb.value);

        if (models.length === 0) {
          showToast('Please select at least one separation model', 'warning');
          return;
        }

        try {
          const res = await fetch('/api/jobs/batch_benchmark', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              speech_dataset: speechDs,
              music_dataset: musicDs,
              snr_levels: snrs,
              models,
            }),
          });
          const job = await res.json();
          if (res.ok) {
            showToast('Benchmark evaluation job enqueued!', 'success');
            switchTab('tab-queue');
          } else {
            showToast(job.error || 'Benchmark failed', 'danger');
          }
        } catch (err) {
          showToast('Error submitting benchmark job', 'danger');
        }
      });
    }
  }

  async function loadBenchmarkHistory() {
    try {
      const res = await fetch('/api/benchmarks');
      const reports = await res.json();
      if (!res.ok) throw new Error(reports.error || 'Failed to load benchmark history');

      if (els.benchmarkHistorySelect) {
        els.benchmarkHistorySelect.innerHTML = reports.length
          ? reports.map((report, index) => `<option value="${escapeHtml(report.job_id || '')}">${index === 0 ? 'Latest: ' : ''}${new Date((report.timestamp || 0) * 1000).toLocaleString()}</option>`).join('')
          : '<option value="">No benchmark runs</option>';
        els.benchmarkHistorySelect.onchange = async () => {
          const jobId = els.benchmarkHistorySelect.value;
          if (!jobId) {
            renderLeaderboard({});
            return;
          }
          const detailRes = await fetch(`/api/benchmarks/${encodeURIComponent(jobId)}`);
          const detail = await detailRes.json();
          if (!detailRes.ok) throw new Error(detail.error || 'Failed to load benchmark report');
          renderLeaderboard(detail.leaderboard);
        };
      }

      if (reports.length > 0) {
        renderLeaderboard(reports[0].leaderboard);
      } else {
        renderLeaderboard({});
      }
    } catch (err) {
      console.error('Failed loading benchmark history:', err);
    }
  }

  function renderLeaderboard(leaderboard) {
    if (!leaderboard || Object.keys(leaderboard).length === 0) {
      els.leaderboardTbody.innerHTML = '<tr><td colspan="6" class="text-center text-muted">No benchmark runs recorded yet.</td></tr>';
      return;
    }

    const rows = Object.values(leaderboard).map((item) => `
      <tr>
        <td><strong>${escapeHtml(item.model)}</strong></td>
        <td>${item.samples_count}</td>
        <td><strong style="color: var(--color-emerald);">${item.mean_si_sdri_db >= 0 ? '+' : ''}${item.mean_si_sdri_db} dB</strong></td>
        <td>${item.median_si_sdri_db} dB</td>
        <td>${item.mean_vocals_si_sdr_db} dB</td>
        <td>${item.avg_speed_sec}s</td>
      </tr>
    `);
    els.leaderboardTbody.innerHTML = rows.join('');
  }

  // -----------------------------------------------------------------------
  // Dataset Hub & Manifest Handlers (with Inline Auditioning Player)
  // -----------------------------------------------------------------------
  async function loadItems() {
    const dataset = els.itemFilterDataset.value;
    const query = els.itemFilterQuery.value.trim();
    const stems = els.itemFilterStems.value;

    const params = new URLSearchParams();
    if (dataset && dataset !== 'all') params.append('dataset', dataset);
    if (query) params.append('query', query);
    if (stems) params.append('has_stems', stems);

    try {
      const res = await fetch(`/api/items?${params.toString()}`);
      const data = await res.json();
      state.items = data.items;
      renderItemsTable();
    } catch (err) {
      console.error('Failed to load items:', err);
    }
  }

  function renderItemsTable() {
    if (state.items.length === 0) {
      els.itemsTbody.innerHTML = '<tr><td colspan="10" class="text-center text-muted">No audio assets found. Ingest audio files or playlists to populate.</td></tr>';
      return;
    }

    const rows = state.items.map((item) => {
      const isChecked = state.selectedItemIds.has(item.id);
      const isPlaying = state.currentPlayingItemId === item.id;
      const stemCount = Object.keys(item.stems || {}).length;
      const stemBadge = stemCount > 0 ? `<span class="stem-badge">${stemCount} model(s)</span>` : '<span class="text-muted">None</span>';
      const diarBadge = item.diarization ? `<span class="badge badge-accent">${item.diarization.speaker_count} spk</span>` : '<span class="text-muted">No</span>';
      const tags = (item.tags || []).map((t) => `<span class="tag-pill">${escapeHtml(t)}</span>`).join(' ');

      return `
        <tr data-id="${item.id}">
          <td><input type="checkbox" class="row-checkbox" data-id="${item.id}" ${isChecked ? 'checked' : ''}></td>
          <td>
            <button class="btn-inline-play ${isPlaying ? 'playing' : ''}" data-id="${item.id}" title="${isPlaying ? 'Pause' : 'Play & Audition'}" aria-label="Audition track">
              ${isPlaying ? '❚❚' : '▶'}
            </button>
          </td>
          <td>
            <strong>${escapeHtml(item.title)}</strong><br>
            <small class="text-muted font-mono">${escapeHtml(item.id)}</small>
          </td>
          <td><span class="tag-pill">${escapeHtml(item.dataset)}</span></td>
          <td>${(item.duration || 0).toFixed(2)}s</td>
          <td>${item.sample_rate.toLocaleString()} Hz / ${item.channels === 1 ? 'Mono' : 'Stereo'}</td>
          <td>${stemBadge}</td>
          <td>${diarBadge}</td>
          <td>${tags}</td>
          <td>
            <button class="btn btn-secondary btn-xs btn-inspect-item" data-id="${item.id}">Inspect</button>
          </td>
        </tr>
      `;
    });

    els.itemsTbody.innerHTML = rows.join('');
    attachTableListeners();
    updateBulkBar();
  }

  function attachTableListeners() {
    // Checkboxes
    document.querySelectorAll('.row-checkbox').forEach((cb) => {
      cb.addEventListener('change', () => {
        const id = cb.getAttribute('data-id');
        if (cb.checked) {
          state.selectedItemIds.add(id);
        } else {
          state.selectedItemIds.delete(id);
        }
        updateBulkBar();
      });
    });

    // Inline Audition Player
    document.querySelectorAll('.btn-inline-play').forEach((b) => {
      b.addEventListener('click', () => {
        const id = b.getAttribute('data-id');
        toggleInlinePlay(id);
      });
    });

    // Inspect buttons
    document.querySelectorAll('.btn-inspect-item').forEach((b) => {
      b.addEventListener('click', () => {
        const id = b.getAttribute('data-id');
        openInspector(id);
      });
    });
  }

  function toggleInlinePlay(itemId) {
    if (!els.auditionAudio) return;

    if (state.currentPlayingItemId === itemId && !els.auditionAudio.paused) {
      els.auditionAudio.pause();
      state.currentPlayingItemId = null;
    } else {
      state.currentPlayingItemId = itemId;
      els.auditionAudio.src = `/api/items/${itemId}/stream`;
      els.auditionAudio.play().catch(e => console.error("Inline play prevented:", e));
    }
    renderItemsTable();
  }

  function initAuditionAudioElement() {
    if (!els.auditionAudio) return;

    els.auditionAudio.addEventListener('ended', () => {
      state.currentPlayingItemId = null;
      renderItemsTable();
    });
    els.auditionAudio.addEventListener('pause', () => {
      state.currentPlayingItemId = null;
      renderItemsTable();
    });
    els.auditionAudio.addEventListener('error', () => {
      state.currentPlayingItemId = null;
      renderItemsTable();
      showToast('Audio playback error', 'danger');
    });
  }

  function updateBulkBar() {
    const count = state.selectedItemIds.size;
    if (els.bulkSelectedCount) els.bulkSelectedCount.textContent = count;
    if (els.bulkBar) els.bulkBar.style.display = count > 0 ? 'flex' : 'none';
  }

  function openInspector(itemId) {
    const item = state.items.find((i) => i.id === itemId);
    if (!item) return;

    els.inspectorTitle.textContent = `Asset Inspector: ${item.title}`;
    let stemsHtml = '';
    if (item.stems && Object.keys(item.stems).length > 0) {
      stemsHtml = `
        <h4 style="margin: 16px 0 8px 0; font-size: 0.9rem; color: var(--text-main);">Separated Stems</h4>
        ${Object.entries(item.stems).map(([model, stems]) => `
          <div style="margin-bottom: 12px; background: var(--bg-panel); padding: 12px; border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
            <strong style="color: var(--color-primary);">${escapeHtml(model)}</strong>
            <div style="display: flex; flex-direction: column; gap: 8px; margin-top: 8px;">
              ${Object.entries(stems).map(([stemName, path]) => `
                <div style="display: flex; align-items: center; justify-content: space-between; gap: 8px;">
                  <span class="tag-pill">${escapeHtml(stemName)}</span>
                  <audio controls src="/api/items/${item.id}/stems/${model}/${stemName}/stream" style="height: 32px; flex: 1; max-width: 340px;"></audio>
                </div>
              `).join('')}
            </div>
          </div>
        `).join('')}
      `;
    }

    els.inspectorBody.innerHTML = `
      <div style="display: flex; flex-direction: column; gap: 14px;">
        <div>
          <label class="form-label">Master Source Stream</label>
          <audio controls src="/api/items/${item.id}/stream" style="width: 100%; margin-top: 6px;"></audio>
        </div>
        <div style="font-family: var(--font-mono); font-size: 12px; background: var(--bg-input); padding: 12px; border-radius: var(--radius-sm); border: 1px solid var(--border-subtle); display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
          <div><strong>Duration:</strong> ${(item.duration || 0).toFixed(2)}s</div>
          <div><strong>Sample Rate:</strong> ${item.sample_rate.toLocaleString()} Hz</div>
          <div><strong>Channels:</strong> ${item.channels === 1 ? 'Mono' : 'Stereo'}</div>
          <div><strong>Dataset:</strong> ${escapeHtml(item.dataset)}</div>
          <div style="grid-column: 1 / -1; word-break: break-all;"><strong>File Path:</strong> ${escapeHtml(item.path)}</div>
        </div>
        ${stemsHtml}
      </div>
    `;

    els.modalInspector.style.display = 'flex';
  }

  function initBulkActions() {
    // Bulk Tag
    if (els.btnBulkTag) {
      els.btnBulkTag.addEventListener('click', async () => {
        const tag = prompt("Enter tag to apply to selected items:");
        if (!tag) return;
        try {
          const itemIds = Array.from(state.selectedItemIds);
          const res = await fetch('/api/items/bulk_tag', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ item_ids: itemIds, add_tags: [tag], remove_tags: [] }),
          });
          const data = await res.json();
          if (!res.ok) throw new Error(data.error || 'Bulk tag failed');
          showToast(`Tagged ${itemIds.length} items with '${tag}'`, 'success');
          loadItems();
        } catch (err) {
          showToast('Bulk tag failed', 'danger');
        }
      });
    }

    // Bulk Move Dataset
    if (els.btnBulkMove) {
      els.btnBulkMove.addEventListener('click', async () => {
        const targetDataset = prompt("Enter target dataset name:", "Speech Corpus");
        if (!targetDataset) return;
        try {
          const itemIds = Array.from(state.selectedItemIds);
          await fetch('/api/items/bulk_dataset', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ item_ids: itemIds, dataset: targetDataset }),
          });
          showToast(`Moved ${itemIds.length} items to '${targetDataset}'`, 'success');
          loadDatasets();
          loadItems();
        } catch (err) {
          showToast('Bulk move failed', 'danger');
        }
      });
    }

    // Bulk Delete
    if (els.btnBulkDelete) {
      els.btnBulkDelete.addEventListener('click', async () => {
        const count = state.selectedItemIds.size;
        if (!confirm(`Are you sure you want to delete ${count} selected item(s)?`)) return;
        try {
          const itemIds = Array.from(state.selectedItemIds);
          await fetch('/api/items/delete', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ item_ids: itemIds }),
          });
          showToast(`Deleted ${count} items`, 'info');
          state.selectedItemIds.clear();
          loadDatasets();
          loadItems();
        } catch (err) {
          showToast('Bulk delete failed', 'danger');
        }
      });
    }

    // Bulk Separate
    if (els.btnBulkSeparate) {
      els.btnBulkSeparate.addEventListener('click', () => {
        switchTab('tab-separation');
      });
    }
  }

  function initManifestAndExport() {
    // Export manifest JSONL
    if (els.btnExportManifest) {
      els.btnExportManifest.addEventListener('click', async () => {
        try {
          const dataset = els.itemFilterDataset.value;
          const res = await fetch('/api/manifests/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              dataset: dataset === 'all' ? null : dataset,
              format: 'jsonl',
            }),
          });
          const blob = await res.blob();
          const url = window.URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = `manifest_${dataset || 'all'}.jsonl`;
          a.click();
          showToast('Manifest JSONL downloaded', 'success');
        } catch (err) {
          showToast('Failed to export manifest', 'danger');
        }
      });
    }

    // Export ZIP bundle
    if (els.btnExportZip) {
      els.btnExportZip.addEventListener('click', async () => {
        try {
          const dataset = els.itemFilterDataset.value;
          showToast('Packaging ZIP bundle...', 'info');
          const res = await fetch('/api/exports/create', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              dataset: dataset === 'all' ? null : dataset,
              include_stems: true,
              include_manifests: true,
            }),
          });
          const data = await res.json();
          if (res.ok && data.download_url) {
            const a = document.createElement('a');
            a.href = data.download_url;
            a.download = data.export_id;
            a.click();
            showToast(`ZIP package created (${data.size_mb} MB)`, 'success');
          } else {
            showToast(data.error || 'Export failed', 'danger');
          }
        } catch (err) {
          showToast('Failed to create ZIP export', 'danger');
        }
      });
    }
  }

  // -----------------------------------------------------------------------
  // Initialization
  // -----------------------------------------------------------------------
  function init() {
    initTheme();
    initTabs();
    initEventSource();
    initLogFeedControls();
    initIngestForms();
    initSeparationForm();
    initDiarizationForm();
    initBenchmark();
    initManifestAndExport();
    initBulkActions();
    initAuditionAudioElement();

    // Filters and search
    if (els.itemFilterDataset) els.itemFilterDataset.addEventListener('change', loadItems);
    if (els.itemFilterStems) els.itemFilterStems.addEventListener('change', loadItems);
    if (els.itemFilterQuery) {
      els.itemFilterQuery.addEventListener('input', () => {
        clearTimeout(window._searchTimer);
        window._searchTimer = setTimeout(loadItems, 300);
      });
    }

    if (els.checkSelectAll) {
      els.checkSelectAll.addEventListener('change', () => {
        const checked = els.checkSelectAll.checked;
        if (checked) {
          state.items.forEach((i) => state.selectedItemIds.add(i.id));
        } else {
          state.selectedItemIds.clear();
        }
        renderItemsTable();
      });
    }

    if (els.btnCloseInspector) {
      els.btnCloseInspector.addEventListener('click', () => {
        els.modalInspector.style.display = 'none';
      });
    }

    // Filter pills in queue
    els.filterPills.forEach((p) => {
      p.addEventListener('click', () => {
        els.filterPills.forEach((x) => x.classList.remove('active'));
        p.classList.add('active');
        state.jobFilter = p.getAttribute('data-status');
        renderJobs();
      });
    });

    // Clear completed jobs
    if (els.btnClearCompleted) {
      els.btnClearCompleted.addEventListener('click', async () => {
        const completedJobs = state.jobs.filter(j => j.status === 'completed' || j.status === 'cancelled');
        for (const job of completedJobs) {
          try {
            await fetch(`/api/jobs/${job.id}`, { method: 'DELETE' });
          } catch (_) {}
        }
        showToast(`Cleared ${completedJobs.length} completed job records`, 'info');
        loadJobs();
      });
    }

    // Queue Concurrency
    if (els.queueConcurrency) {
      els.queueConcurrency.addEventListener('change', async () => {
        const conc = parseInt(els.queueConcurrency.value, 10);
        await fetch('/api/queue/controls', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action: 'set_concurrency', concurrency: conc }),
        });
        showToast(`Concurrency updated to ${conc} worker slots`, 'info');
      });
    }

    // Pause / Resume Queue
    if (els.btnPauseQueue) {
      els.btnPauseQueue.addEventListener('click', async () => {
        state.isPaused = !state.isPaused;
        await fetch('/api/queue/controls', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action: state.isPaused ? 'pause' : 'resume' }),
        });
        els.labelPauseQueue.textContent = state.isPaused ? '▶ Resume Queue' : '⏸ Pause Queue';
        showToast(state.isPaused ? 'Queue paused' : 'Queue resumed', 'warning');
      });
    }

    if (els.btnRefreshTelemetry) {
      els.btnRefreshTelemetry.addEventListener('click', async () => {
        try {
          const res = await fetch('/api/telemetry');
          const data = await res.json();
          updateTelemetryUI(data);
          showToast('Telemetry refreshed', 'info');
        } catch (_) {}
      });
    }

    // Initial load
    loadDatasets();
    loadJobs();
  }

  document.addEventListener('DOMContentLoaded', init);
})();
