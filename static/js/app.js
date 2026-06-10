/**
 * app.js — Predictive Analytics Dashboard
 * ----------------------------------------
 * Wizard state management, API calls, Plotly chart rendering,
 * drag-and-drop upload, toasts, and all UI interactions.
 */

'use strict';

// ═══════════════════════════════════════════════════════════════
// State
// ═══════════════════════════════════════════════════════════════
const State = {
  currentStep   : 0,
  uploadedFile  : null,
  columns       : [],
  numericCols   : [],
  modelInfo     : null,
  trainResult   : null,
  downloadToken : null,
};

const STEPS = [
  { id: 'step-upload',     label: 'Upload' },
  { id: 'step-preprocess', label: 'Preprocess' },
  { id: 'step-eda',        label: 'EDA' },
  { id: 'step-train',      label: 'Train' },
  { id: 'step-forecast',   label: 'Forecast' },
  { id: 'step-download',   label: 'Download' },
];

// ═══════════════════════════════════════════════════════════════
// DOM Ready
// ═══════════════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
  initStepProgress();
  initUploadZone();
  initSamplePicker();
  initPeriodSlider();
  goToStep(0);
});

// ═══════════════════════════════════════════════════════════════
// Step Progress
// ═══════════════════════════════════════════════════════════════
function initStepProgress() {
  const container = document.getElementById('step-progress');
  STEPS.forEach((step, idx) => {
    const item = document.createElement('div');
    item.className = 'step-item';
    item.id = `step-indicator-${idx}`;
    item.innerHTML = `
      <div class="step-dot">${idx + 1}</div>
      <span class="step-label">${step.label}</span>
    `;
    item.addEventListener('click', () => {
      if (idx <= State.currentStep) goToStep(idx);
    });
    container.appendChild(item);
  });
}

function goToStep(idx) {
  State.currentStep = idx;

  // Update step indicators
  document.querySelectorAll('.step-item').forEach((el, i) => {
    el.classList.toggle('active', i === idx);
    el.classList.toggle('completed', i < idx);
    const dot = el.querySelector('.step-dot');
    if (i < idx) dot.innerHTML = '✓';
    else dot.innerHTML = i + 1;
  });

  // Show the correct panel
  document.querySelectorAll('.section-panel').forEach((el, i) => {
    el.classList.toggle('active', i === idx);
  });

  // Scroll to top of content
  document.querySelector('.main-container')?.scrollIntoView({ behavior: 'smooth' });
}

function nextStep() { goToStep(State.currentStep + 1); }

// ═══════════════════════════════════════════════════════════════
// Upload Zone
// ═══════════════════════════════════════════════════════════════
function initUploadZone() {
  const zone  = document.getElementById('upload-zone');
  const input = document.getElementById('file-input');

  zone.addEventListener('click', () => input.click());

  zone.addEventListener('dragover', e => {
    e.preventDefault();
    zone.classList.add('dragover');
  });

  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));

  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file) handleFileSelect(file);
  });

  input.addEventListener('change', () => {
    if (input.files[0]) handleFileSelect(input.files[0]);
  });
}

async function handleFileSelect(file) {
  if (!file.name.endsWith('.csv')) {
    showToast('Only CSV files are supported.', 'error');
    return;
  }

  showLoading('upload-loading', true);
  const formData = new FormData();
  formData.append('file', file);

  try {
    const res  = await fetch('/api/upload', { method: 'POST', body: formData });
    const data = await res.json();

    if (data.error) { showToast(data.error, 'error'); return; }

    State.uploadedFile = file.name;
    State.columns      = data.columns;
    renderUploadSuccess(data);
    showToast(`✅ "${data.filename}" uploaded successfully!`, 'success');
  } catch (e) {
    showToast('Upload failed: ' + e.message, 'error');
  } finally {
    showLoading('upload-loading', false);
  }
}

function renderUploadSuccess(data) {
  document.getElementById('upload-filename').textContent = data.filename;
  document.getElementById('upload-rows').textContent   = data.raw_shape[0].toLocaleString();
  document.getElementById('upload-cols').textContent   = data.raw_shape[1];
  document.getElementById('upload-success').classList.remove('hidden');
  document.getElementById('upload-zone').classList.add('hidden');
  renderPreviewTable('upload-preview-table', data.preview);
}

// ═══════════════════════════════════════════════════════════════
// Sample Picker
// ═══════════════════════════════════════════════════════════════
const SAMPLE_META = {
  'housing.csv'     : { icon: '🏠', desc: 'Regression — House price prediction' },
  'stock_prices.csv': { icon: '📈', desc: 'Time-Series — ARIMA stock forecasting' },
};

async function initSamplePicker() {
  try {
    const res     = await fetch('/api/samples');
    const data    = await res.json();
    const grid    = document.getElementById('sample-grid');
    grid.innerHTML = '';

    data.samples.forEach(name => {
      const meta = SAMPLE_META[name] || { icon: '📂', desc: name };
      const card = document.createElement('div');
      card.className = 'sample-card';
      card.innerHTML = `
        <span class="sample-icon">${meta.icon}</span>
        <div class="sample-name">${name}</div>
        <div class="sample-desc">${meta.desc}</div>
      `;
      card.addEventListener('click', () => loadSample(name));
      grid.appendChild(card);
    });
  } catch (e) {
    console.warn('Could not load samples:', e);
  }
}

async function loadSample(name) {
  showLoading('upload-loading', true);
  try {
    const res  = await fetch('/api/load_sample', {
      method : 'POST',
      headers: { 'Content-Type': 'application/json' },
      body   : JSON.stringify({ name }),
    });
    const data = await res.json();
    if (data.error) { showToast(data.error, 'error'); return; }

    State.uploadedFile = name;
    State.columns      = data.columns;
    renderUploadSuccess(data);
    showToast(`✅ Loaded sample dataset: ${name}`, 'success');
  } catch (e) {
    showToast('Could not load sample: ' + e.message, 'error');
  } finally {
    showLoading('upload-loading', false);
  }
}

// ═══════════════════════════════════════════════════════════════
// Preprocess
// ═══════════════════════════════════════════════════════════════
async function runPreprocess() {
  if (!State.uploadedFile) { showToast('Please upload a file first.', 'error'); return; }

  const btn = document.getElementById('btn-preprocess');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Cleaning…';
  showLoading('preprocess-loading', true);

  try {
    const res  = await fetch('/api/preprocess', { method: 'POST' });
    const data = await res.json();
    if (data.error) { showToast(data.error, 'error'); return; }

    renderPreprocessReport(data.report);
    renderPreviewTable('preprocess-preview-table', data.preview);
    document.getElementById('preprocess-results').classList.remove('hidden');
    document.getElementById('btn-next-eda').classList.remove('hidden');
    showToast('✅ Data cleaned successfully!', 'success');
  } catch (e) {
    showToast('Preprocessing failed: ' + e.message, 'error');
  } finally {
    btn.disabled = false;
    btn.innerHTML = '🧹 Clean & Preprocess Data';
    showLoading('preprocess-loading', false);
  }
}

function renderPreprocessReport(report) {
  document.getElementById('stat-original-shape').textContent =
    report.original_shape.join(' × ');
  document.getElementById('stat-cleaned-shape').textContent =
    report.cleaned_shape.join(' × ');
  document.getElementById('stat-duplicates').textContent =
    report.duplicates_removed;
  document.getElementById('stat-date-cols').textContent =
    report.date_columns.length || '—';

  const nullInfo = document.getElementById('null-info');
  if (Object.keys(report.null_counts_before).length > 0) {
    nullInfo.textContent = 'Filled nulls in: ' +
      Object.entries(report.null_counts_before)
            .map(([c, n]) => `${c} (${n})`)
            .join(', ');
  } else {
    nullInfo.textContent = 'No missing values detected.';
  }
}

// ═══════════════════════════════════════════════════════════════
// EDA
// ═══════════════════════════════════════════════════════════════
async function runEDA() {
  showLoading('eda-loading', true);
  document.getElementById('eda-content').classList.add('hidden');

  try {
    const res  = await fetch('/api/eda', { method: 'POST' });
    const data = await res.json();
    if (data.error) { showToast(data.error, 'error'); return; }

    State.modelInfo = data.model_info;
    renderEDA(data);
    document.getElementById('eda-content').classList.remove('hidden');
    showToast('✅ EDA complete!', 'success');
  } catch (e) {
    showToast('EDA failed: ' + e.message, 'error');
  } finally {
    showLoading('eda-loading', false);
  }
}

function renderEDA(data) {
  // Model recommendation
  const mi = data.model_info;
  const badgeClass = mi.model_type === 'timeseries' ? 'timeseries' : 'regression';
  const badgeLabel = mi.model_type === 'timeseries' ? '📅 ARIMA (Time-Series)' : '📐 Linear Regression';
  document.getElementById('model-badge').innerHTML =
    `<span class="model-badge ${badgeClass}">${badgeLabel}</span>`;
  document.getElementById('model-reason').textContent = mi.reason;

  // Stats table
  renderStatsTable(data.stats);

  // Correlation heatmap
  if (data.heatmap && data.heatmap !== '{}') {
    renderChart('chart-heatmap', data.heatmap);
  }

  // Historical overview
  if (data.overview && data.overview !== '{}') {
    renderChart('chart-overview', data.overview);
  }

  // Distribution histograms
  const distGrid = document.getElementById('dist-charts-grid');
  distGrid.innerHTML = '';
  (data.distributions || []).forEach((chartJson, i) => {
    const div = document.createElement('div');
    div.className = 'chart-container';
    div.id = `dist-chart-${i}`;
    distGrid.appendChild(div);
    renderChart(`dist-chart-${i}`, chartJson);
  });
}

function renderStatsTable(stats) {
  const tbody = document.getElementById('stats-table-body');
  tbody.innerHTML = '';
  (stats.numeric_summary || []).forEach(row => {
    const tr = document.createElement('tr');
    const cols = ['column', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'];
    cols.forEach(col => {
      const td = document.createElement('td');
      const val = row[col];
      td.textContent = val !== null && val !== undefined ? Number(val).toLocaleString(undefined, { maximumFractionDigits: 4 }) : '—';
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
}

// ═══════════════════════════════════════════════════════════════
// Train
// ═══════════════════════════════════════════════════════════════
async function runTrain() {
  const btn = document.getElementById('btn-train');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Training…';
  showLoading('train-loading', true);
  document.getElementById('train-results').classList.add('hidden');

  // Read user overrides
  const targetCol  = document.getElementById('target-col-select')?.value || '';
  const modelType  = document.getElementById('model-type-select')?.value || '';

  const body = {};
  if (targetCol)  body.target_col  = targetCol;
  if (modelType)  body.model_type  = modelType;

  // Feature cols = all numeric except target
  if (targetCol && State.numericCols.length > 1) {
    body.feature_cols = State.numericCols.filter(c => c !== targetCol);
  }

  try {
    const res  = await fetch('/api/train', {
      method : 'POST',
      headers: { 'Content-Type': 'application/json' },
      body   : JSON.stringify(body),
    });
    const data = await res.json();
    if (data.error) { showToast(data.error, 'error'); return; }

    State.trainResult = data;
    renderTrainResults(data);
    document.getElementById('train-results').classList.remove('hidden');
    showToast('✅ Model trained successfully!', 'success');
  } catch (e) {
    showToast('Training failed: ' + e.message, 'error');
  } finally {
    btn.disabled = false;
    btn.innerHTML = '🧠 Train Model';
    showLoading('train-loading', false);
  }
}

function renderTrainResults(data) {
  const m = data.metrics;

  // Metrics
  document.getElementById('metric-mae').textContent  = m.mae  ?? '—';
  document.getElementById('metric-mse').textContent  = m.mse  ?? '—';
  document.getElementById('metric-rmse').textContent = m.rmse ?? '—';

  const r2El = document.getElementById('metric-r2');
  if (m.r2 !== undefined) {
    r2El.textContent = m.r2;
    const r2Card = r2El.closest('.metric-card');
    if (r2Card) r2Card.classList.toggle('good', m.r2 >= 0.8);
  } else {
    r2El.textContent = 'N/A';
  }

  // Model type badge
  const typeEl = document.getElementById('trained-model-type');
  if (typeEl) typeEl.textContent = data.model_type === 'timeseries' ? 'ARIMA' : 'Linear Regression';

  // Actual vs predicted chart
  if (data.actual_vs_pred) renderChart('chart-actual-vs-pred', data.actual_vs_pred);

  // Feature importance (regression only)
  const importanceSection = document.getElementById('feature-importance-section');
  if (data.feature_importance && data.feature_importance.length > 0) {
    importanceSection.classList.remove('hidden');
    renderImportanceTable(data.feature_importance);
  } else {
    importanceSection.classList.add('hidden');
  }
}

function renderImportanceTable(importance) {
  const tbody = document.getElementById('importance-table-body');
  tbody.innerHTML = '';
  const maxCoef = Math.max(...importance.map(r => Math.abs(r.coefficient)));

  importance.forEach(row => {
    const pct = maxCoef > 0 ? (Math.abs(row.coefficient) / maxCoef * 100).toFixed(0) : 0;
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><strong>${row.feature}</strong></td>
      <td>${row.coefficient}</td>
      <td>
        <div class="importance-bar">
          <div class="importance-bar-fill" style="width:${pct}%"></div>
          <span style="font-size:0.75rem;color:var(--text-secondary)">${pct}%</span>
        </div>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

// ═══════════════════════════════════════════════════════════════
// Forecast
// ═══════════════════════════════════════════════════════════════
function initPeriodSlider() {
  const slider  = document.getElementById('period-slider');
  const display = document.getElementById('period-display');
  if (!slider) return;
  slider.addEventListener('input', () => {
    display.textContent = slider.value;
  });
}

async function runForecast() {
  const btn = document.getElementById('btn-forecast');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Forecasting…';
  showLoading('forecast-loading', true);
  document.getElementById('forecast-results').classList.add('hidden');

  const nPeriods = parseInt(document.getElementById('period-slider')?.value || '10', 10);

  try {
    const res  = await fetch('/api/forecast', {
      method : 'POST',
      headers: { 'Content-Type': 'application/json' },
      body   : JSON.stringify({ n_periods: nPeriods }),
    });
    const data = await res.json();
    if (data.error) { showToast(data.error, 'error'); return; }

    State.downloadToken = data.download_token;
    renderForecastResults(data, nPeriods);
    document.getElementById('forecast-results').classList.remove('hidden');
    showToast(`✅ Forecast for ${nPeriods} periods ready!`, 'success');
  } catch (e) {
    showToast('Forecast failed: ' + e.message, 'error');
  } finally {
    btn.disabled = false;
    btn.innerHTML = '🔮 Generate Forecast';
    showLoading('forecast-loading', false);
  }
}

function renderForecastResults(data, nPeriods) {
  // Chart
  if (data.forecast_chart) renderChart('chart-forecast', data.forecast_chart);

  // Summary table (first 10)
  const tbody = document.getElementById('forecast-table-body');
  tbody.innerHTML = '';
  const rows = data.csv_rows.slice(1, 11);   // skip header row
  rows.forEach(row => {
    const tr = document.createElement('tr');
    row.forEach(cell => {
      const td = document.createElement('td');
      td.textContent = cell;
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });

  // Set download link
  const dlBtn = document.getElementById('btn-download-csv');
  if (dlBtn && State.downloadToken) {
    dlBtn.onclick = () => {
      window.location.href = `/api/download/${State.downloadToken}`;
    };
  }

  // Populate download step summary
  document.getElementById('dl-n-periods').textContent = nPeriods;
  document.getElementById('dl-token').textContent     = State.downloadToken || '';
}

// ═══════════════════════════════════════════════════════════════
// Column picker population
// ═══════════════════════════════════════════════════════════════
async function populateColumnPickers() {
  try {
    const res  = await fetch('/api/columns');
    const data = await res.json();
    State.numericCols = data.numeric_cols || [];

    const targetSel = document.getElementById('target-col-select');
    if (targetSel) {
      targetSel.innerHTML = '<option value="">Auto-detect</option>' +
        data.numeric_cols.map(c => `<option value="${c}">${c}</option>`).join('');
    }
  } catch (e) {
    console.warn('Could not fetch column info:', e);
  }
}

// ═══════════════════════════════════════════════════════════════
// Chart rendering via Plotly
// ═══════════════════════════════════════════════════════════════
function renderChart(containerId, chartJson) {
  const el = document.getElementById(containerId);
  if (!el) return;

  try {
    const fig = typeof chartJson === 'string' ? JSON.parse(chartJson) : chartJson;
    Plotly.react(el, fig.data, fig.layout, {
      responsive       : true,
      displayModeBar   : true,
      modeBarButtonsToRemove: ['lasso2d', 'select2d'],
      displaylogo      : false,
    });
  } catch (e) {
    console.error('Chart render error:', e);
  }
}

// ═══════════════════════════════════════════════════════════════
// Table rendering utility
// ═══════════════════════════════════════════════════════════════
function renderPreviewTable(tableId, preview) {
  const table = document.getElementById(tableId);
  if (!table || !preview) return;

  const thead = table.querySelector('thead');
  const tbody = table.querySelector('tbody');

  // Header
  thead.innerHTML = '<tr>' + preview.columns.map(c =>
    `<th>${c}</th>`
  ).join('') + '</tr>';

  // Rows
  tbody.innerHTML = preview.rows.map(row =>
    '<tr>' + preview.columns.map(c =>
      `<td>${row[c] !== null && row[c] !== undefined ? row[c] : '—'}</td>`
    ).join('') + '</tr>'
  ).join('');
}

// ═══════════════════════════════════════════════════════════════
// Loading overlay
// ═══════════════════════════════════════════════════════════════
function showLoading(id, show) {
  const el = document.getElementById(id);
  if (!el) return;
  el.classList.toggle('hidden', !show);
}

// ═══════════════════════════════════════════════════════════════
// Toast notifications
// ═══════════════════════════════════════════════════════════════
function showToast(message, type = 'info') {
  const container = document.getElementById('toast-container');
  const icons = { success: '✅', error: '❌', info: 'ℹ️' };

  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `<span>${icons[type] || ''}</span><span>${message}</span>`;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    setTimeout(() => toast.remove(), 300);
  }, 3800);
}

// ═══════════════════════════════════════════════════════════════
// Navigation helpers (called from HTML onclick)
// ═══════════════════════════════════════════════════════════════
function proceedToPreprocess() {
  nextStep();
}

function proceedToEDA() {
  nextStep();
  runEDA();
  populateColumnPickers();
}

function proceedToTrain() {
  nextStep();
}

function proceedToForecast() {
  nextStep();
}

function proceedToDownload() {
  nextStep();
}

function resetUpload() {
  State.uploadedFile = null;
  document.getElementById('upload-success').classList.add('hidden');
  document.getElementById('upload-zone').classList.remove('hidden');
  document.getElementById('file-input').value = '';
}
