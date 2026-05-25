/* ==========================================
   SINGLE PAGE APPLICATION SCRIPT - TABBED VERSION
   ==========================================

   JavaScript untuk halaman single-page dengan tab navigation
*/

import { renderNavbar } from '../components/navbar.js';
import { renderFooter } from '../components/footer.js';
import { API_BASE_URL, formatDateTime, getProbabilityColor, getStatusBadgeClass } from './main.js';

// Render navbar dan footer
renderNavbar('home');
renderFooter();

// ==========================================
// TAB NAVIGATION SYSTEM
// ==========================================

let currentTab = 'home';

function switchTab(tabName) {
  // Hide all tabs
  document.querySelectorAll('.tab-content').forEach(tab => {
    tab.classList.remove('active');
  });

  // Show selected tab
  const targetTab = document.getElementById(`tab-${tabName}`);
  if (targetTab) {
    targetTab.classList.add('active');
    currentTab = tabName;
  }

  // Update navbar active state
  document.querySelectorAll('.navbar-link').forEach(link => {
    link.classList.remove('active');
    if (link.getAttribute('data-tab') === tabName) {
      link.classList.add('active');
    }
  });

  // Scroll to top
  window.scrollTo({ top: 0, behavior: 'smooth' });

  // Load data for specific tabs
  if (tabName === 'history') {
    loadHistory();
  } else if (tabName === 'analytics') {
    loadAnalytics();
  }
}

// Event listeners for tab navigation
document.addEventListener('click', (e) => {
  const tabTrigger = e.target.closest('[data-tab]');
  if (tabTrigger) {
    e.preventDefault();
    const tabName = tabTrigger.getAttribute('data-tab');
    switchTab(tabName);
  }
});

// ==========================================
// HOME SECTION
// ==========================================

// Load stats for home section
async function loadHomeStats() {
  try {
    const response = await fetch(`${API_BASE_URL}/analytics`);
    const data = await response.json();
    document.getElementById('statTotal').textContent = data.total_predictions;
    document.getElementById('statFailRate').textContent = data.failure_rate + '%';
  } catch (error) {
    document.getElementById('statTotal').textContent = '—';
    document.getElementById('statFailRate').textContent = '—';
  }
}

// Animate probability bar
setTimeout(() => {
  const probFill = document.getElementById('probFill');
  if (probFill) {
    probFill.style.width = '12%';
  }
}, 500);

// Animate counter for pipeline stats
function animateCounter(element, target, duration = 2000) {
  const start = 0;
  const increment = target / (duration / 16);
  let current = start;

  const timer = setInterval(() => {
    current += increment;
    if (current >= target) {
      element.textContent = target.toFixed(2);
      clearInterval(timer);
    } else {
      element.textContent = current.toFixed(2);
    }
  }, 16);
}

// Start animation for pipeline stats
function startPipelineStatsAnimation() {
  const statNumbers = document.querySelectorAll('.stat-number[data-target]');
  console.log('Found stat numbers:', statNumbers.length);

  statNumbers.forEach(stat => {
    const target = parseFloat(stat.getAttribute('data-target'));
    console.log('Animating stat to:', target);
    if (target && !isNaN(target)) {
      animateCounter(stat, target);
    }
  });
}

// Run animation after a short delay to ensure DOM is ready
setTimeout(() => {
  startPipelineStatsAnimation();
}, 1000);

// Also use Intersection Observer as backup
const observerOptions = {
  threshold: 0.1,
  rootMargin: '0px'
};

const statsObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      const statNumbers = entry.target.querySelectorAll('.stat-number[data-target]');
      statNumbers.forEach(stat => {
        const target = parseFloat(stat.getAttribute('data-target'));
        const currentValue = parseFloat(stat.textContent);
        // Only animate if still at 0
        if (target && !isNaN(target) && currentValue === 0) {
          animateCounter(stat, target);
        }
      });
      statsObserver.unobserve(entry.target);
    }
  });
}, observerOptions);

// Observe pipeline stats section
const pipelineStatsSection = document.querySelector('.pipeline-stats-section');
if (pipelineStatsSection) {
  statsObserver.observe(pipelineStatsSection);
}

loadHomeStats();

// ==========================================
// PREDICT SECTION
// ==========================================

const FAILURE_RECOMMENDATIONS = {
  TWF: 'Segera hentikan operasi dan ganti alat potong yang aus. Catat durasi pemakaian tool saat ini dan sesuaikan interval penggantian. Periksa parameter cutting speed dan feed rate nilai ekstrem mempercepat keausan tool.',
  HDF: 'Periksa sistem pendingin mesin: cek aliran coolant, bersihkan filter dan saluran yang tersumbat, serta pastikan kipas dan heat sink berfungsi normal. Kurangi kecepatan operasi sementara hingga suhu kembali normal.',
  PWF: 'Periksa tegangan dan arus listrik pada panel, pastikan tidak ada lonjakan atau drop tegangan. Cek kondisi motor listrik, VFD (inverter), dan gearbox. Evaluasi kombinasi torque dan rotational speed agar tidak melebihi batas daya maksimum mesin.',
  OSF: 'Kurangi beban kerja segera,, turunkan nilai torque atau rotational speed agar berada dalam batas spesifikasi teknis. Periksa kondisi tool wear karena keausan tool meningkatkan gaya potong dan memperparah overstrain.',
  RNF: 'Lakukan inspeksi menyeluruh pada semua komponen mesin karena penyebab kegagalan tidak teridentifikasi secara spesifik. Dokumentasikan kondisi sensor saat kegagalan terjadi untuk analisis lebih lanjut.'
};

const btnPredict = document.getElementById('btnPredict');
const btnReset = document.getElementById('btnReset');

if (btnPredict) {
  btnPredict.addEventListener('click', handlePredict);
}

if (btnReset) {
  btnReset.addEventListener('click', resetForm);
}

// Input validation for Rotational Speed
const rotationalSpeedInput = document.getElementById('rotational_speed');
if (rotationalSpeedInput) {
  rotationalSpeedInput.addEventListener('input', (e) => {
    const value = parseFloat(e.target.value);
    const warning = document.getElementById('warning_rotational_speed');

    if (warning) {
      if (value < 1168 || value > 2886) {
        warning.classList.add('show');
      } else {
        warning.classList.remove('show');
      }
    }
  });
}

// Input validation for Torque
const torqueInput = document.getElementById('torque');
if (torqueInput) {
  torqueInput.addEventListener('input', (e) => {
    const value = parseFloat(e.target.value);
    const warning = document.getElementById('warning_torque');

    if (warning) {
      if (value < 3.8 || value > 76.6) {
        warning.classList.add('show');
      } else {
        warning.classList.remove('show');
      }
    }
  });
}

async function handlePredict() {
  const btn = document.getElementById('btnPredict');
  const spinner = document.getElementById('spinner');
  const btnText = document.getElementById('btnText');
  const errorMsg = document.getElementById('errorMsg');

  const payload = {
    type: document.getElementById('type').value,
    air_temperature: parseFloat(document.getElementById('air_temperature').value),
    process_temperature: parseFloat(document.getElementById('process_temperature').value),
    rotational_speed: parseInt(document.getElementById('rotational_speed').value),
    torque: parseFloat(document.getElementById('torque').value),
    tool_wear: parseInt(document.getElementById('tool_wear').value)
  };

  // Validate
  const numericFields = ['air_temperature', 'process_temperature', 'rotational_speed', 'torque', 'tool_wear'];
  const hasInvalidField = numericFields.some(field => isNaN(payload[field]));

  if (hasInvalidField || !payload.type) {
    errorMsg.textContent = 'Semua field harus diisi dengan nilai valid.';
    errorMsg.style.display = 'block';
    return;
  }

  btn.disabled = true;
  spinner.classList.remove('hidden');
  btnText.textContent = 'Memproses...';
  errorMsg.style.display = 'none';

  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail?.[0]?.msg || 'Gagal melakukan prediksi');
    }

    const data = await response.json();
    console.log('Prediction result:', data);
    renderPredictionResult(data);
  } catch (error) {
    console.error('Prediction error:', error);
    errorMsg.textContent = 'Terjadi kesalahan: ' + error.message;
    errorMsg.style.display = 'block';
  } finally {
    btn.disabled = false;
    spinner.classList.add('hidden');
    btnText.textContent = 'Jalankan Prediksi';
  }
}

function renderPredictionResult(data) {
  const resultCard = document.getElementById('resultCard');
  const isFailure = data.status === 'Failure';

  document.getElementById('resultStatus').textContent = data.status;
  document.getElementById('resultStatus').style.color = isFailure ? 'var(--danger)' : 'var(--success)';

  document.getElementById('resultSubtitle').textContent = isFailure
    ? 'Mesin terdeteksi akan mengalami kegagalan'
    : 'Mesin dalam kondisi normal';

  document.getElementById('resultFailureType').textContent = data.failure_type || '—';
  document.getElementById('resultProbText').textContent = data.probability_failure.toFixed(4);

  const probPct = Math.round(data.probability_failure * 100);
  document.getElementById('probPct').textContent = probPct + '%';

  const probFill = document.getElementById('probFill2');
  probFill.style.width = probPct + '%';
  probFill.style.backgroundColor = getProbabilityColor(data.probability_failure);

  // Confidence Badge
  const confidenceBadge = document.getElementById('confidenceBadge');
  let badgeClass = '';
  let badgeText = '';

  console.log('=== CONFIDENCE BADGE DEBUG ===');
  console.log('Probability:', data.probability_failure);
  console.log('Probability Percent:', probPct);

  if (probPct <= 30) {
    badgeClass = 'confidence-low';
    badgeText = 'Risiko Rendah';
  } else if (probPct <= 60) {
    badgeClass = 'confidence-medium';
    badgeText = 'Risiko Sedang';
  } else {
    badgeClass = 'confidence-high';
    badgeText = 'Risiko Tinggi';
  }

  console.log('Badge Class:', badgeClass);
  console.log('Badge Text:', badgeText);

  confidenceBadge.innerHTML = `<span class="confidence-badge ${badgeClass}">${badgeText}</span>`;

  console.log('Badge HTML:', confidenceBadge.innerHTML);

  const recommendation = data.failure_type
    ? FAILURE_RECOMMENDATIONS[data.failure_type]
    : 'Tidak ada tindakan khusus diperlukan. Lanjutkan operasi normal dan pantau kondisi mesin secara berkala.';

  document.getElementById('resultRecommendation').textContent = recommendation;
  document.getElementById('checkedAt').textContent = 'Prediksi dilakukan pada ' + formatDateTime(data.checked_at);

  resultCard.classList.add('visible');
  resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function resetForm() {
  document.getElementById('type').value = 'L';
  document.getElementById('air_temperature').value = '298.1';
  document.getElementById('process_temperature').value = '308.6';
  document.getElementById('rotational_speed').value = '1500';
  document.getElementById('torque').value = '40.0';
  document.getElementById('tool_wear').value = '50';
  document.getElementById('errorMsg').textContent = '';
  document.getElementById('resultCard').classList.remove('visible');

  // Clear warnings
  const warningRotational = document.getElementById('warning_rotational_speed');
  const warningTorque = document.getElementById('warning_torque');

  if (warningRotational) warningRotational.classList.remove('show');
  if (warningTorque) warningTorque.classList.remove('show');
}

// ==========================================
// HISTORY SECTION
// ==========================================

let historyData = []; // Store history data for export
let currentPage = 1;
let itemsPerPage = 10;

const btnFilter = document.getElementById('btnFilter');
const btnResetFilter = document.getElementById('btnResetFilter');
const btnExportCSV = document.getElementById('btnExportCSV');

if (btnFilter) {
  btnFilter.addEventListener('click', loadHistory);
}

if (btnResetFilter) {
  btnResetFilter.addEventListener('click', resetFilters);
}

if (btnExportCSV) {
  btnExportCSV.addEventListener('click', exportToCSV);
}

async function loadHistory() {
  const status = document.getElementById('filterStatus').value;
  const failureType = document.getElementById('filterType').value;
  const limit = document.getElementById('filterLimit').value;

  // Build URL - if "all" is selected, use max limit (500)
  let url = `${API_BASE_URL}/history?limit=${limit === 'all' ? '500' : limit}`;
  if (status) url += `&status=${status}`;
  if (failureType) url += `&failure_type=${failureType}`;

  console.log('Loading history with limit:', limit === 'all' ? 'ALL (500)' : limit);

  document.getElementById('tableBody').innerHTML = '<div class="loading-state">Memuat data...</div>';

  try {
    const response = await fetch(url);
    const data = await response.json();

    // Store data for export
    historyData = data;

    // Reset to page 1
    currentPage = 1;

    renderHistoryTable(data);
  } catch (error) {
    console.error('Error loading history:', error);
    document.getElementById('tableBody').innerHTML = `
      <div class="empty-state">
        <p class="empty-state-text">Tidak dapat terhubung ke API.</p>
      </div>
    `;
  }
}

function renderHistoryTable(data) {
  const normal = data.filter(d => d.status === 'Normal').length;
  const failure = data.filter(d => d.status === 'Failure').length;
  const rate = data.length > 0 ? ((failure / data.length) * 100).toFixed(1) : 0;

  document.getElementById('statShown').textContent = data.length;
  document.getElementById('statNormal').textContent = normal;
  document.getElementById('statFailure').textContent = failure;
  document.getElementById('statRate').textContent = rate + '%';
  document.getElementById('tableCount').textContent = `${data.length} record`;

  if (data.length === 0) {
    document.getElementById('tableBody').innerHTML = `
      <div class="empty-state">
        <p class="empty-state-text">Belum ada data prediksi.</p>
      </div>
    `;
    return;
  }

  // Pagination calculation
  const totalPages = Math.ceil(data.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = startIndex + itemsPerPage;
  const paginatedData = data.slice(startIndex, endIndex);

  console.log('=== PAGINATION DEBUG ===');
  console.log('Total data:', data.length);
  console.log('Items per page:', itemsPerPage);
  console.log('Total pages:', totalPages);
  console.log('Current page:', currentPage);
  console.log('Showing records:', startIndex, 'to', endIndex);

  const rows = paginatedData.map(d => {
    const prob = d.probability_failure;
    const probPct = Math.round(prob * 100);
    const color = getProbabilityColor(prob);
    const badgeClass = getStatusBadgeClass(d.status);

    const ftTag = d.failure_type
      ? `<span class="badge badge-neutral">${d.failure_type}</span>`
      : '<span style="color:var(--text-muted)">—</span>';

    // Confidence Badge
    let confidenceBadgeClass = '';
    let confidenceBadgeText = '';

    if (probPct <= 30) {
      confidenceBadgeClass = 'confidence-low';
      confidenceBadgeText = 'Rendah';
    } else if (probPct <= 60) {
      confidenceBadgeClass = 'confidence-medium';
      confidenceBadgeText = 'Sedang';
    } else {
      confidenceBadgeClass = 'confidence-high';
      confidenceBadgeText = 'Tinggi';
    }

    return `
      <tr>
        <td class="font-mono">${d.id}</td>
        <td><span class="badge ${badgeClass}">${d.status}</span></td>
        <td>${ftTag}</td>
        <td>
          <div class="prob-cell">
            <div class="prob-mini-track">
              <div class="prob-mini-fill" style="width:${probPct}%;background-color:${color}"></div>
            </div>
            <span class="font-mono text-secondary">${prob.toFixed(4)}</span>
          </div>
        </td>
        <td><span class="confidence-badge ${confidenceBadgeClass}" style="font-size:11px;padding:4px 8px;">${confidenceBadgeText}</span></td>
        <td class="font-mono">${d.type}</td>
        <td class="font-mono">${d.air_temperature?.toFixed(1)}</td>
        <td class="font-mono">${d.torque?.toFixed(1)}</td>
        <td class="font-mono">${d.tool_wear}</td>
        <td class="text-muted" style="font-size:var(--font-size-xs)">${formatDateTime(d.checked_at)}</td>
      </tr>
    `;
  }).join('');

  document.getElementById('tableBody').innerHTML = `
    <table class="table">
      <thead>
        <tr>
          <th>ID</th>
          <th>Status</th>
          <th>Failure Type</th>
          <th>Probabilitas</th>
          <th>Risiko</th>
          <th>Type</th>
          <th>Air Temp</th>
          <th>Torque</th>
          <th>Tool Wear</th>
          <th>Waktu</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;

  // Render pagination
  renderPagination(totalPages);
}

// Render pagination controls
function renderPagination(totalPages) {
  const paginationContainer = document.getElementById('paginationContainer');

  if (!paginationContainer) {
    console.error('paginationContainer not found!');
    return;
  }

  // Clear previous pagination
  paginationContainer.innerHTML = '';

  console.log('renderPagination called with totalPages:', totalPages);

  if (totalPages <= 1) {
    console.log('Pagination skipped: totalPages <= 1');
    return;
  }

  const maxVisiblePages = 5;
  let startPage = Math.max(1, currentPage - Math.floor(maxVisiblePages / 2));
  let endPage = Math.min(totalPages, startPage + maxVisiblePages - 1);

  if (endPage - startPage < maxVisiblePages - 1) {
    startPage = Math.max(1, endPage - maxVisiblePages + 1);
  }

  // Generate page numbers
  let pageNumbers = '';

  // First page
  if (startPage > 1) {
    pageNumbers += `<button class="pagination-btn page-number" data-page="1">1</button>`;
    if (startPage > 2) {
      pageNumbers += `<span class="pagination-ellipsis">...</span>`;
    }
  }

  // Visible pages
  for (let i = startPage; i <= endPage; i++) {
    const activeClass = i === currentPage ? 'active' : '';
    pageNumbers += `<button class="pagination-btn page-number ${activeClass}" data-page="${i}">${i}</button>`;
  }

  // Last page
  if (endPage < totalPages) {
    if (endPage < totalPages - 1) {
      pageNumbers += `<span class="pagination-ellipsis">...</span>`;
    }
    pageNumbers += `<button class="pagination-btn page-number" data-page="${totalPages}">${totalPages}</button>`;
  }

  const paginationHTML = `
    <div class="pagination-controls">
      <button class="pagination-btn" id="btnPrevPage" ${currentPage === 1 ? 'disabled' : ''}>
        ← Prev
      </button>
      ${pageNumbers}
      <button class="pagination-btn" id="btnNextPage" ${currentPage === totalPages ? 'disabled' : ''}>
        Next →
      </button>
    </div>
  `;

  console.log('Inserting pagination HTML');
  paginationContainer.innerHTML = paginationHTML;
  console.log('Pagination HTML inserted');

  // Add event listeners
  const btnPrev = document.getElementById('btnPrevPage');
  const btnNext = document.getElementById('btnNextPage');

  if (btnPrev) {
    btnPrev.addEventListener('click', () => {
      if (currentPage > 1) {
        currentPage--;
        renderHistoryTable(historyData);
      }
    });
  }

  if (btnNext) {
    btnNext.addEventListener('click', () => {
      if (currentPage < totalPages) {
        currentPage++;
        renderHistoryTable(historyData);
      }
    });
  }

  // Page number buttons
  const pageButtons = document.querySelectorAll('.pagination-btn.page-number');
  console.log('Page buttons found:', pageButtons.length);
  pageButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const page = parseInt(btn.getAttribute('data-page'));
      if (page !== currentPage) {
        currentPage = page;
        renderHistoryTable(historyData);
      }
    });
  });
}

function resetFilters() {
  document.getElementById('filterStatus').value = '';
  document.getElementById('filterType').value = '';
  document.getElementById('filterLimit').value = '50';
  document.getElementById('filterDateStart').value = '';
  document.getElementById('filterDateEnd').value = '';
  loadHistory();
}

// Export to CSV function
function exportToCSV() {
  if (!historyData || historyData.length === 0) {
    alert('Tidak ada data untuk diekspor.');
    return;
  }

  console.log('Exporting', historyData.length, 'records to CSV');

  // CSV Headers
  const headers = [
    'ID',
    'Status',
    'Failure Type',
    'Probability Failure',
    'Type',
    'Air Temperature',
    'Process Temperature',
    'Rotational Speed',
    'Torque',
    'Tool Wear',
    'Checked At'
  ];

  // CSV Rows
  const rows = historyData.map(d => [
    d.id,
    d.status,
    d.failure_type || '',
    d.probability_failure.toFixed(4),
    d.type,
    d.air_temperature?.toFixed(1) || '',
    d.process_temperature?.toFixed(1) || '',
    d.rotational_speed || '',
    d.torque?.toFixed(1) || '',
    d.tool_wear || '',
    d.checked_at
  ]);

  // Combine headers and rows
  const csvContent = [
    headers.join(','),
    ...rows.map(row => row.join(','))
  ].join('\n');

  // Create Blob and download
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const link = document.createElement('a');
  const url = URL.createObjectURL(blob);

  link.setAttribute('href', url);
  link.setAttribute('download', `faultsense_history_${new Date().toISOString().split('T')[0]}.csv`);
  link.style.visibility = 'hidden';

  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);

  console.log('CSV export completed');
}

// ==========================================
// ANALYTICS SECTION
// ==========================================

const FT_COLORS = {
  TWF: '#22C55E',
  HDF: '#16A34A',
  PWF: '#4ADE80',
  OSF: '#86EFAC',
  RNF: '#BBF7D0'
};

let donutChart, barChart;

const btnRefresh = document.getElementById('btnRefresh');
if (btnRefresh) {
  btnRefresh.addEventListener('click', loadAnalytics);
}

async function loadAnalytics() {
  try {
    const response = await fetch(`${API_BASE_URL}/analytics`);
    const data = await response.json();
    renderAnalytics(data);
  } catch (error) {
    console.error('Error loading analytics:', error);
    document.getElementById('statTotalAnalytics').textContent = 'Error';
  }
}

function renderAnalytics(data) {
  const total = data.total_predictions;
  const normal = data.total_normal;
  const failure = data.total_failure;

  document.getElementById('lastChecked').textContent = formatDateTime(data.latest_checked_at);

  document.getElementById('statTotalAnalytics').textContent = total;
  document.getElementById('statNormalAnalytics').textContent = normal;
  document.getElementById('statFailureAnalytics').textContent = failure;
  document.getElementById('statRateAnalytics').textContent = data.failure_rate + '%';

  document.getElementById('statNormalPct').textContent =
    total > 0 ? ((normal / total) * 100).toFixed(1) + '% dari total' : '—';
  document.getElementById('statFailurePct').textContent =
    total > 0 ? ((failure / total) * 100).toFixed(1) + '% dari total' : '—';

  // Donut Chart
  if (donutChart) donutChart.destroy();
  const donutCtx = document.getElementById('donutChart').getContext('2d');
  donutChart = new Chart(donutCtx, {
    type: 'doughnut',
    data: {
      labels: ['Normal', 'Failure'],
      datasets: [{
        data: [normal, failure],
        backgroundColor: ['#22C55E', '#DC2626'],
        borderWidth: 0,
        hoverOffset: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '68%',
      plugins: {
        legend: {
          position: 'bottom',
          labels: {
            font: { family: 'Poppins', size: 12 },
            padding: 16,
            color: '#D1D5DB'
          }
        }
      }
    }
  });

  // Bar Chart
  const ftData = data.per_failure_type;
  const ftLabels = Object.keys(ftData);
  const ftValues = Object.values(ftData);
  const ftColors = ftLabels.map(k => FT_COLORS[k] || '#6b7280');

  if (barChart) barChart.destroy();
  const barCtx = document.getElementById('barChart').getContext('2d');
  barChart = new Chart(barCtx, {
    type: 'bar',
    data: {
      labels: ftLabels,
      datasets: [{
        label: 'Jumlah Kasus',
        data: ftValues,
        backgroundColor: ftColors,
        borderRadius: 6,
        borderWidth: 0
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false }
      },
      scales: {
        y: {
          beginAtZero: true,
          ticks: {
            stepSize: 1,
            font: { family: 'Poppins', size: 11 },
            color: '#D1D5DB'
          },
          grid: { color: '#374151' }
        },
        x: {
          ticks: {
            font: { family: 'Poppins', size: 11 },
            color: '#D1D5DB'
          },
          grid: { display: false }
        }
      }
    }
  });

  // Breakdown Table
  const maxVal = Math.max(...ftValues, 1);
  const rows = ftLabels.map(ft => {
    const val = ftData[ft];
    const pct = failure > 0 ? ((val / failure) * 100).toFixed(1) : 0;
    const color = FT_COLORS[ft] || '#22C55E';
    const width = ((val / maxVal) * 100).toFixed(1);

    return `
      <tr>
        <td><strong>${ft}</strong></td>
        <td class="font-mono">${val}</td>
        <td class="font-mono">${pct}%</td>
        <td style="width: 40%">
          <div class="ft-bar-wrap">
            <div class="ft-bar-track">
              <div class="ft-bar-fill" style="width:${width}%;background-color:${color}"></div>
            </div>
          </div>
        </td>
      </tr>
    `;
  }).join('');

  document.getElementById('breakdownTable').innerHTML = `
    <table class="breakdown-table">
      <thead>
        <tr>
          <th>Failure Type</th>
          <th>Jumlah Kasus</th>
          <th>% dari Failure</th>
          <th>Proporsi</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}
