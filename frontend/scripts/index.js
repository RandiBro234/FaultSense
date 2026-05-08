/* ==========================================
   SINGLE PAGE APPLICATION SCRIPT
   ==========================================

   JavaScript untuk halaman single-page dengan semua section
*/

import { renderNavbar } from '../components/navbar.js';
import { renderFooter } from '../components/footer.js';
import { API_BASE_URL, formatDateTime, getProbabilityColor, getStatusBadgeClass } from './main.js';

// Render navbar dan footer
renderNavbar('home');
renderFooter();

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
  document.getElementById('probFill').style.width = '12%';
}, 500);

loadHomeStats();

// ==========================================
// PREDICT SECTION
// ==========================================

const FAILURE_RECOMMENDATIONS = {
  TWF: 'Segera ganti alat potong (tool) yang sudah aus. Periksa jadwal maintenance rutin untuk mencegah keausan berlebih.',
  HDF: 'Periksa sistem pendingin mesin. Pastikan sirkulasi udara lancar dan tidak ada penyumbatan pada heat sink atau kipas.',
  PWF: 'Periksa suplai daya dan kondisi motor. Pastikan tegangan listrik stabil dan tidak ada komponen kelistrikan yang rusak.',
  OSF: 'Kurangi beban kerja mesin atau tingkatkan kapasitas. Hindari operasi di luar spesifikasi teknis yang direkomendasikan.',
  RNF: 'Lakukan inspeksi menyeluruh pada semua komponen mesin. Kegagalan ini bersifat acak dan memerlukan pengecekan komprehensif.'
};

document.getElementById('btnPredict').addEventListener('click', handlePredict);
document.getElementById('btnReset').addEventListener('click', resetForm);

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

  btn.disabled = true;
  spinner.classList.remove('hidden');
  btnText.textContent = 'Memproses...';
  errorMsg.textContent = '';

  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!response.ok) throw new Error('Gagal melakukan prediksi');

    const data = await response.json();
    renderPredictionResult(data);
  } catch (error) {
    errorMsg.textContent = 'Terjadi kesalahan: ' + error.message;
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

  const recommendation = data.failure_type
    ? FAILURE_RECOMMENDATIONS[data.failure_type]
    : 'Tidak ada tindakan khusus diperlukan. Lanjutkan operasi normal dan pantau kondisi mesin secara berkala.';

  document.getElementById('resultRecommendation').textContent = recommendation;
  document.getElementById('checkedAt').textContent = 'Prediksi dilakukan pada ' + formatDateTime(data.checked_at);

  resultCard.classList.remove('hidden');
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
}

// ==========================================
// HISTORY SECTION
// ==========================================

loadHistory();

document.getElementById('btnFilter').addEventListener('click', loadHistory);
document.getElementById('btnResetFilter').addEventListener('click', resetFilters);

async function loadHistory() {
  const status = document.getElementById('filterStatus').value;
  const failureType = document.getElementById('filterType').value;
  const limit = document.getElementById('filterLimit').value;

  let url = `${API_BASE_URL}/history?limit=${limit}`;
  if (status) url += `&status=${status}`;
  if (failureType) url += `&failure_type=${failureType}`;

  document.getElementById('tableBody').innerHTML = '<div class="loading-state">Memuat data...</div>';

  try {
    const response = await fetch(url);
    const data = await response.json();
    renderHistoryTable(data);
  } catch (error) {
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

  const rows = data.map(d => {
    const prob = d.probability_failure;
    const probPct = Math.round(prob * 100);
    const color = getProbabilityColor(prob);
    const badgeClass = getStatusBadgeClass(d.status);

    const ftTag = d.failure_type
      ? `<span class="badge badge-neutral">${d.failure_type}</span>`
      : '<span style="color:var(--text-muted)">—</span>';

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
}

function resetFilters() {
  document.getElementById('filterStatus').value = '';
  document.getElementById('filterType').value = '';
  document.getElementById('filterLimit').value = '50';
  loadHistory();
}

// ==========================================
// ANALYTICS SECTION
// ==========================================

const FT_COLORS = {
  TWF: '#ef4444',
  HDF: '#f97316',
  PWF: '#3b82f6',
  OSF: '#a855f7',
  RNF: '#6b7280'
};

let donutChart, barChart;

loadAnalytics();

document.getElementById('btnRefresh').addEventListener('click', loadAnalytics);

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
        backgroundColor: ['#10b981', '#ef4444'],
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
            font: { family: 'DM Sans', size: 12 },
            padding: 16,
            color: '#94a3b8'
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
            font: { family: 'DM Mono', size: 11 },
            color: '#94a3b8'
          },
          grid: { color: '#334155' }
        },
        x: {
          ticks: {
            font: { family: 'DM Mono', size: 11 },
            color: '#94a3b8'
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
    const color = FT_COLORS[ft] || '#6b7280';
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
