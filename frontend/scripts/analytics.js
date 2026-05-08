/* ==========================================
   ANALYTICS PAGE SCRIPT
   ==========================================

   JavaScript untuk halaman analytics/dashboard
*/

import { renderNavbarForPages } from '../components/navbar.js';
import { renderFooter } from '../components/footer.js';
import { API_BASE_URL, formatDateTime } from './main.js';

// Render navbar dan footer
renderNavbarForPages('analytics');
renderFooter();

// Failure type colors
const FT_COLORS = {
  TWF: '#ef4444',
  HDF: '#f97316',
  PWF: '#3b82f6',
  OSF: '#a855f7',
  RNF: '#6b7280'
};

let donutChart, barChart;

// Load analytics on page load
loadAnalytics();

// Event listener
document.getElementById('btnRefresh').addEventListener('click', loadAnalytics);

// Load analytics function
async function loadAnalytics() {
  try {
    const response = await fetch(`${API_BASE_URL}/analytics`);
    const data = await response.json();
    renderAnalytics(data);
  } catch (error) {
    console.error('Error loading analytics:', error);
    document.getElementById('statTotal').textContent = 'Error';
  }
}

// Render analytics function
function renderAnalytics(data) {
  const total = data.total_predictions;
  const normal = data.total_normal;
  const failure = data.total_failure;

  // Last checked
  document.getElementById('lastChecked').textContent = formatDateTime(data.latest_checked_at);

  // Stats
  document.getElementById('statTotal').textContent = total;
  document.getElementById('statNormal').textContent = normal;
  document.getElementById('statFailure').textContent = failure;
  document.getElementById('statRate').textContent = data.failure_rate + '%';

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
