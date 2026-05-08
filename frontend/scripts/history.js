/* ==========================================
   HISTORY PAGE SCRIPT
   ==========================================

   JavaScript untuk halaman history/riwayat prediksi
*/

import { renderNavbarForPages } from '../components/navbar.js';
import { renderFooter } from '../components/footer.js';
import { API_BASE_URL, formatDateTime, getProbabilityColor, getStatusBadgeClass } from './main.js';

// Render navbar dan footer
renderNavbarForPages('history');
renderFooter();

// Load history on page load
loadHistory();

// Event listeners
document.getElementById('btnFilter').addEventListener('click', loadHistory);
document.getElementById('btnReset').addEventListener('click', resetFilters);

// Load history function
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
    renderTable(data);
  } catch (error) {
    document.getElementById('tableBody').innerHTML = `
      <div class="empty-state">
        <p class="empty-state-text">Tidak dapat terhubung ke API.</p>
      </div>
    `;
  }
}

// Render table function
function renderTable(data) {
  // Calculate stats
  const normal = data.filter(d => d.status === 'Normal').length;
  const failure = data.filter(d => d.status === 'Failure').length;
  const rate = data.length > 0 ? ((failure / data.length) * 100).toFixed(1) : 0;

  // Update stats
  document.getElementById('statShown').textContent = data.length;
  document.getElementById('statNormal').textContent = normal;
  document.getElementById('statFailure').textContent = failure;
  document.getElementById('statRate').textContent = rate + '%';
  document.getElementById('tableCount').textContent = `${data.length} record`;

  // Empty state
  if (data.length === 0) {
    document.getElementById('tableBody').innerHTML = `
      <div class="empty-state">
        <p class="empty-state-text">Belum ada data prediksi.</p>
      </div>
    `;
    return;
  }

  // Generate table rows
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

  // Render table
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

// Reset filters function
function resetFilters() {
  document.getElementById('filterStatus').value = '';
  document.getElementById('filterType').value = '';
  document.getElementById('filterLimit').value = '50';
  loadHistory();
}
