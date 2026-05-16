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

// Pagination state
let currentPage = 1;
let itemsPerPage = 10;
let allData = [];
let filteredData = [];

// Load history on page load
loadHistory();

// Event listeners
document.getElementById('btnFilter').addEventListener('click', loadHistory);
document.getElementById('btnResetFilter').addEventListener('click', resetFilters);
document.getElementById('btnExportCSV').addEventListener('click', exportToCSV);

// Load history function
async function loadHistory() {
  const status = document.getElementById('filterStatus').value;
  const failureType = document.getElementById('filterType').value;
  const limit = document.getElementById('filterLimit').value;
  const dateStart = document.getElementById('filterDateStart').value;
  const dateEnd = document.getElementById('filterDateEnd').value;

  console.log('=== LOAD HISTORY START ===');
  console.log('Filter - Status:', status, 'Type:', failureType, 'Limit:', limit);

  let url = `${API_BASE_URL}/history?limit=${limit}`;
  if (status) url += `&status=${status}`;
  if (failureType) url += `&failure_type=${failureType}`;

  console.log('Fetching URL:', url);

  document.getElementById('tableBody').innerHTML = '<div class="loading-state">Memuat data...</div>';

  try {
    const response = await fetch(url);
    const data = await response.json();

    console.log('Data received:', data.length, 'records');

    // Store all data
    allData = data;

    // Apply date filter if set
    filteredData = filterByDateRange(data, dateStart, dateEnd);

    console.log('After date filter:', filteredData.length, 'records');

    // Reset to page 1
    currentPage = 1;

    renderTable(filteredData);
  } catch (error) {
    console.error('Error loading history:', error);
    document.getElementById('tableBody').innerHTML = `
      <div class="empty-state">
        <p class="empty-state-text">Tidak dapat terhubung ke API.</p>
      </div>
    `;
  }
}

// Filter by date range
function filterByDateRange(data, dateStart, dateEnd) {
  if (!dateStart && !dateEnd) return data;

  return data.filter(d => {
    const checkedDate = new Date(d.checked_at).toISOString().split('T')[0];

    if (dateStart && dateEnd) {
      return checkedDate >= dateStart && checkedDate <= dateEnd;
    } else if (dateStart) {
      return checkedDate >= dateStart;
    } else if (dateEnd) {
      return checkedDate <= dateEnd;
    }

    return true;
  });
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

  // Pagination calculation
  const totalPages = Math.ceil(data.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = startIndex + itemsPerPage;
  const paginatedData = data.slice(startIndex, endIndex);

  // Generate table rows
  const rows = paginatedData.map(d => {
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

  // Debug log
  console.log('Total data:', data.length);
  console.log('Items per page:', itemsPerPage);
  console.log('Total pages:', totalPages);
  console.log('Current page:', currentPage);

  // Render pagination controls
  renderPagination(totalPages);
}

// Render pagination controls
function renderPagination(totalPages) {
  console.log('renderPagination called with totalPages:', totalPages);

  // Clear previous pagination
  const paginationContainer = document.getElementById('paginationContainer');
  paginationContainer.innerHTML = '';

  if (totalPages <= 1) {
    console.log('Pagination skipped: totalPages <= 1');
    return;
  }

  const maxVisiblePages = 5; // Jumlah maksimal nomor halaman yang ditampilkan
  let startPage = Math.max(1, currentPage - Math.floor(maxVisiblePages / 2));
  let endPage = Math.min(totalPages, startPage + maxVisiblePages - 1);

  // Adjust startPage jika endPage sudah mentok di akhir
  if (endPage - startPage < maxVisiblePages - 1) {
    startPage = Math.max(1, endPage - maxVisiblePages + 1);
  }

  // Generate page numbers
  let pageNumbers = '';

  // Tombol halaman pertama jika tidak terlihat
  if (startPage > 1) {
    pageNumbers += `<button class="pagination-btn page-number" data-page="1">1</button>`;
    if (startPage > 2) {
      pageNumbers += `<span class="pagination-ellipsis">...</span>`;
    }
  }

  // Tombol halaman yang terlihat
  for (let i = startPage; i <= endPage; i++) {
    const activeClass = i === currentPage ? 'active' : '';
    pageNumbers += `<button class="pagination-btn page-number ${activeClass}" data-page="${i}">${i}</button>`;
  }

  // Tombol halaman terakhir jika tidak terlihat
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

  console.log('Inserting pagination HTML into paginationContainer');
  paginationContainer.innerHTML = paginationHTML;
  console.log('Pagination HTML inserted');

  // Add event listeners untuk Prev dan Next
  const btnPrev = document.getElementById('btnPrevPage');
  const btnNext = document.getElementById('btnNextPage');

  if (btnPrev) {
    btnPrev.addEventListener('click', () => {
      if (currentPage > 1) {
        currentPage--;
        renderTable(filteredData);
      }
    });
  }

  if (btnNext) {
    btnNext.addEventListener('click', () => {
      if (currentPage < totalPages) {
        currentPage++;
        renderTable(filteredData);
      }
    });
  }

  // Add event listeners untuk nomor halaman
  const pageButtons = document.querySelectorAll('.pagination-btn.page-number');
  console.log('Page buttons found:', pageButtons.length);
  pageButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const page = parseInt(btn.getAttribute('data-page'));
      if (page !== currentPage) {
        currentPage = page;
        renderTable(filteredData);
      }
    });
  });
}

// Reset filters function
function resetFilters() {
  document.getElementById('filterStatus').value = '';
  document.getElementById('filterType').value = '';
  document.getElementById('filterLimit').value = '50';
  document.getElementById('filterDateStart').value = '';
  document.getElementById('filterDateEnd').value = '';
  currentPage = 1;
  loadHistory();
}

// Export to CSV function
function exportToCSV() {
  if (filteredData.length === 0) {
    alert('Tidak ada data untuk diekspor.');
    return;
  }

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
  const rows = filteredData.map(d => [
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
}
