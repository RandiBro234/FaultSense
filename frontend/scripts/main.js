/* ==========================================
   MAIN.JS - Shared Utilities
   ==========================================

   File ini berisi fungsi-fungsi yang dipakai
   di semua halaman:
   - API base URL
   - Format tanggal/waktu
   - Helper functions
*/

// ===== API CONFIGURATION =====
export const API_BASE_URL = 'http://localhost:8000';

// ===== DATE & TIME UTILITIES =====

/**
 * Format ISO date string ke format Indonesia
 * @param {string} isoString - ISO date string (e.g., "2024-01-15T10:30:00")
 * @returns {string} - Formatted date (e.g., "15 Jan 2024, 10:30")
 */
export function formatDateTime(isoString) {
  if (!isoString) return '—';

  const date = new Date(isoString);

  const dateStr = date.toLocaleDateString('id-ID', {
    day: '2-digit',
    month: 'short',
    year: 'numeric'
  });

  const timeStr = date.toLocaleTimeString('id-ID', {
    hour: '2-digit',
    minute: '2-digit'
  });

  return `${dateStr}, ${timeStr}`;
}

/**
 * Format date only (tanpa waktu)
 * @param {string} isoString - ISO date string
 * @returns {string} - Formatted date (e.g., "15 Jan 2024")
 */
export function formatDate(isoString) {
  if (!isoString) return '—';

  const date = new Date(isoString);
  return date.toLocaleDateString('id-ID', {
    day: '2-digit',
    month: 'short',
    year: 'numeric'
  });
}

/**
 * Format time only (tanpa tanggal)
 * @param {string} isoString - ISO date string
 * @returns {string} - Formatted time (e.g., "10:30")
 */
export function formatTime(isoString) {
  if (!isoString) return '—';

  const date = new Date(isoString);
  return date.toLocaleTimeString('id-ID', {
    hour: '2-digit',
    minute: '2-digit'
  });
}

// ===== NUMBER UTILITIES =====

/**
 * Format number dengan desimal
 * @param {number} num - Number to format
 * @param {number} decimals - Jumlah desimal (default: 2)
 * @returns {string} - Formatted number
 */
export function formatNumber(num, decimals = 2) {
  if (num === null || num === undefined || isNaN(num)) return '—';
  return num.toFixed(decimals);
}

/**
 * Format percentage
 * @param {number} value - Value (0-1 atau 0-100)
 * @param {boolean} isDecimal - Apakah value dalam bentuk desimal (0-1)
 * @returns {string} - Formatted percentage (e.g., "85%")
 */
export function formatPercentage(value, isDecimal = true) {
  if (value === null || value === undefined || isNaN(value)) return '—';

  const percentage = isDecimal ? value * 100 : value;
  return `${Math.round(percentage)}%`;
}

// ===== COLOR UTILITIES =====

/**
 * Get color berdasarkan probability value
 * @param {number} probability - Probability value (0-1)
 * @returns {string} - CSS color variable
 */
export function getProbabilityColor(probability) {
  if (probability < 0.3) return 'var(--success)';
  if (probability < 0.6) return 'var(--warning)';
  return 'var(--danger)';
}

/**
 * Get status badge class
 * @param {string} status - Status ("Normal" atau "Failure")
 * @returns {string} - CSS class name
 */
export function getStatusBadgeClass(status) {
  return status === 'Normal' ? 'badge-success' : 'badge-danger';
}

// ===== API UTILITIES =====

/**
 * Fetch data dari API dengan error handling
 * @param {string} endpoint - API endpoint (e.g., "/predict")
 * @param {object} options - Fetch options
 * @returns {Promise<object>} - Response data
 */
export async function fetchAPI(endpoint, options = {}) {
  try {
    const url = `${API_BASE_URL}${endpoint}`;
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers
      },
      ...options
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail?.[0]?.msg || 'Terjadi kesalahan pada server');
    }

    return await response.json();
  } catch (error) {
    console.error('API Error:', error);
    throw error;
  }
}

// ===== DOM UTILITIES =====

/**
 * Show element
 * @param {string|HTMLElement} element - Element ID atau element object
 */
export function showElement(element) {
  const el = typeof element === 'string' ? document.getElementById(element) : element;
  if (el) el.classList.remove('hidden');
}

/**
 * Hide element
 * @param {string|HTMLElement} element - Element ID atau element object
 */
export function hideElement(element) {
  const el = typeof element === 'string' ? document.getElementById(element) : element;
  if (el) el.classList.add('hidden');
}

/**
 * Toggle element visibility
 * @param {string|HTMLElement} element - Element ID atau element object
 */
export function toggleElement(element) {
  const el = typeof element === 'string' ? document.getElementById(element) : element;
  if (el) el.classList.toggle('hidden');
}

// ===== VALIDATION UTILITIES =====

/**
 * Validate form input values
 * @param {object} values - Object dengan key-value pairs
 * @returns {boolean} - True jika semua valid
 */
export function validateInputs(values) {
  return Object.values(values).every(value => {
    return value !== null && value !== undefined && value !== '' && !isNaN(value);
  });
}
