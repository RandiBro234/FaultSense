/* ==========================================
   HOME PAGE SCRIPT
   ==========================================

   JavaScript untuk halaman home/landing page
*/

import { renderNavbar } from '../components/navbar.js';
import { renderFooter } from '../components/footer.js';
import { API_BASE_URL } from './main.js';

// Render navbar dan footer
renderNavbar('home');
renderFooter();

// Load stats
async function loadStats() {
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

loadStats();
