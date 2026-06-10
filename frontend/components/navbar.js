/* ==========================================
   NAVBAR COMPONENT
   ==========================================

   Component untuk render navbar secara dinamis
   dengan active state berdasarkan halaman saat ini.

   Usage:
   import { renderNavbar } from './components/navbar.js';
   renderNavbar('predict'); // 'predict', 'history', 'analytics', atau 'home'
*/

/**
 * Render navbar ke dalam DOM
 * @param {string} activePage - Nama halaman aktif ('home', 'predict', 'history', 'analytics')
 */
export function renderNavbar(activePage = 'home') {
  const navbarHTML = `
    <nav class="navbar">
      <div class="navbar-container">
        <!-- Logo -->
        <a href="javascript:void(0)" class="navbar-logo" data-tab="home">
          <span class="navbar-logo-text">
            <span class="navbar-logo-dot"></span>
            FaultSense
          </span>
        </a>

        <!-- Hamburger Toggle -->
        <button class="navbar-toggle" id="navbarToggle" aria-label="Toggle menu">
          <span></span>
          <span></span>
          <span></span>
        </button>

        <!-- Menu Links -->
        <div class="navbar-menu" id="navbarMenu">
          <a href="javascript:void(0)" class="navbar-link ${activePage === 'home' ? 'active' : ''}" data-tab="home">
            Home
          </a>
          <a href="javascript:void(0)" class="navbar-link ${activePage === 'predict' ? 'active' : ''}" data-tab="predict">
            Prediksi
          </a>
          <a href="javascript:void(0)" class="navbar-link ${activePage === 'history' ? 'active' : ''}" data-tab="history">
            History
          </a>
          <a href="javascript:void(0)" class="navbar-link ${activePage === 'analytics' ? 'active' : ''}" data-tab="analytics">
            Analytics
          </a>
        </div>
      </div>
    </nav>
  `;

  // Insert navbar di awal body
  document.body.insertAdjacentHTML('afterbegin', navbarHTML);

  // Toggle hamburger menu
  const toggle = document.getElementById('navbarToggle');
  const menu = document.getElementById('navbarMenu');
  if (toggle && menu) {
    toggle.addEventListener('click', () => {
      menu.classList.toggle('open');
      toggle.classList.toggle('open');
    });

    // Tutup menu kalau klik link
    menu.querySelectorAll('.navbar-link').forEach(link => {
      link.addEventListener('click', () => {
        menu.classList.remove('open');
        toggle.classList.remove('open');
      });
    });
  }
}

/**
 * Render navbar untuk halaman di dalam folder pages/
 * Path relatif disesuaikan (../ untuk kembali ke root)
 * @param {string} activePage - Nama halaman aktif
 */
export function renderNavbarForPages(activePage) {
  const navbarHTML = `
    <nav class="navbar">
      <div class="navbar-container">
        <!-- Logo -->
        <a href="../index.html" class="navbar-logo">
          <span class="navbar-logo-text">
            <span class="navbar-logo-dot"></span>
            FaultSense
          </span>
        </a>

        <!-- Hamburger Toggle -->
        <button class="navbar-toggle" id="navbarToggle" aria-label="Toggle menu">
          <span></span>
          <span></span>
          <span></span>
        </button>

        <!-- Menu Links -->
        <div class="navbar-menu" id="navbarMenu">
          <a href="dashboard.html" class="navbar-link ${activePage === 'predict' ? 'active' : ''}">
            Predict
          </a>
          <a href="history.html" class="navbar-link ${activePage === 'history' ? 'active' : ''}">
            History
          </a>
          <a href="analytics.html" class="navbar-link ${activePage === 'analytics' ? 'active' : ''}">
            Analytics
          </a>
          <a href="http://localhost:8000/docs" target="_blank" class="btn btn-primary btn-sm">
            API Docs
          </a>
        </div>
      </div>
    </nav>
  `;

  // Insert navbar di awal body
  document.body.insertAdjacentHTML('afterbegin', navbarHTML);

  // Toggle hamburger menu
  const toggle = document.getElementById('navbarToggle');
  const menu = document.getElementById('navbarMenu');
  if (toggle && menu) {
    toggle.addEventListener('click', () => {
      menu.classList.toggle('open');
      toggle.classList.toggle('open');
    });

    // Tutup menu kalau klik link
    menu.querySelectorAll('.navbar-link').forEach(link => {
      link.addEventListener('click', () => {
        menu.classList.remove('open');
        toggle.classList.remove('open');
      });
    });
  }
}