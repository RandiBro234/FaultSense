/* ==========================================
   FOOTER COMPONENT
   ==========================================

   Component untuk render footer secara dinamis.

   Usage:
   import { renderFooter } from './components/footer.js';
   renderFooter();
*/

/**
 * Render footer ke dalam DOM
 */
export function renderFooter() {
  const footerHTML = `
    <footer class="footer">
      <div class="footer-container">
        <div class="footer-brand">
          <div class="footer-logo">
            <img src="images/logopens.png" alt="PENS Logo" class="footer-logo-img">
            <span class="footer-logo-text">FaultSense</span>
          </div>
          <p class="footer-desc">
            Sistem monitoring dan deteksi fault motor listrik berbasis AI untuk industri.
          </p>
        </div>

        <div class="footer-links-group">
          <div class="footer-column">
            <h4 class="footer-column-title">Produk</h4>
            <a href="#predict" class="footer-link">Input Data Sensor</a>
            <a href="#predict" class="footer-link">Live Prediction</a>
            <a href="#history" class="footer-link">History Log</a>
            <a href="#analytics" class="footer-link">Analytics</a>
          </div>

          <div class="footer-column">
            <h4 class="footer-column-title">Resources</h4>
            <a href="http://localhost:8000/docs" target="_blank" class="footer-link">API Documentation</a>
            <a href="http://localhost:8000/health" target="_blank" class="footer-link">Health Check</a>
          </div>

          <div class="footer-column">
            <h4 class="footer-column-title">Tentang</h4>
            <p class="footer-info">Politeknik Elektronika Negeri Surabaya</p>
            <p class="footer-info">Dataset: AI4I 2020 Predictive Maintenance</p>
          </div>
        </div>
      </div>

      <div class="footer-bottom">
        <div class="footer-container">
          <p class="footer-copyright">© 2026 FaultSense - PENS. Predictive Maintenance System.</p>
        </div>
      </div>
    </footer>
  `;

  // Insert footer di akhir body
  document.body.insertAdjacentHTML('beforeend', footerHTML);
}
