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
        <div class="footer-left">
          <strong>FaultSense</strong> — Predictive Maintenance System
          <span class="text-muted"> · AI4I 2020 Dataset</span>
        </div>
        <div class="footer-links">
          <a href="http://localhost:8000/docs" target="_blank">API Docs</a>
          <a href="http://localhost:8000/health" target="_blank">Health Check</a>
        </div>
      </div>
    </footer>
  `;

  // Insert footer di akhir body
  document.body.insertAdjacentHTML('beforeend', footerHTML);
}
