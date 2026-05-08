/* ==========================================
   DASHBOARD PAGE SCRIPT
   ==========================================

   JavaScript untuk halaman dashboard/predict
*/

import { renderNavbarForPages } from '../components/navbar.js';
import { renderFooter } from '../components/footer.js';
import { API_BASE_URL, formatDateTime, getProbabilityColor } from './main.js';

// Render navbar dan footer
renderNavbarForPages('predict');
renderFooter();

// Default values
const defaults = {
  air_temperature: 298.1,
  process_temperature: 308.6,
  rotational_speed: 1500,
  torque: 40.0,
  tool_wear: 50
};

// Reset form
document.getElementById('btnReset').addEventListener('click', () => {
  document.getElementById('type').value = 'L';
  Object.keys(defaults).forEach(key => {
    document.getElementById(key).value = defaults[key];
  });
  document.getElementById('resultCard').classList.remove('visible');
  document.getElementById('errorMsg').style.display = 'none';
});

// Predict function
document.getElementById('btnPredict').addEventListener('click', async () => {
  const btn = document.getElementById('btnPredict');
  const spinner = document.getElementById('spinner');
  const btnText = document.getElementById('btnText');
  const errorMsg = document.getElementById('errorMsg');

  // Collect input
  const payload = {
    type: document.getElementById('type').value,
    air_temperature: parseFloat(document.getElementById('air_temperature').value),
    process_temperature: parseFloat(document.getElementById('process_temperature').value),
    rotational_speed: parseInt(document.getElementById('rotational_speed').value),
    torque: parseFloat(document.getElementById('torque').value),
    tool_wear: parseInt(document.getElementById('tool_wear').value)
  };

  // Validate (exclude 'type' from NaN check since it's a string)
  const numericFields = ['air_temperature', 'process_temperature', 'rotational_speed', 'torque', 'tool_wear'];
  const hasInvalidField = numericFields.some(field => isNaN(payload[field]));

  if (hasInvalidField || !payload.type) {
    errorMsg.textContent = 'Semua field harus diisi dengan nilai valid.';
    errorMsg.style.display = 'block';
    return;
  }

  // Loading state
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
      throw new Error(error.detail?.[0]?.msg || 'Terjadi kesalahan pada server.');
    }

    const data = await response.json();
    renderResult(data);

  } catch (error) {
    errorMsg.textContent = error.message || 'Tidak dapat terhubung ke API.';
    errorMsg.style.display = 'block';
  } finally {
    btn.disabled = false;
    spinner.classList.add('hidden');
    btnText.textContent = 'Jalankan Prediksi';
  }
});

// Render result
function renderResult(data) {
  const card = document.getElementById('resultCard');
  const isNormal = data.status === 'Normal';
  const prob = data.probability_failure;
  const probPct = Math.round(prob * 100);

  // Status
  document.getElementById('resultStatus').textContent = isNormal ? 'Mesin Normal' : 'Terdeteksi Failure';
  document.getElementById('resultStatus').style.color = isNormal ? 'var(--success)' : 'var(--danger)';
  document.getElementById('resultSubtitle').textContent = isNormal
    ? 'Tidak ada indikasi kegagalan'
    : `Jenis: ${data.failure_type}`;

  // Failure Type
  if (data.failure_type) {
    document.getElementById('resultFailureType').innerHTML =
      `<span class="badge badge-danger">${data.failure_type}</span>`;
  } else {
    document.getElementById('resultFailureType').innerHTML =
      `<span style="color:var(--text-muted)">—</span>`;
  }

  // Probability
  document.getElementById('resultProbText').textContent = prob.toFixed(4);

  // Probability bar
  const fill = document.getElementById('probFill');
  const color = getProbabilityColor(prob);
  fill.style.backgroundColor = color;
  document.getElementById('probPct').textContent = probPct + '%';
  document.getElementById('probPct').style.color = color;

  setTimeout(() => {
    fill.style.width = probPct + '%';
  }, 100);

  // Recommendation
  document.getElementById('resultRecommendation').textContent = data.recommendation;

  // Timestamp
  document.getElementById('checkedAt').textContent = `Diperiksa: ${formatDateTime(data.checked_at)}`;

  // Show card
  card.classList.add('visible');
  card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}
