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

// Animate counter for pipeline stats
function animateCounter(element, target, duration = 2000) {
  const start = 0;
  const increment = target / (duration / 16);
  let current = start;

  const timer = setInterval(() => {
    current += increment;
    if (current >= target) {
      element.textContent = target.toFixed(2);
      clearInterval(timer);
    } else {
      element.textContent = current.toFixed(2);
    }
  }, 16);
}

// Start animation immediately when DOM is ready
function startStatsAnimation() {
  const statNumbers = document.querySelectorAll('.stat-number[data-target]');
  console.log('Found stat numbers:', statNumbers.length);

  statNumbers.forEach(stat => {
    const target = parseFloat(stat.getAttribute('data-target'));
    console.log('Animating stat to:', target);
    if (target && !isNaN(target)) {
      animateCounter(stat, target);
    }
  });
}

// Run animation after a short delay to ensure DOM is ready
setTimeout(() => {
  startStatsAnimation();
}, 1000);

// Also use Intersection Observer as backup
const observerOptions = {
  threshold: 0.1,
  rootMargin: '0px'
};

const statsObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      const statNumbers = entry.target.querySelectorAll('.stat-number[data-target]');
      statNumbers.forEach(stat => {
        const target = parseFloat(stat.getAttribute('data-target'));
        const currentValue = parseFloat(stat.textContent);
        // Only animate if still at 0
        if (target && !isNaN(target) && currentValue === 0) {
          animateCounter(stat, target);
        }
      });
      statsObserver.unobserve(entry.target);
    }
  });
}, observerOptions);

// Observe pipeline stats section
const pipelineStatsSection = document.querySelector('.pipeline-stats-section');
if (pipelineStatsSection) {
  statsObserver.observe(pipelineStatsSection);
}

loadStats();
