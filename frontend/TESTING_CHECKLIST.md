/* ==========================================
   TESTING CHECKLIST
   ==========================================

   File ini berisi checklist untuk testing manual
   semua halaman setelah refactor.
*/

# TESTING CHECKLIST - FaultSense Frontend Refactor

## 1. FILE STRUCTURE VERIFICATION ✅
- [x] styles/variables.css exists
- [x] styles/global.css exists
- [x] styles/layout.css exists
- [x] styles/components.css exists
- [x] components/navbar.js exists
- [x] components/footer.js exists
- [x] scripts/main.js exists
- [x] pages/dashboard.html exists
- [x] pages/history.html exists
- [x] pages/analytics.html exists
- [x] index.html (refactored) exists

## 2. NAVIGATION TESTING
### Landing Page (index.html)
- [ ] Navbar logo links to index.html
- [ ] "Predict" link → pages/dashboard.html
- [ ] "History" link → pages/history.html
- [ ] "Analytics" link → pages/analytics.html
- [ ] "API Docs" button → http://localhost:8000/docs (new tab)
- [ ] Hero "Mulai Prediksi" button → pages/dashboard.html
- [ ] Hero "Lihat Dashboard" button → pages/analytics.html
- [ ] Page cards navigation works
- [ ] Footer links work

### Dashboard Page (pages/dashboard.html)
- [ ] Navbar logo links to ../index.html
- [ ] Breadcrumb "Home" link → ../index.html
- [ ] All navbar links work with correct paths
- [ ] Active state on "Predict" link
- [ ] Footer renders correctly

### History Page (pages/history.html)
- [ ] Navbar logo links to ../index.html
- [ ] Breadcrumb "Home" link → ../index.html
- [ ] All navbar links work
- [ ] Active state on "History" link
- [ ] Footer renders correctly

### Analytics Page (pages/analytics.html)
- [ ] Navbar logo links to ../index.html
- [ ] Breadcrumb "Home" link → ../index.html
- [ ] All navbar links work
- [ ] Active state on "Analytics" link
- [ ] Footer renders correctly

## 3. COMPONENT RENDERING
### Navbar Component
- [ ] Logo renders with animated dot
- [ ] All menu links render
- [ ] Active state highlights correctly
- [ ] API Docs button styled correctly
- [ ] Navbar is fixed at top
- [ ] Backdrop blur effect works

### Footer Component
- [ ] Footer renders at bottom
- [ ] "FaultSense" text bold
- [ ] API Docs link works
- [ ] Health Check link works
- [ ] Layout: left text, right links

## 4. CSS IMPORTS & STYLING
### All Pages Should Have:
- [ ] Dark blue theme (#0f172a background)
- [ ] Correct text colors (slate palette)
- [ ] Accent blue (#3b82f6) for interactive elements
- [ ] Consistent spacing using CSS variables
- [ ] DM Sans font for body
- [ ] DM Serif Display for headings
- [ ] DM Mono for code/numbers

### Specific Checks:
- [ ] Cards have secondary-bg (#1e293b)
- [ ] Borders use --border color (#334155)
- [ ] Buttons have hover effects
- [ ] Badges have correct colors (success/danger/warning)
- [ ] Forms styled consistently
- [ ] Tables styled consistently

## 5. JAVASCRIPT MODULES
### Import Checks:
- [ ] No console errors for missing modules
- [ ] navbar.js imports successfully
- [ ] footer.js imports successfully
- [ ] main.js utilities import successfully
- [ ] ES6 modules work (type="module")

### Functionality Checks:
- [ ] Navbar renders on page load
- [ ] Footer renders on page load
- [ ] API calls work (if backend running)
- [ ] formatDateTime() works correctly
- [ ] getProbabilityColor() returns correct colors
- [ ] Form validation works (dashboard)
- [ ] Filter functionality works (history)
- [ ] Charts render (analytics - requires Chart.js)

## 6. RESPONSIVE DESIGN
### Mobile (< 768px)
- [ ] Navbar menu hidden (expected behavior)
- [ ] Hero section stacks vertically
- [ ] Stats grid: 2 columns
- [ ] Features grid: 1 column
- [ ] Pages grid: 1 column
- [ ] Form inputs full width
- [ ] Tables hide extra columns
- [ ] Cards padding reduced
- [ ] Text sizes readable

### Tablet (768px - 1024px)
- [ ] Layout adapts smoothly
- [ ] Grids adjust appropriately
- [ ] No horizontal scroll
- [ ] Touch targets adequate size

### Desktop (> 1024px)
- [ ] Full layout displays
- [ ] Max-width container (1200px)
- [ ] All columns visible
- [ ] Proper spacing

## 7. DARK THEME CONSISTENCY
- [ ] All backgrounds dark
- [ ] Text readable (good contrast)
- [ ] No white flashes
- [ ] Hover states visible
- [ ] Focus states visible
- [ ] Scrollbar styled dark
- [ ] Selection color styled

## 8. BROWSER COMPATIBILITY
- [ ] Chrome/Edge (Chromium)
- [ ] Firefox
- [ ] Safari (if available)
- [ ] No console errors
- [ ] ES6 modules supported

## 9. PERFORMANCE
- [ ] Page loads quickly
- [ ] No layout shift
- [ ] Smooth animations
- [ ] No janky scrolling
- [ ] CSS files cached

## 10. ACCESSIBILITY (Basic)
- [ ] Semantic HTML used
- [ ] Links have descriptive text
- [ ] Buttons have clear labels
- [ ] Form labels associated
- [ ] Color contrast adequate
- [ ] Keyboard navigation works

## KNOWN ISSUES TO FIX:
1. Mobile navbar menu not implemented (hidden by CSS)
2. Need to test with backend API running
3. Chart.js CDN dependency (analytics page)

## NEXT STEPS AFTER TESTING:
1. Fix any broken links/imports
2. Cleanup old files
3. Add README.md
4. Add UI polish & animations
