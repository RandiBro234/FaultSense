# FaultSense - Predictive Maintenance System

![FaultSense Banner](./refrensi%20design/landing%20page.png)

**FaultSense** adalah sistem prediksi kerusakan mesin industri berbasis machine learning yang menggunakan **Cascade Classification** untuk mendeteksi kondisi mesin dan mengidentifikasi jenis kegagalan secara otomatis.

---

## 📋 Deskripsi Project

FaultSense mengimplementasikan sistem prediksi dua tahap:
1. **Binary Classification** - Mendeteksi apakah mesin dalam kondisi Normal atau Failure
2. **Multi-class Classification** - Jika terdeteksi Failure, sistem mengidentifikasi jenis kegagalan spesifik (TWF, HDF, PWF, OSF, RNF)

Sistem ini dibangun untuk membantu maintenance engineer melakukan **predictive maintenance** dan mencegah downtime mesin yang tidak terduga.

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ POST /predict   │ ──▶ │ Binary RF + thr  │ ──▶ │ Multiclass RF    │
│ (sensor data)   │     │ Normal / Failure │     │ (jenis failure)  │
└─────────────────┘     └──────────────────┘     └──────────────────┘
                                  │                        │
                                  ▼                        ▼
                          ┌──────────────────────────────────────┐
                          │ prediction_logs (PostgreSQL)         │
                          └──────────────────────────────────────┘
```

---

## ✨ Features

### 🔮 Prediksi Real-time
- Input data sensor mesin (temperature, speed, torque, tool wear)
- Hasil prediksi instan dengan probability score
- Rekomendasi tindakan berdasarkan jenis failure

### 📊 Analytics Dashboard
- Visualisasi distribusi status (Normal vs Failure)
- Breakdown per jenis failure dengan chart interaktif
- Metrik performa model (ROC-AUC, Recall, F1-Score)

### 📋 History Management
- Riwayat lengkap semua prediksi
- Filter berdasarkan status dan failure type
- Export data untuk analisis lebih lanjut

### 🎨 Modern UI/UX
- Dark blue theme yang modern dan profesional
- Responsive design (mobile, tablet, desktop)
- Smooth animations dan transitions
- Component-based architecture

---

## 🛠️ Tech Stack

### Frontend
- **HTML5** - Semantic markup
- **CSS3** - Modern styling dengan CSS Variables
- **Vanilla JavaScript (ES6+)** - Modular architecture dengan ES6 modules
- **Chart.js** - Data visualization untuk analytics

### Backend
- **FastAPI** - Python web framework
- **Scikit-learn** - Machine learning models
- **PostgreSQL** - Database untuk menyimpan prediksi
- **Docker Compose** - Container orchestration

### Machine Learning
- **Random Forest Classifier** - Binary classification (Normal/Failure)
- **Random Forest Classifier** - Multi-class classification (Failure types)
- **Dataset**: AI4I 2020 Predictive Maintenance Dataset

---

## 📁 Struktur Folder

### Frontend Structure
```
frontend/
├── index.html                 # Landing page
├── pages/
│   ├── dashboard.html         # Halaman prediksi mesin
│   ├── history.html           # Halaman riwayat prediksi
│   └── analytics.html         # Halaman analytics & visualisasi
├── components/
│   ├── navbar.js              # Dynamic navbar component
│   └── footer.js              # Dynamic footer component
├── scripts/
│   └── main.js                # Shared utilities & helper functions
├── styles/
│   ├── variables.css          # CSS Variables (colors, spacing, typography)
│   ├── global.css             # Global styles & reset
│   ├── layout.css             # Layout utilities (grid, flexbox)
│   └── components.css         # Reusable component styles
└── TESTING_CHECKLIST.md       # Testing checklist
```

### Backend Structure
```
app/
├── main.py                    # FastAPI application
├── models/                    # ML model artifacts
│   ├── model_binary_rf.pkl
│   ├── model_multiclass_rf.pkl
│   ├── scaler.pkl
│   └── threshold.pkl
└── notebook/
    └── training.ipynb         # Model training notebook
```

---

## 🚀 Cara Menjalankan Project

### Prerequisites
- Docker & Docker Compose v2 (recommended)
- Python 3.8+ (untuk development lokal)
- Browser modern (Chrome, Firefox, Edge)

### 1. Quick Start dengan Docker Compose (Recommended)

```bash
# Clone repository
git clone https://github.com/RandiBro234/FaultSense.git
cd FaultSense

# Setup environment
cp .env.example .env

# Start services
docker compose up --build
```

**Services yang berjalan:**
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- PostgreSQL: localhost:5433 (host) / 5432 (container)

### 2. Setup Frontend

**Opsi A: Menggunakan Live Server (Recommended)**
```bash
# Install Live Server (VS Code Extension)
# Atau gunakan Python HTTP server
cd frontend
python -m http.server 8080

# Buka browser: http://localhost:8080
```

**Opsi B: Menggunakan Node.js HTTP Server**
```bash
npx http-server frontend -p 8080
```

### 3. Setup Backend Lokal (Tanpa Docker)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup database (PostgreSQL harus running)
export DATABASE_URL=postgresql://faultsense_user:faultsense_pass@localhost:5432/faultsense_db

# Run server
uvicorn app.main:app --reload
```

### 4. Akses Aplikasi

- **Landing Page**: http://localhost:8080/
- **Dashboard Prediksi**: http://localhost:8080/pages/dashboard.html
- **History**: http://localhost:8080/pages/history.html
- **Analytics**: http://localhost:8080/pages/analytics.html
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

---

## 🏗️ Frontend Architecture

### Design System

**Color Palette (Dark Blue Theme)**
```css
--primary-bg: #0f172a        /* Background utama (slate 900) */
--secondary-bg: #1e293b      /* Card background (slate 800) */
--accent-blue: #3b82f6       /* Primary accent (blue 500) */
--text-primary: #f1f5f9      /* Heading (slate 100) */
--text-secondary: #94a3b8    /* Body text (slate 400) */
--success: #10b981           /* Success state (green 500) */
--danger: #ef4444            /* Danger state (red 500) */
--warning: #f59e0b           /* Warning state (amber 500) */
```

**Typography**
- **Body**: DM Sans (400, 500, 600)
- **Headings**: DM Serif Display
- **Code/Numbers**: DM Mono

**Spacing System**
- Base unit: 4px (0.25rem)
- Scale: xs(4px), sm(8px), md(16px), lg(24px), xl(32px), 2xl(48px), 3xl(64px)

### Component Architecture

**Reusable Components:**
1. **Navbar** - Fixed navbar dengan active state
2. **Footer** - Simple footer dengan links
3. **Card** - Container untuk konten
4. **Button** - Primary, secondary, outline variants
5. **Badge** - Status indicators (success, danger, warning)
6. **Form** - Input, select, label dengan validation
7. **Table** - Responsive table dengan hover state

**JavaScript Modules:**
- `navbar.js` - Dynamic navbar rendering
- `footer.js` - Dynamic footer rendering
- `main.js` - Utility functions (formatDateTime, API calls, color mapping)

### Code Organization

**CSS Architecture (ITCSS-inspired):**
1. **variables.css** - Design tokens (colors, spacing, typography)
2. **global.css** - Reset & base styles
3. **layout.css** - Layout utilities (grid, flexbox, container)
4. **components.css** - Component styles (navbar, button, card, etc)
5. **Page-specific styles** - Inline `<style>` untuk styles unik per halaman

**JavaScript Pattern:**
- ES6 Modules untuk code splitting
- Single Responsibility Principle
- Reusable utility functions
- Minimal external dependencies (hanya Chart.js)

---

## 📸 Screenshots

### Landing Page
![Landing Page](./refrensi%20design/landing%20page.png)

### Dashboard Prediksi
![Dashboard](./refrensi%20design/dashboard.png)

---

## 📝 API Endpoints

### Prediction
```http
POST /predict
Content-Type: application/json

{
  "type": "L",
  "air_temperature": 298.1,
  "process_temperature": 308.6,
  "rotational_speed": 1500,
  "torque": 40.0,
  "tool_wear": 50
}
```

**Response:**
```json
{
  "status": "Normal",
  "failure_type": null,
  "probability_failure": 0.0123,
  "recommendation": "Mesin dalam kondisi normal. Lanjutkan monitoring rutin.",
  "checked_at": "2026-05-07T12:34:56.000000"
}
```

### History
```http
GET /history?limit=50&status=Failure&failure_type=TWF
```

### Analytics
```http
GET /analytics
```

Dokumentasi lengkap: http://localhost:8000/docs

---

## 🧪 Training Model

### Setup Training Environment
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Launch Jupyter Lab
jupyter lab notebook/training.ipynb
```

### Model Artifacts
Training menghasilkan artefak ke folder `models/`:
- `model_binary_rf.pkl` — RandomForest binary (Normal vs Failure)
- `model_multiclass_rf.pkl` — RandomForest multiclass (jenis failure)
- `scaler.pkl` — StandardScaler untuk fitur numerik
- `ohe_columns.pkl` — Daftar kolom hasil one-hot encoding
- `threshold.pkl` — Threshold optimal dari tuning

### MLflow Tracking
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5002
```

---

## 📊 Dataset Notes (AI4I 2020)

Dataset memiliki karakteristik khusus:
- 339 baris dengan `Machine failure = 1` dari total 10000
- 9 baris failure tanpa failure type spesifik (di-drop)
- 24 baris dengan multiple failure types (di-resolve dengan prioritas)
- 18 dari 19 baris `RNF = 1` memiliki `Machine failure = 0` (anomali dataset)

---

## 🔮 Future Improvements

### Features
- [ ] Real-time monitoring dengan WebSocket
- [ ] Export data ke CSV/Excel
- [ ] User authentication & authorization
- [ ] Multi-language support (ID/EN)
- [ ] Dark/Light theme toggle
- [ ] Notification system untuk critical failures
- [ ] Mobile app (PWA)

### Technical
- [ ] Add unit tests (Jest)
- [ ] Add E2E tests (Playwright)
- [ ] Implement service worker untuk offline support
- [ ] Add loading skeletons
- [ ] Optimize bundle size
- [ ] Add error boundary
- [ ] Implement retry logic untuk API calls
- [ ] Add accessibility improvements (ARIA labels, keyboard navigation)

### UI/UX
- [ ] Add onboarding tutorial
- [ ] Improve mobile navigation (hamburger menu)
- [ ] Add data visualization animations
- [ ] Implement drag-and-drop untuk file upload
- [ ] Add comparison mode (compare multiple predictions)

---

## 🧪 Testing

Lihat `frontend/TESTING_CHECKLIST.md` untuk checklist testing lengkap.

**Testing Areas:**
- Navigation & routing
- Component rendering
- CSS imports & styling
- JavaScript modules
- Responsive design (mobile/tablet/desktop)
- Dark theme consistency
- API integration

---

## 👥 Contributors

- **Randi** - Backend & ML Development
- **[Your Name]** - Frontend Architecture & UI/UX

---

## 📄 License

Proyek tugas akhir / penelitian. Belum ada lisensi formal.

---

## 🙏 Acknowledgments

- Dataset: [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
- Fonts: [DM Sans, DM Serif Display, DM Mono](https://fonts.google.com/)
- Icons: Emoji (native)
- Charts: [Chart.js](https://www.chartjs.org/)

---

## 📞 Contact

Untuk pertanyaan atau feedback, silakan buka issue di repository ini.

**Built with ❤️ using Vanilla JavaScript**
