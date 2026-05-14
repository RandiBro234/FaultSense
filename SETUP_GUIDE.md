# 🚀 Panduan Menjalankan FaultSense (Frontend + Backend + Training)

## 📋 Arsitektur Project

Project ini menggunakan **3 Docker Compose files terpisah**:

1. **docker-compose.yml** - Backend API + Database (PostgreSQL + pgAdmin)
2. **docker-compose.frontend.yml** - Frontend Next.js
3. **docker-compose.training.yml** - Jupyter Lab untuk ML training

---

## 🔧 Cara Menjalankan Project

### Opsi 1: Menggunakan Docker Compose (Recommended)

#### 1. Setup Environment
```bash
cd "C:\SEMESTER 4\(Praktikum) Teknologi Web Service\FaultSense"

# Copy environment file (jika belum ada)
cp .env.example .env

# Edit .env jika perlu (opsional)
```

#### 2. Start Backend + Database
```bash
# Start API + PostgreSQL + pgAdmin
docker compose up -d --build

# Check status
docker compose ps

# Check API health
curl http://localhost:8000/health
```

#### 3. Start Frontend (Opsional)
```bash
# Start frontend Next.js di Docker
docker compose -f docker-compose.frontend.yml up -d --build

# Check status
docker compose -f docker-compose.frontend.yml ps
```

#### 4. Start Training Environment (Opsional)
```bash
# Start Jupyter Lab untuk ML training
docker compose -f docker-compose.training.yml up -d

# Check status
docker compose -f docker-compose.training.yml ps
```

#### 5. Akses Aplikasi
- **Frontend**: http://localhost:3000/ (jika pakai Docker) atau http://localhost:8080/ (jika pakai Python server)
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **pgAdmin**: http://localhost:5050 (login: admin@faultsense.com / admin123)
- **Jupyter Lab**: http://localhost:8888 (untuk training ML)

---

## 🎯 Quick Start Commands

### Jalankan Semua Services
```bash
# Backend + Database
docker compose up -d

# Frontend (pilih salah satu)
docker compose -f docker-compose.frontend.yml up -d  # Docker
# ATAU
cd frontend && python -m http.server 8080            # Python server

# Training (opsional, untuk ML development)
docker compose -f docker-compose.training.yml up -d
```

### Jalankan Hanya Backend
```bash
docker compose up -d
```

### Jalankan Hanya Frontend
```bash
docker compose -f docker-compose.frontend.yml up -d
```

### Jalankan Hanya Training
```bash
docker compose -f docker-compose.training.yml up -d
```

### Stop Services
```bash
# Stop backend
docker compose down

# Stop frontend
docker compose -f docker-compose.frontend.yml down

# Stop training
docker compose -f docker-compose.training.yml down

# Stop semua sekaligus
docker compose down && docker compose -f docker-compose.frontend.yml down && docker compose -f docker-compose.training.yml down
```

---

### Opsi 2: Menjalankan Lokal (Tanpa Docker)

#### 1. Setup Python Environment
```bash
cd "C:\SEMESTER 4\(Praktikum) Teknologi Web Service\FaultSense"

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### 2. Setup PostgreSQL
```bash
# Install PostgreSQL (jika belum)
# Download dari: https://www.postgresql.org/download/windows/

# Create database
psql -U postgres
CREATE DATABASE faultsense_db;
CREATE USER faultsense_user WITH PASSWORD 'faultsense_pass';
GRANT ALL PRIVILEGES ON DATABASE faultsense_db TO faultsense_user;
\q
```

#### 3. Setup Environment Variables
```bash
# Set DATABASE_URL
set DATABASE_URL=postgresql://faultsense_user:faultsense_pass@localhost:5432/faultsense_db

# Atau buat file .env
echo DATABASE_URL=postgresql://faultsense_user:faultsense_pass@localhost:5432/faultsense_db > .env
```

#### 4. Run Backend
```bash
# Jalankan FastAPI
uvicorn app.main:app --reload

# Server akan berjalan di http://localhost:8000
```

#### 5. Start Frontend
```bash
# Buka terminal baru
cd frontend
python -m http.server 8080
```

---

## 🧪 Testing Koneksi Frontend-Backend

### 1. Test API Health
```bash
curl http://localhost:8000/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-05-07T00:36:13.059Z"
}
```

### 2. Test Prediction Endpoint
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "type": "L",
    "air_temperature": 298.1,
    "process_temperature": 308.6,
    "rotational_speed": 1500,
    "torque": 40.0,
    "tool_wear": 50
  }'
```

**Expected Response:**
```json
{
  "status": "Normal",
  "failure_type": null,
  "probability_failure": 0.0123,
  "recommendation": "Mesin dalam kondisi normal...",
  "checked_at": "2026-05-07T00:36:13.059Z"
}
```

### 3. Test Frontend
1. Buka browser: http://localhost:8080/
2. Klik "Mulai Prediksi"
3. Isi form dengan data sensor
4. Klik "Jalankan Prediksi"
5. Hasil prediksi harus muncul

### 4. Test History
1. Buka: http://localhost:8080/pages/history.html
2. Tabel riwayat prediksi harus muncul
3. Filter harus berfungsi

### 5. Test Analytics
1. Buka: http://localhost:8080/pages/analytics.html
2. Chart dan statistik harus muncul
3. Data harus ter-load dari API

---

## 🐛 Troubleshooting

### Problem: Backend tidak bisa start

**Error: "Port 8000 already in use"**
```bash
# Windows: Kill process di port 8000
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Atau gunakan port lain
uvicorn app.main:app --reload --port 8001
```

**Error: "Database connection failed"**
```bash
# Check PostgreSQL running
docker compose ps

# Check DATABASE_URL
echo %DATABASE_URL%

# Restart PostgreSQL
docker compose restart postgres

# View logs
docker compose logs postgres
```

### Problem: Frontend tidak bisa start

**Error: "Port 3000 already in use"**
```bash
# Stop container yang menggunakan port 3000
docker compose -f docker-compose.frontend.yml down

# Atau gunakan port lain (edit docker-compose.frontend.yml)
```

**Error: "Cannot connect to API"**
- Pastikan backend sudah running di http://localhost:8000
- Check environment variable `NEXT_PUBLIC_API_URL`
- Test API: `curl http://localhost:8000/health`

### Problem: Training tidak bisa start

**Error: "Port 8888 already in use"**
```bash
# Stop Jupyter Lab
docker compose -f docker-compose.training.yml down

# Atau kill process di port 8888
netstat -ano | findstr :8888
taskkill /PID <PID> /F
```

**Error: "Cannot install requirements"**
```bash
# Check logs
docker compose -f docker-compose.training.yml logs

# Rebuild container
docker compose -f docker-compose.training.yml up -d --build
```

### Problem: Frontend tidak bisa akses API

**Error: "CORS error" atau "Failed to fetch"**
- Pastikan backend sudah running di http://localhost:8000
- Check CORS settings di backend (FastAPI)
- Pastikan tidak ada firewall blocking

**Error: "Network error"**
```bash
# Test API dari terminal
curl http://localhost:8000/health

# Jika gagal, backend belum running
```

### Problem: Data tidak muncul di frontend

**Stats menunjukkan "—"**
- Backend belum running
- Database kosong (belum ada prediksi)
- API endpoint error

**Solution:**
1. Check backend logs: `docker compose logs api`
2. Test API manual dengan curl
3. Check browser console untuk error

### Problem: pgAdmin tidak bisa connect ke PostgreSQL

**Error: "Unable to connect to server"**
1. Buka pgAdmin: http://localhost:5050
2. Login: admin@faultsense.com / admin123
3. Add New Server:
   - Name: FaultSense
   - Host: postgres (bukan localhost!)
   - Port: 5432
   - Username: faultsense_user
   - Password: faultsense_pass

---

## 📊 Verifikasi Koneksi

### Checklist:
- [ ] Backend API running (`docker compose ps`)
- [ ] Backend API responding (`curl http://localhost:8000/health`)
- [ ] PostgreSQL running (`docker compose ps`)
- [ ] pgAdmin accessible (http://localhost:5050)
- [ ] Frontend running (Docker atau Python server)
- [ ] Frontend accessible (http://localhost:3000 atau http://localhost:8080)
- [ ] Jupyter Lab running (opsional, http://localhost:8888)
- [ ] Landing page loads
- [ ] Dashboard page loads
- [ ] Prediction form works
- [ ] History page shows data
- [ ] Analytics charts render

---

## 🔄 Workflow Development

### Start Development (Semua Services)
```bash
# Terminal 1: Backend + Database
cd "C:\SEMESTER 4\(Praktikum) Teknologi Web Service\FaultSense"
docker compose up

# Terminal 2: Frontend (pilih salah satu)
# Opsi A: Docker
docker compose -f docker-compose.frontend.yml up

# Opsi B: Python server
cd frontend
python -m http.server 8080

# Terminal 3: Training (opsional)
docker compose -f docker-compose.training.yml up
```

### Start Development (Backend Only)
```bash
# Untuk development API saja
docker compose up
```

### Stop Services
```bash
# Stop backend
docker compose down

# Stop frontend
docker compose -f docker-compose.frontend.yml down

# Stop training
docker compose -f docker-compose.training.yml down
```

### View Logs
```bash
# Backend logs
docker compose logs -f api
docker compose logs -f postgres

# Frontend logs
docker compose -f docker-compose.frontend.yml logs -f

# Training logs
docker compose -f docker-compose.training.yml logs -f
```

### Rebuild Services
```bash
# Rebuild backend
docker compose up -d --build

# Rebuild frontend
docker compose -f docker-compose.frontend.yml up -d --build

# Rebuild training
docker compose -f docker-compose.training.yml up -d --build
```

---

## 📝 Environment Variables

### Required Variables (.env)
```env
# Database
DATABASE_URL=postgresql://faultsense_user:faultsense_pass@postgres:5432/faultsense_db

# API
API_HOST=0.0.0.0
API_PORT=8000

# Optional
DEBUG=True
LOG_LEVEL=INFO
```

---

## 🎯 Next Steps

### 1. Start Backend + Database
```bash
docker compose up -d
```

### 2. Verify API
```bash
curl http://localhost:8000/health
```

### 3. Start Frontend (pilih salah satu)
```bash
# Opsi A: Docker
docker compose -f docker-compose.frontend.yml up -d

# Opsi B: Python server
cd frontend
python -m http.server 8080
```

### 4. Start Training (opsional, untuk ML development)
```bash
docker compose -f docker-compose.training.yml up -d
```

### 5. Test Integration
- Buka http://localhost:3000/ (Docker) atau http://localhost:8080/ (Python)
- Test prediction form
- Check history & analytics
- Access pgAdmin: http://localhost:5050
- Access Jupyter Lab: http://localhost:8888 (jika training running)

---

## 📦 Docker Compose Files

Project ini menggunakan 3 file docker-compose terpisah:

### 1. `docker-compose.yml` (Backend + Database)
Services:
- **postgres**: PostgreSQL database (port 5433)
- **api**: FastAPI backend (port 8000)
- **pgadmin**: Database management UI (port 5050)

### 2. `docker-compose.frontend.yml` (Frontend)
Services:
- **frontend**: Next.js application (port 3000)

### 3. `docker-compose.training.yml` (ML Training)
Services:
- **training**: Jupyter Lab untuk ML development (port 8888)
- Auto-install dependencies dari `requirements-dev.txt`
- Volume mounting: `/notebook`, `/data`, `/models`, `/outputs`

---

## 📞 Support

Jika masih ada masalah:
1. Check logs: 
   - Backend: `docker compose logs -f api`
   - Frontend: `docker compose -f docker-compose.frontend.yml logs -f`
   - Training: `docker compose -f docker-compose.training.yml logs -f`
2. Check API docs: http://localhost:8000/docs
3. Check browser console (F12)
4. Verify ports tidak bentrok
5. Check pgAdmin: http://localhost:5050

---

## 🚀 Use Cases

### Development Backend API
```bash
docker compose up
# API: http://localhost:8000
# pgAdmin: http://localhost:5050
```

### Development Frontend
```bash
docker compose up -d  # Backend di background
docker compose -f docker-compose.frontend.yml up
# Frontend: http://localhost:3000
```

### ML Training & Experimentation
```bash
docker compose -f docker-compose.training.yml up
# Jupyter Lab: http://localhost:8888
# Akses notebook: work/training.ipynb
```

### Full Stack Development
```bash
# Terminal 1: Backend
docker compose up

# Terminal 2: Frontend
docker compose -f docker-compose.frontend.yml up

# Terminal 3: Training (opsional)
docker compose -f docker-compose.training.yml up
```

---

**Setelah semua services running, aplikasi siap digunakan! 🚀**
