# 🚀 Panduan Menjalankan FaultSense (Frontend + Backend)

## 📋 Status Saat Ini

✅ **Frontend** - Sudah selesai di-refactor, siap digunakan  
❌ **Backend API** - Belum running  
❌ **Database** - Belum running  

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

#### 2. Start Services
```bash
# Start semua services (API + PostgreSQL)
docker compose up --build


# Atau run di background
docker compose up -d --build
```

#### 3. Verify Services Running
```bash
# Check containers
docker compose ps

# Check API health
curl http://localhost:8000/health

# Check API docs
# Buka browser: http://localhost:8000/docs
```

#### 4. Start Frontend
```bash
# Buka terminal baru
cd "C:\SEMESTER 4\(Praktikum) Teknologi Web Service\FaultSense\frontend"

# Start HTTP server
python -m http.server 8080

# Atau gunakan Live Server (VS Code extension)
```

#### 5. Akses Aplikasi
- **Frontend**: http://localhost:8080/
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

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

---

## 📊 Verifikasi Koneksi

### Checklist:
- [ ] Docker Compose running (`docker compose ps`)
- [ ] Backend API responding (`curl http://localhost:8000/health`)
- [ ] PostgreSQL running (check docker compose)
- [ ] Frontend server running (http://localhost:8080)
- [ ] Landing page loads
- [ ] Dashboard page loads
- [ ] Prediction form works
- [ ] History page shows data
- [ ] Analytics charts render

---

## 🔄 Workflow Development

### Start Development
```bash
# Terminal 1: Backend
cd "C:\SEMESTER 4\(Praktikum) Teknologi Web Service\FaultSense"
docker compose up

# Terminal 2: Frontend
cd "C:\SEMESTER 4\(Praktikum) Teknologi Web Service\FaultSense\frontend"
python -m http.server 8080
```

### Stop Services
```bash
# Stop Docker Compose
docker compose down

# Stop frontend (Ctrl+C di terminal)
```

### View Logs
```bash
# All services
docker compose logs

# Specific service
docker compose logs api
docker compose logs postgres

# Follow logs
docker compose logs -f api
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

1. **Start Backend**
   ```bash
   docker compose up -d
   ```

2. **Verify API**
   ```bash
   curl http://localhost:8000/health
   ```

3. **Start Frontend**
   ```bash
   cd frontend
   python -m http.server 8080
   ```

4. **Test Integration**
   - Buka http://localhost:8080/
   - Test prediction form
   - Check history & analytics

---

## 📞 Support

Jika masih ada masalah:
1. Check logs: `docker compose logs`
2. Check API docs: http://localhost:8000/docs
3. Check browser console (F12)
4. Verify ports tidak bentrok

---

**Setelah backend running, frontend akan otomatis terhubung ke API! 🚀**
