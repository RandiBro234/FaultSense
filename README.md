# FaultSense

Predictive maintenance API untuk mendeteksi kegagalan mesin manufaktur menggunakan klasifikasi cascade (binary failure detection → multiclass failure type) di atas dataset [AI4I 2020](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset).

Stack: FastAPI · scikit-learn · PostgreSQL · Docker Compose.

---

## Arsitektur singkat

1. **Binary classifier** memprediksi probabilitas mesin akan gagal. Threshold dipilih lewat tuning di validation split (bukan test set).
2. Jika probabilitas ≥ threshold, **multiclass classifier** menentukan jenis kegagalan: TWF, HDF, PWF, OSF, atau RNF — sebatas kelas yang dapat dipelajari model dari data training.
3. Setiap permintaan prediksi dicatat ke PostgreSQL untuk endpoint `/history` dan `/analytics`.

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

## Quick start (Docker Compose)

Prasyarat: Docker & Docker Compose v2.

```bash
git clone https://github.com/RandiBro234/FaultSense.git
cd FaultSense
cp .env.example .env       # ubah kredensial sebelum production
docker compose up --build
```

API tersedia di `http://localhost:8000`. Dokumentasi interaktif Swagger di `http://localhost:8000/docs`.

Postgres di-expose ke host di port **5433** (di dalam network compose tetap `5432`).

---

## Quick start (lokal, tanpa Docker)

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Postgres harus jalan secara terpisah
export DATABASE_URL=postgresql://faultsense_user:faultsense_pass@localhost:5432/faultsense_db

uvicorn app.main:app --reload
```

---

## Konfigurasi

Semua konfigurasi runtime via environment variable. Lihat <code>.env.example</code> untuk daftar lengkap.

| Var | Default | Keterangan |
| --- | --- | --- |
| `DATABASE_URL` | `postgresql://faultsense_user:faultsense_pass@postgres:5432/faultsense_db` | URL koneksi SQLAlchemy. Hostname `postgres` adalah service name di Docker Compose; gunakan `localhost:5433` jika menjalankan API di host. |

---

## Endpoint utama

| Method | Path | Deskripsi |
| --- | --- | --- |
| GET | `/` | Info ringkas API. |
| GET | `/health` | Health check. |
| POST | `/predict` | Prediksi kondisi mesin dari data sensor. |
| GET | `/history` | Riwayat prediksi (filter `status`, `failure_type`, `start_date`, `end_date`, `limit`). |
| GET | `/analytics` | Ringkasan agregat: total, distribusi per kelas, rata-rata probabilitas. |

### Contoh `POST /predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "type": "L",
        "air_temperature": 298.0,
        "process_temperature": 308.0,
        "rotational_speed": 1500,
        "torque": 40.0,
        "tool_wear": 50
      }'
```

Response:

```json
{
  "status": "Normal",
  "failure_type": null,
  "probability_failure": 0.0123,
  "recommendation": "Mesin dalam kondisi normal. Lanjutkan monitoring rutin.",
  "checked_at": "2025-01-01T12:34:56.000000"
}
```

---

## Training ulang model

Dependensi training dipisah agar image runtime tetap ramping.

```bash
pip install -r requirements-dev.txt
jupyter lab notebook/training.ipynb
```

Notebook menghasilkan artefak ke folder `models/`:

- `model_binary_rf.pkl` — RandomForest binary (Normal vs Failure).
- `model_multiclass_rf.pkl` — RandomForest multiclass (jenis failure).
- `scaler.pkl` — `StandardScaler` untuk fitur numerik.
- `ohe_columns.pkl` — daftar kolom hasil one-hot encoding `Type`.
- `label_map.pkl` — pemetaan label numerik multiclass ke nama failure type.
- `threshold.pkl` — threshold yang dipilih dari validation split.

Tracking eksperimen menggunakan MLflow (SQLite store di `mlflow.db`). Untuk membuka UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5002
```

---

## Catatan dataset (AI4I 2020)

Dataset memiliki kuirk yang relevan untuk training cascade:

- 339 baris dengan `Machine failure = 1` dari total 10000.
- 9 baris `Machine failure = 1` namun tidak memiliki failure type spesifik (TWF/HDF/PWF/OSF/RNF) — di-drop secara eksplisit oleh notebook dengan log peringatan.
- 24 baris memiliki lebih dari satu failure type aktif — di-resolve dengan urutan prioritas TWF → HDF → PWF → OSF → RNF.
- 18 dari 19 baris dengan `RNF = 1` justru memiliki `Machine failure = 0`. Akibatnya kelas RNF nyaris tidak pernah masuk ke pipeline multiclass dan model umumnya tidak mempelajarinya. API hanya akan mengembalikan kelas yang benar-benar dipelajari (lihat `model.classes_`).

---

## Pengembangan

- Test: `pytest`
- Lint (jika ditambahkan): `ruff check .`
- Pre-commit (opsional): `pre-commit install`

---

## Lisensi

Proyek tugas akhir / penelitian. Belum ada lisensi formal.
