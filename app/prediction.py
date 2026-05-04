"""Cascade prediction: binary failure detection → multiclass failure type.

Model artefak di-load sekali saat modul di-import dan dipakai bersama oleh
seluruh worker FastAPI. Validasi input dilakukan di `app.schemas`, bukan di
sini, sehingga modul ini bisa fokus pada inference saja.
"""

from __future__ import annotations

import os
import pickle
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd

# ============================================================
# PATH MODEL
# ============================================================
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(BASE_DIR)
MODEL_DIR = os.path.join(ROOT_DIR, "models")


def _load_pickle(filename: str) -> Any:
    with open(os.path.join(MODEL_DIR, filename), "rb") as fh:
        return pickle.load(fh)


# ============================================================
# LOAD MODEL DAN PREPROCESSOR
# ============================================================
model_binary     = _load_pickle("model_binary_rf.pkl")
model_multiclass = _load_pickle("model_multiclass_rf.pkl")
scaler           = _load_pickle("scaler.pkl")
ohe_columns: List[str] = _load_pickle("ohe_columns.pkl")
threshold: float = float(_load_pickle("threshold.pkl"))

# Default label map (fallback bila artefak label_map.pkl belum tersedia,
# misal model lama yang dilatih sebelum refactor notebook).
_DEFAULT_LABEL_MAP: Dict[int, str] = {0: "TWF", 1: "HDF", 2: "PWF", 3: "OSF", 4: "RNF"}

try:
    label_map: Dict[int, str] = _load_pickle("label_map.pkl")
except FileNotFoundError:
    label_map = _DEFAULT_LABEL_MAP

# ============================================================
# REKOMENDASI
# ============================================================
recommendations: Dict[str, str] = {
    "Normal": "Mesin dalam kondisi normal. Lanjutkan monitoring rutin.",
    "TWF"   : "Periksa kondisi tool, lakukan penggantian atau perawatan alat potong, dan pantau tingkat keausan alat.",
    "HDF"   : "Periksa sistem pendingin, suhu mesin, ventilasi, dan kurangi beban kerja sementara.",
    "PWF"   : "Periksa suplai daya, motor, kestabilan torsi, dan kecepatan putar mesin.",
    "OSF"   : "Periksa beban kerja mesin, kurangi overload, dan cek nilai torque serta tool wear.",
    "RNF"   : "Lakukan inspeksi manual karena penyebab kegagalan bersifat acak atau tidak terklasifikasi jelas.",
}

# Urutan fitur HARUS sama dengan training (lihat notebook/training.ipynb):
# fitur numerik dulu, lalu hasil one-hot encoding kolom Type.
_NUMERIC_FEATURES: List[str] = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
FEATURE_COLUMNS: List[str] = _NUMERIC_FEATURES + ohe_columns


def _build_input_frame(data: Dict[str, Any]) -> pd.DataFrame:
    """Susun DataFrame satu baris dengan urutan kolom sesuai training.

    Pakai DataFrame dengan kolom bernama agar tidak memicu warning
    `X does not have valid feature names` dari scikit-learn.
    """
    type_value = str(data["type"]).upper()
    row = {
        "Air temperature [K]"     : data["air_temperature"],
        "Process temperature [K]" : data["process_temperature"],
        "Rotational speed [rpm]"  : data["rotational_speed"],
        "Torque [Nm]"             : data["torque"],
        "Tool wear [min]"         : data["tool_wear"],
    }
    for col in ohe_columns:
        # Kolom OHE biasanya berbentuk "Type_H", "Type_L", "Type_M".
        suffix = col.split("_", 1)[1] if "_" in col else col
        row[col] = 1 if suffix == type_value else 0
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def _resolve_failure_label(prediction: int) -> str:
    """Map output multiclass model ke nama failure type."""
    if prediction in label_map:
        return label_map[int(prediction)]
    return f"UNKNOWN_{int(prediction)}"


# ============================================================
# FUNGSI PREDIKSI CASCADE
# ============================================================
def predict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Jalankan cascade: binary → multiclass jika probabilitas ≥ threshold."""
    now = datetime.now()

    input_frame  = _build_input_frame(data)
    input_scaled = pd.DataFrame(
        scaler.transform(input_frame),
        columns=FEATURE_COLUMNS,
        index=input_frame.index,
    )

    probability_failure = float(model_binary.predict_proba(input_scaled)[0][1])

    if probability_failure < threshold:
        return {
            "status"              : "Normal",
            "failure_type"        : None,
            "probability_failure" : round(probability_failure, 4),
            "recommendation"      : recommendations["Normal"],
            "checked_at"          : now.isoformat(),
        }

    multiclass_prediction = int(model_multiclass.predict(input_scaled)[0])
    failure_type          = _resolve_failure_label(multiclass_prediction)

    return {
        "status"              : "Failure",
        "failure_type"        : failure_type,
        "probability_failure" : round(probability_failure, 4),
        "recommendation"      : recommendations.get(
            failure_type, "Kondisi failure terdeteksi; lakukan inspeksi manual."
        ),
        "checked_at"          : now.isoformat(),
    }
