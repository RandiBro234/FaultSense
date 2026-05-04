import os
import pickle
import numpy as np
from datetime import datetime

# ============================================================
# PATH MODEL
# ============================================================
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(BASE_DIR)
MODEL_DIR = os.path.join(ROOT_DIR, "models")

# ============================================================
# LOAD MODEL DAN PREPROCESSOR
# ============================================================
with open(os.path.join(MODEL_DIR, "model_binary_rf.pkl"), "rb") as f:
    model_binary = pickle.load(f)

with open(os.path.join(MODEL_DIR, "model_multiclass_rf.pkl"), "rb") as f:
    model_multiclass = pickle.load(f)

with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(MODEL_DIR, "ohe_columns.pkl"), "rb") as f:
    ohe_columns = pickle.load(f)

with open(os.path.join(MODEL_DIR, "threshold.pkl"), "rb") as f:
    threshold = pickle.load(f)

# ============================================================
# LABEL DAN REKOMENDASI
# ============================================================
failure_labels = ["TWF", "HDF", "PWF", "OSF", "RNF"]

recommendations = {
    "Normal": "Mesin dalam kondisi normal. Lanjutkan monitoring rutin.",
    "TWF"   : "Periksa kondisi tool, lakukan penggantian atau perawatan alat potong, dan pantau tingkat keausan alat.",
    "HDF"   : "Periksa sistem pendingin, suhu mesin, ventilasi, dan kurangi beban kerja sementara.",
    "PWF"   : "Periksa suplai daya, motor, kestabilan torsi, dan kecepatan putar mesin.",
    "OSF"   : "Periksa beban kerja mesin, kurangi overload, dan cek nilai torque serta tool wear.",
    "RNF"   : "Lakukan inspeksi manual karena penyebab kegagalan bersifat acak atau tidak terklasifikasi jelas."
}

VALID_TYPES = ['H', 'L', 'M']

# ============================================================
# FUNGSI PREDIKSI CASCADE
# ============================================================
def predict(data: dict):
    type_value = data["type"].upper()

    if type_value not in VALID_TYPES:
        raise ValueError(f"Type '{type_value}' tidak valid. Gunakan salah satu: {VALID_TYPES}")

    # Ambil timestamp otomatis saat request masuk
    now = datetime.now()

    # OHE manual — urutan HARUS sama dengan feature_cols di notebook
    input_array = np.array([[
        data["air_temperature"],
        data["process_temperature"],
        data["rotational_speed"],
        data["torque"],
        data["tool_wear"],
        1 if type_value == 'H' else 0,   # Type_H
        1 if type_value == 'L' else 0,   # Type_L
        1 if type_value == 'M' else 0,   # Type_M
    ]])

    input_scaled        = scaler.transform(input_array)
    probability_failure = model_binary.predict_proba(input_scaled)[0][1]

    if probability_failure < threshold:
        return {
            "status"              : "Normal",
            "failure_type"        : None,
            "probability_failure" : round(float(probability_failure), 4),
            "recommendation"      : recommendations["Normal"],
            "checked_at"          : now.isoformat()
        }

    multiclass_prediction = model_multiclass.predict(input_scaled)[0]
    failure_type          = failure_labels[int(multiclass_prediction)]

    return {
        "status"              : "Failure",
        "failure_type"        : failure_type,
        "probability_failure" : round(float(probability_failure), 4),
        "recommendation"      : recommendations[failure_type],
        "checked_at"          : now.isoformat()
    }