import os
import pickle
import numpy as np

# ============================================================
# PATH MODEL
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
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

with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
    label_encoder = pickle.load(f)

with open(os.path.join(MODEL_DIR, "threshold.pkl"), "rb") as f:
    threshold = pickle.load(f)


# ============================================================
# LABEL DAN REKOMENDASI
# ============================================================

failure_labels = ["TWF", "HDF", "PWF", "OSF", "RNF"]

recommendations = {
    "Normal": "Mesin dalam kondisi normal. Lanjutkan monitoring rutin.",
    "TWF": "Periksa kondisi tool, lakukan penggantian atau perawatan alat potong, dan pantau tingkat keausan alat.",
    "HDF": "Periksa sistem pendingin, suhu mesin, ventilasi, dan kurangi beban kerja sementara.",
    "PWF": "Periksa suplai daya, motor, kestabilan torsi, dan kecepatan putar mesin.",
    "OSF": "Periksa beban kerja mesin, kurangi overload, dan cek nilai torque serta tool wear.",
    "RNF": "Lakukan inspeksi manual karena penyebab kegagalan bersifat acak atau tidak terklasifikasi jelas."
}


# ============================================================
# FUNGSI PREDIKSI CASCADE
# ============================================================

def predict(data: dict):
    type_value = data["type"].upper()

    if type_value not in label_encoder.classes_:
        raise ValueError(f"Type '{type_value}' tidak dikenali. Gunakan salah satu: {list(label_encoder.classes_)}")

    type_encoded = label_encoder.transform([type_value])[0]

    input_array = np.array([[
        type_encoded,
        data["air_temperature"],
        data["process_temperature"],
        data["rotational_speed"],
        data["torque"],
        data["tool_wear"]
    ]])

    input_scaled = scaler.transform(input_array)

    probability_failure = model_binary.predict_proba(input_scaled)[0][1]

    if probability_failure < threshold:
        return {
            "status": "Normal",
            "failure_type": None,
            "probability_failure": round(float(probability_failure), 4),
            "recommendation": recommendations["Normal"]
        }

    multiclass_prediction = model_multiclass.predict(input_scaled)[0]
    failure_type = failure_labels[int(multiclass_prediction)]

    return {
        "status": "Failure",
        "failure_type": failure_type,
        "probability_failure": round(float(probability_failure), 4),
        "recommendation": recommendations[failure_type]
    }