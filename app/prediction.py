from __future__ import annotations

import os
import pickle
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(BASE_DIR)
MODEL_DIR = os.path.join(ROOT_DIR, "models")


def _load_pickle(filename: str) -> Any:
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File model tidak ditemukan: {path}")
    with open(path, "rb") as fh:
        return pickle.load(fh)


model_binary     = _load_pickle("model_binary_rf.pkl")
model_multiclass = _load_pickle("model_multiclass_rf.pkl")
scaler           = _load_pickle("scaler.pkl")
threshold        = float(_load_pickle("threshold.pkl"))

FULL_FEATURE_COLUMNS: List[str] = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Type_H",
    "Type_L",
    "Type_M",
    "power",
    "temp_diff",
    "wear_torque",
]

SELECTED_FEATURE_COLUMNS: List[str] = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Type_M",
    "power",
    "temp_diff",
    "wear_torque",
]


label_map: Dict[int, str] = {
    0: "TWF",
    1: "HDF",
    2: "PWF",
    3: "OSF",
    4: "RNF",
}

recommendations: Dict[str, str] = {
    "Normal": "Mesin dalam kondisi normal. Lanjutkan monitoring rutin.",
    "TWF": "Periksa kondisi tool, lakukan penggantian atau perawatan alat potong, dan pantau tingkat keausan alat.",
    "HDF": "Periksa sistem pendingin, suhu mesin, ventilasi, dan kurangi beban kerja sementara.",
    "PWF": "Periksa suplai daya, motor, kestabilan torsi, dan kecepatan putar mesin.",
    "OSF": "Periksa beban kerja mesin, kurangi overload, dan cek nilai torque serta tool wear.",
    "RNF": "Lakukan inspeksi manual karena penyebab kegagalan bersifat acak atau tidak terklasifikasi jelas.",
}


def _build_input_frame(data: Dict[str, Any]) -> pd.DataFrame:
    type_value   = str(data["type"]).upper()
    air_temp     = float(data["air_temperature"])
    process_temp = float(data["process_temperature"])
    rot_speed    = float(data["rotational_speed"])
    torque       = float(data["torque"])
    tool_wear    = float(data["tool_wear"])

    row = {
        "Air temperature [K]"     : air_temp,
        "Process temperature [K]" : process_temp,
        "Rotational speed [rpm]"  : rot_speed,
        "Torque [Nm]"             : torque,
        "Tool wear [min]"         : tool_wear,

        "Type_H"                  : 1 if type_value == "H" else 0,
        "Type_L"                  : 1 if type_value == "L" else 0,
        "Type_M"                  : 1 if type_value == "M" else 0,

        "power"                   : torque * rot_speed,
        "temp_diff"               : process_temp - air_temp,
        "wear_torque"             : tool_wear * torque,
    }

    return pd.DataFrame([row], columns=FULL_FEATURE_COLUMNS)


def _resolve_failure_label(prediction: int) -> str:
    return label_map.get(int(prediction), f"UNKNOWN_{int(prediction)}")


def _validate_model_features() -> None:
    if hasattr(scaler, "feature_names_in_"):
        scaler_features = list(scaler.feature_names_in_)
        if scaler_features != FULL_FEATURE_COLUMNS:
            raise ValueError(
                "Urutan fitur scaler tidak sama dengan backend.\n"
                f"Scaler : {scaler_features}\n"
                f"Backend: {FULL_FEATURE_COLUMNS}"
            )

    if hasattr(model_binary, "feature_names_in_"):
        binary_features = list(model_binary.feature_names_in_)
        if binary_features != SELECTED_FEATURE_COLUMNS:
            raise ValueError(
                "Urutan fitur model binary tidak sama dengan backend.\n"
                f"Model  : {binary_features}\n"
                f"Backend: {SELECTED_FEATURE_COLUMNS}"
            )

    if hasattr(model_multiclass, "feature_names_in_"):
        multiclass_features = list(model_multiclass.feature_names_in_)
        if multiclass_features != SELECTED_FEATURE_COLUMNS:
            raise ValueError(
                "Urutan fitur model multiclass tidak sama dengan backend.\n"
                f"Model  : {multiclass_features}\n"
                f"Backend: {SELECTED_FEATURE_COLUMNS}"
            )


_validate_model_features()


def predict(data: Dict[str, Any]) -> Dict[str, Any]:
    now = datetime.now()

    input_frame = _build_input_frame(data)

    input_scaled_full = pd.DataFrame(
        scaler.transform(input_frame),
        columns=FULL_FEATURE_COLUMNS,
        index=input_frame.index,
    )

    input_scaled_selected = input_scaled_full[SELECTED_FEATURE_COLUMNS]

    probability_failure = float(
        model_binary.predict_proba(input_scaled_selected)[0][1]
    )

    if probability_failure < threshold:
        return {
            "status"              : "Normal",
            "failure_type"        : None,
            "probability_failure" : round(probability_failure, 4),
            "recommendation"      : recommendations["Normal"],
            "checked_at"          : now.isoformat(),
        }

    multiclass_prediction = int(
        model_multiclass.predict(input_scaled_selected)[0]
    )
    failure_type = _resolve_failure_label(multiclass_prediction)

    return {
        "status"              : "Failure",
        "failure_type"        : failure_type,
        "probability_failure" : round(probability_failure, 4),
        "recommendation"      : recommendations.get(
            failure_type,
            "Kondisi failure terdetksi; lakukan inspeksi manual."
        ),
        "checked_at"          : now.isoformat(),
    }