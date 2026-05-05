from datetime import datetime
from typing import Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Type produk: H = High quality, L = Low, M = Medium (sesuai dataset AI4I 2020).
ProductType = Literal["H", "L", "M"]


class PredictionInput(BaseModel):
    """Schema input untuk endpoint /predict.

    Semua field divalidasi di sini agar request yang tidak valid mendapat
    HTTP 422 (validation error) yang informatif, bukan 500.
    """

    model_config = ConfigDict(populate_by_name=True)

    type                : ProductType = Field(..., description="Kategori produk: H, L, atau M.")
    air_temperature     : float       = Field(..., gt=0, description="Air temperature dalam Kelvin.")
    process_temperature : float       = Field(..., gt=0, description="Process temperature dalam Kelvin.")
    rotational_speed    : int         = Field(..., gt=0, description="Rotational speed dalam rpm.")
    torque              : float       = Field(..., ge=0, description="Torque dalam Nm.")
    tool_wear           : int         = Field(..., ge=0, description="Tool wear dalam menit.")

    @field_validator("type", mode="before")
    @classmethod
    def _normalize_type(cls, value: object) -> object:
        # Terima input lower-case dengan menormalisasi ke upper-case.
        if isinstance(value, str):
            return value.upper()
        return value


class PredictionResponse(BaseModel):
    status              : Literal["Normal", "Failure"]
    failure_type        : Optional[str]   = None
    probability_failure : float
    recommendation      : str
    checked_at          : datetime


class AnalyticsResponse(BaseModel):
    total_predictions   : int
    total_normal        : int
    total_failure       : int
    failure_rate        : float
    per_failure_type    : Dict[str, int]
    avg_probability     : float
    latest_checked_at   : Optional[str]   = None
