from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class PredictionInput(BaseModel):
    type                : str
    air_temperature     : float
    process_temperature : float
    rotational_speed    : int
    torque              : float
    tool_wear           : int

class PredictionResponse(BaseModel):
    status              : str
    failure_type        : Optional[str] = None
    probability_failure : float
    recommendation      : str
    checked_at          : datetime

class AnalyticsResponse(BaseModel):
    total_predictions   : int
    total_normal        : int
    total_failure       : int
    failure_rate        : float
    per_failure_type    : dict
    avg_probability     : float
    latest_checked_at   : Optional[str] = None