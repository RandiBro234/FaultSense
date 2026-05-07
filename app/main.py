from datetime import datetime
from typing import Optional

from fastapi import Depends, FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.crud import create_prediction_log, get_analytics, get_prediction_history
from app.database import Base, engine, get_db
from app.prediction import predict
from app.schemas import PredictionInput

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="FaultSense API",
    description="Predictive Maintenance System — Cascade Classification",
    version="1.0.0",
)

# CORS — izinkan frontend akses API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ROOT
# ============================================================
@app.get("/")
def root():
    return {
        "message"  : "FaultSense API is running",
        "version"  : "1.0.0",
        "docs"     : "/docs",
        "endpoints": {
            "health"   : "GET /health",
            "predict"  : "POST /predict",
            "history"  : "GET /history",
            "analytics": "GET /analytics",
        },
    }


# ============================================================
# HEALTH CHECK
# ============================================================
@app.get("/health")
def health():
    return {
        "status"   : "API is running",
        "timestamp": datetime.now().isoformat(),
    }


# ============================================================
# PREDICT
# ============================================================
@app.post("/predict")
def predict_api(data: PredictionInput, db: Session = Depends(get_db)):
    result = predict(data.model_dump())
    create_prediction_log(db=db, input_data=data, result=result)
    return result


# ============================================================
# HISTORY — dengan filter opsional
# ============================================================
@app.get("/history")
def history(
    db          : Session            = Depends(get_db),
    status      : Optional[str]      = Query(None, description="Filter: Normal atau Failure"),
    failure_type: Optional[str]      = Query(None, description="Filter: TWF, HDF, PWF, OSF, RNF"),
    start_date  : Optional[datetime] = Query(None, description="Filter dari tanggal (ISO format)"),
    end_date    : Optional[datetime] = Query(None, description="Filter sampai tanggal (ISO format)"),
    limit       : int                = Query(50,   ge=1, le=500, description="Jumlah data maksimal"),
):
    return get_prediction_history(
        db           = db,
        status       = status,
        failure_type = failure_type,
        start_date   = start_date,
        end_date     = end_date,
        limit        = limit,
    )


# ============================================================
# ANALYTICS
# ============================================================
@app.get("/analytics")
def analytics(db: Session = Depends(get_db)):
    return get_analytics(db)
