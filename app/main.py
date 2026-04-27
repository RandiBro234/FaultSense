from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session

from app.schemas import PredictionInput
from app.prediction import predict
from app.database import Base, engine, get_db
from app.crud import create_prediction_log, get_prediction_history

Base.metadata.create_all(bind=engine)

app = FastAPI(title="FaultSense API")


@app.get("/")
def root():
    return {
        "message": "FaultSense API is running",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
        "history": "/history"
    }


@app.get("/health")
def health():
    return {"status": "API is running"}


@app.post("/predict")
def predict_api(data: PredictionInput, db: Session = Depends(get_db)):
    result = predict(data.dict())

    create_prediction_log(
        db=db,
        input_data=data,
        result=result
    )

    return result


@app.get("/history")
def history(db: Session = Depends(get_db)):
    return get_prediction_history(db)