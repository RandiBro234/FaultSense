from datetime import datetime
from sqlalchemy import func
from app.models_db import PredictionLog

def create_prediction_log(db, input_data, result):
    log = PredictionLog(
        type                = input_data.type,
        air_temperature     = input_data.air_temperature,
        process_temperature = input_data.process_temperature,
        rotational_speed    = input_data.rotational_speed,
        torque              = input_data.torque,
        tool_wear           = input_data.tool_wear,
        status              = result["status"],
        failure_type        = result.get("failure_type"),
        probability_failure = result["probability_failure"],
        recommendation      = result["recommendation"],
        checked_at          = datetime.fromisoformat(result["checked_at"])
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log


def get_prediction_history(db, status=None, failure_type=None,
                           start_date=None, end_date=None, limit=50):
    query = db.query(PredictionLog)

    # Filter opsional
    if status:
        query = query.filter(PredictionLog.status == status)
    if failure_type:
        query = query.filter(PredictionLog.failure_type == failure_type)
    if start_date:
        query = query.filter(PredictionLog.checked_at >= start_date)
    if end_date:
        query = query.filter(PredictionLog.checked_at <= end_date)

    return (
        query
        .order_by(PredictionLog.checked_at.desc())
        .limit(limit)
        .all()
    )


def get_analytics(db):
    total       = db.query(PredictionLog).count()
    total_normal  = db.query(PredictionLog).filter(PredictionLog.status == "Normal").count()
    total_failure = db.query(PredictionLog).filter(PredictionLog.status == "Failure").count()

    # Hitung per failure type
    failure_types = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    per_type = {}
    for ft in failure_types:
        count = db.query(PredictionLog).filter(PredictionLog.failure_type == ft).count()
        per_type[ft] = count

    # Data terbaru
    latest = (
        db.query(PredictionLog)
        .order_by(PredictionLog.checked_at.desc())
        .first()
    )

    # Rata-rata probability failure
    avg_prob = db.query(func.avg(PredictionLog.probability_failure)).scalar()

    return {
        "total_predictions"  : total,
        "total_normal"       : total_normal,
        "total_failure"      : total_failure,
        "failure_rate"       : round(total_failure / total * 100, 2) if total > 0 else 0.0,
        "per_failure_type"   : per_type,
        "avg_probability"    : round(float(avg_prob), 4) if avg_prob else 0.0,
        "latest_checked_at"  : latest.checked_at.isoformat() if latest else None
    }