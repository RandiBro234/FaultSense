from app.models_db import PredictionLog


def create_prediction_log(db, input_data, result):
    log = PredictionLog(
        type=input_data.type,
        air_temperature=input_data.air_temperature,
        process_temperature=input_data.process_temperature,
        rotational_speed=input_data.rotational_speed,
        torque=input_data.torque,
        tool_wear=input_data.tool_wear,

        status=result["status"],
        failure_type=result["failure_type"],
        probability_failure=result["probability_failure"],
        recommendation=result["recommendation"]
    )

    db.add(log)
    db.commit()
    db.refresh(log)

    return log


def get_prediction_history(db, limit=20):
    return (
        db.query(PredictionLog)
        .order_by(PredictionLog.created_at.desc())
        .limit(limit)
        .all()
    )