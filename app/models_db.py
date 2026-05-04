from sqlalchemy import Column, Integer, Float, String, Text, DateTime
from sqlalchemy.sql import func
from app.database import Base

class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id                  = Column(Integer, primary_key=True, index=True)
    type                = Column(String(5))
    air_temperature     = Column(Float)
    process_temperature = Column(Float)
    rotational_speed    = Column(Integer)
    torque              = Column(Float)
    tool_wear           = Column(Integer)
    status              = Column(String(20))
    failure_type        = Column(String(20), nullable=True)
    probability_failure = Column(Float)
    recommendation      = Column(Text)
    checked_at          = Column(DateTime(timezone=True))           # ← baru
    created_at          = Column(DateTime(timezone=True), server_default=func.now())