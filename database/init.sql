CREATE TABLE IF NOT EXISTS prediction_logs (
    id                  SERIAL PRIMARY KEY,
    type                VARCHAR(5),
    air_temperature     FLOAT,
    process_temperature FLOAT,
    rotational_speed    INTEGER,
    torque              FLOAT,
    tool_wear           INTEGER,
    status              VARCHAR(20),
    failure_type        VARCHAR(20),
    probability_failure FLOAT,
    recommendation      TEXT,
    checked_at          TIMESTAMPTZ,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);