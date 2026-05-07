import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

load_dotenv()

# Default mengikuti konfigurasi docker-compose: hostname `postgres` adalah
# service name di dalam network compose. Untuk menjalankan API langsung
# di host (di luar Docker), set DATABASE_URL ke `localhost:5433` di .env.
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://faultsense_user:faultsense_pass@postgres:5432/faultsense_db"
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
