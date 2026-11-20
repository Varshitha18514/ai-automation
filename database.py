
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.orm import sessionmaker, declarative_base

load_dotenv()

# Select database using DATABASE_PROVIDER
DATABASE_PROVIDER = os.getenv("DATABASE_PROVIDER", "sqlite").lower()

connect_args = {}

if DATABASE_PROVIDER == "supabase":
    # Supabase managed PostgreSQL
    connection_url = URL.create(
        drivername="postgresql+psycopg2",
        username=os.getenv("SUPABASE_DB_USER"),
        password=os.getenv("SUPABASE_DB_PASSWORD"),
        host=os.getenv("SUPABASE_DB_HOST"),
        port=int(os.getenv("SUPABASE_DB_PORT", "5432")),
        database=os.getenv("SUPABASE_DB_NAME"),
    )

    DATABASE_URL = connection_url
    connect_args = {"sslmode": "require"}

elif DATABASE_PROVIDER == "docker":
    # Existing Docker PostgreSQL
    DATABASE_URL = os.getenv("DATABASE_URL")

elif DATABASE_PROVIDER == "sqlite":
    # Local SQLite
    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        "sqlite:///./ai_automation.db"
    )
    connect_args = {"check_same_thread": False}

else:
    raise ValueError(
        f"Unsupported DATABASE_PROVIDER: {DATABASE_PROVIDER}"
    )

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True
)

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
