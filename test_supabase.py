import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

load_dotenv(override=True)

connection_url = URL.create(
    drivername="postgresql+psycopg2",
    username=os.getenv("SUPABASE_DB_USER"),
    password=os.getenv("SUPABASE_DB_PASSWORD"),
    host=os.getenv("SUPABASE_DB_HOST"),
    port=int(os.getenv("SUPABASE_DB_PORT", "5432")),
    database=os.getenv("SUPABASE_DB_NAME"),
)

engine = create_engine(
    connection_url,
    connect_args={"sslmode": "require"},
    pool_pre_ping=True,
)

try:
    with engine.connect() as connection:
        result = connection.execute(text("SELECT 1"))
        print("Supabase connection successful!")
        print("Test result:", result.scalar())
except Exception as error:
    print("Supabase connection failed.")
    print("Error:", str(error).splitlines()[0])
finally:
    engine.dispose()