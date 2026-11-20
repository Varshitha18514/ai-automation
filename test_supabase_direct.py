import os
import getpass
import psycopg2
from dotenv import load_dotenv

load_dotenv(override=True)

password = getpass.getpass("Enter your current Supabase database password: ")

try:
    connection = psycopg2.connect(
        host=os.getenv("SUPABASE_DB_HOST"),
        port=os.getenv("SUPABASE_DB_PORT", "5432"),
        database=os.getenv("SUPABASE_DB_NAME"),
        user=os.getenv("SUPABASE_DB_USER"),
        password=password,
        sslmode="require",
        connect_timeout=15
    )

    print("SUCCESS: Supabase PostgreSQL connected!")

    cursor = connection.cursor()
    cursor.execute("SELECT version();")
    print("Database:", cursor.fetchone()[0])

    cursor.close()
    connection.close()

except Exception as error:
    print("Connection failed.")
    print("Error:", str(error).splitlines()[0])
