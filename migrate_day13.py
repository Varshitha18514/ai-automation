
from sqlalchemy import inspect, text

from database import engine


def migrate_database():
    with engine.begin() as connection:

        inspector = inspect(connection)

        if "jobs" not in inspector.get_table_names():
            raise RuntimeError(
                "The jobs table does not exist. "
                "Check your database first."
            )

        existing_columns = {
            column["name"]
            for column in inspector.get_columns("jobs")
        }

        if "webhook_payload" not in existing_columns:
            connection.execute(
                text(
                    "ALTER TABLE jobs "
                    "ADD COLUMN webhook_payload TEXT"
                )
            )
            print("Added webhook_payload column.")

        else:
            print("webhook_payload already exists.")

        if "ai_result" not in existing_columns:
            connection.execute(
                text(
                    "ALTER TABLE jobs "
                    "ADD COLUMN ai_result TEXT"
                )
            )
            print("Added ai_result column.")

        else:
            print("ai_result already exists.")

    print("\nDay 13 database migration completed!")


if __name__ == "__main__":
    migrate_database()

