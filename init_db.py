from database import Base, engine
import models


def create_tables():
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully!")

    print("\nAvailable tables:")
    for table_name in Base.metadata.tables:
        print(f"- {table_name}")


if __name__ == "__main__":
    create_tables()