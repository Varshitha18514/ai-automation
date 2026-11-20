
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker


# Database configuration
DATABASE_URL = "sqlite:///./ai_automation.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base = declarative_base()


# Conversation message database model
class ConversationMessage(Base):
    __tablename__ = "conversation_messages"

    id = Column(Integer, primary_key=True, index=True)

    session_id = Column(
        String(100),
        index=True,
        nullable=False,
    )

    role = Column(String(20), nullable=False)

    content = Column(Text, nullable=False)

    created_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
    )


# Create the conversation table
Base.metadata.create_all(bind=engine)


def save_message(session_id: str, role: str, content: str):
    """Save a conversation message."""

    db = SessionLocal()

    try:
        message = ConversationMessage(
            session_id=session_id,
            role=role,
            content=content,
        )

        db.add(message)
        db.commit()

        print(f"Saved {role} message for session {session_id}")

    finally:
        db.close()


def get_recent_messages(session_id: str, limit: int = 10):
    """Retrieve the most recent conversation messages."""

    db = SessionLocal()

    try:
        messages = (
            db.query(ConversationMessage)
            .filter(
                ConversationMessage.session_id == session_id
            )
            .order_by(
                ConversationMessage.id.desc()
            )
            .limit(limit)
            .all()
        )

        messages.reverse()

        return [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in messages
        ]

    finally:
        db.close()


def clear_conversation(session_id: str):
    """Delete messages belonging to one session."""

    db = SessionLocal()

    try:
        db.query(ConversationMessage).filter(
            ConversationMessage.session_id == session_id
        ).delete()

        db.commit()

        print(f"Cleared conversation session: {session_id}")

    finally:
        db.close()


if __name__ == "__main__":

    test_session = "day20_test_session"

    # Save test messages
    save_message(
        test_session,
        "user",
        "What are vector embeddings?",
    )

    save_message(
        test_session,
        "assistant",
        "Vector embeddings are numerical representations of text.",
    )

    save_message(
        test_session,
        "user",
        "Why are they useful?",
    )

    # Retrieve only the latest two messages
    history = get_recent_messages(test_session, limit=2)

    print("\n========== CONVERSATION HISTORY ==========")

    for message in history:
        print(f"\n{message['role']}: {message['content']}")
