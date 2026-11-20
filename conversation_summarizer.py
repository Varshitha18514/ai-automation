
import os

from dotenv import load_dotenv
from google import genai
from sqlalchemy import Column, Integer, String, Text
from sqlalchemy.orm import Session

from conversation_memory import (
    Base,
    engine,
    SessionLocal,
    ConversationMessage,
)


# Load Gemini API configuration
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing from your .env file")

gemini = genai.Client(api_key=api_key)


# Conversation summary database model
class ConversationSummary(Base):
    __tablename__ = "conversation_summaries"

    id = Column(Integer, primary_key=True, index=True)

    session_id = Column(
        String(100),
        unique=True,
        index=True,
        nullable=False,
    )

    summary = Column(Text, nullable=False)


# Create the summary table
Base.metadata.create_all(bind=engine)


def summarize_conversation(session_id: str):
    """Summarize a conversation using Gemini and save it."""

    db: Session = SessionLocal()

    try:
        # Retrieve conversation messages
        messages = (
            db.query(ConversationMessage)
            .filter(
                ConversationMessage.session_id == session_id
            )
            .order_by(ConversationMessage.id.asc())
            .all()
        )

        if not messages:
            print("No conversation messages found.")
            return None

        # Convert messages into readable text
        conversation_text = "\n".join(
            f"{message.role}: {message.content}"
            for message in messages
        )

        # Generate summary using Gemini
        prompt = f"""
You are an assistant that summarizes conversations.

Summarize the following conversation.
Keep important facts, questions, answers, and context.
Use clear and concise language.

CONVERSATION:
{conversation_text}

CONVERSATION SUMMARY:
"""

        print("Generating conversation summary with Gemini...")

        response = gemini.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt,
        )

        summary_text = response.text

        if not summary_text:
            print("Gemini did not return a summary.")
            return None

        # Check whether a summary already exists
        existing_summary = (
            db.query(ConversationSummary)
            .filter(
                ConversationSummary.session_id == session_id
            )
            .first()
        )

        if existing_summary:
            existing_summary.summary = summary_text
        else:
            new_summary = ConversationSummary(
                session_id=session_id,
                summary=summary_text,
            )

            db.add(new_summary)

        db.commit()

        print("\n========== CONVERSATION SUMMARY ==========")
        print(summary_text)

        print("\nSummary successfully saved to SQLite.")

        return summary_text

    finally:
        db.close()


def get_conversation_summary(session_id: str):
    """Retrieve a saved conversation summary."""

    db = SessionLocal()

    try:
        saved_summary = (
            db.query(ConversationSummary)
            .filter(
                ConversationSummary.session_id == session_id
            )
            .first()
        )

        if saved_summary:
            return saved_summary.summary

        return None

    finally:
        db.close()


if __name__ == "__main__":

    test_session = "day20_test_session"

    summarize_conversation(test_session)

    print("\n========== RETRIEVED SAVED SUMMARY ==========")

    saved_summary = get_conversation_summary(test_session)

    if saved_summary:
        print(saved_summary)
    else:
        print("No saved summary found.")

