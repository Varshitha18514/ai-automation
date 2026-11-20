
import os

from dotenv import load_dotenv
from google import genai

from hybrid_search import hybrid_search
from reranker import rerank_results

from conversation_memory import (
    save_message,
    get_recent_messages,
)

from conversation_summarizer import (
    get_conversation_summary,
)


# Load environment variables
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing from your .env file")

gemini = genai.Client(api_key=api_key)


def generate_answer(question: str, session_id: str):
    """Generate an answer using RAG and conversational memory."""

    print("\nStep 1: Loading conversation memory...")

    # Retrieve previous conversation messages
    history = get_recent_messages(
        session_id,
        limit=10,
    )

    # Retrieve the saved conversation summary
    summary = get_conversation_summary(session_id)

    # Save the current user question
    save_message(
        session_id,
        "user",
        question,
    )

    # Format conversation history
    history_text = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in history
    )

    print("Conversation memory loaded.")

    print("\nStep 2: Running hybrid search...")

    search_results = hybrid_search(
        question,
        limit=5,
    )

    if not search_results:
        answer = "I couldn't find relevant information in the documents."

        save_message(session_id, "assistant", answer)

        return answer

    print("\nStep 3: Re-ranking search results...")

    reranked_results = rerank_results(
        question,
        search_results,
        limit=3,
    )

    if not reranked_results:
        answer = "I couldn't find relevant information in the documents."

        save_message(session_id, "assistant", answer)

        return answer

    print("\nStep 4: Preparing document context...")

    context_text = "\n\n".join(
        item["content"]
        for item in reranked_results
    )

    # Prepare memory context
    memory_context = ""

    if summary:
        memory_context += f"""
PREVIOUS CONVERSATION SUMMARY:
{summary}
"""

    if history_text:
        memory_context += f"""
RECENT CONVERSATION HISTORY:
{history_text}
"""

    # Build final prompt
    prompt = f"""
You are a helpful AI assistant with conversational memory.

Use the provided document context and conversation memory
to answer the user's latest question.

Instructions:
- Use document context as the primary source for factual answers.
- Use conversation history to understand follow-up questions.
- Use the previous summary to remember older conversation details.
- Do not invent information that is not available.
- If the documents do not contain enough information,
  clearly explain that.

CONVERSATION MEMORY:
{memory_context}

DOCUMENT CONTEXT:
{context_text}

CURRENT USER QUESTION:
{question}

Provide a clear and easy-to-understand answer.
"""

    print("\nStep 5: Generating answer with Gemini...")

    response = gemini.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
    )

    answer = response.text

    # Save assistant response to conversation memory
    save_message(
        session_id,
        "assistant",
        answer,
    )

    return answer


if __name__ == "__main__":

    print("\n========== CONVERSATIONAL RAG CHATBOT ==========")

    session_id = input(
        "\nEnter conversation session ID: "
    ).strip()

    if not session_id:
        session_id = "day20_test_session"

    print(f"\nActive session: {session_id}")

    while True:

        question = input(
            "\nAsk a question (or type 'exit' to quit): "
        ).strip()

        if question.lower() == "exit":
            print("\nConversation ended.")
            break

        if not question:
            print("Please enter a question.")
            continue

        print("\nProcessing your question...")

        answer = generate_answer(
            question,
            session_id,
        )

        print("\n========== RAG CHATBOT ANSWER ==========")
        print(answer)
