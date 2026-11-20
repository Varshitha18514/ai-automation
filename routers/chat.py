
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from security import verify_api_key
from rag_pipeline import gemini

from hybrid_search import hybrid_search
from reranker import rerank_results

from conversation_memory import (
    save_message,
    get_recent_messages,
)

from conversation_summarizer import (
    get_conversation_summary,
)


router = APIRouter()


class ChatRequest(BaseModel):
    question: str
    session_id: str


def stream_answer(question: str, session_id: str):
    """Retrieve document context and stream Gemini's response."""

    history = get_recent_messages(session_id, limit=10)
    summary = get_conversation_summary(session_id)

    save_message(session_id, "user", question)

    history_text = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in history
    )

    search_results = hybrid_search(question, limit=5)

    if not search_results:
        answer = "I couldn't find relevant information in the documents."
        save_message(session_id, "assistant", answer)
        yield answer
        return

    reranked_results = rerank_results(
        question,
        search_results,
        limit=3,
    )

    if not reranked_results:
        answer = "I couldn't find relevant information in the documents."
        save_message(session_id, "assistant", answer)
        yield answer
        return

    context_text = "\n\n".join(
        item["content"] for item in reranked_results
    )

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

    full_answer = ""

    response_stream = gemini.models.generate_content_stream(
        model="gemini-3.5-flash-lite",
        contents=prompt,
    )

    for chunk in response_stream:
        if chunk.text:
            full_answer += chunk.text
            yield chunk.text

    if full_answer:
        save_message(session_id, "assistant", full_answer)


@router.post("/chat/stream")
async def chat_stream(
    chat_request: ChatRequest,
    authenticated: bool = Depends(verify_api_key),
):
    return StreamingResponse(
        stream_answer(
            chat_request.question,
            chat_request.session_id,
        ),
        media_type="text/plain",
    )
