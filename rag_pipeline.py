
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

from telemetry import tracer

from security import sanitize_output

from cache_service import (
    get_cached_response,
    set_cached_response,
    get_semantic_cached_response,
    set_semantic_cached_response,
)


# --------------------------------------------------
# INITIALIZATION
# --------------------------------------------------

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing from your .env file")

gemini = genai.Client(api_key=api_key)


# --------------------------------------------------
# RAG PIPELINE
# --------------------------------------------------

def generate_answer(question: str, session_id: str):
    """Generate an answer using RAG, memory, exact caching,
    semantic caching, security, and telemetry.
    """

    # Step 1: Load conversation memory
    with tracer.start_as_current_span("load_conversation_memory"):

        history = get_recent_messages(
            session_id,
            limit=10,
        )

        summary = get_conversation_summary(session_id)

        save_message(
            session_id,
            "user",
            question,
        )

        history_text = "\n".join(
            f"{message['role']}: {message['content']}"
            for message in history
        )

    # Step 2: Hybrid search
    with tracer.start_as_current_span("hybrid_search"):

        search_results = hybrid_search(
            question,
            limit=5,
        )

    if not search_results:
        answer = "I couldn't find relevant information in the documents."
        save_message(session_id, "assistant", answer)
        return answer

    # Step 3: Re-rank results
    with tracer.start_as_current_span("rerank_results"):

        reranked_results = rerank_results(
            question,
            search_results,
            limit=3,
        )

    if not reranked_results:
        answer = "I couldn't find relevant information in the documents."
        save_message(session_id, "assistant", answer)
        return answer

    # Step 4: Prepare complete prompt
    with tracer.start_as_current_span("prepare_document_context"):

        context_text = "\n\n".join(
            item["content"]
            for item in reranked_results
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

    # Step 5: Exact and semantic cache lookup
    answer = None
    query_embedding = None

    with tracer.start_as_current_span("cache_lookup") as cache_span:

        # First check exact-match cache
        try:
            cached_answer = get_cached_response(prompt)
        except Exception:
            cached_answer = None
            cache_span.set_attribute("cache.exact_error", True)

        if cached_answer is not None:

            cache_span.set_attribute("cache.hit", True)
            cache_span.set_attribute("cache.type", "exact")

            answer = sanitize_output(cached_answer)

        else:

            cache_span.set_attribute("cache.exact_hit", False)

            # Check semantic cache if exact cache misses
            try:
                semantic_answer, query_embedding = (
                    get_semantic_cached_response(prompt)
                )

                if semantic_answer is not None:

                    cache_span.set_attribute("cache.hit", True)
                    cache_span.set_attribute("cache.type", "semantic")

                    answer = sanitize_output(semantic_answer)

                    # Also store semantic hit in exact cache
                    if answer:
                        try:
                            set_cached_response(prompt, answer)
                        except Exception:
                            pass

            except Exception:
                # Continue to Gemini if semantic caching fails
                cache_span.set_attribute(
                    "cache.semantic_error",
                    True,
                )

    # Step 6: Generate answer if both caches miss
    if answer is None:

        with tracer.start_as_current_span(
            "gemini_generate_answer"
        ) as span:

            response = gemini.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=prompt,
            )

            raw_answer = response.text or ""

            span.set_attribute(
                "llm.model",
                "gemini-3.5-flash-lite",
            )

        # Sanitize Gemini output before caching
        answer = sanitize_output(raw_answer)

        # Store safe answer in exact cache
        if answer:

            try:
                set_cached_response(
                    prompt,
                    answer,
                )
            except Exception:
                pass

            # Store safe answer in semantic cache
            try:

                # Generate embedding if semantic lookup failed
                if query_embedding is None:

                    from embedding_generator import create_embedding

                    query_embedding = create_embedding(
                        prompt,
                        task_type="RETRIEVAL_QUERY",
                    ).tolist()

                set_semantic_cached_response(
                    prompt,
                    answer,
                    query_embedding,
                )

            except Exception:
                # Caching failure must not break the chatbot
                pass

    # Step 7: Save assistant response
    with tracer.start_as_current_span("save_assistant_response"):

        if answer:
            save_message(
                session_id,
                "assistant",
                answer,
            )

    return answer


# --------------------------------------------------
# COMMAND LINE INTERFACE
# --------------------------------------------------

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
