
import os
import json
import time
import asyncio
from datetime import datetime

from dotenv import load_dotenv
from google import genai
from openai import AsyncOpenAI

from ragas.llms import llm_factory
from ragas.embeddings import GoogleEmbeddings
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)

from hybrid_search import hybrid_search
from reranker import rerank_results


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY was not found in .env")

GENERATION_MODEL = "gemini-3.5-flash-lite"

EVALUATION_MODEL = os.getenv(
    "RAGAS_EVALUATION_MODEL",
    GENERATION_MODEL
)

# Wait between individual evaluation operations.
REQUEST_DELAY = 12

# Wait before retrying a rate-limited request.
RETRY_DELAY = 35

gemini_client = genai.Client(api_key=API_KEY)

evaluator_client = AsyncOpenAI(
    api_key=API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

evaluator_llm = llm_factory(
    EVALUATION_MODEL,
    client=evaluator_client,
)

evaluator_embeddings = GoogleEmbeddings(
    client=gemini_client,
    model="gemini-embedding-001",
)


# --------------------------------------------------
# 2. Metrics
# --------------------------------------------------

metrics = {
    "faithfulness": Faithfulness(llm=evaluator_llm),

    "answer_relevancy": AnswerRelevancy(
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    ),

    "context_precision": ContextPrecision(
        llm=evaluator_llm,
    ),

    "context_recall": ContextRecall(
        llm=evaluator_llm,
    ),
}


# --------------------------------------------------
# 3. Benchmark questions
# --------------------------------------------------

benchmark = [
    {
        "question": "What is AI automation?",
        "reference": (
            "AI automation uses artificial intelligence to perform "
            "tasks that traditionally require human effort. It improves "
            "productivity, reduces repetitive work, and minimizes errors."
        ),
    },
    {
        "question": "What can Large Language Models do?",
        "reference": (
            "Large Language Models can summarize documents, answer "
            "questions, classify text, and extract structured information."
        ),
    },
    {
        "question": "Why is text chunking important?",
        "reference": (
            "Chunking divides long documents into smaller sections, "
            "making them easier to process and retrieve."
        ),
    },
    {
        "question": "What is overlap chunking?",
        "reference": (
            "Overlap chunking keeps a small portion of text from the "
            "previous chunk in the next chunk. This helps preserve context."
        ),
    },
    {
        "question": "What is Retrieval-Augmented Generation?",
        "reference": (
            "Retrieval-Augmented Generation retrieves relevant document "
            "chunks before generating an answer. It helps language models "
            "answer using external documents instead of relying only "
            "on their training data."
        ),
    },
]


# --------------------------------------------------
# 4. Detect rate-limit errors
# --------------------------------------------------

def is_rate_limit_error(error):

    message = str(error).upper()

    return (
        "429" in message
        or "RESOURCE_EXHAUSTED" in message
        or "RATE LIMIT" in message
    )


# --------------------------------------------------
# 5. Generate answer with retry
# --------------------------------------------------

def generate_rag_answer(question, contexts):

    context_text = "\n\n".join(contexts)

    prompt = f"""
You are an AI assistant. Answer using only the supplied
document context.

If the answer is unavailable in the context, say so.

DOCUMENT CONTEXT:
{context_text}

QUESTION:
{question}

ANSWER:
"""

    for attempt in range(2):

        time.sleep(REQUEST_DELAY)

        try:

            response = gemini_client.models.generate_content(
                model=GENERATION_MODEL,
                contents=prompt,
            )

            return response.text or ""

        except Exception as error:

            if is_rate_limit_error(error) and attempt == 0:

                print(
                    f"Rate limit reached. Waiting {RETRY_DELAY} seconds..."
                )

                time.sleep(RETRY_DELAY)
                continue

            raise


# --------------------------------------------------
# 6. Run one metric with retry
# --------------------------------------------------

async def run_metric(metric_name, metric, metric_input):

    for attempt in range(2):

        await asyncio.sleep(REQUEST_DELAY)

        try:

            result = await metric.ascore(**metric_input)

            value = getattr(result, "value", result)

            return float(value)

        except Exception as error:

            if is_rate_limit_error(error) and attempt == 0:

                print(
                    f"{metric_name}: Rate limit reached. "
                    f"Waiting {RETRY_DELAY} seconds before retry..."
                )

                await asyncio.sleep(RETRY_DELAY)
                continue

            print(f"{metric_name} failed: {error}")
            return None

    return None


# --------------------------------------------------
# 7. Evaluate one question
# --------------------------------------------------

async def evaluate_question(question, reference):

    print("\n" + "=" * 65)
    print("QUESTION:", question)

    search_results = hybrid_search(
        question,
        limit=5,
    )

    ranked_results = rerank_results(
        question,
        search_results,
        limit=3,
    )

    contexts = [
        item["content"]
        for item in ranked_results
        if item.get("content")
    ]

    if not contexts:

        print("No document context was retrieved.")

        return {
            "question": question,
            "reference": reference,
            "answer": "",
            "contexts": [],
            "scores": {},
        }

    print("Retrieved contexts:", len(contexts))

    try:

        answer = generate_rag_answer(
            question,
            contexts,
        )

    except Exception as error:

        print("Answer generation failed:", error)

        return {
            "question": question,
            "reference": reference,
            "answer": "",
            "contexts": contexts,
            "scores": {},
        }

    print("\nGENERATED ANSWER:")
    print(answer)

    metric_inputs = {
        "faithfulness": {
            "user_input": question,
            "response": answer,
            "retrieved_contexts": contexts,
        },

        "answer_relevancy": {
            "user_input": question,
            "response": answer,
        },

        "context_precision": {
            "user_input": question,
            "reference": reference,
            "retrieved_contexts": contexts,
        },

        "context_recall": {
            "user_input": question,
            "reference": reference,
            "retrieved_contexts": contexts,
        },
    }

    scores = {}

    for metric_name, metric in metrics.items():

        score = await run_metric(
            metric_name,
            metric,
            metric_inputs[metric_name],
        )

        scores[metric_name] = score

        print(f"{metric_name}: {score}")

    return {
        "question": question,
        "reference": reference,
        "answer": answer,
        "contexts": contexts,
        "scores": scores,
    }


# --------------------------------------------------
# 8. Run complete evaluation
# --------------------------------------------------

async def run_evaluation():

    print("\nDAY 21 - RAG EVALUATION")
    print("-" * 65)

    print("Generation model:", GENERATION_MODEL)
    print("Evaluation model:", EVALUATION_MODEL)

    results = []

    try:

        for item in benchmark:

            result = await evaluate_question(
                item["question"],
                item["reference"],
            )

            results.append(result)

    finally:

        await evaluator_client.close()

    report = {
        "date": datetime.now().isoformat(),
        "generation_model": GENERATION_MODEL,
        "evaluation_model": EVALUATION_MODEL,
        "total_questions": len(results),
        "results": results,
    }

    report_file = "day21_evaluation_report.json"

    with open(
        report_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print("\n" + "=" * 65)
    print("EVALUATION FINISHED")
    print("Report saved:", report_file)

    print("\nAVERAGE METRIC SCORES")

    for metric_name in metrics:

        values = [
            result["scores"].get(metric_name)
            for result in results
            if isinstance(
                result["scores"].get(metric_name),
                (int, float),
            )
        ]

        if values:

            average = sum(values) / len(values)

            print(f"{metric_name}: {average:.4f}")

        else:

            print(f"{metric_name}: No successful scores")


# --------------------------------------------------
# 9. Start evaluation
# --------------------------------------------------

if __name__ == "__main__":

    asyncio.run(run_evaluation())
