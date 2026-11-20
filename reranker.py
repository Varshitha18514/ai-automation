from functools import lru_cache

from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def load_model():
    return CrossEncoder(MODEL_NAME)


def rerank_results(
    query: str,
    search_results: list,
    limit: int = 3,
):
    """Efficiently rerank search results using a cached cross-encoder."""

    if not search_results or limit <= 0:
        return []

    model = load_model()

    pairs = [
        [query, result.get("content", "")]
        for result in search_results
    ]

    scores = model.predict(
        pairs,
        batch_size=16,
        show_progress_bar=False,
    )

    reranked_results = []

    for result, score in zip(search_results, scores):
        updated_result = result.copy()
        updated_result["rerank_score"] = float(score)
        reranked_results.append(updated_result)

    reranked_results.sort(
        key=lambda item: item["rerank_score"],
        reverse=True,
    )

    return reranked_results[:limit]


if __name__ == "__main__":

    from hybrid_search import hybrid_search

    question = input("\nEnter your question: ")

    print("\nRunning hybrid search...")

    results = hybrid_search(question, limit=5)

    print("\nRunning cross-encoder re-ranking...")

    reranked = rerank_results(
        question,
        results,
        limit=3,
    )

    print("\n========== OPTIMIZED RERANKED RESULTS ==========")

    for index, result in enumerate(reranked, start=1):
        print(f"\nResult {index}")
        print("Relevance Score:", round(result["rerank_score"], 4))
        print("Original RRF Score:", round(result["rrf_score"], 5))
        print("Content:", result["content"])
        print("Metadata:", result["metadata"])