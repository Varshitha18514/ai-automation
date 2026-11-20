
import re
from functools import lru_cache

from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient

from embedding_generator import create_embedding


COLLECTION_NAME = "ai_automation_documents"
RRF_K = 60
KEYWORD_LIMIT = 20


@lru_cache(maxsize=1)
def get_qdrant_client():
    return QdrantClient(path="./qdrant_storage")


@lru_cache(maxsize=1)
def load_bm25_index():

    qdrant = get_qdrant_client()

    points = []
    offset = None

    while True:

        batch, offset = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=500,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        points.extend(batch)

        if offset is None:
            break

    if not points:
        return [], None

    documents = [
        point.payload.get("content", "")
        for point in points
    ]

    tokenized = [
        re.findall(r"\b\w+\b", doc.lower())
        for doc in documents
    ]

    return points, BM25Okapi(tokenized)


def hybrid_search(query: str, limit: int = 3):

    points, bm25 = load_bm25_index()

    if not points or bm25 is None:
        return []

    tokens = re.findall(r"\b\w+\b", query.lower())

    if not tokens:
        return []

    # BM25 keyword search
    scores = bm25.get_scores(tokens)

    keyword_indices = scores.argsort()[::-1][:KEYWORD_LIMIT]

    keyword_rankings = {}

    for rank, index in enumerate(keyword_indices, start=1):

        if scores[index] > 0:
            keyword_rankings[points[index].id] = rank

    # Vector search
    query_embedding = create_embedding(
        query,
        task_type="RETRIEVAL_QUERY",
    )

    qdrant = get_qdrant_client()

    vector_response = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding.tolist(),
        limit=limit * 3,
        with_payload=True,
    )

    vector_rankings = {
        point.id: rank
        for rank, point in enumerate(
            vector_response.points,
            start=1,
        )
    }

    # Reciprocal Rank Fusion
    point_lookup = {
        point.id: point
        for point in points
    }

    all_ids = set(keyword_rankings) | set(vector_rankings)

    results = []

    for point_id in all_ids:

        score = 0.0

        if point_id in keyword_rankings:
            score += 1 / (
                RRF_K + keyword_rankings[point_id]
            )

        if point_id in vector_rankings:
            score += 1 / (
                RRF_K + vector_rankings[point_id]
            )

        point = point_lookup[point_id]
        payload = point.payload or {}

        results.append({
            "content": payload.get("content", ""),
            "metadata": payload.get("metadata", {}),
            "rrf_score": score,
            "keyword_rank": keyword_rankings.get(point_id),
            "vector_rank": vector_rankings.get(point_id),
        })

    results.sort(
        key=lambda item: item["rrf_score"],
        reverse=True,
    )

    return results[:limit]


if __name__ == "__main__":

    try:

        question = input("\nEnter your search query: ")

        results = hybrid_search(question)

        print("\n========== HYBRID SEARCH RESULTS ==========")

        for index, result in enumerate(results, start=1):

            print(f"\nResult {index}")
            print("RRF Score:", round(result["rrf_score"], 5))
            print("BM25 Rank:", result["keyword_rank"])
            print("Vector Rank:", result["vector_rank"])
            print("Content:", result["content"])
            print("Metadata:", result["metadata"])

    finally:

        load_bm25_index.cache_clear()

        if get_qdrant_client.cache_info().currsize:
            get_qdrant_client().close()

        get_qdrant_client.cache_clear()
