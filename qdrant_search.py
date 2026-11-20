from qdrant_client import QdrantClient

from embedding_generator import create_embedding

COLLECTION_NAME = "ai_automation_documents"

client = QdrantClient(path="./qdrant_storage")


def search_documents(query: str, limit: int = 3):
    # Convert the question into a query embedding
    query_vector = create_embedding(
        query,
        task_type="RETRIEVAL_QUERY",
    )

    # Search Qdrant for similar document vectors
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector.tolist(),
        limit=limit,
        with_payload=True,
    )

    return response.points


if __name__ == "__main__":
    question = input("\nEnter your question: ")

    results = search_documents(question)

    print("\n========== QDRANT SEARCH RESULTS ==========")

    for index, result in enumerate(results, start=1):
        print(f"\nResult {index}")
        print(f"Similarity score: {result.score:.4f}")
        print(f"Metadata: {result.payload['metadata']}")
        print(f"Content:\n{result.payload['content']}")
        print("-" * 60)

    client.close()