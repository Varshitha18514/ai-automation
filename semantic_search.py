
import json

import numpy as np

from embedding_generator import create_embedding


def load_embeddings(file_path: str = "document_embeddings.json"):
    """Load previously generated document embeddings."""

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def search_documents(
    query: str,
    top_k: int = 3,
):
    """Find the most relevant document chunks for a question."""

    documents = load_embeddings()

    # Generate an embedding for the user's question
    query_embedding = create_embedding(
        query,
        task_type="RETRIEVAL_QUERY",
    )

    results = []

    for document in documents:

        document_embedding = np.array(
            document["embedding"],
            dtype=np.float32,
        )

        # Both vectors are normalized, so dot product
        # gives cosine similarity.
        similarity = float(
            np.dot(query_embedding, document_embedding)
        )

        results.append({
            "similarity": similarity,
            "content": document["content"],
            "metadata": document["metadata"],
        })

    # Sort from highest similarity to lowest
    results.sort(
        key=lambda item: item["similarity"],
        reverse=True,
    )

    return results[:top_k]


if __name__ == "__main__":

    question = input("\nEnter your question: ")

    search_results = search_documents(question)

    print("\n========== SEARCH RESULTS ==========")

    for index, result in enumerate(search_results, start=1):

        print(f"\nResult {index}")
        print(f"Similarity: {result['similarity']:.4f}")
        print(f"Metadata: {result['metadata']}")
        print(f"Content:\n{result['content']}")

        print("-" * 60)

