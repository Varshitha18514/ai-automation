
import os
import json

import numpy as np
from dotenv import load_dotenv
from google import genai

from document_processor import process_document

load_dotenv()


def create_embedding(
    text: str,
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> np.ndarray:
    """Generate a normalized Gemini embedding."""

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env")

    client = genai.Client(api_key=api_key)

    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config={
            "output_dimensionality": 768,
            "task_type": task_type,
        },
    )

    embedding = np.array(
        response.embeddings[0].values,
        dtype=np.float32,
    )

    norm = np.linalg.norm(embedding)

    if norm == 0:
        raise ValueError("Embedding has zero magnitude")

    return embedding / norm


def cosine_similarity(vector_a, vector_b) -> float:
    """Calculate cosine similarity between normalized vectors."""
    return float(np.dot(vector_a, vector_b))


def embed_document_chunks(file_path: str):
    """Generate retrieval embeddings for document chunks."""

    chunks = process_document(file_path)
    embedded_chunks = []

    print(f"Total document chunks: {len(chunks)}")

    for index, chunk in enumerate(chunks):
        print(f"Generating embedding {index + 1}/{len(chunks)}...")

        embedding = create_embedding(
            chunk.page_content,
            task_type="RETRIEVAL_DOCUMENT",
        )

        embedded_chunks.append({
            "content": chunk.page_content,
            "metadata": chunk.metadata,
            "embedding": embedding.tolist(),
        })

    return embedded_chunks


if __name__ == "__main__":

    document_path = "documents/ai_automation_guide.pdf"

    results = embed_document_chunks(document_path)

    output_path = "document_embeddings.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("\nDocument embedding generation completed!")
    print("Total embedded chunks:", len(results))
    print("Vector dimensions:", len(results[0]["embedding"]))
    print("Saved embeddings to:", output_path)
