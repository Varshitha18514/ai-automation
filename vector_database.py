import json
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

client = QdrantClient(path="./qdrant_storage")

COLLECTION_NAME = "ai_automation_documents"
VECTOR_SIZE = 768


def create_collection():
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )
        print("Collection created successfully!")


def load_embeddings():
    with open(
        "document_embeddings.json",
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def upload_embeddings():
    documents = load_embeddings()
    points = []

    for document in documents:
        metadata = document["metadata"]

        # Stable ID prevents duplicate points on repeated uploads.
        unique_key = (
            f"{metadata.get('source', '')}_"
            f"{metadata.get('page', '')}_"
            f"{metadata.get('chunk_index', '')}"
        )

        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_key))

        points.append(
            PointStruct(
                id=point_id,
                vector=document["embedding"],
                payload={
                    "content": document["content"],
                    "metadata": metadata,
                },
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    print(f"Successfully uploaded {len(points)} document chunks!")
def verify_collection():
    info = client.get_collection(COLLECTION_NAME)

    print("\n========== QDRANT DATABASE STATUS ==========")
    print("Collection:", COLLECTION_NAME)
    print("Stored vectors:", info.points_count)
    print("Vector dimensions:", VECTOR_SIZE)

if __name__ == "__main__":
    create_collection()
    upload_embeddings()
    verify_collection()

    client.close()