
import os
import json
import hashlib

import numpy as np
import redis

# Redis connection
redis_client = redis.Redis.from_url(
    os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    decode_responses=True,
)

# Cache settings
CACHE_TTL = 3600
SEMANTIC_CACHE_PREFIX = "ai_semantic_cache:"
SIMILARITY_THRESHOLD = float(
    os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.95")
)


# --------------------------------------------------
# EXACT MATCH CACHE
# --------------------------------------------------

def generate_cache_key(prompt: str) -> str:
    """Generate a SHA-256 key for an exact prompt."""

    prompt_hash = hashlib.sha256(
        prompt.encode("utf-8")
    ).hexdigest()

    return f"ai_cache:{prompt_hash}"


def get_cached_response(prompt: str):
    """Retrieve an exact-match cached response."""

    key = generate_cache_key(prompt)

    cached = redis_client.get(key)

    if cached is None:
        return None

    return json.loads(cached)


def set_cached_response(prompt: str, response: str):
    """Store an exact-match response in Redis."""

    key = generate_cache_key(prompt)

    redis_client.set(
        key,
        json.dumps(response),
        ex=CACHE_TTL,
    )


# --------------------------------------------------
# SEMANTIC CACHE
# --------------------------------------------------

def generate_query_embedding(text: str):
    """Generate a normalized query embedding using Gemini."""

    from embedding_generator import create_embedding

    embedding = create_embedding(
        text,
        task_type="RETRIEVAL_QUERY",
    )

    return embedding.tolist()


def get_semantic_cached_response(text: str):
    """
    Search Redis for a semantically similar prompt.

    Returns:
        cached response or None,
        query embedding for reuse when storing a new response.
    """

    query_embedding = generate_query_embedding(text)

    query_vector = np.array(
        query_embedding,
        dtype=np.float32,
    )

    best_response = None
    best_similarity = -1.0

    # Search existing semantic cache entries
    for key in redis_client.scan_iter(
        match=f"{SEMANTIC_CACHE_PREFIX}*"
    ):

        cached_data = redis_client.get(key)

        if cached_data is None:
            continue

        try:
            data = json.loads(cached_data)

            cached_vector = np.array(
                data["embedding"],
                dtype=np.float32,
            )

            if cached_vector.shape != query_vector.shape:
                continue

            similarity = float(
                np.dot(query_vector, cached_vector)
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_response = data["response"]

        except (ValueError, KeyError, TypeError):
            continue

    # Return only sufficiently similar matches
    if best_similarity >= SIMILARITY_THRESHOLD:
        return best_response, query_embedding

    return None, query_embedding


def set_semantic_cached_response(
    text: str,
    response: str,
    embedding: list,
):
    """Store a response and its embedding in Redis."""

    text_hash = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

    key = f"{SEMANTIC_CACHE_PREFIX}{text_hash}"

    cache_data = {
        "text": text,
        "embedding": embedding,
        "response": response,
    }

    redis_client.set(
        key,
        json.dumps(cache_data),
        ex=CACHE_TTL,
    )
