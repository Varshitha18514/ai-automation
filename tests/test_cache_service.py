
import pytest

from cache_service import (
    get_cached_response,
    set_cached_response,
    generate_cache_key,
    get_semantic_cached_response,
    set_semantic_cached_response,
    redis_client,
)


def test_cache_miss():
    prompt = "unique cache miss test prompt"

    assert get_cached_response(prompt) is None


def test_cache_store_and_retrieve():
    prompt = "unique cache retrieval test prompt"
    answer = "Redis caching works!"

    set_cached_response(prompt, answer)

    assert get_cached_response(prompt) == answer


def test_different_prompts_have_different_keys():
    key1 = generate_cache_key("What is AI?")
    key2 = generate_cache_key("What is Python?")

    assert key1 != key2


def test_same_prompt_has_same_key():
    key1 = generate_cache_key("What is AI?")
    key2 = generate_cache_key("What is AI?")

    assert key1 == key2


def test_semantic_cache_hit(monkeypatch):
    prompt = "unique semantic cache hit test"

    key = (
        "ai_semantic_cache:"
        + __import__("hashlib").sha256(
            prompt.encode("utf-8")
        ).hexdigest()
    )

    redis_client.delete(key)

    try:
        set_semantic_cached_response(
            prompt,
            "Artificial intelligence explanation",
            [1.0, 0.0],
        )

        monkeypatch.setattr(
            "cache_service.generate_query_embedding",
            lambda text: [1.0, 0.0],
        )

        answer, embedding = get_semantic_cached_response(
            "another similar question"
        )

        assert answer == "Artificial intelligence explanation"
        assert embedding == [1.0, 0.0]

    finally:
        redis_client.delete(key)


def test_semantic_cache_miss(monkeypatch):
    prompt = "unique semantic cache miss test"

    key = (
        "ai_semantic_cache:"
        + __import__("hashlib").sha256(
            prompt.encode("utf-8")
        ).hexdigest()
    )

    redis_client.delete(key)

    try:
        set_semantic_cached_response(
            prompt,
            "Cached answer",
            [1.0, 0.0],
        )

        monkeypatch.setattr(
            "cache_service.generate_query_embedding",
            lambda text: [0.0, 1.0],
        )

        answer, embedding = get_semantic_cached_response(
            "completely different question"
        )

        assert answer is None
        assert embedding == [0.0, 1.0]

    finally:
        redis_client.delete(key)
