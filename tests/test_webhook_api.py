
import hashlib
import hmac
import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import main
import routers.webhooks as webhooks
from database import Base
from models import Job, ExecutionLog


@pytest.fixture
def test_db(tmp_path, monkeypatch):
    # Create a separate temporary SQLite database.
    database_path = tmp_path / "test_database.db"

    engine = create_engine(
        f"sqlite:///{database_path}",
        connect_args={"check_same_thread": False}
    )

    TestingSessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine
    )

    Base.metadata.create_all(bind=engine)

    # Replace the webhook router's database session.
    monkeypatch.setattr(
        webhooks,
        "SessionLocal",
        TestingSessionLocal
    )

    yield TestingSessionLocal

    engine.dispose()


@pytest.fixture
def client():
    return TestClient(main.app)


def create_signature(body):
    return hmac.new(
        webhooks.WEBHOOK_SECRET.encode("utf-8"),
        body,
        hashlib.sha256
    ).hexdigest()


def test_webhook_missing_signature(client, test_db):
    response = client.post(
        "/webhooks/receive",
        json={"event": "new_order"}
    )

    assert response.status_code == 401
    assert response.json()["detail"] == (
        "Missing webhook signature"
    )


def test_webhook_invalid_signature(client, test_db):
    response = client.post(
        "/webhooks/receive",
        content=b'{"event": "new_order"}',
        headers={
            "Content-Type": "application/json",
            "X-Signature": "invalid-signature"
        }
    )

    assert response.status_code == 401
    assert response.json()["detail"] == (
        "Invalid webhook signature"
    )


def test_webhook_invalid_json(client, test_db):
    body = b'{"event": invalid-json}'

    response = client.post(
        "/webhooks/receive",
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-Signature": create_signature(body)
        }
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Invalid JSON payload"
    )


def test_webhook_non_object_payload(client, test_db):
    body = json.dumps(["order", 101]).encode("utf-8")

    response = client.post(
        "/webhooks/receive",
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-Signature": create_signature(body)
        }
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Webhook payload must be a JSON object"
    )


def test_webhook_success(client, test_db, monkeypatch):
    # Mock Gemini so no real API request is made.
    fake_ai_result = {
        "event": "new_order",
        "summary": "New order 101 created for John.",
        "order_id": "101",
        "customer": "John",
        "status": ""
    }

    monkeypatch.setattr(
        webhooks,
        "analyze_webhook",
        lambda payload: fake_ai_result
    )

    payload = {
        "event": "new_order",
        "order_id": 101,
        "customer": "John"
    }

    body = json.dumps(payload).encode("utf-8")

    response = client.post(
        "/webhooks/receive",
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-Signature": create_signature(body)
        }
    )

    assert response.status_code == 200

    response_data = response.json()

    assert response_data["message"] == (
        "Webhook accepted successfully!"
    )
    assert response_data["status"] == "Pending"

    # Verify the background task saved its results.
    db = test_db()

    try:
        job = db.query(Job).filter(
            Job.id == response_data["job_id"]
        ).first()

        assert job is not None
        assert job.status == "Completed"

        saved_payload = json.loads(job.webhook_payload)
        assert saved_payload == payload

        saved_result = json.loads(job.ai_result)
        assert saved_result == fake_ai_result

        logs = db.query(ExecutionLog).filter(
            ExecutionLog.job_id == job.id
        ).all()

        assert len(logs) == 2

    finally:
        db.close()

