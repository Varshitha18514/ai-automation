
import json
from unittest.mock import MagicMock

import webhook_ai


def test_analyze_webhook_success(monkeypatch):
    # Provide a dummy API key so no real key is needed.
    monkeypatch.setenv("GEMINI_API_KEY", "test-api-key")

    # Create a fake Gemini response.
    fake_response = MagicMock()
    fake_response.text = json.dumps({
        "event": "new_order",
        "summary": "New order 101 created for customer John.",
        "order_id": "101",
        "customer": "John",
        "status": ""
    })

    # Create a fake Gemini client.
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = fake_response

    # Replace the real Gemini client with our fake client.
    monkeypatch.setattr(
        webhook_ai.genai,
        "Client",
        lambda api_key: fake_client
    )

    # Run the function.
    payload = {
        "event": "new_order",
        "order_id": 101,
        "customer": "John"
    }

    result = webhook_ai.analyze_webhook(payload)

    # Verify the returned result.
    assert result["event"] == "new_order"
    assert result["summary"] == (
        "New order 101 created for customer John."
    )
    assert result["order_id"] == "101"
    assert result["customer"] == "John"
    assert result["status"] == ""

    # Verify Gemini was called exactly once.
    fake_client.models.generate_content.assert_called_once()
