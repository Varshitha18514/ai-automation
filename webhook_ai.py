
import json
import os
import time

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel


load_dotenv()


class WebhookAnalysis(BaseModel):
    event: str
    summary: str
    order_id: str
    customer: str
    status: str


def analyze_webhook(payload: dict) -> dict:
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env")

    client = genai.Client(api_key=api_key)

    last_error = None

    # Retry up to 3 times
    for attempt in range(1, 4):
        try:
            print(f"Gemini API attempt {attempt}/3")

            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=(
                    "Analyze this webhook payload and extract its "
                    "information. Return a concise summary. "
                    "Use supplied values only. If a field is missing, "
                    "use an empty string.\n\n"
                    f"Webhook payload: {json.dumps(payload)}"
                ),
                config={
                    "response_mime_type": "application/json",
                    "response_schema": WebhookAnalysis,
                },
            )

            result = WebhookAnalysis.model_validate_json(
                response.text
            )

            return result.model_dump()

        except Exception as error:
            last_error = error
            print(f"Gemini attempt {attempt} failed: {error}")

            if attempt < 3:
                time.sleep(2 ** attempt)

    raise RuntimeError(
        f"Gemini failed after 3 attempts: {last_error}"
    )
