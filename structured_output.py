from pydantic import BaseModel
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


class Topic(BaseModel):
    name: str
    category: str
    difficulty: str


response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents="Describe Python as a programming topic.",
    config={
        "response_mime_type": "application/json",
        "response_schema": Topic,
    },
)

topic = Topic.model_validate_json(response.text)

print(topic)
print(topic.model_dump())