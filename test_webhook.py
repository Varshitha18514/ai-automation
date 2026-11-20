import hashlib
import hmac
import json
import urllib.request
import urllib.error

# Must match the secret in routers/webhooks.py
WEBHOOK_SECRET = "local-demo-secret"

URL = "http://127.0.0.1:8000/webhooks/receive"

# Sample webhook data
payload = {
    "event": "new_order",
    "order_id": 101,
    "customer": "John"
}

# Convert payload to JSON bytes
body = json.dumps(payload).encode("utf-8")

# Generate valid HMAC-SHA256 signature
signature = hmac.new(
    WEBHOOK_SECRET.encode("utf-8"),
    body,
    hashlib.sha256
).hexdigest()

# Prepare request
request = urllib.request.Request(
    URL,
    data=body,
    headers={
        "Content-Type": "application/json",
        "X-Signature": signature
    },
    method="POST"
)

print("Sending webhook...")

try:
    with urllib.request.urlopen(request) as response:
        print("Status code:", response.status)
        print("Response:", response.read().decode("utf-8"))

except urllib.error.HTTPError as error:
    print("Request failed:", error.code)
    print(error.read().decode("utf-8"))
    