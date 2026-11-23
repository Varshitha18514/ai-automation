
import os
import hmac
import re

from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

load_dotenv()

API_SECRET_KEY = os.getenv("API_SECRET_KEY")

bearer_scheme = HTTPBearer(auto_error=False)

# Basic prompt-injection detection patterns
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"disregard\s+(all\s+)?(previous\s+)?instructions",
    r"reveal\s+(your\s+)?system\s+prompt",
    r"show\s+(me\s+)?your\s+system\s+prompt",
    r"ignore\s+the\s+system\s+prompt",
    r"act\s+as\s+an\s+unrestricted",
]


def sanitize_input(text: str) -> str:
    """Validate and sanitize user input."""

    if not isinstance(text, str) or not text.strip():
        raise HTTPException(
            status_code=400,
            detail="Input cannot be empty",
        )

    if len(text) > 5000:
        raise HTTPException(
            status_code=413,
            detail="Input is too long",
        )

    # Remove non-printable control characters
    text = "".join(
        char for char in text
        if char.isprintable() or char in "\n\t"
    )

    # Detect common prompt-injection attempts
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            raise HTTPException(
                status_code=400,
                detail="Potential prompt injection detected",
            )

    return text.strip()


async def verify_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme)
):
    if not API_SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API authentication is not configured"
        )

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
            headers={"WWW-Authenticate": "Bearer"}
        )

    if not hmac.compare_digest(
        credentials.credentials,
        API_SECRET_KEY
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"}
        )

    return True
def sanitize_output(text: str) -> str:
    """Prevent common secret patterns from being exposed."""

    sensitive_patterns = [
        r"AIza[0-9A-Za-z_-]{35}",       # Google API keys
        r"AKIA[0-9A-Z]{16}",            # AWS access keys
        r"-----BEGIN PRIVATE KEY-----",
        r"-----BEGIN RSA PRIVATE KEY-----",
        r"Bearer\s+[A-Za-z0-9._-]{20,}",
        r"sk-[A-Za-z0-9_-]{20,}",       # Secret-key patterns
    ]

    for pattern in sensitive_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return "I cannot provide this response because it may contain sensitive information."

    return text
