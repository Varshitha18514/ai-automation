
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from rate_limiter import limiter

from routers.health import router as health_router
from routers.tasks import router as tasks_router
from routers.webhooks import router as webhooks_router
from routers.chat import router as chat_router


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Automation API",
    description="API for our 30-Day AI Automation Project",
    version="1.0.0"
)

app.state.limiter = limiter

app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(tasks_router)
app.include_router(webhooks_router)
app.include_router(chat_router)


@app.get("/")
@limiter.limit("5/minute")
async def root(request: Request):
    logger.info("Root endpoint accessed")

    return {
        "message": "Welcome to the AI Automation API!"
    }
