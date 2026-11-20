
import hashlib
import hmac
import json
import os
import logging

from fastapi import APIRouter, Request, HTTPException, BackgroundTasks

from database import SessionLocal
from models import Job, ExecutionLog
from webhook_ai import analyze_webhook


router = APIRouter()
logger = logging.getLogger(__name__)

WEBHOOK_SECRET = os.getenv(
    "WEBHOOK_SECRET",
    "local-demo-secret"
)


def process_webhook(job_id: int, payload: dict):
    db = SessionLocal()

    try:
        job = db.query(Job).filter(Job.id == job_id).first()

        if job is None:
            return

        job.status = "Processing"

        db.add(
            ExecutionLog(
                job_id=job_id,
                message="Webhook background processing started.",
                level="INFO"
            )
        )
        db.commit()

        logger.info("Processing webhook job %s", job_id)

        # Send webhook payload to Gemini
        ai_result = analyze_webhook(payload)

        # Save Gemini result
        job.ai_result = json.dumps(ai_result)
        job.status = "Completed"

        db.add(
            ExecutionLog(
                job_id=job_id,
                message="Gemini analysis completed successfully.",
                level="INFO"
            )
        )

        db.commit()

        logger.info("Webhook job %s completed", job_id)

    except Exception as error:
        db.rollback()

        logger.exception("Webhook job %s failed", job_id)

        job = db.query(Job).filter(Job.id == job_id).first()

        if job is not None:
            job.status = "Failed"

            db.add(
                ExecutionLog(
                    job_id=job_id,
                    message=f"Webhook processing failed: {str(error)}",
                    level="ERROR"
                )
            )

            db.commit()

    finally:
        db.close()


@router.post("/webhooks/receive")
async def receive_webhook(
    request: Request,
    background_tasks: BackgroundTasks
):

    # 1. Read original request body
    body = await request.body()

    # 2. Verify signature
    received_signature = request.headers.get("X-Signature")

    if not received_signature:
        raise HTTPException(
            status_code=401,
            detail="Missing webhook signature"
        )

    expected_signature = hmac.new(
        WEBHOOK_SECRET.encode("utf-8"),
        body,
        hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(
        received_signature,
        expected_signature
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid webhook signature"
        )

    # 3. Parse JSON
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Invalid JSON payload"
        )

    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=400,
            detail="Webhook payload must be a JSON object"
        )

    # 4. Save job in SQLite
    db = SessionLocal()

    try:
        new_job = Job(
            task_name=f"Webhook: {payload.get('event', 'Unknown event')}",
            status="Pending",
            webhook_payload=json.dumps(payload)
        )

        db.add(new_job)
        db.commit()
        db.refresh(new_job)

        job_id = new_job.id

    except Exception:
        db.rollback()
        logger.exception("Could not save webhook job")

        raise HTTPException(
            status_code=500,
            detail="Could not save webhook job"
        )

    finally:
        db.close()

    # 5. Schedule Gemini processing
    background_tasks.add_task(
        process_webhook,
        job_id,
        payload
    )

    # 6. Return immediately
    return {
        "message": "Webhook accepted successfully!",
        "job_id": job_id,
        "status": "Pending"
    }
