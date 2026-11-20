from fastapi import APIRouter, BackgroundTasks, Depends, Request
from pydantic import BaseModel
import logging
import time

from database import SessionLocal
from models import Job, ExecutionLog
from security import verify_api_key
from rate_limiter import limiter

router = APIRouter()
logger = logging.getLogger(__name__)


class TaskRequest(BaseModel):
    task_name: str


def process_task(job_id: int, task_name: str):
    db = SessionLocal()

    try:
        job = db.query(Job).filter(Job.id == job_id).first()

        if job is None:
            return

        job.status = "Processing"
        db.add(
            ExecutionLog(
                job_id=job_id,
                message=f"Background task started: {task_name}",
                level="INFO"
            )
        )
        db.commit()

        logger.info(f"Background task started: {task_name}")

        time.sleep(5)

        job.status = "Completed"
        db.add(
            ExecutionLog(
                job_id=job_id,
                message=f"Background task completed: {task_name}",
                level="INFO"
            )
        )
        db.commit()

        logger.info(f"Background task completed: {task_name}")

    except Exception:
        db.rollback()
        logger.exception("Background task failed")

        job = db.query(Job).filter(Job.id == job_id).first()

        if job is not None:
            job.status = "Failed"
            db.add(
                ExecutionLog(
                    job_id=job_id,
                    message=f"Background task failed: {task_name}",
                    level="ERROR"
                )
            )
            db.commit()

    finally:
        db.close()


@router.post("/tasks")
@limiter.limit("5/minute")
async def create_task(
    request: Request,
    task_request: TaskRequest,
    background_tasks: BackgroundTasks,
    authenticated: bool = Depends(verify_api_key)
):
    db = SessionLocal()

    try:
        new_job = Job(
            task_name=task_request.task_name,
            status="Pending"
        )

        db.add(new_job)
        db.commit()
        db.refresh(new_job)

        job_id = new_job.id

        background_tasks.add_task(
            process_task,
            job_id,
            task_request.task_name
        )

        return {
            "message": "Task saved to database successfully!",
            "job_id": job_id,
            "task_name": task_request.task_name,
            "status": "processing in background"
        }

    finally:
        db.close()