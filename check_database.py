from database import SessionLocal
from models import Job, ExecutionLog

db = SessionLocal()

try:
    print("\n========== SAVED JOBS ==========")

    jobs = db.query(Job).order_by(Job.id.desc()).all()

    for job in jobs:
        print(
            f"ID: {job.id} | "
            f"Task: {job.task_name} | "
            f"Status: {job.status}"
        )

    print("\n========== EXECUTION LOGS ==========")

    logs = (
        db.query(ExecutionLog)
        .order_by(ExecutionLog.id.desc())
        .all()
    )

    for log in logs:
        print(
            f"Job ID: {log.job_id} | "
            f"Level: {log.level} | "
            f"Message: {log.message}"
        )

finally:
    db.close()