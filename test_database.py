from database import SessionLocal
from models import Job, TokenMetric, ExecutionLog


db = SessionLocal()

try:
    # Create a test job
    new_job = Job(
        task_name="Day 11 Database Test",
        status="Completed"
    )

    db.add(new_job)
    db.commit()
    db.refresh(new_job)

    # Create a token usage record
    new_metric = TokenMetric(
        model="Gemini",
        input_tokens=100,
        output_tokens=50,
        estimated_cost_usd=0.001
    )

    db.add(new_metric)

    # Create an execution log
    new_log = ExecutionLog(
        job_id=new_job.id,
        message="Database test completed successfully!",
        level="INFO"
    )

    db.add(new_log)
    db.commit()

    print("Database records saved successfully!")

    print("\nSaved Job:")
    print(new_job.id, new_job.task_name, new_job.status)

    print("\nSaved Token Metric:")
    print(new_metric.model, new_metric.input_tokens,
          new_metric.output_tokens)

    print("\nSaved Execution Log:")
    print(new_log.message)

finally:
    db.close()