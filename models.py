
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    Text,
    Float,
    ForeignKey,
)

from database import Base


# Table 1: Store job information and webhook results
class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    task_name = Column(String(200), nullable=False)
    status = Column(String(50), default="Pending")

    # Day 13: Webhook pipeline fields
    webhook_payload = Column(Text, nullable=True)
    ai_result = Column(Text, nullable=True)

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )


# Table 2: Store AI token usage
class TokenMetric(Base):
    __tablename__ = "token_metrics"

    id = Column(Integer, primary_key=True, index=True)
    model = Column(String(100), nullable=False)
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    estimated_cost_usd = Column(Float, default=0.0)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )


# Table 3: Store execution logs
class ExecutionLog(Base):
    __tablename__ = "execution_logs"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=True)
    message = Column(Text, nullable=False)
    level = Column(String(20), default="INFO")
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )
