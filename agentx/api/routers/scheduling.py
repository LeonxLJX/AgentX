# ============================================================================
# AgentX.api.routers.scheduling - Scheduler endpoints
# ============================================================================

"""Endpoint
---------
- ``POST /api/schedule`` : schedule jobs with priorities, DAG dependencies
  and resources across N parallel machines; returns a Gantt-ready timeline.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from agentx.core import get_logger
from agentx.core.exceptions import SchedulingError, ValidationError
from agentx.core.schema import Job
from agentx.scheduler import JobScheduler

logger = get_logger("agentx.api.scheduling")
router = APIRouter()


class JobRequest(BaseModel):
    """One schedulable job."""

    job_id: str = Field(..., description="Unique job id")
    duration: float = Field(..., gt=0, description="Processing time")
    priority: int = Field(0, description="Higher = more urgent")
    dependencies: List[str] = Field(default_factory=list, description="Ids that must finish first")
    resources: float = Field(1.0, gt=0, description="Resource units consumed")
    deadline: Optional[float] = Field(None, description="Optional due time")
    weight: float = Field(1.0, gt=0, description="Weight for WSPT ordering")


class ScheduleRequest(BaseModel):
    """Scheduling problem description."""

    machines: int = Field(1, ge=1, le=64, description="Number of parallel machines")
    jobs: List[JobRequest] = Field(..., min_length=1)


@router.post("/schedule", summary="Schedule jobs across parallel machines")
def schedule(payload: ScheduleRequest) -> Dict[str, Any]:
    """Return timeline, makespan and per-machine utilization.

    The endpoint translates :class:`SchedulingError` /
    :class:`ValidationError` raised by :class:`JobScheduler` into HTTP 400
    responses automatically via the registered exception handlers.
    """
    jobs = [
        Job(
            job_id=j.job_id,
            duration=j.duration,
            priority=j.priority,
            dependencies=j.dependencies,
            resources=j.resources,
            deadline=j.deadline,
            weight=j.weight,
        )
        for j in payload.jobs
    ]
    result = JobScheduler(machines=payload.machines).schedule(jobs)
    logger.info(
        "schedule solved: machines=%d jobs=%d makespan=%.2f",
        payload.machines,
        len(jobs),
        result.makespan,
    )
    return {
        "entries": [asdict(e) for e in result.entries],
        "makespan": result.makespan,
        "machine_utilization": result.machine_utilization,
        "scheduled_jobs": result.scheduled_jobs,
        "total_work": result.total_work,
    }
