# ============================================================================
# AgentX.scheduler.scheduler - JobScheduler implementation
# ============================================================================

"""Deterministic scheduling engine.

Algorithm overview
------------------
1. **Topological order** (Kahn's algorithm): resolve dependency edges; a job
   becomes *ready* when all its predecessors have completed.
2. **Priority discipline**: ready jobs are dispatched by WSPT - highest
   ``weight / duration`` first - which is optimal for minimizing total
   weighted completion time on a single machine.
3. **Machine placement**: with ``machines > 1``, each machine tracks its next
   free time; a dispatched job goes to the earliest-free machine (ties broken
   by machine id for determinism).

Cycles in the dependency graph are detected and rejected with a clear error.
"""

from __future__ import annotations

import heapq
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from agentx.core import get_logger
from agentx.core.exceptions import SchedulingError, ValidationError
from agentx.core.schema import Job, ScheduleEntry

logger = get_logger("agentx.scheduler")


@dataclass
class ScheduleResult:
    """Result of a scheduling run, ready for Gantt rendering."""

    entries: List[ScheduleEntry] = field(default_factory=list)
    makespan: float = 0.0
    machine_utilization: Dict[str, float] = field(default_factory=dict)
    scheduled_jobs: List[str] = field(default_factory=list)
    total_work: float = 0.0


class JobScheduler:
    """Schedule a set of jobs on one or more identical machines.

    Parameters
    ----------
    machines : int
        Number of identical parallel machines (>= 1).
    tie_breaker : str, optional
        Not used directly - kept for API stability.
    """

    def __init__(self, machines: int = 1) -> None:
        if machines < 1:
            raise ValidationError("machines must be >= 1")
        self.machines = machines

    # --------------------------------------------------------------- public
    def schedule(self, jobs: List[Job]) -> ScheduleResult:
        """Schedule the jobs and return the timeline.

        Parameters
        ----------
        jobs : list[Job]
            Jobs with unique ``job_id``.

        Returns
        -------
        ScheduleResult
            Ordered timeline, makespan and per-machine utilization.

        Raises
        ------
        SchedulingError
            On duplicate ids or a cyclic dependency graph.
        """
        by_id: Dict[str, Job] = {}
        for job in jobs:
            if job.job_id in by_id:
                raise SchedulingError(f"duplicate job_id: {job.job_id}")
            by_id[job.job_id] = job

        # Validate dependencies point to existing jobs.
        for job in jobs:
            for dep in job.dependencies:
                if dep not in by_id:
                    raise SchedulingError(
                        f"job '{job.job_id}' depends on unknown '{dep}'"
                    )

        order = self._topological_order(jobs)
        result = self._dispatch(order, by_id)

        total_work = sum(j.duration for j in jobs)
        result.total_work = total_work
        result.scheduled_jobs = [e.job_id for e in result.entries]
        logger.info(
            "scheduled %d jobs on %d machines, makespan=%.2f",
            len(jobs), self.machines, result.makespan,
        )
        return result

    # --------------------------------------------------------------- core
    def _topological_order(self, jobs: List[Job]) -> List[str]:
        """Kahn's algorithm; raises ValueError on a cycle."""
        indegree: Dict[str, int] = {j.job_id: 0 for j in jobs}
        dependents: Dict[str, List[str]] = defaultdict(list)
        for job in jobs:
            for dep in job.dependencies:
                dependents[dep].append(job.job_id)
                indegree[job.job_id] += 1

        ready = deque([jid for jid, d in indegree.items() if d == 0])
        order: List[str] = []
        while ready:
            jid = ready.popleft()
            order.append(jid)
            for nxt in dependents[jid]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    ready.append(nxt)

        if len(order) != len(jobs):
            cyclic = [jid for jid, d in indegree.items() if d > 0]
            raise SchedulingError(f"dependency cycle detected among jobs: {cyclic}")
        return order

    def _dispatch(
        self, order: List[str], by_id: Dict[str, Job]
    ) -> ScheduleResult:
        """Greedy WSPT dispatch over earliest-free machines."""
        result = ScheduleResult()
        done_time: Dict[str, float] = {}
        machine_free: List[float] = [0.0] * self.machines
        machine_work: List[float] = [0.0] * self.machines

        # Ready set: jobs whose dependencies are all finished.
        remaining = set(order)
        dispatched_at: Dict[str, int] = {}

        while remaining:
            # Find the ready job with the highest weight/duration ratio.
            best: Optional[Job] = None
            best_rank = -1.0
            for jid in list(remaining):
                job = by_id[jid]
                if all(dep in done_time for dep in job.dependencies):
                    rank = (job.weight / job.duration) if job.duration > 0 else float("inf")
                    # Stable tie-breaking by id.
                    if rank > best_rank or (rank == best_rank and (best is None or jid < best.job_id)):
                        best = job
                        best_rank = rank

            if best is None:
                raise SchedulingError("scheduling deadlock - inconsistent dependency state")

            # Place on the earliest-free machine (lowest id wins ties).
            machine = int(min(range(self.machines), key=lambda m: (machine_free[m], m)))
            start = machine_free[machine]
            end = start + best.duration
            machine_free[machine] = end
            machine_work[machine] += best.duration
            done_time[best.job_id] = end
            dispatched_at[best.job_id] = machine
            remaining.discard(best.job_id)

            result.entries.append(
                ScheduleEntry(
                    job_id=best.job_id,
                    start=round(start, 4),
                    end=round(end, 4),
                    resource=best.resources,
                    machine=machine,
                )
            )

        result.makespan = max(machine_free) if machine_free else 0.0
        total_span = max(machine_free) if machine_free else 1.0
        result.machine_utilization = {
            f"machine_{i}": round(work / total_span, 4)
            for i, work in enumerate(machine_work)
        }
        return result

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<JobScheduler machines={self.machines}>"
