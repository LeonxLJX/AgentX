# ============================================================================
# AgentX.scheduler - Module 4: AI Job Scheduling
# ============================================================================

"""Job scheduling engine for AI-automation and dispatch outsourcing.

Supports the scheduling patterns clients most often ask for:

- **Dependency scheduling** : jobs form a DAG; a job starts only after all of
  its dependencies finish (Kahn topological ordering).
- **Priority scheduling**   : weighted shortest processing time (WSPT)
  minimizes total weighted completion time for ready jobs.
- **Parallel machines**     : schedule across ``machines`` identical machines
  and get makespan + per-machine utilization.

Output is a Gantt-friendly list of ``(job, start, end)`` entries that can be
rendered directly in the API frontend.

Quick start
-----------
>>> from agentx.scheduler import JobScheduler
>>> from agentx.core.schema import Job
>>> jobs = [Job("A", 3, priority=5), Job("B", 2, priority=1, dependencies=["A"])]
>>> result = JobScheduler().schedule(jobs, machines=2)
>>> result.makespan
5.0
"""

from agentx.scheduler.scheduler import JobScheduler, ScheduleResult

__all__ = ["JobScheduler", "ScheduleResult"]
