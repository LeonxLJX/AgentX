# ============================================================================
# Example: JobScheduler - WSPT priorities, DAG dependencies, parallel machines
# Run:  python examples/demo_scheduler.py
# ============================================================================

"""Schedules a small project-like DAG and prints the Gantt-ready timeline."""

from agentx.core.schema import Job
from agentx.scheduler import JobScheduler

if __name__ == "__main__":
    jobs = [
        Job(job_id="A", duration=3, priority=5, weight=2.0),
        Job(job_id="B", duration=2, priority=1, dependencies=["A"]),
        Job(job_id="C", duration=4, priority=3),
        Job(job_id="D", duration=1, priority=2, dependencies=["B", "C"]),
        Job(job_id="E", duration=5, priority=4),
        Job(job_id="F", duration=2, priority=1, dependencies=["D"]),
    ]

    result = JobScheduler(machines=2).schedule(jobs)
    print(f"makespan = {result.makespan}")
    print(f"utilization = {result.machine_utilization}")
    print(f"\n{'job':<6}{'start':>8}{'end':>8}{'machine':>10}")
    for entry in result.entries:
        print(f"{entry.job_id:<6}{entry.start:>8.1f}{entry.end:>8.1f}{entry.machine:>10}")
