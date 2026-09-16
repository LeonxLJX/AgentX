# ============================================================================
# Tests: JobScheduler
# ============================================================================

import pytest

from agentx.core.schema import Job
from agentx.scheduler import JobScheduler


def test_single_machine_ordering():
    jobs = [
        Job(job_id="A", duration=2, priority=1),
        Job(job_id="B", duration=1, priority=2),
    ]
    result = JobScheduler(machines=1).schedule(jobs)
    assert result.entries[0].job_id == "B"  # higher priority first
    assert result.makespan == pytest.approx(3.0)


def test_dependencies_forced_order():
    jobs = [
        Job(job_id="A", duration=1),
        Job(job_id="B", duration=1, dependencies=["A"]),
    ]
    result = JobScheduler(machines=1).schedule(jobs)
    ids = [e.job_id for e in result.entries]
    assert ids.index("A") < ids.index("B")
    assert result.entries[1].start >= result.entries[0].end


def test_parallel_machines_shorten_makespan():
    jobs = [Job(job_id=f"J{i}", duration=5) for i in range(4)]
    single = JobScheduler(machines=1).schedule(jobs)
    parallel = JobScheduler(machines=2).schedule(jobs)
    assert parallel.makespan <= single.makespan
    assert parallel.makespan == pytest.approx(10.0)


def test_cycle_detected():
    jobs = [
        Job(job_id="A", duration=1, dependencies=["B"]),
        Job(job_id="B", duration=1, dependencies=["A"]),
    ]
    with pytest.raises(ValueError, match="cycle"):
        JobScheduler().schedule(jobs)


def test_unknown_dependency_rejected():
    jobs = [Job(job_id="A", duration=1, dependencies=["ghost"])]
    with pytest.raises(ValueError, match="unknown"):
        JobScheduler().schedule(jobs)


def test_duplicate_id_rejected():
    jobs = [Job(job_id="A", duration=1), Job(job_id="A", duration=2)]
    with pytest.raises(ValueError, match="duplicate"):
        JobScheduler().schedule(jobs)


def test_utilization_sums_to_one_single_machine():
    jobs = [Job(job_id="A", duration=2), Job(job_id="B", duration=3)]
    result = JobScheduler(machines=1).schedule(jobs)
    assert result.machine_utilization["machine_0"] == pytest.approx(1.0)
