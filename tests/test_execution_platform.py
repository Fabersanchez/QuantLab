"""
QuantLab Execution, Jobs & Orchestration Platform Test Suite.

Validates all Phase 19.4 components:
JobRecord, JobManager, WorkflowEngine, WorkflowStage, BaseTask, GenericTask, TaskEngine,
ExecutionQueue, ResourceManager, ResourceAllocationGrant, ExecutionSupervisor, BaseWorker,
WorkerInfo, GenericLocalWorker, WorkerFramework, PipelineEngine, PipelineStage, and ExecutionCenter.
"""

import unittest
from studio import (
    ExecutionCenter,
    ExecutionQueue,
    ExecutionSupervisor,
    GenericLocalWorker,
    GenericTask,
    JobManager,
    PipelineEngine,
    ResourceManager,
    StudioEventBus,
    TaskEngine,
    WorkerFramework,
    WorkflowEngine,
)


class TestQuantLabExecutionPlatform(unittest.TestCase):
    """Comprehensive Test Case for QuantLab Execution & Orchestration Platform."""

    def test_job_manager_lifecycle(self) -> None:
        """Test JobManager job creation, start, complete, and priority sorting."""
        bus = StudioEventBus()
        jm = JobManager(event_bus=bus)

        j1 = jm.create_job("LowPriorityJob", job_type="Backtest", priority=20)
        j2 = jm.create_job("HighPriorityJob", job_type="Optimization", priority=90)

        jobs = jm.list_jobs()
        self.assertEqual(jobs[0].job_id, j2.job_id)  # High priority first

        self.assertTrue(jm.start_job(j2.job_id))
        self.assertEqual(j2.status, "RUNNING")

        self.assertTrue(jm.complete_job(j2.job_id, result_payload={"sharpe": 2.3}))
        self.assertEqual(j2.status, "SUCCESS")

    def test_workflow_engine(self) -> None:
        """Test WorkflowEngine multi-stage pipeline execution."""
        wf = WorkflowEngine(workflow_name="BacktestPipeline")
        results = []

        wf.add_stage("s1", "Import Data", lambda: {"rows": 1000})
        wf.add_stage("s2", "Run Strategy", lambda: {"trades": 50})

        success = wf.execute_workflow()
        self.assertTrue(success)
        self.assertEqual(wf.status, "SUCCESS")
        self.assertEqual(wf.stages[0].output["rows"], 1000)

    def test_task_engine_lifecycle(self) -> None:
        """Test TaskEngine atomic task lifecycle execution."""
        task = GenericTask("t1", "CalcMetrics", lambda: {"sharpe": 1.95})
        res = TaskEngine.run_task(task)
        self.assertEqual(res["sharpe"], 1.95)

    def test_execution_queue_and_dlq(self) -> None:
        """Test ExecutionQueue priority heap, FIFO, and Dead Letter Queue."""
        eq = ExecutionQueue()
        jm = JobManager()

        j1 = jm.create_job("Job1", priority=10)
        j2 = jm.create_job("Job2", priority=80)

        eq.enqueue_priority(j1)
        eq.enqueue_priority(j2)

        popped1 = eq.dequeue_next()
        self.assertEqual(popped1.job_id, j2.job_id)  # Higher priority first

        eq.move_to_dead_letter_queue(j1)
        self.assertEqual(len(eq.get_dlq_jobs()), 1)

    def test_resource_manager(self) -> None:
        """Test ResourceManager hardware resource governor grants."""
        rm = ResourceManager(max_cpu_percent=100.0, max_mem_percent=100.0)
        grant = rm.request_allocation("job_101", required_threads=2, required_mem_mb=512)
        self.assertIsNotNone(grant)
        self.assertEqual(grant.granted_threads, 2)

        rm.release_allocation("job_101")

    def test_execution_supervisor(self) -> None:
        """Test ExecutionSupervisor process monitoring and timeout auto-recovery."""
        jm = JobManager()
        sup = ExecutionSupervisor(job_manager=jm, max_run_seconds=1.0)

        j = jm.create_job("TimeoutJob", priority=50)
        jm.start_job(j.job_id)
        j.elapsed_seconds = 10.0  # Exceed limit

        cancelled = sup.inspect_and_supervise()
        self.assertIn(j.job_id, cancelled)
        self.assertEqual(j.status, "FAILED")

    def test_worker_framework(self) -> None:
        """Test WorkerFramework local worker pool task assignment."""
        wf = WorkerFramework()
        worker = GenericLocalWorker("w1", "LocalWorker-1")
        wf.register_worker(worker)

        avail = wf.find_available_worker()
        self.assertIsNotNone(avail)
        self.assertTrue(avail.accept_task("task_001"))

    def test_pipeline_engine(self) -> None:
        """Test PipelineEngine reusable pipeline stage chaining."""
        pe = PipelineEngine("FeaturePipeline")
        pe.add_stage("scale", lambda x: x * 2.0, input_validator=lambda x: x > 0)

        out = pe.run_pipeline(10.0)
        self.assertEqual(out, 20.0)

    def test_execution_center(self) -> None:
        """Test ExecutionCenter real-time control hub telemetry."""
        ec = ExecutionCenter()
        telemetry = ec.get_execution_telemetry()
        self.assertIn("active_jobs_count", telemetry)
        self.assertTrue(telemetry["can_allocate_resources"])


if __name__ == "__main__":
    unittest.main()
