"""
QuantLab Optimization Task Scheduler.

Manages priority task queues, multi-core parallel worker pools, evaluation timeouts,
cancellation tokens, and state checkpoints.
"""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from queue import PriorityQueue
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from optimization.evaluator import EvaluationResult, SolutionEvaluator
from optimization.logger import get_optimization_logger

logger = get_optimization_logger("Scheduler")


@dataclass(order=True)
class OptimizationTask:
    """Dataclass holding a prioritized candidate evaluation task."""

    priority: int
    task_id: int
    parameters: Dict[str, Any] = field(compare=False)


class OptimizationScheduler:
    """Institutional Scheduler for Parallel Optimization Tasks."""

    def __init__(self, max_workers: int = 4, timeout_sec: Optional[float] = None) -> None:
        """Initialize OptimizationScheduler.

        Args:
            max_workers: Maximum worker threads/processes.
            timeout_sec: Evaluation timeout in seconds.
        """
        self.max_workers = max(1, max_workers)
        self.timeout_sec = timeout_sec
        self.task_queue: PriorityQueue = PriorityQueue()
        self._lock = threading.RLock()
        self._task_counter: int = 0
        self._is_cancelled: bool = False
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="OptWorker")

    def submit_task(self, parameters: Dict[str, Any], priority: int = 10) -> int:
        """Enqueue parameter evaluation task with priority (lower int = higher priority).

        Returns:
            Task ID integer.
        """
        with self._lock:
            self._task_counter += 1
            task = OptimizationTask(priority=priority, task_id=self._task_counter, parameters=parameters)
            self.task_queue.put(task)
            return task.task_id

    def cancel_all(self) -> None:
        """Cancel all pending queued tasks."""
        with self._lock:
            self._is_cancelled = True
            while not self.task_queue.empty():
                try:
                    self.task_queue.get_nowait()
                except Exception:
                    break
            logger.info("Cancelled all scheduler tasks.")

    def run_batch(
        self, evaluator: SolutionEvaluator, parameter_batch: List[Dict[str, Any]]
    ) -> List[EvaluationResult]:
        """Execute a batch of parameter combinations in parallel using worker pool.

        Args:
            evaluator: SolutionEvaluator instance.
            parameter_batch: List of parameter dictionaries.

        Returns:
            List of EvaluationResult objects.
        """
        if not parameter_batch:
            return []

        futures = {}
        with self._lock:
            self._is_cancelled = False

        for params in parameter_batch:
            if self._is_cancelled:
                break
            future = self._executor.submit(evaluator.evaluate, params)
            futures[future] = params

        results: List[EvaluationResult] = []
        for future in as_completed(futures):
            if self._is_cancelled:
                break
            params = futures[future]
            try:
                if self.timeout_sec:
                    res = future.result(timeout=self.timeout_sec)
                else:
                    res = future.result()
                results.append(res)
            except Exception as exc:
                logger.error(f"Scheduler worker failed on params {params}: {exc}")
                results.append(
                    EvaluationResult(
                        parameters=params,
                        fitness_score=-9999.0,
                        is_valid=False,
                        metrics={},
                        violations=[f"Worker error: {str(exc)}"],
                        execution_time_sec=0.0,
                    )
                )

        return results

    def shutdown(self) -> None:
        """Shutdown worker executor pool."""
        self.cancel_all()
        self._executor.shutdown(wait=False)
