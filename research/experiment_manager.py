"""
QuantLab Master Experiment Manager.

Orchestrates the lifecycle, state updates, search, versioning, cloning, deletion, and
concurrent execution of quantitative research experiments with institutional thread safety.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import Any, Callable, Dict, List, Optional, Union

from research.experiment import Experiment, ExperimentStatus
from research.experiment_registry import ExperimentRegistry
from research.logger import get_research_logger
from research.metadata import MetadataExtractor
from research.reproducibility import ReproducibilityManager

logger = get_research_logger("ExperimentManager")


class ExperimentManager:
    """Institutional Manager for QuantLab scientific research experiments."""

    def __init__(self, registry: Optional[ExperimentRegistry] = None, max_workers: int = 4) -> None:
        """Initialize ExperimentManager.

        Args:
            registry: Optional persistent ExperimentRegistry instance.
            max_workers: Maximum threads for concurrent execution pool.
        """
        self.registry = registry or ExperimentRegistry(":memory:")
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ExpManagerWorker")
        self._memory_cache: Dict[str, Experiment] = {}

    def create_experiment(
        self,
        name: str,
        description: str = "",
        author: str = "QuantLab_Researcher",
        dataset: Optional[Dict[str, Any]] = None,
        broker: str = "GenericBroker",
        asset: str = "EURUSD",
        timeframe: str = "1h",
        parameters: Optional[Dict[str, Any]] = None,
        indicators: Optional[List[Dict[str, Any]]] = None,
        random_seed: int = 42,
        configuration: Optional[Dict[str, Any]] = None,
    ) -> Experiment:
        """Create, initialize, and register a new scientific experiment.

        Returns:
            Newly instantiated and registered Experiment.
        """
        with self._lock:
            config = configuration or {}
            params = parameters or {}
            ds = dataset or {}

            # Collect system metadata & reproducibility
            context = ReproducibilityManager.capture_context(
                seed=random_seed,
                config=config,
                dataset_repr=ds,
                broker=broker,
            )

            exp = Experiment(
                name=name,
                description=description,
                author=author,
                dataset=ds,
                broker=broker,
                asset=asset,
                timeframe=timeframe,
                parameters=params,
                indicators=indicators or [],
                status=ExperimentStatus.INITIALIZED,
                random_seed=random_seed,
                configuration=config,
                system_metadata=context.environment_metadata,
            )

            self._memory_cache[exp.uuid] = exp
            self.registry.register(exp, log_message="Created and registered experiment.")
            logger.log_creation(exp.uuid, exp.name, exp.author)
            return exp

    def get_experiment(self, exp_uuid: str) -> Optional[Experiment]:
        """Retrieve experiment by UUID from cache or registry.

        Args:
            exp_uuid: Experiment UUID.

        Returns:
            Experiment object or None if not found.
        """
        with self._lock:
            if exp_uuid in self._memory_cache:
                return self._memory_cache[exp_uuid]
            exp = self.registry.get(exp_uuid)
            if exp:
                self._memory_cache[exp.uuid] = exp
            return exp

    def update_status(
        self,
        exp_uuid: str,
        status: Union[ExperimentStatus, str],
        log_message: Optional[str] = None,
        results: Optional[Dict[str, Any]] = None,
        execution_time: Optional[float] = None,
    ) -> Experiment:
        """Update experiment status and optional results/execution metrics.

        Args:
            exp_uuid: Target experiment UUID.
            status: New ExperimentStatus.
            log_message: Status transition explanation.
            results: Results dictionary update.
            execution_time: Total duration seconds update.

        Returns:
            Updated Experiment instance.
        """
        with self._lock:
            exp = self.get_experiment(exp_uuid)
            if not exp:
                raise ValueError(f"Experiment with UUID '{exp_uuid}' not found.")

            status_enum = ExperimentStatus(status) if isinstance(status, str) else status
            exp.status = status_enum

            if results is not None:
                exp.results.update(results)
                exp.checksum = exp.calculate_checksum()
            if execution_time is not None:
                exp.execution_time = float(execution_time)

            note = log_message or f"Status updated to '{status_enum.value}'."
            exp.logs.append({"timestamp": exp.date, "status": status_enum.value, "message": note})

            self._memory_cache[exp.uuid] = exp
            self.registry.update(exp, log_message=note)
            logger.info(f"Updated experiment status: UUID={exp.uuid} -> {status_enum.value}")
            return exp

    def cancel_experiment(self, exp_uuid: str, reason: str = "User cancelled execution.") -> Experiment:
        """Cancel an ongoing experiment execution."""
        return self.update_status(exp_uuid, ExperimentStatus.CANCELLED, log_message=reason)

    def pause_experiment(self, exp_uuid: str) -> Experiment:
        """Pause an ongoing experiment execution."""
        exp = self.update_status(exp_uuid, ExperimentStatus.PAUSED, log_message="Execution paused.")
        logger.log_pause(exp.uuid)
        return exp

    def resume_experiment(self, exp_uuid: str) -> Experiment:
        """Resume a paused experiment execution."""
        exp = self.update_status(exp_uuid, ExperimentStatus.RUNNING, log_message="Execution resumed.")
        logger.log_resume(exp.uuid)
        return exp

    def delete_experiment(self, exp_uuid: str) -> bool:
        """Delete an experiment from memory cache and registry database."""
        with self._lock:
            if exp_uuid in self._memory_cache:
                del self._memory_cache[exp_uuid]
            return self.registry.delete(exp_uuid)

    def search_experiments(
        self,
        query: Optional[str] = None,
        author: Optional[str] = None,
        status: Optional[Union[ExperimentStatus, str]] = None,
        asset: Optional[str] = None,
    ) -> List[Experiment]:
        """Search registered experiments using criteria filters.

        Returns:
            List of matching Experiments.
        """
        with self._lock:
            return self.registry.query(name=query, author=author, status=status, asset=asset)

    def list_experiments(
        self,
        filter_fn: Optional[Callable[[Experiment], bool]] = None,
        sort_by: str = "date",
        reverse: bool = True,
    ) -> List[Experiment]:
        """List experiments stored in registry database with filtering and sorting.

        Returns:
            Sorted list of Experiment objects.
        """
        with self._lock:
            all_experiments = self.registry.query(limit=1000)
            if filter_fn:
                all_experiments = [e for e in all_experiments if filter_fn(e)]

            all_experiments.sort(key=lambda e: getattr(e, sort_by, e.date), reverse=reverse)
            return all_experiments

    def clone_experiment(
        self,
        exp_uuid: str,
        new_name: Optional[str] = None,
        param_overrides: Optional[Dict[str, Any]] = None,
    ) -> Experiment:
        """Clone an existing experiment with a new identity and parameter overrides.

        Returns:
            New registered cloned Experiment.
        """
        with self._lock:
            source = self.get_experiment(exp_uuid)
            if not source:
                raise ValueError(f"Source experiment '{exp_uuid}' not found.")

            cloned_exp = source.clone(new_name=new_name, param_overrides=param_overrides)
            self._memory_cache[cloned_exp.uuid] = cloned_exp
            self.registry.register(cloned_exp, log_message=f"Cloned from source experiment {exp_uuid}")
            logger.info(f"Cloned experiment: Source={exp_uuid} -> Cloned={cloned_exp.uuid}")
            return cloned_exp

    def version_experiment(self, exp_uuid: str, version_type: str = "patch") -> Experiment:
        """Increment semantic version of an experiment and log update in registry.

        Returns:
            Experiment instance with updated version.
        """
        with self._lock:
            exp = self.get_experiment(exp_uuid)
            if not exp:
                raise ValueError(f"Experiment '{exp_uuid}' not found.")

            old_version = exp.version
            new_version = exp.increment_version(version_type=version_type)
            self.registry.update(exp, log_message=f"Version updated: {old_version} -> {new_version}")
            logger.info(f"Version updated for {exp.uuid}: {old_version} -> {new_version}")
            return exp

    def run_concurrently(
        self,
        task_fn: Callable[[Experiment], Dict[str, Any]],
        experiments: List[Experiment],
    ) -> List[Experiment]:
        """Execute experiment task function concurrently across thread pool.

        Args:
            task_fn: Function accepting Experiment and returning result dictionary.
            experiments: List of experiments to run.

        Returns:
            List of completed Experiment instances.
        """
        futures = {}
        for exp in experiments:
            self.update_status(exp.uuid, ExperimentStatus.RUNNING, log_message="Concurrent execution started.")
            future = self._executor.submit(task_fn, exp)
            futures[future] = exp

        completed: List[Experiment] = []
        for future in as_completed(futures):
            exp = futures[future]
            try:
                result_data = future.result()
                updated = self.update_status(
                    exp.uuid,
                    ExperimentStatus.COMPLETED,
                    results=result_data,
                    log_message="Concurrent task finished successfully.",
                )
                completed.append(updated)
            except Exception as exc:
                updated = self.update_status(
                    exp.uuid,
                    ExperimentStatus.FAILED,
                    log_message=f"Concurrent execution failed: {str(exc)}",
                )
                completed.append(updated)
                logger.log_error(exp.uuid, str(exc))

        return completed
