"""
QuantLab Institutional Experiment Registry & Database Sinks.

Provides relational persistence and historical tracking for scientific experiments, storing
complete execution histories, parameter configurations, logs, hashes, versioning trees,
and detailed resource consumption metrics (RAM, CPU, execution times per stage).
"""

from datetime import datetime, timezone
import json
import sqlite3
from typing import Any, Dict, List, Optional, Union

from research.experiment import Experiment, ExperimentStatus
from research.logger import get_research_logger

logger = get_research_logger("ExperimentRegistry")


class ExperimentRegistry:
    """Institutional Persistent Registry for QuantLab Scientific Experiments."""

    def __init__(self, db_path: str = ":memory:") -> None:
        """Initialize ExperimentRegistry.

        Args:
            db_path: Path to SQLite database file or ':memory:'.
        """
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create database connection instance."""
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        """Initialize relational database tables for institutional experiment persistence."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Main experiments table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    uuid TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    author TEXT,
                    date TEXT,
                    version TEXT,
                    broker TEXT,
                    asset TEXT,
                    timeframe TEXT,
                    status TEXT,
                    execution_time REAL,
                    hash TEXT,
                    checksum TEXT,
                    random_seed INTEGER,
                    parameters_json TEXT,
                    results_json TEXT,
                    config_json TEXT,
                    system_metadata_json TEXT,
                    payload_json TEXT
                )
                """
            )

            # History table tracking status changes & versioning
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS experiment_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_uuid TEXT NOT NULL,
                    status TEXT NOT NULL,
                    version TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    log_message TEXT,
                    FOREIGN KEY (experiment_uuid) REFERENCES experiments (uuid)
                )
                """
            )

            # Logs table recording detailed execution log messages
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS experiment_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_uuid TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    FOREIGN KEY (experiment_uuid) REFERENCES experiments (uuid)
                )
                """
            )

            # Resource consumption metrics table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS experiment_resource_usage (
                    experiment_uuid TEXT PRIMARY KEY,
                    ram_peak_mb REAL,
                    cpu_usage_pct REAL,
                    backtest_time_sec REAL,
                    walk_forward_time_sec REAL,
                    monte_carlo_time_sec REAL,
                    total_execution_time_sec REAL,
                    FOREIGN KEY (experiment_uuid) REFERENCES experiments (uuid)
                )
                """
            )
            conn.commit()

    def register(self, experiment: Experiment, log_message: str = "Initial experiment registration.") -> None:
        """Register and persist a new experiment in the institutional registry.

        Args:
            experiment: Experiment instance.
            log_message: Registration note.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            status_val = (
                experiment.status.value if isinstance(experiment.status, ExperimentStatus) else str(experiment.status)
            )

            cursor.execute(
                """
                INSERT OR REPLACE INTO experiments (
                    uuid, name, description, author, date, version, broker, asset,
                    timeframe, status, execution_time, hash, checksum, random_seed,
                    parameters_json, results_json, config_json, system_metadata_json, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment.uuid,
                    experiment.name,
                    experiment.description,
                    experiment.author,
                    experiment.date,
                    experiment.version,
                    experiment.broker,
                    experiment.asset,
                    experiment.timeframe,
                    status_val,
                    experiment.execution_time,
                    experiment.hash,
                    experiment.checksum,
                    experiment.random_seed,
                    json.dumps(experiment.parameters, default=str),
                    json.dumps(experiment.results, default=str),
                    json.dumps(experiment.configuration, default=str),
                    json.dumps(experiment.system_metadata, default=str),
                    experiment.to_json(),
                ),
            )

            # Record initial history
            cursor.execute(
                """
                INSERT INTO experiment_history (experiment_uuid, status, version, timestamp, log_message)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    experiment.uuid,
                    status_val,
                    experiment.version,
                    datetime.now(timezone.utc).isoformat(),
                    log_message,
                ),
            )

            # Record resource metrics
            res = experiment.resource_metrics or {}
            cursor.execute(
                """
                INSERT OR REPLACE INTO experiment_resource_usage (
                    experiment_uuid, ram_peak_mb, cpu_usage_pct,
                    backtest_time_sec, walk_forward_time_sec, monte_carlo_time_sec, total_execution_time_sec
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment.uuid,
                    float(res.get("ram_peak_mb", 0.0)),
                    float(res.get("cpu_usage_pct", 0.0)),
                    float(res.get("backtest_time_sec", 0.0)),
                    float(res.get("walk_forward_time_sec", 0.0)),
                    float(res.get("monte_carlo_time_sec", 0.0)),
                    float(experiment.execution_time),
                ),
            )

            conn.commit()
            logger.info(f"Experiment registered in database: UUID={experiment.uuid}")

    def update(self, experiment: Experiment, log_message: Optional[str] = None) -> None:
        """Update existing experiment record in registry.

        Args:
            experiment: Updated Experiment instance.
            log_message: Update note.
        """
        self.register(experiment, log_message=log_message or "Experiment state updated.")

    def get(self, exp_uuid: str) -> Optional[Experiment]:
        """Fetch Experiment by UUID from database.

        Args:
            exp_uuid: Target experiment UUID.

        Returns:
            Experiment instance or None if not found.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT payload_json FROM experiments WHERE uuid = ?", (exp_uuid,))
            row = cursor.fetchone()
            if not row:
                return None
            return Experiment.from_json(row[0])

    def delete(self, exp_uuid: str) -> bool:
        """Delete experiment record and related historical logs from database.

        Args:
            exp_uuid: Target experiment UUID.

        Returns:
            True if deleted, False otherwise.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM experiment_history WHERE experiment_uuid = ?", (exp_uuid,))
            cursor.execute("DELETE FROM experiment_logs WHERE experiment_uuid = ?", (exp_uuid,))
            cursor.execute("DELETE FROM experiment_resource_usage WHERE experiment_uuid = ?", (exp_uuid,))
            cursor.execute("DELETE FROM experiments WHERE uuid = ?", (exp_uuid,))
            deleted = cursor.rowcount > 0
            conn.commit()
            return deleted

    def query(
        self,
        name: Optional[str] = None,
        author: Optional[str] = None,
        status: Optional[Union[ExperimentStatus, str]] = None,
        asset: Optional[str] = None,
        limit: int = 100,
    ) -> List[Experiment]:
        """Query experiments with flexible parameter filters.

        Args:
            name: Name substring search filter.
            author: Author filter.
            status: Status filter.
            asset: Asset filter.
            limit: Maximum result rows.

        Returns:
            List of matching Experiment objects.
        """
        conditions = []
        params: List[Any] = []

        if name:
            conditions.append("name LIKE ?")
            params.append(f"%{name}%")
        if author:
            conditions.append("author = ?")
            params.append(author)
        if status:
            status_val = status.value if isinstance(status, ExperimentStatus) else str(status)
            conditions.append("status = ?")
            params.append(status_val)
        if asset:
            conditions.append("asset = ?")
            params.append(asset)

        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
        sql = f"SELECT payload_json FROM experiments{where_clause} ORDER BY date DESC LIMIT ?"
        params.append(limit)

        results: List[Experiment] = []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            for row in rows:
                try:
                    results.append(Experiment.from_json(row[0]))
                except Exception:
                    pass
        return results

    def get_history(self, exp_uuid: str) -> List[Dict[str, Any]]:
        """Retrieve status change and versioning history for an experiment.

        Args:
            exp_uuid: Target experiment UUID.

        Returns:
            List of history record dicts.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT status, version, timestamp, log_message
                FROM experiment_history
                WHERE experiment_uuid = ?
                ORDER BY id ASC
                """,
                (exp_uuid,),
            )
            rows = cursor.fetchall()
            return [
                {
                    "status": row[0],
                    "version": row[1],
                    "timestamp": row[2],
                    "log_message": row[3],
                }
                for row in rows
            ]

    def get_resource_consumption(self, exp_uuid: str) -> Dict[str, Any]:
        """Retrieve hardware resource consumption and duration metrics for an experiment.

        Args:
            exp_uuid: Target experiment UUID.

        Returns:
            Dictionary containing hardware and timing resource usage breakdown.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT ram_peak_mb, cpu_usage_pct, backtest_time_sec,
                       walk_forward_time_sec, monte_carlo_time_sec, total_execution_time_sec
                FROM experiment_resource_usage
                WHERE experiment_uuid = ?
                """,
                (exp_uuid,),
            )
            row = cursor.fetchone()
            if not row:
                return {}
            return {
                "ram_peak_mb": row[0],
                "cpu_usage_pct": row[1],
                "backtest_time_sec": row[2],
                "walk_forward_time_sec": row[3],
                "monte_carlo_time_sec": row[4],
                "total_execution_time_sec": row[5],
            }
