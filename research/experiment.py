"""
QuantLab Master Research Experiment Entity.

Defines Experiment entity dataclass, lifecycle status Enum, and automatic serialization
interfaces for JSON, YAML, and SQLite format persistence.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import os
import sqlite3
import uuid
from typing import Any, Dict, List, Optional, Union

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


class ExperimentStatus(str, Enum):
    """Lifecycle status enumeration for scientific experiments."""

    CREATED = "CREATED"
    INITIALIZED = "INITIALIZED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


@dataclass
class Experiment:
    """Institutional QuantLab Scientific Experiment container."""

    uuid: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Quantitative_Experiment"
    description: str = "Institutional quantitative strategy research experiment."
    author: str = "QuantLab_Researcher"
    date: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    version: str = "1.0.0"
    dataset: Dict[str, Any] = field(default_factory=dict)
    broker: str = "GenericBroker"
    asset: str = "EURUSD"
    timeframe: str = "1h"
    parameters: Dict[str, Any] = field(default_factory=dict)
    indicators: List[Dict[str, Any]] = field(default_factory=list)
    status: Union[ExperimentStatus, str] = ExperimentStatus.CREATED
    execution_time: float = 0.0
    results: Dict[str, Any] = field(default_factory=dict)
    hash: str = ""
    checksum: str = ""
    random_seed: int = 42
    configuration: Dict[str, Any] = field(default_factory=dict)
    system_metadata: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    resource_metrics: Dict[str, Any] = field(
        default_factory=lambda: {
            "ram_peak_mb": 0.0,
            "cpu_usage_pct": 0.0,
            "backtest_time_sec": 0.0,
            "walk_forward_time_sec": 0.0,
            "monte_carlo_time_sec": 0.0,
        }
    )

    def __post_init__(self) -> None:
        """Post-initialization validation and hash generation."""
        if isinstance(self.status, str):
            try:
                self.status = ExperimentStatus(self.status)
            except ValueError:
                pass
        if not self.hash:
            self.hash = self.calculate_hash()
        if not self.checksum:
            self.checksum = self.calculate_checksum()

    def calculate_hash(self) -> str:
        """Compute SHA-256 hash digest of experiment metadata and configuration.

        Returns:
            Hexadecimal SHA-256 string.
        """
        payload = {
            "uuid": self.uuid,
            "name": self.name,
            "version": self.version,
            "asset": self.asset,
            "timeframe": self.timeframe,
            "parameters": self.parameters,
            "random_seed": self.random_seed,
            "configuration": self.configuration,
        }
        raw_bytes = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(raw_bytes).hexdigest()

    def calculate_checksum(self) -> str:
        """Compute SHA-256 checksum digest of dataset and full results.

        Returns:
            Hexadecimal SHA-256 string.
        """
        payload = {
            "dataset": self.dataset,
            "results": self.results,
        }
        raw_bytes = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(raw_bytes).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert experiment instance into dictionary representation."""
        data = asdict(self)
        if isinstance(self.status, Enum):
            data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Experiment":
        """Reconstruct Experiment instance from dictionary.

        Args:
            data: Dictionary representation.

        Returns:
            Experiment object.
        """
        data_copy = dict(data)
        if "status" in data_copy and isinstance(data_copy["status"], str):
            try:
                data_copy["status"] = ExperimentStatus(data_copy["status"])
            except ValueError:
                pass
        return cls(**data_copy)

    def to_json(self, filepath: Optional[str] = None, indent: int = 2) -> str:
        """Serialize experiment to JSON string or JSON file.

        Args:
            filepath: Optional destination file path.
            indent: Indentation spaces.

        Returns:
            Serialized JSON string.
        """
        json_str = json.dumps(self.to_dict(), indent=indent, default=str)
        if filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(json_str)
        return json_str

    @classmethod
    def from_json(cls, json_str_or_path: str) -> "Experiment":
        """Deserialize Experiment from JSON string or file path.

        Args:
            json_str_or_path: JSON content string or file path.

        Returns:
            Experiment object.
        """
        if json_str_or_path.strip().startswith("{"):
            data = json.loads(json_str_or_path)
        else:
            with open(json_str_or_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        return cls.from_dict(data)

    def to_yaml(self, filepath: Optional[str] = None) -> str:
        """Serialize experiment to YAML string or YAML file.

        Args:
            filepath: Optional destination file path.

        Returns:
            Serialized YAML string.
        """
        data = self.to_dict()
        if HAS_YAML:
            yaml_str = yaml.dump(data, sort_keys=False)
        else:
            # Fallback simple custom YAML formatting
            yaml_str = json.dumps(data, indent=2, default=str)

        if filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(yaml_str)
        return yaml_str

    @classmethod
    def from_yaml(cls, yaml_str_or_path: str) -> "Experiment":
        """Deserialize Experiment from YAML string or file path.

        Args:
            yaml_str_or_path: YAML content string or file path.

        Returns:
            Experiment object.
        """
        if HAS_YAML:
            if os.path.exists(yaml_str_or_path):
                with open(yaml_str_or_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
            else:
                data = yaml.safe_load(yaml_str_or_path)
        else:
            return cls.from_json(yaml_str_or_path)
        return cls.from_dict(data)

    def to_sqlite(self, db_path_or_conn: Union[str, sqlite3.Connection], table_name: str = "experiments") -> None:
        """Serialize and persist experiment into SQLite database table.

        Args:
            db_path_or_conn: Path to SQLite DB file or active Connection instance.
            table_name: Table name.
        """
        conn_created = False
        if isinstance(db_path_or_conn, str):
            conn = sqlite3.connect(db_path_or_conn)
            conn_created = True
        else:
            conn = db_path_or_conn

        try:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    uuid TEXT PRIMARY KEY,
                    name TEXT,
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
                    payload_json TEXT
                )
                """
            )
            payload_json = self.to_json()
            status_val = self.status.value if isinstance(self.status, Enum) else str(self.status)
            cursor.execute(
                f"""
                INSERT OR REPLACE INTO {table_name} (
                    uuid, name, description, author, date, version, broker, asset,
                    timeframe, status, execution_time, hash, checksum, random_seed, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.uuid,
                    self.name,
                    self.description,
                    self.author,
                    self.date,
                    self.version,
                    self.broker,
                    self.asset,
                    self.timeframe,
                    status_val,
                    self.execution_time,
                    self.hash,
                    self.checksum,
                    self.random_seed,
                    payload_json,
                ),
            )
            conn.commit()
        finally:
            if conn_created:
                conn.close()

    @classmethod
    def from_sqlite(
        cls,
        db_path_or_conn: Union[str, sqlite3.Connection],
        exp_uuid: str,
        table_name: str = "experiments",
    ) -> "Experiment":
        """Deserialize Experiment from SQLite database by UUID.

        Args:
            db_path_or_conn: SQLite DB file path or Connection.
            exp_uuid: Target experiment UUID.
            table_name: Table name.

        Returns:
            Experiment instance.
        """
        conn_created = False
        if isinstance(db_path_or_conn, str):
            conn = sqlite3.connect(db_path_or_conn)
            conn_created = True
        else:
            conn = db_path_or_conn

        try:
            cursor = conn.cursor()
            cursor.execute(f"SELECT payload_json FROM {table_name} WHERE uuid = ?", (exp_uuid,))
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Experiment with UUID '{exp_uuid}' not found in SQLite table '{table_name}'.")
            return cls.from_json(row[0])
        finally:
            if conn_created:
                conn.close()

    def clone(self, new_name: Optional[str] = None, param_overrides: Optional[Dict[str, Any]] = None) -> "Experiment":
        """Clone current experiment with updated identity and parameter overrides.

        Args:
            new_name: Optional new experiment name.
            param_overrides: Optional parameter overrides dictionary.

        Returns:
            New cloned Experiment object.
        """
        cloned_dict = self.to_dict()
        cloned_dict["uuid"] = str(uuid.uuid4())
        cloned_dict["name"] = new_name or f"{self.name}_Clone"
        cloned_dict["date"] = datetime.now(timezone.utc).isoformat()
        cloned_dict["status"] = ExperimentStatus.CREATED.value
        cloned_dict["logs"] = []
        if param_overrides:
            cloned_dict["parameters"] = {**cloned_dict["parameters"], **param_overrides}

        cloned_exp = Experiment.from_dict(cloned_dict)
        cloned_exp.hash = cloned_exp.calculate_hash()
        cloned_exp.checksum = cloned_exp.calculate_checksum()
        return cloned_exp

    def increment_version(self, version_type: str = "patch") -> str:
        """Increment semantic version string.

        Args:
            version_type: One of 'major', 'minor', 'patch'.

        Returns:
            New version string.
        """
        parts = self.version.split(".")
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2]) if len(parts) > 2 else 0
        if version_type == "major":
            major += 1
            minor = 0
            patch = 0
        elif version_type == "minor":
            minor += 1
            patch = 0
        else:
            patch += 1
        self.version = f"{major}.{minor}.{patch}"
        self.hash = self.calculate_hash()
        return self.version
