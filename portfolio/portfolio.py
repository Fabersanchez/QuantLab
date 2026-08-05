"""
QuantLab Master Portfolio Entity Container.

Defines Portfolio dataclass encapsulating asset universes, capital weights, cash reserves,
positions, versioning, history tracking, and JSON/SQLite serialization.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import sqlite3
import uuid
from typing import Any, Dict, List, Optional, Union

from portfolio.asset import Asset


@dataclass
class Portfolio:
    """Institutional QuantLab Portfolio Container."""

    portfolio_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "QuantLab_Institutional_Portfolio"
    description: str = "Multi-asset quantitative portfolio."
    version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    initial_capital: float = 100000.0
    current_cash: float = 100000.0
    assets: Dict[str, Asset] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)
    positions: Dict[str, float] = field(default_factory=dict)
    history_events: List[Dict[str, Any]] = field(default_factory=list)

    def add_asset(self, asset: Asset, weight: float = 0.0) -> None:
        """Add asset to portfolio universe with target allocation weight.

        Args:
            asset: Asset instance.
            weight: Allocation weight float (0.0 to 1.0).
        """
        self.assets[asset.symbol] = asset
        self.weights[asset.symbol] = float(weight)

    def remove_asset(self, symbol: str) -> None:
        """Remove asset from portfolio universe."""
        if symbol in self.assets:
            del self.assets[symbol]
        if symbol in self.weights:
            del self.weights[symbol]
        if symbol in self.positions:
            del self.positions[symbol]

    def set_weights(self, weights: Dict[str, float]) -> None:
        """Update target asset allocation weights."""
        self.weights = {k: float(v) for k, v in weights.items()}

    def to_dict(self) -> Dict[str, Any]:
        """Convert Portfolio instance into dictionary representation."""
        data = {
            "portfolio_id": self.portfolio_id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "created_at": self.created_at,
            "initial_capital": self.initial_capital,
            "current_cash": self.current_cash,
            "assets": {sym: ast.to_dict() for sym, ast in self.assets.items()},
            "weights": self.weights,
            "positions": self.positions,
            "history_events": self.history_events,
        }
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Portfolio":
        """Reconstruct Portfolio instance from dictionary representation."""
        data_copy = dict(data)
        if "assets" in data_copy and isinstance(data_copy["assets"], dict):
            assets_dict = {}
            for sym, ast_data in data_copy["assets"].items():
                assets_dict[sym] = Asset.from_dict(ast_data)
            data_copy["assets"] = assets_dict
        return cls(**data_copy)

    def to_json(self, filepath: Optional[str] = None, indent: int = 2) -> str:
        """Serialize Portfolio to JSON string or JSON file."""
        json_str = json.dumps(self.to_dict(), indent=indent, default=str)
        if filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(json_str)
        return json_str

    @classmethod
    def from_json(cls, json_str_or_path: str) -> "Portfolio":
        """Deserialize Portfolio from JSON string or file path."""
        import os

        if os.path.exists(json_str_or_path):
            with open(json_str_or_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = json.loads(json_str_or_path)
        return cls.from_dict(data)

    def to_sqlite(self, db_path_or_conn: Union[str, sqlite3.Connection], table_name: str = "portfolios") -> None:
        """Persist portfolio record to SQLite database table."""
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
                    portfolio_id TEXT PRIMARY KEY,
                    name TEXT,
                    description TEXT,
                    version TEXT,
                    created_at TEXT,
                    initial_capital REAL,
                    current_cash REAL,
                    payload_json TEXT
                )
                """
            )
            cursor.execute(
                f"""
                INSERT OR REPLACE INTO {table_name} (
                    portfolio_id, name, description, version, created_at, initial_capital, current_cash, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.portfolio_id,
                    self.name,
                    self.description,
                    self.version,
                    self.created_at,
                    self.initial_capital,
                    self.current_cash,
                    self.to_json(),
                ),
            )
            conn.commit()
        finally:
            if conn_created:
                conn.close()

    @classmethod
    def from_sqlite(
        cls, db_path_or_conn: Union[str, sqlite3.Connection], portfolio_id: str, table_name: str = "portfolios"
    ) -> "Portfolio":
        """Deserialize Portfolio from SQLite database by ID."""
        conn_created = False
        if isinstance(db_path_or_conn, str):
            conn = sqlite3.connect(db_path_or_conn)
            conn_created = True
        else:
            conn = db_path_or_conn

        try:
            cursor = conn.cursor()
            cursor.execute(f"SELECT payload_json FROM {table_name} WHERE portfolio_id = ?", (portfolio_id,))
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Portfolio '{portfolio_id}' not found in SQLite table '{table_name}'.")
            return cls.from_json(row[0])
        finally:
            if conn_created:
                conn.close()

    def clone(self, new_name: Optional[str] = None) -> "Portfolio":
        """Clone portfolio with new unique ID."""
        cloned_dict = self.to_dict()
        cloned_dict["portfolio_id"] = str(uuid.uuid4())
        cloned_dict["name"] = new_name or f"{self.name}_Clone"
        cloned_dict["created_at"] = datetime.now(timezone.utc).isoformat()
        return Portfolio.from_dict(cloned_dict)

    def increment_version(self, version_type: str = "patch") -> str:
        """Increment semantic version string."""
        parts = self.version.split(".")
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2]) if len(parts) > 2 else 0
        if version_type == "major":
            major += 1
            minor, patch = 0, 0
        elif version_type == "minor":
            minor += 1
            patch = 0
        else:
            patch += 1
        self.version = f"{major}.{minor}.{patch}"
        return self.version
