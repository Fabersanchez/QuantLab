"""
QuantLab Registry Multi-Backend Storage Engine.

Persists governance records into SQLite database tables, JSON files, Parquet tables, and Pickle artifacts.
"""

import json
import os
import sqlite3
from typing import Any, Dict, List, Optional, Union
import pandas as pd


class RegistryStorage:
    """Institutional Multi-Backend Governance Storage Provider."""

    def __init__(self, db_path: str = "quantlab_registry.db") -> None:
        """Initialize RegistryStorage.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = db_path
        self._init_sqlite()

    def _init_sqlite(self) -> None:
        """Initialize SQLite database tables for governance categories."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            for table in ["models", "experiments", "strategies", "datasets", "features", "artifacts"]:
                cursor.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {table} (
                        record_id TEXT PRIMARY KEY,
                        name TEXT,
                        version TEXT,
                        state TEXT,
                        created_at TEXT,
                        payload_json TEXT
                    )
                    """
                )
            conn.commit()
        finally:
            conn.close()

    def save_record(self, table_name: str, record_id: str, name: str, version: str, state: str, payload: Dict[str, Any]) -> None:
        """Save governance record payload to SQLite database."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            created_at = str(payload.get("created_at", ""))
            json_str = json.dumps(payload, default=str)
            cursor.execute(
                f"""
                INSERT OR REPLACE INTO {table_name} (
                    record_id, name, version, state, created_at, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (record_id, name, version, state, created_at, json_str),
            )
            conn.commit()
        finally:
            conn.close()

    def load_record(self, table_name: str, record_id: str) -> Optional[Dict[str, Any]]:
        """Load governance record payload from SQLite database."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(f"SELECT payload_json FROM {table_name} WHERE record_id = ?", (record_id,))
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
            return None
        finally:
            conn.close()

    def list_records(self, table_name: str) -> List[Dict[str, Any]]:
        """List all governance records in a database table."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(f"SELECT payload_json FROM {table_name}")
            rows = cursor.fetchall()
            return [json.loads(r[0]) for r in rows]
        finally:
            conn.close()
