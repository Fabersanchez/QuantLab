"""
QuantLab Governance Registry Comparator Engine.

Provides side-by-side diff matrices and comparative scoring between versions, models,
experiments, datasets, and strategies.
"""

from typing import Any, Dict, List, Optional
import pandas as pd


class RegistryComparator:
    """Institutional Governance Registry Comparator Engine."""

    @staticmethod
    def compare_models(model_records: List[Dict[str, Any]]) -> pd.DataFrame:
        """Compare multiple model records side-by-side.

        Returns:
            DataFrame matrix comparing models.
        """
        rows = []
        for r in model_records:
            rows.append(
                {
                    "model_id": r.get("model_id"),
                    "name": r.get("name"),
                    "version": r.get("version"),
                    "framework": r.get("framework"),
                    "state": r.get("state"),
                    "scores": str(r.get("scores")),
                    "checksum": r.get("payload_hash", "")[:12],
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def compare_experiments(experiment_records: List[Dict[str, Any]]) -> pd.DataFrame:
        """Compare multiple research experiment runs side-by-side.

        Returns:
            DataFrame matrix comparing experiments.
        """
        rows = []
        for r in experiment_records:
            rows.append(
                {
                    "experiment_id": r.get("experiment_id"),
                    "name": r.get("name"),
                    "category": r.get("category"),
                    "duration_sec": r.get("duration_sec"),
                    "status": r.get("status"),
                    "metrics": str(r.get("metrics")),
                }
            )
        return pd.DataFrame(rows)
