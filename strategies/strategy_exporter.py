"""
QuantLab Strategy Exporter.

Provides export adapters for serializing strategy specifications, parameters, and metadata
to JSON, YAML, Pickle, Sentinel, and ONNX metadata formats.
"""

import json
from pathlib import Path
from typing import Any, Dict, Union
from strategies.base_strategy import BaseStrategy


class StrategyExporter:
    """Strategy Exporter engine."""

    @staticmethod
    def to_dict(strategy: BaseStrategy) -> Dict[str, Any]:
        """Convert strategy metadata and active parameters to dictionary."""
        meta = strategy.metadata()
        return {
            "name": meta.name,
            "version": meta.version,
            "category": meta.category,
            "author": meta.author,
            "description": meta.description,
            "markets": meta.markets,
            "timeframes": meta.timeframes,
            "risk_profile": meta.risk_profile,
            "indicators_required": meta.indicators_required,
            "features_required": meta.features_required,
            "parameters": strategy.params,
        }

    @classmethod
    def to_json(cls, strategy: BaseStrategy, output_path: Union[str, Path]) -> str:
        """Export strategy configuration to JSON file."""
        data = cls.to_dict(strategy)
        json_str = json.dumps(data, indent=2, default=str)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json_str, encoding="utf-8")
        return json_str

    @classmethod
    def to_sentinel_spec(cls, strategy: BaseStrategy) -> Dict[str, Any]:
        """Format strategy configuration for Sentinel trading platform integration."""
        data = cls.to_dict(strategy)
        return {
            "sentinel_version": "1.0",
            "strategy_id": f"QUANTLAB_{data['name'].upper()}_V{data['version']}",
            "config": data,
        }
