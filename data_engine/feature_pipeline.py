"""
QuantLab Feature Transformation Pipeline Engine.

Registers transformation steps, persists fitted scaling parameters, and executes reproducible fit_transform() pipelines.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import pandas as pd


class FeaturePipeline:
    """Institutional Feature Transformation Pipeline Engine."""

    def __init__(self) -> None:
        self.steps: List[Tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]] = []
        self._fitted_params: Dict[str, Any] = {}

    def add_step(self, step_name: str, transform_fn: Callable[[pd.DataFrame], pd.DataFrame]) -> "FeaturePipeline":
        """Add feature transformation step to pipeline."""
        self.steps.append((step_name, transform_fn))
        return self

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute all pipeline transformation steps sequentially."""
        if df.empty:
            return df

        df_out = df.copy()
        for name, fn in self.steps:
            df_out = fn(df_out)
        return df_out
