"""
QuantLab Multi-Asset Data Merger.

Merges multiple asset DataFrames with synchronized time-series joins.
"""

from typing import Any, Dict, List, Optional
import pandas as pd


class DataMerger:
    """Institutional Multi-Asset Data Merger."""

    @staticmethod
    def merge_asset_close_prices(df_dict: Dict[str, pd.DataFrame], join_type: str = "outer") -> pd.DataFrame:
        """Merge close price columns from a dictionary of asset DataFrames.

        Args:
            df_dict: Dictionary mapping asset symbol to asset DataFrame.
            join_type: One of 'outer', 'inner'.

        Returns:
            Merged DataFrame indexed by datetime with asset symbol columns.
        """
        series_dict: Dict[str, pd.Series] = {}
        for sym, df in df_dict.items():
            if not df.empty:
                cols_lower = {str(c).lower(): c for c in df.columns}
                if "close" in cols_lower:
                    series_dict[sym] = df[cols_lower["close"]]

        if not series_dict:
            return pd.DataFrame()

        merged = pd.DataFrame(series_dict)
        if join_type == "inner":
            merged = merged.dropna()
        else:
            merged = merged.ffill().bfill()

        return merged
