"""
QuantLab Feature Metadata.

Defines the metadata record for registering, tracking, and auditing
predictive features within QuantLab.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class FeatureMetadata:
    """Metadata attribute record for a predictive feature.

    Attributes:
        name: Unique string identifier for the feature.
        version: Feature version identifier.
        author: Creator or maintainer name.
        description: Functional description of feature logic.
        category: Feature category (Price, Volume, Volatility, Time, Statistical, etc.).
        source_dataset: Origin dataset identifier.
        dependencies: List of input column names required to compute feature.
        data_type: Data type string (e.g., 'float64', 'int64', 'category').
        status: Feature operational state ('active', 'deprecated', 'experimental').
        created_at: Creation timestamp.
        extra_attributes: Custom key-value attributes.
    """

    name: str
    version: str = "1.0.0"
    author: str = "QuantLab Engineering"
    description: str = ""
    category: str = "General"
    source_dataset: str = "RawMarketData"
    dependencies: List[str] = field(default_factory=list)
    data_type: str = "float64"
    status: str = "active"
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    extra_attributes: Dict[str, Any] = field(default_factory=dict)
