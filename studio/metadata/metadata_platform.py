"""
QuantLab Enriched Metadata Platform Engine.

Enriches datasets, models, experiments, and research lines with owner, origin, dependencies,
relations, tags, description, status, version, digital signature, compatibility, license, and audit logs.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class EnrichedMetadata:
    """Dataclass holding enriched entity metadata."""

    entity_id: str
    entity_type: str  # 'DATASET', 'MODEL', 'EXPERIMENT', 'RESEARCH_LINE', 'ARTIFACT'
    owner: str = "QuantLabResearcher"
    origin: str = "Internal"
    dependencies: List[str] = field(default_factory=list)
    relations: Dict[str, str] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    description: str = ""
    status: str = "ACTIVE"
    version: str = "1.0.0"
    digital_signature: str = ""
    compatibility: str = "Python 3.11 / PyTorch / sklearn"
    license: str = "Institutional Proprietary"
    audit_history: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary representation."""
        return asdict(self)


class MetadataPlatform:
    """Institutional Enriched Metadata Platform Engine."""

    def __init__(self) -> None:
        self._registry: Dict[str, EnrichedMetadata] = {}

    def attach_metadata(
        self,
        entity_id: str,
        entity_type: str,
        owner: str = "QuantLabResearcher",
        tags: Optional[List[str]] = None,
        description: str = "",
    ) -> EnrichedMetadata:
        """Attach enriched metadata payload to target entity."""
        meta = EnrichedMetadata(
            entity_id=entity_id,
            entity_type=entity_type,
            owner=owner,
            tags=tags or [],
            description=description,
        )
        meta.audit_history.append(f"Created metadata at {datetime.now(timezone.utc).isoformat()}")
        self._registry[entity_id] = meta
        return meta

    def get_metadata(self, entity_id: str) -> Optional[EnrichedMetadata]:
        """Fetch metadata for entity ID."""
        return self._registry.get(entity_id)

    def search_by_tag(self, tag: str) -> List[EnrichedMetadata]:
        """Search metadata records by tag match."""
        t_low = tag.lower()
        return [m for m in self._registry.values() if any(t_low in t.lower() for t in m.tags)]
