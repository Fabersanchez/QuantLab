"""
QuantLab High-Performance Multi-Attribute Search & Catalog Index Engine.

Provides high-performance multi-attribute search indexing across datasets, experiments,
models, research lines, artifacts, projects, and workspaces.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set
from studio.logging.studio_logger import get_studio_logger

logger = get_studio_logger("CatalogEngine")


@dataclass
class CatalogEntry:
    """Dataclass holding catalog index entry metadata."""

    entry_id: str
    entity_type: str  # 'DATASET', 'EXPERIMENT', 'MODEL', 'RESEARCH_LINE', 'ARTIFACT', 'PROJECT'
    name: str
    symbol: str = ""
    market: str = ""
    author: str = ""
    status: str = "ACTIVE"
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list) if 'field' in globals() else None


class CatalogEngine:
    """Institutional High-Performance Multi-Attribute Search Index Engine."""

    def __init__(self) -> None:
        self._entries: Dict[str, Dict[str, Any]] = {}
        self._tag_index: Dict[str, Set[str]] = {}  # tag -> set of entry_ids
        self._type_index: Dict[str, Set[str]] = {}  # entity_type -> set of entry_ids

    def index_entity(
        self,
        entry_id: str,
        entity_type: str,
        name: str,
        symbol: str = "",
        market: str = "",
        author: str = "QuantResearcher",
        status: str = "ACTIVE",
        version: str = "1.0.0",
        tags: Optional[List[str]] = None,
    ) -> None:
        """Index entity metadata into high-performance search engine."""
        entry = {
            "entry_id": entry_id,
            "entity_type": entity_type.upper(),
            "name": name,
            "symbol": symbol.upper(),
            "market": market.upper(),
            "author": author,
            "status": status.upper(),
            "version": version,
            "tags": tags or [],
        }
        self._entries[entry_id] = entry

        # Update indexes
        if entity_type.upper() not in self._type_index:
            self._type_index[entity_type.upper()] = set()
        self._type_index[entity_type.upper()].add(entry_id)

        for t in tags or []:
            t_low = t.lower()
            if t_low not in self._tag_index:
                self._tag_index[t_low] = set()
            self._tag_index[t_low].add(entry_id)

        logger.debug(f"Indexed {entity_type} '{name}' (ID={entry_id})")

    def search(
        self,
        query: str = "",
        entity_type: Optional[str] = None,
        tag: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Multi-attribute search query.

        Returns:
            List of matching catalog entry dictionaries.
        """
        candidate_ids: Optional[Set[str]] = None

        if entity_type:
            candidate_ids = set(self._type_index.get(entity_type.upper(), set()))

        if tag:
            tagged_ids = set(self._tag_index.get(tag.lower(), set()))
            candidate_ids = tagged_ids if candidate_ids is None else candidate_ids.intersection(tagged_ids)

        if candidate_ids is None:
            candidate_ids = set(self._entries.keys())

        results: List[Dict[str, Any]] = []
        q_low = query.lower()

        for eid in candidate_ids:
            e = self._entries[eid]
            if symbol and e["symbol"] != symbol.upper():
                continue
            if q_low and (q_low not in e["name"].lower() and q_low not in e["author"].lower()):
                continue
            results.append(e)

        return results
