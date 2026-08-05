"""
QuantLab Lineage Graph & Provenance Traceability Engine.

Builds a Directed Acyclic Graph (DAG) tracking relationships:
Dataset -> Features -> Optimization -> Model -> Experiment -> Strategy -> Artifacts.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class LineageNodeType(str, Enum):
    """Lineage node entity type categorization."""

    DATASET = "DATASET"
    FEATURE = "FEATURE"
    OPTIMIZATION = "OPTIMIZATION"
    MODEL = "MODEL"
    EXPERIMENT = "EXPERIMENT"
    STRATEGY = "STRATEGY"
    ARTIFACT = "ARTIFACT"


@dataclass
class LineageNode:
    """Dataclass representing a node in the governance lineage DAG."""

    node_id: str
    name: str
    node_type: LineageNodeType
    version: str = "1.0.0"
    metadata: Dict[str, Any] = field(default_factory=dict)


class LineageGraph:
    """Institutional Lineage DAG Provenance Engine."""

    def __init__(self) -> None:
        """Initialize LineageGraph."""
        self.nodes: Dict[str, LineageNode] = {}
        self.parents: Dict[str, Set[str]] = {}
        self.children: Dict[str, Set[str]] = {}

    def add_node(
        self,
        node_id: str,
        name: str,
        node_type: LineageNodeType,
        version: str = "1.0.0",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LineageNode:
        """Add node to lineage DAG."""
        node = LineageNode(
            node_id=node_id,
            name=name,
            node_type=node_type,
            version=version,
            metadata=metadata or {},
        )
        self.nodes[node_id] = node
        if node_id not in self.parents:
            self.parents[node_id] = set()
        if node_id not in self.children:
            self.children[node_id] = set()
        return node

    def add_edge(self, parent_id: str, child_id: str) -> None:
        """Add directed dependency edge from parent to child."""
        if parent_id in self.nodes and child_id in self.nodes:
            self.parents[child_id].add(parent_id)
            self.children[parent_id].add(child_id)

    def get_parents(self, node_id: str) -> List[LineageNode]:
        """Get direct parent nodes for a given node."""
        parent_ids = self.parents.get(node_id, set())
        return [self.nodes[pid] for pid in parent_ids if pid in self.nodes]

    def get_ancestors(self, node_id: str) -> List[LineageNode]:
        """Get all upstream ancestor nodes recursively."""
        visited: Set[str] = set()
        stack = list(self.parents.get(node_id, set()))

        while stack:
            curr = stack.pop()
            if curr not in visited:
                visited.add(curr)
                stack.extend(list(self.parents.get(curr, set())))

        return [self.nodes[aid] for aid in visited if aid in self.nodes]

    def to_mermaid(self) -> str:
        """Render Mermaid DAG flowchart diagram string."""
        lines = ["graph TD"]
        for parent_id, child_set in self.children.items():
            p_node = self.nodes.get(parent_id)
            p_label = f"{p_node.name} ({p_node.node_type.value})" if p_node else parent_id
            for child_id in child_set:
                c_node = self.nodes.get(child_id)
                c_label = f"{c_node.name} ({c_node.node_type.value})" if c_node else child_id
                lines.append(f'    "{p_label}" --> "{c_label}"')
        return "\n".join(lines)
