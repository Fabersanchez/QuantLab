"""
QuantLab Data Lineage Provenance Engine.

Provides complete DAG provenance tracking answering dataset origin, transformation, model version,
experiment parameters, and researcher identity with Mermaid DAG rendering.
"""

from typing import Any, Dict, List, Optional
from registry.lineage import LineageGraph, LineageNode, LineageNodeType
from studio.logging.studio_logger import get_studio_logger

logger = get_studio_logger("DataLineageEngine")


class DataLineageEngine:
    """Institutional Data Lineage Provenance Engine."""

    def __init__(self, lineage_graph: Optional[LineageGraph] = None) -> None:
        self.graph = lineage_graph or LineageGraph()

    def record_provenance(
        self,
        dataset_id: str,
        model_id: str,
        experiment_id: str,
        dataset_name: str = "Dataset",
        model_name: str = "Model",
        experiment_name: str = "Experiment",
    ) -> None:
        """Record full dataset -> model -> experiment DAG provenance chain."""
        self.graph.add_node(dataset_id, dataset_name, LineageNodeType.DATASET)
        self.graph.add_node(model_id, model_name, LineageNodeType.MODEL)
        self.graph.add_node(experiment_id, experiment_name, LineageNodeType.EXPERIMENT)

        self.graph.add_edge(dataset_id, model_id)
        self.graph.add_edge(model_id, experiment_id)
        logger.info(f"Recorded provenance chain: '{dataset_name}' -> '{model_name}' -> '{experiment_name}'")

    def get_ancestors(self, node_id: str) -> List[LineageNode]:
        """Fetch upstream origin ancestors for target node ID."""
        return self.graph.get_ancestors(node_id)

    def render_mermaid_dag(self) -> str:
        """Render Mermaid DAG flowchart string for provenance visualization."""
        return self.graph.to_mermaid()
