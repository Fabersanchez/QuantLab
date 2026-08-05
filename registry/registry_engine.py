"""
QuantLab Master Governance Registry Engine.

Centralizes and coordinates Model Registry, Experiment Registry, Strategy Registry, Dataset Registry,
Feature Registry, Artifact Registry, Version Manager, Lineage DAG Traceability, Approval Workflows,
Storage, and Reporting.
"""

from typing import Any, Dict, List, Optional

from registry.approval import ApprovalState
from registry.artifact_registry import ArtifactRecord, ArtifactRegistry
from registry.cache import RegistryCache
from registry.dataset_registry import DatasetRecord, DatasetRegistry
from registry.experiment_registry import ExperimentRecord, ExperimentRegistry
from registry.exporter import RegistryExporter
from registry.feature_registry import FeatureRecord, FeatureRegistry
from registry.lineage import LineageGraph, LineageNodeType
from registry.logger import get_registry_logger
from registry.model_registry import ModelRecord, ModelRegistry
from registry.reports import RegistryReportEngine
from registry.storage import RegistryStorage
from registry.strategy_registry import StrategyRecord, StrategyRegistry

logger = get_registry_logger("RegistryEngine")


class RegistryEngine:
    """Master Institutional Governance Registry Engine for QuantLab."""

    def __init__(self, db_path: str = "quantlab_registry.db") -> None:
        """Initialize RegistryEngine.

        Args:
            db_path: Path to SQLite database file.
        """
        self.model_registry = ModelRegistry()
        self.experiment_registry = ExperimentRegistry()
        self.strategy_registry = StrategyRegistry()
        self.dataset_registry = DatasetRegistry()
        self.feature_registry = FeatureRegistry()
        self.artifact_registry = ArtifactRegistry()
        self.lineage = LineageGraph()
        self.storage = RegistryStorage(db_path=db_path)
        self.cache = RegistryCache()

    # --- Model Registration ---

    def register_model(
        self,
        name: str,
        framework: str = "scikit-learn",
        model_type: str = "MachineLearning",
        dataset_id: str = "",
        features: Optional[List[str]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        architecture: str = "StandardClassifier",
        weights_path: str = "",
        scores: Optional[Dict[str, float]] = None,
        author: str = "QuantLabResearcher",
    ) -> ModelRecord:
        """Register model and update lineage graph."""
        record = self.model_registry.register_model(
            name=name,
            framework=framework,
            model_type=model_type,
            dataset_id=dataset_id,
            features=features,
            hyperparameters=hyperparameters,
            architecture=architecture,
            weights_path=weights_path,
            scores=scores,
            author=author,
        )
        # Lineage update
        node = self.lineage.add_node(
            node_id=record.model_id,
            name=record.name,
            node_type=LineageNodeType.MODEL,
            version=record.version,
        )
        if dataset_id:
            self.lineage.add_edge(dataset_id, record.model_id)

        self.storage.save_record("models", record.model_id, record.name, record.version, record.state.value, record.to_dict())
        self.cache.put(record.model_id, record.to_dict())
        logger.log_registration("Model", record.model_id, record.name, record.version)
        return record

    # --- Experiment Registration ---

    def register_experiment(
        self,
        name: str,
        category: str = "Backtest",
        strategy_id: str = "",
        model_id: str = "",
        parameters: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        duration_sec: float = 0.0,
        status: str = "SUCCESS",
        notes: str = "",
    ) -> ExperimentRecord:
        """Register experiment run and update lineage graph."""
        record = self.experiment_registry.register_experiment(
            name=name,
            category=category,
            strategy_id=strategy_id,
            model_id=model_id,
            parameters=parameters,
            metrics=metrics,
            duration_sec=duration_sec,
            status=status,
            notes=notes,
        )
        # Lineage update
        self.lineage.add_node(
            node_id=record.experiment_id,
            name=record.name,
            node_type=LineageNodeType.EXPERIMENT,
        )
        if model_id:
            self.lineage.add_edge(model_id, record.experiment_id)
        if strategy_id:
            self.lineage.add_edge(strategy_id, record.experiment_id)

        self.storage.save_record("experiments", record.experiment_id, record.name, "1.0.0", record.status, record.to_dict())
        logger.log_registration("Experiment", record.experiment_id, record.name, "1.0.0")
        return record

    # --- Governance Approval Transitions ---

    def update_model_approval_state(
        self, model_id: str, new_state: ApprovalState, approver: str = "SystemAdmin", comments: str = ""
    ) -> bool:
        """Update model governance approval state."""
        success = self.model_registry.update_approval_state(model_id, new_state, approver=approver, comments=comments)
        if success:
            record = self.model_registry.get_model(model_id)
            if record:
                self.storage.save_record("models", record.model_id, record.name, record.version, record.state.value, record.to_dict())
                self.cache.put(record.model_id, record.to_dict())
                logger.log_approval(model_id, "PREV", new_state.value, approver)
        return success

    # --- Reports & Exporters ---

    def generate_audit_report(self) -> str:
        """Generate comprehensive Markdown governance audit report string."""
        models_data = [m.to_dict() for m in self.model_registry.list_models()]
        exp_data = [e.to_dict() for e in self.experiment_registry.list_experiments()]
        return RegistryReportEngine.generate_audit_report(models_data, exp_data, self.lineage)

    def export_registry(self, category: str = "models", filepath: str = "registry.json", export_format: str = "json") -> str:
        """Export specified registry category to file."""
        if category == "experiments":
            records = [e.to_dict() for e in self.experiment_registry.list_experiments()]
        elif category == "strategies":
            records = [s.to_dict() for s in self.strategy_registry.list_strategies()]
        elif category == "datasets":
            records = [d.to_dict() for d in self.dataset_registry.list_datasets()]
        else:
            records = [m.to_dict() for m in self.model_registry.list_models()]

        fmt = export_format.lower()
        if fmt == "csv":
            return RegistryExporter.to_csv(records, filepath)
        elif fmt in ("excel", "xlsx"):
            return RegistryExporter.to_excel(records, filepath)
        elif fmt in ("parquet",):
            return RegistryExporter.to_parquet(records, filepath)
        elif fmt in ("markdown", "md"):
            return RegistryExporter.to_markdown(records, filepath)
        elif fmt in ("pdf",):
            return RegistryExporter.to_pdf(records, filepath)
        else:
            return RegistryExporter.to_json(records, filepath)
