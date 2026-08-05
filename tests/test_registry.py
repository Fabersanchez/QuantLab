"""
QuantLab Master Governance Registry Engine Test Suite.

Validates all 21 components of the Registry Engine:
SystemEnvironmentMetadata, IntegrityChecker, DigitalSignature, ApprovalWorkflow, VersionManager,
LineageGraph, ModelRegistry, ExperimentRegistry, StrategyRegistry, DatasetRegistry, FeatureRegistry,
ArtifactRegistry, RegistryComparator, DeploymentPackage, RegistryStorage, RegistryCache,
RegistryExporter, RegistryReportEngine, and RegistryEngine.
"""

import os
import shutil
import tempfile
import unittest

from registry import (
    ApprovalState,
    ApprovalWorkflow,
    ArtifactRecord,
    ArtifactRegistry,
    DatasetRecord,
    DatasetRegistry,
    DeploymentPackage,
    DigitalSignature,
    ExperimentRecord,
    ExperimentRegistry,
    FeatureRecord,
    FeatureRegistry,
    IntegrityChecker,
    LineageGraph,
    LineageNodeType,
    ModelRecord,
    ModelRegistry,
    RegistryCache,
    RegistryComparator,
    RegistryEngine,
    RegistryExporter,
    RegistryReportEngine,
    RegistryStorage,
    StrategyRecord,
    StrategyRegistry,
    SystemEnvironmentMetadata,
    VersionManager,
)


class TestQuantLabRegistryEngine(unittest.TestCase):
    """Comprehensive Test Case for QuantLab Model & Experiment Registry Engine."""

    def setUp(self) -> None:
        """Set up temporary directory and test database file."""
        self.temp_dir = tempfile.mkdtemp(prefix="quantlab_reg_test_")
        self.db_path = os.path.join(self.temp_dir, "test_registry.db")

    def tearDown(self) -> None:
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_metadata_and_integrity_and_signatures(self) -> None:
        """Test SystemEnvironmentMetadata, IntegrityChecker, and DigitalSignature."""
        meta = SystemEnvironmentMetadata()
        self.assertTrue(len(meta.os_platform) > 0)
        self.assertTrue(meta.total_ram_mb > 0)

        data = "QuantLab_Model_Weights_V1"
        h = IntegrityChecker.compute_sha256(data)
        self.assertTrue(len(h) == 64)
        self.assertTrue(IntegrityChecker.verify_integrity(data, h))

        sig = DigitalSignature.generate_signature("MODEL-001", h, "QuantResearcher")
        self.assertTrue(DigitalSignature.verify_signature(sig, "MODEL-001", h, "QuantResearcher"))

    def test_approval_workflow_and_version_manager(self) -> None:
        """Test ApprovalWorkflow state transitions and VersionManager snapshots/rollback."""
        wf = ApprovalWorkflow(initial_state=ApprovalState.DRAFT)
        self.assertTrue(wf.transition_to(ApprovalState.TRAINING))
        self.assertTrue(wf.transition_to(ApprovalState.TESTING))
        self.assertTrue(wf.transition_to(ApprovalState.VALIDATED))
        self.assertTrue(wf.transition_to(ApprovalState.APPROVED))
        self.assertEqual(wf.current_state, ApprovalState.APPROVED)

        vm = VersionManager(initial_version="1.0.0")
        vm.create_snapshot("M-1", {"weights": [0.1, 0.2]})
        v2 = vm.bump_version("minor")
        self.assertEqual(v2, "1.1.0")
        vm.create_snapshot("M-1", {"weights": [0.5, 0.6]})

        rolled = vm.rollback_to("1.0.0")
        self.assertIsNotNone(rolled)
        self.assertEqual(rolled["weights"], [0.1, 0.2])

    def test_lineage_graph(self) -> None:
        """Test LineageGraph DAG dependency tracking and Mermaid diagram rendering."""
        graph = LineageGraph()
        n_ds = graph.add_node("DS-1", "EURUSD_1H", LineageNodeType.DATASET)
        n_mod = graph.add_node("MOD-1", "RandomForest", LineageNodeType.MODEL)
        n_exp = graph.add_node("EXP-1", "BacktestRun", LineageNodeType.EXPERIMENT)

        graph.add_edge("DS-1", "MOD-1")
        graph.add_edge("MOD-1", "EXP-1")

        ancestors = graph.get_ancestors("EXP-1")
        self.assertEqual(len(ancestors), 2)
        mermaid_str = graph.to_mermaid()
        self.assertIn("graph TD", mermaid_str)

    def test_sub_registries(self) -> None:
        """Test ModelRegistry, ExperimentRegistry, StrategyRegistry, DatasetRegistry, FeatureRegistry, ArtifactRegistry."""
        # Model Registry
        mr = ModelRegistry()
        m_rec = mr.register_model("XGBoost_Strategy", framework="scikit-learn", scores={"sharpe": 1.85})
        self.assertEqual(m_rec.name, "XGBoost_Strategy")
        self.assertTrue(mr.update_approval_state(m_rec.model_id, ApprovalState.TRAINING))

        # Experiment Registry
        er = ExperimentRegistry()
        e_rec = er.register_experiment("MonteCarlo_Robustness", category="MonteCarlo", duration_sec=5.2)
        self.assertEqual(e_rec.category, "MonteCarlo")

        # Strategy Registry
        sr = StrategyRegistry()
        s_rec = sr.register_strategy("EMA_Cross", indicators=["EMA_10", "EMA_50"])
        self.assertEqual(len(s_rec.indicators), 2)

        # Dataset Registry
        dr = DatasetRegistry()
        d_rec = dr.register_dataset("EURUSD_2025", market="FOREX", n_rows=10000)
        self.assertEqual(d_rec.n_rows, 10000)

        # Feature Registry
        fr = FeatureRegistry()
        f_rec = fr.register_feature("feature_rsi_14", importance_score=0.45)
        self.assertEqual(f_rec.importance_score, 0.45)

        # Artifact Registry
        ar = ArtifactRegistry()
        a_rec = ar.register_artifact("model_weights.pkl", artifact_type="MODEL_WEIGHTS")
        self.assertEqual(a_rec.artifact_type, "MODEL_WEIGHTS")

    def test_comparator_and_deployment(self) -> None:
        """Test RegistryComparator matrices and DeploymentPackage manifest creation."""
        models = [
            {"model_id": "M1", "name": "Model1", "version": "1.0.0", "framework": "PyTorch", "state": "APPROVED", "scores": {"sharpe": 2.1}},
            {"model_id": "M2", "name": "Model2", "version": "1.0.0", "framework": "sklearn", "state": "VALIDATED", "scores": {"sharpe": 1.7}},
        ]
        df_comp = RegistryComparator.compare_models(models)
        self.assertEqual(len(df_comp), 2)

        dp = DeploymentPackage(package_id="DEP-1", model_id="M1", model_name="Model1", version="1.0.0", state="APPROVED", artifacts_included=["weights.pkl"])
        manifest = dp.build_bundle(self.temp_dir)
        self.assertTrue(os.path.exists(manifest))

    def test_storage_and_cache(self) -> None:
        """Test RegistryStorage SQLite database persistence and RegistryCache."""
        storage = RegistryStorage(db_path=self.db_path)
        storage.save_record("models", "M-100", "TestModel", "1.0.0", "APPROVED", {"name": "TestModel"})

        rec = storage.load_record("models", "M-100")
        self.assertIsNotNone(rec)
        self.assertEqual(rec["name"], "TestModel")

        cache = RegistryCache()
        cache.put("M-100", {"name": "TestModel"})
        self.assertEqual(cache.get("M-100")["name"], "TestModel")

    def test_exporter_and_report_engine(self) -> None:
        """Test RegistryExporter formats and RegistryReportEngine markdown reports."""
        records = [{"id": "1", "name": "ModelA"}, {"id": "2", "name": "ModelB"}]

        csv_p = RegistryExporter.to_csv(records, os.path.join(self.temp_dir, "reg.csv"))
        self.assertTrue(os.path.exists(csv_p))

        excel_p = RegistryExporter.to_excel(records, os.path.join(self.temp_dir, "reg.xlsx"))
        self.assertTrue(os.path.exists(excel_p))

        json_p = RegistryExporter.to_json(records, os.path.join(self.temp_dir, "reg.json"))
        self.assertTrue(os.path.exists(json_p))

        parq_p = RegistryExporter.to_parquet(records, os.path.join(self.temp_dir, "reg.parquet"))
        self.assertTrue(os.path.exists(parq_p))

        md_p = RegistryExporter.to_markdown(records, os.path.join(self.temp_dir, "reg.md"))
        self.assertTrue(os.path.exists(md_p))

        report_md = RegistryReportEngine.generate_audit_report(records, [])
        self.assertIn("QuantLab Governance & Lineage Audit Report", report_md)

    def test_master_registry_engine(self) -> None:
        """Test master RegistryEngine end-to-end registration, approval, and exporting."""
        engine = RegistryEngine(db_path=self.db_path)
        m = engine.register_model("MasterModel", framework="PyTorch", model_type="DeepLearning")
        self.assertIsNotNone(m)

        e = engine.register_experiment("MasterBacktest", category="Backtest", model_id=m.model_id)
        self.assertIsNotNone(e)

        self.assertTrue(engine.update_model_approval_state(m.model_id, ApprovalState.TRAINING))

        report = engine.generate_audit_report()
        self.assertIn("MasterModel", report)

        exp_json = engine.export_registry(category="models", filepath=os.path.join(self.temp_dir, "master_models.json"))
        self.assertTrue(os.path.exists(exp_json))


if __name__ == "__main__":
    unittest.main()
