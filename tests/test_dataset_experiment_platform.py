"""
QuantLab Dataset, Experiment & Research Management Platform Test Suite.

Validates all Phase 19.3 components:
DatasetRecord, DatasetCenter, DataQualityEngine, QualityAlert, ExperimentRecord, ExperimentCenter,
ResearchLine, ResearchCenter, ArtifactManager, MetadataPlatform, EnrichedMetadata,
DataLineageEngine, PreviewEngine, DatasetPreviewSummary, and CatalogEngine.
"""

import os
import shutil
import tempfile
import unittest
import pandas as pd

from studio import (
    ArtifactManager,
    CatalogEngine,
    DataLineageEngine,
    DataQualityEngine,
    DatasetCenter,
    DatasetPreviewSummary,
    DatasetRecord,
    EnrichedMetadata,
    ExperimentCenter,
    ExperimentRecord,
    MetadataPlatform,
    PreviewEngine,
    QualityAlert,
    ResearchCenter,
    ResearchLine,
)


class TestQuantLabDatasetExperimentPlatform(unittest.TestCase):
    """Comprehensive Test Case for QuantLab Dataset, Experiment & Research Management Platform."""

    def setUp(self) -> None:
        """Set up temporary directory and test data file."""
        self.temp_dir = tempfile.mkdtemp(prefix="quantlab_d_e_test_")
        self.csv_path = os.path.join(self.temp_dir, "preview_test.csv")

        df = pd.DataFrame(
            {
                "timestamp": ["2025-01-01 10:00:00", "2025-01-01 11:00:00", "2025-01-01 12:00:00"],
                "open": [1.1000, 1.1010, 1.1020],
                "high": [1.1050, 1.1060, 1.1070],
                "low": [1.0990, 1.1000, 1.1010],
                "close": [1.1010, 1.1020, 1.1030],
                "volume": [1000, 1500, 1200],
            }
        )
        df.to_csv(self.csv_path, index=False)

    def tearDown(self) -> None:
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_dataset_center_and_deduplication(self) -> None:
        """Test DatasetCenter registration, metadata tracking, and deduplication."""
        dc = DatasetCenter()
        ds1 = dc.register_dataset("EURUSD_1H", filepath=self.csv_path, row_count=3)
        self.assertEqual(ds1.name, "EURUSD_1H")

        # Re-register exact same dataset to verify deduplication
        ds2 = dc.register_dataset("EURUSD_1H", filepath=self.csv_path, row_count=3)
        self.assertEqual(ds1.dataset_id, ds2.dataset_id)
        self.assertEqual(len(dc.list_datasets()), 1)

    def test_data_quality_engine(self) -> None:
        """Test DataQualityEngine telemetry evaluation and alerts."""
        df = pd.read_csv(self.csv_path)
        report, alerts = DataQualityEngine.evaluate_quality(df)

        self.assertTrue(report.is_valid)
        self.assertGreater(report.quality_score, 80.0)

    def test_experiment_center_and_reproducibility(self) -> None:
        """Test ExperimentCenter reproducible experiment run registration."""
        ec = ExperimentCenter()
        exp = ec.register_experiment(
            "XGBoost_HyperOpt", category="Optimization", parameters={"n_estimators": 100}, metrics={"sharpe": 2.1}
        )
        self.assertEqual(exp.name, "XGBoost_HyperOpt")
        self.assertEqual(exp.metrics["sharpe"], 2.1)

    def test_research_center(self) -> None:
        """Test ResearchCenter research line creation and experiment association."""
        rc = ResearchCenter()
        rl = rc.create_research_line("StatArb_Pairs", hypothesis="Trading cointegrated EURUSD vs GBPUSD")
        self.assertEqual(rl.title, "StatArb_Pairs")

        self.assertTrue(rc.add_experiment_to_line(rl.line_id, "EXP-101"))
        self.assertIn("EXP-101", rl.experiments)

    def test_artifact_manager_and_metadata_platform(self) -> None:
        """Test ArtifactManager registration and MetadataPlatform tagging."""
        am = ArtifactManager()
        art = am.register_artifact("xgboost_weights.pkl", artifact_type="MODEL_WEIGHTS")
        self.assertEqual(art.artifact_type, "MODEL_WEIGHTS")

        mp = MetadataPlatform()
        meta = mp.attach_metadata(art.artifact_id, "ARTIFACT", owner="ChiefQuant", tags=["production", "v1"])
        self.assertEqual(meta.owner, "ChiefQuant")

        search_meta = mp.search_by_tag("production")
        self.assertEqual(len(search_meta), 1)

    def test_data_lineage_and_preview_engine(self) -> None:
        """Test DataLineageEngine Mermaid DAG rendering and PreviewEngine incremental reading."""
        lineage = DataLineageEngine()
        lineage.record_provenance("DS-1", "MOD-1", "EXP-1")

        mermaid = lineage.render_mermaid_dag()
        self.assertIn("graph TD", mermaid)

        preview = PreviewEngine.generate_preview(self.csv_path, preview_rows=2)
        self.assertIsNotNone(preview)
        self.assertEqual(len(preview.column_names), 6)

    def test_catalog_engine(self) -> None:
        """Test CatalogEngine multi-attribute search index query."""
        catalog = CatalogEngine()
        catalog.index_entity("DS-1", "DATASET", "EURUSD_1H", symbol="EURUSD", market="FOREX", tags=["hft"])
        catalog.index_entity("MOD-1", "MODEL", "DeepAR_LSTM", symbol="EURUSD", market="FOREX", tags=["dl"])

        results = catalog.search(query="EURUSD", entity_type="DATASET")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "EURUSD_1H")


if __name__ == "__main__":
    unittest.main()
