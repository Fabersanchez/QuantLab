"""
QuantLab Scientific Reproducibility Engine.

Enforces deterministic random seed initialization across standard libraries, NumPy,
and PyTorch. Manages reproducibility contracts, captures execution snapshots, and
validates auditability across experiments.
"""

from dataclasses import asdict, dataclass, field
import hashlib
import json
import random
from typing import Any, Dict, Optional
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from research.metadata import MetadataExtractor, SystemMetadata


@dataclass
class ReproducibilityContext:
    """Dataclass encapsulating reproducible experiment parameters and checksums."""

    random_seed: int
    config_hash: str
    dataset_checksum: str
    quantlab_version: str
    git_commit: str
    environment_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary representation."""
        return asdict(self)


class ReproducibilityManager:
    """Institutional Reproducibility Manager ensuring 100% deterministic scientific execution."""

    @staticmethod
    def set_seed(seed: int = 42) -> None:
        """Enforce global deterministic random seed across all python engines.

        Args:
            seed: Integer random seed.
        """
        random.seed(seed)
        np.random.seed(seed)
        if HAS_TORCH:
            try:
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except Exception:
                pass

    @staticmethod
    def compute_config_hash(config: Dict[str, Any]) -> str:
        """Compute SHA-256 hash digest of configuration dictionary.

        Args:
            config: Configuration dictionary.

        Returns:
            Hexadecimal SHA-256 hash string.
        """
        serialized = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()

    @staticmethod
    def compute_dataset_checksum(dataset_repr: Any) -> str:
        """Compute SHA-256 checksum digest of dataset or dataset metadata.

        Args:
            dataset_repr: Dataset object, pandas DataFrame, or dictionary representation.

        Returns:
            Hexadecimal SHA-256 checksum string.
        """
        if hasattr(dataset_repr, "to_csv"):
            data_bytes = dataset_repr.to_csv(index=False).encode("utf-8")
        elif isinstance(dataset_repr, dict):
            data_bytes = json.dumps(dataset_repr, sort_keys=True, default=str).encode("utf-8")
        elif isinstance(dataset_repr, bytes):
            data_bytes = dataset_repr
        else:
            data_bytes = str(dataset_repr).encode("utf-8")

        return hashlib.sha256(data_bytes).hexdigest()

    @classmethod
    def capture_context(
        cls,
        seed: int,
        config: Dict[str, Any],
        dataset_repr: Any,
        quantlab_version: str = "1.0.0",
        broker: str = "GenericBroker",
    ) -> ReproducibilityContext:
        """Capture complete reproducibility context snapshot.

        Args:
            seed: Random seed integer.
            config: Configuration payload.
            dataset_repr: Dataset instance or metadata.
            quantlab_version: Platform version string.
            broker: Broker identifier string.

        Returns:
            ReproducibilityContext instance.
        """
        cls.set_seed(seed)
        config_hash = cls.compute_config_hash(config)
        dataset_checksum = cls.compute_dataset_checksum(dataset_repr)
        metadata = MetadataExtractor.collect(
            broker=broker,
            quantlab_version=quantlab_version,
            random_seed=seed,
            config_hash=config_hash,
        )

        return ReproducibilityContext(
            random_seed=seed,
            config_hash=config_hash,
            dataset_checksum=dataset_checksum,
            quantlab_version=quantlab_version,
            git_commit=metadata.git_commit,
            environment_metadata=metadata.to_dict(),
        )

    @classmethod
    def verify_reproducibility(
        cls,
        context_a: ReproducibilityContext,
        context_b: ReproducibilityContext,
        strict_git: bool = False,
    ) -> Dict[str, Any]:
        """Verify reproducibility compatibility between two experiment contexts.

        Args:
            context_a: First experiment context.
            context_b: Second experiment context.
            strict_git: Whether Git commit hash mismatch triggers failure.

        Returns:
            Dict containing 'is_reproducible' boolean and breakdown of match checks.
        """
        seed_match = context_a.random_seed == context_b.random_seed
        config_match = context_a.config_hash == context_b.config_hash
        dataset_match = context_a.dataset_checksum == context_b.dataset_checksum
        version_match = context_a.quantlab_version == context_b.quantlab_version
        git_match = context_a.git_commit == context_b.git_commit

        is_reproducible = seed_match and config_match and dataset_match and version_match
        if strict_git:
            is_reproducible = is_reproducible and git_match

        return {
            "is_reproducible": is_reproducible,
            "seed_match": seed_match,
            "config_match": config_match,
            "dataset_match": dataset_match,
            "version_match": version_match,
            "git_match": git_match,
        }
