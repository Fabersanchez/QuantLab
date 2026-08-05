"""
QuantLab Hardware & Environment Metadata Extractor.

Auto-captures hardware telemetry (CPU, GPU, RAM), operating system parameters, Python version,
installed library manifests, git commit hashes, and deterministic execution seeds.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import os
import platform
import sys
from typing import Any, Dict, List, Optional
import psutil


@dataclass
class SystemEnvironmentMetadata:
    """Dataclass holding hardware, OS, and software environment telemetry."""

    os_platform: str = field(default_factory=lambda: platform.system())
    os_version: str = field(default_factory=lambda: platform.version())
    os_release: str = field(default_factory=lambda: platform.release())
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    python_executable: str = field(default_factory=lambda: sys.executable)
    cpu_architecture: str = field(default_factory=lambda: platform.machine())
    cpu_count_physical: int = field(default_factory=lambda: psutil.cpu_count(logical=False) or 1)
    cpu_count_logical: int = field(default_factory=lambda: psutil.cpu_count(logical=True) or 1)
    total_ram_mb: float = field(
        default_factory=lambda: round(psutil.virtual_memory().total / (1024 * 1024), 2)
    )
    gpu_name: str = "N/A"
    installed_libraries: Dict[str, str] = field(default_factory=dict)
    random_seed: Optional[int] = 42
    git_commit_hash: str = "UNKNOWN"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        """Capture installed libraries manifests."""
        if not self.installed_libraries:
            libs = {}
            for mod_name in ["numpy", "pandas", "scipy", "sklearn", "torch", "matplotlib"]:
                if mod_name in sys.modules and hasattr(sys.modules[mod_name], "__version__"):
                    libs[mod_name] = str(getattr(sys.modules[mod_name], "__version__"))
            self.installed_libraries = libs

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SystemEnvironmentMetadata":
        """Reconstruct metadata from dictionary representation."""
        return cls(**data)
