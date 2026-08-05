"""
QuantLab Hardware and Environment Metadata Engine.

Automatically extracts environment, operating system, hardware resources (CPU, RAM, GPU),
python runtime, git VCS metadata, broker details, and system configurations.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import os
import platform
import subprocess
import sys
from typing import Any, Dict, Optional
import psutil

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class SystemMetadata:
    """Dataclass holding comprehensive environment and hardware execution metadata."""

    os_name: str = field(default_factory=platform.system)
    os_release: str = field(default_factory=platform.release)
    os_version: str = field(default_factory=platform.version)
    architecture: str = field(default_factory=platform.machine)
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    python_executable: str = field(default_factory=lambda: sys.executable)
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0
    ram_used_gb: float = 0.0
    ram_usage_pct: float = 0.0
    cpu_cores_physical: int = 0
    cpu_cores_logical: int = 0
    cpu_frequency_mhz: float = 0.0
    cpu_usage_pct: float = 0.0
    gpu_available: bool = False
    gpu_name: str = "N/A"
    gpu_memory_total_gb: float = 0.0
    broker: str = "GenericBroker"
    quantlab_version: str = "1.0.0"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    git_commit: str = "N/A"
    git_branch: str = "N/A"
    random_seed: int = 42
    config_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata object to dictionary representation."""
        return asdict(self)


class MetadataExtractor:
    """Automated Metadata Extractor for QuantLab scientific experiments."""

    @staticmethod
    def _extract_git_info() -> Dict[str, str]:
        """Extract current Git repository commit hash and branch.

        Returns:
            Dict containing 'commit' and 'branch'.
        """
        commit = "N/A"
        branch = "N/A"
        try:
            commit = (
                subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
                .decode("utf-8")
                .strip()
            )
            branch = (
                subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL)
                .decode("utf-8")
                .strip()
            )
        except Exception:
            pass
        return {"commit": commit, "branch": branch}

    @staticmethod
    def _extract_gpu_info() -> Dict[str, Any]:
        """Extract GPU hardware specifications if available.

        Returns:
            Dict containing gpu_available, gpu_name, and gpu_memory_total_gb.
        """
        gpu_available = False
        gpu_name = "N/A"
        gpu_memory_total_gb = 0.0

        if HAS_TORCH and torch.cuda.is_available():
            try:
                gpu_available = True
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory_total_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
            except Exception:
                pass
        return {
            "gpu_available": gpu_available,
            "gpu_name": gpu_name,
            "gpu_memory_total_gb": gpu_memory_total_gb,
        }

    @classmethod
    def collect(
        cls,
        broker: str = "GenericBroker",
        quantlab_version: str = "1.0.0",
        random_seed: int = 42,
        config_hash: str = "",
    ) -> SystemMetadata:
        """Collect current system, runtime, hardware, and environment metadata.

        Args:
            broker: Broker identifier.
            quantlab_version: Version of QuantLab platform.
            random_seed: Execution random seed.
            config_hash: Configuration SHA256 checksum.

        Returns:
            Populated SystemMetadata instance.
        """
        # Memory metrics
        mem = psutil.virtual_memory()
        ram_total_gb = round(mem.total / (1024**3), 2)
        ram_available_gb = round(mem.available / (1024**3), 2)
        ram_used_gb = round(mem.used / (1024**3), 2)
        ram_usage_pct = float(mem.percent)

        # CPU metrics
        cpu_phys = psutil.cpu_count(logical=False) or 1
        cpu_log = psutil.cpu_count(logical=True) or 1
        cpu_freq_info = psutil.cpu_freq()
        cpu_freq = float(cpu_freq_info.current) if cpu_freq_info else 0.0
        cpu_usage = float(psutil.cpu_percent(interval=None))

        # Git info
        git_info = cls._extract_git_info()

        # GPU info
        gpu_info = cls._extract_gpu_info()

        return SystemMetadata(
            os_name=platform.system(),
            os_release=platform.release(),
            os_version=platform.version(),
            architecture=platform.machine(),
            python_version=sys.version.split()[0],
            python_executable=sys.executable,
            ram_total_gb=ram_total_gb,
            ram_available_gb=ram_available_gb,
            ram_used_gb=ram_used_gb,
            ram_usage_pct=ram_usage_pct,
            cpu_cores_physical=cpu_phys,
            cpu_cores_logical=cpu_log,
            cpu_frequency_mhz=cpu_freq,
            cpu_usage_pct=cpu_usage,
            gpu_available=gpu_info["gpu_available"],
            gpu_name=gpu_info["gpu_name"],
            gpu_memory_total_gb=gpu_info["gpu_memory_total_gb"],
            broker=broker,
            quantlab_version=quantlab_version,
            timestamp=datetime.now(timezone.utc).isoformat(),
            git_commit=git_info["commit"],
            git_branch=git_info["branch"],
            random_seed=random_seed,
            config_hash=config_hash,
        )
