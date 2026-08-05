"""
QuantLab Cryptographic Integrity & Checksum Verification Engine.

Computes SHA-256 and MD5 cryptographic hashes, verifies payload checksums, and detects
data corruption or duplicate model weights/dataset artifacts.
"""

import hashlib
import json
import os
from typing import Any, Dict, Optional, Union
import numpy as np


class IntegrityChecker:
    """Institutional Cryptographic Integrity & Checksum Verification Engine."""

    @staticmethod
    def compute_sha256(data: Union[str, bytes, dict, np.ndarray]) -> str:
        """Compute SHA-256 hex digest string for string, bytes, dict, or numpy array.

        Args:
            data: Data payload.

        Returns:
            Hexadecimal SHA-256 digest string.
        """
        hasher = hashlib.sha256()
        if isinstance(data, str):
            hasher.update(data.encode("utf-8"))
        elif isinstance(data, bytes):
            hasher.update(data)
        elif isinstance(data, dict):
            hasher.update(json.dumps(data, sort_keys=True, default=str).encode("utf-8"))
        elif isinstance(data, np.ndarray):
            hasher.update(data.tobytes())
        else:
            hasher.update(str(data).encode("utf-8"))
        return hasher.hexdigest()

    @staticmethod
    def compute_file_sha256(filepath: str) -> str:
        """Compute SHA-256 checksum for a file on disk.

        Args:
            filepath: Destination file path.

        Returns:
            Hexadecimal SHA-256 digest string.
        """
        if not os.path.exists(filepath):
            return ""
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def verify_integrity(data_or_filepath: Union[str, bytes, dict], expected_hash: str) -> bool:
        """Verify payload or file integrity matches expected SHA-256 digest.

        Returns:
            Boolean indicating whether integrity is verified intact.
        """
        if isinstance(data_or_filepath, str) and os.path.exists(data_or_filepath):
            actual = IntegrityChecker.compute_file_sha256(data_or_filepath)
        else:
            actual = IntegrityChecker.compute_sha256(data_or_filepath)
        return actual.lower() == expected_hash.lower()
