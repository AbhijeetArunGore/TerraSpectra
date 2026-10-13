"""Utility functions for TerraSpectra."""

import os
from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    """Ensure that the parent directory or specified directory exists."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
