"""Data ingestion, validation, and exploration modules."""

from terraspectra.data.loader import (
    SalinasDataset,
    SalinasLoader,
    DatasetMetadata,
    load_salinas_data,
    create_synthetic_salinas_fixture,
    SALINAS_CLASS_NAMES,
)
from terraspectra.data.exploration import (
    HyperspectralExplorer,
    QualityAuditReport,
    ExplorationSummary,
)

__all__ = [
    "SalinasDataset",
    "SalinasLoader",
    "DatasetMetadata",
    "load_salinas_data",
    "create_synthetic_salinas_fixture",
    "SALINAS_CLASS_NAMES",
    "HyperspectralExplorer",
    "QualityAuditReport",
    "ExplorationSummary",
]

