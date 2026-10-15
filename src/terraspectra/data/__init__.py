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
from terraspectra.data.patches import (
    PatchSample,
    extract_spatial_patch,
    pad_cube_spatially,
    validate_patch_parameters,
    iter_pixel_patches,
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
    "PatchSample",
    "extract_spatial_patch",
    "pad_cube_spatially",
    "validate_patch_parameters",
    "iter_pixel_patches",
]

