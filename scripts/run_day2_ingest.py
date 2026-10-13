"""Script for Day 2: Ingest Salinas hyperspectral data and ground truth."""

import argparse
import sys
from pathlib import Path

# Add src to pythonpath if running directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from terraspectra.data.loader import SalinasLoader, SALINAS_CLASS_NAMES


def main():
    parser = argparse.ArgumentParser(description="Ingest Salinas Hyperspectral Dataset")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Directory to store or load Salinas dataset .mat files",
    )
    parser.add_argument(
        "--auto-download",
        action="store_true",
        default=True,
        help="Attempt to download files if missing (default: True)",
    )
    parser.add_argument(
        "--allow-synthetic-fallback",
        action="store_true",
        default=True,
        help="Generate synthetic test fixture if remote download fails or is unavailable",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("TerraSpectra — Day 2: Salinas Hyperspectral Data Ingestion")
    print("=" * 60)

    loader = SalinasLoader(raw_data_dir=args.data_dir)
    print(f"Target directory: {loader.raw_data_dir.resolve()}")
    print(f"Image path:       {loader.image_path}")
    print(f"Ground-truth:     {loader.gt_path}")

    try:
        dataset = loader.load(
            auto_download=args.auto_download,
            allow_synthetic_fallback=args.allow_synthetic_fallback,
        )
    except FileNotFoundError as e:
        print(f"\n[ERROR] Dataset files could not be loaded: {e}", file=sys.stderr)
        sys.exit(1)

    meta = dataset.metadata
    print("\n--- Ingestion & Validation Report ---")
    print(f"Dataset Status:        {'SYNTHETIC FIXTURE' if meta.is_synthetic else 'GENUINE SALINAS DATASET'}")
    print(f"Image Cube Dimensions: {meta.image_shape} (Height: {meta.image_shape[0]}, Width: {meta.image_shape[1]}, Bands: {meta.image_shape[2]})")
    print(f"Spectral Band Count:   {meta.spectral_band_count} bands")
    print(f"Label Grid Dimensions: {meta.label_shape}")
    print(f"Cube Data Type:        {meta.image_dtype}")
    print(f"Labels Data Type:      {meta.label_dtype}")
    print(f"Memory Footprint:      {meta.memory_usage_mb:.2f} MB")
    print(f"Unique Label Classes:  {len(meta.unique_classes)} classes {meta.unique_classes}")

    print("\nSample Classes Detected:")
    for cls_id in meta.unique_classes[:10]:
        cls_name = SALINAS_CLASS_NAMES[cls_id] if cls_id < len(SALINAS_CLASS_NAMES) else f"Class_{cls_id}"
        count = int((dataset.gt == cls_id).sum())
        print(f"  - Class {cls_id:2d} ({cls_name:<28}): {count:6d} pixels")
    if len(meta.unique_classes) > 10:
        print(f"  ... and {len(meta.unique_classes) - 10} more classes.")

    print("\n[SUCCESS] Day 2 Ingestion and Validation completed successfully.\n")


if __name__ == "__main__":
    main()
