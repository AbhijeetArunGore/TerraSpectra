"""Script for Day 3: Dataset exploration, quality audit, and visualization."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from terraspectra.data.loader import SalinasLoader
from terraspectra.data.exploration import HyperspectralExplorer


def main():
    parser = argparse.ArgumentParser(description="Day 3: Explore Salinas Hyperspectral Dataset")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Directory containing Salinas dataset .mat files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save figures and exploration reports",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("TerraSpectra — Day 3: Hyperspectral Exploration & Signatures")
    print("=" * 65)

    loader = SalinasLoader(raw_data_dir=args.data_dir)
    dataset = loader.load(auto_download=True, allow_synthetic_fallback=True)

    explorer = HyperspectralExplorer(dataset=dataset, output_dir=args.output_dir)
    print("\n[1/4] Auditing numerical quality...")
    quality = explorer.audit_numerical_quality()
    print(f"      Total Pixels:  {quality.total_pixels:,}")
    print(f"      NaN Count:     {quality.nan_count}")
    print(f"      Inf Count:     {quality.inf_count}")
    print(f"      Value Range:   [{quality.min_value:.2f}, {quality.max_value:.2f}]")
    print(f"      Mean ± Std:    {quality.mean_value:.2f} ± {quality.std_value:.2f}")
    print(f"      Finite Status: {'PASS (All finite)' if quality.is_finite else 'FAIL (Contains NaN/Inf)'}")

    print("\n[2/4] Generating false-color composite (NIR: 45, Red: 25, Green: 15)...")
    explorer.generate_false_color_composite(bands=(45, 25, 15))
    print(f"      Saved: outputs/figures/salinas_false_color_composite.png")

    print("\n[3/4] Extracting representative spectral signatures...")
    explorer.generate_spectral_signatures()
    print(f"      Saved: outputs/figures/salinas_spectral_signatures.png")

    print("\n[4/4] Computing class distribution and metadata analysis...")
    explorer.generate_class_distribution_plot()
    print(f"      Saved: outputs/figures/salinas_class_distribution.png")

    summary = explorer.run_full_exploration()
    report_path = Path(args.output_dir) / "reports" / "day3_exploration_report.json"
    print(f"      Saved: {report_path}")

    print("\n--- Spatial & Geospatial Coordinate Investigation ---")
    print(f"Coordinate System:    {summary.coordinate_system}")
    print(f"Georeferencing Audit: {summary.georeferencing_status}")
    print("\n[SUCCESS] Day 3 Exploration workflow finished successfully.\n")


if __name__ == "__main__":
    main()
