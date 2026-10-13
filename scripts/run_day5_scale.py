"""Script for Day 5: Feature normalization and scaling on PCA-reduced data."""

import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from terraspectra.processing.scaling import HyperspectralScaler, ScalingMethod


def main():
    parser = argparse.ArgumentParser(description="Day 5: Feature Scaling and Normalization")
    parser.add_argument(
        "--pca-data-path",
        type=str,
        default="data/processed/salinas_pca_reduced.npz",
        help="Path to PCA-reduced NPZ data file",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="standard",
        choices=["standard", "minmax"],
        help="Scaling method to apply (standard or minmax)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save models and reports",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("TerraSpectra — Day 5: Feature Normalization & Scaling")
    print("=" * 65)

    pca_file = Path(args.pca_data_path)
    if not pca_file.exists():
        print(f"[ERROR] PCA reduced data file not found at {pca_file}. Run Day 4 first!", file=sys.stderr)
        sys.exit(1)

    loaded = np.load(str(pca_file))
    reduced_cube = loaded["reduced_cube"]
    print(f"Loaded PCA Cube Shape: {reduced_cube.shape}")

    scaler = HyperspectralScaler(method=args.method)
    scaled_cube = scaler.fit_transform(reduced_cube)

    # Save scaled data, model artifact, and metadata report
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    scaled_out = processed_dir / f"salinas_scaled_{args.method}.npz"
    np.savez_compressed(str(scaled_out), scaled_cube=scaled_cube)

    model_path = Path("outputs/models") / f"scaler_{args.method}.joblib"
    report_path = Path(args.output_dir) / "reports" / f"day5_scaling_{args.method}_report.json"
    scaler.save(model_path, report_path)

    meta = scaler.get_metadata()
    print("\n--- Scaling Verification Report ---")
    print(f"Applied Method:     {meta.method.upper()}")
    print(f"Scaled Cube Shape:  {scaled_cube.shape}")
    print(f"Scaled Value Range: [{scaled_cube.min():.4f}, {scaled_cube.max():.4f}]")
    print(f"Scaled Mean ± Std:  {scaled_cube.mean():.4f} ± {scaled_cube.std():.4f}")
    print(f"Constant Features:  {meta.constant_features_detected}")
    print(f"Saved Scaled Data:  {scaled_out}")
    print(f"Saved Fitted Model: {model_path}")
    print(f"Saved Scaler JSON:  {report_path}")
    print("\n[SUCCESS] Day 5 Feature Scaling and Normalization completed successfully.\n")


if __name__ == "__main__":
    main()
