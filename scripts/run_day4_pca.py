"""Script for Day 4: PCA dimensionality reduction on Salinas dataset."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from terraspectra.data.loader import SalinasLoader
from terraspectra.processing.pca import HyperspectralPCA


def main():
    parser = argparse.ArgumentParser(description="Day 4: PCA Dimensionality Reduction")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Directory containing raw Salinas data")
    parser.add_argument("--n-components", type=int, default=15, help="Target number of principal components")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory for reports & figures")
    args = parser.parse_args()

    print("=" * 65)
    print("TerraSpectra — Day 4: PCA Dimensionality Reduction")
    print("=" * 65)

    loader = SalinasLoader(raw_data_dir=args.data_dir)
    dataset = loader.load(auto_download=True, allow_synthetic_fallback=True)

    print(f"\nOriginal Cube Shape: {dataset.cube.shape} (Bands: {dataset.bands})")
    print(f"Fitting PCA with {args.n_components} components...")

    pca_model = HyperspectralPCA(n_components=args.n_components)
    results = pca_model.fit_exploratory(dataset.cube)

    # Save visualization and compressed array
    fig_path = Path(args.output_dir) / "figures" / "salinas_pca_explained_variance.png"
    pca_model.plot_explained_variance(results, fig_path)

    out_data = Path("data/processed") / "salinas_pca_reduced.npz"
    rep_path = Path(args.output_dir) / "reports" / "day4_pca_variance_report.json"
    pca_model.save_results(results, out_data, rep_path)

    print("\n--- PCA Results Summary ---")
    print(f"Reduced Cube Shape:       {results.reduced_cube.shape}")
    print(f"Reduced Samples Matrix:   {results.reduced_samples.shape}")
    print(f"Top 3 Component Variance: {[round(v * 100.0, 2) for v in results.explained_variance_ratio[:3]]}%")
    print(f"Total Variance Captured:  {results.total_explained_variance * 100.0:.2f}%")
    print(f"Saved Variance Plot:      {fig_path}")
    print(f"Saved Reduced Data:       {out_data}")
    print(f"Saved Metadata Report:    {rep_path}")
    print("\n[SUCCESS] Day 4 PCA Dimensionality Reduction completed successfully.\n")


if __name__ == "__main__":
    main()
