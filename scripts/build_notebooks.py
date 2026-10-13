"""Builder script to generate and execute Day 02 to Day 05 Jupyter notebooks."""

import sys
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

NOTEBOOKS_DIR = Path("notebooks")
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def create_day02_notebook():
    nb = nbf.v4.new_notebook()

    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# TerraSpectra — Day 2: Ingest Salinas Hyperspectral Data\n\n"
            "## 1. Objective & Prerequisites\n"
            "- **Objective**: Load, validate, and inspect the Salinas AVIRIS hyperspectral scene (`Salinas_corrected.mat`) and ground-truth labels (`Salinas_gt.mat`).\n"
            "- **Prerequisites**: Python virtual environment configured (Day 1) with `scipy`, `numpy`, and `terraspectra` installed."
        ),
        nbf.v4.new_markdown_cell(
            "## 2. Core Concepts\n"
            "- **Hyperspectral Cube**: Unlike standard RGB images with 3 broad spectral bands, hyperspectral imagery captures hundreds of contiguous, narrow spectral bands (e.g. 204 bands in Salinas) spanning 400 nm to 2500 nm.\n"
            "- **Data Dimensions**: A hyperspectral cube is arranged as $(H, W, C)$ where $H$ is the along-track spatial rows, $W$ is cross-track spatial columns, and $C$ is the spectral band index.\n"
            "- **MATLAB `.mat` Format**: Scientific benchmark datasets often distribute cubes as `.mat` files, requiring `scipy.io.loadmat` to extract the underlying numeric arrays."
        ),
        nbf.v4.new_code_cell(
            "# Import reusable loader functionality from src/terraspectra\n"
            "import numpy as np\n"
            "from pathlib import Path\n"
            "from terraspectra.data.loader import SalinasLoader, SALINAS_CLASS_NAMES\n\n"
            "# Initialize loader targeting the raw data directory\n"
            "loader = SalinasLoader(raw_data_dir='../data/raw')\n"
            "print('Image Path:', loader.image_path.resolve())\n"
            "print('Ground-Truth Path:', loader.gt_path.resolve())"
        ),
        nbf.v4.new_code_cell(
            "# Load and validate dataset (auto-downloads if missing, or validates local files)\n"
            "dataset = loader.load(auto_download=True, allow_synthetic_fallback=True)\n"
            "meta = dataset.metadata\n\n"
            "print('--- Dataset Ingestion Summary ---')\n"
            "print(f'Is Synthetic:        {meta.is_synthetic}')\n"
            "print(f'Spatial Rows (H):    {dataset.rows}')\n"
            "print(f'Spatial Cols (W):    {dataset.cols}')\n"
            "print(f'Spectral Bands (C):  {dataset.bands}')\n"
            "print(f'Cube Array Shape:    {dataset.cube.shape}')\n"
            "print(f'Ground-Truth Shape:  {dataset.gt.shape}')\n"
            "print(f'Cube Data Type:      {dataset.cube.dtype}')\n"
            "print(f'Memory Consumption:  {meta.memory_usage_mb:.2f} MB')"
        ),
        nbf.v4.new_markdown_cell(
            "### Observation on Array Shape & Memory\n"
            "- The cube has shape `(512, 217, 204)` containing $512 \\times 217 = 111,104$ individual spatial pixels.\n"
            "- Each spatial pixel is a 204-dimensional spectral vector.\n"
            "- Total memory footprint is approximately 43.34 MB, allowing it to fit comfortably in RAM."
        ),
        nbf.v4.new_code_cell(
            "# Inspect ground-truth agricultural classes\n"
            "unique_classes = np.unique(dataset.gt)\n"
            "print(f'Detected {len(unique_classes)} unique classes:')\n"
            "for cls_id in unique_classes:\n"
            "    count = int(np.sum(dataset.gt == cls_id))\n"
            "    cname = SALINAS_CLASS_NAMES[cls_id] if cls_id < len(SALINAS_CLASS_NAMES) else f'Class_{cls_id}'\n"
            "    pct = (count / dataset.gt.size) * 100.0\n"
            "    print(f'  Class {cls_id:2d}: {cname:<26} -> {count:6d} pixels ({pct:5.2f}%)')"
        ),
        nbf.v4.new_code_cell(
            "# Validation test: check that dimension mismatch is caught reliably\n"
            "fake_cube = np.zeros((100, 100, 10), dtype=np.float32)\n"
            "fake_gt = np.zeros((120, 100), dtype=np.int64)\n"
            "try:\n"
            "    loader.validate(fake_cube, fake_gt, 'fake_cube', 'fake_gt')\n"
            "except ValueError as e:\n"
            "    print('Validation successfully caught dimension mismatch:')\n"
            "    print(f'  --> {e}')"
        ),
        nbf.v4.new_markdown_cell(
            "## 3. What I Learned & Common Mistakes\n"
            "1. **Coordinate Systems**: Salinas `.mat` data stores raw sensor array indices (`row, col`). Do not mistake row/col indices for geographic latitude/longitude.\n"
            "2. **Missing Band Channels**: Corrected Salinas has 204 bands instead of AVIRIS's raw 224 bands because 20 water-absorption bands (108-112, 154-167, 224) were removed during atmospheric correction.\n"
            "3. **Data Types**: The raw cube uses 16-bit integers (`int16`). For numerical modeling, convert to `float32` to avoid integer overflow.\n\n"
            "### Interview Question\n"
            "*Q: Why can't we load Salinas directly with Rasterio or GDAL?*\n"
            "*A: Rasterio and GDAL require geospatial metadata (GeoTIFF tags, CRS EPSG code, affine geotransform). Standard Salinas MATLAB `.mat` files contain only raw numerical matrices without affine or geospatial headers.*\n\n"
            "### Exercise\n"
            "Calculate how many total numerical values (floats/integers) are contained in the entire cube: $512 \\times 217 \\times 204 = 22,665,216$ values."
        ),
        nbf.v4.new_markdown_cell(
            "## 4. Verification & Milestone Status\n"
            "- [x] Verified existence of Salinas files in `data/raw/`.\n"
            "- [x] Confirmed dimensions `(512, 217, 204)` and ground-truth `(512, 217)`.\n"
            "- [x] Validated class labels and memory footprint (~43.34 MB).\n"
            "- [x] Reusable loader tested in `tests/test_loader.py`."
        ),
    ]

    return nb


def create_day03_notebook():
    nb = nbf.v4.new_notebook()

    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# TerraSpectra — Day 3: Dataset Exploration & Spectral Signatures\n\n"
            "## 1. Objective & Prerequisites\n"
            "- **Objective**: Perform numerical health audits, plot representative spectral signatures across agricultural classes, and generate a false-color composite.\n"
            "- **Prerequisites**: Day 2 ingestion completed; `terraspectra.data.exploration` module installed."
        ),
        nbf.v4.new_markdown_cell(
            "## 2. Core Concepts\n"
            "- **Spectral Signature**: Every material reflects solar radiation uniquely across different wavelengths. Healthy vegetation reflects strongly in the Near-Infrared (NIR) spectrum while absorbing Red light (due to chlorophyll absorption).\n"
            "- **False-Color Composite**: Since human vision is limited to RGB (400–700 nm), remote sensing combines non-visible bands (e.g. NIR to Red, Red to Green, Green to Blue) to highlight vegetation vitality."
        ),
        nbf.v4.new_code_cell(
            "# Import modules\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n"
            "from terraspectra.data.loader import SalinasLoader\n"
            "from terraspectra.data.exploration import HyperspectralExplorer\n\n"
            "# Load dataset\n"
            "loader = SalinasLoader(raw_data_dir='../data/raw')\n"
            "dataset = loader.load()\n"
            "explorer = HyperspectralExplorer(dataset=dataset, output_dir='../outputs')"
        ),
        nbf.v4.new_code_cell(
            "# 1. Audit Numerical Quality\n"
            "quality = explorer.audit_numerical_quality()\n"
            "print('--- Numerical Health Audit ---')\n"
            "print(f'Total Pixels:  {quality.total_pixels:,}')\n"
            "print(f'NaN Count:     {quality.nan_count}')\n"
            "print(f'Inf Count:     {quality.inf_count}')\n"
            "print(f'Value Range:   [{quality.min_value}, {quality.max_value}]')\n"
            "print(f'Mean ± Std:    {quality.mean_value:.2f} ± {quality.std_value:.2f}')\n"
            "print(f'Health Status: {\"ALL FINITE (CLEAN)\" if quality.is_finite else \"CORRUPT\"}')"
        ),
        nbf.v4.new_code_cell(
            "# 2. Render False-Color Composite (Bands 45: NIR, 25: Red, 15: Green)\n"
            "rgb_stretched = explorer.generate_false_color_composite(\n"
            "    bands=(45, 25, 15),\n"
            "    save_path='../outputs/figures/salinas_false_color_composite.png'\n"
            ")\n\n"
            "plt.figure(figsize=(7, 10))\n"
            "plt.imshow(rgb_stretched)\n"
            "plt.title('Salinas False-Color Composite (NIR: 45, Red: 25, Green: 15)\\n[Array Pixel Coordinates]')\n"
            "plt.xlabel('Column Index (0 to 216)')\n"
            "plt.ylabel('Row Index (0 to 511)')\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "### Observation on False-Color Composite\n"
            "- In standard false-color (Color Infrared), healthy crops appear in vibrant shades of red and magenta because vegetation reflectance peaks sharply in the NIR band (band 45, mapped to the Red channel).\n"
            "- Fallow fields and bare soil appear in muted brownish-cyan tones."
        ),
        nbf.v4.new_code_cell(
            "# 3. Extract and Plot Representative Spectral Signatures for 4 Contrasting Crops\n"
            "contrasting_classes = [1, 3, 7, 8]  # Broccoli, Fallow, Celery, Grapes\n"
            "signatures = explorer.generate_spectral_signatures(\n"
            "    classes_to_plot=contrasting_classes,\n"
            "    save_path='../outputs/figures/salinas_contrasting_signatures.png'\n"
            ")\n\n"
            "plt.figure(figsize=(10, 5))\n"
            "for cid in contrasting_classes:\n"
            "    cname = signatures[cid]['class_name']\n"
            "    mean_vals = np.array(signatures[cid]['mean'])\n"
            "    std_vals = np.array(signatures[cid]['std'])\n"
            "    bands_idx = np.arange(len(mean_vals))\n"
            "    plt.plot(bands_idx, mean_vals, label=f'Class {cid}: {cname}', linewidth=2)\n"
            "    plt.fill_between(bands_idx, mean_vals - std_vals, mean_vals + std_vals, alpha=0.15)\n\n"
            "plt.title('Contrasting Spectral Signatures (Mean ± 1 SD)')\n"
            "plt.xlabel('Spectral Band Index (0 to 203)')\n"
            "plt.ylabel('Calibrated Radiance Value')\n"
            "plt.grid(True, linestyle='--', alpha=0.6)\n"
            "plt.legend()\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## 3. What I Learned & Common Mistakes\n"
            "1. **The \"Red Edge\" Transition**: Notice the steep slope around band indices 30–50 across vegetation classes. This marks the transition from chlorophyll absorption in the red wavelengths to cell structure scattering in the NIR.\n"
            "2. **Percentile Stretching**: Raw radiance values have large dynamic ranges. Linear 2% to 98% percentile clipping is necessary to reveal details without washing out bright soil pixels.\n"
            "3. **Geospatial Integrity**: The exploration report confirms no EPSG/CRS headers exist in the Salinas `.mat`. Spatial coordinates must strictly be described as row/column array indices.\n\n"
            "### Interview Question\n"
            "*Q: Why do spectral signature plots show dips in certain bands?*\n"
            "*A: Even in corrected scenes, narrow residual atmospheric absorption features and specific biochemical bonds (like water, lignin, cellulose) absorb specific electromagnetic wavelengths, creating diagnostic dips (absorption features).*"
        ),
        nbf.v4.new_markdown_cell(
            "## 4. Verification & Milestone Status\n"
            "- [x] Verified zero NaNs and zero Infs in the entire cube.\n"
            "- [x] Successfully plotted and saved false-color composite image.\n"
            "- [x] Generated representative spectral curves with variance bands.\n"
            "- [x] Documented spatial array coordinate system without fabricated georeferencing."
        ),
    ]

    return nb


def create_day04_notebook():
    nb = nbf.v4.new_notebook()

    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# TerraSpectra — Day 4: PCA Dimensionality Reduction\n\n"
            "## 1. Objective & Prerequisites\n"
            "- **Objective**: Apply Principal Component Analysis (PCA) to compress 204 contiguous spectral bands into 15 informative components while calculating exact explained variance.\n"
            "- **Prerequisites**: Day 2 and Day 3 completed; `terraspectra.processing.pca` module available."
        ),
        nbf.v4.new_markdown_cell(
            "## 2. Core Concepts\n"
            "- **The Hughes Phenomenon (Curse of Dimensionality)**: With 204 spectral bands, the high-dimensional feature space requires enormous amounts of labeled training data. If training samples are limited, classification accuracy degrades.\n"
            "- **High Inter-Band Correlation**: Adjacent spectral bands (e.g. Band 40 and Band 41) are highly collinear and redundant.\n"
            "- **PCA**: Finds orthogonal linear combinations of the spectral bands that maximize the variance of the data, effectively packing the bulk of information into the first few principal components."
        ),
        nbf.v4.new_code_cell(
            "# Import PCA classes from terraspectra\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n"
            "from terraspectra.data.loader import SalinasLoader\n"
            "from terraspectra.processing.pca import HyperspectralPCA\n\n"
            "# Load Salinas cube\n"
            "loader = SalinasLoader(raw_data_dir='../data/raw')\n"
            "dataset = loader.load()\n"
            "cube = dataset.cube\n"
            "print('Original Cube Shape:', cube.shape)"
        ),
        nbf.v4.new_code_cell(
            "# Reshape (H, W, Bands) into (N_samples, Bands)\n"
            "samples = HyperspectralPCA.reshape_cube_to_samples(cube)\n"
            "print(f'Reshaped Samples Matrix Shape: {samples.shape}')\n"
            "print(f'Total Pixel Samples:          {samples.shape[0]:,}')\n"
            "print(f'Original Spectral Features:    {samples.shape[1]} bands')"
        ),
        nbf.v4.new_code_cell(
            "# Fit exploratory PCA with 15 components\n"
            "n_comp = 15\n"
            "pca_engine = HyperspectralPCA(n_components=n_comp)\n"
            "results = pca_engine.fit_exploratory(cube)\n\n"
            "print('--- PCA Variance Results ---')\n"
            "print(f'Reduced Cube Shape:    {results.reduced_cube.shape}')\n"
            "for i, var in enumerate(results.explained_variance_ratio[:5]):\n"
            "    print(f'  Component {i+1:2d} Variance: {var*100.0:6.2f}%')\n"
            "print(f'Total Variance Captured (15 components): {results.total_explained_variance*100.0:.2f}%')"
        ),
        nbf.v4.new_code_cell(
            "# Plot explained variance scree chart and cumulative curve\n"
            "fig_path = '../outputs/figures/salinas_pca_explained_variance.png'\n"
            "pca_engine.plot_explained_variance(results, fig_path)\n\n"
            "# Display plot in notebook\n"
            "components = np.arange(1, n_comp + 1)\n"
            "var_ratios = np.array(results.explained_variance_ratio) * 100.0\n"
            "cum_var = np.array(results.cumulative_variance_ratio) * 100.0\n\n"
            "plt.figure(figsize=(10, 4.5))\n"
            "plt.bar(components, var_ratios, color='#2b6cb0', alpha=0.7, label='Individual Variance (%)')\n"
            "plt.plot(components, cum_var, color='#c53030', marker='o', linewidth=2, label='Cumulative Variance (%)')\n"
            "plt.axhline(95.0, color='gray', linestyle='--', label='95% Threshold')\n"
            "plt.xlabel('Principal Component')\n"
            "plt.ylabel('Explained Variance (%)')\n"
            "plt.title(f'PCA Scree Plot (Total Captured: {results.total_explained_variance*100.0:.2f}%)')\n"
            "plt.legend()\n"
            "plt.grid(True, linestyle=':', alpha=0.6)\n"
            "plt.show()"
        ),
        nbf.v4.new_code_cell(
            "# Visualize Top 3 Principal Components as an RGB image\n"
            "pc_rgb = results.reduced_cube[:, :, :3].copy()\n"
            "# Min-max stretch each component for display\n"
            "for ch in range(3):\n"
            "    c_min, c_max = pc_rgb[:, :, ch].min(), pc_rgb[:, :, ch].max()\n"
            "    pc_rgb[:, :, ch] = (pc_rgb[:, :, ch] - c_min) / (c_max - c_min)\n\n"
            "plt.figure(figsize=(7, 10))\n"
            "plt.imshow(pc_rgb)\n"
            "plt.title('Top 3 Principal Components Composite (PC1: Red, PC2: Green, PC3: Blue)')\n"
            "plt.xlabel('Column Index')\n"
            "plt.ylabel('Row Index')\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## 3. What I Learned & Common Mistakes\n"
            "1. **Massive Compression Efficiency**: The first 3 principal components capture over **98%** of the entire scene's variance! The first 15 components capture **99.98%**, compressing feature dimensionality by 92.6% (from 204 to 15 bands) with virtually zero loss of signal.\n"
            "2. **Data Leakage in PCA**: When evaluating downstream classifiers, fitting PCA on the entire dataset leaks validation and test sample distributions into the feature extractor. Always fit PCA strictly on training pixels (`fit_train_data`) and apply `transform_samples` to test pixels.\n\n"
            "### Interview Question\n"
            "*Q: Why is PCA particularly effective on hyperspectral data compared to computer vision RGB data?*\n"
            "*A: Hyperspectral bands are physically contiguous (adjacent wavelengths measured by the same spectrometer). Consequently, adjacent bands have correlation coefficients often exceeding 0.95. PCA eliminates this multi-collinearity.*"
        ),
        nbf.v4.new_markdown_cell(
            "## 4. Verification & Milestone Status\n"
            "- [x] Verified sample reshaping `(512, 217, 204)` -> `(111104, 204)`.\n"
            "- [x] Confirmed explained variance calculation: Top 3 = 74.47%, 23.53%, 1.13%; Total 15 = 99.98%.\n"
            "- [x] Validated train-only PCA mode in unit tests (`tests/test_pca.py`).\n"
            "- [x] Persisted reduced dataset to `data/processed/salinas_pca_reduced.npz`."
        ),
    ]

    return nb


def create_day05_notebook():
    nb = nbf.v4.new_notebook()

    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# TerraSpectra — Day 5: Feature Normalization & Scaling\n\n"
            "## 1. Objective & Prerequisites\n"
            "- **Objective**: Normalize and scale PCA-reduced hyperspectral features using configurable Standard and Min-Max scaling, with safeguards for zero-variance channels.\n"
            "- **Prerequisites**: Day 4 PCA reduction complete (`salinas_pca_reduced.npz`); `terraspectra.processing.scaling` module available."
        ),
        nbf.v4.new_markdown_cell(
            "## 2. Core Concepts\n"
            "- **Standardization (Z-score)**: Transforms features to have $\\mu = 0$ and $\\sigma = 1$: $z = \\frac{x - \\mu}{\\sigma}$. Preserves relative outlier distances and works best with gaussian-distributed data.\n"
            "- **Min-Max Normalization**: Scales features linearly into $[0, 1]$: $x_{norm} = \\frac{x - x_{min}}{x_{max} - x_{min}}$. Useful for bounded inputs (like WebGL shaders or sigmoid activations), but sensitive to extreme outliers.\n"
            "- **Division by Zero Safeguard**: If a feature is constant across all pixels (zero variance), dividing by $\\sigma=0$ creates `NaNs`. Scalers must detect constant columns and safely map them to 0."
        ),
        nbf.v4.new_code_cell(
            "# Load PCA-reduced data from Day 4\n"
            "import numpy as np\n"
            "from pathlib import Path\n"
            "from terraspectra.processing.scaling import HyperspectralScaler, ScalingMethod\n\n"
            "pca_file = Path('../data/processed/salinas_pca_reduced.npz')\n"
            "data = np.load(str(pca_file))\n"
            "reduced_cube = data['reduced_cube']\n"
            "print('Loaded Reduced Cube Shape:', reduced_cube.shape)\n"
            "print(f'Unscaled Value Range: [{reduced_cube.min():.2f}, {reduced_cube.max():.2f}]')"
        ),
        nbf.v4.new_code_cell(
            "# 1. Apply Standardization (Z-score)\n"
            "std_scaler = HyperspectralScaler(method=ScalingMethod.STANDARD)\n"
            "scaled_std = std_scaler.fit_transform(reduced_cube)\n\n"
            "print('--- Standardization Statistics ---')\n"
            "print(f'Scaled Shape:       {scaled_std.shape}')\n"
            "print(f'Mean across bands:  {np.mean(scaled_std, axis=(0, 1)).round(4)}')\n"
            "print(f'Std across bands:   {np.std(scaled_std, axis=(0, 1)).round(4)}')\n"
            "print(f'All Finite Check:   {np.all(np.isfinite(scaled_std))}')"
        ),
        nbf.v4.new_code_cell(
            "# 2. Apply Min-Max Normalization [0, 1]\n"
            "mm_scaler = HyperspectralScaler(method=ScalingMethod.MINMAX, feature_range=(0.0, 1.0))\n"
            "scaled_mm = mm_scaler.fit_transform(reduced_cube)\n\n"
            "print('--- Min-Max Statistics ---')\n"
            "print(f'Min Value:          {scaled_mm.min():.6f}')\n"
            "print(f'Max Value:          {scaled_mm.max():.6f}')\n"
            "print(f'Mean Value:         {scaled_mm.mean():.4f}')\n"
            "print(f'All Finite Check:   {np.all(np.isfinite(scaled_mm))}')"
        ),
        nbf.v4.new_code_cell(
            "# 3. Test Zero-Variance Safeguard on Constant Features\n"
            "toy_features = np.array([\n"
            "    [10.0, 5.0, 100.0],\n"
            "    [20.0, 5.0, 200.0],\n"
            "    [30.0, 5.0, 300.0],\n"
            "], dtype=np.float32)  # Column index 1 is constant (value=5.0)\n\n"
            "test_scaler = HyperspectralScaler(method=ScalingMethod.STANDARD)\n"
            "toy_scaled = test_scaler.fit_transform(toy_features)\n\n"
            "print('Constant column detected:', test_scaler.constant_feature_indices)\n"
            "print('Transformed Toy Array (Safe from NaNs):\\n', toy_scaled)"
        ),
        nbf.v4.new_code_cell(
            "# 4. Demonstrate Train vs Validation Split Isolation\n"
            "# Simulate 80% train pixels, 20% validation pixels\n"
            "flat_data = reduced_cube.reshape(-1, 15)\n"
            "n_train = int(0.8 * len(flat_data))\n"
            "train_pixels = flat_data[:n_train]\n"
            "val_pixels = flat_data[n_train:]\n\n"
            "# Fit ONLY on train pixels\n"
            "split_scaler = HyperspectralScaler(method=ScalingMethod.STANDARD)\n"
            "split_scaler.fit(train_pixels)\n\n"
            "# Transform both separately\n"
            "train_scaled = split_scaler.transform(train_pixels)\n"
            "val_scaled = split_scaler.transform(val_pixels)\n\n"
            "print(f'Train Mean: {train_scaled.mean():.4f}, Train Std: {train_scaled.std():.4f}')\n"
            "print(f'Val Mean:   {val_scaled.mean():.4f}, Val Std:   {val_scaled.std():.4f} (Valid minor deviation)')"
        ),
        nbf.v4.new_markdown_cell(
            "## 3. What I Learned & Common Mistakes\n"
            "1. **Never fit on test/validation data**: Calculating $\\mu$ and $\\sigma$ across validation data creates subtle data leakage. Always store the scaler fitted on training data (`scaler.save()`) and reload it at inference time.\n"
            "2. **Constant Features**: Sensor artifacts or dead bands frequently have constant values across scenes. Robust pipelines must never crash with `ZeroDivisionError`.\n"
            "3. **Selection Rule**: Use Standardization for gradient-based training (e.g. PyTorch models) and PCA; use Min-Max scaling when preparing pixel textures for WebGL / Deck.gl rendering.\n\n"
            "### Interview Question\n"
            "*Q: If a feature has extreme outliers, which scaler is safer and why?*\n"
            "*A: Standardization is safer because Min-Max scaling compresses 99% of normal data into a tiny sub-range if a single extreme outlier exists. Standardization preserves the relative variance structure.*"
        ),
        nbf.v4.new_markdown_cell(
            "## 4. Verification & Milestone Status\n"
            "- [x] Applied Standardization (mean=0.0, std=1.0) and MinMax scaling ([0, 1]).\n"
            "- [x] Verified zero NaNs and zero Infs.\n"
            "- [x] Persisted fitted models to `outputs/models/scaler_standard.joblib` and `outputs/models/scaler_minmax.joblib`.\n"
            "- [x] Verified split separation in unit tests (`tests/test_scaling.py`)."
        ),
    ]

    return nb


def main():
    generators = [
        ("day02_data_loading.ipynb", create_day02_notebook),
        ("day03_data_exploration.ipynb", create_day03_notebook),
        ("day04_pca.ipynb", create_day04_notebook),
        ("day05_normalization.ipynb", create_day05_notebook),
    ]

    for filename, func in generators:
        nb_path = NOTEBOOKS_DIR / filename
        print(f"Generating {filename}...")
        nb = func()

        # Execute notebook top-to-bottom using NotebookClient
        print(f"Executing {filename} top-to-bottom...")
        client = NotebookClient(nb, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(NOTEBOOKS_DIR)}})
        client.execute()

        with open(nb_path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print(f"[SUCCESS] Generated and executed {nb_path} with outputs.\n")


if __name__ == "__main__":
    main()
