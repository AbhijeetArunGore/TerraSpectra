# TerraSpectra 🛰️

**Hyperspectral Data Engineering, Preprocessing, Deep Learning, and Interactive Visualization Platform**

TerraSpectra is an end-to-end hyperspectral remote sensing platform. This repository implements Days 1–11 of the 30-day roadmap, covering hyperspectral ingestion, validation, exploratory analysis, PCA dimensionality reduction, feature scaling, a high-performance Deck.gl interactive visualization scaffold, a genuine 3D-CNN baseline architecture, PyTorch DataLoaders, reproducible baseline model training, rigorous test-set evaluation, and security-conscious release hygiene.

---

## Roadmap Scope (Days 1–11)

- **Day 1: Repository & Python Environment**
  - Reproducible virtual environment and dependency configuration (`requirements.txt`, `pyproject.toml`).
  - Modular project structure: `terraspectra.data`, `terraspectra.processing`, `terraspectra.utils`.
  - Comprehensive `.gitignore` protecting against committing large raw datasets or binaries.
- **Day 2: Ingest Salinas Hyperspectral Data**
  - Download and loader pipeline for Salinas corrected hyperspectral cube (`Salinas_corrected.mat`) and ground-truth labels (`Salinas_gt.mat`).
  - File existence, data type, dimension, and spectral compatibility validation.
  - Transparent memory reporting and fallback synthetic fixtures for offline testing.
- **Day 3: Dataset Exploration & Signatures**
  - Spectral band count, missing/non-finite value audits, unique class frequencies.
  - Class-conditional representative spectral signature extraction.
  - False-color composite image rendering using documented AVIRIS band indices.
  - Explicit spatial array coordinate handling (no fabricated GPS/CRS).
- **Day 4: PCA Dimensionality Reduction**
  - Configurable Principal Component Analysis (e.g., 204 bands down to 10–30 principal components).
  - Explained variance and cumulative variance ratio computations.
  - Strict separation between exploratory PCA (unsupervised scene-wide) and model-evaluation PCA (fit on training pixels only to prevent data leakage).
- **Day 5: Normalization & Feature Scaling**
  - Standardization (Z-score) and Min-Max scaling for reduced spectral features.
  - Protection against zero-variance/constant features.
  - Fitted scaler persistence (`joblib`) and metadata reporting.
- **Day 6: React Frontend Scaffold & Deck.gl**
  - Vite + React + TypeScript web application (`frontend/`).
  - Deck.gl spatial pixel grid rendering without fabricated geographic projections.
- **Day 7: Geographical Base Map Foundation**
  - Interactive Web Mercator (`EPSG:3857`) basemap using MapLibre GL JS and Deck.gl `MapboxOverlay`.
  - Token-free default basemaps (CARTO Dark Matter, CARTO Positron, OpenStreetMap) with proper attribution.
  - Optional Mapbox GL integration via `VITE_MAPBOX_ACCESS_TOKEN` in `frontend/.env`.
  - Regional Salinas Valley demonstration extent (`[-121.6555°, 36.6777°]`, zoom 10.5).
  - Explicit distinction between WGS84 geographic coordinates and raw 512x217 raster array indices.
  - View mode toggling between Geographical Base Map (Day 7) and Orthographic Raster Scene (Day 6).
- **Day 8: 3D-CNN Baseline Architecture in PyTorch**
  - Genuine 3D-CNN model architecture (`Hyperspectral3DCNN`) operating directly on 5D tensors `(B, C, D, H, W)` with spectral depth $D$ and spatial window $H \times W$.
  - Spatial patch extraction utility with reflection/edge padding to handle image borders without border artifacts or eager memory exhaustion.
  - Dual 3D convolution stages with BatchNorm3d, ReLU, and Adaptive 3D Average Pooling (`(1, 1, 1)`) decoupling spatial/spectral input sizes from dense classifier weights (~24.2k parameters).
  - Verification on both raw 204-band spectral cubes and 15-band PCA-reduced cubes.
  - Comprehensive unit test suite (forward pass, backward gradient propagation, shape checks, CPU execution) and step-by-step verified learning notebook.
- **Day 9: PyTorch DataLoaders for Hyperspectral Tensor Batches**
  - Custom `HyperspectralPatchDataset` yielding 5D-compatible tensors `(1, D, H, W)` and zero-indexed `torch.long` targets.
  - Reusable `create_dataloaders(...)` generating reproducible train, validation, and test loaders with stratified sampling (70/15/15%).
  - Pre-padded $O(1)$ patch slicing preventing repeated padding allocations across 37,000+ training iterations.
  - Strict documentation of the spatial autocorrelation caveat in pixel-based random splits.
  - Full compatibility and verified forward pass with Day 8 `Hyperspectral3DCNN`.
- **Day 10: Baseline 3D-CNN Supervised Training & Checkpointing**
  - Complete training pipeline (`train_model`, `train_one_epoch`, `validate_one_epoch`) with AdamW and `nn.CrossEntropyLoss`.
  - Sample-weighted loss and multiclass accuracy aggregation handling incomplete final batches.
  - Atomic checkpoint persistence (`outputs/checkpoints/best_3d_cnn.pt`) with full model config, epoch, metric, and class-mapping metadata.
  - Hardened `.gitignore` and `.dockerignore` ensuring checkpoints and secrets remain excluded from version control.
  - Interactive learning notebook (`notebooks/day10_train_3d_cnn.ipynb`) executing a verified training run on Salinas.
- **Day 11: Rigorous 3D-CNN Evaluation, Error Analysis & GitHub Security**
  - Modular evaluation engine (`src/terraspectra/models/eval.py`) running with `torch.inference_mode()` on held-out test data (8,120 samples across 16 land-cover classes).
  - Honest classification metrics: Overall Accuracy (97.02%), Average Accuracy (98.38%), Macro Precision (98.70%), Macro Recall (98.38%), Macro F1 (98.49%), Weighted F1 (96.97%), and Cohen's Kappa ($\kappa = 0.9668$).
  - Per-class breakdown and error analysis identifying subtle agronomic confusions (e.g., `Vinyard_untrained` vs. `Grapes_untrained` due to matching canopy architecture and soil background).
  - Confusion matrix generation (raw counts and row-normalized percentages) with publication-quality heatmaps.
  - Checkpoint reload and forward-pass reproducibility verification against bit-level drift.
  - Unit tests (`tests/test_evaluation.py`) validating zero-division safety, incomplete final batch handling, weight immutability, and deterministic reproducibility.
  - Mandatory GitHub security audit ensuring no checkpoints, large datasets, or credentials are leaked.

---

## Project Structure

```text
Tetraspectra/
├── .gitignore
├── .dockerignore
├── .env.example
├── pyproject.toml
├── requirements.txt
├── README.md
├── src/
│   └── terraspectra/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loader.py        # Dataset loading, validation, downloading
│       │   ├── exploration.py   # Spectral inspection, signatures, false-color
│       │   ├── patches.py       # Spatial patch extraction & reflection padding
│       │   └── dataset.py       # PyTorch Dataset, stratified splits & DataLoaders
│       ├── models/
│       │   ├── __init__.py
│       │   ├── cnn3d.py         # Hyperspectral 3D-CNN architecture & config
│       │   ├── train.py         # Training loop, validation & checkpointing
│       │   └── eval.py          # Rigorous metrics, error analysis & confusion matrix
│       ├── processing/
│       │   ├── __init__.py
│       │   ├── pca.py           # PCA dimensionality reduction
│       │   └── scaling.py       # Normalization and feature scaling
│       └── utils/
│           ├── __init__.py
│           └── io.py
├── scripts/
│   ├── run_day2_ingest.py
│   ├── run_day3_explore.py
│   ├── run_day4_pca.py
│   ├── run_day5_scale.py
│   └── build_day08_notebook.py
├── tests/
│   ├── test_loader.py
│   ├── test_exploration.py
│   ├── test_pca.py
│   ├── test_scaling.py
│   ├── test_cnn3d.py
│   ├── test_patches.py
│   ├── test_dataset.py
│   ├── test_train.py
│   └── test_evaluation.py
├── notebooks/
│   ├── day02_data_loading.ipynb
│   ├── day03_data_exploration.ipynb
│   ├── day04_pca.ipynb
│   ├── day05_normalization.ipynb
│   ├── day08_3d_cnn.ipynb
│   ├── day09_dataloaders.ipynb
│   ├── day10_train_3d_cnn.ipynb
│   └── day11_model_evaluation.ipynb
├── data/
│   ├── raw/                     # Raw .mat files (git-ignored)
│   ├── processed/               # PCA & scaled outputs (git-ignored)
│   └── test_fixtures/           # Synthetic fixtures for automated tests
├── outputs/
│   ├── checkpoints/             # Best model checkpoints (git-ignored)
│   ├── figures/                 # Generated plots and false-color composites
│   └── reports/                 # Exploration and variance reports
└── frontend/                    # Vite + React + TypeScript + Deck.gl
```

---

## Setup & Quickstart

### 1. Python Environment Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Interactive Learning Notebooks

Explore each day's concepts and step-by-step verified outputs:

- [`notebooks/day02_data_loading.ipynb`](notebooks/day02_data_loading.ipynb): Ingesting and validating Salinas `.mat` data.
- [`notebooks/day03_data_exploration.ipynb`](notebooks/day03_data_exploration.ipynb): Numerical health audits, spectral curves, and false-color compositing.
- [`notebooks/day04_pca.ipynb`](notebooks/day04_pca.ipynb): Hughes phenomenon, PCA reduction, scree plots, and variance analysis.
- [`notebooks/day05_normalization.ipynb`](notebooks/day05_normalization.ipynb): Standardization vs. Min-Max scaling, zero-variance safeguards, and model persistence.
- [`notebooks/day08_3d_cnn.ipynb`](notebooks/day08_3d_cnn.ipynb): 3D-CNN architecture, 5D tensor layouts, central-pixel patch extraction with reflection padding, parameter counting, and PyTorch backward pass verification.
- [`notebooks/day09_dataloaders.ipynb`](notebooks/day09_dataloaders.ipynb): HyperspectralPatchDataset, 5D tensor batches, stratified train/val/test splits, class-label remapping, and smoke test integration with Day 8 3D-CNN.
- [`notebooks/day10_train_3d_cnn.ipynb`](notebooks/day10_train_3d_cnn.ipynb): Supervised 3D-CNN training loop, AdamW optimizer, loss descent curves, atomic checkpoint saving, and reload verification.
- [`notebooks/day11_model_evaluation.ipynb`](notebooks/day11_model_evaluation.ipynb): Rigorous 3D-CNN evaluation on held-out test split, overall and per-class metrics, confusion matrix analysis, and agronomic error investigation.

To launch Jupyter Lab / Notebook:
```bash
jupyter lab notebooks/
```

### 3. Running Pipeline Scripts & Tests

```bash
# Day 2: Ingest Salinas dataset (verifies or auto-downloads to data/raw/)
python scripts/run_day2_ingest.py

# Day 3: Dataset exploration and false-color composite generation
python scripts/run_day3_explore.py

# Day 4: PCA dimensionality reduction (15 components)
python scripts/run_day4_pca.py --n-components 15

# Day 5: Feature normalization and scaling (standard or minmax)
python scripts/run_day5_scale.py --method standard
python scripts/run_day5_scale.py --method minmax

# Run Day 8, 9, 10, 11 deep learning test suites
pytest tests/test_cnn3d.py tests/test_patches.py tests/test_dataset.py tests/test_train.py tests/test_evaluation.py

# Run all 58 automated test suites across the repository
pytest
```

### 4. Running the React Frontend (Day 6 & Day 7)

```bash
cd frontend

# Install dependencies (React, Deck.gl, MapLibre GL)
npm install

# Optional: configure Mapbox token (copy template)
cp .env.example .env
# Edit .env and set VITE_MAPBOX_ACCESS_TOKEN if using custom Mapbox styles

# Launch Vite development server
npm run dev
```

Navigate to `http://localhost:5173` to explore:
- **Geographical Base Map (Day 7)**: True Web Mercator (`EPSG:3857`) basemap using MapLibre GL + Deck.gl `MapboxOverlay`. Includes interactive zoom/pan, HUD coordinates readout, reset view button, and basemap style switcher (CARTO Dark Matter, CARTO Positron, OpenStreetMap).
- **Orthographic Raster Scene (Day 6)**: Raw 2D Cartesian array viewer (`0..216`, `0..511`) displaying Ground-Truth classification, False-Color composite, and PCA-1 intensity layers.

For in-depth architectural details, see [`docs/day07_basemap.md`](docs/day07_basemap.md).


