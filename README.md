# TerraSpectra 🛰️

**Hyperspectral Data Engineering, Preprocessing, and Interactive Visualization Platform**

TerraSpectra is an end-to-end hyperspectral remote sensing platform. This repository implements Days 1–6 of the 30-day roadmap, focusing on rigorous hyperspectral ingestion, validation, exploratory analysis, PCA dimensionality reduction, feature scaling, and a high-performance Deck.gl interactive visualization scaffold.

---

## Roadmap Scope (Days 1–6)

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

---

## Project Structure

```text
Tetraspectra/
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── README.md
├── src/
│   └── terraspectra/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loader.py        # Dataset loading, validation, downloading
│       │   └── exploration.py   # Spectral inspection, signatures, false-color
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
│   └── run_day5_scale.py
├── tests/
│   ├── test_loader.py
│   ├── test_exploration.py
│   ├── test_pca.py
│   └── test_scaling.py
├── data/
│   ├── raw/                     # Raw .mat files (git-ignored)
│   ├── processed/               # PCA & scaled outputs (git-ignored)
│   └── test_fixtures/           # Synthetic fixtures for automated tests
├── outputs/
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

### 2. Running Pipeline Steps

```bash
# Day 2: Ingest Salinas dataset (downloads or verifies local data)
python scripts/run_day2_ingest.py

# Day 3: Dataset exploration and false-color composite generation
python scripts/run_day3_explore.py

# Day 4: PCA dimensionality reduction
python scripts/run_day4_pca.py --n-components 15

# Day 5: Feature normalization and scaling
python scripts/run_day5_scale.py --method standard

# Run all test suites
pytest
```

### 3. Running the React Frontend (Day 6)

```bash
cd frontend
npm install
npm run dev
```
Navigate to `http://localhost:5173` to view the Deck.gl hyperspectral visualizer.
