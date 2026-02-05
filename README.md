# Credit Risk Prediction with Explainability

**TC2038 - Midterm 1**

**Team:**
- Santiago Arista Viramontes
- Diego Vergara Hernández
- José Leobardo Navarro Márquez

## Overview

End-to-end explainable AI system for credit risk scoring using SHAP on Home Credit Default Risk dataset.

**Implemented features:**
- **M1**: SHAP explanations with pathological case analysis (demonstrating XAI limits)
- **M2**: Formal robustness criterion with explanation stability guarantees
- **M3**: Fairness metrics with explicit synthetic group caveats
- **M4**: Governance framework with reproducibility guarantees (config hashing)

## Prerequisites

- Python 3.9 or higher
- `uv` package manager (recommended) or `pip`

## Dataset Setup

This project uses the **Home Credit Default Risk** dataset from Kaggle.

### Option 1: Download Manually (Recommended)

1. Visit the [Home Credit Default Risk competition page](https://www.kaggle.com/competitions/home-credit-default-risk/data)
2. Download `application_train.csv` and `application_test.csv`
3. Place them in the `data/` directory:
   ```
   midterm1/
   ├── data/
   │   ├── application_train.csv
   │   └── application_test.csv
   ```

### Option 2: Download Using Kaggle API

If you have Kaggle API credentials configured:

```bash
# Install Kaggle API
pip install kaggle

# Download dataset
kaggle competitions download -c home-credit-default-risk

# Unzip files to data directory
unzip home-credit-default-risk.zip -d data/
```

**Note:** The dataset files are not included in the repository due to their size (~150MB).

## Installation

### Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver.

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

### Using pip (Alternative)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

Once the dataset is in place and dependencies are installed:

```bash
# Make sure virtual environment is activated
python src/train.py
```

This will:
1. Load and preprocess the data
2. Train the credit risk model
3. Generate SHAP explanations
4. Run robustness tests
5. Compute fairness metrics
6. Create governance logs
7. Save all results to `experiments/`

**That's it!** Everything is automated in one script.

## Team Contributions

- **Santiago Arista Viramontes**: Model architecture, governance framework, integration
- **Diego Vergara Hernández**: Data preprocessing, robustness testing, regularization experiments
- **José Leobardo Navarro Márquez**: SHAP explainability, fairness metrics, evaluation

**Detailed contributions**: See [CONTRIBUTORS.md](CONTRIBUTORS.md) for a complete breakdown of individual contributions and module ownership.

**Note on Git History**: While commits may appear from a single account, work was distributed among all team members as documented in CONTRIBUTORS.md and code attribution comments.

## Environment Notes

This project has been tested on:
- macOS (Apple Silicon & Intel)
- Linux (Ubuntu 20.04+)
- Windows 10/11

All experiments use deterministic random seeds for reproducibility.
