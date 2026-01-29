# Quick Start Guide

## Prerequisites

1. Python 3.8 or higher
2. pip package manager
3. (Optional) LaTeX distribution for compiling the paper

## Setup Instructions

### 1. Download the Dataset

Download the Home Credit Default Risk dataset from Kaggle:
https://www.kaggle.com/competitions/home-credit-default-risk/data

You'll need these files:
- `application_train.csv` (main training data)
- `application_test.csv` (test data, optional)

Place them in a `data/` directory at the project root.

### 2. Install Dependencies

Run the setup script (macOS/Linux):
```bash
chmod +x setup.sh
./setup.sh
```

Or manually:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the Pipeline

Train the model and generate all results:
```bash
python src/train.py
```

This will:
- Load and preprocess the data
- Train credit risk models
- Generate SHAP explanations
- Run robustness tests
- Save all plots and results to `experiments/`

## Project Structure

```
midterm1/
├── data/                   # Place Kaggle dataset here
├── src/
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # ML models (logistic, LightGBM)
│   ├── explainability/    # SHAP implementation (M1)
│   ├── robustness/        # Robustness tests (M2)
│   └── train.py           # Main pipeline script
├── experiments/           # Generated plots and results
├── paper/
│   └── midterm1.tex       # LaTeX paper (both parts)
├── requirements.txt       # Python dependencies
└── README.md
```

## Running Individual Components

### Test Data Loading
```bash
python src/data/data_loader.py
```

### Test Model Training
```bash
python src/models/credit_model.py
```

### Test SHAP Explanations
```bash
python src/explainability/shap_explainer.py
```

### Test Robustness
```bash
python src/robustness/robustness_tests.py
```

## Compiling the Paper

The paper is located in `paper/midterm1.tex` and contains both Part I (Algorithmic Analysis) and Part II (Project Progress).

To compile:
```bash
cd paper
pdflatex midterm1.tex
pdflatex midterm1.tex  # Run twice for references
```

Or use your preferred LaTeX editor (Overleaf, TeXShop, etc.).

## Expected Outputs

After running the pipeline, you should see:

### In `experiments/`
- `shap_waterfall_default.png` - Individual explanation
- `shap_summary.png` - Global feature importance
- `feature_importance.csv` - Feature rankings
- `regularization_results.csv` - L1/L2 comparison
- `regularization_comparison.png` - Regularization plots
- `robustness_noise.csv` - Noise injection results
- `robustness_noise.png` - Noise robustness plot
- `robustness_dropout.csv` - Feature dropout results
- `robustness_dropout.png` - Dropout robustness plot
- `summary_report.txt` - Complete summary

### In `models/`
- `credit_model.pkl` - Trained model (saved)

### In `paper/`
- `midterm1.pdf` - Final submission document (after compiling)

## Customization

Edit configuration in `src/train.py`:
- `SAMPLE_SIZE`: Number of records to use (default: 10000)
- `MODEL_TYPE`: "logistic" or "lightgbm"
- `RANDOM_STATE`: Random seed for reproducibility

## Troubleshooting

### "Dataset not found" error
Make sure `data/application_train.csv` exists.

### Memory errors
Reduce `SAMPLE_SIZE` in `src/train.py`.

### SHAP taking too long
Reduce background sample size in `SHAPExplainer` (default: 500).

### LaTeX compilation errors
Make sure you have a complete LaTeX distribution (TeX Live, MiKTeX, or MacTeX).

## Next Steps for Midterm 2

- Implement M3: Fairness auditing metrics
- Implement M4: Governance and deployment
- Complete full repository with documentation
- Run comprehensive experiments on full dataset

## Contact

For questions or issues, contact the project team:
- Santiago Arista Viramontes (A01028372)
- Diego Vergara Hernández (A01425660)
- José Leobardo Navarro Márquez (A91541324)
