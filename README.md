# Explainable Credit Risk Prediction with XAI

**TC2038 - Analysis and Design of Advanced Algorithms - Midterm 1**

Authors: Santiago Arista Viramontes, Diego Vergara Hernández, José Leobardo Navarro Márquez

## Project Overview

This project implements an explainable credit risk assessment system using the Home Credit Default Risk dataset. The system integrates:

- **M1**: SHAP-based explainable AI for credit decisions
- **M2**: Robustness testing with regularization and perturbation analysis
- **M3**: Fairness auditing framework (planned)
- **M4**: Governance and deployment readiness (planned)

## Repository Structure

```
midterm1/
├── data/               # Dataset (download from Kaggle)
├── notebooks/          # Jupyter notebooks for exploration
├── src/
│   ├── data/          # Data loading and preprocessing
│   ├── models/        # ML models (LightGBM, Logistic Regression)
│   ├── explainability/ # SHAP implementation
│   ├── robustness/    # Regularization and stress testing
│   └── utils/         # Helper functions
├── experiments/       # Experiment results and plots
├── paper/            # LaTeX source for midterm report
└── requirements.txt  # Python dependencies
```

## Setup Instructions

1. Download data from Kaggle: https://www.kaggle.com/competitions/home-credit-default-risk/data
2. Place CSV files in `data/` directory
3. Install dependencies: `pip install -r requirements.txt`
4. Run notebooks in order or execute scripts

## Key Features

- Supervised learning models (Logistic Regression, LightGBM)
- SHAP explanations for individual predictions
- Feature importance analysis
- Robustness testing under noise and perturbations
- Regularization experiments (L1/L2)
- Calibrated probability outputs

## Running the Pipeline

```bash
# Train baseline model
python src/train.py

# Generate explanations
python src/explain.py

# Run robustness tests
python src/robustness_tests.py
```
