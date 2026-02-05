# Contributors

This project was developed as a team effort for TC2038 - Midterm 1.

## Team Members

### Santiago Arista Viramontes
- **Role**: Project Lead & Integration
- **Contributions**:
  - Model architecture and implementation ([models/credit_model.py](src/models/credit_model.py))
  - Governance framework and logging ([governance/monitoring.py](src/governance/monitoring.py))
  - Main training pipeline integration ([train.py](src/train.py))
  - Project structure and orchestration
  - Documentation and setup scripts

### Diego Vergara Hernández
- **Role**: Data Engineering & Robustness Testing
- **Contributions**:
  - Data loading and preprocessing ([data/data_loader.py](src/data/data_loader.py))
  - Robustness testing implementation ([robustness/robustness_tests.py](src/robustness/robustness_tests.py))
  - Regularization experiments (L1/L2)
  - Noise injection and dropout testing
  - Feature engineering support

### José Leobardo Navarro Márquez
- **Role**: Explainability & Fairness Analysis
- **Contributions**:
  - SHAP explainability implementation ([explainability/shap_explainer.py](src/explainability/shap_explainer.py))
  - Fairness metrics and analysis ([fairness/fairness_metrics.py](src/fairness/fairness_metrics.py))
  - Feature importance visualization
  - Demographic parity and equalized odds evaluation
  - Model evaluation and reporting

## Collaboration

All team members contributed to:
- Project design and architecture discussions
- Testing and debugging
- Documentation and code review
- LaTeX report writing
- Results analysis and interpretation

## Attribution in Code

Each module includes attribution comments at the top indicating primary contributors. This helps identify who led each component while acknowledging that all members contributed to the overall project success.

## Version Control

All changes are tracked through Git commits. While the repository may show commits from a single account, the actual work was distributed as documented above.
