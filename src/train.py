"""
Training script for credit risk model with SHAP explanations.

END-TO-END XAI PIPELINE: This script implements a complete XAI system from
data loading through model training, post-hoc explanations, robustness testing,
fairness evaluation, and governance.

Team: Santiago Arista Viramontes, Diego Vergara Hernández, José Leobardo Navarro Márquez
Integration and orchestration by: Santiago Arista Viramontes
"""

import sys
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_loader import CreditDataLoader
from src.models.credit_model import CreditRiskModel
from src.models.baseline_benchmarks import BaselineBenchmarks
from src.explainability.shap_explainer import SHAPExplainer
from src.robustness.robustness_tests import (
    RobustnessEvaluator, RegularizationExperiments, ExplanationStabilityEvaluator
)
from src.fairness.fairness_metrics import FairnessEvaluator
from src.governance.monitoring import GovernanceLogger, ModelMonitor, AuditTrail
from src.utils.reproducibility import create_experiment_config


def main():
    """Main training pipeline implementing end-to-end XAI system."""
    print("="*70)
    print("CREDIT RISK XAI PIPELINE - END-TO-END EXPLAINABLE AI SYSTEM")
    print("="*70)
    
    # ============================================================
    # STEP 0: Reproducibility Setup
    # ============================================================
    print("\n[0/9] Setting up reproducibility guarantees...")
    
    config = create_experiment_config(
        sample_size=10000,
        model_type="lightgbm",
        random_state=42,
        regularization='l2',
        C=1.0
    )
    
    config.print_reproducibility_info()
    config.save_config("experiments")
    
    # Extract config values
    SAMPLE_SIZE = config.config['sample_size']
    MODEL_TYPE = config.config['model_type']
    RANDOM_STATE = config.config['random_state']
    
    # Create experiments directory
    Path("experiments").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    
    # ============================================================
    # STEP 1: Load and prepare data
    # ============================================================
    print("\n[1/9] Loading and preprocessing data...")
    loader = CreditDataLoader()
    X, y, feature_names = loader.load_and_prepare(sample_size=SAMPLE_SIZE)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
    )
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Val set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # ============================================================
    # STEP 2: Baseline Benchmarks (From Class)
    # ============================================================
    print("\n[2/9] Training baseline benchmarks for comparison...")
    
    benchmarks = BaselineBenchmarks(random_state=RANDOM_STATE)
    baseline_results = benchmarks.train_all_baselines(X_train, y_train, X_test, y_test)
    baseline_results.to_csv("experiments/baseline_benchmarks.csv", index=False)
    
    # ============================================================
    # STEP 3: Train sophisticated model
    # ============================================================
    print(f"\n[3/9] Training sophisticated {MODEL_TYPE} model...")
    model = CreditRiskModel(model_type=MODEL_TYPE, random_state=RANDOM_STATE)
    model.build_model(regularization='l2', C=1.0)
    model.train(X_train, y_train, feature_names)
    
    # Evaluate
    print("\n--- Training Set Performance ---")
    model.print_evaluation(X_train, y_train, "Training Set")
    
    print("\n--- Validation Set Performance ---")
    model.print_evaluation(X_val, y_val, "Validation Set")
    
    print("\n--- Test Set Performance ---")
    metrics_test = model.print_evaluation(X_test, y_test, "Test Set")
    
    # Compare with baselines
    improvements = benchmarks.compare_with_model(model, X_test, y_test, baseline_results)
    
    # Get predictions for fairness evaluation
    y_pred_test = model.model.predict_proba(X_test)[:, 1]
    
    # Save model
    model.save("models/credit_model.pkl")
    
    # ============================================================
    # STEP 4: Generate SHAP explanations (M1)
    # ============================================================
    print("\n[4/9] Generating SHAP explanations (M1 - XAI Feature)...")
    explainer = SHAPExplainer(model, X_train[:500], feature_names)
    
    # Explain individual predictions
    print("\n--- Example Explanation 1 (Default Case) ---")
    # Find a default case
    default_idx = np.where(y_test == 1)[0][0]
    explanation_text = explainer.explain_prediction_text(X_test, default_idx)
    print(explanation_text)
    
    print("\n--- Example Explanation 2 (Non-Default Case) ---")
    # Find a non-default case
    non_default_idx = np.where(y_test == 0)[0][0]
    explanation_text = explainer.explain_prediction_text(X_test, non_default_idx)
    print(explanation_text)
    
    # Create visualizations
    print("\nGenerating waterfall plot...")
    explainer.plot_waterfall(X_test, default_idx, "experiments/shap_waterfall_default.png")
    
    print("\nGenerating summary plot (this may take a moment)...")
    explainer.plot_summary(X_test[:200], max_display=15, save_path="experiments/shap_summary.png")
    
    # Global feature importance
    print("\nCalculating global feature importance...")
    importance_df = explainer.get_global_importance(X_test[:200])
    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10).to_string(index=False))
    importance_df.to_csv("experiments/feature_importance.csv", index=False)
    
    # PATHOLOGICAL CASE ANALYSIS - Demonstrating SHAP Limits
    print("\n--- Pathological Case Analysis (Demonstrating XAI Limits) ---")
    pathological_cases = explainer.analyze_pathological_cases(
        X_test[:500], y_test[:500],
        save_path="experiments/pathological_cases.json"
    )
    
    # ============================================================
    # STEP 5: Regularization experiments (M2)
    # ============================================================
    print("\n[5/9] Running regularization experiments (M2)...")
    reg_experiments = RegularizationExperiments(
        X_train, y_train, X_val, y_val, feature_names
    )
    
    reg_results = reg_experiments.compare_regularization(model_type=MODEL_TYPE)
    print("\nRegularization Results:")
    print(reg_results.to_string(index=False))
    reg_results.to_csv("experiments/regularization_results.csv", index=False)
    
    reg_experiments.plot_regularization_comparison(
        reg_results, 
        save_path="experiments/regularization_comparison.png"
    )
    
    # ============================================================
    # STEP 5: Robustness testing (M2)
    # ============================================================
    print("\n[5/6] Running robustness tests (M2)...")
    robustness_eval = RobustnessEvaluator(
        model, X_train, y_train, X_test, y_test, feature_names
    )
    
    # Test noise robustness
    print("\n--- Noise Injection Test ---")
    noise_results = robustness_eval.test_noise_injection()
    noise_results.to_csv("experiments/robustness_noise.csv", index=False)
    robustness_eval.plot_robustness_results(
        noise_results, 
        perturbation_col='noise_std',
        save_path="experiments/robustness_noise.png"
    )
    
    # Test feature dropout
    print("\n--- Feature Dropout Test ---")
    dropout_results = robustness_eval.test_feature_dropout()
    dropout_results.to_csv("experiments/robustness_dropout.csv", index=False)
    robustness_eval.plot_robustness_results(
        dropout_results,
        perturbation_col='dropout_rate',
        save_path="experiments/robustness_dropout.png"
    )
    
    # Test distribution shift
    print("\n--- Distribution Shift Test ---")
    shift_results = robustness_eval.test_distribution_shift(shift_magnitude=0.5)
    
    # ============================================================
    # STEP 7: Fairness evaluation (M3)
    # ============================================================
    print("\n[7/9] Evaluating fairness metrics (M3)...")
    fairness_eval = FairnessEvaluator()
    fairness_results = fairness_eval.evaluate_all(
        y_test, y_pred_test, X_test
    )
    fairness_eval.save_results("experiments/fairness_metrics.csv")
    
    # ============================================================
    # STEP 8: Governance & monitoring (M4)
    # ============================================================
    print("\n[8/9] Setting up governance and monitoring (M4)...")
    
    # Initialize governance
    gov_logger = GovernanceLogger("experiments/governance_logs")
    monitor = ModelMonitor()
    audit = AuditTrail(model_version="1.0-midterm1")
    
    # Log sample predictions with governance
    print("Logging predictions for audit trail...")
    import time as time_module
    for i in range(min(100, len(X_test))):
        start = time_module.time()
        pred = model.model.predict_proba(X_test[i:i+1])[:, 1][0]
        latency = (time_module.time() - start) * 1000  # ms
        
        # Log with transaction ID
        tx_id = f"TX_{int(time_module.time()*1000)}_{i}"
        gov_logger.log_prediction(
            prediction_id=tx_id,
            features=X_test[i],
            prediction=pred,
            metadata={'sample_index': i}
        )
        monitor.record_inference(pred, latency)
    
    # Generate monitoring report
    monitor.save_report("experiments/monitoring_report.txt")
    
    # Generate audit report
    model_metrics = {
        'roc_auc': metrics_test['roc_auc'],
        'accuracy': metrics_test['accuracy'],
        'precision': metrics_test['precision'],
        'recall': metrics_test['recall'],
        'f1_score': metrics_test['f1'],
        'ece': metrics_test['ece']
    }
    
    audit.generate_audit_report(
        model_metrics=model_metrics,
        fairness_metrics=fairness_results,
        output_path="experiments/audit_report.txt"
    )
    
    # ============================================================
    # STEP 9: Summary report
    # ============================================================
    print("\n[9/9] Generating comprehensive summary...")
    
    summary = f"""
{'='*70}
COMPLETE PROJECT - ALL MILESTONES (M1-M4)
{'='*70}

MODEL CONFIGURATION
-------------------
Model Type: {MODEL_TYPE}
Sample Size: {SAMPLE_SIZE}
Features: {len(feature_names)}
Random State: {RANDOM_STATE}

TEST SET PERFORMANCE
--------------------
ROC-AUC: {metrics_test['roc_auc']:.4f}
Accuracy: {metrics_test['accuracy']:.4f}
Precision: {metrics_test['precision']:.4f}
Recall: {metrics_test['recall']:.4f}
F1-Score: {metrics_test['f1']:.4f}
Expected Calibration Error: {metrics_test['ece']:.4f}

M1: EXPLAINABILITY
------------------
✓ SHAP explanations implemented
✓ Individual instance explanations generated
✓ Global feature importance computed
✓ Visualizations saved to experiments/

TOP 5 MOST IMPORTANT FEATURES
{importance_df.head(5).to_string(index=False)}

M2: ROBUSTNESS
--------------
Noise Injection (std=0.2):
  - ROC-AUC degradation: {noise_results[noise_results['noise_std']==0.2]['roc_auc'].values[0] - noise_results[noise_results['noise_std']==0.0]['roc_auc'].values[0]:.4f}

Feature Dropout (30% missing):
  - ROC-AUC degradation: {dropout_results[dropout_results['dropout_rate']==0.3]['roc_auc'].values[0] - dropout_results[dropout_results['dropout_rate']==0.0]['roc_auc'].values[0]:.4f}

Distribution Shift:
  - ROC-AUC drop: {shift_results['roc_auc_drop']:.4f}
  - ECE increase: {shift_results['ece_increase']:.4f}

Regularization (Best):
  - {reg_results.loc[reg_results['val_auc'].idxmax(), 'config']}
  - Val AUC: {reg_results['val_auc'].max():.4f}

M3: FAIRNESS
------------
✓ Disparate Impact Ratio: {fairness_results['disparate_impact']['disparate_impact_ratio']:.4f}
✓ Passes 80% Rule: {fairness_results['disparate_impact']['passes_80_rule']}
✓ TPR Disparity: {fairness_results['equalized_odds']['tpr_disparity']:.4f}
✓ FPR Disparity: {fairness_results['equalized_odds']['fpr_disparity']:.4f}
✓ Demographic Parity Diff: {fairness_results['demographic_parity']['max_difference']:.4f}

M4: GOVERNANCE
--------------
✓ Audit trail: 100 predictions logged with transaction IDs
✓ Mean latency: {monitor.get_statistics()['mean_latency_ms']:.2f}ms
✓ Audit report generated with compliance checks
✓ Model version: 1.0-midterm1

OUTPUT FILES (15+ total)
------------------------
✓ models/credit_model.pkl
✓ experiments/shap_waterfall_default.png
✓ experiments/shap_summary.png
✓ experiments/feature_importance.csv
✓ experiments/regularization_results.csv
✓ experiments/regularization_comparison.png
✓ experiments/robustness_noise.csv
✓ experiments/robustness_noise.png
✓ experiments/robustness_dropout.csv
✓ experiments/robustness_dropout.png
✓ experiments/fairness_metrics.csv
✓ experiments/governance_logs/*.jsonl
✓ experiments/monitoring_report.txt
✓ experiments/audit_report.txt
✓ experiments/summary_report.txt

{'='*70}
PROJECT COMPLETE - ALL MILESTONES IMPLEMENTED
{'='*70}

M1: Explainability - IMPLEMENTED
M2: Robustness - IMPLEMENTED  
M3: Fairness - IMPLEMENTED
M4: Governance - IMPLEMENTED
{'='*70}
"""
    
    print(summary)
    
    # Save summary
    with open("experiments/summary_report.txt", "w") as f:
        f.write(summary)
    
    print("\nPipeline complete! Check the experiments/ folder for all outputs.")


if __name__ == "__main__":
    main()
