"""
Baseline benchmark models for comparison.

Implements simple baselines from class to establish performance floor.

Primary contributor: Santiago Arista Viramontes
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


class BaselineBenchmarks:
    """
    Baseline models for credit risk prediction.
    
    Establishes performance floor following class guidelines:
    1. Random baseline (stratified)
    2. Majority class baseline
    3. Simple decision tree (depth=3, interpretable)
    4. Unregularized logistic regression
    
    These baselines provide context for evaluating sophisticated models.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.baselines = {}
        
    def train_all_baselines(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """
        Train all baseline models and return comparison table.
        
        Returns DataFrame with performance metrics for each baseline.
        """
        print("\n" + "="*70)
        print("BASELINE BENCHMARK COMPARISON (From Class)")
        print("="*70)
        print("\nEstablishing performance floor with simple baselines...")
        print("These provide context for evaluating our sophisticated model.\n")
        
        results = []
        
        # Baseline 1: Random Stratified
        print("[1/4] Random Stratified Classifier...")
        random_clf = DummyClassifier(strategy='stratified', random_state=self.random_state)
        random_clf.fit(X_train, y_train)
        self.baselines['random'] = random_clf
        
        y_pred_random = random_clf.predict(X_test)
        y_prob_random = random_clf.predict_proba(X_test)[:, 1]
        
        results.append({
            'Baseline': 'Random (Stratified)',
            'Type': 'Dummy',
            'ROC-AUC': roc_auc_score(y_test, y_prob_random),
            'Accuracy': accuracy_score(y_test, y_pred_random),
            'F1-Score': f1_score(y_test, y_pred_random),
            'Description': 'Samples randomly based on class distribution'
        })
        print(f"  ROC-AUC: {results[-1]['ROC-AUC']:.4f}")
        
        # Baseline 2: Majority Class
        print("\n[2/4] Majority Class Classifier...")
        majority_clf = DummyClassifier(strategy='most_frequent', random_state=self.random_state)
        majority_clf.fit(X_train, y_train)
        self.baselines['majority'] = majority_clf
        
        y_pred_majority = majority_clf.predict(X_test)
        y_prob_majority = majority_clf.predict_proba(X_test)[:, 1]
        
        results.append({
            'Baseline': 'Majority Class',
            'Type': 'Dummy',
            'ROC-AUC': roc_auc_score(y_test, y_prob_majority),
            'Accuracy': accuracy_score(y_test, y_pred_majority),
            'F1-Score': f1_score(y_test, y_pred_majority),
            'Description': 'Always predicts most frequent class'
        })
        print(f"  ROC-AUC: {results[-1]['ROC-AUC']:.4f}")
        
        # Baseline 3: Simple Decision Tree (Interpretable)
        print("\n[3/4] Simple Decision Tree (depth=3)...")
        tree_clf = DecisionTreeClassifier(
            max_depth=3,  # Shallow for interpretability
            min_samples_split=100,
            min_samples_leaf=50,
            random_state=self.random_state,
            class_weight='balanced'
        )
        tree_clf.fit(X_train, y_train)
        self.baselines['tree'] = tree_clf
        
        y_pred_tree = tree_clf.predict(X_test)
        y_prob_tree = tree_clf.predict_proba(X_test)[:, 1]
        
        results.append({
            'Baseline': 'Decision Tree',
            'Type': 'Intrinsically Interpretable',
            'ROC-AUC': roc_auc_score(y_test, y_prob_tree),
            'Accuracy': accuracy_score(y_test, y_pred_tree),
            'F1-Score': f1_score(y_test, y_pred_tree),
            'Description': 'Shallow tree (depth=3), fully interpretable'
        })
        print(f"  ROC-AUC: {results[-1]['ROC-AUC']:.4f}")
        
        # Baseline 4: Unregularized Logistic Regression
        print("\n[4/4] Unregularized Logistic Regression...")
        lr_clf = LogisticRegression(
            C=1e10,  # Effectively no regularization
            max_iter=1000,
            random_state=self.random_state,
            class_weight='balanced'
        )
        lr_clf.fit(X_train, y_train)
        self.baselines['logistic'] = lr_clf
        
        y_pred_lr = lr_clf.predict(X_test)
        y_prob_lr = lr_clf.predict_proba(X_test)[:, 1]
        
        results.append({
            'Baseline': 'Logistic Reg (No Reg)',
            'Type': 'Linear',
            'ROC-AUC': roc_auc_score(y_test, y_prob_lr),
            'Accuracy': accuracy_score(y_test, y_pred_lr),
            'F1-Score': f1_score(y_test, y_pred_lr),
            'Description': 'Logistic regression without regularization'
        })
        print(f"  ROC-AUC: {results[-1]['ROC-AUC']:.4f}")
        
        df_results = pd.DataFrame(results)
        
        print("\n" + "="*70)
        print("BASELINE SUMMARY")
        print("="*70)
        print(df_results[['Baseline', 'ROC-AUC', 'Accuracy', 'F1-Score']].to_string(index=False))
        print("\n→ Our sophisticated model must significantly outperform these baselines")
        print("→ Decision Tree (depth=3) represents intrinsic interpretability alternative")
        print("="*70 + "\n")
        
        return df_results
    
    def compare_with_model(self, model, X_test: np.ndarray, y_test: np.ndarray,
                          baseline_results: pd.DataFrame) -> Dict[str, float]:
        """
        Compare sophisticated model against baselines.
        
        Returns improvement metrics showing how much better the model is.
        """
        print("\n" + "="*70)
        print("MODEL vs BASELINE COMPARISON")
        print("="*70)
        
        # Evaluate sophisticated model
        y_prob_model = model.model.predict_proba(X_test)[:, 1]
        y_pred_model = model.model.predict(X_test)
        
        model_metrics = {
            'ROC-AUC': roc_auc_score(y_test, y_prob_model),
            'Accuracy': accuracy_score(y_test, y_pred_model),
            'F1-Score': f1_score(y_test, y_pred_model)
        }
        
        print(f"\nSophisticated Model Performance:")
        print(f"  ROC-AUC:  {model_metrics['ROC-AUC']:.4f}")
        print(f"  Accuracy: {model_metrics['Accuracy']:.4f}")
        print(f"  F1-Score: {model_metrics['F1-Score']:.4f}")
        
        # Calculate improvements over each baseline
        print("\nImprovement over Baselines:")
        print("-" * 70)
        
        improvements = {}
        
        for _, row in baseline_results.iterrows():
            baseline_name = row['Baseline']
            
            auc_improvement = model_metrics['ROC-AUC'] - row['ROC-AUC']
            acc_improvement = model_metrics['Accuracy'] - row['Accuracy']
            f1_improvement = model_metrics['F1-Score'] - row['F1-Score']
            
            improvements[baseline_name] = {
                'AUC_improvement': auc_improvement,
                'Accuracy_improvement': acc_improvement,
                'F1_improvement': f1_improvement
            }
            
            print(f"\nvs {baseline_name}:")
            print(f"  ROC-AUC: +{auc_improvement:+.4f} ({auc_improvement/row['ROC-AUC']*100:+.1f}%)")
            print(f"  Accuracy: +{acc_improvement:+.4f}")
            print(f"  F1-Score: +{f1_improvement:+.4f}")
        
        # Key comparison: vs Decision Tree (intrinsic interpretability)
        tree_row = baseline_results[baseline_results['Baseline'] == 'Decision Tree'].iloc[0]
        tree_auc = tree_row['ROC-AUC']
        model_auc = model_metrics['ROC-AUC']
        
        print("\n" + "="*70)
        print("INTRINSIC INTERPRETABILITY vs POST-HOC XAI TRADE-OFF")
        print("="*70)
        print(f"\nDecision Tree (Intrinsically Interpretable): AUC = {tree_auc:.4f}")
        print(f"Our Model + SHAP (Post-hoc XAI):             AUC = {model_auc:.4f}")
        print(f"Performance Gain:                            +{(model_auc - tree_auc):.4f}")
        
        if model_auc > tree_auc + 0.02:
            print("\n→ JUSTIFICATION: Significant performance gain (+{:.1f}%) justifies".format(
                (model_auc - tree_auc)/tree_auc * 100))
            print("  using complex model with post-hoc SHAP explanations over")
            print("  intrinsically interpretable but less accurate decision tree.")
        else:
            print("\n WARNING: Small performance gain. Consider intrinsically")
            print("   interpretable model (decision tree) for regulatory contexts.")
        
        print("="*70 + "\n")
        
        return improvements


if __name__ == "__main__":
    # Test baseline benchmarks
    from src.data.data_loader import CreditDataLoader
    from sklearn.model_selection import train_test_split
    
    loader = CreditDataLoader()
    X, y, features = loader.load_and_prepare(sample_size=5000)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    benchmarks = BaselineBenchmarks()
    results = benchmarks.train_all_baselines(X_train, y_train, X_test, y_test)
