"""
Robustness testing and regularization experiments (M2).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split, cross_val_score
from src.models.credit_model import CreditRiskModel


class RobustnessEvaluator:
    """Test model robustness under various stress conditions."""
    
    def __init__(self, model, X_train: np.ndarray, y_train: np.ndarray, 
                 X_test: np.ndarray, y_test: np.ndarray, feature_names: list):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names
    
    def test_noise_injection(self, noise_levels: List[float] = [0.0, 0.05, 0.1, 0.2, 0.3]) -> pd.DataFrame:
        """
        Test model robustness under Gaussian noise.
        
        Args:
            noise_levels: List of noise standard deviations
        """
        results = []
        
        print("Testing robustness under noise injection...")
        
        for noise_std in noise_levels:
            # Add Gaussian noise
            X_noisy = self.X_test + np.random.normal(0, noise_std, self.X_test.shape)
            
            # Evaluate
            metrics = self.model.evaluate(X_noisy, self.y_test)
            
            results.append({
                'noise_std': noise_std,
                'accuracy': metrics['accuracy'],
                'roc_auc': metrics['roc_auc'],
                'f1': metrics['f1'],
                'ece': metrics['ece']
            })
            
            print(f"Noise std={noise_std:.2f}: ROC-AUC={metrics['roc_auc']:.4f}, "
                  f"Accuracy={metrics['accuracy']:.4f}")
        
        return pd.DataFrame(results)
    
    def test_feature_dropout(self, dropout_rates: List[float] = [0.0, 0.1, 0.2, 0.3, 0.5]) -> pd.DataFrame:
        """
        Test robustness when features are randomly dropped (set to 0).
        """
        results = []
        
        print("Testing robustness under feature dropout...")
        
        for dropout_rate in dropout_rates:
            # Create dropout mask
            mask = np.random.binomial(1, 1 - dropout_rate, self.X_test.shape)
            X_dropout = self.X_test * mask
            
            # Evaluate
            metrics = self.model.evaluate(X_dropout, self.y_test)
            
            results.append({
                'dropout_rate': dropout_rate,
                'accuracy': metrics['accuracy'],
                'roc_auc': metrics['roc_auc'],
                'f1': metrics['f1'],
                'ece': metrics['ece']
            })
            
            print(f"Dropout={dropout_rate:.1%}: ROC-AUC={metrics['roc_auc']:.4f}, "
                  f"Accuracy={metrics['accuracy']:.4f}")
        
        return pd.DataFrame(results)
    
    def test_distribution_shift(self, shift_magnitude: float = 0.5) -> Dict:
        """
        Test performance under distribution shift by sampling from different quantiles.
        """
        print(f"Testing robustness under distribution shift (magnitude={shift_magnitude})...")
        
        # Shift data by adding a bias to all features
        X_shifted = self.X_test + shift_magnitude
        
        metrics_original = self.model.evaluate(self.X_test, self.y_test)
        metrics_shifted = self.model.evaluate(X_shifted, self.y_test)
        
        degradation = {
            'roc_auc_drop': metrics_original['roc_auc'] - metrics_shifted['roc_auc'],
            'accuracy_drop': metrics_original['accuracy'] - metrics_shifted['accuracy'],
            'ece_increase': metrics_shifted['ece'] - metrics_original['ece']
        }
        
        print(f"Performance degradation under shift:")
        print(f"  ROC-AUC drop: {degradation['roc_auc_drop']:.4f}")
        print(f"  Accuracy drop: {degradation['accuracy_drop']:.4f}")
        print(f"  ECE increase: {degradation['ece_increase']:.4f}")
        
        return degradation
    
    def plot_robustness_results(self, results_df: pd.DataFrame, 
                                 perturbation_col: str = 'noise_std',
                                 save_path: str = None):
        """Plot robustness test results."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        metrics = ['accuracy', 'roc_auc', 'f1']
        titles = ['Accuracy', 'ROC-AUC', 'F1-Score']
        
        for ax, metric, title in zip(axes, metrics, titles):
            ax.plot(results_df[perturbation_col], results_df[metric], 
                   marker='o', linewidth=2, markersize=8)
            ax.set_xlabel(perturbation_col.replace('_', ' ').title())
            ax.set_ylabel(title)
            ax.set_title(f'{title} vs Perturbation')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()


class RegularizationExperiments:
    """Compare different regularization strategies."""
    
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray, feature_names: list):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.feature_names = feature_names
    
    def compare_regularization(self, model_type: str = "logistic") -> pd.DataFrame:
        """
        Compare L1, L2, and no regularization.
        """
        results = []
        
        # Regularization configurations to test
        configs = [
            {'name': 'No Reg (C=1e10)', 'penalty': 'l2', 'C': 1e10},
            {'name': 'L2 Weak (C=10)', 'penalty': 'l2', 'C': 10.0},
            {'name': 'L2 Medium (C=1)', 'penalty': 'l2', 'C': 1.0},
            {'name': 'L2 Strong (C=0.1)', 'penalty': 'l2', 'C': 0.1},
            {'name': 'L1 Medium (C=1)', 'penalty': 'l1', 'C': 1.0},
            {'name': 'L1 Strong (C=0.1)', 'penalty': 'l1', 'C': 0.1},
        ]
        
        print("Comparing regularization strategies...")
        
        for config in configs:
            print(f"\nTesting {config['name']}...")
            
            # Train model
            model = CreditRiskModel(model_type=model_type)
            model.build_model(regularization=config['penalty'], C=config['C'])
            model.train(self.X_train, self.y_train, self.feature_names)
            
            # Evaluate on validation set
            metrics_train = model.evaluate(self.X_train, self.y_train)
            metrics_val = model.evaluate(self.X_val, self.y_val)
            
            # Calculate overfitting
            overfit = metrics_train['roc_auc'] - metrics_val['roc_auc']
            
            results.append({
                'config': config['name'],
                'penalty': config['penalty'],
                'C': config['C'],
                'train_auc': metrics_train['roc_auc'],
                'val_auc': metrics_val['roc_auc'],
                'val_accuracy': metrics_val['accuracy'],
                'val_f1': metrics_val['f1'],
                'val_ece': metrics_val['ece'],
                'overfitting': overfit
            })
            
            print(f"  Train AUC: {metrics_train['roc_auc']:.4f}")
            print(f"  Val AUC: {metrics_val['roc_auc']:.4f}")
            print(f"  Overfitting: {overfit:.4f}")
        
        df_results = pd.DataFrame(results)
        return df_results
    
    def plot_regularization_comparison(self, results_df: pd.DataFrame, save_path: str = None):
        """Plot regularization comparison results."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Train vs Val AUC
        x = np.arange(len(results_df))
        width = 0.35
        
        axes[0].bar(x - width/2, results_df['train_auc'], width, label='Train AUC', alpha=0.8)
        axes[0].bar(x + width/2, results_df['val_auc'], width, label='Val AUC', alpha=0.8)
        axes[0].set_xlabel('Regularization Config')
        axes[0].set_ylabel('ROC-AUC')
        axes[0].set_title('Train vs Validation AUC')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(results_df['config'], rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Overfitting
        axes[1].bar(x, results_df['overfitting'], alpha=0.8, color='coral')
        axes[1].set_xlabel('Regularization Config')
        axes[1].set_ylabel('Overfitting (Train AUC - Val AUC)')
        axes[1].set_title('Overfitting Analysis')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(results_df['config'], rotation=45, ha='right')
        axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()


if __name__ == "__main__":
    # Test robustness evaluator
    from src.data.data_loader import CreditDataLoader
    
    loader = CreditDataLoader()
    X, y, features = loader.load_and_prepare(sample_size=5000)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = CreditRiskModel(model_type="logistic")
    model.train(X_train, y_train, features)
    
    evaluator = RobustnessEvaluator(model, X_train, y_train, X_test, y_test, features)
    results = evaluator.test_noise_injection()
    evaluator.plot_robustness_results(results, save_path="experiments/robustness_noise.png")
