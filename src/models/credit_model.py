"""
Credit risk prediction models with calibration support.
"""

import numpy as np
from typing import Tuple, Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, 
    recall_score, f1_score, confusion_matrix, classification_report
)
from sklearn.calibration import calibration_curve
import lightgbm as lgb
import pickle


class CreditRiskModel:
    """Base class for credit risk models with explainability support."""
    
    def __init__(self, model_type: str = "logistic", random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.feature_names = None
        
    def build_model(self, regularization: str = 'l2', C: float = 1.0):
        """Build the predictive model."""
        if self.model_type == "logistic":
            self.model = LogisticRegression(
                penalty=regularization,
                C=C,
                solver='liblinear' if regularization == 'l1' else 'lbfgs',
                max_iter=1000,
                random_state=self.random_state,
                class_weight='balanced'
            )
        elif self.model_type == "lightgbm":
            self.model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.05,
                num_leaves=31,
                reg_alpha=C if regularization == 'l1' else 0.0,
                reg_lambda=C if regularization == 'l2' else 0.0,
                random_state=self.random_state,
                class_weight='balanced',
                verbose=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return self.model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: list = None):
        """Train the model."""
        if self.model is None:
            self.build_model()
        
        self.feature_names = feature_names
        
        print(f"Training {self.model_type} model...")
        self.model.fit(X_train, y_train)
        print("Training complete!")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities (for default class)."""
        return self.model.predict_proba(X)[:, 1]
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Comprehensive model evaluation."""
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'roc_auc': roc_auc_score(y, y_proba),
            'confusion_matrix': confusion_matrix(y, y_pred).tolist()
        }
        
        # Calculate calibration error (ECE)
        prob_true, prob_pred = calibration_curve(y, y_proba, n_bins=10)
        ece = np.mean(np.abs(prob_true - prob_pred))
        metrics['ece'] = ece
        
        return metrics
    
    def print_evaluation(self, X: np.ndarray, y: np.ndarray, dataset_name: str = ""):
        """Print evaluation metrics."""
        metrics = self.evaluate(X, y)
        
        print(f"\n{'='*50}")
        print(f"Evaluation Results - {dataset_name}")
        print(f"{'='*50}")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1']:.4f}")
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"ECE:       {metrics['ece']:.4f}")
        print(f"\nConfusion Matrix:")
        print(np.array(metrics['confusion_matrix']))
        
        return metrics
    
    def save(self, filepath: str):
        """Save model to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names,
                'model_type': self.model_type
            }, f)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.feature_names = data['feature_names']
            self.model_type = data['model_type']
        print(f"Model loaded from {filepath}")
        return self


if __name__ == "__main__":
    # Test model
    from src.data.data_loader import CreditDataLoader
    
    loader = CreditDataLoader()
    X, y, features = loader.load_and_prepare(sample_size=5000)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = CreditRiskModel(model_type="logistic")
    model.train(X_train, y_train, features)
    model.print_evaluation(X_test, y_test, "Test Set")
