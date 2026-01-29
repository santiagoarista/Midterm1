"""
SHAP-based explainability for credit risk models (M1 - XAI Feature).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from typing import Optional, Dict, Any


class SHAPExplainer:
    """Generate SHAP explanations for credit risk predictions."""
    
    def __init__(self, model, X_background: np.ndarray, feature_names: list):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained model with predict_proba method
            X_background: Background dataset for SHAP (typically training set sample)
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        self.X_background = X_background
        
        # Initialize appropriate explainer based on model type
        print("Initializing SHAP explainer...")
        
        # Use TreeExplainer for tree models, KernelExplainer for others
        try:
            # Try TreeExplainer first (faster for tree models)
            self.explainer = shap.TreeExplainer(model.model)
            print("Using TreeExplainer")
        except:
            # Fallback to KernelExplainer for linear models
            # Use a sample of background data to speed up
            background_sample = shap.sample(X_background, min(100, len(X_background)))
            self.explainer = shap.KernelExplainer(
                model.predict_proba, 
                background_sample
            )
            print("Using KernelExplainer")
    
    def explain_instance(self, X: np.ndarray, index: int = 0) -> Dict[str, Any]:
        """
        Generate SHAP explanation for a single instance.
        
        Returns dict with:
            - shap_values: SHAP values for the instance
            - base_value: Expected value (average prediction)
            - prediction: Model prediction
            - top_features: Most influential features
        """
        instance = X[index:index+1]
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(instance)
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Get values for positive class
        
        shap_values = shap_values[0]  # Get first instance
        
        # Get base value
        if hasattr(self.explainer, 'expected_value'):
            base_value = self.explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                base_value = base_value[1] if len(base_value) > 1 else base_value[0]
        else:
            base_value = 0.0
        
        # Get prediction
        prediction = self.model.predict_proba(instance)[0]
        
        # Get top contributing features
        feature_importance = list(zip(self.feature_names, shap_values, instance[0]))
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        return {
            'shap_values': shap_values,
            'base_value': base_value,
            'prediction': prediction,
            'top_features': feature_importance[:10],
            'instance_values': instance[0]
        }
    
    def plot_waterfall(self, X: np.ndarray, index: int = 0, save_path: Optional[str] = None):
        """Create waterfall plot showing feature contributions."""
        explanation = self.explain_instance(X, index)
        
        plt.figure(figsize=(10, 6))
        
        # Get top 10 features
        top_features = explanation['top_features'][:10]
        feature_names = [f[0] for f in top_features]
        shap_vals = [f[1] for f in top_features]
        
        # Create waterfall-style bar plot
        colors = ['red' if x < 0 else 'green' for x in shap_vals]
        y_pos = np.arange(len(feature_names))
        
        plt.barh(y_pos, shap_vals, color=colors, alpha=0.7)
        plt.yticks(y_pos, feature_names)
        plt.xlabel('SHAP Value (Impact on Prediction)')
        plt.title(f'Feature Contributions for Instance {index}\n'
                  f'Prediction: {explanation["prediction"]:.3f}')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Waterfall plot saved to {save_path}")
        
        plt.show()
        
        return explanation
    
    def plot_summary(self, X: np.ndarray, max_display: int = 20, save_path: Optional[str] = None):
        """Create summary plot showing feature importance across all instances."""
        print("Computing SHAP values for all instances...")
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X)
        
        # Handle different output formats
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, 
            X, 
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Summary plot saved to {save_path}")
        
        plt.show()
    
    def get_global_importance(self, X: np.ndarray) -> pd.DataFrame:
        """Calculate global feature importance using mean absolute SHAP values."""
        print("Computing global feature importance...")
        
        shap_values = self.explainer.shap_values(X)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Calculate mean absolute SHAP value for each feature
        importance = np.abs(shap_values).mean(axis=0)
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df
    
    def explain_prediction_text(self, X: np.ndarray, index: int = 0) -> str:
        """Generate human-readable explanation text."""
        explanation = self.explain_instance(X, index)
        
        decision = "APPROVE" if explanation['prediction'] < 0.5 else "DENY"
        confidence = abs(explanation['prediction'] - 0.5) * 200
        
        text = f"Credit Decision: {decision}\n"
        text += f"Default Risk: {explanation['prediction']:.1%}\n"
        text += f"Confidence: {confidence:.1f}%\n\n"
        text += f"Top Contributing Factors:\n"
        
        for i, (feature, shap_val, feature_val) in enumerate(explanation['top_features'][:5], 1):
            impact = "INCREASES" if shap_val > 0 else "DECREASES"
            text += f"{i}. {feature} = {feature_val:.2f} {impact} risk by {abs(shap_val):.3f}\n"
        
        return text


if __name__ == "__main__":
    # Test explainer
    from src.data.data_loader import CreditDataLoader
    from src.models.credit_model import CreditRiskModel
    from sklearn.model_selection import train_test_split
    
    loader = CreditDataLoader()
    X, y, features = loader.load_and_prepare(sample_size=5000)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = CreditRiskModel(model_type="logistic")
    model.train(X_train, y_train, features)
    
    explainer = SHAPExplainer(model, X_train, features)
    print("\n" + explainer.explain_prediction_text(X_test, 0))
