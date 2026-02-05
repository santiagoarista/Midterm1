"""
SHAP-based explainability for credit risk models (M1 - XAI Feature).

Primary contributor: José Leobardo Navarro Márquez
Additional work by: Santiago Arista Viramontes
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
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
        
        plt.close()  # Close instead of show
        
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
        
        plt.close()  # Close instead of show
    
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
    
    def analyze_pathological_cases(self, X: np.ndarray, y_true: np.ndarray, 
                                   save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Identify and analyze pathological cases where SHAP explanations may be misleading.
        
        This demonstrates the LIMITS of SHAP explanations including:
        1. High-confidence misclassifications (model is wrong but SHAP explains why)
        2. Feature interactions that SHAP attributes individually  
        3. Prediction instability with similar explanations
        
        Returns dict with pathological case analyses showing XAI limitations.
        """
        print("\n" + "="*70)
        print("PATHOLOGICAL CASE ANALYSIS - Demonstrating SHAP Limitations")
        print("="*70)
        
        y_pred = self.model.model.predict_proba(X)[:, 1]
        y_pred_class = (y_pred > 0.5).astype(int)
        
        pathological_cases = {
            'high_confidence_errors': [],
            'explanation_instability': [],
            'summary': {}
        }
        
        # 1. Find high-confidence misclassifications
        print("\n1. HIGH-CONFIDENCE ERRORS (Model confident but wrong)")
        print("   Limitation: SHAP explains the model's reasoning, NOT ground truth")
        print("-" * 70)
        
        errors = np.where(y_pred_class != y_true)[0]
        high_conf_errors = [(idx, abs(y_pred[idx] - 0.5)) for idx in errors]
        high_conf_errors.sort(key=lambda x: x[1], reverse=True)
        
        for rank, (idx, confidence) in enumerate(high_conf_errors[:3], 1):
            explanation = self.explain_instance(X, idx)
            pathological_cases['high_confidence_errors'].append({
                'index': int(idx),
                'predicted': float(y_pred[idx]),
                'actual': int(y_true[idx]),
                'confidence': float(confidence),
                'top_features': [(f, float(s)) for f, s, _ in explanation['top_features'][:5]]
            })
            
            print(f"\n   Case {rank}: Index={idx}")
            print(f"   Predicted: {y_pred[idx]:.3f} | Actual: {y_true[idx]}")
            print(f"   → SHAP explains WHY the model made this WRONG prediction")
            print(f"   Top factors: {explanation['top_features'][0][0]} (SHAP={explanation['top_features'][0][1]:.3f})")
        
        # 2. Explanation instability - similar predictions, different explanations
        print("\n\n2. EXPLANATION INSTABILITY")
        print("   Limitation: Similar predictions may have different SHAP attributions")
        print("-" * 70)
        
        # Find pairs of instances with similar predictions
        pred_range = (0.4, 0.6)  # Near decision boundary
        boundary_indices = np.where((y_pred > pred_range[0]) & (y_pred < pred_range[1]))[0]
        
        if len(boundary_indices) >= 2:
            idx1, idx2 = boundary_indices[0], boundary_indices[1]
            exp1 = self.explain_instance(X, idx1)
            exp2 = self.explain_instance(X, idx2)
            
            # Calculate explanation similarity (cosine of SHAP vectors)
            shap1 = exp1['shap_values']
            shap2 = exp2['shap_values']
            explanation_similarity = np.dot(shap1, shap2) / (np.linalg.norm(shap1) * np.linalg.norm(shap2))
            
            pathological_cases['explanation_instability'] = {
                'idx1': int(idx1),
                'idx2': int(idx2),
                'pred1': float(y_pred[idx1]),
                'pred2': float(y_pred[idx2]),
                'prediction_diff': float(abs(y_pred[idx1] - y_pred[idx2])),
                'explanation_similarity': float(explanation_similarity),
                'top_feature_1': exp1['top_features'][0][0],
                'top_feature_2': exp2['top_features'][0][0]
            }
            
            print(f"   Instance {idx1}: Prediction={y_pred[idx1]:.3f}, Top Feature={exp1['top_features'][0][0]}")
            print(f"   Instance {idx2}: Prediction={y_pred[idx2]:.3f}, Top Feature={exp2['top_features'][0][0]}")
            print(f"   → Prediction Difference: {abs(y_pred[idx1] - y_pred[idx2]):.4f}")
            print(f"   → Explanation Similarity: {explanation_similarity:.4f}")
            print(f"   → Shows that similar predictions can have different explanations")
        
        # 3. Summary statistics
        print("\n\n3. SHAP EXPLANATION LIMITATIONS SUMMARY")
        print("-" * 70)
        
        error_rate = len(errors) / len(y_true)
        pathological_cases['summary'] = {
            'total_errors': int(len(errors)),
            'error_rate': float(error_rate),
            'high_confidence_error_count': len(pathological_cases['high_confidence_errors']),
            'caveat': 'SHAP explains model predictions, not ground truth. '
                     'Even wrong predictions get plausible explanations.'
        }
        
        print(f"   Total Errors: {len(errors)} ({error_rate:.1%})")
        print(f"   High-Confidence Errors: {len(pathological_cases['high_confidence_errors'])}")
        print(f"\n CRITICAL CAVEAT:")
        print(f"   SHAP values explain WHY the model made a prediction,")
        print(f"   NOT whether the prediction is correct.")
        print(f"   Wrong predictions still get coherent SHAP explanations!")
        
        # Save if requested
        if save_path:
            import json
            with open(save_path, 'w') as f:
                json.dump(pathological_cases, f, indent=2)
            print(f"\n   Pathological case analysis saved to {save_path}")
        
        print("="*70)
        return pathological_cases


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
