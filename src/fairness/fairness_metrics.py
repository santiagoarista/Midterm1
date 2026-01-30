"""
Fairness metrics for credit risk model.
Implements disparate impact, equalized odds, and demographic parity.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.metrics import confusion_matrix


class FairnessEvaluator:
    """Evaluate fairness metrics across protected groups."""
    
    def __init__(self):
        self.results = {}
        
    def create_synthetic_groups(self, X: np.ndarray, n_groups: int = 2) -> np.ndarray:
        """
        Create synthetic protected groups for demonstration.
        In production, use actual demographic data.
        """
        # Use clustering on features to create synthetic groups
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_groups, random_state=42)
        groups = kmeans.fit_predict(X[:, :5])  # Use first 5 features
        return groups
    
    def disparate_impact(self, y_pred: np.ndarray, groups: np.ndarray, 
                        threshold: float = 0.5) -> Dict:
        """
        Calculate disparate impact ratio.
        DI = P(positive|group=0) / P(positive|group=1)
        """
        binary_pred = (y_pred >= threshold).astype(int)
        
        results = {}
        unique_groups = np.unique(groups)
        
        # Calculate selection rates
        selection_rates = {}
        for group in unique_groups:
            mask = groups == group
            selection_rate = binary_pred[mask].mean()
            selection_rates[f"group_{group}"] = selection_rate
        
        # Calculate disparate impact (min/max ratio)
        rates = list(selection_rates.values())
        di_ratio = min(rates) / max(rates) if max(rates) > 0 else 0
        
        results['selection_rates'] = selection_rates
        results['disparate_impact_ratio'] = di_ratio
        results['passes_80_rule'] = 0.8 <= di_ratio <= 1.25
        
        return results
    
    def equalized_odds(self, y_true: np.ndarray, y_pred: np.ndarray, 
                       groups: np.ndarray, threshold: float = 0.5) -> Dict:
        """
        Calculate equalized odds: equal TPR and FPR across groups.
        """
        binary_pred = (y_pred >= threshold).astype(int)
        
        results = {}
        unique_groups = np.unique(groups)
        
        tpr_list = []
        fpr_list = []
        
        for group in unique_groups:
            mask = groups == group
            y_true_g = y_true[mask]
            y_pred_g = binary_pred[mask]
            
            if len(y_true_g) == 0:
                continue
                
            tn, fp, fn, tp = confusion_matrix(y_true_g, y_pred_g).ravel()
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            tpr_list.append(tpr)
            fpr_list.append(fpr)
            
            results[f"group_{group}_tpr"] = tpr
            results[f"group_{group}_fpr"] = fpr
        
        # Calculate disparities
        results['tpr_disparity'] = max(tpr_list) - min(tpr_list) if tpr_list else 0
        results['fpr_disparity'] = max(fpr_list) - min(fpr_list) if fpr_list else 0
        results['max_disparity'] = max(results['tpr_disparity'], results['fpr_disparity'])
        
        return results
    
    def demographic_parity(self, y_pred: np.ndarray, groups: np.ndarray,
                          threshold: float = 0.5) -> Dict:
        """
        Calculate demographic parity: equal positive prediction rates.
        """
        binary_pred = (y_pred >= threshold).astype(int)
        
        results = {}
        unique_groups = np.unique(groups)
        
        pos_rates = []
        for group in unique_groups:
            mask = groups == group
            pos_rate = binary_pred[mask].mean()
            results[f"group_{group}_positive_rate"] = pos_rate
            pos_rates.append(pos_rate)
        
        results['max_difference'] = max(pos_rates) - min(pos_rates) if pos_rates else 0
        
        return results
    
    def evaluate_all(self, y_true: np.ndarray, y_pred: np.ndarray,
                     X: np.ndarray, threshold: float = 0.5) -> Dict:
        """
        Run all fairness metrics.
        """
        # Create synthetic groups for demonstration
        groups = self.create_synthetic_groups(X)
        
        print("\n" + "="*70)
        print("FAIRNESS EVALUATION")
        print("="*70)
        
        # Disparate Impact
        di_results = self.disparate_impact(y_pred, groups, threshold)
        print(f"\n1. Disparate Impact:")
        print(f"   Ratio: {di_results['disparate_impact_ratio']:.4f}")
        print(f"   Passes 80% Rule: {di_results['passes_80_rule']}")
        for group, rate in di_results['selection_rates'].items():
            print(f"   {group} selection rate: {rate:.4f}")
        
        # Equalized Odds
        eo_results = self.equalized_odds(y_true, y_pred, groups, threshold)
        print(f"\n2. Equalized Odds:")
        print(f"   TPR Disparity: {eo_results['tpr_disparity']:.4f}")
        print(f"   FPR Disparity: {eo_results['fpr_disparity']:.4f}")
        print(f"   Max Disparity: {eo_results['max_disparity']:.4f}")
        
        # Demographic Parity
        dp_results = self.demographic_parity(y_pred, groups, threshold)
        print(f"\n3. Demographic Parity:")
        print(f"   Max Difference: {dp_results['max_difference']:.4f}")
        
        # Combine all results
        all_results = {
            'disparate_impact': di_results,
            'equalized_odds': eo_results,
            'demographic_parity': dp_results,
            'groups': groups
        }
        
        self.results = all_results
        return all_results
    
    def save_results(self, filepath: str):
        """Save fairness results to CSV."""
        if not self.results:
            print("No results to save. Run evaluate_all first.")
            return
        
        # Flatten results for CSV
        rows = []
        
        # Disparate Impact
        di = self.results['disparate_impact']
        rows.append({
            'metric': 'Disparate Impact Ratio',
            'value': di['disparate_impact_ratio'],
            'threshold': '0.8-1.25',
            'passes': di['passes_80_rule']
        })
        
        # Equalized Odds
        eo = self.results['equalized_odds']
        rows.append({
            'metric': 'TPR Disparity',
            'value': eo['tpr_disparity'],
            'threshold': '<0.1',
            'passes': eo['tpr_disparity'] < 0.1
        })
        rows.append({
            'metric': 'FPR Disparity',
            'value': eo['fpr_disparity'],
            'threshold': '<0.1',
            'passes': eo['fpr_disparity'] < 0.1
        })
        
        # Demographic Parity
        dp = self.results['demographic_parity']
        rows.append({
            'metric': 'Demographic Parity Difference',
            'value': dp['max_difference'],
            'threshold': '<0.1',
            'passes': dp['max_difference'] < 0.1
        })
        
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        print(f"\nFairness results saved to {filepath}")
