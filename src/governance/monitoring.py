"""
Governance framework for model monitoring and audit trails.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import hashlib


class GovernanceLogger:
    """Log predictions and explanations for audit trail."""
    
    def __init__(self, log_dir: str = "governance_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def log_prediction(self, prediction_id: str, features: np.ndarray,
                      prediction: float, shap_values: Optional[np.ndarray] = None,
                      metadata: Optional[Dict] = None) -> str:
        """
        Log a single prediction with full audit trail.
        
        Returns:
            Transaction ID for this prediction
        """
        timestamp = datetime.now().isoformat()
        
        # Create transaction record
        record = {
            'transaction_id': prediction_id,
            'session_id': self.session_id,
            'timestamp': timestamp,
            'prediction': float(prediction),
            'feature_hash': self._hash_features(features),
            'metadata': metadata or {}
        }
        
        # Add SHAP values if provided
        if shap_values is not None:
            record['shap_values'] = {
                f'feature_{i}': float(val) 
                for i, val in enumerate(shap_values)
            }
            record['top_3_features'] = self._get_top_features(shap_values)
        
        # Save to log file
        log_file = self.log_dir / f"predictions_{self.session_id}.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(record) + '\n')
        
        return record['transaction_id']
    
    def _hash_features(self, features: np.ndarray) -> str:
        """Create hash of features for verification."""
        return hashlib.md5(features.tobytes()).hexdigest()[:16]
    
    def _get_top_features(self, shap_values: np.ndarray, top_k: int = 3) -> list:
        """Get indices and values of top contributing features."""
        abs_values = np.abs(shap_values)
        top_indices = np.argsort(abs_values)[-top_k:][::-1]
        return [
            {'feature_idx': int(idx), 'shap_value': float(shap_values[idx])}
            for idx in top_indices
        ]


class ModelMonitor:
    """Monitor model performance and data drift."""
    
    def __init__(self):
        self.monitoring_data = {
            'predictions': [],
            'latencies': [],
            'timestamps': []
        }
        
    def record_inference(self, prediction: float, latency_ms: float):
        """Record inference metrics."""
        self.monitoring_data['predictions'].append(prediction)
        self.monitoring_data['latencies'].append(latency_ms)
        self.monitoring_data['timestamps'].append(time.time())
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        if not self.monitoring_data['predictions']:
            return {}
        
        predictions = np.array(self.monitoring_data['predictions'])
        latencies = np.array(self.monitoring_data['latencies'])
        
        return {
            'total_predictions': len(predictions),
            'mean_prediction': float(predictions.mean()),
            'prediction_std': float(predictions.std()),
            'mean_latency_ms': float(latencies.mean()),
            'p95_latency_ms': float(np.percentile(latencies, 95)),
            'p99_latency_ms': float(np.percentile(latencies, 99)),
            'max_latency_ms': float(latencies.max())
        }
    
    def save_report(self, filepath: str):
        """Save monitoring report."""
        stats = self.get_statistics()
        
        report_lines = [
            "="*70,
            "MODEL MONITORING REPORT",
            "="*70,
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\nPrediction Statistics:",
            f"  Total Predictions: {stats.get('total_predictions', 0)}",
            f"  Mean Prediction: {stats.get('mean_prediction', 0):.4f}",
            f"  Prediction Std Dev: {stats.get('prediction_std', 0):.4f}",
            f"\nLatency Statistics:",
            f"  Mean Latency: {stats.get('mean_latency_ms', 0):.2f} ms",
            f"  P95 Latency: {stats.get('p95_latency_ms', 0):.2f} ms",
            f"  P99 Latency: {stats.get('p99_latency_ms', 0):.2f} ms",
            f"  Max Latency: {stats.get('max_latency_ms', 0):.2f} ms",
            "\nPerformance Status:",
        ]
        
        # Check SLA compliance
        mean_latency = stats.get('mean_latency_ms', 0)
        if mean_latency < 100:
            report_lines.append(f"  ✓ Mean latency under 100ms target")
        else:
            report_lines.append(f"  ✗ Mean latency exceeds 100ms target")
        
        with open(filepath, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"\nMonitoring report saved to {filepath}")


class AuditTrail:
    """Create audit reports for regulatory compliance."""
    
    def __init__(self, model_version: str = "1.0"):
        self.model_version = model_version
        
    def generate_audit_report(self, model_metrics: Dict, fairness_metrics: Dict,
                             output_path: str):
        """
        Generate comprehensive audit report.
        """
        report = []
        report.append("="*70)
        report.append("MODEL AUDIT REPORT")
        report.append("="*70)
        report.append(f"\nReport Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Model Version: {self.model_version}")
        
        # Model Performance
        report.append("\n" + "-"*70)
        report.append("1. MODEL PERFORMANCE METRICS")
        report.append("-"*70)
        for key, value in model_metrics.items():
            if isinstance(value, float):
                report.append(f"  {key}: {value:.4f}")
            else:
                report.append(f"  {key}: {value}")
        
        # Fairness Assessment
        report.append("\n" + "-"*70)
        report.append("2. FAIRNESS ASSESSMENT")
        report.append("-"*70)
        
        if 'disparate_impact' in fairness_metrics:
            di = fairness_metrics['disparate_impact']
            report.append(f"  Disparate Impact Ratio: {di.get('disparate_impact_ratio', 0):.4f}")
            report.append(f"  Passes 80% Rule: {di.get('passes_80_rule', False)}")
        
        if 'equalized_odds' in fairness_metrics:
            eo = fairness_metrics['equalized_odds']
            report.append(f"  TPR Disparity: {eo.get('tpr_disparity', 0):.4f}")
            report.append(f"  FPR Disparity: {eo.get('fpr_disparity', 0):.4f}")
        
        # Compliance Status
        report.append("\n" + "-"*70)
        report.append("3. COMPLIANCE STATUS")
        report.append("-"*70)
        
        compliance_checks = []
        
        # Check model performance
        if model_metrics.get('roc_auc', 0) > 0.65:
            compliance_checks.append(("Model Performance", "PASS", "AUC > 0.65"))
        else:
            compliance_checks.append(("Model Performance", "FAIL", "AUC < 0.65"))
        
        # Check fairness
        if fairness_metrics.get('disparate_impact', {}).get('passes_80_rule', False):
            compliance_checks.append(("Fairness - Disparate Impact", "PASS", "Within 80% rule"))
        else:
            compliance_checks.append(("Fairness - Disparate Impact", "WARNING", "Outside 80% rule"))
        
        # Check equalized odds
        eo_disparity = fairness_metrics.get('equalized_odds', {}).get('max_disparity', 1.0)
        if eo_disparity < 0.1:
            compliance_checks.append(("Fairness - Equalized Odds", "PASS", "Disparity < 0.1"))
        else:
            compliance_checks.append(("Fairness - Equalized Odds", "WARNING", f"Disparity = {eo_disparity:.3f}"))
        
        for check_name, status, detail in compliance_checks:
            report.append(f"  [{status}] {check_name}: {detail}")
        
        # Save report
        with open(output_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"\n✓ Audit report generated: {output_path}")
        
        return '\n'.join(report)
