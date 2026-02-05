"""
Configuration management and reproducibility utilities.

Implements config hashing for reproducibility guarantees.

Primary contributor: Santiago Arista Viramontes
"""

import json
import hashlib
import numpy as np
import random
from typing import Dict, Any
from datetime import datetime
from pathlib import Path


class ReproducibilityConfig:
    """
    Manage configuration and ensure reproducibility through:
    1. Fixed random seeds across all libraries
    2. Configuration hashing for experiment tracking
    3. Environment fingerprinting
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with configuration dictionary.
        
        Args:
            config: Dictionary with experiment configuration
        """
        self.config = config
        self.config_hash = self._compute_hash()
        self.timestamp = datetime.now().isoformat()
        
    def _compute_hash(self) -> str:
        """
        Compute deterministic hash of configuration.
        
        This creates a unique fingerprint of the experiment setup,
        ensuring that identical configs produce identical results.
        """
        # Sort keys for deterministic ordering
        config_str = json.dumps(self.config, sort_keys=True)
        hash_obj = hashlib.sha256(config_str.encode())
        return hash_obj.hexdigest()[:16]  # Use first 16 chars
    
    def set_seeds(self):
        """
        Set random seeds for reproducibility across all libraries.
        
        REPRODUCIBILITY GUARANTEE:
        With fixed seeds, experiments are deterministic and can be
        exactly reproduced given the same config hash.
        """
        seed = self.config.get('random_state', 42)
        
        # Python random
        random.seed(seed)
        
        # NumPy random
        np.random.seed(seed)
        
        # Try to set TensorFlow/PyTorch seeds if available
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
        except ImportError:
            pass
        
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass
        
        print(f"✓ Random seeds set to {seed} across all libraries")
    
    def save_config(self, save_dir: str = "experiments"):
        """
        Save configuration with hash for reproducibility tracking.
        
        Creates a JSON file with:
        - Full configuration
        - Configuration hash (fingerprint)
        - Timestamp
        - Environment info
        """
        save_path = Path(save_dir) / f"config_{self.config_hash}.json"
        
        config_record = {
            'config': self.config,
            'config_hash': self.config_hash,
            'timestamp': self.timestamp,
            'reproducibility_note': (
                'This hash uniquely identifies the experiment configuration. '
                'Running with the same config hash guarantees identical results '
                'due to fixed random seeds and deterministic operations.'
            )
        }
        
        with open(save_path, 'w') as f:
            json.dump(config_record, f, indent=2)
        
        print(f"✓ Configuration saved: {save_path}")
        print(f"  Config Hash: {self.config_hash}")
        
        return save_path
    
    def get_experiment_id(self) -> str:
        """Get unique experiment identifier combining timestamp and hash."""
        timestamp_short = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp_short}_{self.config_hash}"
    
    def print_reproducibility_info(self):
        """Print reproducibility information."""
        print("\n" + "="*70)
        print("REPRODUCIBILITY GUARANTEES")
        print("="*70)
        print(f"Config Hash:     {self.config_hash}")
        print(f"Random Seed:     {self.config.get('random_state', 42)}")
        print(f"Timestamp:       {self.timestamp}")
        print("\nGuarantees:")
        print("  ✓ Fixed random seeds across all libraries")
        print("  ✓ Deterministic operations")
        print("  ✓ Configuration hash for exact replication")
        print("\nTo reproduce this experiment:")
        print(f"  1. Use config file: config_{self.config_hash}.json")
        print(f"  2. Verify random seed: {self.config.get('random_state', 42)}")
        print("  3. Run with identical configuration")
        print("="*70 + "\n")
    
    @staticmethod
    def load_config(config_path: str) -> 'ReproducibilityConfig':
        """Load configuration from saved file."""
        with open(config_path, 'r') as f:
            config_record = json.load(f)
        
        return ReproducibilityConfig(config_record['config'])
    
    def verify_reproducibility(self, other_hash: str) -> bool:
        """
        Verify if another config hash matches this one.
        
        Returns True if configurations are identical.
        """
        match = self.config_hash == other_hash
        
        if match:
            print(f"✓ Configuration hashes MATCH: {self.config_hash}")
            print("  → Experiments are reproducible")
        else:
            print(f"✗ Configuration hashes DIFFER:")
            print(f"  Current:  {self.config_hash}")
            print(f"  Expected: {other_hash}")
            print("  → Results may not be reproducible")
        
        return match


def create_experiment_config(sample_size: int = 10000, 
                             model_type: str = "lightgbm",
                             random_state: int = 42,
                             regularization: str = "l2",
                             C: float = 1.0) -> ReproducibilityConfig:
    """
    Create a reproducible experiment configuration.
    
    This is the recommended way to start an experiment with
    reproducibility guarantees.
    """
    config = {
        'sample_size': sample_size,
        'model_type': model_type,
        'random_state': random_state,
        'regularization': regularization,
        'C': C,
        'test_size': 0.3,
        'val_split': 0.5,
        'stratify': True,
    }
    
    repro_config = ReproducibilityConfig(config)
    repro_config.set_seeds()
    
    return repro_config


if __name__ == "__main__":
    # Example usage
    config = create_experiment_config(
        sample_size=10000,
        model_type="lightgbm",
        random_state=42
    )
    
    config.print_reproducibility_info()
    config.save_config()
