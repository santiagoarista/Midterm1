"""
Data loading and preprocessing module for Home Credit Default Risk dataset.

Primary contributor: Diego Vergara Hernández
Additional work by: Santiago Arista Viramontes
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class CreditDataLoader:
    """Load and preprocess Home Credit Default Risk data."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        
    def load_application_data(self, train: bool = True) -> pd.DataFrame:
        """Load main application data."""
        filename = "application_train.csv" if train else "application_test.csv"
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(
                f"Data file not found: {filepath}\n"
                "Please download from: https://www.kaggle.com/competitions/home-credit-default-risk/data"
            )
        
        print(f"Loading {filename}...")
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} records with {len(df.columns)} columns")
        
        return df
    
    def select_features(self, df: pd.DataFrame, max_features: int = 50) -> pd.DataFrame:
        """Select most important features to reduce dimensionality."""
        # Separate target if present
        has_target = 'TARGET' in df.columns
        if has_target:
            target = df['TARGET']
            df = df.drop('TARGET', axis=1)
        
        # Remove ID column
        if 'SK_ID_CURR' in df.columns:
            df = df.drop('SK_ID_CURR', axis=1)
        
        # Select numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Calculate missing rate and select features with less missing values
        missing_rate = df[numeric_cols].isnull().mean()
        selected_cols = missing_rate[missing_rate < 0.5].index.tolist()
        
        # Limit to max_features
        if len(selected_cols) > max_features:
            selected_cols = selected_cols[:max_features]
        
        df_selected = df[selected_cols].copy()
        
        # Add target back if present
        if has_target:
            df_selected['TARGET'] = target
        
        print(f"Selected {len(selected_cols)} features")
        return df_selected
    
    def preprocess(self, df: pd.DataFrame, fit: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Preprocess features: impute missing values and scale."""
        # Separate target if present
        has_target = 'TARGET' in df.columns
        if has_target:
            y = df['TARGET'].values
            X = df.drop('TARGET', axis=1)
        else:
            y = None
            X = df
        
        # Impute missing values
        if fit:
            X_imputed = self.imputer.fit_transform(X)
        else:
            X_imputed = self.imputer.transform(X)
        
        # Scale features
        if fit:
            X_scaled = self.scaler.fit_transform(X_imputed)
        else:
            X_scaled = self.scaler.transform(X_imputed)
        
        return X_scaled, y
    
    def load_and_prepare(self, sample_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, list]:
        """Complete pipeline: load, select features, and preprocess."""
        # Load data
        df = self.load_application_data(train=True)
        
        # Sample if requested (for faster experimentation)
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
            print(f"Sampled {sample_size} records")
        
        # Select features
        df_selected = self.select_features(df)
        feature_names = [col for col in df_selected.columns if col != 'TARGET']
        
        # Preprocess
        X, y = self.preprocess(df_selected, fit=True)
        
        print(f"Final dataset shape: X={X.shape}, y={y.shape}")
        print(f"Class distribution: {np.bincount(y.astype(int))}")
        
        return X, y, feature_names


if __name__ == "__main__":
    # Test data loading
    loader = CreditDataLoader()
    X, y, features = loader.load_and_prepare(sample_size=10000)
    print(f"\nFeature names: {features[:10]}...")
