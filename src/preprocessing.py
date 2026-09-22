"""
MODULE 2 — DATA PREPROCESSING
Handles cleaning, missing values, deduplication, outlier handling,
feature selection, and feature scaling using StandardScaler.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import config
from src.utils import log_substep

class DataPreprocessor:
    """Class to clean and scale dataset for K-Means clustering."""

    def __init__(self, df: pd.DataFrame, custom_features: Optional[List[str]] = None):
        self.raw_df = df.copy()
        self.cleaned_df = pd.DataFrame()
        self.scaled_df = pd.DataFrame()
        self.custom_features = custom_features
        self.scaler = StandardScaler()
        
        self.selected_features: List[str] = []
        self.removed_features: List[str] = []
        self.records_before = len(df)
        self.records_after = 0
        self.missing_before = int(df.isnull().sum().sum())
        self.missing_after = 0
        self.duplicates_removed = 0
        self.outliers_removed = 0

    def _identify_features(self) -> Tuple[List[str], List[str]]:
        """
        Dynamically determine features to keep for clustering
        and irrelevant columns to exclude (IDs, Dates, Postal Codes).
        """
        if self.custom_features is not None:
            # Use user-specified features
            selected = [col for col in self.custom_features if col in self.raw_df.columns]
            removed = [col for col in self.raw_df.columns if col not in selected]
            return selected, removed

        selected = []
        removed = []

        for col in self.raw_df.columns:
            col_lower = col.strip().lower()
            # Check if column matches identifier/metadata keywords
            is_irrelevant = any(kw in col_lower for kw in config.IDENTIFIER_KEYWORDS)
            
            # Check if column is non-numeric
            is_non_numeric = not np.issubdtype(self.raw_df[col].dtype, np.number)

            if is_irrelevant or is_non_numeric:
                removed.append(col)
            else:
                selected.append(col)

        return selected, removed

    def preprocess(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Execute full preprocessing pipeline:
        1. Deduplication
        2. Outlier filtering
        3. Missing value imputation
        4. Feature selection
        5. StandardScaler transformation
        """
        df = self.raw_df.copy()

        # 1. Remove duplicate records
        n_before_dedup = len(df)
        df = df.drop_duplicates()
        self.duplicates_removed = n_before_dedup - len(df)

        # 2. Outlier / Invalid Value Handling
        # Look dynamically for bedroom column to detect unrealistic records (>20 bedrooms)
        bedroom_col = None
        for col in df.columns:
            if "bedroom" in col.lower():
                bedroom_col = col
                break

        if bedroom_col:
            # Filter out entries with invalid or extreme bedroom counts
            valid_mask = (df[bedroom_col] > 0) & (df[bedroom_col] <= config.BEDROOM_MAX_THRESHOLD)
            n_outliers = (~valid_mask).sum()
            df = df[valid_mask]
            self.outliers_removed = int(n_outliers)

        # 3. Missing Value Handling
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if np.issubdtype(df[col].dtype, np.number):
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])

        self.missing_after = int(df.isnull().sum().sum())
        self.records_after = len(df)

        # 4. Feature Selection
        self.selected_features, self.removed_features = self._identify_features()

        if not self.selected_features:
            raise ValueError("No valid numerical features remaining for clustering.")

        # Keep the clean unscaled dataset separate
        self.cleaned_df = df.reset_index(drop=True)

        # 5. Feature Scaling (StandardScaler)
        clustering_features_df = self.cleaned_df[self.selected_features]
        scaled_array = self.scaler.fit_transform(clustering_features_df)
        self.scaled_df = pd.DataFrame(
            scaled_array, 
            columns=self.selected_features, 
            index=self.cleaned_df.index
        )

        metadata = self.get_summary()
        return self.cleaned_df, self.scaled_df, metadata

    def get_summary(self) -> Dict:
        """Return structured preprocessing summary."""
        return {
            "records_before": self.records_before,
            "records_after": self.records_after,
            "missing_before": self.missing_before,
            "missing_after": self.missing_after,
            "duplicates_removed": self.duplicates_removed,
            "outliers_removed": self.outliers_removed,
            "selected_features": self.selected_features,
            "removed_features": self.removed_features,
            "scaling_method": "StandardScaler (mean=0, std=1)",
            "scaling_status": "Completed successfully"
        }

    def print_summary(self):
        """Print console summary for Module 2."""
        summary = self.get_summary()
        print("\nData Preprocessing Summary")
        print(f"Records Before Preprocessing: {summary['records_before']}")
        print(f"Records After Preprocessing:  {summary['records_after']}")
        print(f"Duplicates Removed:           {summary['duplicates_removed']}")
        print(f"Outliers Handled:             {summary['outliers_removed']}")
        print(f"Missing Values (Before/After):{summary['missing_before']} -> {summary['missing_after']}")
        print(f"Selected Features ({len(summary['selected_features'])}): {', '.join(summary['selected_features'][:6])}...")
        print(f"Removed Features  ({len(summary['removed_features'])}): {', '.join(summary['removed_features'])}")
        print(f"Scaling Status:               {summary['scaling_status']}")
        log_substep(f"Cleaned {summary['records_after']} records, scaled {len(summary['selected_features'])} features")

def preprocess_data(df: pd.DataFrame, custom_features: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Convenience functional interface for Module 2."""
    preprocessor = DataPreprocessor(df, custom_features=custom_features)
    cleaned_df, scaled_df, metadata = preprocessor.preprocess()
    preprocessor.print_summary()
    return cleaned_df, scaled_df, metadata
