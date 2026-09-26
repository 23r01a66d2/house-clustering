"""
PROPERTY INTELLIGENCE MODULE — Data Trust & Feature Categorization
Audits data integrity, constructs the Data Trust Panel, maps domain-specific
feature roles, and fits the single standardized feature transformation.
"""

from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

import config

class PropertyIntelligence:
    """
    Manages data hygiene, audit logging, feature role assignment,
    and standardized feature scaling.
    """
    def __init__(self, raw_df: pd.DataFrame):
        self.raw_df = raw_df.copy()
        self.usable_df: Optional[pd.DataFrame] = None
        self.scaled_df: Optional[pd.DataFrame] = None
        self.scaler: Optional[StandardScaler] = None
        self.segmentation_features: List[str] = []
        self.excluded_features: List[str] = []
        self.data_trust_metrics: Dict[str, Any] = {}
        self.feature_roles: Dict[str, List[str]] = {}

    def audit_and_prepare(self) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler, Dict[str, Any]]:
        """
        Executes data hygiene, documents outlier/anomaly treatments,
        categorizes feature roles, and fits the single StandardScaler.
        """
        raw_count = len(self.raw_df)
        df = self.raw_df.copy()

        # 1. Missing Value Audit
        missing_count = int(df.isna().sum().sum())
        total_cells = df.shape[0] * df.shape[1]
        completeness_pct = ((total_cells - missing_count) / total_cells) * 100.0 if total_cells > 0 else 100.0

        # Fill any missing values if present using median
        if missing_count > 0:
            df = df.fillna(df.median(numeric_only=True))

        # 2. Duplicate Records Check
        duplicate_count = int(df.duplicated().sum())
        if duplicate_count > 0:
            df = df.drop_duplicates().reset_index(drop=True)

        # 3. Documented Anomaly Treatment (33-bedroom entry)
        anomaly_log = []
        bed_col = None
        for col in df.columns:
            if "bedroom" in col.lower():
                bed_col = col
                break

        outliers_treated = 0
        if bed_col and bed_col in df.columns:
            anomalous_mask = df[bed_col] > config.BEDROOM_MAX_THRESHOLD
            outliers_treated = int(anomalous_mask.sum())
            if outliers_treated > 0:
                for idx, row in df[anomalous_mask].iterrows():
                    anomaly_log.append({
                        "id": row.get("id", idx),
                        "reason": (
                            f"Typographical anomaly: recorded {int(row[bed_col])} bedrooms "
                            f"for {int(row.get('living area', 0))} sqft structure. "
                            f"Excluded to prevent standard deviation distortion."
                        )
                    })
                df = df[~anomalous_mask].reset_index(drop=True)

        self.usable_df = df

        # 4. Feature Role Categorization & Feature Selection
        self._categorize_features(df)

        # 5. Fit Single Standardization Scaler
        X = self.usable_df[self.segmentation_features].values
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.scaled_df = pd.DataFrame(
            X_scaled,
            columns=self.segmentation_features,
            index=self.usable_df.index
        )

        # 6. Build Data Trust Panel
        self.data_trust_metrics = {
            "properties_examined": raw_count,
            "usable_properties": len(self.usable_df),
            "missing_value_completeness_pct": round(completeness_pct, 2),
            "missing_cells_count": missing_count,
            "duplicate_records_found": duplicate_count,
            "features_suitable_for_segmentation": len(self.segmentation_features),
            "features_excluded": len(self.excluded_features),
            "excluded_feature_names": self.excluded_features,
            "segmentation_feature_names": self.segmentation_features,
            "outliers_treated_count": outliers_treated,
            "anomaly_log": anomaly_log,
            "feature_roles": self.feature_roles,
            "dimensionality": len(self.segmentation_features)
        }

        return self.usable_df, self.scaled_df, self.scaler, self.data_trust_metrics

    def _categorize_features(self, df: pd.DataFrame):
        """
        Dynamically classifies columns into domain-specific real-estate roles
        and identifies segmentation vs excluded columns.
        """
        roles: Dict[str, List[str]] = {
            "Property Size": [],
            "Layout": [],
            "Property Quality": [],
            "Property Age": [],
            "Economic": [],
            "Spatial": [],
            "System Identifiers": []
        }

        seg_cols = []
        excl_cols = []

        for col in df.columns:
            col_l = col.lower()
            # Identifiers / Non-physical tracking fields
            if any(k in col_l for k in config.IDENTIFIER_KEYWORDS):
                roles["System Identifiers"].append(col)
                excl_cols.append(col)
            # Economic (Price)
            elif "price" in col_l:
                roles["Economic"].append(col)
                seg_cols.append(col)
            # Layout
            elif any(k in col_l for k in ["bedroom", "bathroom", "floor"]):
                roles["Layout"].append(col)
                seg_cols.append(col)
            # Size
            elif any(k in col_l for k in ["area", "living", "lot", "basement"]):
                roles["Property Size"].append(col)
                seg_cols.append(col)
            # Quality & Structural condition
            elif any(k in col_l for k in ["grade", "condition", "view", "waterfront"]):
                roles["Property Quality"].append(col)
                seg_cols.append(col)
            # Age
            elif any(k in col_l for k in ["built", "renovat", "year"]):
                roles["Property Age"].append(col)
                seg_cols.append(col)
            # Spatial / Location context
            elif any(k in col_l for k in ["lattitude", "latitude", "longitude", "airport", "school"]):
                roles["Spatial"].append(col)
                seg_cols.append(col)
            else:
                if pd.api.types.is_numeric_dtype(df[col]):
                    roles["Property Size"].append(col)
                    seg_cols.append(col)
                else:
                    roles["System Identifiers"].append(col)
                    excl_cols.append(col)

        self.feature_roles = {k: v for k, v in roles.items() if len(v) > 0}
        self.segmentation_features = seg_cols
        self.excluded_features = excl_cols
