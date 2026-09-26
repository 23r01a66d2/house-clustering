"""
PROPERTY MATCHER MODULE — Interactive Segment Assignment
Transforms new/user-entered property attributes using the ALREADY-FITTED StandardScaler
and determines the closest property community based on minimum Euclidean distance
to learned K-Means cluster centroids.
Validates inputs against empirical dataset bounds to warn of extreme out-of-distribution entries.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from src.utils import find_column_by_role

class PropertyMatcher:
    """
    Transforms arbitrary property features through the pre-fitted scaling pipeline
    and identifies the nearest market segment centroid in standardized similarity space.
    """
    def __init__(
        self,
        scaler: StandardScaler,
        kmeans_model: KMeans,
        segment_names: Dict[int, str],
        reference_df: pd.DataFrame,
        feature_names: List[str]
    ):
        self.scaler = scaler
        self.kmeans_model = kmeans_model
        self.segment_names = segment_names
        self.reference_df = reference_df
        self.feature_names = feature_names
        self.n_dimensions = len(feature_names)
        
        # Calculate empirical validation bounds from the reference inventory
        self.bounds = {}
        for feat in self.feature_names:
            if feat in self.reference_df.columns:
                s = self.reference_df[feat].dropna()
                self.bounds[feat] = {
                    "min": float(s.min()),
                    "max": float(s.max()),
                    "p01": float(s.quantile(0.01)),
                    "p99": float(s.quantile(0.99)),
                    "median": float(s.median())
                }

    def validate_inputs(self, user_inputs: Dict[str, float]) -> List[str]:
        """
        Validates user entries against empirical dataset distributions.
        Returns explicit warnings for values far outside the 1st-99th percentile range.
        """
        warnings = []
        for feat, val in user_inputs.items():
            if feat in self.bounds:
                b = self.bounds[feat]
                if val < b["min"] or val > b["max"]:
                    warnings.append(
                        f"'{feat}' value ({val:,.1f}) is completely outside the dataset observed spectrum "
                        f"[{b['min']:,.1f} — {b['max']:,.1f}]. Assignment may be skewed."
                    )
                elif val < b["p01"] or val > b["p99"]:
                    warnings.append(
                        f"'{feat}' value ({val:,.1f}) falls in the extreme outer 1% tail of the dataset. "
                        f"Typical range: {b['p01']:,.1f} to {b['p99']:,.1f}."
                    )
        return warnings

    def match_property(
        self,
        user_inputs: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Transforms user input through the ALREADY-FITTED scaler,
        computes standardized Euclidean distance to all cluster centroids,
        and assigns the closest segment.
        """
        # Validate inputs first
        validation_warnings = self.validate_inputs(user_inputs)

        # Assemble feature vector matching exact fitted feature ordering
        feature_vector = []
        for feat in self.feature_names:
            if feat in user_inputs:
                feature_vector.append(float(user_inputs[feat]))
            elif feat in self.bounds:
                # Impute with empirical median for unentered secondary attributes
                feature_vector.append(self.bounds[feat]["median"])
            else:
                feature_vector.append(0.0)

        # Standardize using ALREADY-FITTED scaler (Zero refitting)
        X_input = np.array([feature_vector])
        X_scaled = self.scaler.transform(X_input)

        # Compute Euclidean distance to all cluster centers in standardized space
        centroids = self.kmeans_model.cluster_centers_
        distances = np.linalg.norm(centroids - X_scaled, axis=1)

        assigned_cluster = int(np.argmin(distances))
        min_distance = float(distances[assigned_cluster])
        matched_segment_name = self.segment_names.get(assigned_cluster, f"Segment {assigned_cluster}")

        # Inverse transform matched centroid back to original units for "Why This Match?" comparison
        centroid_orig = self.scaler.inverse_transform([centroids[assigned_cluster]])[0]
        centroid_dict = dict(zip(self.feature_names, centroid_orig))

        # Build comparison details for primary display attributes
        comparison = []
        display_roles = ["price", "living_area", "bedrooms", "bathrooms", "grade", "built_year"]
        for role in display_roles:
            col_name = find_column_by_role(self.reference_df, role)
            if col_name and col_name in user_inputs and col_name in centroid_dict:
                user_val = float(user_inputs[col_name])
                centroid_val = float(centroid_dict[col_name])
                diff = user_val - centroid_val
                pct_diff = (diff / centroid_val * 100.0) if centroid_val != 0 else 0.0

                comparison.append({
                    "attribute": col_name,
                    "your_property": user_val,
                    "segment_typical": centroid_val,
                    "delta": diff,
                    "delta_pct": pct_diff
                })

        return {
            "matched_cluster_id": assigned_cluster,
            "matched_segment_name": matched_segment_name,
            "distance_to_centroid": round(min_distance, 3),
            "all_distances": {
                self.segment_names.get(i, f"Segment {i}"): round(float(d), 3)
                for i, d in enumerate(distances)
            },
            "comparison": comparison,
            "validation_warnings": validation_warnings,
            "dimensionality": self.n_dimensions
        }
