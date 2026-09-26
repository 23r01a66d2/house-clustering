"""
PROPERTY MATCHER MODULE — Interactive Segment Assignment
Transforms new/user-entered property attributes using the ALREADY-FITTED log1p and StandardScaler.
Determines the closest property community based on minimum Euclidean distance to learned K-Means centroids.
Reuses the pre-fitted pipeline without any refitting.
Displays zero fake confidence percentages or prediction probabilities.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import config
from src.utils import format_currency

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
        feature_names: List[str],
        log_features: Optional[List[str]] = None
    ):
        self.scaler = scaler
        self.kmeans_model = kmeans_model
        self.segment_names = segment_names
        self.reference_df = reference_df
        self.feature_names = feature_names
        self.log_features = log_features or list(config.LOG_TRANSFORM_FEATURES)
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
        labels = {
            "price": "Monthly Rent",
            "sqft": "Living Area",
            "bhk": "Bedrooms (BHK)",
            "numBathrooms": "Bathrooms",
            "furnishing_tier": "Furnishing Tier"
        }

        for feat, val in user_inputs.items():
            if feat in self.bounds:
                b = self.bounds[feat]
                lbl = labels.get(feat, feat)
                if val < b["min"] or val > b["max"]:
                    warnings.append(
                        f"'{lbl}' ({val:,.1f}) is outside the observed dataset range "
                        f"[{b['min']:,.1f} — {b['max']:,.1f}]."
                    )
                elif val < b["p01"] or val > b["p99"]:
                    warnings.append(
                        f"'{lbl}' ({val:,.1f}) falls in the extreme outer 1% tail of the dataset "
                        f"(typical range: {b['p01']:,.1f} to {b['p99']:,.1f})."
                    )
        return warnings

    def match_property(
        self,
        user_inputs: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Transforms user input through the ALREADY-FITTED pipeline (log1p + StandardScaler),
        computes Euclidean distance to all learned cluster centroids,
        and assigns the closest segment.
        """
        # Validate inputs first
        validation_warnings = self.validate_inputs(user_inputs)

        # Assemble raw vector in exact fitted feature order
        raw_values = []
        transformed_vector = []
        for feat in self.feature_names:
            if feat in user_inputs:
                raw_val = float(user_inputs[feat])
            elif feat in self.bounds:
                raw_val = float(self.bounds[feat]["median"])
            else:
                raw_val = 0.0
            
            raw_values.append(raw_val)

            # Apply log1p if this feature was log-transformed during training
            if feat in self.log_features:
                val_trans = float(np.log1p(max(0.0, raw_val)))
            else:
                val_trans = raw_val
            
            transformed_vector.append(val_trans)

        # Standardize using ALREADY-FITTED scaler (Zero refitting)
        X_df = pd.DataFrame([transformed_vector], columns=self.feature_names)
        X_scaled = self.scaler.transform(X_df)

        # Compute Euclidean distance to all cluster centers in standardized similarity space
        centroids = self.kmeans_model.cluster_centers_
        distances = np.linalg.norm(centroids - X_scaled, axis=1)

        assigned_cluster = int(np.argmin(distances))
        min_distance = float(distances[assigned_cluster])
        matched_segment_name = self.segment_names.get(assigned_cluster, f"Cluster {assigned_cluster}")

        # Inverse transform matched centroid back to original units for "Why This Match?" comparison
        centroid_scaled = centroids[assigned_cluster]
        centroid_inv_trans = self.scaler.inverse_transform([centroid_scaled])[0]
        
        centroid_orig_units = {}
        for feat, val in zip(self.feature_names, centroid_inv_trans):
            if feat in self.log_features:
                centroid_orig_units[feat] = float(np.expm1(val))
            else:
                centroid_orig_units[feat] = float(val)

        # Build comparison details
        status_names = {0: "Unfurnished", 1: "Semi-Furnished", 2: "Furnished"}
        furn_val = int(round(user_inputs.get("furnishing_tier", 0)))

        comparison = [
            {
                "attribute": "Monthly Rent",
                "your_property": f"₹{float(user_inputs.get('price', 0)):,.0f}/mo",
                "cluster_typical": f"₹{centroid_orig_units.get('price', 0):,.0f}/mo",
                "status": "Aligned" if abs(user_inputs.get('price', 0) - centroid_orig_units.get('price', 0)) / max(1, centroid_orig_units.get('price', 0)) < 0.3 else "Variance"
            },
            {
                "attribute": "Living Area",
                "your_property": f"{float(user_inputs.get('sqft', 0)):,.0f} sqft",
                "cluster_typical": f"{centroid_orig_units.get('sqft', 0):,.0f} sqft",
                "status": "Aligned" if abs(user_inputs.get('sqft', 0) - centroid_orig_units.get('sqft', 0)) / max(1, centroid_orig_units.get('sqft', 0)) < 0.3 else "Variance"
            },
            {
                "attribute": "Bedrooms (BHK)",
                "your_property": f"{int(round(user_inputs.get('bhk', 1)))} BHK",
                "cluster_typical": f"{int(round(centroid_orig_units.get('bhk', 1)))} BHK",
                "status": "Identical" if int(round(user_inputs.get('bhk', 1))) == int(round(centroid_orig_units.get('bhk', 1))) else "Adjacent"
            },
            {
                "attribute": "Bathrooms",
                "your_property": f"{int(round(user_inputs.get('numBathrooms', 1)))} Baths",
                "cluster_typical": f"{int(round(centroid_orig_units.get('numBathrooms', 1)))} Baths",
                "status": "Identical" if int(round(user_inputs.get('numBathrooms', 1))) == int(round(centroid_orig_units.get('numBathrooms', 1))) else "Adjacent"
            },
            {
                "attribute": "Furnishing Status",
                "your_property": status_names.get(furn_val, "Unfurnished"),
                "cluster_typical": status_names.get(int(round(centroid_orig_units.get("furnishing_tier", 0))), "Unfurnished"),
                "status": "Identical" if furn_val == int(round(centroid_orig_units.get("furnishing_tier", 0))) else "Different"
            }
        ]

        # Calculate distances to all clusters for transparent geometric comparison
        cluster_distances = []
        for c_id, d in enumerate(distances):
            cluster_distances.append({
                "cluster_id": c_id,
                "segment_name": self.segment_names.get(c_id, f"Cluster {c_id}"),
                "standardized_distance": round(float(d), 4),
                "is_match": c_id == assigned_cluster
            })
        cluster_distances.sort(key=lambda x: x["standardized_distance"])

        return {
            "headline": f"Your property is closest to {matched_segment_name}",
            "matched_cluster_id": assigned_cluster,
            "matched_segment_name": matched_segment_name,
            "distance_to_centroid": round(min_distance, 4),
            "distance_metric": f"Standardized Euclidean Distance in {self.n_dimensions}D Log-Transformed Space",
            "dimensions_evaluated": self.n_dimensions,
            "warnings": validation_warnings,
            "comparison": comparison,
            "all_cluster_distances": cluster_distances,
            "explanation": (
                f"In the {self.n_dimensions}-dimensional standardized feature space, your property's Euclidean "
                f"distance to the '{matched_segment_name}' centroid is {min_distance:.3f}, making it the closest "
                f"fitted cluster."
            )
        }
