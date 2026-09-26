"""
SIMILARITY ENGINE MODULE — N-Dimensional Property Similarity Space
Constructs the standardized similarity space using the fitted StandardScaler.
Implements nearest-neighbor property retrieval using standardized Euclidean distance.
Strictly excludes the query property itself from the results.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

class SimilarityEngine:
    """
    Manages the N-dimensional standardized similarity space.
    Enables exploratory nearest-neighbor discovery across housing records.
    """
    def __init__(
        self,
        scaled_df: pd.DataFrame,
        original_df: pd.DataFrame,
        scaler: StandardScaler,
        feature_names: List[str]
    ):
        self.scaled_df = scaled_df
        self.original_df = original_df
        self.scaler = scaler
        self.feature_names = feature_names
        self.n_dimensions = len(feature_names)
        
        # Fit NearestNeighbors model on standardized feature space
        self.nn_model = NearestNeighbors(
            n_neighbors=min(20, len(scaled_df)),
            metric="euclidean",
            algorithm="auto"
        )
        self.nn_model.fit(self.scaled_df.values)

    def find_similar_properties(
        self,
        property_id: Any,
        top_n: int = 5
    ) -> Dict[str, Any]:
        """
        Finds the top N most similar properties to a selected property
        in standardized feature space.
        
        CRITICAL: Excludes the query property itself from the returned neighbors.
        """
        # Locate query property index in original DataFrame
        if "id" in self.original_df.columns:
            matches = self.original_df.index[self.original_df["id"] == property_id].tolist()
            if not matches:
                # Try integer index fallback
                try:
                    p_idx = int(property_id)
                    if p_idx in self.original_df.index:
                        matches = [p_idx]
                except (ValueError, TypeError):
                    pass
        else:
            matches = [int(property_id)] if int(property_id) in self.original_df.index else []

        if not matches:
            raise ValueError(f"Property identifier '{property_id}' not found in dataset.")

        query_idx = matches[0]
        query_scaled = self.scaled_df.loc[[query_idx]].values
        query_record = self.original_df.loc[query_idx].to_dict()

        # Query top_n + 1 to account for self-match
        n_to_query = min(top_n + 5, len(self.scaled_df))
        distances, indices = self.nn_model.kneighbors(query_scaled, n_neighbors=n_to_query)

        neighbor_results = []
        for dist, idx in zip(distances[0], indices[0]):
            # CRITICAL REQUIREMENT: Exclude the query property itself
            if idx == query_idx:
                continue

            neighbor_row = self.original_df.iloc[idx].to_dict()
            # Standardized Euclidean distance
            euclidean_dist = float(dist)
            # Normalized similarity index in [0, 1] based on characteristic dimensional scale
            norm_factor = np.sqrt(self.n_dimensions)
            similarity_pct = round(100.0 / (1.0 + (euclidean_dist / norm_factor)), 1)

            neighbor_results.append({
                "index": int(idx),
                "id": neighbor_row.get("id", idx),
                "euclidean_distance": round(euclidean_dist, 3),
                "similarity_index_pct": similarity_pct,
                "price": float(neighbor_row.get("Price", 0)),
                "living_area": float(neighbor_row.get("living area", 0)),
                "bedrooms": float(neighbor_row.get("number of bedrooms", 0)),
                "bathrooms": float(neighbor_row.get("number of bathrooms", 0)),
                "grade": float(neighbor_row.get("grade of the house", 0)),
                "condition": float(neighbor_row.get("condition of the house", 0)),
                "built_year": int(neighbor_row.get("Built Year", 0)),
                "cluster": int(neighbor_row.get("Cluster", 0)) if "Cluster" in neighbor_row else None,
                "segment_name": str(neighbor_row.get("Segment Name", "N/A")),
            })

            if len(neighbor_results) >= top_n:
                break

        return {
            "query_property": {
                "index": int(query_idx),
                "id": query_record.get("id", query_idx),
                "price": float(query_record.get("Price", 0)),
                "living_area": float(query_record.get("living area", 0)),
                "bedrooms": float(query_record.get("number of bedrooms", 0)),
                "bathrooms": float(query_record.get("number of bathrooms", 0)),
                "grade": float(query_record.get("grade of the house", 0)),
                "built_year": int(query_record.get("Built Year", 0)),
                "segment_name": str(query_record.get("Segment Name", "N/A")),
            },
            "similarity_space_dimensions": self.n_dimensions,
            "similar_properties": neighbor_results
        }
