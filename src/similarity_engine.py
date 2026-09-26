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
        self.original_df = original_df.copy().reset_index(drop=True)
        self.scaler = scaler
        self.feature_names = feature_names
        self.n_dimensions = len(feature_names)
        
        # Fit NearestNeighbors model on standardized feature space
        self.nn_model = NearestNeighbors(
            n_neighbors=min(25, len(scaled_df)),
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
        matches = []
        id_col = "property_id" if "property_id" in self.original_df.columns else "id"

        if id_col in self.original_df.columns:
            try:
                numeric_id = int(property_id)
                matches = self.original_df.index[self.original_df[id_col] == numeric_id].tolist()
            except (ValueError, TypeError):
                pass

            if not matches:
                matches = self.original_df.index[self.original_df[id_col].astype(str) == str(property_id)].tolist()

        if not matches:
            try:
                p_idx = int(property_id)
                if 0 <= p_idx < len(self.original_df):
                    matches = [p_idx]
            except (ValueError, TypeError):
                matches = []

        if not matches:
            raise ValueError(f"Property identifier '{property_id}' not found in dataset.")

        query_idx = matches[0]
        query_scaled = self.scaled_df.iloc[[query_idx]].values
        query_record = self.original_df.iloc[query_idx].to_dict()

        # Query top_n + 5 to ensure enough non-self neighbors
        n_to_query = min(top_n + 5, len(self.scaled_df))
        distances, indices = self.nn_model.kneighbors(query_scaled, n_neighbors=n_to_query)

        neighbor_results = []
        for dist, idx in zip(distances[0], indices[0]):
            # CRITICAL REQUIREMENT: Strictly exclude the query property itself
            if idx == query_idx:
                continue

            neighbor_row = self.original_df.iloc[idx].to_dict()
            euclidean_dist = float(dist)
            norm_factor = np.sqrt(self.n_dimensions)
            similarity_pct = round(100.0 / (1.0 + (euclidean_dist / norm_factor)), 1)

            p_id = neighbor_row.get("property_id", neighbor_row.get("id", idx + 1))
            rent_val = float(neighbor_row.get("price", 0))

            neighbor_results.append({
                "id": int(p_id),
                "property_id": int(p_id),
                "euclidean_distance": round(euclidean_dist, 4),
                "relative_similarity_pct": similarity_pct,
                "price": rent_val,
                "price_formatted": f"₹{rent_val:,.0f}/mo",
                "sqft": float(neighbor_row.get("sqft", 0)),
                "bhk": int(round(float(neighbor_row.get("bhk", 1)))),
                "numBathrooms": float(neighbor_row.get("numBathrooms", 1)),
                "city": str(neighbor_row.get("city_clean", neighbor_row.get("city", ""))),
                "location": str(neighbor_row.get("location", "")),
                "Status": str(neighbor_row.get("Status", "")),
                "typology": str(neighbor_row.get("typology", "")),
                "cluster": int(neighbor_row.get("Cluster", 0)),
                "cluster_name": str(neighbor_row.get("cluster_name", f"Cluster {neighbor_row.get('Cluster', 0)}"))
            })

            if len(neighbor_results) >= top_n:
                break

        query_rent = float(query_record.get("price", 0))
        q_id = query_record.get("property_id", query_record.get("id", query_idx + 1))

        return {
            "query_property": {
                "id": int(q_id),
                "property_id": int(q_id),
                "price": query_rent,
                "price_formatted": f"₹{query_rent:,.0f}/mo",
                "sqft": float(query_record.get("sqft", 0)),
                "bhk": int(round(float(query_record.get("bhk", 1)))),
                "numBathrooms": float(query_record.get("numBathrooms", 1)),
                "city": str(query_record.get("city_clean", query_record.get("city", ""))),
                "location": str(query_record.get("location", "")),
                "Status": str(query_record.get("Status", "")),
                "typology": str(query_record.get("typology", "")),
                "cluster": int(query_record.get("Cluster", 0)),
            },
            "similarity_metric": f"Standardized Euclidean Distance in {self.n_dimensions}D Space",
            "dimensions_evaluated": self.n_dimensions,
            "similar_properties": neighbor_results
        }
