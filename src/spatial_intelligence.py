"""
SPATIAL INTELLIGENCE MODULE — Geographic Distribution of Property Segments
Validates empirical coordinates, documents provenance without fabricated geographic claims,
and enables multi-attribute spatial exploration and descriptive segment mapping.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

class SpatialIntelligence:
    """
    Validates geographic coordinates and computes descriptive spatial patterns
    across discovered property communities.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.lat_col = "Lattitude" if "Lattitude" in self.df.columns else ("latitude" if "latitude" in self.df.columns else None)
        self.lon_col = "Longitude" if "Longitude" in self.df.columns else ("longitude" if "longitude" in self.df.columns else None)
        self.is_valid_spatial = self._validate_coordinates()

    def _validate_coordinates(self) -> bool:
        """Verify presence of valid non-null numerical coordinates."""
        if not self.lat_col or not self.lon_col:
            return False
        valid = self.df[[self.lat_col, self.lon_col]].dropna()
        if len(valid) == 0:
            return False
        # Check that coordinates are in valid lat/lon numeric bounds
        lat_valid = valid[self.lat_col].between(-90, 90).all()
        lon_valid = valid[self.lon_col].between(-180, 180).all()
        return bool(lat_valid and lon_valid)

    def get_provenance_audit(self) -> Dict[str, Any]:
        """
        Provides factual geographical bounding box and explicit provenance documentation
        to ensure zero fabricated location claims.
        """
        if not self.is_valid_spatial:
            return {
                "valid": False,
                "message": "Valid geographical coordinates are not present in this dataset."
            }

        valid_df = self.df[[self.lat_col, self.lon_col]].dropna()
        min_lat = float(valid_df[self.lat_col].min())
        max_lat = float(valid_df[self.lat_col].max())
        min_lon = float(valid_df[self.lon_col].min())
        max_lon = float(valid_df[self.lon_col].max())
        mean_lat = float(valid_df[self.lat_col].mean())
        mean_lon = float(valid_df[self.lon_col].mean())

        provenance_statement = (
            f"Coordinate verification confirms valid spatial coordinates spanning "
            f"Latitude {min_lat:.4f}° to {max_lat:.4f}° N and "
            f"Longitude {min_lon:.4f}° to {max_lon:.4f}° W. "
            f"Although widely hosted under the title 'House Price India.csv', this recorded bounding box "
            f"corresponds geographically to North America (approx. 52.8°N, -114.4°W). "
            f"Our system visualizes and analyzes the true empirical coordinates as recorded in the data "
            f"without asserting speculative or fabricated geographic identities."
        )

        return {
            "valid": True,
            "record_count": len(valid_df),
            "bounds": {
                "min_lat": min_lat,
                "max_lat": max_lat,
                "min_lon": min_lon,
                "max_lon": max_lon,
                "center_lat": mean_lat,
                "center_lon": mean_lon,
            },
            "provenance_statement": provenance_statement
        }

    def filter_spatial_inventory(
        self,
        segment_filter: Optional[List[int]] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        min_area: Optional[float] = None,
        max_area: Optional[float] = None,
        sample_limit: int = 4000
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Applies multi-attribute filters to produce map-ready property coordinates
        and factual descriptive spatial metrics.
        """
        filtered = self.df.dropna(subset=[self.lat_col, self.lon_col]).copy()

        # Rename to standard lowercase for Streamlit st.map compatibility
        filtered["latitude"] = filtered[self.lat_col]
        filtered["longitude"] = filtered[self.lon_col]

        if segment_filter is not None and len(segment_filter) > 0 and "Cluster" in filtered.columns:
            filtered = filtered[filtered["Cluster"].isin(segment_filter)]

        if min_price is not None and "Price" in filtered.columns:
            filtered = filtered[filtered["Price"] >= min_price]
        if max_price is not None and "Price" in filtered.columns:
            filtered = filtered[filtered["Price"] <= max_price]

        if min_area is not None and "living area" in filtered.columns:
            filtered = filtered[filtered["living area"] >= min_area]
        if max_area is not None and "living area" in filtered.columns:
            filtered = filtered[filtered["living area"] <= max_area]

        total_matching = len(filtered)

        # Descriptive statistics of filtered cohort
        composition = {}
        if "Segment Name" in filtered.columns and total_matching > 0:
            counts = filtered["Segment Name"].value_counts()
            for s_name, count in counts.items():
                composition[s_name] = {
                    "count": int(count),
                    "share_pct": round((count / total_matching) * 100.0, 1)
                }

        avg_price = float(filtered["Price"].mean()) if "Price" in filtered.columns and total_matching > 0 else 0.0
        avg_area = float(filtered["living area"].mean()) if "living area" in filtered.columns and total_matching > 0 else 0.0

        stats = {
            "total_matching": total_matching,
            "sample_displayed": min(total_matching, sample_limit),
            "composition": composition,
            "avg_price_filtered": avg_price,
            "avg_area_filtered": avg_area,
        }

        # Subsample for responsive browser mapping if cohort is large
        if total_matching > sample_limit:
            filtered_sample = filtered.sample(n=sample_limit, random_state=config.RANDOM_STATE)
        else:
            filtered_sample = filtered

        return filtered_sample, stats
