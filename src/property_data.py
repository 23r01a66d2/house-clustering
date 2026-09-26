"""
PROPERTY DATA MODULE — Property Universe Management
Loads and audits the Indian rental property inventory dynamically.
Computes complete empirical spectrums without hardcoded figures.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np

import config
from src.utils import find_column_by_role

class PropertyDataLoader:
    """
    Loads processed Indian rental housing records and computes high-level inventory statistics
    for the Property Universe stage.
    """
    def __init__(self, csv_path: Optional[Path] = None):
        self.csv_path = csv_path or config.RAW_DATA_PATH
        self.raw_df: Optional[pd.DataFrame] = None
        self.col_roles: Dict[str, Optional[str]] = {}

    def load_data(self) -> pd.DataFrame:
        """Load dataset from disk with integrity verification."""
        if not self.csv_path.exists():
            # Trigger data pipeline if processed file is missing
            from src.data_pipeline import run_data_pipeline
            run_data_pipeline(config.RAW_DATA_DIR, self.csv_path)

        self.raw_df = pd.read_csv(self.csv_path)
        self._detect_roles()
        return self.raw_df

    def _detect_roles(self):
        """Map canonical roles to actual DataFrame columns dynamically."""
        if self.raw_df is not None:
            for role in config.FEATURE_ROLES:
                self.col_roles[role] = find_column_by_role(self.raw_df, role)

    def get_universe_metrics(self) -> Dict[str, Any]:
        """
        Dynamically calculate inventory spectrums and summary indicators
        directly from the actual dataset. Zero hardcoded figures.
        """
        if self.raw_df is None:
            self.load_data()

        df = self.raw_df
        total_properties = len(df)
        total_attributes = len(df.columns)

        price_col = self.col_roles.get("price") or "price"
        area_col = self.col_roles.get("living_area") or "sqft"
        bed_col = self.col_roles.get("bedrooms") or "bhk"
        bath_col = self.col_roles.get("bathrooms") or "numBathrooms"
        city_col = self.col_roles.get("city") or "city_clean"
        furnish_col = "furnishing_tier"
        psqft_col = "price_per_sqft"

        # 1. Price Spectrum (Monthly Rent ₹)
        if price_col in df.columns:
            price_s = df[price_col].dropna()
            price_metrics = {
                "min": float(price_s.min()),
                "max": float(price_s.max()),
                "mean": float(price_s.mean()),
                "median": float(price_s.median()),
                "q25": float(price_s.quantile(0.25)),
                "q75": float(price_s.quantile(0.75)),
            }
        else:
            price_metrics = {"min": 0, "max": 0, "mean": 0, "median": 0, "q25": 0, "q75": 0}

        # 2. Living Area Spectrum (sqft)
        if area_col in df.columns:
            area_s = df[area_col].dropna()
            area_metrics = {
                "min": float(area_s.min()),
                "max": float(area_s.max()),
                "mean": float(area_s.mean()),
                "median": float(area_s.median()),
                "q25": float(area_s.quantile(0.25)),
                "q75": float(area_s.quantile(0.75)),
            }
        else:
            area_metrics = {"min": 0, "max": 0, "mean": 0, "median": 0, "q25": 0, "q75": 0}

        # 3. Price per Sqft Spectrum
        if psqft_col in df.columns:
            psqft_s = df[psqft_col].dropna()
            psqft_metrics = {
                "min": float(psqft_s.min()),
                "max": float(psqft_s.max()),
                "mean": float(psqft_s.mean()),
                "median": float(psqft_s.median()),
            }
        else:
            psqft_metrics = {"min": 0, "max": 0, "mean": 0, "median": 0}

        # 4. Bedroom and Bathroom Distributions
        bed_counts = df[bed_col].value_counts().sort_index().to_dict() if bed_col in df.columns else {}
        avg_beds = float(df[bed_col].mean()) if bed_col in df.columns else 0.0
        avg_baths = float(df[bath_col].mean()) if bath_col in df.columns else 0.0

        # 5. City and Locality Counts
        city_counts = df[city_col].value_counts().to_dict() if city_col in df.columns else {}
        locality_count = int(df["location"].nunique()) if "location" in df.columns else 0

        # 6. Furnishing Breakdown
        status_counts = df["Status"].value_counts().to_dict() if "Status" in df.columns else {}

        # 7. Spatial Coordinates Coverage
        has_coords = False
        coord_bounds = {}
        if "latitude" in df.columns and "longitude" in df.columns:
            has_coords = True
            # Restrict bounds calculation to valid coordinates
            valid_mask = df["is_valid_coord"] if "is_valid_coord" in df.columns else pd.Series(True, index=df.index)
            valid_coords = df[valid_mask][["latitude", "longitude"]].dropna()
            coord_bounds = {
                "count": len(valid_coords),
                "total_rows": len(df),
                "valid_pct": round(len(valid_coords) / len(df) * 100, 2),
                "min_lat": float(valid_coords["latitude"].min()),
                "max_lat": float(valid_coords["latitude"].max()),
                "min_lon": float(valid_coords["longitude"].min()),
                "max_lon": float(valid_coords["longitude"].max()),
                "mean_lat": float(valid_coords["latitude"].mean()),
                "mean_lon": float(valid_coords["longitude"].mean()),
            }

        return {
            "total_properties": total_properties,
            "total_attributes": total_attributes,
            "attribute_names": list(df.columns),
            "price_metrics": price_metrics,
            "area_metrics": area_metrics,
            "psqft_metrics": psqft_metrics,
            "bedroom_counts": bed_counts,
            "avg_bedrooms": round(avg_beds, 2),
            "avg_bathrooms": round(avg_baths, 2),
            "city_counts": city_counts,
            "locality_count": locality_count,
            "furnishing_counts": status_counts,
            "has_coordinates": has_coords,
            "coordinate_bounds": coord_bounds,
            "column_roles": self.col_roles,
        }
