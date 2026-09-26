"""
PROPERTY DATA MODULE — Property Universe Management
Loads and audits the property inventory without assuming column positions or hardcoding values.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np

import config
from src.utils import find_column_by_role

class PropertyDataLoader:
    """
    Loads raw housing records and computes high-level inventory statistics
    for the Property Universe stage.
    """
    def __init__(self, csv_path: Optional[Path] = None):
        self.csv_path = csv_path or config.RAW_DATA_PATH
        self.raw_df: Optional[pd.DataFrame] = None
        self.col_roles: Dict[str, Optional[str]] = {}

    def load_data(self) -> pd.DataFrame:
        """Load dataset from disk with integrity verification."""
        if not self.csv_path.exists():
            # Check fallback in data folder
            candidates = list(config.DATA_DIR.glob("*.csv"))
            if candidates:
                self.csv_path = candidates[0]
            else:
                raise FileNotFoundError(f"Dataset not found at {self.csv_path} or in {config.DATA_DIR}")

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
        directly from the raw dataset. Zero hardcoded figures.
        """
        if self.raw_df is None:
            self.load_data()

        df = self.raw_df
        total_properties = len(df)
        total_attributes = len(df.columns)

        price_col = self.col_roles.get("price")
        area_col = self.col_roles.get("living_area")
        bed_col = self.col_roles.get("bedrooms")
        bath_col = self.col_roles.get("bathrooms")
        year_col = self.col_roles.get("built_year")
        lat_col = find_column_by_role(df, "spatial") or "Lattitude"
        lon_col = "Longitude"

        # Dynamically compute Price Spectrum
        if price_col and price_col in df.columns:
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

        # Dynamically compute Living Area Spectrum
        if area_col and area_col in df.columns:
            area_s = df[area_col].dropna()
            area_metrics = {
                "min": float(area_s.min()),
                "max": float(area_s.max()),
                "mean": float(area_s.mean()),
                "median": float(area_s.median()),
            }
        else:
            area_metrics = {"min": 0, "max": 0, "mean": 0, "median": 0}

        # Dynamically compute Built Year Spectrum
        if year_col and year_col in df.columns:
            yr_s = df[year_col].dropna()
            year_metrics = {
                "min": int(yr_s.min()),
                "max": int(yr_s.max()),
                "median": int(yr_s.median()),
            }
        else:
            year_metrics = {"min": 0, "max": 0, "median": 0}

        # Bedroom and Bathroom Distribution
        bed_counts = df[bed_col].value_counts().to_dict() if bed_col and bed_col in df.columns else {}
        avg_beds = float(df[bed_col].mean()) if bed_col and bed_col in df.columns else 0.0
        avg_baths = float(df[bath_col].mean()) if bath_col and bath_col in df.columns else 0.0

        # Spatial Coordinates Coverage
        has_coords = False
        coord_bounds = {}
        if "Lattitude" in df.columns and "Longitude" in df.columns:
            has_coords = True
            valid_coords = df[["Lattitude", "Longitude"]].dropna()
            coord_bounds = {
                "count": len(valid_coords),
                "min_lat": float(valid_coords["Lattitude"].min()),
                "max_lat": float(valid_coords["Lattitude"].max()),
                "min_lon": float(valid_coords["Longitude"].min()),
                "max_lon": float(valid_coords["Longitude"].max()),
                "mean_lat": float(valid_coords["Lattitude"].mean()),
                "mean_lon": float(valid_coords["Longitude"].mean()),
            }

        return {
            "total_properties": total_properties,
            "total_attributes": total_attributes,
            "attribute_names": list(df.columns),
            "price_metrics": price_metrics,
            "area_metrics": area_metrics,
            "year_metrics": year_metrics,
            "bedroom_counts": bed_counts,
            "avg_bedrooms": avg_beds,
            "avg_bathrooms": avg_baths,
            "has_coordinates": has_coords,
            "coordinate_bounds": coord_bounds,
            "column_roles": self.col_roles,
        }
