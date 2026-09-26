"""
SEGMENT DNA MODULE — Cluster Profiling & Dynamic Segment Archetypes
Extracts multidimensional statistical profiles for each discovered property community.
Generates descriptive, data-grounded segment names and radar metrics directly from empirical centroids.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

import config
from src.utils import format_currency

class SegmentDNAProfiler:
    """
    Extracts statistical DNA profiles and generates dynamic, meaningful segment archetypes
    for each discovered property community.
    """
    def __init__(self, clustered_df: pd.DataFrame):
        self.df = clustered_df.copy()
        self.cluster_col = "Cluster"

    def generate_segment_names(self) -> Dict[int, str]:
        """
        Dynamically derives descriptive segment names based on actual statistical centroids.
        Evaluates rent level, living area, room count, and furnishing status.
        """
        clusters = sorted(self.df[self.cluster_col].unique())
        names = {}

        # Compute median statistics per cluster to inform naming
        profiles = []
        for c in clusters:
            sub = self.df[self.df[self.cluster_col] == c]
            med_rent = float(sub["price"].median())
            med_sqft = float(sub["sqft"].median())
            med_bhk = float(sub["bhk"].median())
            med_baths = float(sub["numBathrooms"].median())
            pct_furnished = float((sub["furnishing_tier"] == 2).mean() * 100)
            profiles.append({
                "cluster": c,
                "rent": med_rent,
                "sqft": med_sqft,
                "bhk": med_bhk,
                "baths": med_baths,
                "pct_furnished": pct_furnished
            })

        # Sort clusters by rent to assign intuitive tiered names
        sorted_by_rent = sorted(profiles, key=lambda x: x["rent"])
        total_clusters = len(sorted_by_rent)

        for rank, p in enumerate(sorted_by_rent):
            c = p["cluster"]
            rent = p["rent"]
            sqft = p["sqft"]
            bhk = p["bhk"]
            pct_furn = p["pct_furnished"]

            if rent >= 200_000 or (bhk >= 4 and sqft >= 3_500):
                name = "Luxury Estates & Penthouses"
            elif rank == 0:
                if bhk <= 1.5 and rent <= 22_000:
                    name = "Affordable Compact Living"
                else:
                    name = "Budget Urban Rentals"
            elif pct_furn >= 75.0:
                name = "Turnkey Executive Suites"
            elif bhk >= 3.0 and rent >= 50_000:
                name = "Premium Family Residences"
            elif bhk >= 2.0 and rent <= 35_000:
                name = "Mid-Market Urban Homes"
            elif rank == total_clusters - 1:
                name = "High-End Prestige Living"
            elif rank == 1:
                name = "Standard City Residences"
            else:
                name = f"Urban Rental Tier {rank + 1}"

            # Ensure uniqueness
            base_name = name
            counter = 2
            while name in names.values():
                name = f"{base_name} ({counter})"
                counter += 1
            names[c] = name

        return names

    def extract_segment_dna(self, segment_names: Optional[Dict[int, str]] = None) -> Dict[int, Dict[str, Any]]:
        """
        Calculates comprehensive statistical indicators for each property community.
        """
        if segment_names is None:
            segment_names = self.generate_segment_names()

        clusters = sorted(self.df[self.cluster_col].unique())
        total_properties = len(self.df)
        dna = {}

        # Precompute global medians for relative comparison
        global_rent = float(self.df["price"].median())
        global_sqft = float(self.df["sqft"].median())
        global_psqft = float(self.df["price_per_sqft"].median())

        for c in clusters:
            sub = self.df[self.df[self.cluster_col] == c]
            count = len(sub)
            share_pct = round((count / total_properties) * 100, 1)

            # Price Metrics (Monthly Rent ₹)
            price_s = sub["price"]
            med_rent = float(price_s.median())
            mean_rent = float(price_s.mean())
            q25_rent = float(price_s.quantile(0.25))
            q75_rent = float(price_s.quantile(0.75))

            # Area Metrics (sqft)
            area_s = sub["sqft"]
            med_sqft = float(area_s.median())
            mean_sqft = float(area_s.mean())

            # Price per SqFt
            psqft_s = sub["price_per_sqft"]
            med_psqft = float(psqft_s.median())
            mean_psqft = float(psqft_s.mean())

            # Bedrooms & Bathrooms
            med_bhk = float(sub["bhk"].median())
            mode_bhk = float(sub["bhk"].mode()[0]) if len(sub["bhk"].mode()) > 0 else med_bhk
            med_baths = float(sub["numBathrooms"].median())

            # Furnishing Status distribution
            furn_counts = sub["Status"].value_counts().to_dict()
            pct_unfurnished = round(float((sub["furnishing_tier"] == 0).mean() * 100), 1)
            pct_semi = round(float((sub["furnishing_tier"] == 1).mean() * 100), 1)
            pct_furnished = round(float((sub["furnishing_tier"] == 2).mean() * 100), 1)

            # City distribution
            city_counts = sub["city_clean"].value_counts().to_dict()
            city_pcts = {city: round(cnt / count * 100, 1) for city, cnt in city_counts.items()}

            # Typology distribution
            typology_counts = sub["typology"].value_counts().to_dict()
            top_typologies = list(typology_counts.keys())[:3]

            # Security Deposit
            med_deposit = float(sub["deposit_clean"].median())
            pct_no_deposit = round(float((sub["deposit_clean"] == 0).mean() * 100), 1)

            # Normalized Radar Spectrum (0 to 100)
            # Clamped relative to empirical bounds for fair visual comparison
            radar_rent = min(100, max(5, int(round((np.log1p(med_rent) / np.log1p(2_000_000)) * 100))))
            radar_space = min(100, max(5, int(round((np.log1p(med_sqft) / np.log1p(10_000)) * 100))))
            radar_beds = min(100, max(10, int(round((med_bhk / 5.0) * 100))))
            radar_baths = min(100, max(10, int(round((med_baths / 5.0) * 100))))
            radar_density = min(100, max(5, int(round((med_psqft / 150.0) * 100))))
            radar_furnishing = int(round(pct_furnished))

            radar_axes = [
                {"axis": "Rent Budget", "score": radar_rent},
                {"axis": "Living Area", "score": radar_space},
                {"axis": "Bedrooms", "score": radar_beds},
                {"axis": "Bathrooms", "score": radar_baths},
                {"axis": "Rent / SqFt", "score": radar_density},
                {"axis": "Furnishing", "score": radar_furnishing}
            ]

            dna[c] = {
                "cluster_id": c,
                "segment_name": segment_names.get(c, f"Cluster {c}"),
                "property_count": count,
                "market_share_pct": share_pct,
                "price": {
                    "median": med_rent,
                    "mean": mean_rent,
                    "q25": q25_rent,
                    "q75": q75_rent,
                    "formatted_median": format_currency(med_rent),
                    "formatted_mean": format_currency(mean_rent),
                    "vs_market": f"{((med_rent - global_rent) / global_rent * 100):+.1f}% vs market median"
                },
                "area": {
                    "median": med_sqft,
                    "mean": mean_sqft,
                    "formatted": f"{med_sqft:,.0f} sqft",
                    "vs_market": f"{((med_sqft - global_sqft) / global_sqft * 100):+.1f}% vs market median"
                },
                "price_per_sqft": {
                    "median": med_psqft,
                    "mean": mean_psqft,
                    "formatted": f"₹{med_psqft:.1f}/sqft"
                },
                "specs": {
                    "typical_bhk": int(round(med_bhk)),
                    "bhk_mode": int(round(mode_bhk)),
                    "typical_bathrooms": int(round(med_baths)),
                    "bhk_display": f"{int(round(med_bhk))} BHK",
                    "baths_display": f"{int(round(med_baths))} Baths"
                },
                "furnishing": {
                    "pct_furnished": pct_furnished,
                    "pct_semi": pct_semi,
                    "pct_unfurnished": pct_unfurnished,
                    "dominant_status": sub["Status"].mode()[0] if len(sub["Status"].mode()) > 0 else "Unfurnished"
                },
                "geography": {
                    "city_distribution": city_pcts,
                    "dominant_city": sub["city_clean"].mode()[0] if len(sub["city_clean"].mode()) > 0 else "Delhi",
                    "top_localities": sub["location"].value_counts().head(5).to_dict()
                },
                "typology": {
                    "distribution": typology_counts,
                    "top_typologies": top_typologies
                },
                "deposit": {
                    "median": med_deposit,
                    "pct_no_deposit": pct_no_deposit,
                    "formatted": format_currency(med_deposit) if med_deposit > 0 else "No Deposit"
                },
                "radar_axes": radar_axes
            }

        return dna
