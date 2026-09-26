"""
SEGMENT DNA MODULE — Visual Fingerprints and Dynamic Segment Profiling
Extracts empirical cluster statistics, assigns data-driven market segment names,
and generates visual Segment DNA fingerprints using normalized relative indicators.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

import config
from src.utils import find_column_by_role
from src.property_signature import PropertySignatureEngine

class SegmentDNAProfiler:
    """
    Analyzes discovered property communities to produce data-driven names,
    typical metric summaries, and normalized visual DNA fingerprints.
    """
    def __init__(self, clustered_df: pd.DataFrame):
        self.df = clustered_df.copy()
        self.price_col = find_column_by_role(self.df, "price") or "Price"
        self.area_col = find_column_by_role(self.df, "living_area") or "living area"
        self.lot_col = find_column_by_role(self.df, "lot_area") or "lot area"
        self.bed_col = find_column_by_role(self.df, "bedrooms") or "number of bedrooms"
        self.bath_col = find_column_by_role(self.df, "bathrooms") or "number of bathrooms"
        self.floor_col = find_column_by_role(self.df, "floors") or "number of floors"
        self.grade_col = find_column_by_role(self.df, "grade") or "grade of the house"
        self.cond_col = find_column_by_role(self.df, "condition") or "condition of the house"
        self.year_col = find_column_by_role(self.df, "built_year") or "Built Year"
        
        self.sig_engine = PropertySignatureEngine(self.df)
        self.overall_means = {
            "price": float(self.df[self.price_col].mean()),
            "area": float(self.df[self.area_col].mean()),
            "bedrooms": float(self.df[self.bed_col].mean()) if self.bed_col in self.df.columns else 3.0,
            "bathrooms": float(self.df[self.bath_col].mean()) if self.bath_col in self.df.columns else 2.0,
            "grade": float(self.df[self.grade_col].mean()) if self.grade_col in self.df.columns else 7.0,
            "built_year": float(self.df[self.year_col].mean()) if self.year_col in self.df.columns else 1970.0,
        }

    def generate_segment_names(self) -> Dict[int, str]:
        """
        Dynamically derives descriptive segment names based on actual relative
        percentiles of price, living area, grade, and built year.
        Zero hardcoded labels.
        """
        cluster_ids = sorted(self.df["Cluster"].unique())
        means = self.df.groupby("Cluster")[[self.price_col, self.area_col]].mean()
        
        price_ranks = means[self.price_col].rank(ascending=True)
        area_ranks = means[self.area_col].rank(ascending=True)
        k = len(cluster_ids)
        
        names = {}
        for c_id in cluster_ids:
            p_rank = price_ranks.loc[c_id]
            a_rank = area_ranks.loc[c_id]
            
            c_data = self.df[self.df["Cluster"] == c_id]
            avg_grade = c_data[self.grade_col].mean() if self.grade_col in c_data.columns else 7.0
            avg_year = c_data[self.year_col].mean() if self.year_col in c_data.columns else 1970.0

            if k == 2:
                if p_rank == 2:
                    names[c_id] = "Higher-Priced Spacious Properties"
                else:
                    names[c_id] = "Lower-Priced Compact Properties"
            elif k == 3:
                if p_rank == 3:
                    names[c_id] = "Premium Luxury Residences"
                elif p_rank == 2:
                    names[c_id] = "Mid-Tier Balanced Family Homes"
                else:
                    names[c_id] = "Economical Compact Properties"
            elif k == 4:
                if p_rank == 4:
                    names[c_id] = "High-End Luxury Estates"
                elif p_rank == 3:
                    names[c_id] = "Upper-Middle Family Residences"
                elif p_rank == 2:
                    names[c_id] = "Standard Suburban Dwellings"
                else:
                    names[c_id] = "Budget Compact Units"
            else:
                p_ratio = means.loc[c_id, self.price_col] / self.overall_means["price"]
                a_ratio = means.loc[c_id, self.area_col] / self.overall_means["area"]
                if p_ratio > 1.3:
                    p_desc = "Premium High-Value"
                elif p_ratio < 0.8:
                    p_desc = "Value-Oriented"
                else:
                    p_desc = "Mid-Market"
                    
                if a_ratio > 1.2:
                    a_desc = "Expansive Properties"
                elif a_ratio < 0.85:
                    a_desc = "Compact Properties"
                else:
                    a_desc = "Balanced Properties"
                names[c_id] = f"{p_desc} {a_desc}"

        return names

    def extract_segment_dna(self, segment_names: Optional[Dict[int, str]] = None) -> List[Dict[str, Any]]:
        """
        Builds the complete Segment DNA portfolio including relative indicator bars,
        typical traits, market share, and comparative profiles.
        """
        if segment_names is None:
            segment_names = self.generate_segment_names()

        total_houses = len(self.df)
        dna_profiles = []

        for c_id in sorted(self.df["Cluster"].unique()):
            c_df = self.df[self.df["Cluster"] == c_id]
            count = len(c_df)
            share_pct = round((count / total_houses) * 100.0, 1)

            # Empirical averages
            avg_price = float(c_df[self.price_col].mean())
            med_price = float(c_df[self.price_col].median())
            avg_area = float(c_df[self.area_col].mean())
            avg_beds = float(c_df[self.bed_col].mean()) if self.bed_col in c_df.columns else 0.0
            avg_baths = float(c_df[self.bath_col].mean()) if self.bath_col in c_df.columns else 0.0
            avg_grade = float(c_df[self.grade_col].mean()) if self.grade_col in c_df.columns else 0.0
            avg_cond = float(c_df[self.cond_col].mean()) if self.cond_col in c_df.columns else 0.0
            avg_year = float(c_df[self.year_col].mean()) if self.year_col in c_df.columns else 0.0
            avg_lot = float(c_df[self.lot_col].mean()) if self.lot_col in c_df.columns else 0.0

            # Directional relative indicators against overall mean
            p_dir = "↑ Substantially Higher" if avg_price > self.overall_means["price"] * 1.15 else ("↓ Economical / Lower" if avg_price < self.overall_means["price"] * 0.85 else "→ Typical Market Mean")
            a_dir = "↑ Expansive / Spacious" if avg_area > self.overall_means["area"] * 1.15 else ("↓ Compact Living" if avg_area < self.overall_means["area"] * 0.85 else "→ Standard Scale")
            b_dir = "↑ Higher Bedroom Capacity" if avg_beds > self.overall_means["bedrooms"] * 1.1 else ("↓ Compact Layout" if avg_beds < self.overall_means["bedrooms"] * 0.9 else "→ Typical Family Layout")
            g_dir = "↑ Superior Build Grade" if avg_grade > self.overall_means["grade"] * 1.05 else ("↓ Standard / Economy Build" if avg_grade < self.overall_means["grade"] * 0.95 else "→ Moderate Build Quality")

            # Generate normalized visual fingerprint using PropertySignatureEngine
            typical_vector = {
                self.price_col: avg_price,
                self.area_col: avg_area,
                self.bed_col: avg_beds,
                self.bath_col: avg_baths,
                self.grade_col: avg_grade,
                self.year_col: avg_year
            }
            sig = self.sig_engine.generate_signature(typical_vector)

            # Build narrative
            name = segment_names.get(c_id, f"Segment {c_id}")
            narrative = (
                f"{name} accounts for {count:,} properties ({share_pct}% of analyzed inventory). "
                f"Properties in this segment exhibit an average price of ${avg_price:,.0f} "
                f"(median ${med_price:,.0f}) with typical living quarters spanning {avg_area:,.0f} sqft. "
                f"Architectural profiles typically offer {avg_beds:.1f} bedrooms and {avg_baths:.2f} bathrooms "
                f"with an average construction grade of {avg_grade:.1f}/13 and average build year around {int(avg_year)}."
            )

            dna_profiles.append({
                "cluster_id": int(c_id),
                "segment_name": name,
                "share_pct": share_pct,
                "property_count": count,
                "avg_price": avg_price,
                "median_price": med_price,
                "avg_living_area": avg_area,
                "avg_lot_area": avg_lot,
                "avg_bedrooms": avg_beds,
                "avg_bathrooms": avg_baths,
                "avg_grade": avg_grade,
                "avg_condition": avg_cond,
                "avg_built_year": avg_year,
                "traits": {
                    "Price": p_dir,
                    "Area": a_dir,
                    "Bedrooms": b_dir,
                    "Grade": g_dir,
                },
                "fingerprint_dimensions": sig["dimensions"],
                "narrative": narrative
            })

        return dna_profiles

    def get_comparison_table(self, segment_names: Optional[Dict[int, str]] = None) -> pd.DataFrame:
        """Generates a side-by-side comparative profiling table across all segments."""
        profiles = self.extract_segment_dna(segment_names)
        data = {}
        for p in profiles:
            col_name = f"Segment {p['cluster_id']}: {p['segment_name']}"
            data[col_name] = {
                "Market Share": f"{p['share_pct']}% ({p['property_count']:,} homes)",
                "Average Price": f"${p['avg_price']:,.0f}",
                "Median Price": f"${p['median_price']:,.0f}",
                "Average Living Area": f"{p['avg_living_area']:,.0f} sqft",
                "Average Lot Area": f"{p['avg_lot_area']:,.0f} sqft",
                "Average Bedrooms": f"{p['avg_bedrooms']:.2f}",
                "Average Bathrooms": f"{p['avg_bathrooms']:.2f}",
                "Average Grade": f"{p['avg_grade']:.2f} / 13",
                "Average Condition": f"{p['avg_condition']:.2f} / 5",
                "Typical Built Year": f"{int(p['avg_built_year'])}",
            }
        return pd.DataFrame(data)
