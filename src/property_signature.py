"""
PROPERTY SIGNATURE MODULE — Normalized Property Profiles
Translates physical property attributes into normalized relative indicators (0-100% position
within the property universe). Explicitly labeled as relative dataset positions, NOT probabilities.
"""

from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np

from src.utils import format_currency

class PropertySignatureEngine:
    """
    Computes normalized relative signatures for individual properties
    or aggregated segment profiles based on empirical dataset percentiles.
    """
    def __init__(self, reference_df: pd.DataFrame):
        self.df = reference_df
        self.price_col = "price"
        self.area_col = "sqft"
        self.bed_col = "bhk"
        self.bath_col = "numBathrooms"
        self.psqft_col = "price_per_sqft"

        # Precompute empirical bounds for min-max relative normalization
        self.bounds = {}
        for key, col in [
            ("price", self.price_col),
            ("area", self.area_col),
            ("bedrooms", self.bed_col),
            ("bathrooms", self.bath_col),
            ("psqft", self.psqft_col)
        ]:
            if col in self.df.columns:
                s = self.df[col].dropna()
                self.bounds[key] = {
                    "min": float(s.min()),
                    "max": float(s.max()),
                    "p05": float(s.quantile(0.05)),
                    "p95": float(s.quantile(0.95)),
                }

    def _relative_score(self, val: float, key: str) -> float:
        """
        Calculates relative position (0.0 to 1.0) of a value
        clamped against the 5th and 95th percentiles to avoid extreme outlier distortion.
        """
        b = self.bounds.get(key)
        if not b or b["p95"] == b["p05"]:
            return 0.5
        clamped = max(b["p05"], min(float(val), b["p95"]))
        return (clamped - b["p05"]) / (b["p95"] - b["p05"])

    def generate_signature(self, prop_data: Union[pd.Series, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generates a normalized relative Property Signature.
        All scores are strictly relative indicators (0-100% position within dataset).
        """
        def _get_val(col_name: str, fallback: float = 0.0) -> float:
            if isinstance(prop_data, pd.Series):
                return float(prop_data.get(col_name, fallback))
            return float(prop_data.get(col_name, fallback))

        raw_price = _get_val(self.price_col)
        raw_area = _get_val(self.area_col)
        raw_beds = _get_val(self.bed_col, 1.0)
        raw_baths = _get_val(self.bath_col, 1.0)
        raw_psqft = _get_val(self.psqft_col, raw_price / max(1.0, raw_area))

        # Calculate relative indicators (0.0 to 1.0)
        sig_price = self._relative_score(raw_price, "price")
        sig_size = self._relative_score(raw_area, "area")
        sig_beds = self._relative_score(raw_beds, "bedrooms")
        sig_baths = self._relative_score(raw_baths, "bathrooms")
        sig_psqft = self._relative_score(raw_psqft, "psqft")

        dimensions = [
            {
                "dimension": "RENT BUDGET",
                "score_pct": int(round(sig_price * 100)),
                "raw_value": raw_price,
                "formatted_raw": f"₹{raw_price:,.0f}/mo",
                "descriptor": "Relative Monthly Rent Position",
                "bar": self._render_bar(sig_price)
            },
            {
                "dimension": "LIVING SPACE",
                "score_pct": int(round(sig_size * 100)),
                "raw_value": raw_area,
                "formatted_raw": f"{raw_area:,.0f} sqft",
                "descriptor": "Relative Built-Up Area Position",
                "bar": self._render_bar(sig_size)
            },
            {
                "dimension": "BEDROOM CAPACITY",
                "score_pct": int(round(sig_beds * 100)),
                "raw_value": raw_beds,
                "formatted_raw": f"{int(round(raw_beds))} BHK",
                "descriptor": "Relative Bedroom Capacity",
                "bar": self._render_bar(sig_beds)
            },
            {
                "dimension": "BATHROOM PROVISION",
                "score_pct": int(round(sig_baths * 100)),
                "raw_value": raw_baths,
                "formatted_raw": f"{int(round(raw_baths))} Baths",
                "descriptor": "Relative Bathroom Provision",
                "bar": self._render_bar(sig_baths)
            },
            {
                "dimension": "RENT DENSITY",
                "score_pct": int(round(sig_psqft * 100)),
                "raw_value": raw_psqft,
                "formatted_raw": f"₹{raw_psqft:.1f}/sqft",
                "descriptor": "Relative Rent per SqFt Density",
                "bar": self._render_bar(sig_psqft)
            }
        ]

        return {
            "dimensions": dimensions,
            "interpretation_label": "Relative position within dataset (Normalized 0-100% Spectrum)",
            "summary": {d["dimension"]: d["score_pct"] for d in dimensions}
        }

    @staticmethod
    def _render_bar(fraction: float, total_blocks: int = 10) -> str:
        """Render a clean block-style text visual bar."""
        clamped = max(0.0, min(1.0, fraction))
        filled = int(round(clamped * total_blocks))
        empty = total_blocks - filled
        return "█" * filled + "░" * empty
