"""
INSIGHT ENGINE MODULE — Market Insights, Architecture Visualizer, and Exports
Synthesizes high-level property intelligence, formats the Core Architecture diagram,
and packages analytical deliverables for export.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
from pathlib import Path

import config
from src.utils import format_currency

class InsightEngine:
    """
    Synthesizes executive market insights, outputs architecture diagrams,
    and manages export artifacts.
    """
    @staticmethod
    def get_architecture_diagram(n_dimensions: Optional[int] = None) -> str:
        """Returns the Core Property Intelligence Architecture diagram."""
        dim_str = f"{n_dimensions}D" if n_dimensions is not None else f"{len(config.CLUSTERING_FEATURES)}D"
        return f"""
                 PROPERTY UNIVERSE
           (Delhi, Mumbai, Pune Rentals)
                        │
                        ▼
               PROPERTY INTELLIGENCE
            (Data Trust & Preprocessing)
                        │
                        ▼
               SIMILARITY SPACE ({dim_str})
           (Log1p + Standardized Scaler)
                        │
                  ┌─────┴─────┐
                  ▼           ▼
          CLUSTER DISCOVERY  K DISCOVERY
                  │           │
                  └─────┬─────┘
                        ▼
               DISCOVERED CLUSTERS
            (Optimal K Discovered)
                        │
           ┌────────────┼────────────┐
           ▼            ▼            ▼
      SEGMENT DNA  SPATIAL MAPS  PROPERTY MATCH
           │            │            │
           └────────────┼────────────┘
                        ▼
                 MARKET INSIGHTS
        """.strip()

    @staticmethod
    def generate_executive_insights(
        clustered_df: pd.DataFrame,
        segment_dna_dict: Dict[int, Dict[str, Any]],
        data_trust_metrics: Dict[str, Any]
    ) -> List[str]:
        """
        Synthesizes core analytical takeaways dynamically from the fitted data.
        """
        total = len(clustered_df)
        k = len(segment_dna_dict)
        dim = data_trust_metrics.get("dimensionality", len(config.CLUSTERING_FEATURES))

        insights = [
            f"**Inventory Segmentation**: The examined property universe of {total:,} Indian rental properties naturally organizes into {k} distinctive clusters within a {dim}-dimensional standardized similarity space.",
            f"**Data Trust Verification**: Post-processing completeness of {data_trust_metrics.get('missing_value_completeness_pct', 100.0)}% across {dim} clustering attributes ({', '.join(data_trust_metrics.get('segmentation_feature_names', []))}), with log-variance stabilization applied to positively skewed monetary and area distributions.",
        ]

        # Add segment-specific highlights
        for c, p in segment_dna_dict.items():
            name = p.get("segment_name", f"Cluster {c}")
            share = p.get("market_share_pct", 0)
            med_rent = p.get("price", {}).get("formatted_median", "N/A")
            area = p.get("area", {}).get("formatted", "N/A")
            bhk = p.get("specs", {}).get("bhk_display", "N/A")
            furn = p.get("furnishing", {}).get("dominant_status", "Unfurnished")
            city = p.get("geography", {}).get("dominant_city", "India")
            insights.append(
                f"**{name} ({share}% of market)**: Typical rent of {med_rent} for {area} ({bhk}, {furn}), concentrated primarily in {city}."
            )

        return insights

    @staticmethod
    def export_deliverables(
        clustered_df: pd.DataFrame,
        segment_profiles_df: pd.DataFrame,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Path]:
        """
        Saves clean analytical deliverables to outputs directory.
        """
        out_dir = output_dir or config.OUTPUTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        clustered_path = out_dir / "clustered_indian_rentals.csv"
        profiles_path = out_dir / "segment_profiles.csv"

        clustered_df.to_csv(clustered_path, index=False)
        segment_profiles_df.to_csv(profiles_path, index=True)

        return {
            "clustered_dataset": clustered_path,
            "segment_profiles": profiles_path
        }
