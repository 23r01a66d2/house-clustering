"""
INSIGHT ENGINE MODULE — Market Insights, Architecture Visualizer, and Exports
Synthesizes high-level property intelligence, formats the Core Architecture diagram,
and packages analytical deliverables for export.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
from pathlib import Path

import config

class InsightEngine:
    """
    Synthesizes executive market insights, outputs architecture diagrams,
    and manages export artifacts.
    """
    @staticmethod
    def get_architecture_diagram() -> str:
        """Returns the Core Property Intelligence Architecture diagram."""
        return """
                 PROPERTY UNIVERSE
                        │
                        ▼
              PROPERTY INTELLIGENCE
                        │
                        ▼
               PROPERTY SIGNATURES
                        │
                        ▼
                SIMILARITY SPACE
                        │
                 ┌──────┴──────┐
                 ▼             ▼
          SEGMENT DISCOVERY   K DISCOVERY
                 │             │
                 └──────┬──────┘
                        ▼
              PROPERTY COMMUNITIES
                        │
          ┌─────────────┼──────────────┐
          ▼             ▼              ▼
    SEGMENT DNA    SPATIAL PATTERNS  PROPERTY MATCH
          │             │              │
          └─────────────┼──────────────┘
                        ▼
                 MARKET INSIGHTS
        """.strip()

    @staticmethod
    def generate_executive_insights(
        clustered_df: pd.DataFrame,
        segment_dna_profiles: List[Dict[str, Any]],
        data_trust_metrics: Dict[str, Any]
    ) -> List[str]:
        """
        Synthesizes core analytical takeaways dynamically from the fitted data.
        """
        total = len(clustered_df)
        k = len(segment_dna_profiles)
        dim = data_trust_metrics.get("dimensionality", 20)

        insights = [
            f"**Inventory Segmentation**: The examined property universe of {total:,} homes naturally organizes into {k} distinctive communities within an {dim}-dimensional standardized similarity space.",
            f"**Data Trust Verification**: {data_trust_metrics.get('missing_value_completeness_pct', 100.0)}% data completeness across {dim} segmentation attributes, with {data_trust_metrics.get('outliers_treated_count', 0)} typographical anomalies documented and isolated.",
        ]

        # Add segment-specific highlights
        for p in segment_dna_profiles:
            insights.append(
                f"**{p['segment_name']} ({p['share_pct']}% of market)**: Averages ${p['avg_price']:,.0f} with typical living quarters of {p['avg_living_area']:,.0f} sqft and {p['avg_bedrooms']:.1f} bedrooms."
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

        clustered_path = out_dir / "clustered_house_data.csv"
        profiles_path = out_dir / "segment_profiles.csv"

        clustered_df.to_csv(clustered_path, index=False)
        segment_profiles_df.to_csv(profiles_path, index=True)

        return {
            "clustered_dataset": clustered_path,
            "segment_profiles": profiles_path
        }
