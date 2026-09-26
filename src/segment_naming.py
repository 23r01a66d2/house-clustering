"""
Segment Naming Module for Property Segmentation Analytics.
Dynamically derives descriptive, evidence-based market segment names from
actual cluster statistics without hard-coding static labels.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from src.utils import detect_column_roles

def generate_dynamic_segment_names(clustered_df: pd.DataFrame) -> Dict[int, str]:
    """
    Generate dynamic, data-driven segment titles for each cluster
    based on relative price, living area, and build quality comparisons.
    NO hard-coded assumptions or static mappings.
    """
    roles = detect_column_roles(clustered_df)
    price_col = roles.get("price")
    area_col = roles.get("living_area")
    grade_col = roles.get("grade")
    bed_col = roles.get("bedrooms")

    if not price_col or not area_col:
        # Fallback if specific roles cannot be resolved
        return {c: f"Property Segment {c}" for c in sorted(clustered_df["Cluster"].unique())}

    pop_price = clustered_df[price_col].mean()
    pop_area = clustered_df[area_col].mean()
    pop_grade = clustered_df[grade_col].mean() if grade_col else None

    cluster_ids = sorted(clustered_df["Cluster"].unique())
    num_clusters = len(cluster_ids)

    # Compute means per cluster
    c_means = {}
    for c in cluster_ids:
        c_sub = clustered_df[clustered_df["Cluster"] == c]
        c_means[c] = {
            "price": c_sub[price_col].mean(),
            "area": c_sub[area_col].mean(),
            "beds": c_sub[bed_col].mean() if bed_col else 3.0,
            "grade": c_sub[grade_col].mean() if grade_col else 7.0,
            "count": len(c_sub)
        }

    # Rank clusters by price and area
    price_ranks = {c: r for r, c in enumerate(sorted(cluster_ids, key=lambda x: c_means[x]["price"]))}
    area_ranks = {c: r for r, c in enumerate(sorted(cluster_ids, key=lambda x: c_means[x]["area"]))}

    segment_names = {}
    used_names = set()

    for c in cluster_ids:
        p_val = c_means[c]["price"]
        a_val = c_means[c]["area"]
        p_ratio = p_val / (pop_price + 1e-9)
        a_ratio = a_val / (pop_area + 1e-9)

        # 1. Price tier description
        if num_clusters == 2:
            price_tier = "Higher-Priced" if price_ranks[c] == 1 else "Lower-Priced"
        else:
            if p_ratio >= 1.35:
                price_tier = "Upper-Tier"
            elif p_ratio <= 0.80:
                price_tier = "Entry-Level"
            elif price_ranks[c] == num_clusters - 1:
                price_tier = "Premium-Tier"
            elif price_ranks[c] == 0:
                price_tier = "Affordable"
            else:
                price_tier = "Mid-Market"

        # 2. Area/Size description
        if num_clusters == 2:
            size_tier = "Spacious Properties" if area_ranks[c] == 1 else "Compact Properties"
        else:
            if a_ratio >= 1.25:
                size_tier = "Spacious Residences"
            elif a_ratio <= 0.80:
                size_tier = "Compact Properties"
            else:
                size_tier = "Standard-Sized Homes"

        candidate_name = f"{price_tier} {size_tier}"

        # Prevent duplicate names when multiple clusters share tiers
        if candidate_name in used_names:
            if grade_col and c_means[c]["grade"] > (pop_grade or 7.0) + 0.5:
                candidate_name = f"{candidate_name} (High Grade)"
            elif bed_col and c_means[c]["beds"] >= 4.0:
                candidate_name = f"{candidate_name} (Multi-Bed)"
            else:
                candidate_name = f"{candidate_name} (Group {c + 1})"

        used_names.add(candidate_name)
        segment_names[c] = candidate_name

    return segment_names

def attach_segment_names(clustered_df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach 'Segment Name' and 'Segment Label' to a clustered DataFrame.
    """
    df = clustered_df.copy()
    seg_names = generate_dynamic_segment_names(df)
    df["Segment Name"] = df["Cluster"].map(seg_names)
    df["Segment Label"] = df.apply(lambda r: f"Segment {r['Cluster']}: {r['Segment Name']}", axis=1)
    return df

def get_segment_summary_cards(clustered_df: pd.DataFrame) -> List[Dict]:
    """
    Return clean structured data for rendering visual segment profile cards.
    All metrics computed dynamically.
    """
    roles = detect_column_roles(clustered_df)
    price_col = roles.get("price", "Price")
    area_col = roles.get("living_area", "living area")
    bed_col = roles.get("bedrooms", "number of bedrooms")
    bath_col = roles.get("bathrooms", "number of bathrooms")
    grade_col = roles.get("grade", "grade of the house")

    total_properties = len(clustered_df)
    seg_names = generate_dynamic_segment_names(clustered_df)
    cards = []

    for c in sorted(clustered_df["Cluster"].unique()):
        sub = clustered_df[clustered_df["Cluster"] == c]
        count = len(sub)
        pct = (count / total_properties) * 100 if total_properties > 0 else 0

        cards.append({
            "cluster_id": c,
            "segment_name": seg_names[c],
            "display_title": f"Segment {c}: {seg_names[c]}",
            "count": count,
            "percentage": pct,
            "avg_price": sub[price_col].mean() if price_col in sub else 0,
            "median_price": sub[price_col].median() if price_col in sub else 0,
            "avg_area": sub[area_col].mean() if area_col in sub else 0,
            "avg_beds": sub[bed_col].mean() if bed_col in sub else 0,
            "avg_baths": sub[bath_col].mean() if bath_col in sub else 0,
            "avg_grade": sub[grade_col].mean() if grade_col in sub else 0,
        })

    return cards
