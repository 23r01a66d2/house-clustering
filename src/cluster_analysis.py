"""
MODULE 7 — CLUSTER ANALYSIS & INTERPRETATION
Computes comprehensive cluster profiles and dynamically synthesizes
human-readable interpretations grounded strictly in statistical evidence.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from src.utils import detect_column_roles, format_number, log_substep

def build_cluster_profiles(clustered_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute cluster profiles:
    - profile_table: formatted for display (Features as rows, Clusters as columns)
    - numeric_profiles: raw aggregated statistics for programmatic interpretation
    """
    roles = detect_column_roles(clustered_df)
    
    # Priority ordered attributes to summarize
    candidate_features = [
        ("Price", roles.get("price")),
        ("Living Area (sqft)", roles.get("living_area")),
        ("Lot Area (sqft)", roles.get("lot_area")),
        ("Bedrooms", roles.get("bedrooms")),
        ("Bathrooms", roles.get("bathrooms")),
        ("Floors", roles.get("floors")),
        ("House Grade", roles.get("grade")),
        ("House Condition", roles.get("condition")),
        ("Year Built", roles.get("built_year")),
        ("Schools Nearby", roles.get("schools")),
        ("Distance from Airport", roles.get("airport_dist")),
    ]
    
    valid_features = [(label, col) for label, col in candidate_features if col is not None]

    cluster_ids = sorted(clustered_df["Cluster"].unique())
    counts = clustered_df["Cluster"].value_counts()
    total_houses = len(clustered_df)

    # Dictionary to build transposed display table
    table_dict = {}
    table_dict["Houses"] = {f"Cluster {c}": f"{counts[c]:,} ({counts[c] / total_houses * 100:.1f}%)" for c in cluster_ids}

    # Store raw means for interpretation
    numeric_means = {}

    for label, col in valid_features:
        means = clustered_df.groupby("Cluster")[col].mean()
        numeric_means[col] = means
        
        # Formatting
        row_formatted = {}
        for c in cluster_ids:
            val = means[c]
            if "price" in col.lower():
                row_formatted[f"Cluster {c}"] = f"${val:,.0f}"
            elif "area" in col.lower() or "lot" in col.lower():
                row_formatted[f"Cluster {c}"] = f"{val:,.0f} sqft"
            elif "year" in col.lower():
                row_formatted[f"Cluster {c}"] = f"{int(round(val))}"
            else:
                row_formatted[f"Cluster {c}"] = f"{val:.2f}"
        table_dict[f"Avg {label}"] = row_formatted

    profile_display_df = pd.DataFrame(table_dict).T
    numeric_profiles_df = pd.DataFrame(numeric_means)

    return profile_display_df, numeric_profiles_df

def generate_cluster_interpretations(
    clustered_df: pd.DataFrame, 
    numeric_profiles: pd.DataFrame
) -> Dict[int, str]:
    """
    Dynamically generate cluster interpretations based on relative statistical differences
    between each cluster's average and the dataset overall population mean.
    DO NOT HARD-CODE STATIC DESCRIPTIONS.
    """
    interpretations = {}
    roles = detect_column_roles(clustered_df)

    price_col = roles.get("price")
    area_col = roles.get("living_area")
    bed_col = roles.get("bedrooms")
    bath_col = roles.get("bathrooms")
    grade_col = roles.get("grade")
    built_col = roles.get("built_year")

    # Overall dataset means
    overall_means = {
        col: clustered_df[col].mean() 
        for col in [price_col, area_col, bed_col, bath_col, grade_col, built_col] 
        if col is not None
    }

    cluster_counts = clustered_df["Cluster"].value_counts()
    total_houses = len(clustered_df)

    for cluster_id in sorted(clustered_df["Cluster"].unique()):
        lines = []
        count = cluster_counts[cluster_id]
        pct = (count / total_houses) * 100
        
        # 1. Size & Share
        lines.append(f"Cluster {cluster_id} encompasses {count:,} properties ({pct:.1f}% of analyzed dataset).")

        # 2. Price Characterization
        if price_col and price_col in numeric_profiles.columns:
            c_price = numeric_profiles.loc[cluster_id, price_col]
            pop_price = overall_means[price_col]
            ratio = c_price / pop_price
            if ratio >= 1.35:
                price_desc = f"substantially above average prices (${c_price:,.0f} vs overall ${pop_price:,.0f})"
            elif ratio <= 0.80:
                price_desc = f"economical, below-average prices (${c_price:,.0f} vs overall ${pop_price:,.0f})"
            else:
                price_desc = f"moderate, mid-range pricing (${c_price:,.0f} near overall ${pop_price:,.0f})"
        else:
            price_desc = "unspecified pricing"

        # 3. Size / Living Area Characterization
        if area_col and area_col in numeric_profiles.columns:
            c_area = numeric_profiles.loc[cluster_id, area_col]
            pop_area = overall_means[area_col]
            ratio_area = c_area / pop_area
            if ratio_area >= 1.25:
                area_desc = f"spacious living areas ({c_area:,.0f} sqft)"
            elif ratio_area <= 0.80:
                area_desc = f"compact living quarters ({c_area:,.0f} sqft)"
            else:
                area_desc = f"standard/medium property dimensions ({c_area:,.0f} sqft)"
        else:
            area_desc = "unspecified area"

        lines.append(f"• Characterized by {price_desc} with {area_desc}.")

        # 4. Bedroom & Bathroom Layout
        specs = []
        if bed_col and bed_col in numeric_profiles.columns:
            c_bed = numeric_profiles.loc[cluster_id, bed_col]
            specs.append(f"{c_bed:.1f} bedrooms")
        if bath_col and bath_col in numeric_profiles.columns:
            c_bath = numeric_profiles.loc[cluster_id, bath_col]
            specs.append(f"{c_bath:.2f} bathrooms")
        if specs:
            lines.append(f"• Typical layout averages {', '.join(specs)}.")

        # 5. Construction Grade & Age
        qualities = []
        if grade_col and grade_col in numeric_profiles.columns:
            c_grade = numeric_profiles.loc[cluster_id, grade_col]
            pop_grade = overall_means[grade_col]
            if c_grade > pop_grade + 0.5:
                qualities.append(f"superior construction quality (grade {c_grade:.1f}/13)")
            elif c_grade < pop_grade - 0.5:
                qualities.append(f"standard/economy construction build (grade {c_grade:.1f}/13)")
            else:
                qualities.append(f"average build quality (grade {c_grade:.1f}/13)")

        if built_col and built_col in numeric_profiles.columns:
            c_built = int(round(numeric_profiles.loc[cluster_id, built_col]))
            qualities.append(f"average build year around {c_built}")

        if qualities:
            lines.append(f"• Construction profile features {'; '.join(qualities)}.")

        interpretations[cluster_id] = "\n".join(lines)

    return interpretations

def run_cluster_analysis(clustered_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int, str]]:
    """Execute complete Module 7 profiling and dynamic interpretation."""
    profile_table, numeric_profiles = build_cluster_profiles(clustered_df)
    interpretations = generate_cluster_interpretations(clustered_df, numeric_profiles)

    print("\nCluster Profile Table:")
    print(profile_table.to_string())

    print("\nAutomated Cluster Interpretations:")
    for c_id, text in interpretations.items():
        print(f"\n--- Cluster {c_id} ---")
        print(text)

    log_substep(f"Generated profiles and dynamic interpretations for {len(interpretations)} clusters")
    return profile_table, interpretations
