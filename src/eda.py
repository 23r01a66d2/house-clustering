"""
MODULE 3 — EXPLORATORY DATA ANALYSIS
Provides statistical summaries, correlation heatmaps, feature distributions,
and relationship scatter plots.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import config
from src.utils import detect_column_roles, log_substep

# Set aesthetic styling
sns.set_theme(style=config.SEABORN_STYLE)
plt.rcParams["font.sans-serif"] = "DejaVu Sans"

def compute_descriptive_statistics(df: pd.DataFrame, features: Optional[List[str]] = None) -> pd.DataFrame:
    """Compute detailed summary statistics including skewness."""
    cols = features if features else df.select_dtypes(include=["number"]).columns.tolist()
    stats = df[cols].describe().T
    stats["skew"] = df[cols].skew()
    stats["median"] = df[cols].median()
    return stats[["count", "mean", "std", "min", "25%", "median", "75%", "max", "skew"]]

def plot_correlation_heatmap(
    df: pd.DataFrame, 
    features: Optional[List[str]] = None, 
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Generate and return a correlation heatmap figure."""
    cols = features if features else df.select_dtypes(include=["number"]).columns.tolist()
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"},
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
        ax=ax
    )
    ax.set_title("Correlation Heatmap of House Features", fontsize=14, pad=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)

    return fig

def plot_feature_distribution(
    df: pd.DataFrame, 
    feature: str, 
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Plot distribution histogram with KDE and a corresponding boxplot."""
    fig, (ax_box, ax_hist) = plt.subplots(
        2, 1, 
        figsize=(8, 6), 
        gridspec_kw={"height_ratios": [0.25, 0.75]},
        sharex=True
    )

    sns.boxplot(x=df[feature], ax=ax_box, color="#4c72b0", fliersize=3)
    ax_box.set(xlabel="")
    ax_box.set_title(f"Distribution & Outlier Analysis: {feature}", fontsize=12, fontweight="bold")

    sns.histplot(df[feature], kde=True, ax=ax_hist, color="#4c72b0", bins=30, stat="density")
    ax_hist.set_xlabel(feature, fontsize=10)
    ax_hist.set_ylabel("Density", fontsize=10)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)

    return fig

def plot_bivariate_scatter(
    df: pd.DataFrame, 
    x_col: str, 
    y_col: str, 
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Plot relationship between two features with a trendline."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sample_df = df.sample(n=min(3000, len(df)), random_state=config.RANDOM_STATE)
    sns.regplot(
        data=sample_df,
        x=x_col,
        y=y_col,
        scatter_kws={"alpha": 0.35, "color": "#1f77b4", "s": 15},
        line_kws={"color": "#d62728", "linewidth": 2},
        ax=ax
    )
    ax.set_title(f"{y_col} vs {x_col}", fontsize=12, fontweight="bold")
    ax.set_xlabel(x_col, fontsize=10)
    ax.set_ylabel(y_col, fontsize=10)
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)

    return fig

def generate_eda_suite(
    df: pd.DataFrame, 
    features: List[str], 
    plots_dir: Path = config.PLOTS_DIR
) -> Dict[str, Path]:
    """
    Generate and save standard EDA plots required by Module 3.
    Returns mapping of plot names to saved file paths.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = {}

    # 1. Correlation Heatmap
    heatmap_path = plots_dir / "eda_correlation_heatmap.png"
    plot_correlation_heatmap(df, features, save_path=heatmap_path)
    saved_paths["correlation_heatmap"] = heatmap_path

    # 2. Key Bivariate Relationships (Dynamic Lookup)
    roles = detect_column_roles(df)
    price_col = roles.get("price")
    area_col = roles.get("living_area")
    bed_col = roles.get("bedrooms")
    bath_col = roles.get("bathrooms")

    pairs = []
    if price_col and area_col:
        pairs.append((area_col, price_col, "eda_scatter_price_vs_area.png"))
    if price_col and bed_col:
        pairs.append((bed_col, price_col, "eda_scatter_price_vs_bedrooms.png"))
    if price_col and bath_col:
        pairs.append((bath_col, price_col, "eda_scatter_price_vs_bathrooms.png"))
    if area_col and bed_col:
        pairs.append((bed_col, area_col, "eda_scatter_area_vs_bedrooms.png"))

    for x, y, fname in pairs:
        path = plots_dir / fname
        plot_bivariate_scatter(df, x, y, save_path=path)
        saved_paths[fname] = path

    # 3. Distribution of Key Target Feature (Price)
    if price_col:
        price_dist_path = plots_dir / "eda_distribution_price.png"
        plot_feature_distribution(df, price_col, save_path=price_dist_path)
        saved_paths["price_distribution"] = price_dist_path

    if area_col:
        area_dist_path = plots_dir / "eda_distribution_area.png"
        plot_feature_distribution(df, area_col, save_path=area_dist_path)
        saved_paths["area_distribution"] = area_dist_path

    log_substep(f"Generated {len(saved_paths)} EDA visualizations in '{plots_dir.name}'")
    return saved_paths
