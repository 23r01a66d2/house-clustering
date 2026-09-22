"""
MODULE 6 — VISUALIZATION & RESULTS
Generates 2D PCA cluster projections with transformed centroids,
cluster distribution bars, and multi-feature comparative visualizations.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
import config
from src.utils import detect_column_roles, log_substep

def plot_pca_clusters(
    scaled_df: pd.DataFrame, 
    labels: np.ndarray, 
    centroids: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None
) -> Tuple[plt.Figure, PCA]:
    """
    Reduce scaled features to 2 principal components via PCA.
    Project cluster points and centroids onto PC1 vs PC2.
    """
    feature_cols = [c for c in scaled_df.columns if c != "Cluster"]
    X = scaled_df[feature_cols].values

    pca = PCA(n_components=2, random_state=config.RANDOM_STATE)
    X_pca = pca.fit_transform(X)
    
    var_exp = pca.explained_variance_ratio_ * 100

    pca_df = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "Cluster": [f"Cluster {lbl}" for lbl in labels]
    })

    fig, ax = plt.subplots(figsize=(10, 7))
    unique_clusters = sorted(pca_df["Cluster"].unique())
    palette = sns.color_palette("tab10", n_colors=len(unique_clusters))

    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="Cluster",
        palette=palette,
        alpha=0.45,
        s=25,
        ax=ax,
        edgecolor=None
    )

    # Project and plot centroids if provided
    if centroids is not None:
        centroids_pca = pca.transform(centroids)
        for idx, pt in enumerate(centroids_pca):
            ax.scatter(
                pt[0], pt[1],
                s=260,
                marker="X",
                color="black",
                edgecolors="white",
                linewidth=2,
                label="Centroid" if idx == 0 else "",
                zorder=10
            )
            ax.text(
                pt[0] + 0.15, pt[1] + 0.15,
                f"C{idx}",
                fontsize=11,
                fontweight="bold",
                color="black",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none")
            )

    ax.set_title(
        f"K-Means Clusters in 2D PCA Space\n(PC1: {var_exp[0]:.1f}%, PC2: {var_exp[1]:.1f}% variance explained)",
        fontsize=13,
        fontweight="bold"
    )
    ax.set_xlabel(f"Principal Component 1 ({var_exp[0]:.1f}%)", fontsize=10)
    ax.set_ylabel(f"Principal Component 2 ({var_exp[1]:.1f}%)", fontsize=10)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)

    return fig, pca

def plot_cluster_distribution(
    clustered_df: pd.DataFrame, 
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Plot bar chart showing the house count and percentage per cluster."""
    counts = clustered_df["Cluster"].value_counts().sort_index()
    total = len(clustered_df)
    percentages = (counts / total) * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    palette = sns.color_palette("tab10", n_colors=len(counts))
    
    bars = ax.bar([f"Cluster {c}" for c in counts.index], counts.values, color=palette, edgecolor="black", linewidth=0.8)

    for bar, pct, count in zip(bars, percentages, counts.values):
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            yval + total * 0.01,
            f"{count:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )

    ax.set_title("House Distribution Across Clusters", fontsize=13, fontweight="bold")
    ax.set_xlabel("Cluster", fontsize=10)
    ax.set_ylabel("Number of Houses", fontsize=10)
    ax.set_ylim(0, max(counts.values) * 1.18)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)

    return fig

def plot_cluster_feature_comparisons(
    clustered_df: pd.DataFrame, 
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Compare key house metrics across clusters using subplots."""
    roles = detect_column_roles(clustered_df)
    key_features = [
        ("Average Price", roles.get("price")),
        ("Average Living Area (sqft)", roles.get("living_area")),
        ("Average Bedrooms", roles.get("bedrooms")),
        ("Average Bathrooms", roles.get("bathrooms")),
    ]
    # Filter only available features
    valid_features = [(title, col) for title, col in key_features if col is not None]
    
    if not valid_features:
        return plt.figure()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, (title, col) in enumerate(valid_features):
        ax = axes[i]
        means = clustered_df.groupby("Cluster")[col].mean()
        palette = sns.color_palette("tab10", n_colors=len(means))
        
        bars = ax.bar([f"Cluster {c}" for c in means.index], means.values, color=palette, edgecolor="black", linewidth=0.8)
        for bar in bars:
            yval = bar.get_height()
            label = f"{yval:,.0f}" if yval >= 100 else f"{yval:.2f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                yval * 1.02,
                label,
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold"
            )

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Cluster", fontsize=9)
        ax.set_ylabel("Mean Value", fontsize=9)
        ax.set_ylim(0, max(means.values) * 1.16)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)

    return fig

def generate_cluster_visualizations(
    clustered_orig: pd.DataFrame, 
    clustered_scaled: pd.DataFrame,
    centroids: np.ndarray,
    plots_dir: Path = config.PLOTS_DIR
) -> Dict[str, Path]:
    """Execute complete Module 6 visualization suite and save artifacts."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = {}

    # 1. PCA 2D Cluster Scatter Plot
    pca_path = plots_dir / "cluster_pca_scatter.png"
    plot_pca_clusters(
        clustered_scaled, 
        clustered_scaled["Cluster"].values, 
        centroids=centroids, 
        save_path=pca_path
    )
    saved_paths["pca_scatter"] = pca_path

    # 2. Cluster Size Distribution
    dist_path = plots_dir / "cluster_distribution.png"
    plot_cluster_distribution(clustered_orig, save_path=dist_path)
    saved_paths["cluster_distribution"] = dist_path

    # 3. Cluster Feature Comparisons
    comp_path = plots_dir / "cluster_feature_comparisons.png"
    plot_cluster_feature_comparisons(clustered_orig, save_path=comp_path)
    saved_paths["feature_comparisons"] = comp_path

    log_substep(f"Generated {len(saved_paths)} clustering visualizations in '{plots_dir.name}'")
    return saved_paths
