"""
MODULE 4 — OPTIMAL K SELECTION
Calculates Inertia (Elbow Method) and Silhouette Scores across multiple K values.
Implements automated, mathematically grounded K selection logic.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import config
from src.utils import log_substep

def evaluate_k_range(
    scaled_df: pd.DataFrame, 
    k_min: int = config.K_MIN, 
    k_max: int = config.K_MAX,
    sample_size: Optional[int] = 5000,
    random_state: int = config.RANDOM_STATE
) -> pd.DataFrame:
    """
    Train K-Means for K in [k_min, k_max].
    Calculate Inertia and Silhouette Score for each K.
    """
    results = []
    X = scaled_df.values

    for k in range(k_min, k_max + 1):
        km = KMeans(
            n_clusters=k, 
            random_state=random_state, 
            n_init=config.N_INIT,
            max_iter=config.MAX_ITER
        )
        labels = km.fit_predict(X)
        inertia = float(km.inertia_)

        # Calculate silhouette score
        sil_score = float(silhouette_score(
            X, 
            labels, 
            sample_size=min(sample_size, len(X)) if sample_size else None,
            random_state=random_state
        ))

        results.append({
            "k": k,
            "inertia": inertia,
            "silhouette_score": sil_score
        })

    return pd.DataFrame(results)

def find_elbow_point(k_values: np.ndarray, inertias: np.ndarray) -> int:
    """
    Find elbow point using maximum perpendicular distance to the secant line
    connecting the first and last (k, inertia) points (Kneedle geometric method).
    Normalizes coordinates to [0, 1] to prevent scale distortion.
    """
    k_vals = np.array(k_values, dtype=float)
    in_vals = np.array(inertias, dtype=float)

    # Normalize to [0, 1]
    k_norm = (k_vals - k_vals.min()) / (k_vals.max() - k_vals.min() + 1e-9)
    in_norm = (in_vals - in_vals.min()) / (in_vals.max() - in_vals.min() + 1e-9)

    x1, y1 = k_norm[0], in_norm[0]
    x2, y2 = k_norm[-1], in_norm[-1]

    dx = x2 - x1
    dy = y2 - y1
    denom = np.sqrt(dx**2 + dy**2)
    if denom == 0:
        return int(k_values[0])

    # 2D perpendicular distance formula: |(y2 - y1)*x - (x2 - x1)*y + x2*y1 - y2*x1| / sqrt(dx^2 + dy^2)
    distances = np.abs(dy * k_norm - dx * in_norm + x2 * y1 - y2 * x1) / denom
    best_idx = int(np.argmax(distances))
    return int(k_values[best_idx])

def determine_optimal_k(results_df: pd.DataFrame) -> Tuple[int, Dict]:
    """
    Apply transparent selection logic:
    - Calculates k_elbow (diminishing returns inflection point)
    - Calculates k_silhouette (maximum silhouette coefficient)
    - Recommends best K and documents the selection rationale.
    """
    k_vals = results_df["k"].values
    inertias = results_df["inertia"].values
    sil_scores = results_df["silhouette_score"].values

    k_elbow = find_elbow_point(k_vals, inertias)
    k_sil = int(k_vals[np.argmax(sil_scores)])

    # Selection Rule:
    # 1. If both agree, select unanimous K.
    # 2. If differ, inspect if silhouette at k_elbow is competitive (within 10% of max).
    #    If so, k_elbow provides better feature variance partitioning; otherwise k_sil.
    elbow_sil = float(results_df.loc[results_df["k"] == k_elbow, "silhouette_score"].iloc[0])
    max_sil = float(np.max(sil_scores))

    if k_elbow == k_sil:
        recommended_k = k_elbow
        rationale = f"Both Elbow Method and Silhouette Score agree unanimously on K = {recommended_k}."
    elif (max_sil - elbow_sil) / (abs(max_sil) + 1e-6) < 0.15:
        recommended_k = k_elbow
        rationale = (
            f"Elbow Method indicates inflection at K = {k_elbow}, where Silhouette Score ({elbow_sil:.4f}) "
            f"remains near peak ({max_sil:.4f} at K = {k_sil}). K = {k_elbow} chosen for optimal variance explained."
        )
    else:
        recommended_k = k_sil
        rationale = (
            f"Silhouette Score peaks strongly at K = {k_sil} ({max_sil:.4f}), providing significantly "
            f"more distinct cluster boundaries than Elbow inflection K = {k_elbow}."
        )

    decision_info = {
        "k_elbow": k_elbow,
        "k_silhouette": k_sil,
        "recommended_k": recommended_k,
        "rationale": rationale,
        "max_silhouette_score": max_sil,
        "elbow_silhouette_score": elbow_sil,
    }

    return recommended_k, decision_info

def plot_elbow_curve(
    results_df: pd.DataFrame, 
    selected_k: Optional[int] = None, 
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Plot K vs Inertia curve with selected K highlighted."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(results_df["k"], results_df["inertia"], "o-", color="#1f77b4", linewidth=2.5, markersize=7)
    
    if selected_k:
        sel_row = results_df[results_df["k"] == selected_k]
        if not sel_row.empty:
            ax.axvline(x=selected_k, color="#d62728", linestyle="--", linewidth=1.5, label=f"Selected K={selected_k}")
            ax.scatter(selected_k, sel_row["inertia"].values[0], color="#d62728", s=130, zorder=5)

    ax.set_title("Elbow Method: K vs Inertia (Within-Cluster Sum of Squares)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of Clusters (K)", fontsize=10)
    ax.set_ylabel("Inertia (WCSS)", fontsize=10)
    ax.set_xticks(results_df["k"])
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)

    return fig

def plot_silhouette_curve(
    results_df: pd.DataFrame, 
    selected_k: Optional[int] = None, 
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Plot K vs Silhouette Score curve with selected K highlighted."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(results_df["k"], results_df["silhouette_score"], "s-", color="#2ca02c", linewidth=2.5, markersize=7)
    
    if selected_k:
        sel_row = results_df[results_df["k"] == selected_k]
        if not sel_row.empty:
            ax.axvline(x=selected_k, color="#d62728", linestyle="--", linewidth=1.5, label=f"Selected K={selected_k}")
            ax.scatter(selected_k, sel_row["silhouette_score"].values[0], color="#d62728", s=130, zorder=5)

    ax.set_title("Silhouette Analysis: K vs Silhouette Score", fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of Clusters (K)", fontsize=10)
    ax.set_ylabel("Silhouette Score", fontsize=10)
    ax.set_xticks(results_df["k"])
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)

    return fig

def run_optimal_k_selection(
    scaled_df: pd.DataFrame, 
    plots_dir: Path = config.PLOTS_DIR
) -> Tuple[int, pd.DataFrame, Dict]:
    """Execute complete Module 4 workflow and save elbow & silhouette plots."""
    results_df = evaluate_k_range(scaled_df)
    recommended_k, decision_info = determine_optimal_k(results_df)

    # Save plots
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_elbow_curve(results_df, selected_k=recommended_k, save_path=plots_dir / "optimal_k_elbow.png")
    plot_silhouette_curve(results_df, selected_k=recommended_k, save_path=plots_dir / "optimal_k_silhouette.png")

    print("\nOptimal K Selection Results:")
    for _, row in results_df.iterrows():
        print(f"  K = {int(row['k'])}: Inertia = {row['inertia']:,.1f}, Silhouette = {row['silhouette_score']:.4f}")
    
    print(f"\nSelection Rationale:\n  {decision_info['rationale']}")
    print(f"\n>>> Selected K = {recommended_k} <<<")
    log_substep(f"Selected K = {recommended_k} based on Elbow ({decision_info['k_elbow']}) & Silhouette ({decision_info['k_silhouette']})")

    return recommended_k, results_df, decision_info
