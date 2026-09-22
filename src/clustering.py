"""
MODULE 5 — K-MEANS CLUSTERING
Fits KMeans model, extracts cluster centroids, computes final inertia & silhouette score,
and attaches cluster assignments to the dataset.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import config
from src.utils import log_substep

class HouseClusteringModel:
    """Wrapper for scikit-learn KMeans clustering."""

    def __init__(
        self, 
        k: int, 
        random_state: int = config.RANDOM_STATE, 
        n_init: int = config.N_INIT,
        max_iter: int = config.MAX_ITER
    ):
        self.k = k
        self.random_state = random_state
        self.n_init = n_init
        self.max_iter = max_iter
        
        self.model = KMeans(
            n_clusters=self.k,
            random_state=self.random_state,
            n_init=self.n_init,
            max_iter=self.max_iter
        )
        self.is_fitted = False
        self.inertia: float = 0.0
        self.silhouette: float = 0.0
        self.n_iter: int = 0
        self.labels: np.ndarray = np.array([])
        self.cluster_centers_: np.ndarray = np.array([])

    def fit_predict(
        self, 
        scaled_df: pd.DataFrame, 
        original_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Train K-Means on scaled_df and attach 'Cluster' column to both
        original_df and scaled_df without mutating in-place.
        """
        X = scaled_df.values
        self.labels = self.model.fit_predict(X)
        self.is_fitted = True
        self.inertia = float(self.model.inertia_)
        self.n_iter = int(self.model.n_iter_)
        self.cluster_centers_ = self.model.cluster_centers_

        # Compute silhouette score for the final model
        self.silhouette = float(silhouette_score(
            X, 
            self.labels, 
            sample_size=min(5000, len(X)), 
            random_state=self.random_state
        ))

        # Create output DataFrames with 'Cluster' column
        clustered_original = original_df.copy()
        clustered_original["Cluster"] = self.labels

        clustered_scaled = scaled_df.copy()
        clustered_scaled["Cluster"] = self.labels

        metrics = {
            "selected_k": self.k,
            "n_iterations": self.n_iter,
            "inertia": self.inertia,
            "silhouette_score": self.silhouette,
            "cluster_counts": dict(pd.Series(self.labels).value_counts().sort_index())
        }

        return clustered_original, clustered_scaled, metrics

    def print_summary(self, metrics: Dict):
        """Print console summary for Module 5."""
        print(f"\nK-Means Clustering Results (K = {self.k})")
        print(f"  Iterations to Converge: {metrics['n_iterations']}")
        print(f"  Inertia (WCSS):         {metrics['inertia']:,.1f}")
        print(f"  Silhouette Score:       {metrics['silhouette_score']:.4f}")
        print("  Cluster Distribution:")
        for cluster_id, count in metrics["cluster_counts"].items():
            print(f"    Cluster {cluster_id}: {count:,} houses")
        log_substep(f"Clustering converged in {metrics['n_iterations']} iterations with Silhouette {metrics['silhouette_score']:.4f}")

def train_kmeans(
    scaled_df: pd.DataFrame, 
    original_df: pd.DataFrame, 
    k: int
) -> Tuple[pd.DataFrame, pd.DataFrame, HouseClusteringModel, Dict]:
    """Convenience functional interface for Module 5."""
    model = HouseClusteringModel(k=k)
    clustered_orig, clustered_scaled, metrics = model.fit_predict(scaled_df, original_df)
    model.print_summary(metrics)
    return clustered_orig, clustered_scaled, model, metrics
