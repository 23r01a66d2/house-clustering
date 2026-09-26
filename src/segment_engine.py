"""
SEGMENT ENGINE MODULE — Property Community Discovery via K-Means
Executes K-Means clustering on the standardized similarity space.
Projects high-dimensional property similarity into a 2D PCA landscape strictly for visualization.
"""

from typing import Tuple, Dict, Any, List, Optional
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import config

class SegmentEngine:
    """
    Discovers natural property communities using K-Means clustering.
    Projects high-dimensional property similarities into a 2D landscape for visualization.
    """
    def __init__(self, k: int = config.K_DEFAULT, random_state: int = config.RANDOM_STATE):
        self.k = k
        self.random_state = random_state
        self.kmeans_model: Optional[KMeans] = None
        self.pca_model: Optional[PCA] = None
        self.pca_variance_ratio: List[float] = []

    def fit_communities(
        self,
        scaled_df: pd.DataFrame,
        usable_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, KMeans, PCA, pd.DataFrame, np.ndarray, Dict[str, Any]]:
        """
        Executes K-Means clustering and generates 2D PCA landscape projection.
        """
        X_scaled = scaled_df.values

        # 1. Initialize and Fit K-Means
        self.kmeans_model = KMeans(
            n_clusters=self.k,
            init="k-means++",
            n_init=config.N_INIT,
            max_iter=config.MAX_ITER,
            random_state=self.random_state
        )
        cluster_labels = self.kmeans_model.fit_predict(X_scaled)

        # 2. Attach Community Assignments to Datasets
        clustered_df = usable_df.copy()
        clustered_df["Cluster"] = cluster_labels

        # 3. Calculate Unsupervised Convergence Metrics
        inertia = float(self.kmeans_model.inertia_)
        n_iter = int(self.kmeans_model.n_iter_)

        # Fast sample for silhouette score
        if len(X_scaled) > 5000:
            np.random.seed(self.random_state)
            idx_sample = np.random.choice(len(X_scaled), size=5000, replace=False)
            sil_score = float(silhouette_score(X_scaled[idx_sample], cluster_labels[idx_sample]))
        else:
            sil_score = float(silhouette_score(X_scaled, cluster_labels))

        cluster_counts = pd.Series(cluster_labels).value_counts().sort_index().to_dict()

        convergence_info = {
            "k": self.k,
            "inertia": round(inertia, 1),
            "iterations_to_converge": n_iter,
            "silhouette_score": round(sil_score, 4),
            "cluster_counts": cluster_counts,
            "convergence_summary": (
                f"K-Means converged in {n_iter} iterations. "
                f"Final Within-Cluster Sum of Squares (Inertia): {inertia:,.1f}. "
                f"Silhouette Boundary Score: {sil_score:.4f}."
            )
        }

        # 4. Fit 2D PCA Projection Strictly for Visualization
        self.pca_model = PCA(n_components=2, random_state=self.random_state)
        pca_coords = self.pca_model.fit_transform(X_scaled)
        self.pca_variance_ratio = [round(float(v), 4) for v in self.pca_model.explained_variance_ratio_]

        # Project cluster centers consistently into the same 2D PCA space
        pca_centroids = self.pca_model.transform(self.kmeans_model.cluster_centers_)

        pca_df = pd.DataFrame(
            pca_coords,
            columns=["PCA_1", "PCA_2"],
            index=clustered_df.index
        )
        pca_df["Cluster"] = cluster_labels

        # Attach essential metadata columns for interactive tooltip inspection
        cols_to_attach = [
            "property_id", "price", "sqft", "price_per_sqft", "bhk",
            "numBathrooms", "city_clean", "location", "Status", "typology"
        ]
        for c in cols_to_attach:
            if c in clustered_df.columns:
                pca_df[c] = clustered_df[c]

        return clustered_df, self.kmeans_model, self.pca_model, pca_df, pca_centroids, convergence_info
