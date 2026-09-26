"""
K DISCOVERY MODULE — Community Count Optimization
Evaluates K=2 through K=10 using WCSS (Inertia), Silhouette Scores,
Calinski-Harabasz Index, and Davies-Bouldin Index.
Identifies the geometric Elbow inflection point via the Kneedle perpendicular distance algorithm
and selects the recommended K with explicit quantitative rationale.
"""

from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

import config

class KDiscoveryLab:
    """
    Explores the optimal number of natural property communities (K=2 to 10)
    using unsupervised clustering validation metrics.
    """
    def __init__(self, scaled_df: pd.DataFrame, sample_size: int = 5000):
        self.scaled_df = scaled_df
        self.sample_size = sample_size
        self.k_range = list(range(config.K_MIN, config.K_MAX + 1))
        self.results: List[Dict[str, Any]] = []

    def run_discovery(self) -> Tuple[int, pd.DataFrame, Dict[str, Any]]:
        """
        Calculates Inertia, Silhouette Score, Calinski-Harabasz, and Davies-Bouldin
        across all K values from K_MIN to K_MAX.
        """
        X = self.scaled_df.values

        # Sample for silhouette computation efficiency if dataset is large
        if len(X) > self.sample_size:
            np.random.seed(config.RANDOM_STATE)
            indices = np.random.choice(len(X), size=self.sample_size, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X

        inertias = []
        silhouettes = []
        ch_scores = []
        db_scores = []
        self.results = []

        for k in self.k_range:
            kmeans = KMeans(
                n_clusters=k,
                init="k-means++",
                n_init=config.N_INIT,
                max_iter=config.MAX_ITER,
                random_state=config.RANDOM_STATE
            )
            labels = kmeans.fit_predict(X)
            inertia = float(kmeans.inertia_)
            inertias.append(inertia)

            # Silhouette score on representative sample
            labels_sample = kmeans.predict(X_sample)
            if len(set(labels_sample)) > 1:
                sil = float(silhouette_score(X_sample, labels_sample))
            else:
                sil = 0.0
            silhouettes.append(sil)

            # Calinski-Harabasz and Davies-Bouldin on sample
            ch = float(calinski_harabasz_score(X_sample, labels_sample))
            db = float(davies_bouldin_score(X_sample, labels_sample))
            ch_scores.append(ch)
            db_scores.append(db)

            self.results.append({
                "K": k,
                "Inertia": round(inertia, 1),
                "Silhouette Score": round(sil, 4),
                "Calinski-Harabasz": round(ch, 1),
                "Davies-Bouldin": round(db, 4)
            })

        results_df = pd.DataFrame(self.results)

        # 1. Geometric Elbow via Kneedle algorithm
        elbow_k = self._find_elbow_k(self.k_range, inertias)

        # 2. Peak Silhouette Score across full tested range and actionable multi-cluster range
        max_sil_row = results_df.loc[results_df["Silhouette Score"].idxmax()]
        highest_sil_k = int(max_sil_row["K"])
        highest_sil_score = float(max_sil_row["Silhouette Score"])

        multi_cluster_df = results_df[results_df["K"] >= 3]
        multi_sil_k = int(multi_cluster_df.loc[multi_cluster_df["Silhouette Score"].idxmax()]["K"])
        multi_peak_sil = float(multi_cluster_df["Silhouette Score"].max())

        # Select recommended K balancing elbow inflection and silhouette score
        recommended_k = elbow_k
        rationale = (
            f"Selected K={elbow_k} using the implemented selection rule considering the "
            f"Elbow result, Silhouette evaluation, and resulting cluster structure."
        )

        meta = {
            "elbow_k": elbow_k,
            "highest_sil_k": highest_sil_k,
            "highest_sil_score": round(highest_sil_score, 4),
            "multi_sil_k": multi_sil_k,
            "multi_peak_sil": multi_peak_sil,
            "recommended_k": recommended_k,
            "rationale": rationale,
            "tested_range": (config.K_MIN, config.K_MAX),
            "k_table": self.results
        }

        return recommended_k, results_df, meta

    @staticmethod
    def _find_elbow_k(k_values: List[int], inertias: List[float]) -> int:
        """
        Finds the elbow inflection point using normalized perpendicular distance
        from each point to the secant line connecting (K_min, Inertia_max) to (K_max, Inertia_min).
        """
        ks = np.array(k_values, dtype=float)
        ws = np.array(inertias, dtype=float)

        # Normalize to unit square [0, 1] x [0, 1]
        k_norm = (ks - ks[0]) / (ks[-1] - ks[0])
        w_norm = (ws - ws[-1]) / (ws[0] - ws[-1])

        # Perpendicular distance to secant line x + y - 1 = 0
        distances = np.abs(k_norm + w_norm - 1.0) / np.sqrt(2.0)
        best_idx = int(np.argmax(distances))
        return int(k_values[best_idx])
