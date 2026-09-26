"""
K DISCOVERY MODULE — Community Count Optimization
Evaluates K=2 through K=10 using WCSS (Inertia) and Silhouette Scores.
Identifies the Elbow inflection point via 2D perpendicular distance and finds
the Silhouette peak to select the Official Recommended K with explicit rationale.
"""

from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import config

class KDiscoveryLab:
    """
    Explores the optimal number of natural property communities (K=2 to 10)
    using mathematical clustering validation metrics.
    """
    def __init__(self, scaled_df: pd.DataFrame, sample_size: int = 5000):
        self.scaled_df = scaled_df
        self.sample_size = sample_size
        self.k_range = list(range(config.K_MIN, config.K_MAX + 1))
        self.results: List[Dict[str, Any]] = []

    def run_discovery(self) -> Tuple[int, pd.DataFrame, Dict[str, Any]]:
        """
        Calculates Inertia and Silhouette Scores across all K values in range.
        Determines the official recommended community count.
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

        for k in self.k_range:
            kmeans = KMeans(
                n_clusters=k,
                init="k-means++",
                n_init=config.N_INIT,
                max_iter=config.MAX_ITER,
                random_state=config.RANDOM_STATE
            )
            kmeans.fit(X)
            inertia = float(kmeans.inertia_)
            inertias.append(inertia)

            # Silhouette score on sample
            labels_sample = kmeans.predict(X_sample)
            # Guard against edge cases where a single cluster forms
            if len(set(labels_sample)) > 1:
                sil = float(silhouette_score(X_sample, labels_sample))
            else:
                sil = 0.0
            silhouettes.append(sil)

            self.results.append({
                "K": k,
                "Inertia": inertia,
                "Silhouette Score": round(sil, 4)
            })

        results_df = pd.DataFrame(self.results)

        # 1. Identify Elbow via 2D Perpendicular Distance to Line (Kneedle)
        elbow_k = self._find_elbow_k(self.k_range, inertias)

        # 2. Identify Peak Silhouette Score
        sil_k = int(results_df.loc[results_df["Silhouette Score"].idxmax()]["K"])
        peak_sil_val = float(results_df["Silhouette Score"].max())

        # 3. Decision Rationale
        # If silhouette score peaks decisively at a specific K, select it;
        # otherwise balance with elbow inflection.
        if sil_k == 2 and peak_sil_val > 0.20:
            recommended_k = sil_k
            rationale = (
                f"Silhouette Score peaks decisively at K = {sil_k} ({peak_sil_val:.4f}), "
                f"demonstrating that properties partition into two major structural communities "
                f"with maximum boundary separation, whereas higher K values fragment these boundaries."
            )
        elif peak_sil_val >= 0.25:
            recommended_k = sil_k
            rationale = (
                f"Selected K = {sil_k} based on maximum Silhouette Score ({peak_sil_val:.4f}), "
                f"indicating optimal cluster separation and cohesion."
            )
        else:
            recommended_k = elbow_k
            rationale = (
                f"Selected K = {elbow_k} based on the geometric Elbow inflection point, "
                f"which provides the most balanced diminishing-returns trade-off in WCSS inertia."
            )

        meta = {
            "elbow_k": elbow_k,
            "silhouette_k": sil_k,
            "peak_silhouette_score": peak_sil_val,
            "recommended_k": recommended_k,
            "rationale": rationale,
            "tested_range": (config.K_MIN, config.K_MAX)
        }

        return recommended_k, results_df, meta

    @staticmethod
    def _find_elbow_k(k_values: List[int], inertias: List[float]) -> int:
        """
        Finds the elbow inflection point using normalized perpendicular distance
        from each point to the secant line connecting K_min to K_max.
        """
        k_arr = np.array(k_values, dtype=float)
        in_arr = np.array(inertias, dtype=float)

        # Min-max normalization
        k_norm = (k_arr - k_arr.min()) / (k_arr.max() - k_arr.min())
        in_norm = (in_arr - in_arr.min()) / (in_arr.max() - in_arr.min())

        # Line from first to last point
        p1 = np.array([k_norm[0], in_norm[0]])
        p2 = np.array([k_norm[-1], in_norm[-1]])

        # Perpendicular distance formula in 2D: |dy*x - dx*y + x2*y1 - y2*x1| / sqrt(dx^2 + dy^2)
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        denom = np.sqrt(dx**2 + dy**2)

        distances = np.abs(dy * k_norm - dx * in_norm + p2[0] * p1[1] - p2[1] * p1[0]) / denom
        elbow_idx = int(np.argmax(distances))
        return int(k_values[elbow_idx])
