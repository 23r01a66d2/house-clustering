"""
MAIN PIPELINE RUNNER — House Clustering Using K-Means Clustering Techniques
Executes the full 8-module unsupervised machine-learning workflow via command-line.
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from src.utils import ensure_directories, log_header, log_step, log_substep, log_footer
from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.eda import generate_eda_suite
from src.optimal_k import run_optimal_k_selection
from src.clustering import train_kmeans
from src.visualization import generate_cluster_visualizations
from src.cluster_analysis import run_cluster_analysis
from src.segment_naming import attach_segment_names, generate_dynamic_segment_names

def run_pipeline(csv_path: Path = config.RAW_DATA_PATH):
    """Run all 8 modules sequentially with rich logging."""
    ensure_directories()
    log_header("HOUSE CLUSTERING SYSTEM - K-MEANS PIPELINE")

    # MODULE 1 — DATA COLLECTION
    log_step(1, 8, "Loading dataset")
    loader = DataLoader(csv_path)
    try:
        raw_df = loader.load_data()
    except Exception as e:
        print(f"\n[ERROR] Failed to load dataset: {e}")
        sys.exit(1)
    loader.print_summary()
    log_substep("Dataset loaded successfully")

    # MODULE 2 — DATA PREPROCESSING
    log_step(2, 8, "Preprocessing")
    preprocessor = DataPreprocessor(raw_df)
    cleaned_df, scaled_df, prep_meta = preprocessor.preprocess()
    preprocessor.print_summary()
    log_substep("Missing values handled")
    log_substep(f"Features selected ({len(prep_meta['selected_features'])} features)")
    log_substep("Scaling completed (StandardScaler)")

    # MODULE 3 — EXPLORATORY DATA ANALYSIS
    log_step(3, 8, "Performing EDA")
    eda_plots = generate_eda_suite(cleaned_df, prep_meta["selected_features"])
    log_substep(f"EDA completed (Saved {len(eda_plots)} plots to '{config.PLOTS_DIR.name}')")

    # MODULE 4 — OPTIMAL K SELECTION
    log_step(4, 8, "Finding optimal K")
    optimal_k, k_results, k_decision = run_optimal_k_selection(scaled_df)
    log_substep("Elbow analysis completed")
    log_substep("Silhouette analysis completed")
    log_substep(f"Selected K = {optimal_k}")

    # MODULE 5 — K-MEANS CLUSTERING
    log_step(5, 8, "Training K-Means")
    clustered_orig, clustered_scaled, model, cluster_metrics = train_kmeans(
        scaled_df=scaled_df,
        original_df=cleaned_df,
        k=optimal_k
    )
    log_substep("Clustering completed")

    # MODULE 6 — VISUALIZATION & RESULTS
    log_step(6, 8, "Generating visualizations")
    viz_plots = generate_cluster_visualizations(
        clustered_orig=clustered_orig,
        clustered_scaled=clustered_scaled,
        centroids=model.cluster_centers_
    )
    log_substep(f"Visualizations generated (Saved {len(viz_plots)} plots to '{config.PLOTS_DIR.name}')")

    # MODULE 7 — CLUSTER ANALYSIS & INTERPRETATION
    log_step(7, 8, "Analyzing clusters")
    profile_table, interpretations = run_cluster_analysis(clustered_orig)
    log_substep("Cluster profiles generated")

    # Attach dynamic segment names
    clustered_orig = attach_segment_names(clustered_orig)
    seg_names = generate_dynamic_segment_names(clustered_orig)

    # MODULE 8 — FINAL OUTPUT
    log_step(8, 8, "Saving results")
    output_csv = config.CLUSTERED_DATA_PATH
    clustered_orig.to_csv(output_csv, index=False)
    log_substep(f"Results saved to '{output_csv.name}' ({len(clustered_orig):,} rows with 'Segment Name')")

    # Display Final Project Summary
    print("\n" + "-" * 60)
    print("FINAL EXECUTIVE SUMMARY")
    print("-" * 60)
    print(f"Total Houses Analyzed: {len(clustered_orig):,}")
    print(f"Selected K:            {optimal_k}")
    print(f"Clustering Inertia:    {cluster_metrics['inertia']:,.1f}")
    print(f"Silhouette Score:      {cluster_metrics['silhouette_score']:.4f}")
    print(f"Exported CSV:          {output_csv}")
    print("\nProperty Segment Breakdown:")
    for c_id, count in cluster_metrics["cluster_counts"].items():
        pct = (count / len(clustered_orig)) * 100
        print(f"  Segment {c_id} ({seg_names.get(c_id, 'N/A')}): {count:,} houses ({pct:.1f}%)")

    log_footer("PIPELINE COMPLETED SUCCESSFULLY")
    return clustered_orig, profile_table, interpretations

if __name__ == "__main__":
    run_pipeline()
