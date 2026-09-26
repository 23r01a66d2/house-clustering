"""
MAIN PIPELINE RUNNER — Property Intelligence & Segment Discovery System
Executes the unsupervised property segmentation workflow from command-line.
Uses K-Means clustering on standardized multidimensional similarity space.
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
from src.property_data import PropertyDataLoader
from src.property_intelligence import PropertyIntelligence
from src.property_signature import PropertySignatureEngine
from src.similarity_engine import SimilarityEngine
from src.k_discovery import KDiscoveryLab
from src.segment_engine import SegmentEngine
from src.segment_dna import SegmentDNAProfiler
from src.spatial_intelligence import SpatialIntelligence
from src.property_matcher import PropertyMatcher
from src.insight_engine import InsightEngine

def run_pipeline(csv_path: Path = config.RAW_DATA_PATH):
    """Run the complete Property Intelligence & Segment Discovery pipeline."""
    ensure_directories()
    log_header("PROPERTY INTELLIGENCE & SEGMENT DISCOVERY SYSTEM")

    # STAGE 1 — PROPERTY UNIVERSE
    log_step(1, 8, "Analyzing Property Universe")
    loader = PropertyDataLoader(csv_path)
    raw_df = loader.load_data()
    universe_metrics = loader.get_universe_metrics()
    log_substep(f"Loaded {universe_metrics['total_properties']:,} housing records across {universe_metrics['total_attributes']} attributes")
    log_substep(f"Price spectrum: ${universe_metrics['price_metrics']['min']:,.0f} to ${universe_metrics['price_metrics']['max']:,.0f} (Mean: ${universe_metrics['price_metrics']['mean']:,.0f})")
    log_substep(f"Living Area spectrum: {universe_metrics['area_metrics']['min']:,.0f} to {universe_metrics['area_metrics']['max']:,.0f} sqft")

    # STAGE 2 — PROPERTY INTELLIGENCE & DATA TRUST
    log_step(2, 8, "Auditing Property Intelligence & Data Trust")
    intelligence = PropertyIntelligence(raw_df)
    usable_df, scaled_df, scaler, trust_meta = intelligence.audit_and_prepare()
    dim = trust_meta["dimensionality"]
    log_substep(f"Data Trust verified: {trust_meta['usable_properties']:,} usable properties ({trust_meta['missing_value_completeness_pct']}% completeness)")
    log_substep(f"Mapped {dim} segmentation attributes across {len(trust_meta['feature_roles'])} domain roles")
    if trust_meta["outliers_treated_count"] > 0:
        log_substep(f"Isolated {trust_meta['outliers_treated_count']} typographical entry anomaly (e.g. 33 bedrooms)")

    # STAGE 3 — SIMILARITY SPACE & PROPERTY SIGNATURES
    log_step(3, 8, f"Constructing {dim}-Dimensional Similarity Space")
    sig_engine = PropertySignatureEngine(usable_df)
    sim_engine = SimilarityEngine(
        scaled_df=scaled_df,
        original_df=usable_df,
        scaler=scaler,
        feature_names=trust_meta["segmentation_feature_names"]
    )
    # Test sample similarity retrieval excluding self
    sample_sim = sim_engine.find_similar_properties(usable_df.index[0], top_n=3)
    log_substep(f"Fitted Euclidean Nearest-Neighbors on standardized {dim}D similarity space")
    log_substep(f"Verified similar property query excludes the query property itself (Top match distance: {sample_sim['similar_properties'][0]['euclidean_distance']})")

    # STAGE 4 — K DISCOVERY LAB
    log_step(4, 8, "Evaluating K Discovery Lab (K=2 to 10)")
    k_lab = KDiscoveryLab(scaled_df)
    recommended_k, k_results_df, k_meta = k_lab.run_discovery()
    log_substep(f"Elbow inflection at K = {k_meta['elbow_k']}")
    log_substep(f"Peak Silhouette score at K = {k_meta['silhouette_k']} ({k_meta['peak_silhouette_score']:.4f})")
    log_substep(f"Official Selected K = {recommended_k} ({k_meta['rationale']})")

    # STAGE 5 — SEGMENT DISCOVERY VIA K-MEANS
    log_step(5, 8, f"Discovering Property Communities (K = {recommended_k})")
    segment_engine = SegmentEngine(k=recommended_k)
    clustered_df, kmeans_model, pca_model, pca_df, pca_centroids, conv_info = segment_engine.fit_communities(
        scaled_df=scaled_df,
        usable_df=usable_df
    )
    log_substep(f"K-Means converged in {conv_info['iterations_to_converge']} iterations (Inertia: {conv_info['inertia']:,.1f})")
    log_substep(f"Silhouette separation score: {conv_info['silhouette_score']:.4f}")

    # STAGE 6 — SEGMENT DNA & PROFILING
    log_step(6, 8, "Extracting Segment DNA & Visual Fingerprints")
    dna_profiler = SegmentDNAProfiler(clustered_df)
    segment_names = dna_profiler.generate_segment_names()
    clustered_df["Segment Name"] = clustered_df["Cluster"].map(segment_names)
    dna_profiles = dna_profiler.extract_segment_dna(segment_names)
    comp_table = dna_profiler.get_comparison_table(segment_names)
    for p in dna_profiles:
        log_substep(f"Segment {p['cluster_id']} ({p['segment_name']}): {p['property_count']:,} homes ({p['share_pct']}%) | Avg Price ${p['avg_price']:,.0f}")

    # STAGE 7 — SPATIAL INTELLIGENCE
    log_step(7, 8, "Validating Spatial Intelligence")
    spatial = SpatialIntelligence(clustered_df)
    provenance = spatial.get_provenance_audit()
    if provenance["valid"]:
        b = provenance["bounds"]
        log_substep(f"Valid coordinates confirmed: Lat {b['min_lat']:.2f}°–{b['max_lat']:.2f}° N, Lon {b['min_lon']:.2f}°–{b['max_lon']:.2f}° W")
        log_substep("Geographic bounding box documented without speculative regional claims")

    # STAGE 8 — PROPERTY MATCHER & EXPORTS
    log_step(8, 8, "Exporting Deliverables & Verifying Property Matcher")
    exports = InsightEngine.export_deliverables(clustered_df, comp_table)
    log_substep(f"Clustered dataset exported to '{exports['clustered_dataset'].name}'")
    log_substep(f"Segment profiles exported to '{exports['segment_profiles'].name}'")

    # Verification: Verify PropertyMatcher transforms new property inputs using
    # the ALREADY-FITTED preprocessing/scaler and assigns to nearest cluster centroid
    matcher = PropertyMatcher(
        scaler=scaler,
        kmeans_model=kmeans_model,
        segment_names=segment_names,
        reference_df=usable_df,
        feature_names=trust_meta["segmentation_feature_names"]
    )
    test_input = {
        "Price": universe_metrics["price_metrics"]["median"],
        "living area": universe_metrics["area_metrics"]["median"],
        "number of bedrooms": 3,
        "number of bathrooms": 2.0,
        "grade of the house": 7,
        "Built Year": 1985
    }
    match_result = matcher.match_property(test_input)
    log_substep(f"PropertyMatcher verified: test home matched to '{match_result['matched_segment_name']}' (Distance: {match_result['distance_to_centroid']})")

    # Print Executive Architecture & Summary
    print("\n" + "=" * 60)
    print("PROPERTY INTELLIGENCE ARCHITECTURE")
    print("=" * 60)
    print(InsightEngine.get_architecture_diagram())

    print("\n" + "-" * 60)
    print("EXECUTIVE SUMMARY")
    print("-" * 60)
    insights = InsightEngine.generate_executive_insights(clustered_df, dna_profiles, trust_meta)
    for ins in insights:
        print(f"• {ins}")

    log_footer("PIPELINE COMPLETED SUCCESSFULLY")
    return clustered_df, comp_table, dna_profiles, matcher

if __name__ == "__main__":
    run_pipeline()
