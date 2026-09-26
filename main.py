"""
MAIN PIPELINE RUNNER — Indian Rental Property Intelligence & Segment Discovery
Uses K-Means clustering on standardized multi-attribute log-transformed similarity space.
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from src.utils import ensure_directories, log_header, log_step, log_substep, log_footer, format_currency
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
    """Run the complete Indian Rental Property Intelligence pipeline."""
    ensure_directories()
    log_header("INDIAN RENTAL PROPERTY INTELLIGENCE & SEGMENT DISCOVERY")

    # STAGE 1 — PROPERTY UNIVERSE
    log_step(1, 8, "Analyzing Indian Rental Universe")
    loader = PropertyDataLoader(csv_path)
    raw_df = loader.load_data()
    universe_metrics = loader.get_universe_metrics()
    log_substep(f"Loaded {universe_metrics['total_properties']:,} rental properties across {universe_metrics['total_attributes']} attributes")
    log_substep(f"Rent spectrum: {format_currency(universe_metrics['price_metrics']['min'])} to {format_currency(universe_metrics['price_metrics']['max'])} / month (Median: {format_currency(universe_metrics['price_metrics']['median'])})")
    log_substep(f"Living Area spectrum: {universe_metrics['area_metrics']['min']:,.0f} to {universe_metrics['area_metrics']['max']:,.0f} sqft (Median: {universe_metrics['area_metrics']['median']:,.0f} sqft)")
    log_substep(f"Metropolitan coverage: {universe_metrics['city_counts']}")

    # STAGE 2 — DATA TRUST & SIMILARITY SPACE CONSTRUCTION
    log_step(2, 8, "Auditing Data Trust & Preparing Similarity Space")
    intelligence = PropertyIntelligence(raw_df)
    usable_df, scaled_df, scaler, trust_meta = intelligence.audit_and_prepare()
    dim = trust_meta["dimensionality"]
    log_substep(f"Data Trust verified: {trust_meta['usable_properties']:,} properties ({trust_meta['missing_value_completeness_pct']}% completeness)")
    log_substep(f"Constructed {dim}D similarity space: {trust_meta['segmentation_feature_names']}")
    log_substep(f"Applied log1p transformation to positive right-skewed features: {trust_meta['log_transformed_features']}")

    # STAGE 3 — SIMILARITY ENGINE & PROPERTY SIGNATURES
    log_step(3, 8, f"Constructing {dim}-Dimensional Similarity Search")
    sig_engine = PropertySignatureEngine(usable_df)
    sim_engine = SimilarityEngine(
        scaled_df=scaled_df,
        original_df=usable_df,
        scaler=scaler,
        feature_names=trust_meta["segmentation_feature_names"]
    )
    first_id = usable_df["property_id"].iloc[0] if "property_id" in usable_df.columns else 1
    sample_sim = sim_engine.find_similar_properties(first_id, top_n=3)
    log_substep(f"Fitted Euclidean Nearest-Neighbors on standardized {dim}D similarity space")
    log_substep(f"Verified similar property query excludes the query property itself (Top match distance: {sample_sim['similar_properties'][0]['euclidean_distance']})")

    # STAGE 4 — K DISCOVERY LAB
    log_step(4, 8, "Evaluating K Discovery Lab (K=2 to 10)")
    k_lab = KDiscoveryLab(scaled_df)
    recommended_k, k_results_df, k_meta = k_lab.run_discovery()
    log_substep(f"Elbow inflection at K = {k_meta['elbow_k']}")
    log_substep(f"Selected K = {recommended_k} ({k_meta['rationale']})")

    # STAGE 5 — SEGMENT DISCOVERY VIA K-MEANS
    log_step(5, 8, f"Discovering Rental Communities (K = {recommended_k})")
    segment_engine = SegmentEngine(k=recommended_k)
    clustered_df, kmeans_model, pca_model, pca_df, pca_centroids, conv_info = segment_engine.fit_communities(
        scaled_df=scaled_df,
        usable_df=usable_df
    )
    log_substep(f"K-Means converged in {conv_info['iterations_to_converge']} iterations (Inertia: {conv_info['inertia']:,.1f})")
    log_substep(f"Silhouette separation score: {conv_info['silhouette_score']:.4f}")
    log_substep(f"2D PCA projection explained variance: {segment_engine.pca_variance_ratio} (Total: {sum(segment_engine.pca_variance_ratio)*100:.1f}%)")

    # STAGE 6 — SEGMENT DNA & PROFILING
    log_step(6, 8, "Extracting Segment DNA & Dynamic Profiles")
    dna_profiler = SegmentDNAProfiler(clustered_df)
    segment_names = dna_profiler.generate_segment_names()
    clustered_df["cluster_name"] = clustered_df["Cluster"].map(segment_names)
    dna_dict = dna_profiler.extract_segment_dna(segment_names)

    # Build comparison summary DataFrame
    comp_rows = []
    for c_id, profile in dna_dict.items():
        comp_rows.append({
            "Cluster": c_id,
            "Segment Name": profile["segment_name"],
            "Properties": profile["property_count"],
            "Market Share": f"{profile['market_share_pct']}%",
            "Median Rent": profile["price"]["formatted_median"],
            "Median Area": profile["area"]["formatted"],
            "Typical BHK": profile["specs"]["bhk_display"],
            "Typical Baths": profile["specs"]["baths_display"],
            "Dominant Furnishing": profile["furnishing"]["dominant_status"],
            "Top City": profile["geography"]["dominant_city"]
        })
        log_substep(f"Segment {c_id} [{profile['segment_name']}]: {profile['property_count']:,} homes ({profile['market_share_pct']}%) | Median Rent: {profile['price']['formatted_median']} | {profile['specs']['bhk_display']}")

    comp_df = pd.DataFrame(comp_rows)

    # STAGE 7 — SPATIAL INTELLIGENCE
    log_step(7, 8, "Validating Spatial Intelligence")
    spatial = SpatialIntelligence(clustered_df)
    provenance = spatial.get_provenance_audit()
    b = provenance["bounds"]
    log_substep(f"Validated Indian coordinates confirmed: Lat {b['min_lat']:.4f}°–{b['max_lat']:.4f}° N, Lon {b['min_lon']:.4f}°–{b['max_lon']:.4f}° E")
    log_substep(f"Verified map coordinates: {provenance['valid_record_count']:,} / {provenance['total_records']:,} ({provenance['valid_pct']}%)")

    # STAGE 8 — PROPERTY MATCHER & EXPORTS
    log_step(8, 8, "Exporting Deliverables & Verifying Property Matcher")
    exports = InsightEngine.export_deliverables(clustered_df, comp_df)
    log_substep(f"Clustered dataset exported to '{exports['clustered_dataset'].name}'")
    log_substep(f"Segment profiles exported to '{exports['segment_profiles'].name}'")

    # Verify PropertyMatcher
    matcher = PropertyMatcher(
        scaler=scaler,
        kmeans_model=kmeans_model,
        segment_names=segment_names,
        reference_df=usable_df,
        feature_names=trust_meta["segmentation_feature_names"],
        log_features=trust_meta["log_transformed_features"]
    )
    test_input = {
        "price": universe_metrics["price_metrics"]["median"],
        "sqft": universe_metrics["area_metrics"]["median"],
        "bhk": 2.0,
        "numBathrooms": 2.0,
        "furnishing_tier": 1
    }
    match_result = matcher.match_property(test_input)
    log_substep(f"Property Matcher test: Input {test_input} -> {match_result['headline']} (Distance: {match_result['distance_to_centroid']})")

    # Executive Insights
    insights = InsightEngine.generate_executive_insights(clustered_df, dna_dict, trust_meta)
    print("\n--- EXECUTIVE MARKET INSIGHTS ---")
    for ins in insights:
        print(f"• {ins}")

    log_footer("INDIAN RENTAL PROPERTY INTELLIGENCE PIPELINE COMPLETED")
    return clustered_df, kmeans_model, pca_model, scaler, dna_dict, segment_names

if __name__ == "__main__":
    run_pipeline()
