"""
Automated Verification Suite for Indian Rental Property Intelligence System.
Verifies all 9 stages: Data Trust, Multi-Attribute Similarity Space, Self-Exclusion in Similarity,
K Discovery (2-10), Segment DNA, Indian Coordinates Spatial Intelligence, and Property Matcher.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
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

def run_tests():
    print("RUNNING AUTOMATED VERIFICATION SUITE (INDIAN RENTAL DATASET)...\n")

    # 1. Property Universe
    loader = PropertyDataLoader(config.RAW_DATA_PATH)
    raw_df = loader.load_data()
    metrics = loader.get_universe_metrics()
    assert metrics["total_properties"] == 12847, f"Failed: total properties expected 12,847, got {metrics['total_properties']}"
    assert metrics["price_metrics"]["min"] >= 1500, "Failed: min rent must be >= 1,500 INR"
    assert "Mumbai" in metrics["city_counts"] and "Delhi" in metrics["city_counts"] and "Pune" in metrics["city_counts"], "Failed: 3 cities must be present"
    print(f"✓ Test 1 Passed: Property Universe verified ({metrics['total_properties']:,} properties across Delhi, Mumbai, Pune)")

    # 2. Property Intelligence & Data Trust
    intelligence = PropertyIntelligence(raw_df)
    usable_df, scaled_df, scaler, trust_meta = intelligence.audit_and_prepare()
    dim = trust_meta["dimensionality"]
    assert dim == 5, f"Failed: dimensionality must be exactly 5, got {dim}"
    assert trust_meta["usable_properties"] == 12847, "Failed: usable count mismatch"
    assert trust_meta["missing_value_completeness_pct"] == 100.0, "Failed: completeness must be 100%"
    assert set(trust_meta["log_transformed_features"]) == {"price", "sqft"}, "Failed: log1p must apply to price and sqft"
    print(f"✓ Test 2 Passed: Data Trust & Feature Roles verified ({dim}D similarity space, 100% completeness)")

    # 3. Property Signatures
    sig_engine = PropertySignatureEngine(usable_df)
    sample_sig = sig_engine.generate_signature(usable_df.iloc[0])
    for d in sample_sig["dimensions"]:
        assert 0 <= d["score_pct"] <= 100, f"Failed: signature pct out of range: {d['score_pct']}"
        assert "probability" not in d["descriptor"].lower(), "Failed: must not use probability in signature"
    assert len(sample_sig["dimensions"]) == 5, "Failed: signature must have 5 dimensions"
    print("✓ Test 3 Passed: Property Signature generated with relative normalized indicators (no probabilities)")

    # 4. Similarity Space & Self-Exclusion
    sim_engine = SimilarityEngine(scaled_df, usable_df, scaler, trust_meta["segmentation_feature_names"])
    query_id = usable_df.iloc[0].get("property_id", 1)
    sim_result = sim_engine.find_similar_properties(query_id, top_n=5)
    assert len(sim_result["similar_properties"]) == 5, "Failed: must return 5 similar properties"
    for peer in sim_result["similar_properties"]:
        assert peer["id"] != query_id, f"CRITICAL FAILED: Queried property {query_id} was returned in similarity results!"
        assert peer["euclidean_distance"] >= 0, "Failed: distance must be non-negative"
    print("✓ Test 4 Passed: Similar Property Finder correctly excludes the query property itself")

    # 5. K Discovery Lab
    k_lab = KDiscoveryLab(scaled_df, sample_size=3000)
    rec_k, k_df, k_meta = k_lab.run_discovery()
    assert rec_k in range(config.K_MIN, config.K_MAX + 1), "Failed: recommended K out of range"
    assert len(k_df) == 9, "Failed: must test K=2 through 10"
    print(f"✓ Test 5 Passed: K Discovery Lab evaluated K=2..10 (Selected K = {rec_k})")

    # 6. Segment Engine & PCA Landscape
    seg_engine = SegmentEngine(k=rec_k)
    clustered_df, kmeans_model, pca_model, pca_df, pca_centroids, conv_info = seg_engine.fit_communities(
        scaled_df, usable_df
    )
    assert "Cluster" in clustered_df.columns, "Failed: Cluster column missing"
    assert len(pca_centroids) == rec_k, "Failed: centroid count mismatch in PCA projection"
    assert sum(seg_engine.pca_variance_ratio) > 0.80, "Failed: PCA explained variance must exceed 80%"
    print(f"✓ Test 6 Passed: Segment Engine converged in {conv_info['iterations_to_converge']} iterations (PCA variance: {sum(seg_engine.pca_variance_ratio)*100:.1f}%)")

    # 7. Segment DNA & Dynamic Naming
    dna_profiler = SegmentDNAProfiler(clustered_df)
    names = dna_profiler.generate_segment_names()
    assert len(names) == rec_k, "Failed: segment name count mismatch"
    dna_profiles = dna_profiler.extract_segment_dna(names)
    assert len(dna_profiles) == rec_k, "Failed: DNA profile count mismatch"
    for c_id, profile in dna_profiles.items():
        assert profile["property_count"] > 0, "Failed: cluster must not be empty"
        assert profile["price"]["median"] > 0, "Failed: median rent must be positive"
    print(f"✓ Test 7 Passed: Segment DNA generated dynamic names: {list(names.values())}")

    # 8. Spatial Intelligence & Validated Indian Coordinates
    spatial = SpatialIntelligence(clustered_df)
    provenance = spatial.get_provenance_audit()
    assert provenance["valid"] is True, "Failed: coordinates should be valid"
    assert 18.0 <= provenance["bounds"]["min_lat"] <= 20.0, "Failed: min latitude must be around Mumbai/Pune (~18-19°N)"
    assert 28.0 <= provenance["bounds"]["max_lat"] <= 30.0, "Failed: max latitude must be around Delhi NCR (~28-29°N)"
    assert 72.0 <= provenance["bounds"]["min_lon"] <= 74.0, "Failed: min longitude must be Western India (~72-73°E)"
    assert provenance["valid_pct"] > 99.0, "Failed: over 99% of coordinates must be valid"
    print(f"✓ Test 8 Passed: Spatial Intelligence confirmed validated Indian coordinates ({provenance['bounds']['min_lat']:.2f}° to {provenance['bounds']['max_lat']:.2f}° N, {provenance['valid_pct']}% valid)")

    # 9. Property Matcher with Fitted Pipeline & Bound Validation
    matcher = PropertyMatcher(
        scaler=scaler,
        kmeans_model=kmeans_model,
        segment_names=names,
        reference_df=usable_df,
        feature_names=trust_meta["segmentation_feature_names"],
        log_features=trust_meta["log_transformed_features"]
    )
    test_property = {
        "price": 28000,
        "sqft": 950,
        "bhk": 2,
        "numBathrooms": 2,
        "furnishing_tier": 1
    }
    match = matcher.match_property(test_property)
    assert "closest to" in match["headline"].lower(), "Failed: match headline must state closest segment"
    assert match["distance_to_centroid"] >= 0, "Failed: distance must be non-negative"
    assert "accuracy" not in str(match).lower(), "Failed: must not show fake accuracy"
    assert "probability" not in str(match).lower(), "Failed: must not show fake probabilities"
    print(f"✓ Test 9 Passed: Property Matcher verified using pre-fitted pipeline -> {match['headline']}")

    print("\nALL 9 SYSTEM INTEGRATION TESTS PASSED WITH 100% INTEGRITY!")

if __name__ == "__main__":
    run_tests()
