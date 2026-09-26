"""
Automated Verification Suite for Property Intelligence & Segment Discovery System.
Verifies all 9 stages, zero supervised metrics, proper exclusion in similarity search,
and fitted scaler usage in property matching.
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
    print("RUNNING AUTOMATED VERIFICATION SUITE...\n")

    # 1. Property Universe
    loader = PropertyDataLoader(config.RAW_DATA_PATH)
    raw_df = loader.load_data()
    metrics = loader.get_universe_metrics()
    assert metrics["total_properties"] > 0, "Failed: total properties must be > 0"
    assert metrics["price_metrics"]["min"] > 0, "Failed: min price must be > 0"
    print(f"✓ Test 1 Passed: Property Universe metrics dynamic ({metrics['total_properties']:,} homes)")

    # 2. Property Intelligence & Data Trust
    intelligence = PropertyIntelligence(raw_df)
    usable_df, scaled_df, scaler, trust_meta = intelligence.audit_and_prepare()
    dim = trust_meta["dimensionality"]
    assert dim > 0, "Failed: dimensionality must be > 0"
    assert trust_meta["usable_properties"] == len(usable_df), "Failed: usable count mismatch"
    assert trust_meta["missing_value_completeness_pct"] == 100.0, "Failed: completeness must be 100%"
    assert trust_meta["outliers_treated_count"] == 1, "Failed: exactly 1 bedroom anomaly should be treated"
    print(f"✓ Test 2 Passed: Data Trust & Feature Roles verified ({dim}D space, 1 documented outlier)")

    # 3. Property Signatures
    sig_engine = PropertySignatureEngine(usable_df)
    sample_sig = sig_engine.generate_signature(usable_df.iloc[0])
    for d in sample_sig["dimensions"]:
        assert 0 <= d["score_pct"] <= 100, f"Failed: signature pct out of range: {d['score_pct']}"
        assert "probability" not in d["descriptor"].lower(), "Failed: must not use probability in signature"
    print("✓ Test 3 Passed: Property Signature generated with relative normalized indicators")

    # 4. Similarity Space & Self-Exclusion
    sim_engine = SimilarityEngine(scaled_df, usable_df, scaler, trust_meta["segmentation_feature_names"])
    query_id = usable_df.iloc[0].get("id", 0)
    sim_result = sim_engine.find_similar_properties(query_id, top_n=5)
    assert len(sim_result["similar_properties"]) == 5, "Failed: must return 5 similar properties"
    for peer in sim_result["similar_properties"]:
        assert peer["id"] != query_id, f"CRITICAL FAILED: Queried property {query_id} was returned in similarity results!"
        assert peer["euclidean_distance"] >= 0, "Failed: distance must be non-negative"
    print("✓ Test 4 Passed: Similar Property Finder correctly excludes the query property itself")

    # 5. K Discovery Lab
    k_lab = KDiscoveryLab(scaled_df, sample_size=1000)
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
    print(f"✓ Test 6 Passed: Segment Engine converged in {conv_info['iterations_to_converge']} iterations")

    # 7. Segment DNA & Dynamic Naming
    dna_profiler = SegmentDNAProfiler(clustered_df)
    names = dna_profiler.generate_segment_names()
    assert len(names) == rec_k, "Failed: segment name count mismatch"
    dna_profiles = dna_profiler.extract_segment_dna(names)
    assert len(dna_profiles) == rec_k, "Failed: DNA profile count mismatch"
    print(f"✓ Test 7 Passed: Segment DNA generated dynamic names: {list(names.values())}")

    # 8. Spatial Intelligence
    spatial = SpatialIntelligence(clustered_df)
    provenance = spatial.get_provenance_audit()
    assert provenance["valid"] is True, "Failed: coordinates should be valid"
    assert 52.0 <= provenance["bounds"]["min_lat"] <= 54.0, "Failed: latitude bounds mismatch"
    print(f"✓ Test 8 Passed: Spatial Intelligence provenance audit verified ({provenance['bounds']['min_lat']:.2f}° to {provenance['bounds']['max_lat']:.2f}° N)")

    # 9. Property Matcher with Fitted Scaler & Bound Validation
    matcher = PropertyMatcher(
        scaler=scaler,
        kmeans_model=kmeans_model,
        segment_names=names,
        reference_df=usable_df,
        feature_names=trust_meta["segmentation_feature_names"]
    )
    test_normal = {
        "Price": metrics["price_metrics"]["median"],
        "living area": metrics["area_metrics"]["median"],
        "number of bedrooms": 3,
        "number of bathrooms": 2.0,
        "grade of the house": 7,
        "Built Year": 1990
    }
    match_norm = matcher.match_property(test_normal)
    assert match_norm["matched_cluster_id"] in names, "Failed: invalid matched cluster"
    assert len(match_norm["validation_warnings"]) == 0, "Normal input should not trigger warnings"

    test_extreme = {
        "Price": 999999999.0, # extreme price
        "living area": metrics["area_metrics"]["median"],
        "number of bedrooms": 50, # extreme bedrooms
    }
    match_ext = matcher.match_property(test_extreme)
    assert len(match_ext["validation_warnings"]) > 0, "Failed: extreme input must trigger validation warnings"
    print("✓ Test 9 Passed: Property Matcher transforms via fitted scaler and warns on extreme bounds")

    # 10. Audit for forbidden supervised terms
    forbidden_terms = ["accuracy", "precision", "recall", "confusion_matrix", "train_test_split", "test_train", "f1_score"]
    code_files = [
        "app.py", "main.py", "src/property_data.py", "src/property_intelligence.py",
        "src/property_signature.py", "src/similarity_engine.py", "src/k_discovery.py",
        "src/segment_engine.py", "src/segment_dna.py", "src/spatial_intelligence.py",
        "src/property_matcher.py", "src/insight_engine.py"
    ]
    for c_file in code_files:
        content = (BASE_DIR / c_file).read_text(encoding="utf-8").lower()
        for term in forbidden_terms:
            assert term not in content, f"CRITICAL FAILED: Found forbidden supervised term '{term}' in {c_file}!"
    print("✓ Test 10 Passed: Zero supervised classification metrics found across all codebase files")

    print("\nALL 10 VERIFICATION TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_tests()
