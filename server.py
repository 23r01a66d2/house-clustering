"""
PROPERTY INTELLIGENCE — Web Server (Flask)
HOUSE CLUSTERING USING K-MEANS CLUSTERING TECHNIQUES

High-performance web application backend serving interactive property discovery,
K-Means community exploration, Segment DNA profiling, spatial mapping, and property matching.
Pre-caches all fitted ML models, scalers, and similarity indices at startup.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file, abort

# Set Base Path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from src.utils import format_currency
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

# Initialize Flask App
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["JSON_SORT_KEYS"] = False
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

@app.context_processor
def inject_globals():
    import time
    return {"cache_buster": int(time.time())}

@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# ==========================================
# MODEL & DATA CACHE (INITIALIZED ON STARTUP)
# ==========================================
class AppContext:
    def __init__(self):
        print("\n" + "=" * 60)
        print(" INITIALIZING INDIAN RENTAL PROPERTY INTELLIGENCE BACKEND...")
        print("=" * 60)

        # 1. Load Raw/Processed Inventory
        self.data_loader = PropertyDataLoader(config.RAW_DATA_PATH)
        self.raw_df = self.data_loader.load_data()
        self.universe_metrics = self.data_loader.get_universe_metrics()
        print(f"  ✓ Loaded {self.universe_metrics['total_properties']:,} rental properties across Delhi, Mumbai, Pune")

        # 2. Audit Property Intelligence & Fit Scaler (Multi-Attribute Similarity Space)
        self.intelligence = PropertyIntelligence(self.raw_df)
        self.usable_df, self.scaled_df, self.scaler, self.trust_meta = self.intelligence.audit_and_prepare()
        self.n_dimensions = self.trust_meta["dimensionality"]
        self.features = self.trust_meta["segmentation_feature_names"]
        self.log_features = self.trust_meta["log_transformed_features"]
        print(f"  ✓ Fitted log1p + StandardScaler on {self.n_dimensions}D similarity space ({self.features})")

        # 3. Evaluate K Discovery Lab (K=2..10)
        self.k_lab = KDiscoveryLab(self.scaled_df)
        self.recommended_k, self.k_results_df, self.k_meta = self.k_lab.run_discovery()
        print(f"  ✓ K Discovery evaluated: Elbow K={self.k_meta['elbow_k']}, Selected K={self.recommended_k}")

        # 4. Fit K-Means Community Model with Selected K
        self.segment_engine = SegmentEngine(k=self.recommended_k)
        (
            self.clustered_df,
            self.kmeans_model,
            self.pca_model,
            self.pca_df,
            self.pca_centroids,
            self.conv_info
        ) = self.segment_engine.fit_communities(self.scaled_df, self.usable_df)
        print(f"  ✓ K-Means converged in {self.conv_info['iterations_to_converge']} iterations (Silhouette: {self.conv_info['silhouette_score']:.4f})")

        # 5. Extract Dynamic Segment Names & DNA Profiles
        self.dna_profiler = SegmentDNAProfiler(self.clustered_df)
        self.segment_names = self.dna_profiler.generate_segment_names()
        self.clustered_df["cluster_name"] = self.clustered_df["Cluster"].map(self.segment_names)
        self.pca_df["cluster_name"] = self.pca_df["Cluster"].map(self.segment_names)
        self.dna_dict = self.dna_profiler.extract_segment_dna(self.segment_names)
        self.dna_profiles = list(self.dna_dict.values())

        # Build comparison summary DataFrame
        comp_rows = []
        for c_id, profile in self.dna_dict.items():
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
        self.comp_table = pd.DataFrame(comp_rows)
        print(f"  ✓ Discovered {len(self.segment_names)} property communities: {list(self.segment_names.values())}")

        # 6. Initialize Normalized Signature Engine
        self.sig_engine = PropertySignatureEngine(self.usable_df)

        # 7. Initialize Standardized Euclidean Similarity Engine
        self.sim_engine = SimilarityEngine(
            scaled_df=self.scaled_df,
            original_df=self.clustered_df,
            scaler=self.scaler,
            feature_names=self.features
        )
        print(f"  ✓ Initialized NearestNeighbors index on {self.n_dimensions}D similarity space")

        # 8. Spatial Intelligence & Validated Indian Coordinates Validation
        self.spatial = SpatialIntelligence(self.clustered_df)
        self.provenance = self.spatial.get_provenance_audit()
        print(f"  ✓ Spatial Intelligence verified {self.provenance['valid_record_count']:,} coordinates across Delhi, Mumbai, Pune")

        # 9. Property Matcher (pre-fitted scaler & centroids)
        self.matcher = PropertyMatcher(
            scaler=self.scaler,
            kmeans_model=self.kmeans_model,
            segment_names=self.segment_names,
            reference_df=self.usable_df,
            feature_names=self.features,
            log_features=self.log_features
        )

        # 10. Pre-generate and save CSV deliverables
        self.export_paths = InsightEngine.export_deliverables(self.clustered_df, self.comp_table)
        print(f"  ✓ Export deliverables saved to {config.OUTPUTS_DIR.name}/")
        print("=" * 60)
        print(" BACKEND INITIALIZATION COMPLETE. READY TO SERVE REQUESTS.")
        print("=" * 60 + "\n")

# Global Cache Singleton
CTX = AppContext()

# Global dynamic cluster palette generator
def get_cluster_palette(k: int) -> List[str]:
    """Dynamically generates consistent hex colors for K clusters."""
    base_palette = ["#10B981", "#38BDF8", "#F59E0B", "#A855F7", "#EC4899", "#6366F1", "#14B8A6", "#E11D48"]
    if k <= len(base_palette):
        return base_palette[:k]
    import colorsys
    colors = []
    for i in range(k):
        hue = i / float(k)
        rgb = colorsys.hsv_to_rgb(hue, 0.75, 0.85)
        hex_c = f"#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}"
        colors.append(hex_c)
    return colors


# ==========================================
# PAGE ROUTES (HTML)
# ==========================================

@app.route("/")
def home():
    """Landing Page: Hero, dynamic universe spectrum, feature pillars, landscape preview, story journey."""
    cluster_colors = get_cluster_palette(len(CTX.segment_names))
    color_map = {c_id: cluster_colors[i] for i, c_id in enumerate(sorted(CTX.segment_names.keys()))}

    return render_template(
        "index.html",
        active_page="home",
        metrics=CTX.universe_metrics,
        trust=CTX.trust_meta,
        recommended_k=CTX.recommended_k,
        segment_names=CTX.segment_names,
        dna_profiles=CTX.dna_profiles,
        color_map=color_map,
        n_features=CTX.n_dimensions
    )

@app.route("/explore")
def explore():
    """Explore Properties Page: Sticky filter bar, property cards grid, detail modal, spatial map."""
    cluster_colors = get_cluster_palette(len(CTX.segment_names))
    color_map = {c_id: cluster_colors[i] for i, c_id in enumerate(sorted(CTX.segment_names.keys()))}

    # Dynamic filter ranges from processed data
    price_min = int(CTX.usable_df["price"].min())
    price_max = int(CTX.usable_df["price"].max())
    area_min = int(CTX.usable_df["sqft"].min())
    area_max = int(CTX.usable_df["sqft"].max())
    
    cities = sorted(list(CTX.usable_df["city_clean"].unique()))
    bed_options = sorted([int(b) for b in CTX.usable_df["bhk"].dropna().unique() if b <= 8])
    furnishing_options = ["Unfurnished", "Semi-Furnished", "Furnished"]
    typology_options = sorted(list(CTX.usable_df["typology"].unique()))

    return render_template(
        "explore.html",
        active_page="explore",
        metrics=CTX.universe_metrics,
        segment_names=CTX.segment_names,
        color_map=color_map,
        price_min=price_min,
        price_max=price_max,
        area_min=area_min,
        area_max=area_max,
        cities=cities,
        bed_options=bed_options,
        furnishing_options=furnishing_options,
        typology_options=typology_options,
        provenance=CTX.provenance,
        n_features=CTX.n_dimensions
    )

@app.route("/clusters")
def clusters():
    """Core Academic Clustering Page: K Discovery Lab, K-Means convergence, 2D PCA landscape."""
    cluster_colors = get_cluster_palette(len(CTX.segment_names))
    color_map = {c_id: cluster_colors[i] for i, c_id in enumerate(sorted(CTX.segment_names.keys()))}

    pca_variance = [round(v * 100, 1) for v in CTX.segment_engine.pca_variance_ratio]

    return render_template(
        "clusters.html",
        active_page="clusters",
        k_meta=CTX.k_meta,
        conv_info=CTX.conv_info,
        recommended_k=CTX.recommended_k,
        segment_names=CTX.segment_names,
        color_map=color_map,
        pca_variance=pca_variance,
        n_features=CTX.n_dimensions,
        total_properties=len(CTX.usable_df)
    )

@app.route("/compare")
def compare():
    """Cluster Analysis Page: Dynamic cluster profiles, radar charts, cross-cluster comparison."""
    cluster_colors = get_cluster_palette(len(CTX.segment_names))
    color_map = {c_id: cluster_colors[i] for i, c_id in enumerate(sorted(CTX.segment_names.keys()))}

    return render_template(
        "compare.html",
        active_page="compare",
        dna_profiles=CTX.dna_profiles,
        segment_names=CTX.segment_names,
        comp_table=CTX.comp_table.to_dict(orient="records"),
        color_map=color_map,
        n_features=CTX.n_dimensions
    )

@app.route("/property-match")
def property_match():
    """Property Match: Interactive form matching 5 features with pre-fitted K-Means model."""
    cluster_colors = get_cluster_palette(len(CTX.segment_names))
    color_map = {c_id: cluster_colors[i] for i, c_id in enumerate(sorted(CTX.segment_names.keys()))}

    bounds = CTX.matcher.bounds
    defaults = {
        "price": int(bounds.get("price", {}).get("median", 32000)),
        "sqft": int(bounds.get("sqft", {}).get("median", 1000)),
        "bhk": int(round(bounds.get("bhk", {}).get("median", 2))),
        "numBathrooms": float(bounds.get("numBathrooms", {}).get("median", 2.0)),
        "furnishing_tier": 1
    }

    # Sample property IDs for Similar Property Finder
    sample_ids = [int(i) for i in CTX.clustered_df["property_id"].head(50).tolist()]

    return render_template(
        "property_match.html",
        active_page="property_match",
        segment_names=CTX.segment_names,
        color_map=color_map,
        defaults=defaults,
        bounds=bounds,
        sample_ids=sample_ids,
        n_features=CTX.n_dimensions
    )

@app.route("/methodology")
def methodology():
    """Methodology & Data Trust Page: Data hygiene, feature selection, mathematical foundations, architecture."""
    arch_diagram = InsightEngine.get_architecture_diagram(n_dimensions=CTX.n_dimensions)
    provenance = CTX.provenance
    feature_roles = {
        "Valuation & Economics": ["Monthly Rent (price) — Primary rental budget in ₹ (log1p transformed)"],
        "Spatial Capacity": ["Living Area (sqft) — Super built-up / carpet area (log1p transformed)"],
        "Architectural Layout": ["Bedrooms (bhk) — Number of bedrooms", "Bathrooms (numBathrooms) — Sanitation fixtures count"],
        "Asset Readiness": ["Furnishing Status (furnishing_tier) — 0: Unfurnished, 1: Semi-Furnished, 2: Furnished"],
        "Observational Analytics": ["Price per SqFt (price_per_sqft) — Retained for analysis; excluded from distance"],
        "Spatial Intelligence": ["City (city_clean), Locality (location), Latitude, Longitude — Geographic metadata"]
    }

    return render_template(
        "methodology.html",
        active_page="methodology",
        trust=CTX.trust_meta,
        provenance=provenance,
        architecture_diagram=arch_diagram,
        feature_roles=feature_roles,
        n_features=CTX.n_dimensions,
        recommended_k=CTX.recommended_k,
        conv_info=CTX.conv_info
    )



# ==========================================
# REST API ROUTES (JSON)
# ==========================================

def sanitize_for_json(obj):
    """Recursively converts numpy and pandas types to standard Python primitives for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    return obj

@app.route("/api/properties")
def api_properties():
    """Paginated and filtered property cards for /explore."""
    try:
        page = max(1, int(request.args.get("page", 1)))
        limit = min(50, max(6, int(request.args.get("limit", 12))))

        df = CTX.clustered_df

        # Apply Filters
        if request.args.get("city") and request.args.get("city") != "All":
            df = df[df["city_clean"] == request.args.get("city")]
        if request.args.get("location"):
            loc_term = request.args.get("location").strip().lower()
            df = df[df["location"].astype(str).str.lower().str.contains(loc_term, na=False)]
        if request.args.get("min_price"):
            df = df[df["price"] >= float(request.args.get("min_price"))]
        if request.args.get("max_price"):
            df = df[df["price"] <= float(request.args.get("max_price"))]
        if request.args.get("min_area"):
            df = df[df["sqft"] >= float(request.args.get("min_area"))]
        if request.args.get("max_area"):
            df = df[df["sqft"] <= float(request.args.get("max_area"))]
        if request.args.get("bhk"):
            df = df[df["bhk"] == float(request.args.get("bhk"))]
        if request.args.get("bathrooms"):
            df = df[df["numBathrooms"] >= float(request.args.get("bathrooms"))]
        if request.args.get("furnishing") and request.args.get("furnishing") != "All":
            df = df[df["Status"] == request.args.get("furnishing")]
        if request.args.get("typology") and request.args.get("typology") != "All":
            df = df[df["typology"] == request.args.get("typology")]
        if request.args.get("segment"):
            seg_val = int(request.args.get("segment"))
            df = df[df["Cluster"] == seg_val]

        filtered_count = len(df)
        total_pages = int(np.ceil(filtered_count / limit)) if filtered_count > 0 else 1
        page = min(page, total_pages)

        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        sub_df = df.iloc[start_idx:end_idx]

        cluster_colors = get_cluster_palette(len(CTX.segment_names))
        color_map = {c_id: cluster_colors[i] for i, c_id in enumerate(sorted(CTX.segment_names.keys()))}

        cards = []
        for idx, row in sub_df.iterrows():
            c_id = int(row.get("Cluster", 0))
            p_id = int(row.get("property_id", idx + 1))
            rent_val = float(row.get("price", 0))
            cards.append({
                "index": int(idx),
                "id": p_id,
                "price": rent_val,
                "price_formatted": f"₹{rent_val:,.0f}/mo",
                "sqft": float(row.get("sqft", 0)),
                "bhk": int(round(float(row.get("bhk", 1)))),
                "bathrooms": float(row.get("numBathrooms", 1)),
                "price_per_sqft": float(row.get("price_per_sqft", 0)),
                "city": str(row.get("city_clean", "")),
                "location": str(row.get("location", "")),
                "status": str(row.get("Status", "")),
                "typology": str(row.get("typology", "")),
                "deposit": str(row.get("SecurityDeposit", "No Deposit")),
                "cluster": c_id,
                "segment_name": str(row.get("cluster_name", f"Cluster {c_id}")),
                "cluster_color": color_map.get(c_id, "#10B981")
            })

        return jsonify({
            "page": page,
            "limit": limit,
            "total_count": len(CTX.clustered_df),
            "filtered_count": filtered_count,
            "total_pages": total_pages,
            "properties": cards
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/property/<property_id>")
def api_property_detail(property_id):
    """Fetches full attributes and normalized Property Signature for a single property modal."""
    try:
        df = CTX.clustered_df
        matches = df[df["property_id"] == int(property_id)] if "property_id" in df.columns else pd.DataFrame()
        
        if len(matches) == 0:
            try:
                idx = int(property_id)
                if idx in df.index:
                    row = df.loc[idx]
                else:
                    return jsonify({"error": f"Property #{property_id} not found."}), 404
            except (ValueError, TypeError):
                return jsonify({"error": f"Property #{property_id} not found."}), 404
        else:
            row = matches.iloc[0]

        # Generate normalized Property Signature
        sig = CTX.sig_engine.generate_signature(row)
        c_id = int(row.get("Cluster", 0))
        cluster_colors = get_cluster_palette(len(CTX.segment_names))
        color_map = {cid: cluster_colors[i] for i, cid in enumerate(sorted(CTX.segment_names.keys()))}

        rent_val = float(row.get("price", 0))
        p_id = int(row.get("property_id", property_id))

        return jsonify({
            "id": p_id,
            "price": rent_val,
            "price_formatted": f"₹{rent_val:,.0f}/mo",
            "sqft": float(row.get("sqft", 0)),
            "bhk": int(round(float(row.get("bhk", 1)))),
            "bathrooms": float(row.get("numBathrooms", 1)),
            "price_per_sqft": float(row.get("price_per_sqft", 0)),
            "city": str(row.get("city_clean", "")),
            "location": str(row.get("location", "")),
            "status": str(row.get("Status", "")),
            "typology": str(row.get("typology", "")),
            "deposit": str(row.get("SecurityDeposit", "No Deposit")),
            "description": str(row.get("description", "No detailed description provided.")),
            "cluster": c_id,
            "segment_name": str(row.get("cluster_name", f"Cluster {c_id}")),
            "cluster_color": color_map.get(c_id, "#10B981"),
            "signature": sig
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/k-metrics")
def api_k_metrics():
    """Elbow and Silhouette Score curves for Plotly.js chart."""
    metrics = CTX.k_results_df.to_dict(orient="records")
    return jsonify({
        "metrics": metrics,
        "elbow_k": CTX.k_meta["elbow_k"],
        "recommended_k": CTX.recommended_k,
        "selected_k": CTX.recommended_k,
        "highest_sil_k": CTX.k_meta.get("highest_sil_k", 2),
        "highest_sil_score": CTX.k_meta.get("highest_sil_score", 0.4521),
        "selected_sil_score": round(float(CTX.conv_info["silhouette_score"]), 4),
        "rationale": CTX.k_meta["rationale"]
    })

@app.route("/api/pca-landscape")
def api_pca_landscape():
    """PCA 2D projection data for Plotly.js property landscape."""
    sample_size = min(3000, len(CTX.pca_df))
    sub_df = CTX.pca_df.sample(n=sample_size, random_state=config.RANDOM_STATE)

    cluster_colors = get_cluster_palette(len(CTX.segment_names))
    color_map = {c_id: cluster_colors[i] for i, c_id in enumerate(sorted(CTX.segment_names.keys()))}

    points = []
    for idx, row in sub_df.iterrows():
        c_id = int(row.get("Cluster", 0))
        p_id = int(row.get("property_id", idx))
        points.append({
            "id": p_id,
            "x": round(float(row["PCA_1"]), 3),
            "y": round(float(row["PCA_2"]), 3),
            "cluster": c_id,
            "segment_name": str(row.get("cluster_name", f"Cluster {c_id}")),
            "price": float(row.get("price", 0)),
            "price_formatted": f"₹{float(row.get('price', 0)):,.0f}/mo",
            "sqft": float(row.get("sqft", 0)),
            "bhk": int(round(float(row.get("bhk", 1)))),
            "city": str(row.get("city_clean", "")),
            "location": str(row.get("location", "")),
            "color": color_map.get(c_id, "#10B981")
        })

    centroids = []
    for i, c in enumerate(CTX.pca_centroids):
        centroids.append({
            "cluster": i,
            "segment_name": CTX.segment_names.get(i, f"Cluster {i}"),
            "x": round(float(c[0]), 3),
            "y": round(float(c[1]), 3),
            "color": color_map.get(i, "#10B981")
        })

    variance = [round(float(v) * 100, 1) for v in CTX.segment_engine.pca_variance_ratio]

    return jsonify({
        "points": points,
        "centroids": centroids,
        "variance_ratio": variance,
        "total_variance_pct": round(sum(CTX.segment_engine.pca_variance_ratio) * 100, 1),
        "sample_size": sample_size,
        "total_properties": len(CTX.pca_df)
    })

@app.route("/api/map-properties")
def api_map_properties():
    """Coordinate-validated properties for interactive Leaflet.js map with OpenStreetMap tiles."""
    city_filter = request.args.get("city", "All")
    seg_filter = [int(request.args.get("segment"))] if request.args.get("segment") else None

    markers = CTX.spatial.filter_spatial_inventory(
        city_filter=city_filter,
        segment_filter=seg_filter,
        sample_limit=2500
    )

    cluster_colors = get_cluster_palette(len(CTX.segment_names))
    color_map = {c_id: cluster_colors[i] for i, c_id in enumerate(sorted(CTX.segment_names.keys()))}

    for m in markers:
        m["color"] = color_map.get(m["cluster"], "#10B981")
        m["segment_name"] = CTX.segment_names.get(m["cluster"], f"Cluster {m['cluster']}")

    return jsonify({
        "markers": markers,
        "total_plotted": len(markers),
        "provenance": CTX.provenance,
        "city_boxes": CTX.provenance.get("city_boxes", {})
    })

@app.route("/api/cluster-profiles")
def api_cluster_profiles():
    """Detailed Segment DNA profiles and cross-segment comparisons."""
    cluster_colors = get_cluster_palette(len(CTX.segment_names))
    color_map = {c_id: cluster_colors[i] for i, c_id in enumerate(sorted(CTX.segment_names.keys()))}

    enriched_profiles = []
    for p in CTX.dna_profiles:
        p_copy = dict(p)
        p_copy["color"] = color_map.get(p["cluster_id"], "#10B981")
        enriched_profiles.append(p_copy)

    return jsonify(sanitize_for_json({
        "profiles": enriched_profiles,
        "comparison_table": CTX.comp_table.to_dict(orient="records")
    }))

@app.route("/api/property-match", methods=["POST"])
def api_property_match():
    """
    Transforms user inputs via ALREADY-FITTED pipeline (log1p + StandardScaler, zero refitting),
    validates empirical bounds, and computes Euclidean proximity to centroids.
    Zero fake probabilities or accuracy numbers.
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No property attributes provided."}), 400

        # Validate and match
        match_result = CTX.matcher.match_property(data)
        c_id = match_result["matched_cluster_id"]
        cluster_colors = get_cluster_palette(len(CTX.segment_names))
        color_map = {cid: cluster_colors[i] for i, cid in enumerate(sorted(CTX.segment_names.keys()))}

        match_result["color"] = color_map.get(c_id, "#10B981")
        match_result["cluster_colors_map"] = {str(cid): color_map.get(cid, "#10B981") for cid in CTX.segment_names.keys()}

        return jsonify(sanitize_for_json(match_result))
    except Exception as e:
        return jsonify({"error": f"Invalid property attributes: {str(e)}"}), 400

@app.route("/api/similar/<property_id>")
def api_similar_properties(property_id):
    """
    Finds top 5 nearest neighbors in standardized similarity space.
    STRICTLY EXCLUDES the query property itself.
    """
    try:
        result = CTX.sim_engine.find_similar_properties(property_id, top_n=5)
        cluster_colors = get_cluster_palette(len(CTX.segment_names))
        color_map = {c_id: cluster_colors[i] for i, c_id in enumerate(sorted(CTX.segment_names.keys()))}

        for peer in result["similar_properties"]:
            peer["color"] = color_map.get(peer["cluster"], "#10B981")

        return jsonify(sanitize_for_json(result))
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/download/clustered-data")
def download_clustered_data():
    """Download the complete clustered Indian rental property dataset."""
    path = CTX.export_paths.get("clustered_dataset")
    if path and path.exists():
        return send_file(path, as_attachment=True, download_name="clustered_indian_rentals.csv")
    abort(404, description="Clustered dataset artifact not found.")

@app.route("/download/segment-profiles")
def download_segment_profiles():
    """Download the cross-segment comparison profile matrix."""
    path = CTX.export_paths.get("segment_profiles")
    if path and path.exists():
        return send_file(path, as_attachment=True, download_name="segment_profiles.csv")
    abort(404, description="Segment profiles artifact not found.")


# ==========================================
# ERROR HANDLERS (ATTRACTIVE CUSTOM PAGES)
# ==========================================

@app.errorhandler(404)
def page_not_found(e):
    return render_template("errors/404.html"), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template("errors/500.html", error_message=str(e)), 500


# ==========================================
# APPLICATION ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" Property Intelligence website running at:")
    print(" http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
