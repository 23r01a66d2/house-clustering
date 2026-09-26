"""
PROPERTY INTELLIGENCE & SEGMENT DISCOVERY SYSTEM
Streamlit Analytics Dashboard

Discover natural property communities, explore multidimensional similarity spaces,
profile segment DNA, map geographic concentrations, and match properties using K-Means clustering.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import streamlit as st

# Setup Path
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

# Check Plotly Availability
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    import matplotlib.pyplot as plt
    import seaborn as sns

# ==========================================
# PAGE CONFIGURATION & THEME STYLING
# ==========================================
st.set_page_config(
    page_title="Property Intelligence & Segment Discovery",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Primary Background & Containers */
    .stApp {
        background-color: #0B0F19;
        color: #F1F5F9;
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #131C31 0%, #0F172A 100%);
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 18px 22px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        margin-bottom: 12px;
    }
    .metric-label {
        color: #94A3B8;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-value {
        color: #38BDF8;
        font-size: 1.75rem;
        font-weight: 800;
        line-height: 1.2;
    }
    .metric-sub {
        color: #64748B;
        font-size: 0.78rem;
        margin-top: 4px;
    }

    /* Segment DNA Card */
    .dna-card {
        background: #111B2E;
        border: 1px solid #233554;
        border-radius: 12px;
        padding: 22px;
        margin-bottom: 18px;
    }
    .dna-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 4px;
    }
    .dna-badge {
        display: inline-block;
        background: #1E3A8A;
        color: #93C5FD;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 12px;
    }

    /* Bar indicator monospace */
    .code-bar {
        font-family: 'JetBrains Mono', monospace;
        color: #38BDF8;
        font-weight: 600;
    }

    /* Highlight badge */
    .tag-badge {
        background: #1E293B;
        border: 1px solid #334155;
        color: #CBD5E1;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        margin-right: 6px;
    }

    /* Sidebar Navigation Header */
    .nav-section-title {
        color: #64748B;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 14px;
        margin-bottom: 6px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==========================================
# DATA & PIPELINE CACHING
# ==========================================
@st.cache_data(show_spinner="Loading Property Universe...")
def load_universe_data():
    """Loads raw dataset and computes universe inventory metrics."""
    loader = PropertyDataLoader(config.RAW_DATA_PATH)
    raw_df = loader.load_data()
    universe_metrics = loader.get_universe_metrics()
    return raw_df, universe_metrics

@st.cache_resource(show_spinner="Auditing Property Intelligence & Scaling...")
def prepare_intelligence(raw_df: pd.DataFrame):
    """Executes data trust audit, classifies feature roles, and fits StandardScaler."""
    intelligence = PropertyIntelligence(raw_df)
    usable_df, scaled_df, scaler, trust_meta = intelligence.audit_and_prepare()
    return usable_df, scaled_df, scaler, trust_meta

@st.cache_data(show_spinner="Evaluating K Discovery Lab...")
def run_k_discovery(scaled_df: pd.DataFrame):
    """Evaluates K=2..10 with WCSS and Silhouette Scores."""
    k_lab = KDiscoveryLab(scaled_df)
    recommended_k, results_df, meta = k_lab.run_discovery()
    return recommended_k, results_df, meta

@st.cache_resource(show_spinner="Discovering Property Communities...")
def fit_community_model(scaled_df: pd.DataFrame, usable_df: pd.DataFrame, k: int):
    """Fits K-Means and generates 2D PCA landscape projection."""
    engine = SegmentEngine(k=k)
    clustered_df, kmeans_model, pca_model, pca_df, pca_centroids, conv_info = engine.fit_communities(
        scaled_df=scaled_df,
        usable_df=usable_df
    )
    # Generate dynamic segment names & DNA
    dna_profiler = SegmentDNAProfiler(clustered_df)
    segment_names = dna_profiler.generate_segment_names()
    clustered_df["Segment Name"] = clustered_df["Cluster"].map(segment_names)
    pca_df["Segment Name"] = pca_df["Cluster"].map(segment_names)
    dna_profiles = dna_profiler.extract_segment_dna(segment_names)
    comp_table = dna_profiler.get_comparison_table(segment_names)

    return (
        clustered_df,
        kmeans_model,
        pca_model,
        pca_df,
        pca_centroids,
        conv_info,
        segment_names,
        dna_profiles,
        comp_table,
    )


# ==========================================
# MAIN APPLICATION CONTROLLER
# ==========================================
def main():
    # 1. Load Data & Pipeline
    raw_df, universe_metrics = load_universe_data()
    usable_df, scaled_df, scaler, trust_meta = prepare_intelligence(raw_df)
    recommended_k, k_results_df, k_meta = run_k_discovery(scaled_df)

    # 2. Sidebar Navigation (Custom Selectbox without radio circles)
    st.sidebar.markdown(
        """
        <div style="padding: 10px 0 16px 0;">
            <div style="font-size: 1.2rem; font-weight: 800; color: #38BDF8; letter-spacing: -0.02em;">
                🏙️ PROPERTY INTELLIGENCE
            </div>
            <div style="font-size: 0.78rem; color: #94A3B8; font-weight: 500;">
                Segment Discovery & Similarity System
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    nav_options = [
        "01 — Property Universe",
        "02 — Property Landscape",
        "03 — Segment Discovery",
        "04 — Segment DNA",
        "05 — Spatial Intelligence",
        "06 — Property Match",
        "07 — Similar Property Finder",
        "08 — Method & Data Trust",
        "09 — Insights & Export"
    ]

    selected_page = st.sidebar.selectbox(
        "Select Analytical Module:",
        options=nav_options,
        index=0
    )

    # Sidebar Pipeline Status
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"""
        <div style="background: #0F172A; border: 1px solid #1E293B; border-radius: 8px; padding: 12px;">
            <div style="font-size: 0.75rem; color: #64748B; font-weight: 700; text-transform: uppercase;">
                Pipeline Architecture
            </div>
            <div style="font-size: 0.85rem; color: #F1F5F9; font-weight: 600; margin-top: 4px;">
                Standardized {trust_meta['dimensionality']}D Similarity Space
            </div>
            <div style="font-size: 0.78rem; color: #38BDF8; margin-top: 2px;">
                Official Recommended K: {recommended_k}
            </div>
            <div style="font-size: 0.75rem; color: #94A3B8; margin-top: 2px;">
                Inventory: {trust_meta['usable_properties']:,} Homes
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Sidebar Experimental Exploration K control
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<div style='font-size: 0.78rem; font-weight: 600; color: #CBD5E1;'>Community Exploration (K)</div>",
        unsafe_allow_html=True
    )
    active_k = st.sidebar.slider(
        "Clustering K",
        min_value=2,
        max_value=8,
        value=recommended_k,
        help="Explore property community structures. Default matches Official Recommended K."
    )
    if active_k != recommended_k:
        st.sidebar.caption(f"⚡ Exploring experimental K={active_k} (Recommended: K={recommended_k})")

    # Fit / retrieve community model for active_k
    (
        clustered_df,
        kmeans_model,
        pca_model,
        pca_df,
        pca_centroids,
        conv_info,
        segment_names,
        dna_profiles,
        comp_table,
    ) = fit_community_model(scaled_df, usable_df, active_k)

    # Instantiate Engines
    sig_engine = PropertySignatureEngine(usable_df)
    sim_engine = SimilarityEngine(
        scaled_df=scaled_df,
        original_df=clustered_df,
        scaler=scaler,
        feature_names=trust_meta["segmentation_feature_names"]
    )
    spatial_engine = SpatialIntelligence(clustered_df)
    matcher = PropertyMatcher(
        scaler=scaler,
        kmeans_model=kmeans_model,
        segment_names=segment_names,
        reference_df=usable_df,
        feature_names=trust_meta["segmentation_feature_names"]
    )

    # 3. Route to Selected Page
    if selected_page == "01 — Property Universe":
        render_page_universe(universe_metrics, usable_df)
    elif selected_page == "02 — Property Landscape":
        render_page_landscape(pca_df, pca_centroids, usable_df, trust_meta)
    elif selected_page == "03 — Segment Discovery":
        render_page_discovery(k_results_df, k_meta, conv_info, active_k, recommended_k)
    elif selected_page == "04 — Segment DNA":
        render_page_segment_dna(dna_profiles, comp_table, clustered_df)
    elif selected_page == "05 — Spatial Intelligence":
        render_page_spatial(spatial_engine, segment_names)
    elif selected_page == "06 — Property Match":
        render_page_property_match(matcher, universe_metrics, usable_df)
    elif selected_page == "07 — Similar Property Finder":
        render_page_similar_finder(sim_engine, sig_engine, clustered_df)
    elif selected_page == "08 — Method & Data Trust":
        render_page_method_trust(trust_meta, k_meta, conv_info)
    elif selected_page == "09 — Insights & Export":
        render_page_insights_export(clustered_df, comp_table, dna_profiles, trust_meta)


# ==========================================
# PAGE 1: 01 — PROPERTY UNIVERSE
# ==========================================
def render_page_universe(universe_metrics: Dict[str, Any], usable_df: pd.DataFrame):
    st.markdown(
        """
        <div style="margin-bottom: 24px;">
            <div style="font-size: 0.85rem; color: #38BDF8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;">
                STAGE 1 — PROPERTY UNIVERSE
            </div>
            <h1 style="font-size: 2.2rem; font-weight: 800; color: #F8FAFC; margin-top: 4px; margin-bottom: 8px;">
                Understanding the Property Inventory
            </h1>
            <p style="color: #94A3B8; font-size: 1.05rem; max-width: 900px;">
                Before clustering, we examine the raw multi-attribute housing universe.
                Explore the empirical price, architectural scale, and spatial distribution across the inventory.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Dynamic Hero Stat Row
    p = universe_metrics["price_metrics"]
    a = universe_metrics["area_metrics"]
    y = universe_metrics["year_metrics"]
    c = universe_metrics["coordinate_bounds"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Property Universe</div>
                <div class="metric-value">{universe_metrics['total_properties']:,}</div>
                <div class="metric-sub">Total examined properties</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Price Spectrum</div>
                <div class="metric-value">${p['median']:,.0f}</div>
                <div class="metric-sub">Range: ${p['min']:,.0f} — ${p['max']:,.0f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Living Area Spectrum</div>
                <div class="metric-value">{a['median']:,.0f} sqft</div>
                <div class="metric-sub">Range: {a['min']:,.0f} — {a['max']:,.0f} sqft</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Structural Era</div>
                <div class="metric-value">{y['median']}</div>
                <div class="metric-sub">Built: {y['min']} — {y['max']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### Explore the Property Universe")
    st.markdown("Inspect how properties naturally distribute across economic value and physical living area.")

    col_chart, col_filter = st.columns([3, 1])

    with col_filter:
        st.markdown("#### Sample Filters")
        sample_size = st.slider("Visualization Sample", 500, min(5000, len(usable_df)), 2500, step=500)
        max_p = float(usable_df["Price"].max())
        price_cutoff = st.slider("Price Cap Filter", int(p["min"]), int(max_p), int(min(max_p, 3000000)), step=100000)
        color_by = st.selectbox("Color By", ["number of bedrooms", "grade of the house", "condition of the house", "Built Year"])

    with col_chart:
        sub_df = usable_df[usable_df["Price"] <= price_cutoff].sample(
            n=min(sample_size, len(usable_df[usable_df["Price"] <= price_cutoff])),
            random_state=config.RANDOM_STATE
        )

        if HAS_PLOTLY:
            fig = px.scatter(
                sub_df,
                x="living area",
                y="Price",
                color=color_by,
                color_continuous_scale="Viridis",
                labels={"living area": "Living Area (sqft)", "Price": "Property Price ($)"},
                hover_data=["number of bedrooms", "number of bathrooms", "grade of the house"],
                title=f"Property Inventory: Price vs. Living Area (Colored by {color_by})"
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15, 23, 42, 0.6)",
                font=dict(color="#CBD5E1"),
                xaxis=dict(gridcolor="#1E293B"),
                yaxis=dict(gridcolor="#1E293B"),
                margin=dict(l=40, r=20, t=50, b=40)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.scatterplot(data=sub_df, x="living area", y="Price", hue=color_by, ax=ax, palette="viridis", alpha=0.7)
            ax.set_title(f"Property Inventory: Price vs. Living Area")
            st.pyplot(fig)

    with st.expander("🔍 View Raw Inventory Records Sample"):
        st.dataframe(
            usable_df[["Price", "living area", "number of bedrooms", "number of bathrooms", "grade of the house", "condition of the house", "Built Year"]].head(25),
            use_container_width=True
        )


# ==========================================
# PAGE 2: 02 — PROPERTY LANDSCAPE
# ==========================================
def render_page_landscape(pca_df: pd.DataFrame, pca_centroids: np.ndarray, usable_df: pd.DataFrame, trust_meta: Dict[str, Any]):
    dim = trust_meta["dimensionality"]
    st.markdown(
        f"""
        <div style="margin-bottom: 24px;">
            <div style="font-size: 0.85rem; color: #38BDF8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;">
                STAGE 8 — PROPERTY LANDSCAPE
            </div>
            <h1 style="font-size: 2.2rem; font-weight: 800; color: #F8FAFC; margin-top: 4px; margin-bottom: 8px;">
                2D Projection of the {dim}-Dimensional Property Space
            </h1>
            <p style="color: #94A3B8; font-size: 1.05rem; max-width: 900px;">
                Properties positioned closer together share more similar standardized characteristics across all {dim} dimensions.
                This 2D PCA projection visualizes the continuous property similarity landscape with projected community centers.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.info(
        f"ℹ️ **Methodology Note**: Principal Component Analysis (PCA) is employed strictly for 2D visual projection. "
        f"The actual K-Means segmentation operates on the full {dim}-dimensional standardized space."
    )

    sample_size = min(3500, len(pca_df))
    sub_pca = pca_df.sample(n=sample_size, random_state=config.RANDOM_STATE)

    if HAS_PLOTLY:
        fig = px.scatter(
            sub_pca,
            x="PCA_1",
            y="PCA_2",
            color="Segment Name",
            labels={"PCA_1": "Principal Dimension 1 (Scale & Quality)", "PCA_2": "Principal Dimension 2 (Age & Structural Variance)"},
            hover_data=["Price", "living area", "number of bedrooms", "number of bathrooms"],
            title=f"Property Similarity Landscape ({sample_size:,} Properties Sampled)",
            color_discrete_sequence=["#38BDF8", "#F59E0B", "#10B981", "#8B5CF6", "#EC4899"]
        )

        # Overlay Community Centroids
        for i, c_coord in enumerate(pca_centroids):
            fig.add_trace(
                go.Scatter(
                    x=[c_coord[0]],
                    y=[c_coord[1]],
                    mode="markers+text",
                    marker=dict(symbol="x", size=14, color="#FFFFFF", line=dict(width=2, color="#000000")),
                    text=[f"Community {i} Center"],
                    textposition="top center",
                    name=f"Centroid {i}",
                    showlegend=False
                )
            )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 23, 42, 0.6)",
            font=dict(color="#CBD5E1"),
            xaxis=dict(gridcolor="#1E293B"),
            yaxis=dict(gridcolor="#1E293B"),
            height=600,
            margin=dict(l=40, r=20, t=50, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Market Pattern Deep Dive (Trendlines using pure NumPy)
    st.markdown("### Bivariate Market Patterns")
    col_p1, col_p2 = st.columns(2)

    with col_p1:
        st.markdown("#### Living Area vs. Price Relationship")
        sample_pat = usable_df.sample(n=min(2000, len(usable_df)), random_state=config.RANDOM_STATE)
        
        if HAS_PLOTLY:
            fig_trend = px.scatter(
                sample_pat,
                x="living area",
                y="Price",
                opacity=0.6,
                color_discrete_sequence=["#38BDF8"]
            )
            # Add pure numpy trendline
            x_vals = sample_pat["living area"].values
            y_vals = sample_pat["Price"].values
            poly = np.polyfit(x_vals, y_vals, 1)
            x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
            y_line = poly[0] * x_line + poly[1]
            fig_trend.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", name="Linear Trend", line=dict(color="#F59E0B", width=2)))
            fig_trend.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15, 23, 42, 0.6)",
                font=dict(color="#CBD5E1"),
                xaxis=dict(title="Living Area (sqft)", gridcolor="#1E293B"),
                yaxis=dict(title="Price ($)", gridcolor="#1E293B"),
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig_trend, use_container_width=True)

    with col_p2:
        st.markdown("#### Price Distribution Across Bedroom Capacity")
        if HAS_PLOTLY:
            bed_filtered = usable_df[usable_df["number of bedrooms"].between(1, 6)]
            fig_box = px.box(
                bed_filtered,
                x="number of bedrooms",
                y="Price",
                color="number of bedrooms",
                color_discrete_sequence=px.colors.sequential.Tealgrn
            )
            fig_box.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15, 23, 42, 0.6)",
                font=dict(color="#CBD5E1"),
                xaxis=dict(title="Number of Bedrooms", gridcolor="#1E293B"),
                yaxis=dict(title="Price ($)", gridcolor="#1E293B"),
                showlegend=False,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig_box, use_container_width=True)


# ==========================================
# PAGE 3: 03 — SEGMENT DISCOVERY
# ==========================================
def render_page_discovery(
    k_results_df: pd.DataFrame,
    k_meta: Dict[str, Any],
    conv_info: Dict[str, Any],
    active_k: int,
    recommended_k: int
):
    st.markdown(
        """
        <div style="margin-bottom: 24px;">
            <div style="font-size: 0.85rem; color: #38BDF8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;">
                STAGE 5 & 6 — K DISCOVERY & COMMUNITY FORMATION
            </div>
            <h1 style="font-size: 2.2rem; font-weight: 800; color: #F8FAFC; margin-top: 4px; margin-bottom: 8px;">
                How Many Property Communities Exist?
            </h1>
            <p style="color: #94A3B8; font-size: 1.05rem; max-width: 900px;">
                Evaluating cluster separation and compactness across K = 2 through 10.
                Zero supervised ground-truth labels or classification targets — community selection is determined by mathematical partition boundaries.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Metrics Summary
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Elbow Inflection</div>
                <div class="metric-value">K = {k_meta['elbow_k']}</div>
                <div class="metric-sub">Geometric WCSS trade-off</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Peak Silhouette</div>
                <div class="metric-value">K = {k_meta['silhouette_k']}</div>
                <div class="metric-sub">Score: {k_meta['peak_silhouette_score']:.4f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Official Recommended K</div>
                <div class="metric-value" style="color: #10B981;">K = {recommended_k}</div>
                <div class="metric-sub">Data-driven selection</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Currently Active K</div>
                <div class="metric-value" style="color: {'#10B981' if active_k == recommended_k else '#F59E0B'};">K = {active_k}</div>
                <div class="metric-sub">Controlled via sidebar slider</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        f"""
        <div style="background: #131C31; border-left: 4px solid #38BDF8; padding: 14px 18px; border-radius: 4px; margin-bottom: 20px;">
            <div style="font-weight: 700; color: #F8FAFC; margin-bottom: 4px;">Selection Rationale:</div>
            <div style="color: #CBD5E1; font-size: 0.95rem;">{k_meta['rationale']}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # K Discovery Lab Charts
    st.markdown("### K Discovery Lab: Mathematical Curves")
    col_elbow, col_sil = st.columns(2)

    with col_elbow:
        if HAS_PLOTLY:
            fig_elbow = px.line(
                k_results_df,
                x="K",
                y="Inertia",
                markers=True,
                title="Elbow Method (Within-Cluster Sum of Squares)",
                labels={"Inertia": "WCSS Inertia", "K": "Number of Clusters (K)"}
            )
            # Highlight elbow
            elbow_val = k_results_df.loc[k_results_df["K"] == k_meta["elbow_k"], "Inertia"].values[0]
            fig_elbow.add_trace(go.Scatter(x=[k_meta["elbow_k"]], y=[elbow_val], mode="markers", marker=dict(color="#F59E0B", size=12), name="Elbow Inflection"))
            fig_elbow.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15, 23, 42, 0.6)",
                font=dict(color="#CBD5E1"),
                xaxis=dict(gridcolor="#1E293B", dtick=1),
                yaxis=dict(gridcolor="#1E293B"),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_elbow, use_container_width=True)

    with col_sil:
        if HAS_PLOTLY:
            fig_sil = px.line(
                k_results_df,
                x="K",
                y="Silhouette Score",
                markers=True,
                title="Silhouette Analysis (Boundary Cohesion & Separation)",
                labels={"Silhouette Score": "Silhouette Score", "K": "Number of Clusters (K)"}
            )
            fig_sil.update_traces(line_color="#10B981")
            peak_val = k_results_df.loc[k_results_df["K"] == k_meta["silhouette_k"], "Silhouette Score"].values[0]
            fig_sil.add_trace(go.Scatter(x=[k_meta["silhouette_k"]], y=[peak_val], mode="markers", marker=dict(color="#38BDF8", size=12), name="Peak Silhouette"))
            fig_sil.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15, 23, 42, 0.6)",
                font=dict(color="#CBD5E1"),
                xaxis=dict(gridcolor="#1E293B", dtick=1),
                yaxis=dict(gridcolor="#1E293B"),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_sil, use_container_width=True)

    st.markdown("### How K-Means Discovers Property Communities")
    st.markdown(
        """
        Rather than a black-box supervised prediction model, K-Means is an iterative unsupervised optimization algorithm:
        1. **Initialize Candidate Centers**: Initial candidate community centers are placed using the k-means++ probabilistic heuristic.
        2. **Standardized Association**: Every property is assigned to its nearest community center in the multidimensional standardized similarity space.
        3. **Recalculate Centroids**: Center positions are updated to the mean vector of all properties assigned to that community.
        4. **Convergence Stabilization**: Steps 2 and 3 repeat until property assignments stabilize and centroids cease to move beyond tolerance.
        """
    )

    st.markdown(
        f"""
        <div style="background: #0F172A; border: 1px solid #1E293B; border-radius: 8px; padding: 16px; margin-top: 12px;">
            <div style="font-weight: 700; color: #38BDF8;">Active Model Convergence Metrics (K = {active_k}):</div>
            <div style="color: #94A3B8; font-size: 0.9rem; margin-top: 6px;">
                • Iterations to Converge: <strong style="color: #F8FAFC;">{conv_info['iterations_to_converge']}</strong><br>
                • Final Inertia (WCSS): <strong style="color: #F8FAFC;">{conv_info['inertia']:,.1f}</strong><br>
                • Silhouette Score: <strong style="color: #F8FAFC;">{conv_info['silhouette_score']:.4f}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ==========================================
# PAGE 4: 04 — SEGMENT DNA
# ==========================================
def render_page_segment_dna(
    dna_profiles: List[Dict[str, Any]],
    comp_table: pd.DataFrame,
    clustered_df: pd.DataFrame
):
    st.markdown(
        """
        <div style="margin-bottom: 24px;">
            <div style="font-size: 0.85rem; color: #38BDF8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;">
                STAGE 7 — SEGMENT DNA
            </div>
            <h1 style="font-size: 2.2rem; font-weight: 800; color: #F8FAFC; margin-top: 4px; margin-bottom: 8px;">
                Property Community DNA & Profiles
            </h1>
            <p style="color: #94A3B8; font-size: 1.05rem; max-width: 900px;">
                Every discovered property community possesses a unique architectural and economic fingerprint.
                Normalized relative indicators display each segment's typical standing across the property universe.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption("⚠️ **Note**: Progress bars represent normalized relative positions (0-100% within the dataset), NOT probabilities or confidence scores.")

    # Render Segment DNA Cards
    for profile in dna_profiles:
        st.markdown(
            f"""
            <div class="dna-card">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div>
                        <div class="dna-title">{profile['segment_name']}</div>
                        <span class="dna-badge">Segment {profile['cluster_id']}</span>
                        <span class="tag-badge">Market Share: {profile['share_pct']}% ({profile['property_count']:,} homes)</span>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 1.5rem; font-weight: 800; color: #38BDF8;">${profile['avg_price']:,.0f}</div>
                        <div style="font-size: 0.78rem; color: #64748B;">Average Segment Price</div>
                    </div>
                </div>
                <div style="color: #CBD5E1; font-size: 0.92rem; margin: 12px 0 16px 0; line-height: 1.5;">
                    {profile['narrative']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Show DNA Fingerprint Bars
        col_bars, col_stats = st.columns([3, 2])
        with col_bars:
            st.markdown(f"**Visual DNA Fingerprint — {profile['segment_name']}**")
            for dim in profile["fingerprint_dimensions"]:
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-size: 0.85rem; color: #94A3B8; width: 180px;">{dim['dimension']}</span>
                        <span class="code-bar" style="font-size: 0.9rem;">{dim['bar']} {dim['score_pct']}%</span>
                        <span style="font-size: 0.8rem; color: #64748B; width: 120px; text-align: right;">{dim['formatted_raw']}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with col_stats:
            st.markdown("**Key Segment Characteristics**")
            for trait_name, trait_desc in profile["traits"].items():
                st.markdown(f"• **{trait_name}**: `{trait_desc}`")
            st.markdown(f"• **Typical Living Area**: `{profile['avg_living_area']:,.0f} sqft`")
            st.markdown(f"• **Typical Layout**: `{profile['avg_bedrooms']:.1f} beds, {profile['avg_bathrooms']:.2f} baths`")
            st.markdown(f"• **Typical Build Era**: `{int(profile['avg_built_year'])}`")

        st.markdown("<hr style='border: 1px solid #1E293B; margin: 24px 0;'>", unsafe_allow_html=True)

    # Cross-Segment Comparison Table
    st.markdown("### Cross-Segment Comparative Profile")
    st.dataframe(comp_table, use_container_width=True)


# ==========================================
# PAGE 5: 05 — SPATIAL INTELLIGENCE
# ==========================================
def render_page_spatial(spatial_engine: SpatialIntelligence, segment_names: Dict[int, str]):
    st.markdown(
        """
        <div style="margin-bottom: 24px;">
            <div style="font-size: 0.85rem; color: #38BDF8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;">
                STAGE 9 — SPATIAL INTELLIGENCE
            </div>
            <h1 style="font-size: 2.2rem; font-weight: 800; color: #F8FAFC; margin-top: 4px; margin-bottom: 8px;">
                Geographic Distribution of Property Segments
            </h1>
            <p style="color: #94A3B8; font-size: 1.05rem; max-width: 900px;">
                Explore how discovered property communities cluster geographically.
                Examines empirical coordinate distributions without fabricating unsupported neighborhood claims.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    provenance = spatial_engine.get_provenance_audit()
    if not provenance["valid"]:
        st.warning("Valid geographic coordinates are not available in this dataset.")
        return

    # Document Provenance Explicitly
    b = provenance["bounds"]
    st.info(f"📍 **Geographic Provenance Notice**: {provenance['provenance_statement']}")

    # Map Filter Controls
    col_map_filter, col_map_disp = st.columns([1, 3])

    with col_map_filter:
        st.markdown("#### Spatial Filters")
        selected_clusters = st.multiselect(
            "Filter by Community",
            options=list(segment_names.keys()),
            format_func=lambda x: f"Segment {x}: {segment_names[x]}",
            default=list(segment_names.keys())
        )
        sample_limit = st.slider("Map Render Limit", 500, 5000, 2500, step=500)

    # Filter spatial data
    filtered_map_df, map_stats = spatial_engine.filter_spatial_inventory(
        segment_filter=selected_clusters if len(selected_clusters) > 0 else None,
        sample_limit=sample_limit
    )

    with col_map_disp:
        st.markdown(
            f"**Displaying {map_stats['sample_displayed']:,} Properties (Total Matching: {map_stats['total_matching']:,})**"
        )
        # Render Streamlit Native Map
        st.map(filtered_map_df[["latitude", "longitude"]], zoom=9)

    # Descriptive Spatial Breakdown
    st.markdown("### Spatial Composition Breakdown")
    comp_cols = st.columns(len(map_stats["composition"]) if len(map_stats["composition"]) > 0 else 1)
    for i, (seg_name, seg_data) in enumerate(map_stats["composition"].items()):
        with comp_cols[i % len(comp_cols)]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{seg_name}</div>
                    <div class="metric-value">{seg_data['share_pct']}%</div>
                    <div class="metric-sub">{seg_data['count']:,} properties visible</div>
                </div>
                """,
                unsafe_allow_html=True
            )


# ==========================================
# PAGE 6: 06 — PROPERTY MATCH
# ==========================================
def render_page_property_match(
    matcher: PropertyMatcher,
    universe_metrics: Dict[str, Any],
    usable_df: pd.DataFrame
):
    st.markdown(
        """
        <div style="margin-bottom: 24px;">
            <div style="font-size: 0.85rem; color: #38BDF8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;">
                STAGE 10 — PROPERTY MATCH
            </div>
            <h1 style="font-size: 2.2rem; font-weight: 800; color: #F8FAFC; margin-top: 4px; margin-bottom: 8px;">
                Find My Property Segment
            </h1>
            <p style="color: #94A3B8; font-size: 1.05rem; max-width: 900px;">
                Simulate a property to find its closest market community.
                Attributes are transformed through the <strong>already-fitted StandardScaler</strong> and assigned to the nearest learned cluster centroid.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption("ℹ️ **Unsupervised Matching**: Assignment is determined by standardized Euclidean distance to learned cluster centroids. This is NOT a supervised prediction or confidence probability.")

    p = universe_metrics["price_metrics"]
    a = universe_metrics["area_metrics"]
    y = universe_metrics["year_metrics"]

    col_in1, col_in2, col_in3 = st.columns(3)
    with col_in1:
        st.markdown("#### Economic & Size")
        input_price = st.number_input(
            "Price ($)",
            min_value=int(p["min"]),
            max_value=int(p["max"]),
            value=int(p["median"]),
            step=25000
        )
        input_area = st.number_input(
            "Living Area (sqft)",
            min_value=int(a["min"]),
            max_value=int(a["max"]),
            value=int(a["median"]),
            step=50
        )
        input_lot = st.number_input(
            "Lot Area (sqft)",
            min_value=500,
            max_value=100000,
            value=7500,
            step=500
        )

    with col_in2:
        st.markdown("#### Layout & Quality")
        input_beds = st.number_input("Bedrooms", min_value=1, max_value=10, value=3, step=1)
        input_baths = st.number_input("Bathrooms", min_value=1.0, max_value=8.0, value=2.0, step=0.25)
        input_floors = st.number_input("Floors", min_value=1.0, max_value=4.0, value=1.5, step=0.5)
        input_grade = st.slider("Construction Grade (1–13)", 1, 13, 7)

    with col_in3:
        st.markdown("#### Structural Condition & Age")
        input_cond = st.slider("House Condition (1–5)", 1, 5, 3)
        input_year = st.slider("Built Year", int(y["min"]), int(y["max"]), int(y["median"]))
        input_views = st.slider("Number of Views", 0, 4, 0)
        input_waterfront = st.selectbox("Waterfront Present", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    # Assemble User Property Dictionary
    user_property = {
        "Price": float(input_price),
        "living area": float(input_area),
        "lot area": float(input_lot),
        "number of bedrooms": float(input_beds),
        "number of bathrooms": float(input_baths),
        "number of floors": float(input_floors),
        "grade of the house": float(input_grade),
        "condition of the house": float(input_cond),
        "Built Year": float(input_year),
        "number of views": float(input_views),
        "waterfront present": float(input_waterfront),
    }

    st.markdown("---")

    # Match Property Button
    if st.button("🚀 Match Property with Market Segment", type="primary"):
        match_result = matcher.match_property(user_property)

        # Show Validation Warnings if values are extreme
        if match_result["validation_warnings"]:
            for warn in match_result["validation_warnings"]:
                st.warning(f"⚠️ {warn}")

        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #1E3A8A 0%, #0F172A 100%); border: 1px solid #38BDF8; border-radius: 12px; padding: 24px; margin: 18px 0;">
                <div style="font-size: 0.85rem; color: #93C5FD; font-weight: 700; text-transform: uppercase;">
                    YOUR PROPERTY MATCH
                </div>
                <div style="font-size: 2rem; font-weight: 800; color: #FFFFFF; margin: 6px 0;">
                    {match_result['matched_segment_name']}
                </div>
                <div style="color: #CBD5E1; font-size: 0.95rem;">
                    Proximity to Community Centroid: <strong style="color: #38BDF8;">{match_result['distance_to_centroid']}</strong> (Standardized Euclidean distance in {match_result['dimensionality']}D space)
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Why this match? Comparison Table
        st.markdown("### Why This Match? Centroid Comparison")
        st.markdown("How your simulated property compares against the typical centroid metrics of the matched community:")

        comp_data = []
        for c in match_result["comparison"]:
            comp_data.append({
                "Attribute": c["attribute"],
                "Your Property": f"{c['your_property']:,.1f}",
                "Segment Typical": f"{c['segment_typical']:,.1f}",
                "Delta": f"{c['delta']:+,.1f}",
                "Difference (%)": f"{c['delta_pct']:+.1f}%"
            })
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True)

        with st.expander("📊 Distance to All Discovered Communities"):
            for seg, dist in match_result["all_distances"].items():
                st.write(f"• **{seg}**: Standardized distance = `{dist}`")


# ==========================================
# PAGE 7: 07 — SIMILAR PROPERTY FINDER
# ==========================================
def render_page_similar_finder(
    sim_engine: SimilarityEngine,
    sig_engine: PropertySignatureEngine,
    clustered_df: pd.DataFrame
):
    st.markdown(
        """
        <div style="margin-bottom: 24px;">
            <div style="font-size: 0.85rem; color: #38BDF8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;">
                STAGE 4 — SIMILAR PROPERTY FINDER
            </div>
            <h1 style="font-size: 2.2rem; font-weight: 800; color: #F8FAFC; margin-top: 4px; margin-bottom: 8px;">
                Nearest-Neighbor Similarity Explorer
            </h1>
            <p style="color: #94A3B8; font-size: 1.05rem; max-width: 900px;">
                Select any property to retrieve its closest peers in the standardized feature space.
                Nearest neighbors are identified via standardized Euclidean distance, strictly excluding the query property itself.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col_pick, col_btn = st.columns([3, 1])
    with col_pick:
        # Allow choosing by ID or index
        id_options = list(clustered_df["id"].values[:500]) if "id" in clustered_df.columns else list(clustered_df.index[:500])
        selected_id = st.selectbox("Select Property Identifier:", options=id_options, index=0)

    # Retrieve similar properties
    try:
        similar_result = sim_engine.find_similar_properties(selected_id, top_n=5)
    except Exception as e:
        st.error(f"Error querying similar properties: {e}")
        return

    q = similar_result["query_property"]

    # Display Query Property Details
    st.markdown("### Selected Query Property")
    q1, q2, q3, q4, q5 = st.columns(5)
    with q1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Property ID</div>
                <div class="metric-value" style="font-size: 1.2rem;">#{q['id']}</div>
                <div class="metric-sub">{q['segment_name']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with q2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Price</div>
                <div class="metric-value" style="font-size: 1.2rem;">${q['price']:,.0f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with q3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Living Area</div>
                <div class="metric-value" style="font-size: 1.2rem;">{q['living_area']:,.0f} sqft</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with q4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Bedrooms</div>
                <div class="metric-value" style="font-size: 1.2rem;">{int(q['bedrooms'])} beds</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with q5:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Built Year</div>
                <div class="metric-value" style="font-size: 1.2rem;">{int(q['built_year'])}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Query Property Signature
    sig = sig_engine.generate_signature(q)
    with st.expander("🧬 View Query Property Signature (Normalized Relative Indicators)"):
        st.caption(sig["interpretation_label"])
        for dim in sig["dimensions"]:
            st.markdown(f"• **{dim['dimension']}**: `{dim['bar']} {dim['score_pct']}%` ({dim['formatted_raw']})")

    st.markdown("### Most Similar Properties in Standardized Space")
    st.caption("ℹ️ **Note**: Standardized Euclidean distance measures multi-attribute distance. The queried property is strictly excluded from these results.")

    peers_df = pd.DataFrame(similar_result["similar_properties"])
    st.dataframe(
        peers_df[["id", "euclidean_distance", "similarity_index_pct", "price", "living_area", "bedrooms", "bathrooms", "grade", "built_year", "segment_name"]],
        use_container_width=True
    )


# ==========================================
# PAGE 8: 08 — METHOD & DATA TRUST
# ==========================================
def render_page_method_trust(
    trust_meta: Dict[str, Any],
    k_meta: Dict[str, Any],
    conv_info: Dict[str, Any]
):
    dim = trust_meta["dimensionality"]
    st.markdown(
        f"""
        <div style="margin-bottom: 24px;">
            <div style="font-size: 0.85rem; color: #38BDF8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;">
                STAGE 2 & 8 — METHODOLOGY & DATA TRUST
            </div>
            <h1 style="font-size: 2.2rem; font-weight: 800; color: #F8FAFC; margin-top: 4px; margin-bottom: 8px;">
                Data Trust Panel & Analytical Foundations
            </h1>
            <p style="color: #94A3B8; font-size: 1.05rem; max-width: 900px;">
                Transparency in data preparation, domain-specific feature roles, and mathematical clustering formulations.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Data Trust Metrics
    st.markdown("### Data Trust Audit Panel")
    t1, t2, t3, t4 = st.columns(4)
    with t1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Examined Records</div>
                <div class="metric-value">{trust_meta['properties_examined']:,}</div>
                <div class="metric-sub">Raw dataset inventory</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with t2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Usable Inventory</div>
                <div class="metric-value">{trust_meta['usable_properties']:,}</div>
                <div class="metric-sub">Completeness: {trust_meta['missing_value_completeness_pct']}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with t3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Segmentation Features</div>
                <div class="metric-value">{trust_meta['features_suitable_for_segmentation']}</div>
                <div class="metric-sub">{dim}-dimensional space</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with t4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Excluded Columns</div>
                <div class="metric-value">{trust_meta['features_excluded']}</div>
                <div class="metric-sub">Identifiers, postal code</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Anomaly Audit Log
    if trust_meta["anomaly_log"]:
        st.markdown("#### Documented Anomaly Treatment")
        for anom in trust_meta["anomaly_log"]:
            st.info(f"📋 Record ID **#{anom['id']}**: {anom['reason']}")

    # Feature Roles Categorization
    st.markdown("### Domain Feature Roles")
    st.markdown("Features are categorized according to their real-estate role rather than generic data types:")
    
    role_cols = st.columns(len(trust_meta["feature_roles"]))
    for i, (role_name, col_list) in enumerate(trust_meta["feature_roles"].items()):
        with role_cols[i % len(role_cols)]:
            st.markdown(f"**{role_name}**")
            for c in col_list:
                st.markdown(f"• `{c}`")

    # Mathematical Foundations
    st.markdown("### Mathematical Similarity Foundations")
    st.markdown(
        r"""
        #### 1. Standardization (StandardScaler)
        Properties use vastly different measurement units. Price is measured in hundreds of thousands, while bedroom counts range from 1 to 10.
        Standardization transforms each feature to mean 0 and unit variance:
        $$z = \frac{x - \mu}{\sigma}$$
        This ensures that no single large-magnitude feature artificially dominates the similarity distance.

        #### 2. Multidimensional Similarity Metric
        Distance between properties $u$ and $v$ is computed via standardized Euclidean distance across all $N$ dimensions:
        $$d(u, v) = \sqrt{\sum_{i=1}^{N} (u_i - v_i)^2}$$

        #### 3. Unsupervised Clustering Objective
        K-Means minimizes Within-Cluster Sum of Squares (Inertia) across all partitions:
        $$J = \sum_{j=1}^{K} \sum_{x \in S_j} \|x - \mu_j\|^2$$
        """
    )


# ==========================================
# PAGE 9: 09 — INSIGHTS & EXPORT
# ==========================================
def render_page_insights_export(
    clustered_df: pd.DataFrame,
    comp_table: pd.DataFrame,
    dna_profiles: List[Dict[str, Any]],
    trust_meta: Dict[str, Any]
):
    st.markdown(
        """
        <div style="margin-bottom: 24px;">
            <div style="font-size: 0.85rem; color: #38BDF8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;">
                STAGE 11 — MARKET INSIGHTS & EXPORTS
            </div>
            <h1 style="font-size: 2.2rem; font-weight: 800; color: #F8FAFC; margin-top: 4px; margin-bottom: 8px;">
                Executive Insights & Analytical Deliverables
            </h1>
            <p style="color: #94A3B8; font-size: 1.05rem; max-width: 900px;">
                Synthesized property intelligence and downloadable segmentation datasets.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Core Architecture Diagram
    st.markdown("### Core System Architecture")
    st.code(InsightEngine.get_architecture_diagram(), language="text")

    # Executive Insights
    st.markdown("### Executive Takeaways")
    insights = InsightEngine.generate_executive_insights(clustered_df, dna_profiles, trust_meta)
    for ins in insights:
        st.markdown(f"• {ins}")

    # Export Section
    st.markdown("### Analytical Deliverables (CSV Downloads)")
    col_d1, col_d2 = st.columns(2)

    with col_d1:
        st.markdown("#### Clustered Property Dataset")
        st.write("Complete dataset with attached `Cluster` ID and data-driven `Segment Name`.")
        csv_clustered = clustered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Clustered Dataset (CSV)",
            data=csv_clustered,
            file_name="clustered_house_data.csv",
            mime="text/csv"
        )

    with col_d2:
        st.markdown("#### Segment DNA Profiles")
        st.write("Side-by-side comparative table of all discovered community metrics.")
        csv_profiles = comp_table.to_csv(index=True).encode("utf-8")
        st.download_button(
            label="📥 Download Segment Profiles (CSV)",
            data=csv_profiles,
            file_name="segment_profiles.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()
