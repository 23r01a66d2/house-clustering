"""
PROPERTY SEGMENTATION ANALYTICS
A Modern Real-Estate Market Analytics and Property Segmentation Platform
Powered by K-Means Unsupervised Learning.
"""

import io
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Safe optional Plotly import
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    px = None
    go = None

import config
from src.cluster_analysis import build_cluster_profiles, generate_cluster_interpretations
from src.clustering import HouseClusteringModel
from src.data_loader import DataLoader
from src.eda import (
    compute_descriptive_statistics, 
    plot_bivariate_scatter, 
    plot_correlation_heatmap, 
    plot_feature_distribution
)
from src.optimal_k import (
    determine_optimal_k, 
    evaluate_k_range, 
    plot_elbow_curve, 
    plot_silhouette_curve
)
from src.preprocessing import DataPreprocessor
from src.segment_naming import (
    attach_segment_names, 
    generate_dynamic_segment_names, 
    get_segment_summary_cards
)
from src.utils import detect_column_roles, ensure_directories, format_number
from src.visualization import (
    plot_cluster_distribution, 
    plot_cluster_feature_comparisons, 
    plot_pca_clusters
)
from sklearn.decomposition import PCA

# ---------------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Property Segmentation Analytics | Real-Estate ML",
    page_icon="🏘️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Real-Estate Analytics Styling
st.markdown("""
<style>
    /* Global Container Adjustments */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
    }
    
    /* Real Estate Analytics Branding */
    .app-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.5px;
        margin-bottom: 0.1rem;
    }
    .app-subtitle {
        font-size: 1.05rem;
        color: #475569;
        font-weight: 400;
        margin-bottom: 1.8rem;
    }
    
    /* Metric Card Styling */
    .re-metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        border-top: 4px solid #2563EB;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    .re-metric-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        color: #64748B;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    .re-metric-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #0F172A;
    }
    .re-metric-subtext {
        font-size: 0.8rem;
        color: #10B981;
        margin-top: 0.2rem;
    }

    /* Segment Profile Card */
    .segment-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.4rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.03);
    }
    .segment-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# CACHED DATA PIPELINE FUNCTIONS
# ---------------------------------------------------------
@st.cache_data
def load_raw_data(csv_path_str: str):
    loader = DataLoader(Path(csv_path_str))
    df = loader.load_data()
    summary = loader.get_summary_dict()
    return df, summary

@st.cache_data
def run_preprocessing(raw_df: pd.DataFrame):
    preprocessor = DataPreprocessor(raw_df)
    cleaned_df, scaled_df, meta = preprocessor.preprocess()
    return cleaned_df, scaled_df, meta

@st.cache_data
def compute_k_evaluation(scaled_df: pd.DataFrame):
    results_df = evaluate_k_range(scaled_df)
    recommended_k, decision_info = determine_optimal_k(results_df)
    return results_df, recommended_k, decision_info

@st.cache_data
def run_clustering_cached(scaled_df: pd.DataFrame, cleaned_df: pd.DataFrame, k: int):
    model = HouseClusteringModel(k=k)
    clustered_orig, clustered_scaled, metrics = model.fit_predict(scaled_df, cleaned_df)
    # Attach dynamic segment names
    clustered_orig = attach_segment_names(clustered_orig)
    profile_table, numeric_profiles = build_cluster_profiles(clustered_orig)
    interpretations = generate_cluster_interpretations(clustered_orig, numeric_profiles)
    return clustered_orig, clustered_scaled, model, metrics, profile_table, interpretations

# ---------------------------------------------------------
# MAIN APP ENTRYPOINT
# ---------------------------------------------------------
def main():
    ensure_directories()

    # Load and Preprocess Dataset
    csv_path = config.RAW_DATA_PATH
    if not csv_path.exists():
        st.error(f"⚠️ Dataset file not found at '{csv_path}'. Please ensure 'data/{config.DEFAULT_DATASET_NAME}' is present.")
        return

    try:
        raw_df, raw_summary = load_raw_data(str(csv_path))
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return

    cleaned_df, scaled_df, prep_meta = run_preprocessing(raw_df)
    k_results_df, recommended_k, k_decision = compute_k_evaluation(scaled_df)

    # Sidebar Navigation - Real-Estate Analytics Layout
    with st.sidebar:
        st.markdown("## 🏘️ **Property Analytics**")
        st.caption("Property Segmentation & Market Intelligence")
        st.divider()

        # Dedicated Navigation Menu
        nav_choice = st.radio(
            "Select Section",
            options=[
                "🏠 Market Overview",
                "🏘 Property Explorer",
                "📊 Market Patterns",
                "🧩 Property Segments",
                "⚖ Segment Comparison",
                "📍 Segment Map / Spatial View",
                "⚙ Clustering Analysis",
                "📁 Dataset & Quality",
                "📑 Property Segmentation Summary"
            ],
            index=0
        )

        st.divider()
        # Clear distinction between Official Recommended K and Model K
        st.markdown("### 🎯 **Segmentation Control**")
        st.markdown(f"**Official Optimal K:** `{recommended_k}`")
        
        # Experimental K slider (clearly labeled as exploration)
        exp_k = st.slider(
            "Explore Alternative K:",
            min_value=config.K_MIN,
            max_value=config.K_MAX,
            value=int(recommended_k),
            help="Defaults to the official mathematically selected K. You can adjust this to test other segment granularities."
        )
        if exp_k != recommended_k:
            st.warning(f"Exploring Experimental K = {exp_k} (Official Recommended K is {recommended_k}).")

        st.divider()
        st.caption(f"Dataset: Kaggle House Price India ({len(cleaned_df):,} records)")

    # Run clustering with active K (defaults to official recommended_k)
    active_k = exp_k
    clustered_orig, clustered_scaled, model, cluster_metrics, profile_table, interpretations = run_clustering_cached(
        scaled_df, cleaned_df, active_k
    )

    # Dynamic Column Role Detection
    roles = detect_column_roles(cleaned_df)
    price_col = roles.get("price", "Price")
    area_col = roles.get("living_area", "living area")
    bed_col = roles.get("bedrooms", "number of bedrooms")
    bath_col = roles.get("bathrooms", "number of bathrooms")
    floor_col = roles.get("floors", "number of floors")
    grade_col = roles.get("grade", "grade of the house")
    cond_col = roles.get("condition", "condition of the house")
    built_col = roles.get("built_year", "Built Year")

    # Header Banner
    st.markdown('<div class="app-title">PROPERTY SEGMENTATION ANALYTICS</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Explore hidden patterns and property segments using K-Means clustering</div>', unsafe_allow_html=True)

    # ---------------------------------------------------------
    # PAGE 1: 🏠 MARKET OVERVIEW
    # ---------------------------------------------------------
    if nav_choice == "🏠 Market Overview":
        st.markdown("### 🏠 Property Market Overview")
        st.markdown("Macro-level portfolio snapshot and foundational property metrics derived dynamically from the dataset.")

        # Large KPI Cards (ALL dynamically computed, NEVER hard-coded!)
        kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5, kpi_col6 = st.columns(6)
        with kpi_col1:
            st.markdown(f"""
            <div class="re-metric-card">
                <div class="re-metric-label">Total Properties</div>
                <div class="re-metric-value">{len(cleaned_df):,}</div>
                <div class="re-metric-subtext">Verified Records</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi_col2:
            st.markdown(f"""
            <div class="re-metric-card">
                <div class="re-metric-label">Average Price</div>
                <div class="re-metric-value">${cleaned_df[price_col].mean():,.0f}</div>
                <div class="re-metric-subtext">Portfolio Mean</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi_col3:
            st.markdown(f"""
            <div class="re-metric-card">
                <div class="re-metric-label">Median Price</div>
                <div class="re-metric-value">${cleaned_df[price_col].median():,.0f}</div>
                <div class="re-metric-subtext">50th Percentile</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi_col4:
            st.markdown(f"""
            <div class="re-metric-card">
                <div class="re-metric-label">Avg Living Area</div>
                <div class="re-metric-value">{cleaned_df[area_col].mean():,.0f} <span style="font-size:1rem;">sqft</span></div>
                <div class="re-metric-subtext">Median: {cleaned_df[area_col].median():,.0f} sqft</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi_col5:
            st.markdown(f"""
            <div class="re-metric-card">
                <div class="re-metric-label">Avg Bedrooms</div>
                <div class="re-metric-value">{cleaned_df[bed_col].mean():.1f}</div>
                <div class="re-metric-subtext">Typical: {int(cleaned_df[bed_col].median())} Beds</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi_col6:
            st.markdown(f"""
            <div class="re-metric-card">
                <div class="re-metric-label">Property Segments</div>
                <div class="re-metric-value">{active_k}</div>
                <div class="re-metric-subtext">Discovered Groups</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Dual Distributions
        col_dist1, col_dist2 = st.columns(2)
        with col_dist1:
            st.markdown("#### Property Price Distribution")
            if HAS_PLOTLY:
                fig_price = px.histogram(
                    cleaned_df,
                    x=price_col,
                    nbins=45,
                    marginal="box",
                    title="Property Price Distribution ($ USD)",
                    color_discrete_sequence=["#1E3A8A"]
                )
                fig_price.update_layout(xaxis_title="Property Price ($)", yaxis_title="Number of Properties", height=380)
                st.plotly_chart(fig_price, use_container_width=True)
            else:
                fig_p = plot_feature_distribution(cleaned_df, price_col)
                st.pyplot(fig_p)

        with col_dist2:
            st.markdown("#### Living Area Distribution")
            if HAS_PLOTLY:
                fig_area = px.histogram(
                    cleaned_df,
                    x=area_col,
                    nbins=45,
                    marginal="box",
                    title="Living Area Distribution (Square Feet)",
                    color_discrete_sequence=["#059669"]
                )
                fig_area.update_layout(xaxis_title="Living Area (sqft)", yaxis_title="Number of Properties", height=380)
                st.plotly_chart(fig_area, use_container_width=True)
            else:
                fig_a = plot_feature_distribution(cleaned_df, area_col)
                st.pyplot(fig_a)

        # Market Snapshot (Purely Data-Driven, Dynamic)
        st.markdown("### 📌 Market Snapshot & Characteristics")
        snap_col1, snap_col2 = st.columns(2)
        
        mode_bed = cleaned_df[bed_col].mode()[0]
        mode_bath = cleaned_df[bath_col].mode()[0]
        min_p = cleaned_df[price_col].min()
        max_p = cleaned_df[price_col].max()
        largest_row = cleaned_df.loc[cleaned_df[area_col].idxmax()]
        smallest_row = cleaned_df.loc[cleaned_df[area_col].idxmin()]

        with snap_col1:
            st.markdown(f"""
            - **Dominant Layout:** Most frequent configuration is **{mode_bed:.0f} bedrooms** and **{mode_bath:.1f} bathrooms**.
            - **Valuation Spread:** Market prices span from **${min_p:,.0f}** to **${max_p:,.0f}**.
            - **Interquartile Price Range:** Middle 50% of homes trade between **${cleaned_df[price_col].quantile(0.25):,.0f}** and **${cleaned_df[price_col].quantile(0.75):,.0f}**.
            """)
        with snap_col2:
            st.markdown(f"""
            - **Largest Property:** **{largest_row[area_col]:,.0f} sqft** with {largest_row[bed_col]} bedrooms (priced at ${largest_row[price_col]:,.0f}).
            - **Smallest Property:** **{smallest_row[area_col]:,.0f} sqft** with {smallest_row[bed_col]} bedrooms (priced at ${smallest_row[price_col]:,.0f}).
            - **Construction Era:** Built years range from **{int(cleaned_df[built_col].min())}** to **{int(cleaned_df[built_col].max())}** (median build year: {int(cleaned_df[built_col].median())}).
            """)

    # ---------------------------------------------------------
    # PAGE 2: 🏘 PROPERTY EXPLORER
    # ---------------------------------------------------------
    elif nav_choice == "🏘 Property Explorer":
        st.markdown("### 🏘 Property Explorer")
        st.markdown("Interactive multi-criteria search to explore individual properties and inspect their assigned market segments.")

        # Interactive Filtering Controls
        with st.expander("🔍 Filter Controls", expanded=True):
            f_col1, f_col2, f_col3 = st.columns(3)
            with f_col1:
                price_min_val = int(cleaned_df[price_col].min())
                price_max_val = int(cleaned_df[price_col].max())
                price_range = st.slider(
                    "Price Range ($)",
                    min_value=price_min_val,
                    max_value=price_max_val,
                    value=(price_min_val, price_max_val),
                    step=10000
                )
            with f_col2:
                area_min_val = int(cleaned_df[area_col].min())
                area_max_val = int(cleaned_df[area_col].max())
                area_range = st.slider(
                    "Living Area Range (sqft)",
                    min_value=area_min_val,
                    max_value=area_max_val,
                    value=(area_min_val, area_max_val),
                    step=100
                )
            with f_col3:
                all_segments = ["All Segments"] + sorted(clustered_orig["Segment Label"].unique().tolist())
                selected_seg = st.selectbox("Market Segment", options=all_segments, index=0)

            f_col4, f_col5, f_col6 = st.columns(3)
            with f_col4:
                bed_options = sorted([int(b) for b in cleaned_df[bed_col].unique()])
                sel_beds = st.multiselect("Bedrooms", options=bed_options, default=bed_options)
            with f_col5:
                cond_options = sorted([int(c) for c in cleaned_df[cond_col].unique()]) if cond_col in cleaned_df else []
                sel_cond = st.multiselect("Condition (1 to 5)", options=cond_options, default=cond_options)
            with f_col6:
                baths_min = float(cleaned_df[bath_col].min())
                baths_max = float(cleaned_df[bath_col].max())
                baths_range = st.slider("Bathrooms", min_value=baths_min, max_value=baths_max, value=(baths_min, baths_max), step=0.25)

        # Apply Filters Dynamically
        filtered_df = clustered_orig[
            (clustered_orig[price_col] >= price_range[0]) & 
            (clustered_orig[price_col] <= price_range[1]) &
            (clustered_orig[area_col] >= area_range[0]) & 
            (clustered_orig[area_col] <= area_range[1]) &
            (clustered_orig[bath_col] >= baths_range[0]) & 
            (clustered_orig[bath_col] <= baths_range[1])
        ]
        if sel_beds:
            filtered_df = filtered_df[filtered_df[bed_col].isin(sel_beds)]
        if cond_col in filtered_df and sel_cond:
            filtered_df = filtered_df[filtered_df[cond_col].isin(sel_cond)]
        if selected_seg != "All Segments":
            filtered_df = filtered_df[filtered_df["Segment Label"] == selected_seg]

        st.markdown(f"**Displaying {len(filtered_df):,} of {len(cleaned_df):,} properties** ({(len(filtered_df)/len(cleaned_df))*100:.1f}% of market portfolio)")

        # Display Clean Table with Key Attributes
        display_cols = [
            col for col in [
                "id", price_col, area_col, bed_col, bath_col, floor_col, grade_col, cond_col, built_col, "Segment Label"
            ] if col in filtered_df.columns
        ]
        st.dataframe(filtered_df[display_cols].head(100), use_container_width=True, height=350)

        # Individual Property Inspector
        st.markdown("---")
        st.markdown("#### 🔎 Individual Property Profile Inspector")
        if not filtered_df.empty:
            sel_idx = st.selectbox(
                "Select a property record to inspect detailed specifications:",
                options=filtered_df.index[:200].tolist(),
                format_func=lambda idx: f"Property Index {idx} — ${filtered_df.loc[idx, price_col]:,.0f} | {filtered_df.loc[idx, area_col]:,.0f} sqft | {filtered_df.loc[idx, bed_col]} Beds | {filtered_df.loc[idx, 'Segment Label']}"
            )
            prop = filtered_df.loc[sel_idx]

            p_col1, p_col2 = st.columns([1, 2])
            with p_col1:
                st.markdown(f"""
                <div class="segment-card" style="border-left: 5px solid #2563EB;">
                    <div style="font-size:0.85rem; font-weight:700; color:#64748B;">ASSIGNED PROPERTY SEGMENT</div>
                    <div style="font-size:1.35rem; font-weight:800; color:#1E3A8A; margin: 0.4rem 0;">{prop['Segment Label']}</div>
                    <hr style="margin: 0.5rem 0;">
                    <div style="font-size:0.9rem; color:#334155;"><b>Property ID:</b> {prop.get('id', sel_idx)}</div>
                    <div style="font-size:0.9rem; color:#334155;"><b>Valuation:</b> ${prop[price_col]:,.0f}</div>
                    <div style="font-size:0.9rem; color:#334155;"><b>Living Area:</b> {prop[area_col]:,.0f} sqft</div>
                    <div style="font-size:0.9rem; color:#334155;"><b>Bedrooms / Baths:</b> {prop[bed_col]} beds / {prop[bath_col]} baths</div>
                </div>
                """, unsafe_allow_html=True)
            with p_col2:
                # Comparison against assigned segment averages
                seg_sub = clustered_orig[clustered_orig["Cluster"] == prop["Cluster"]]
                avg_seg_p = seg_sub[price_col].mean()
                avg_seg_a = seg_sub[area_col].mean()

                p_diff = ((prop[price_col] - avg_seg_p) / avg_seg_p) * 100
                a_diff = ((prop[area_col] - avg_seg_a) / avg_seg_a) * 100

                st.markdown(f"""
                ##### Segment Context Analysis:
                - **Price vs Segment Average:** ${prop[price_col]:,.0f} ({p_diff:+.1f}% relative to segment average ${avg_seg_p:,.0f}).
                - **Area vs Segment Average:** {prop[area_col]:,.0f} sqft ({a_diff:+.1f}% relative to segment average {avg_seg_a:,.0f} sqft).
                - **Structural Build:** Condition rating **{prop.get(cond_col, 'N/A')}/5**, Construction Grade **{prop.get(grade_col, 'N/A')}/13**.
                - **Construction Year:** Built in **{int(prop.get(built_col, 0))}**.
                """)
        else:
            st.warning("No properties match the chosen filter criteria. Please broaden your search ranges.")

    # ---------------------------------------------------------
    # PAGE 3: 📊 MARKET PATTERNS
    # ---------------------------------------------------------
    elif nav_choice == "📊 Market Patterns":
        st.markdown("### 📊 Market Patterns & Correlations")
        st.markdown("Exploratory visual analytics examining structural and economic relationships across the property market before clustering.")

        mp_tab1, mp_tab2, mp_tab3 = st.tabs(["📈 Bivariate Relationships", "📦 Price by Layout", "🔥 Correlation Heatmap"])

        with mp_tab1:
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                st.markdown("#### Price vs Living Area")
                if HAS_PLOTLY:
                    sample_eda = cleaned_df.sample(n=min(3500, len(cleaned_df)), random_state=config.RANDOM_STATE)
                    fig_pa = px.scatter(
                        sample_eda,
                        x=area_col,
                        y=price_col,
                        opacity=0.45,
                        title=f"Price vs Living Area ({len(sample_eda):,} Sampled Homes)",
                        color_discrete_sequence=["#1D4ED8"]
                    )
                    # Add pure numpy trendline without statsmodels
                    x_v = sample_eda[area_col].values.astype(float)
                    y_v = sample_eda[price_col].values.astype(float)
                    sl, inc = np.polyfit(x_v, y_v, 1)
                    x_line = np.linspace(x_v.min(), x_v.max(), 50)
                    fig_pa.add_trace(go.Scatter(x=x_line, y=sl*x_line+inc, mode="lines", line=dict(color="red", width=2.5), name="Trendline"))
                    fig_pa.update_layout(xaxis_title="Living Area (sqft)", yaxis_title="Price ($)", height=450)
                    st.plotly_chart(fig_pa, use_container_width=True)
                else:
                    st.pyplot(plot_bivariate_scatter(cleaned_df, area_col, price_col))

            with col_b2:
                st.markdown("#### Area vs Bedrooms")
                if HAS_PLOTLY:
                    fig_ab = px.scatter(
                        sample_eda,
                        x=bed_col,
                        y=area_col,
                        opacity=0.45,
                        title="Living Area vs Bedroom Count",
                        color_discrete_sequence=["#0D9488"]
                    )
                    x_b = sample_eda[bed_col].values.astype(float)
                    y_a = sample_eda[area_col].values.astype(float)
                    sl2, inc2 = np.polyfit(x_b, y_a, 1)
                    x_line2 = np.linspace(x_b.min(), x_b.max(), 50)
                    fig_ab.add_trace(go.Scatter(x=x_line2, y=sl2*x_line2+inc2, mode="lines", line=dict(color="darkorange", width=2.5), name="Trendline"))
                    fig_ab.update_layout(xaxis_title="Bedrooms", yaxis_title="Living Area (sqft)", height=450)
                    st.plotly_chart(fig_ab, use_container_width=True)
                else:
                    st.pyplot(plot_bivariate_scatter(cleaned_df, bed_col, area_col))

        with mp_tab2:
            st.markdown("#### Property Price Distribution Grouped by Bedroom Count")
            if HAS_PLOTLY:
                fig_bed_box = px.box(
                    cleaned_df[cleaned_df[bed_col] <= 7],
                    x=bed_col,
                    y=price_col,
                    color=bed_col,
                    title="Price Distribution Across Bedroom Configurations (Up to 7 Bedrooms)",
                    color_discrete_sequence=px.colors.qualitative.Prism
                )
                fig_bed_box.update_layout(xaxis_title="Number of Bedrooms", yaxis_title="Price ($)", height=480, showlegend=False)
                st.plotly_chart(fig_bed_box, use_container_width=True)
            else:
                fig_box, ax = plt.subplots(figsize=(10, 5))
                sns.boxplot(data=cleaned_df[cleaned_df[bed_col] <= 7], x=bed_col, y=price_col, ax=ax, palette="viridis")
                ax.set_title("Price Distribution by Bedroom Count")
                st.pyplot(fig_box)

        with mp_tab3:
            st.markdown("#### Feature Correlation Heatmap")
            st.caption("Pearson correlation between structural dimensions, quality metrics, and market price.")
            if HAS_PLOTLY:
                corr_df = cleaned_df[prep_meta["selected_features"]].corr()
                fig_corr = px.imshow(
                    corr_df,
                    text_auto=".2f",
                    aspect="auto",
                    color_continuous_scale="RdBu_r",
                    zmin=-1,
                    zmax=1,
                    title="Correlation Matrix of Selected Property Attributes"
                )
                fig_corr.update_layout(height=650)
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.pyplot(plot_correlation_heatmap(cleaned_df, prep_meta["selected_features"]))

    # ---------------------------------------------------------
    # PAGE 4: 🧩 PROPERTY SEGMENTS
    # ---------------------------------------------------------
    elif nav_choice == "🧩 Property Segments":
        st.markdown("### 🧩 Discovered Property Segments")
        st.markdown(f"Unsupervised clustering grouped **{len(clustered_orig):,} properties** into **{active_k} distinct market segments**.")

        # Segment Profile Cards
        st.markdown("#### Discovered Segment Profiles")
        cards = get_segment_summary_cards(clustered_orig)
        card_cols = st.columns(len(cards))
        
        for i, card in enumerate(cards):
            with card_cols[i]:
                st.markdown(f"""
                <div class="segment-card" style="border-top: 4px solid #2563EB;">
                    <div class="segment-badge" style="background:#DBEAFE; color:#1E40AF;">Segment {card['cluster_id']}</div>
                    <div style="font-size:1.15rem; font-weight:800; color:#0F172A; min-height: 48px;">{card['segment_name']}</div>
                    <div style="font-size:0.85rem; color:#64748B; margin-bottom:0.8rem;"><b>{card['count']:,} Properties</b> ({card['percentage']:.1f}% share)</div>
                    <hr style="margin: 0.5rem 0;">
                    <div style="font-size:0.88rem; color:#334155; margin-bottom: 0.3rem;"><b>Avg Price:</b> ${card['avg_price']:,.0f}</div>
                    <div style="font-size:0.88rem; color:#334155; margin-bottom: 0.3rem;"><b>Avg Living Area:</b> {card['avg_area']:,.0f} sqft</div>
                    <div style="font-size:0.88rem; color:#334155; margin-bottom: 0.3rem;"><b>Avg Layout:</b> {card['avg_beds']:.1f} beds / {card['avg_baths']:.1f} baths</div>
                    <div style="font-size:0.88rem; color:#334155;"><b>Avg Grade:</b> {card['avg_grade']:.1f} / 13</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 2D PCA Cluster Scatter Visualization
        st.markdown("#### High-Dimensional Feature Space Visualization (2D PCA)")
        st.info("💡 **How to interpret:** Each point represents a single property in the dataset. Properties located closer together have more similar characteristics across all scaled features (price, living area, bedrooms, bathrooms, grade, condition, etc.). Centroids represent the mathematical average center of each segment.")

        # Compute PCA projection for visualization
        X_scaled = scaled_df[prep_meta["selected_features"]].values
        pca = PCA(n_components=2, random_state=config.RANDOM_STATE)
        X_pca = pca.fit_transform(X_scaled)
        centroids_pca = pca.transform(model.cluster_centers_)
        var_pca = pca.explained_variance_ratio_ * 100

        pca_plot_df = pd.DataFrame({
            "PC1": X_pca[:, 0],
            "PC2": X_pca[:, 1],
            "Segment": clustered_orig["Segment Label"]
        })
        sample_pca = pca_plot_df.sample(n=min(5000, len(pca_plot_df)), random_state=config.RANDOM_STATE)

        if HAS_PLOTLY:
            fig_pca = px.scatter(
                sample_pca,
                x="PC1",
                y="PC2",
                color="Segment",
                opacity=0.45,
                title=f"Property Segments in 2D PCA Space (PC1: {var_pca[0]:.1f}%, PC2: {var_pca[1]:.1f}% Variance Explained)",
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            # Add Centroid Marks
            for c_idx, pt in enumerate(centroids_pca):
                fig_pca.add_trace(go.Scatter(
                    x=[pt[0]],
                    y=[pt[1]],
                    mode="markers+text",
                    marker=dict(symbol="x", size=15, color="black", line=dict(width=2, color="white")),
                    text=[f"C{c_idx}"],
                    textposition="top center",
                    name=f"Centroid {c_idx}",
                    showlegend=False
                ))
            fig_pca.update_layout(height=580, legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5))
            st.plotly_chart(fig_pca, use_container_width=True)
        else:
            fig_pca_mpl, _ = plot_pca_clusters(scaled_df, clustered_orig["Cluster"].values, centroids=model.cluster_centers_)
            st.pyplot(fig_pca_mpl)

        # Market Share Breakdown
        st.markdown("#### Market Share Distribution Across Discovered Segments")
        counts_df = pd.DataFrame([
            {"Segment": c['display_title'], "Properties": c['count'], "Share (%)": c['percentage']}
            for c in cards
        ])
        if HAS_PLOTLY:
            fig_donut = px.pie(
                counts_df,
                names="Segment",
                values="Properties",
                hole=0.45,
                title="Property Segment Market Share Breakdown",
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig_donut.update_traces(textinfo="percent+label")
            st.plotly_chart(fig_donut, use_container_width=True)
        else:
            st.dataframe(counts_df, use_container_width=True, hide_index=True)

    # ---------------------------------------------------------
    # PAGE 5: ⚖ SEGMENT COMPARISON
    # ---------------------------------------------------------
    elif nav_choice == "⚖ Segment Comparison":
        st.markdown("### ⚖ Segment Comparison Analytics")
        st.markdown("Side-by-side comparison of core physical and valuation dimensions across all discovered property segments.")

        # Comparative Bar Charts
        c_bar1, c_bar2 = st.columns(2)
        cards = get_segment_summary_cards(clustered_orig)
        seg_labels = [c["display_title"] for c in cards]

        with c_bar1:
            st.markdown("#### Average Property Price by Segment")
            if HAS_PLOTLY:
                fig_c1 = px.bar(
                    x=seg_labels,
                    y=[c["avg_price"] for c in cards],
                    labels={"x": "Property Segment", "y": "Average Price ($)"},
                    title="Average Price Comparison ($ USD)",
                    color=seg_labels,
                    color_discrete_sequence=px.colors.qualitative.Safe
                )
                fig_c1.update_layout(showlegend=False, height=360)
                st.plotly_chart(fig_c1, use_container_width=True)
            else:
                st.pyplot(plot_cluster_feature_comparisons(clustered_orig))

        with c_bar2:
            st.markdown("#### Average Living Area by Segment")
            if HAS_PLOTLY:
                fig_c2 = px.bar(
                    x=seg_labels,
                    y=[c["avg_area"] for c in cards],
                    labels={"x": "Property Segment", "y": "Average Living Area (sqft)"},
                    title="Average Living Area Comparison (Square Feet)",
                    color=seg_labels,
                    color_discrete_sequence=px.colors.qualitative.Safe
                )
                fig_c2.update_layout(showlegend=False, height=360)
                st.plotly_chart(fig_c2, use_container_width=True)

        c_bar3, c_bar4 = st.columns(2)
        with c_bar3:
            st.markdown("#### Average Bedrooms by Segment")
            if HAS_PLOTLY:
                fig_c3 = px.bar(
                    x=seg_labels,
                    y=[c["avg_beds"] for c in cards],
                    labels={"x": "Property Segment", "y": "Average Bedrooms"},
                    title="Average Bedroom Count Comparison",
                    color=seg_labels,
                    color_discrete_sequence=px.colors.qualitative.Safe
                )
                fig_c3.update_layout(showlegend=False, height=360)
                st.plotly_chart(fig_c3, use_container_width=True)

        with c_bar4:
            st.markdown("#### Average Bathrooms by Segment")
            if HAS_PLOTLY:
                fig_c4 = px.bar(
                    x=seg_labels,
                    y=[c["avg_baths"] for c in cards],
                    labels={"x": "Property Segment", "y": "Average Bathrooms"},
                    title="Average Bathroom Count Comparison",
                    color=seg_labels,
                    color_discrete_sequence=px.colors.qualitative.Safe
                )
                fig_c4.update_layout(showlegend=False, height=360)
                st.plotly_chart(fig_c4, use_container_width=True)

        # Cross-Segment Metric Comparison Table
        st.markdown("#### Comprehensive Cross-Segment Comparison Table")
        st.dataframe(profile_table, use_container_width=True)

        # Dynamic Narrative Interpretations
        st.markdown("#### Data-Driven Segment Interpretations")
        for c_id, text in interpretations.items():
            with st.expander(f"📌 Segment {c_id}: Detailed Characterization", expanded=True):
                st.markdown(text)

    # ---------------------------------------------------------
    # PAGE 6: 📍 SEGMENT MAP / SPATIAL VIEW
    # ---------------------------------------------------------
    elif nav_choice == "📍 Segment Map / Spatial View":
        st.markdown("### 📍 Geographic & Spatial Segment Distribution")
        st.markdown("Interactive spatial visualization of properties based on coordinates (`Lattitude` and `Longitude`) in the Kaggle dataset.")

        lat_col = "Lattitude" if "Lattitude" in cleaned_df.columns else "latitude"
        lon_col = "Longitude" if "Longitude" in cleaned_df.columns else "longitude"

        has_coords = (
            lat_col in cleaned_df.columns and 
            lon_col in cleaned_df.columns and 
            cleaned_df[lat_col].notnull().any() and 
            cleaned_df[lon_col].notnull().any()
        )

        if has_coords:
            # Map filtering controls
            m_col1, m_col2 = st.columns([1, 3])
            with m_col1:
                st.markdown("##### Spatial Filters")
                map_segs = ["All Segments"] + sorted(clustered_orig["Segment Label"].unique().tolist())
                sel_map_seg = st.selectbox("Filter by Segment:", options=map_segs, index=0)
                map_sample_size = st.slider("Map Sample Size", min_value=500, max_value=min(8000, len(clustered_orig)), value=2500, step=500)

            map_filtered = clustered_orig if sel_map_seg == "All Segments" else clustered_orig[clustered_orig["Segment Label"] == sel_map_seg]
            sample_map_df = map_filtered.sample(n=min(map_sample_size, len(map_filtered)), random_state=config.RANDOM_STATE)

            with m_col2:
                # Standardize lat/lon for Streamlit
                sample_map_df_std = sample_map_df.rename(columns={lat_col: "latitude", lon_col: "longitude"})
                st.map(sample_map_df_std[["latitude", "longitude"]], zoom=8)

            # Coordinate Density Plot by Segment
            st.markdown("#### Spatial Coordinate Distribution (Latitude vs Longitude)")
            if HAS_PLOTLY:
                fig_geo = px.scatter(
                    sample_map_df,
                    x=lon_col,
                    y=lat_col,
                    color="Segment Label",
                    opacity=0.45,
                    title="Properties Grouped by Segment Across Geographic Coordinates",
                    color_discrete_sequence=px.colors.qualitative.Dark24
                )
                fig_geo.update_layout(xaxis_title="Longitude", yaxis_title="Latitude", height=500)
                st.plotly_chart(fig_geo, use_container_width=True)
        else:
            st.warning("No geographic coordinates found in this dataset. Displaying property distribution across distance metrics instead.")
            dist_col = roles.get("airport_dist", "Distance from the airport")
            if dist_col in clustered_orig:
                fig_dist = px.box(clustered_orig, x="Segment Label", y=dist_col, color="Segment Label")
                st.plotly_chart(fig_dist, use_container_width=True)

    # ---------------------------------------------------------
    # PAGE 7: ⚙ CLUSTERING ANALYSIS (TECHNICAL)
    # ---------------------------------------------------------
    elif nav_choice == "⚙ Clustering Analysis":
        st.markdown("### ⚙️ Clustering Methodology & Technical Validation")
        st.markdown("Analytical verification of optimal K selection, Within-Cluster Sum of Squares (Inertia), and Silhouette scores.")

        # Model Status Summary
        st.markdown(f"""
        - **Algorithm:** Unsupervised K-Means (`sklearn.cluster.KMeans`)
        - **Official Recommended K:** `{recommended_k}` (based on Silhouette peak of `{k_decision['max_silhouette_score']:.4f}` and Elbow inflection)
        - **Currently Evaluated K:** `{active_k}`
        - **Convergence Status:** Reached in `{cluster_metrics['n_iterations']}` iterations
        - **Final Inertia (WCSS):** `{cluster_metrics['inertia']:,.1f}`
        - **Final Silhouette Score:** `{cluster_metrics['silhouette_score']:.4f}`
        """)

        # Elbow and Silhouette Curves
        k_col1, k_col2 = st.columns(2)
        with k_col1:
            st.markdown("#### Elbow Method: K vs Inertia")
            if HAS_PLOTLY:
                fig_elb = go.Figure()
                fig_elb.add_trace(go.Scatter(
                    x=k_results_df["k"],
                    y=k_results_df["inertia"],
                    mode="lines+markers",
                    name="Inertia",
                    line=dict(color="#2563EB", width=3),
                    marker=dict(size=8)
                ))
                fig_elb.add_vline(x=recommended_k, line_dash="dash", line_color="green", annotation_text=f"Official K={recommended_k}")
                fig_elb.update_layout(xaxis_title="Clusters (K)", yaxis_title="Inertia (WCSS)", height=380)
                st.plotly_chart(fig_elb, use_container_width=True)
            else:
                st.pyplot(plot_elbow_curve(k_results_df, selected_k=recommended_k))

        with k_col2:
            st.markdown("#### Silhouette Analysis: K vs Silhouette Score")
            if HAS_PLOTLY:
                fig_s = go.Figure()
                fig_s.add_trace(go.Scatter(
                    x=k_results_df["k"],
                    y=k_results_df["silhouette_score"],
                    mode="lines+markers",
                    name="Silhouette Score",
                    line=dict(color="#059669", width=3),
                    marker=dict(size=8)
                ))
                fig_s.add_vline(x=recommended_k, line_dash="dash", line_color="green", annotation_text=f"Official K={recommended_k}")
                fig_s.update_layout(xaxis_title="Clusters (K)", yaxis_title="Silhouette Score", height=380)
                st.plotly_chart(fig_s, use_container_width=True)
            else:
                st.pyplot(plot_silhouette_curve(k_results_df, selected_k=recommended_k))

        # Tested K Values Table
        st.markdown("#### Evaluated Candidate K Values (K = 2 to 10)")
        st.dataframe(
            k_results_df.style.format({"inertia": "{:,.1f}", "silhouette_score": "{:.4f}"}),
            use_container_width=True,
            hide_index=True
        )

        st.info(f"💡 **Automated Decision Rationale:** {k_decision['rationale']}")

        # Features Used
        st.markdown("---")
        st.markdown("#### Selected Clustering Features vs Excluded Identifiers")
        fc1, fc2 = st.columns(2)
        with fc1:
            st.success(f"**Included Property Features ({len(prep_meta['selected_features'])}):**")
            st.write(prep_meta["selected_features"])
            st.caption("Standardized with `StandardScaler` to ensure scale-invariance across units.")
        with fc2:
            st.info(f"**Excluded Non-Physical Identifiers ({len(prep_meta['removed_features'])}):**")
            st.write(prep_meta["removed_features"])
            st.caption("Excluded to prevent arbitrary database indexes or dates from distorting property similarity.")

    # ---------------------------------------------------------
    # PAGE 8: 📁 DATASET & QUALITY
    # ---------------------------------------------------------
    elif nav_choice == "📁 Dataset & Quality":
        st.markdown("### 📁 Dataset Inspection & Data Quality Audit")
        st.markdown("Verification of Kaggle House Price Dataset of India schema, completeness, and preprocessing.")

        d_col1, d_col2, d_col3, d_col4 = st.columns(4)
        with d_col1:
            st.metric("Raw Records", f"{raw_summary['rows']:,}")
        with d_col2:
            st.metric("Cleaned Records", f"{prep_meta['records_after']:,}")
        with d_col3:
            st.metric("Missing Values", f"{prep_meta['missing_after']}")
        with d_col4:
            st.metric("Duplicates Removed", f"{prep_meta['duplicates_removed']}")

        st.markdown("#### Raw Dataset Preview (First 15 Rows)")
        st.dataframe(raw_df.head(15), use_container_width=True)

        st.markdown("#### Schema & Data Types Breakdown")
        schema_df = pd.DataFrame({
            "Column Name": list(raw_summary["dtypes"].keys()),
            "Data Type": list(raw_summary["dtypes"].values()),
            "Missing Values": [raw_summary["missing_values"][c] for c in raw_summary["dtypes"].keys()],
            "Role in Pipeline": ["Included Feature" if c in prep_meta["selected_features"] else "Excluded Identifier" for c in raw_summary["dtypes"].keys()]
        })
        st.dataframe(schema_df, use_container_width=True, height=400, hide_index=True)

    # ---------------------------------------------------------
    # PAGE 9: 📑 PROPERTY SEGMENTATION SUMMARY
    # ---------------------------------------------------------
    elif nav_choice == "📑 Property Segmentation Summary":
        st.markdown("### 📑 Property Segmentation Summary & Export")
        st.markdown("Final executive overview of the market segmentation project and downloadable deliverables.")

        st.markdown(f"""
        #### Executive Summary
        - **Total Properties Analyzed:** **{len(clustered_orig):,}**
        - **Optimal Number of Discovered Segments:** **{recommended_k}** (Currently active: **{active_k}**)
        - **Clustering Silhouette Coefficient:** **{cluster_metrics['silhouette_score']:.4f}**
        - **Dimensionality Reduction:** 2D PCA projects properties explaining **{pca.explained_variance_ratio_[0]*100 + pca.explained_variance_ratio_[1]*100:.1f}%** of total scaled variance.
        """)

        st.markdown("#### Major Discovered Market Segments")
        cards = get_segment_summary_cards(clustered_orig)
        for card in cards:
            st.markdown(f"- **{card['display_title']}:** Represents **{card['count']:,} properties ({card['percentage']:.1f}%)** with an average valuation of **${card['avg_price']:,.0f}**, average living area of **{card['avg_area']:,.0f} sqft**, and average layout of **{card['avg_beds']:.1f} bedrooms**.")

        st.markdown("---")
        st.markdown("#### ⬇️ Download Segment Deliverables")
        
        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            # Export 1: Clustered House Dataset with Cluster ID and Segment Name
            csv_buf1 = io.StringIO()
            clustered_orig.to_csv(csv_buf1, index=False)
            st.download_button(
                label="📥 Download Clustered Dataset (CSV)",
                data=csv_buf1.getvalue(),
                file_name="clustered_house_data.csv",
                mime="text/csv",
                help="Exports the full dataset with assigned Cluster ID and Segment Name.",
                use_container_width=True
            )

        with dl_col2:
            # Export 2: Segment Profiles Summary CSV
            csv_buf2 = io.StringIO()
            profile_table.to_csv(csv_buf2)
            st.download_button(
                label="📥 Download Segment Profiles Summary (CSV)",
                data=csv_buf2.getvalue(),
                file_name="segment_profiles_summary.csv",
                mime="text/csv",
                help="Exports the aggregated statistical profile table for all segments.",
                use_container_width=True
            )

        st.markdown("<br>", unsafe_allow_html=True)
        # Property Segmentation Architecture Diagram
        with st.expander("🏗️ View Property Segmentation System Architecture", expanded=True):
            st.markdown("""
```text
                      HOUSE DATASET (Kaggle India)
                                   ↓
                          DATA QUALITY CHECK
                                   ↓
                         PROPERTY FEATURE SET
                                   ↓
                          DATA PREPROCESSING
                                   ↓
                           FEATURE SCALING (StandardScaler)
                                   ↓
                      ┌────────────┴────────────┐
                      ↓                         ↓
                 ELBOW METHOD             SILHOUETTE SCORE
                      └────────────┬────────────┘
                                   ↓
                            OPTIMAL K (K=2)
                                   ↓
                             K-MEANS MODEL
                                   ↓
                           PROPERTY SEGMENTS
                                   ↓
                    ┌──────────────┼──────────────┐
                    ↓              ↓              ↓
              SEGMENT SIZE   SEGMENT PROFILE   PCA 2D VIEW
                    ↓              ↓              ↓
                    └──────────────┼──────────────┘
                                   ↓
                        PROPERTY MARKET INSIGHTS
```
            """)

if __name__ == "__main__":
    main()
