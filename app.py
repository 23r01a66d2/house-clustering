"""
STREAMLIT WEB DASHBOARD — House Clustering Using K-Means Clustering Techniques
Interactive multi-view machine learning dashboard for Kaggle House Price Dataset of India.
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
from src.utils import detect_column_roles, ensure_directories
from src.visualization import (
    plot_cluster_distribution, 
    plot_cluster_feature_comparisons, 
    plot_pca_clusters
)
from sklearn.decomposition import PCA

# Configure page layout
st.set_page_config(
    page_title="House Clustering | K-Means ML",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, clean UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 4px 4px 0 0;
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
    profile_table, numeric_profiles = build_cluster_profiles(clustered_orig)
    interpretations = generate_cluster_interpretations(clustered_orig, numeric_profiles)
    return clustered_orig, clustered_scaled, model, metrics, profile_table, interpretations

# ---------------------------------------------------------
# MAIN APP FLOW
# ---------------------------------------------------------
def main():
    ensure_directories()

    # Sidebar: Project Info & Controls
    with st.sidebar:
        st.title("🏠 Navigation & Settings")
        st.markdown("Unsupervised Clustering on **Kaggle House Price Dataset of India**.")
        st.divider()

        # CSV Path Config
        csv_file = st.text_input("Dataset CSV Path", value=str(config.RAW_DATA_PATH))
        
        st.markdown("### ⚙️ Model Hyperparameters")
        st.markdown(f"**Random State:** `{config.RANDOM_STATE}`")
        st.markdown(f"**Tested K Range:** `{config.K_MIN}` to `{config.K_MAX}`")
        st.divider()
        st.caption("B.Tech Final-Year Machine Learning Project")

    # Header
    st.markdown('<div class="main-header">House Clustering Using K-Means Clustering Techniques</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Automated End-to-End Unsupervised Learning Pipeline for Property Segmentation</div>', unsafe_allow_html=True)

    # Verify dataset existence
    csv_path = Path(csv_file)
    if not csv_path.exists():
        st.error(f"⚠️ Dataset not found at `{csv_path}`. Please place the CSV in `data/house_price_india.csv`.")
        return

    try:
        raw_df, raw_summary = load_raw_data(str(csv_path))
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return

    # Run Preprocessing
    cleaned_df, scaled_df, prep_meta = run_preprocessing(raw_df)

    # Compute Optimal K
    k_results_df, recommended_k, k_decision = compute_k_evaluation(scaled_df)

    # Cluster Tuning Slider (Defaults to mathematically recommended K)
    with st.sidebar:
        selected_k = st.slider(
            "Selected Number of Clusters (K)",
            min_value=config.K_MIN,
            max_value=config.K_MAX,
            value=int(recommended_k),
            help="Defaults to optimal K selected by Elbow and Silhouette analysis."
        )
        if selected_k != recommended_k:
            st.info(f"Recommended K is {recommended_k}. You selected {selected_k}.")

    # Run Clustering with selected K
    clustered_orig, clustered_scaled, model, cluster_metrics, profile_table, interpretations = run_clustering_cached(
        scaled_df, cleaned_df, selected_k
    )

    # Top Metric KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Houses", f"{len(cleaned_df):,}")
    with col2:
        st.metric("Clustering Features", f"{len(prep_meta['selected_features'])}")
    with col3:
        st.metric("Selected K", f"{selected_k}")
    with col4:
        st.metric("Silhouette Score", f"{cluster_metrics['silhouette_score']:.4f}")
    with col5:
        st.metric("Inertia (WCSS)", f"{cluster_metrics['inertia']/1e6:.1f}M" if cluster_metrics['inertia'] > 1e6 else f"{cluster_metrics['inertia']:,.0f}")

    st.markdown("---")

    # App Tabs
    tabs = st.tabs([
        "📊 Dashboard",
        "📁 Dataset Overview",
        "⚙️ Data Preprocessing",
        "🔍 EDA",
        "📈 Optimal K",
        "🎯 K-Means Clustering",
        "🗺️ Cluster Visualization",
        "📋 Cluster Profiles",
        "📑 Final Results & Export"
    ])

    # ---------------------------------------------------------
    # TAB 1: DASHBOARD
    # ---------------------------------------------------------
    with tabs[0]:
        st.subheader("System Overview & Executive Summary")
        st.markdown(f"""
        This machine learning system performs unsupervised property segmentation on the **Kaggle House Price Dataset of India**.
        Instead of predicting prices, it groups **{len(cleaned_df):,} houses** into **{selected_k} distinct property categories** 
        based on morphological, structural, and valuation similarities.
        """)

        st.markdown("#### Key Pipeline Achievements")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown(f"""
            - **Data Source:** `{csv_path.name}` ({raw_summary['rows']:,} raw records, {raw_summary['columns']} attributes)
            - **Cleaning:** Removed {prep_meta['duplicates_removed']} duplicates, handled {prep_meta['outliers_removed']} unrealistic records
            - **Feature Space:** {len(prep_meta['selected_features'])} property features scaled using `StandardScaler`
            - **Cluster Optimization:** Evaluated K=2..10 via both Elbow and Silhouette methods
            """)
        with c2:
            st.markdown(f"""
            - **Mathematical Optimal K:** `{recommended_k}` ({k_decision['rationale']})
            - **Active Model K:** `{selected_k}`
            - **Convergence:** Reached in `{cluster_metrics['n_iterations']}` iterations
            - **Silhouette Coefficient:** `{cluster_metrics['silhouette_score']:.4f}`
            """)

        st.markdown("#### Cluster Distribution Summary")
        dist_df = pd.DataFrame([
            {"Cluster": f"Cluster {c}", "Houses": count, "Percentage": f"{count/len(cleaned_df)*100:.1f}%"}
            for c, count in cluster_metrics["cluster_counts"].items()
        ])
        st.dataframe(dist_df, use_container_width=True, hide_index=True)

    # ---------------------------------------------------------
    # TAB 2: DATASET OVERVIEW
    # ---------------------------------------------------------
    with tabs[1]:
        st.subheader("Raw Dataset Inspection")
        st.write(f"The raw dataset contains **{raw_summary['rows']:,} rows** and **{raw_summary['columns']} columns**.")

        st.markdown("#### Sample Records (First 10 Rows)")
        st.dataframe(raw_df.head(10), use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Column Data Types & Missing Values")
            dtypes_df = pd.DataFrame({
                "Column": list(raw_summary["dtypes"].keys()),
                "Data Type": list(raw_summary["dtypes"].values()),
                "Missing Values": [raw_summary["missing_values"][col] for col in raw_summary["dtypes"].keys()]
            })
            st.dataframe(dtypes_df, use_container_width=True, height=350, hide_index=True)

        with col_b:
            st.markdown("#### Dataset Characteristics")
            st.markdown(f"""
            - **Total Records:** `{raw_summary['rows']:,}`
            - **Total Attributes:** `{raw_summary['columns']}`
            - **Numerical Columns:** `{len(raw_summary['numerical_cols'])}`
            - **Categorical / String Columns:** `{len(raw_summary['categorical_cols'])}`
            - **Duplicate Rows:** `{raw_summary['duplicates']}`
            - **Total Missing Cells:** `{raw_summary['total_missing']}`
            """)

    # ---------------------------------------------------------
    # TAB 3: DATA PREPROCESSING
    # ---------------------------------------------------------
    with tabs[2]:
        st.subheader("Data Preprocessing & Transformation Audit")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Records Before Preprocessing", f"{prep_meta['records_before']:,}")
        with c2:
            st.metric("Records After Preprocessing", f"{prep_meta['records_after']:,}")
        with c3:
            st.metric("Anomalies / Outliers Filtered", f"{prep_meta['outliers_removed']}")

        st.markdown("#### Feature Selection Details")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.success(f"**Selected Clustering Features ({len(prep_meta['selected_features'])}):**")
            st.write(prep_meta["selected_features"])
            st.caption("All selected features are numerical, scaled with StandardScaler (mean=0, std=1).")

        with col_f2:
            st.info(f"**Excluded Non-Predictive Features ({len(prep_meta['removed_features'])}):**")
            st.write(prep_meta["removed_features"])
            st.caption("Excluded to prevent non-physical identifiers (ID, dates, codes) from distorting distance metrics.")

        st.markdown("#### Cleaned vs Scaled Feature Preview")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("**Cleaned (Original Scale, Sample):**")
            st.dataframe(cleaned_df[prep_meta["selected_features"]].head(5), use_container_width=True)
        with col_s2:
            st.markdown("**Standardized (Zero Mean, Unit Variance):**")
            st.dataframe(scaled_df[prep_meta["selected_features"]].head(5), use_container_width=True)

    # ---------------------------------------------------------
    # TAB 4: EXPLORATORY DATA ANALYSIS (EDA)
    # ---------------------------------------------------------
    with tabs[3]:
        st.subheader("Exploratory Data Analysis")

        st.markdown("#### Descriptive Statistics")
        desc_stats = compute_descriptive_statistics(cleaned_df, prep_meta["selected_features"])
        st.dataframe(desc_stats.style.format("{:,.2f}"), use_container_width=True)

        st.markdown("#### Feature Correlation Heatmap")
        if HAS_PLOTLY:
            corr = cleaned_df[prep_meta["selected_features"]].corr()
            fig_corr = px.imshow(
                corr,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                title="Pearson Correlation Heatmap"
            )
            fig_corr.update_layout(height=650)
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            fig_corr = plot_correlation_heatmap(cleaned_df, prep_meta["selected_features"])
            st.pyplot(fig_corr)

        st.markdown("#### Interactive Feature Distribution")
        sel_dist_col = st.selectbox(
            "Select Feature for Distribution Analysis:",
            options=prep_meta["selected_features"],
            index=0
        )
        if HAS_PLOTLY:
            fig_dist = px.histogram(
                cleaned_df,
                x=sel_dist_col,
                marginal="box",
                nbins=40,
                title=f"Distribution & Spread: {sel_dist_col}",
                color_discrete_sequence=["#3B82F6"]
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        else:
            fig_dist = plot_feature_distribution(cleaned_df, sel_dist_col)
            st.pyplot(fig_dist)

        st.markdown("#### Interactive Bivariate Scatter Analysis")
        col_x, col_y = st.columns(2)
        with col_x:
            x_feat = st.selectbox("X Axis Feature:", prep_meta["selected_features"], index=min(1, len(prep_meta["selected_features"])-1))
        with col_y:
            y_feat = st.selectbox("Y Axis Feature:", prep_meta["selected_features"], index=0)

        if HAS_PLOTLY:
            sample_eda = cleaned_df.sample(n=min(3000, len(cleaned_df)), random_state=config.RANDOM_STATE)
            fig_scatter = px.scatter(
                sample_eda,
                x=x_feat,
                y=y_feat,
                opacity=0.5,
                title=f"{y_feat} vs {x_feat} (with Linear Trendline)",
                color_discrete_sequence=["#2563EB"]
            )
            # Add linear trendline using pure NumPy (zero statsmodels dependency)
            try:
                x_vals = sample_eda[x_feat].values.astype(float)
                y_vals = sample_eda[y_feat].values.astype(float)
                valid = ~(np.isnan(x_vals) | np.isnan(y_vals) | np.isinf(x_vals) | np.isinf(y_vals))
                if np.sum(valid) > 1:
                    slope, intercept = np.polyfit(x_vals[valid], y_vals[valid], 1)
                    x_line = np.linspace(x_vals[valid].min(), x_vals[valid].max(), 50)
                    y_line = slope * x_line + intercept
                    fig_scatter.add_trace(go.Scatter(
                        x=x_line,
                        y=y_line,
                        mode="lines",
                        line=dict(color="red", width=2.5),
                        name="Trendline"
                    ))
            except Exception:
                pass
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            fig_scatter = plot_bivariate_scatter(cleaned_df, x_feat, y_feat)
            st.pyplot(fig_scatter)

    # ---------------------------------------------------------
    # TAB 5: OPTIMAL K SELECTION
    # ---------------------------------------------------------
    with tabs[4]:
        st.subheader("Optimal K Selection: Elbow Method & Silhouette Analysis")
        st.markdown(f"""
        To avoid arbitrarily guessing K, we systematically evaluate candidate cluster counts from **K = {config.K_MIN} to {config.K_MAX}**.
        """)

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            if HAS_PLOTLY:
                fig_elbow = go.Figure()
                fig_elbow.add_trace(go.Scatter(
                    x=k_results_df["k"],
                    y=k_results_df["inertia"],
                    mode="lines+markers",
                    name="Inertia (WCSS)",
                    line=dict(color="#2563EB", width=3),
                    marker=dict(size=8)
                ))
                fig_elbow.add_vline(x=selected_k, line_width=2, line_dash="dash", line_color="red", annotation_text=f"Selected K={selected_k}")
                fig_elbow.update_layout(
                    title="Elbow Method: K vs Inertia",
                    xaxis_title="Number of Clusters (K)",
                    yaxis_title="Inertia (Within-Cluster Sum of Squares)",
                    xaxis=dict(tickmode="linear", tick0=2, dtick=1)
                )
                st.plotly_chart(fig_elbow, use_container_width=True)
            else:
                fig_elbow = plot_elbow_curve(k_results_df, selected_k=selected_k)
                st.pyplot(fig_elbow)

        with col_g2:
            if HAS_PLOTLY:
                fig_sil = go.Figure()
                fig_sil.add_trace(go.Scatter(
                    x=k_results_df["k"],
                    y=k_results_df["silhouette_score"],
                    mode="lines+markers",
                    name="Silhouette Score",
                    line=dict(color="#10B981", width=3),
                    marker=dict(size=8)
                ))
                fig_sil.add_vline(x=selected_k, line_width=2, line_dash="dash", line_color="red", annotation_text=f"Selected K={selected_k}")
                fig_sil.update_layout(
                    title="Silhouette Score: K vs Silhouette Coefficient",
                    xaxis_title="Number of Clusters (K)",
                    yaxis_title="Silhouette Score",
                    xaxis=dict(tickmode="linear", tick0=2, dtick=1)
                )
                st.plotly_chart(fig_sil, use_container_width=True)
            else:
                fig_sil = plot_silhouette_curve(k_results_df, selected_k=selected_k)
                st.pyplot(fig_sil)

        st.markdown("#### Evaluation Metrics Table")
        st.dataframe(
            k_results_df.style.format({"inertia": "{:,.1f}", "silhouette_score": "{:.4f}"}),
            use_container_width=True,
            hide_index=True
        )

        st.info(f"💡 **Automated Decision Rationale:** {k_decision['rationale']}")

    # ---------------------------------------------------------
    # TAB 6: K-MEANS CLUSTERING
    # ---------------------------------------------------------
    with tabs[5]:
        st.subheader(f"K-Means Clustering Execution (K = {selected_k})")
        
        c_m1, c_m2, c_m3, c_m4 = st.columns(4)
        with c_m1:
            st.metric("Clusters", f"{selected_k}")
        with c_m2:
            st.metric("Iterations to Converge", f"{cluster_metrics['n_iterations']}")
        with c_m3:
            st.metric("Total Inertia", f"{cluster_metrics['inertia']:,.0f}")
        with c_m4:
            st.metric("Silhouette Score", f"{cluster_metrics['silhouette_score']:.4f}")

        st.markdown("#### Cluster Centroids (in Standardized Feature Space)")
        centroids_df = pd.DataFrame(
            model.cluster_centers_,
            columns=prep_meta["selected_features"],
            index=[f"Cluster {i}" for i in range(selected_k)]
        )
        st.dataframe(centroids_df.style.format("{:+.2f}"), use_container_width=True)

        st.markdown("#### Clustered Dataset Preview")
        st.dataframe(clustered_orig.head(10), use_container_width=True)

    # ---------------------------------------------------------
    # TAB 7: CLUSTER VISUALIZATION
    # ---------------------------------------------------------
    with tabs[6]:
        st.subheader("2D PCA Projection & Cluster Separation")
        st.markdown("""
        Because clustering is performed in a high-dimensional feature space, we apply **Principal Component Analysis (PCA)**
        to project houses onto the 2 directions of maximum variance for visual inspection.
        """)

        # Perform PCA for visualization
        X_scaled = scaled_df[prep_meta["selected_features"]].values
        pca = PCA(n_components=2, random_state=config.RANDOM_STATE)
        X_pca = pca.fit_transform(X_scaled)
        centroids_pca = pca.transform(model.cluster_centers_)

        if HAS_PLOTLY:
            pca_vis_df = pd.DataFrame({
                "PC1": X_pca[:, 0],
                "PC2": X_pca[:, 1],
                "Cluster": [f"Cluster {lbl}" for lbl in clustered_orig["Cluster"]]
            })

            sample_pca = pca_vis_df.sample(n=min(5000, len(pca_vis_df)), random_state=config.RANDOM_STATE)

            fig_pca = px.scatter(
                sample_pca,
                x="PC1",
                y="PC2",
                color="Cluster",
                opacity=0.4,
                title=f"K-Means Clusters Projected in 2D PCA Space (Variance: PC1 {pca.explained_variance_ratio_[0]*100:.1f}%, PC2 {pca.explained_variance_ratio_[1]*100:.1f}%)",
                color_discrete_sequence=px.colors.qualitative.Plotly
            )

            for idx, pt in enumerate(centroids_pca):
                fig_pca.add_trace(go.Scatter(
                    x=[pt[0]],
                    y=[pt[1]],
                    mode="markers+text",
                    marker=dict(symbol="x", size=14, color="black", line=dict(width=2, color="white")),
                    text=[f"C{idx}"],
                    textposition="top center",
                    name=f"Centroid {idx}",
                    showlegend=False
                ))

            fig_pca.update_layout(height=650)
            st.plotly_chart(fig_pca, use_container_width=True)
        else:
            fig_pca, _ = plot_pca_clusters(
                scaled_df, 
                clustered_orig["Cluster"].values, 
                centroids=model.cluster_centers_
            )
            st.pyplot(fig_pca)

        st.markdown("#### Comparative Feature Distributions by Cluster")
        roles = detect_column_roles(clustered_orig)
        comp_col = st.selectbox(
            "Select Feature to Compare Across Clusters:",
            options=prep_meta["selected_features"],
            index=0
        )
        if HAS_PLOTLY:
            fig_box = px.box(
                clustered_orig,
                x="Cluster",
                y=comp_col,
                color="Cluster",
                points=False,
                title=f"Boxplot of {comp_col} Across Clusters",
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            fig_box = plot_cluster_feature_comparisons(clustered_orig)
            st.pyplot(fig_box)

    # ---------------------------------------------------------
    # TAB 8: CLUSTER PROFILES & DYNAMIC INTERPRETATION
    # ---------------------------------------------------------
    with tabs[7]:
        st.subheader("Cluster Profiles & Statistical Characterization")
        st.markdown("Summary of average property characteristics across clusters:")
        st.dataframe(profile_table, use_container_width=True)

        st.markdown("#### Automated Dynamic Cluster Interpretations")
        st.caption("Generated strictly from relative statistical percentiles and feature deviations without hardcoded templates.")

        for c_id, text in interpretations.items():
            with st.expander(f"📌 Cluster {c_id} Characterization", expanded=True):
                st.markdown(text)

    # ---------------------------------------------------------
    # TAB 9: FINAL RESULTS & EXPORT
    # ---------------------------------------------------------
    with tabs[8]:
        st.subheader("Final Output & Dataset Download")
        st.markdown(f"""
        The final clustering output integrates the original house property records with their assigned cluster (`Cluster` column).
        - **Total Houses Exported:** `{len(clustered_orig):,}`
        - **Number of Clusters:** `{selected_k}`
        """)

        # CSV Download Button
        csv_buffer = io.StringIO()
        clustered_orig.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()

        st.download_button(
            label="⬇️ Download Clustered Dataset (CSV)",
            data=csv_data,
            file_name="clustered_house_data.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.markdown("#### Preview Export File")
        st.dataframe(clustered_orig.head(15), use_container_width=True)

if __name__ == "__main__":
    main()
