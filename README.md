# Property Segmentation Analytics
## House Clustering Using K-Means Clustering Techniques

An end-to-end unsupervised machine learning and business intelligence platform for property segmentation on the **Kaggle House Price Dataset of India**. 

The system discovers natural groupings among properties based on physical attributes, construction quality, layout, spatial coordinates, and market valuation, without relying on arbitrary assumptions or hard-coded rules.

---

## 🏗️ System Architecture

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

---

## 📁 Project Structure

```text
house_clustering/
│
├── data/
│   └── house_price_india.csv          # Kaggle House Price Dataset of India (14,620 rows × 23 columns)
│
├── src/
│   ├── __init__.py
│   ├── config.py                      # Paths, random seeds, K ranges, feature keywords
│   ├── utils.py                       # Logging formatters, directory management, dynamic column matching
│   ├── data_loader.py                 # Module 1: Loading, validation, schema introspection
│   ├── preprocessing.py               # Module 2: Outlier handling, feature scaling (StandardScaler)
│   ├── eda.py                         # Module 3: Summary stats, heatmaps, distributions, scatter plots
│   ├── optimal_k.py                   # Module 4: Inertia (Elbow) and Silhouette analysis for K=2..10
│   ├── clustering.py                  # Module 5: Scikit-learn KMeans training and centroid extraction
│   ├── visualization.py               # Module 6: 2D PCA projection, distribution, feature comparison
│   ├── cluster_analysis.py            # Module 7: Dynamic statistical profiling and rule-based interpretation
│   └── segment_naming.py              # Dynamic market segment naming and profile cards
│
├── outputs/
│   ├── plots/                         # Generated high-resolution visualization figures
│   └── clustered_house_data.csv       # Final export dataset with assigned 'Cluster' and 'Segment Name'
│
├── app.py                             # Interactive Property Segmentation Analytics Web Dashboard
├── main.py                            # Standalone end-to-end CLI execution pipeline
├── requirements.txt                   # Verified dependencies
└── README.md                          # Project documentation and guide
```

---

## 🧭 Dashboard Navigation Journey

The web dashboard is structured around a real-estate business analytics journey:

1. 🏠 **Market Overview:** Macro-level portfolio KPIs (Total Properties, Average/Median Price, Average Area, Average Bedrooms, Discovered Segments), dual price & area distribution charts, and data-driven market snapshot.
2. 🏘 **Property Explorer:** Interactive multi-criteria search (price range, area range, bedrooms, bathrooms, condition, segment) and single property profile inspection.
3. 📊 **Market Patterns:** Pre-clustering market relationships (Price vs Area, Price vs Bedrooms, Area vs Bedrooms, Price by Bedroom boxplots, Pearson correlation heatmap).
4. 🧩 **Property Segments:** Profile cards for all discovered market segments, interactive 2D PCA cluster space visualization with projected centroids, and segment market share breakdown.
5. ⚖ **Segment Comparison:** Side-by-side bar chart benchmarks (Price, Living Area, Bedrooms, Bathrooms by segment) and comprehensive cross-segment comparison table.
6. 📍 **Segment Map / Spatial View:** Interactive geographic distribution map using the actual `Lattitude` and `Longitude` coordinates present in the Kaggle dataset.
7. ⚙ **Clustering Analysis:** Technical section for K-selection (Elbow & Silhouette curves), official recommended K vs experimental exploration slider, and feature selection justification.
8. 📁 **Dataset & Quality:** Data quality audit, schema preview, and missing/duplicate verification.
9. 📑 **Property Segmentation Summary:** Executive overview and downloadable CSV deliverables.

---

## ⚙️ Environment Setup & Local Execution (Windows)

The project runs completely on your local Windows system without requiring cloud services or external APIs.

### 1. Open Terminal & Navigate to Project

```powershell
cd C:\Users\Varsha\.gemini\antigravity\scratch\house_clustering
```

### 2. Activate Virtual Environment

```powershell
.venv\Scripts\activate
```

*(Or use `.venv\Scripts\python.exe` directly if execution policies are restricted).*

---

## 🚀 Running the Project

### Option A: Command-Line Pipeline (`main.py`)

Executes all 8 modules sequentially, logs progress in the terminal, saves plots in `outputs/plots/`, and exports `outputs/clustered_house_data.csv` with both `Cluster` and `Segment Name`:

```powershell
python main.py
```

### Option B: Interactive Real-Estate Analytics Dashboard (`app.py`)

Launches the multi-view property analytics web dashboard:

```powershell
python -m streamlit run app.py
```

*(Or `.\.venv\Scripts\streamlit run app.py`)*

The dashboard will open automatically in your default browser at `http://localhost:8501`.

---

## 🛠️ Technologies Used

- **Language:** Python 3.13 / 3.14
- **Data Manipulation:** `pandas`, `numpy`
- **Machine Learning:** `scikit-learn` (`KMeans`, `StandardScaler`, `PCA`, `silhouette_score`)
- **Visualizations:** `matplotlib`, `seaborn`, `plotly`
- **Web Dashboard:** `streamlit`
