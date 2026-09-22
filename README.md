# House Clustering Using K-Means Clustering Techniques

An end-to-end unsupervised machine learning system for property segmentation on the **Kaggle House Price Dataset of India**. 

The system discovers natural groupings among properties based on physical attributes, construction quality, layout, and valuation, without relying on arbitrary assumptions or hard-coded rules.

---

## 🏗️ Project Architecture & Modules

The pipeline implements an 8-module unsupervised learning workflow:

```text
House Dataset (data/house_price_india.csv)
  │
  ├── [1/8] Module 1: Data Collection (src/data_loader.py)
  ├── [2/8] Module 2: Data Preprocessing (src/preprocessing.py)
  ├── [3/8] Module 3: Exploratory Data Analysis (src/eda.py)
  ├── [4/8] Module 4: Optimal K Selection (src/optimal_k.py)
  ├── [5/8] Module 5: K-Means Clustering (src/clustering.py)
  ├── [6/8] Module 6: Visualization & Results (src/visualization.py)
  ├── [7/8] Module 7: Cluster Analysis & Interpretation (src/cluster_analysis.py)
  └── [8/8] Module 8: Final Output & Export (main.py & app.py)
```

---

## 📁 Project Structure

```text
house_clustering/
│
├── data/
│   └── house_price_india.csv          # Kaggle House Price Dataset of India (14,620 rows)
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
│   └── cluster_analysis.py            # Module 7: Dynamic statistical profiling and rule-based interpretation
│
├── outputs/
│   ├── plots/                         # Generated high-resolution visualization figures
│   └── clustered_house_data.csv       # Final export dataset with assigned 'Cluster' labels
│
├── app.py                             # Interactive Streamlit Web Dashboard
├── main.py                            # Standalone end-to-end CLI execution pipeline
├── requirements.txt                   # Verified dependencies
└── README.md                          # Project documentation and guide
```

---

## ⚙️ Environment Setup & Local Execution (Windows)

The project runs completely on your local Windows system without requiring cloud services or external APIs.

### 1. Open Terminal & Navigate to Project

```powershell
cd C:\Users\Varsha\.gemini\antigravity\scratch\house_clustering
```

### 2. Create Virtual Environment

```powershell
python -m venv .venv
```

### 3. Activate Virtual Environment

```powershell
.venv\Scripts\activate
```

### 4. Install Dependencies

```powershell
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### Option A: Command-Line Pipeline (`main.py`)

Executes all 8 modules sequentially, logs progress in the terminal, generates plots in `outputs/plots/`, and exports `outputs/clustered_house_data.csv`:

```powershell
python main.py
```

### Option B: Interactive Streamlit Web Dashboard (`app.py`)

Launches a multi-tab interactive web interface with dynamic filters, Plotly visualizations, K-tuning slider, and one-click CSV download.

**Method 1 (Recommended on Windows):**
```powershell
python -m streamlit run app.py
```

**Method 2 (Using Virtualenv CLI):**
```powershell
.venv\Scripts\streamlit run app.py
```

Or after activating `.venv`:
```powershell
.venv\Scripts\activate
streamlit run app.py
```

The dashboard will open automatically in your default browser at `http://localhost:8501`.

---

## 🔬 Machine Learning Methodology

1. **Feature Selection**:
   - Automatically drops non-physical identifiers (`id`, `Date`, `Postal Code`) to avoid distorting Euclidean distance.
   - Retains 18 morphological, structural, and valuation attributes (`Price`, `living area`, `lot area`, `bedrooms`, `bathrooms`, `floors`, `grade`, `condition`, `built year`, etc.).
2. **Feature Scaling**:
   - Uses Scikit-learn `StandardScaler` to ensure all features have zero mean and unit variance.
   - Prevents high-magnitude variables (such as price or lot area) from dominating the K-Means distance calculations.
3. **Optimal K Determination**:
   - Tests candidate cluster counts from $K = 2$ to $K = 10$.
   - Calculates **Inertia (Within-Cluster Sum of Squares)** to identify the Elbow point using maximum perpendicular distance.
   - Evaluates the **Silhouette Score** to measure cluster cohesion and separation.
   - Recommends optimal $K$ transparently while allowing interactive exploration.
4. **Dimensionality Reduction for Visualization**:
   - High-dimensional feature space is projected onto 2 Principal Components (PC1 and PC2) via **PCA** strictly for visualization.
   - Actual K-Means centroids are projected into the 2D PCA space for reference.
5. **Dynamic Cluster Interpretation**:
   - Computes statistical profiles (mean, standard deviation, percentiles) for each cluster.
   - Generates human-readable descriptions based on relative deviations from overall population averages without static hard-coded text.

---

## 🛠️ Technologies Used

- **Language:** Python 3.13 / 3.14
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-learn (`KMeans`, `StandardScaler`, `PCA`, `silhouette_score`)
- **Visualizations:** Matplotlib, Seaborn, Plotly
- **Web Dashboard:** Streamlit
