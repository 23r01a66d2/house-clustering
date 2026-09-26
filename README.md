# Property Intelligence & Segment Discovery System
## Multidimensional Property Similarity Space & Community Discovery via K-Means

An advanced unsupervised machine-learning platform that structures housing inventory into a standardized multidimensional similarity space, discovers natural property communities using **K-Means Clustering**, profiles their distinct architectural & economic DNA, and enables interactive property matching and nearest-neighbor peer discovery.

---

## 🏗️ Core System Architecture

```text
                 PROPERTY UNIVERSE
                        │
                        ▼
              PROPERTY INTELLIGENCE
                        │
                        ▼
               PROPERTY SIGNATURES
                        │
                        ▼
                SIMILARITY SPACE
                        │
                 ┌──────┴──────┐
                 ▼             ▼
          SEGMENT DISCOVERY   K DISCOVERY
                 │             │
                 └──────┬──────┘
                        ▼
              PROPERTY COMMUNITIES
                        │
          ┌─────────────┼──────────────┐
          ▼             ▼              ▼
    SEGMENT DNA    SPATIAL PATTERNS  PROPERTY MATCH
          │             │              │
          └─────────────┼──────────────┘
                        ▼
                 MARKET INSIGHTS
```

---

## 🌟 The Analytical Journey

Unlike generic supervised machine-learning pipelines, this platform follows an **Unsupervised Property Similarity & Community Discovery** paradigm:

1. **Property Universe**: Understands the complete housing inventory (14,620 raw records), dynamically computing price spectrums, structural living area ranges, building eras, and spatial bounds without hardcoded figures.
2. **Property Intelligence & Data Trust**: Audits data hygiene (100% completeness across 20 segmentation features, isolating 1 documented typographical entry anomaly) and classifies attributes into domain-specific **Feature Roles** (Property Size, Layout, Property Quality, Property Age, Economic, Spatial).
3. **Property Signatures**: Translates raw attributes into normalized relative indicators (0–100% position within the inventory) representing relative Price, Living Area, Room Capacity, Construction Grade, and Age.
4. **Standardized Similarity Space**: Eliminates unit scale disparities using `StandardScaler` to construct an $N$-dimensional Euclidean similarity space.
5. **K Discovery Lab**: Rigorously evaluates $K = 2 \dots 10$ using Within-Cluster Sum of Squares (Inertia) and Silhouette Scores to select the Official Recommended K based on transparent partition metrics.
6. **Segment Discovery (K-Means)**: Iteratively optimizes cluster centers in standardized space until assignments stabilize.
7. **Segment DNA**: Automatically extracts data-driven segment names (e.g. *Lower-Priced Compact Properties*, *Higher-Priced Spacious Properties*) and generates visual DNA fingerprints.
8. **Spatial Intelligence**: Validates empirical geographic coordinates (Lat 52.39°N–53.01°N, Lon -114.71°W–-113.51°W) and documents dataset provenance without fabricating speculative location claims.
9. **Property Match ("Find My Property Segment")**: Transforms user-simulated properties using the **already-fitted StandardScaler** (zero refitting) and assigns them to the closest learned community centroid with "Why This Match?" delta profiling.
10. **Similar Property Finder**: Retrieves the top nearest-neighbor peers in the standardized feature space, strictly excluding the query property itself.
11. **Market Insights & Exports**: Generates executive portfolio summaries and one-click CSV downloads.

---

## 📁 Project Structure

```text
house_clustering/
│
├── config.py                          # Hyperparameters, column role mappings, paths
├── main.py                            # End-to-end command-line pipeline runner
├── app.py                             # Interactive Streamlit analytics dashboard
├── test_system.py                     # Automated 10-point verification test suite
├── requirements.txt                   # Dependency specifications
│
├── data/
│   └── house_price_india.csv          # Real dataset (14,620 rows × 23 columns)
│
├── src/
│   ├── __init__.py
│   ├── utils.py                       # Console formatters, directory management
│   ├── property_data.py               # Inventory loading and dynamic universe spectrums
│   ├── property_intelligence.py       # Data Trust Panel, feature roles, single fitted StandardScaler
│   ├── property_signature.py          # Normalized relative property signatures (0-100% bars)
│   ├── similarity_engine.py           # Standardized Euclidean space, NearestNeighbors (self-excluded)
│   ├── k_discovery.py                 # K Discovery Lab (K=2..10, Elbow inflection, Silhouette peak)
│   ├── segment_engine.py              # K-Means community discovery, convergence, 2D PCA landscape
│   ├── segment_dna.py                 # Dynamic naming engine, visual fingerprints, comparative profiling
│   ├── spatial_intelligence.py        # Coordinate validation, provenance auditing, spatial filters
│   ├── property_matcher.py            # Fitted scaler transform, bound validation, centroid assignment
│   └── insight_engine.py              # Executive takeaways, architecture diagram, CSV exports
│
└── outputs/
    ├── clustered_house_data.csv       # Exported dataset with Cluster and Segment Name
    └── segment_profiles.csv           # Comparative cross-segment profile table
```

---

## 🚀 Running the Project

### 1. Terminal Pipeline Execution
Run the full 8-stage unsupervised pipeline from PowerShell:
```powershell
cd C:\Users\Varsha\.gemini\antigravity\scratch\house_clustering
.\.venv\Scripts\python.exe main.py
```

### 2. Automated System Verification
Verify all 10 analytical capabilities and confirm zero supervised metrics:
```powershell
.\.venv\Scripts\python.exe test_system.py
```

### 3. Launching the Interactive Web Dashboard
Start the Streamlit application:
```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```
Open your browser to: **http://localhost:8501**

---

## 🧭 Dashboard Navigation Modules

- **DISCOVER**:
  - `01 — Property Universe`: Inventory overview, hero spectrum cards, interactive price vs. area explorer.
  - `02 — Property Landscape`: 2D PCA projection of the similarity space with centroids and bivariate trendlines.
  - `03 — Segment Discovery`: K Discovery Lab (Elbow & Silhouette curves), K-Means convergence, community discovery.
  - `04 — Segment DNA`: Deep community profiles, visual DNA fingerprints (`███████░░░`), relative indicators.
- **EXPLORE**:
  - `05 — Spatial Intelligence`: Validated coordinate map, multi-attribute filtering, provenance documentation.
  - `06 — Property Match`: Interactive simulator to match any property to its closest learned segment centroid.
  - `07 — Similar Property Finder`: Nearest-neighbor peer lookup in standardized space (excluding query property).
- **SYSTEM**:
  - `08 — Method & Data Trust`: Data Trust Panel, real-estate feature roles, mathematical formulations.
  - `09 — Insights & Export`: Executive takeaways, architecture diagram, CSV export downloads.
