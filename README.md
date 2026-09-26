# Property Intelligence
## House Clustering Using K-Means Clustering Techniques

A full-stack, responsive web application for unsupervised property discovery and community segmentation built using **Python + Flask**, Scikit-learn, HTML5, CSS3, vanilla JavaScript, Plotly.js, and Leaflet.js.

The platform structures multi-attribute residential rental housing inventory into a standardized $N$-dimensional similarity space, discovers natural housing communities using **K-Means Clustering**, profiles their distinct architectural & economic DNA, and provides interactive property matching and nearest-neighbor peer lookup.

---

## 📊 Dataset Provenance & Data Hygiene

- **Data Source**: Validated Indian Rental House Price dataset sourced from Makaan.com rental listings collected in April 2024.
- **Raw Volume**: **13,910** raw rental records across three major Indian metropolitan regions:
  - `Indian_housing_Delhi_data.csv` (Delhi NCR: 5,000 records)
  - `Indian_housing_Mumbai_data.csv` (Mumbai MMR: 5,000 records)
  - `Indian_housing_Pune_data.csv` (Pune Metro: 3,910 records)
- **Data Deduplication**: **1,063** exact duplicate rental listings across regional files were identified and removed, yielding **12,847** unique, analysis-ready listings.
- **Missing Value Imputation**: Exactly **55** missing bathroom records post-deduplication (25 in Delhi, 13 in Mumbai, 17 in Pune; 56 in raw files) were median-imputed using locality medians.
- **Spatial Validation**: **12,803** listings (99.66%) contain validated coordinates spanning Latitude 18.35°–28.81° N and Longitude 72.72°–77.34° E across Delhi NCR, Mumbai MMR, and Pune. 44 records with scraper artifacts or out-of-bounds coordinates were flagged and excluded from map plotting while preserved in tabular records.
- **Processed Inventory**: `data/processed_indian_rentals.csv` (12,847 rows × 27 columns).

---

## 🧠 Machine Learning & Clustering Methodology

1. **Similarity Dimensions ($N=5$)**:
   - `price`: Monthly rent in Indian Rupees (₹/month) — *log1p transformed*
   - `sqft`: Living area in square feet — *log1p transformed*
   - `bhk`: Number of bedrooms
   - `numBathrooms`: Sanitation fixture count
   - `furnishing_tier`: Numerical ordinal tier (0: Unfurnished, 1: Semi-Furnished, 2: Furnished)
   - *Note*: `price_per_sqft` is retained for observational analysis but excluded from clustering distance to prevent multi-collinear distortion.

2. **Variance Stabilization & Scaling**:
   - Heavy right-skewed distributions of monthly rent and living area are stabilized via natural logarithm transformation ($\ln(1 + x)$).
   - Attributes are z-score standardized via Scikit-learn's `StandardScaler` to zero mean and unit variance ($\mu = 0, \sigma = 1$), equalizing architectural layout counts with monetary values.

3. **K Discovery Lab ($K \in [2, 10]$)**:
   - Within-Cluster Sum of Squares (WCSS / Inertia) and Silhouette Scores were evaluated across $K=2$ to $K=10$.
   - While $K=2$ achieves a mathematical silhouette peak ($0.4521$) by trivially bifurcating the market into budget vs luxury, Kneedle elbow detection identifies **$K=5$** as the maximum curvature point where incremental variance explained flattens to 12.0%.
   - Fitting $K=5$ partitions the inventory into five distinct, economically meaningful rental archetypes.

4. **Dynamic Segment DNA**:
   - Segments are profiled dynamically from their empirical distributions (medians, ranges, furnishing distributions) rather than arbitrary marketing names.
   - Normalized relative DNA fingerprints ($0\text{--}100\%$ spectrum) provide transparent visual signatures.

5. **2D PCA Projection**:
   - Principal Component Analysis (PCA) is fitted strictly for 2D visualization of the high-dimensional similarity landscape, retaining maximum variance across principal axes without affecting clustering assignments.

6. **Property Matcher (Zero Data Leakage)**:
   - New property queries are transformed using the **already-fitted** pipeline (log1p + StandardScaler) without refitting.
   - Computes standardized Euclidean distances to all fitted centroids and validates user inputs against empirical bounds.

7. **Similar Property Finder**:
   - Employs a pre-fitted Scikit-learn `NearestNeighbors` index in the standardized similarity space to return the top 5 closest peers, **strictly excluding the query property itself**.

---

## 🏗️ Core System Architecture

```text
                 INDIAN RENTAL UNIVERSE
                 (13,910 Raw / 12,847 Processed)
                         │
                         ▼
               PROPERTY INTELLIGENCE
          (Deduplication, Cleaning, Auditing)
                         │
                         ▼
                PROPERTY SIGNATURES
               (Relative 0-100% Scales)
                         │
                         ▼
           SIMILARITY SPACE (STANDARDIZED)
        (log1p Stabilized + StandardScaler)
                         │
                  ┌──────┴──────┐
                  ▼             ▼
           SEGMENT DISCOVERY   K DISCOVERY
             (K-Means, K=5)   (K=2..10 Evaluated)
                  │             │
                  └──────┬──────┘
                         ▼
               PROPERTY COMMUNITIES
                         │
           ┌─────────────┼──────────────┐
           ▼             ▼              ▼
     SEGMENT DNA    SPATIAL PATTERNS  PROPERTY MATCH
   (Fingerprints)   (12,803 Validated) (Zero-Leakage)
           │             │              │
           └─────────────┼──────────────┘
                         ▼
                  MARKET INSIGHTS
```

---

## 🌐 Web Application Features

1. **Home Landing Page (`/`)**:
   - Modern architectural hero section with dynamic inventory statistics.
   - Dynamic ticker strip showing analyzed properties (12,847), discovered communities (5), median rent (₹32,500), and feature dimensions (5).
   - Interactive Property Universe scatter plot (Living Area vs. Monthly Rent, colored by cluster).
   - 5-step visual storytelling journey explaining the unsupervised pipeline.
   - Discovered community preview cards with data-driven segment titles.

2. **Explore Properties (`/explore`)**:
   - Sticky multi-attribute filter bar (Monthly Rent, Area, Bedrooms, Bathrooms, Furnishing Status, City, and Segment).
   - Responsive property card grid with real-estate iconography (Beds, Baths, Area, Rent/sqft).
   - Interactive Property Detail Modal with normalized relative **Property Signature** bars (`███████░░░ 72%`).
   - Integrated Geographic Distribution Map using Leaflet.js with cluster-colored markers and city zoom controls (Delhi NCR, Mumbai MMR, Pune).
   - Shimmer skeleton loading and graceful zero-result empty state.

3. **K-Means Clustering (`/clusters`)**:
   - Core academic clustering page with transparent mathematical explanations.
   - **K Discovery Lab**: Interactive Plotly.js charts for Elbow Curve (WCSS Inertia) and Silhouette Curve for $K=2..10$.
   - Interactive 4-step K-Means mechanical stepper (Init Centers $\rightarrow$ Assign Points $\rightarrow$ Shift Centers $\rightarrow$ Stabilize).
   - Full-width 2D PCA Property Landscape with projected community centroids and empirical variance ratios.

4. **Cluster Analysis (`/compare`)**:
   - Distinct **Segment DNA** profile cards with visual relative indicator bars.
   - Interactive Segment A vs. Segment B side-by-side comparison tool with delta difference calculations.
   - Comparative Plotly.js bar charts for Monthly Rent and Living Area across segments.
   - Comprehensive cross-segment matrix table.

5. **Property Match & Similar Finder (`/property-match`)**:
   - Multi-step interactive simulator form with client-side & server-side empirical bound validation.
   - Transforms input features via the **already-fitted StandardScaler** (zero refitting) and assigns nearest cluster centroid.
   - **"Why This Match?"** centroid comparison table.
   - **Similar Property Finder**: Retrieves top 5 closest peer properties in standardized space, **strictly excluding the query property itself**.

6. **Methodology & Data Trust (`/methodology`)**:
   - Validated dataset provenance notice (Lat 18.35°–28.81° N, Lon 72.72°–77.34° E).
   - Data Trust Panel: Examined records (13,910), usable inventory (12,847), exact duplicates removed (1,063), median-imputed bathrooms (55).
   - Domain-specific Feature Roles breakdown (Monetary, Spatial Capacity, Architectural Layout, Asset Readiness, Observational Analytics, Spatial Intelligence).
   - Mathematical foundations (StandardScaler, Euclidean distance, K-Means objective).
   - Full Core Architecture flowchart and deliverable CSV download links.

---

## 📁 Project Structure

```text
house_clustering/
│
├── config.py                          # Hyperparameters, column role mappings, paths
├── server.py                          # Flask application server (model caching on startup)
├── main.py                            # End-to-end command-line pipeline runner
├── test_system.py                     # Automated 9-stage verification test suite
├── test_server.py                     # Flask HTTP endpoints test suite
├── requirements.txt                   # Dependency specifications
├── README.md                          # Comprehensive project documentation
│
├── data/
│   ├── makaan_raw/                    # Preserved raw regional CSV files
│   │   ├── Indian_housing_Delhi_data.csv
│   │   ├── Indian_housing_Mumbai_data.csv
│   │   └── Indian_housing_Pune_data.csv
│   └── processed_indian_rentals.csv   # Unified, deduplicated, cleaned dataset (12,847 rows)
│
├── src/
│   ├── __init__.py
│   ├── utils.py                       # Logging utilities and directory helpers
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
├── templates/
│   ├── base.html                      # Layout, top navbar, footer, CDNs
│   ├── index.html                     # Home landing page
│   ├── explore.html                   # Property explorer, cards, modal, Leaflet map
│   ├── clusters.html                  # Academic clustering, K Discovery, PCA landscape
│   ├── compare.html                   # Segment DNA, comparison tool, comparative charts
│   ├── property_match.html            # Simulator form, match result, similar finder
│   ├── methodology.html               # Technical docs, Data Trust panel, architecture
│   └── errors/
│       ├── 404.html                   # Custom 404 Not Found page
│       └── 500.html                   # Custom 500 Server Error page
│
├── static/
│   ├── css/
│   │   └── style.css                  # Executive dark-navy & emerald design system
│   ├── js/
│   │   ├── main.js                    # Navbar, count-up animation, icon hydration
│   │   ├── explore.js                 # AJAX filtering, skeletons, modal, Leaflet map
│   │   ├── clusters.js                # K Discovery curves, K-Means stepper, PCA landscape
│   │   ├── compare.js                 # Segment A vs B comparison, comparative bar charts
│   │   └── property_match.js          # Simulator form, match result, similar finder
│   └── assets/
│       ├── hero_architecture.jpg      # Architectural hero visual (Unsplash Free License)
│       ├── explore_header.jpg         # Architectural header visual (Unsplash Free License)
│       ├── property_abstract.jpg      # Architectural texture visual (Unsplash Free License)
│       ├── empty_state.svg            # Custom vector empty state illustration
│       └── error_state.svg            # Custom vector error state illustration
│
└── outputs/
    ├── clustered_indian_rentals.csv   # Exported dataset with Cluster and Segment Name
    └── segment_profiles.csv           # Comparative cross-segment profile table
```

---

## 🎨 Asset Attribution & Licensing

Decorative imagery used strictly for contextual website atmosphere (never attached to individual dataset records):
- `hero_architecture.jpg`: Modern residential architectural exterior by Unsplash (Unsplash Free License).
- `explore_header.jpg`: Contemporary residential facade by Unsplash (Unsplash Free License).
- `property_abstract.jpg`: Architectural minimalist lines by Unsplash (Unsplash Free License).
- `empty_state.svg` & `error_state.svg`: Original vector SVGs created for this application.
- `OpenStreetMap`: Cartography tiles provided by OpenStreetMap contributors (Open Database License, zero API key dependency).

---

## 🚀 Running the Application

### 1. Environment Setup
```powershell
cd C:\Users\Varsha\.gemini\antigravity\scratch\house_clustering
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Verify Backend Pipeline & Automated Test Suite
```powershell
# Run the terminal clustering pipeline:
.\.venv\Scripts\python.exe main.py

# Run the 9-stage system verification suite:
.\.venv\Scripts\python.exe test_system.py

# Run the HTTP endpoints test suite:
.\.venv\Scripts\python.exe test_server.py
```

### 3. Launch the Web Application
```powershell
.\.venv\Scripts\python.exe server.py
```
Open your web browser to:
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**
