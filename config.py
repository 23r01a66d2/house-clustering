"""
Configuration file for House Clustering System.
Centralizes paths, hyperparameters, column mappings, and styling.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"

# Dataset Files
DEFAULT_DATASET_NAME = "house_price_india.csv"
RAW_DATA_PATH = DATA_DIR / DEFAULT_DATASET_NAME
CLUSTERED_DATA_PATH = OUTPUTS_DIR / "clustered_house_data.csv"

# Clustering Hyperparameters
RANDOM_STATE = 42
K_MIN = 2
K_MAX = 10
N_INIT = 10
MAX_ITER = 300

# Identifier & Irrelevant Columns to Exclude from Clustering
IDENTIFIER_KEYWORDS = ["id", "date", "postal", "zip", "serial", "index"]

# Key Feature Keywords for Dynamic Profiling and Auto-Detection
FEATURE_ROLES = {
    "price": ["price"],
    "living_area": ["living area", "sqft_living", "area of the house", "living_area"],
    "lot_area": ["lot area", "sqft_lot", "lot_area"],
    "bedrooms": ["number of bedrooms", "bedrooms", "bedroom", "beds"],
    "bathrooms": ["number of bathrooms", "bathrooms", "bathroom", "baths"],
    "floors": ["number of floors", "floors", "stories"],
    "grade": ["grade of the house", "grade"],
    "condition": ["condition of the house", "condition"],
    "built_year": ["built year", "yr_built", "year built"],
    "renovation_year": ["renovation year", "yr_renovated", "renovation"],
    "waterfront": ["waterfront present", "waterfront"],
    "views": ["number of views", "views", "view"],
    "schools": ["number of schools nearby", "schools"],
    "airport_dist": ["distance from the airport", "airport"],
}

# Outlier Thresholds for Cleaning
BEDROOM_MAX_THRESHOLD = 20  # Identifies anomalies like 33 bedrooms

# Plot Styling
FIGURE_DPI = 300
PALETTE = "viridis"
SEABORN_STYLE = "whitegrid"
