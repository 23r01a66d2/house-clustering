"""
Configuration file for Indian Rental Housing Intelligence System.
Centralizes paths, hyperparameters, column roles, and formatting.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "makaan_raw"
OUTPUTS_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"

# Dataset Files
PROCESSED_DATASET_NAME = "processed_indian_rentals.csv"
RAW_DATA_PATH = DATA_DIR / PROCESSED_DATASET_NAME
CLUSTERED_DATA_PATH = OUTPUTS_DIR / "clustered_indian_rentals.csv"

# Regional Raw Files
RAW_FILES = [
    "Indian_housing_Delhi_data.csv",
    "Indian_housing_Mumbai_data.csv",
    "Indian_housing_Pune_data.csv"
]

# Currency & Terminology
CURRENCY_SYMBOL = "₹"
CURRENCY_CODE = "INR"
PRICE_LABEL = "Monthly Rent"

# Clustering Features (Scaled Similarity Space)
# Note: price_per_sqft is excluded from direct clustering to avoid multi-collinear redundancy with price/sqft
CLUSTERING_FEATURES = ["price", "sqft", "bhk", "numBathrooms", "furnishing_tier"]
LOG_TRANSFORM_FEATURES = ["price", "sqft"]

# Clustering Hyperparameters
RANDOM_STATE = 42
K_MIN = 2
K_MAX = 10
K_FALLBACK = 5  # Fallback only; optimal K is discovered dynamically via KDiscoveryLab
K_DEFAULT = K_FALLBACK
N_INIT = 10
MAX_ITER = 300

# Canonical Column Roles in Processed Dataset
FEATURE_ROLES = {
    "price": ["price"],
    "living_area": ["sqft"],
    "bedrooms": ["bhk"],
    "bathrooms": ["numBathrooms"],
    "furnishing": ["furnishing_tier"],
    "price_per_sqft": ["price_per_sqft"],
    "city": ["city_clean", "city"],
    "location": ["location"],
    "typology": ["typology"],
    "latitude": ["latitude"],
    "longitude": ["longitude"],
    "valid_coords": ["is_valid_coord"]
}

# Display / Non-Clustering Metadata Columns
METADATA_COLUMNS = [
    "property_id",
    "house_type",
    "house_size",
    "location",
    "city",
    "city_clean",
    "typology",
    "Status",
    "SecurityDeposit",
    "deposit_clean",
    "price_per_sqft",
    "latitude",
    "longitude",
    "is_valid_coord"
]

# Plot Styling
FIGURE_DPI = 300
PALETTE = "viridis"
SEABORN_STYLE = "whitegrid"
