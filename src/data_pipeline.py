"""
DATA PREPARATION PIPELINE — Indian Rental Housing Inventory
Unifies regional datasets for Delhi, Mumbai, and Pune into a reproducible,
production-ready processed dataset.
Preserves all original raw CSV files in data/makaan_raw/ unaltered.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

def run_data_pipeline(
    raw_dir: Path,
    output_path: Path
) -> pd.DataFrame:
    """
    Executes the end-to-end data preparation workflow:
    1. Loads raw regional CSVs for Delhi, Mumbai, and Pune
    2. Drops exact row duplicates
    3. Parses built-up area to numeric sqft
    4. Extracts bedroom count (BHK) from property title
    5. Calculates price_per_sqft = price / sqft
    6. Encodes furnishing tier (0: Unfurnished, 1: Semi-Furnished, 2: Furnished)
    7. Median-imputes missing bathroom values
    8. Preserves original city and creates validated city_clean
    9. Flags invalid/suspicious geographic coordinates for map filtering
    10. Saves processed dataset to output_path
    """
    raw_files = [
        "Indian_housing_Delhi_data.csv",
        "Indian_housing_Mumbai_data.csv",
        "Indian_housing_Pune_data.csv"
    ]

    dfs = []
    for f in raw_files:
        file_path = raw_dir / f
        if not file_path.exists():
            raise FileNotFoundError(f"Required raw dataset not found: {file_path}")
        d = pd.read_csv(file_path)
        d["source_file"] = f
        dfs.append(d)

    raw_df = pd.concat(dfs, ignore_index=True)
    raw_count = len(raw_df)

    # 1. Deduplication: drop exact duplicates across the 16 original columns
    original_cols = [c for c in raw_df.columns if c != "source_file"]
    df = raw_df.drop_duplicates(subset=original_cols).copy().reset_index(drop=True)
    dedup_count = len(df)
    dropped_dups = raw_count - dedup_count

    # Create a stable integer ID for each property
    df["property_id"] = np.arange(1, len(df) + 1)

    # 2. Area parsing: house_size -> sqft
    # Extracts numeric digits after removing commas
    df["sqft"] = (
        df["house_size"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.extract(r"(\d+(?:\.\d+)?)")[0]
        .astype(float)
    )

    # 3. BHK extraction: house_type -> bhk
    # Standard BHK pattern: e.g., '2 BHK Apartment' -> 2.0
    df["bhk"] = (
        df["house_type"]
        .astype(str)
        .str.extract(r"(\d+)\s*(?:BHK|bhk)")[0]
        .astype(float)
    )
    # Studio / 1 RK units represent single-room living units -> 1.0 BHK equivalent
    rk_mask = df["house_type"].astype(str).str.contains(r"RK|Studio", case=False, regex=True)
    df.loc[rk_mask, "bhk"] = 1.0
    # Any remaining nulls filled with median
    bhk_median = df["bhk"].median()
    df["bhk"] = df["bhk"].fillna(bhk_median)

    # 4. price_per_sqft calculation: price / sqft
    # Guard against zero or null sqft (all sqft >= 150 in dataset)
    df["price_per_sqft"] = (df["price"] / df["sqft"]).round(2)

    # 5. Furnishing tier encoding: Status -> furnishing_tier
    # 0 = Unfurnished, 1 = Semi-Furnished, 2 = Furnished
    status_map = {
        "Unfurnished": 0,
        "Semi-Furnished": 1,
        "Furnished": 2
    }
    df["furnishing_tier"] = df["Status"].map(status_map).fillna(0).astype(int)

    # 6. Bathrooms imputation
    bath_median = float(df["numBathrooms"].median())
    df["numBathrooms_was_missing"] = df["numBathrooms"].isna().astype(int)
    df["numBathrooms"] = df["numBathrooms"].fillna(bath_median)

    # 7. City cleaning: preserve original city, create city_clean
    df["city_clean"] = df["city"].copy()
    # Handle the 8 Mumbai suburban listings marked 'Hisar'
    hisar_mumbai_mask = (df["source_file"] == "Indian_housing_Mumbai_data.csv") & (df["city"] == "Hisar")
    df.loc[hisar_mumbai_mask, "city_clean"] = "Mumbai"

    # 8. Property typology extraction
    def extract_typology(ht: str) -> str:
        s = str(ht).lower()
        if "villa" in s:
            return "Villa"
        elif "independent floor" in s or "builder floor" in s:
            return "Independent Floor"
        elif "independent house" in s:
            return "Independent House"
        elif "penthouse" in s:
            return "Penthouse"
        elif "studio" in s or "rk" in s:
            return "Studio / 1 RK"
        elif "apartment" in s or "flat" in s:
            return "Apartment"
        return "Apartment"

    df["typology"] = df["house_type"].apply(extract_typology)

    # 9. Clean SecurityDeposit
    def parse_security_deposit(val) -> float:
        if pd.isna(val) or str(val).strip() == "No Deposit":
            return 0.0
        cleaned = str(val).strip().replace(",", "")
        try:
            return float(cleaned)
        except ValueError:
            return 0.0

    df["deposit_clean"] = df["SecurityDeposit"].apply(parse_security_deposit)

    # 10. Geographic coordinate validation flag
    # Valid coordinates for India: Lat [8.0, 37.0], Lon [68.0, 97.0]
    # Metropolitan bounds:
    # Delhi NCR: Lat [28.0, 29.2], Lon [76.5, 77.8]
    # Mumbai MMR: Lat [18.5, 19.8], Lon [72.6, 73.5]
    # Pune: Lat [18.2, 19.0], Lon [73.5, 74.3]
    def validate_coordinates(row) -> bool:
        lat = row["latitude"]
        lon = row["longitude"]
        city = row["city_clean"]

        if pd.isna(lat) or pd.isna(lon):
            return False
        # Filter copy-paste bug (lat == lon)
        if abs(lat - lon) < 0.001:
            return False
        # Geographic India box
        if lat < 8.0 or lat > 37.0 or lon < 68.0 or lon > 97.0:
            return False

        # City regional bounding check
        if city == "Delhi":
            return (28.0 <= lat <= 29.2) and (76.5 <= lon <= 77.8)
        elif city == "Mumbai":
            return (18.5 <= lat <= 19.8) and (72.6 <= lon <= 73.5)
        elif city == "Pune":
            return (18.2 <= lat <= 19.0) and (73.5 <= lon <= 74.3)
        return True

    df["is_valid_coord"] = df.apply(validate_coordinates, axis=1)

    # Save processed unified dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print("================================================================================")
    print("DATA PIPELINE COMPLETE")
    print("================================================================================")
    print(f"Raw Input Rows:            {raw_count:,}")
    print(f"Exact Duplicates Removed:  {dropped_dups:,}")
    print(f"Final Processed Rows:      {len(df):,}")
    print(f"Columns:                   {len(df.columns)}")
    print(f"Valid Map Coordinates:     {df['is_valid_coord'].sum():,} ({df['is_valid_coord'].mean()*100:.2f}%)")
    print(f"Output File:               {output_path}")
    print("================================================================================")

    return df

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    raw_dir = base_dir / "data" / "makaan_raw"
    output_csv = base_dir / "data" / "processed_indian_rentals.csv"
    run_data_pipeline(raw_dir, output_csv)
