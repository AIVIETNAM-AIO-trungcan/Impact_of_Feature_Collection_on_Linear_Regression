import re
import pandas as pd
import numpy as np
from src.config import PROCESSED_DATA_DIR


def preprocess_and_save(df, output_file_name="processed_data.csv"):
    """Clean data and save it to the data/processed/ directory."""
    print("[-] Starting data preprocessing...")

    df_clean = df.copy()

    # ==================================================
    # Standardize Column Names
    # ==================================================

    df_clean.columns = [
        re.sub(
            r"_+",
            "_",
            re.sub(r"[^a-z0-9_]", "_", col.lower().replace("(in rupees)", "").strip()),
        ).strip("_")
        for col in df_clean.columns
    ]

    # ==================================================
    # Drop Unused Columns
    # ==================================================

    df_clean = df_clean.drop(
        columns=[
            "index",
            "title",
            "description",
            "status",
            "society",
            "plot_area",
            "dimensions",
            "price",
            "car_parking",
        ],
        errors="ignore",
    )

    # ==================================================
    # Helper Functions
    # ==================================================

    def convert_amount(value):

        if pd.isna(value):
            return np.nan

        value = str(value).strip()
        match = re.search(r"([\d.]+)", value)

        if not match:
            return np.nan

        number = float(match.group(1))
        value = value.lower()

        if "lac" in value:
            return number * 100000

        if "cr" in value:
            return number * 10000000

        return number

    def extract_count(value):

        if pd.isna(value):
            return np.nan

        match = re.search(r"(\d+)", str(value))

        if not match:
            return np.nan

        count = int(match.group(1))

        return count + 1 if ">" in str(value) else count

    # ==================================================
    # Convert Target Variable
    # ==================================================

    df_clean["amount"] = df_clean["amount"].apply(convert_amount)

    # ==================================================
    # Convert Count Features
    # ==================================================

    for col in ["bathroom", "balcony"]:

        if col in df_clean.columns:
            df_clean[col] = df_clean[col].apply(extract_count).astype("Int64")

    # ==================================================
    # Merge Carpet Area & Super Area
    # ==================================================

    for col in ["carpet_area", "super_area"]:

        if col in df_clean.columns:
            df_clean[col] = (
                df_clean[col].astype(str).str.extract(r"([\d.]+)").astype(float)
            )

    df_clean["area"] = df_clean["carpet_area"].fillna(df_clean["super_area"])

    df_clean = df_clean.drop(columns=["carpet_area", "super_area"], errors="ignore")

    # ==================================================
    # Extract Room Floor & Total Floor
    # ==================================================

    if "floor" in df_clean.columns:

        floor_info = (
            df_clean["floor"].astype(str).str.extract(r"(\d+)\s+out\s+of\s+(\d+)")
        )

        df_clean["room_floor"] = pd.to_numeric(floor_info[0], errors="coerce")
        df_clean["total_floor"] = pd.to_numeric(floor_info[1], errors="coerce")

        df_clean = df_clean.drop(columns=["floor"], errors="ignore")

    # ==================================================
    # Transaction Processing
    # ==================================================

    if "transaction" in df_clean.columns:

        df_clean = df_clean[df_clean["transaction"] != "Rent/Lease"]

        df_clean["transaction"] = df_clean["transaction"].replace("Other", np.nan)

    # ==================================================
    # Overlooking Processing
    # ==================================================

    if "overlooking" in df_clean.columns:

        overlooking = df_clean["overlooking"].fillna("").str.lower()

        df_clean["main_road_view"] = overlooking.str.contains("main road").astype(
            "Int64"
        )

        df_clean["amenity_view"] = (
            overlooking.str.contains("garden/park") | overlooking.str.contains("pool")
        ).astype("Int64")

        df_clean = df_clean.drop(columns=["overlooking"])

    # ==================================================
    # Remove Missing Target
    # ==================================================

    df_clean = df_clean.dropna(subset=["amount"])

    # ==================================================
    # Final Feature Selection
    # ==================================================

    ordered_columns = [
        "amount",
        "area",
        "bathroom",
        "balcony",
        "room_floor",
        "total_floor",
        "transaction",
        "furnishing",
        "facing",
        "location",
        "ownership",
        "main_road_view",
        "amenity_view",
    ]

    df_clean = df_clean[[c for c in ordered_columns if c in df_clean.columns]]

    # =========================================================================
    # END OF PREPROCESSING LOGIC
    # =========================================================================

    # Reset index after cleaning
    df_clean.reset_index(drop=True, inplace=True)

    # Define output routing and save the file
    output_path = PROCESSED_DATA_DIR / output_file_name
    df_clean.to_csv(output_path, index=False)

    print(f"[-] Clean data saved at: {output_path}")

    return df_clean
