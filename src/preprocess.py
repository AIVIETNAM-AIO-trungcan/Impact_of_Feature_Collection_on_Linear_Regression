import pandas as pd
from src.config import PROCESSED_DATA_DIR

def preprocess_and_save(df, output_file_name="processed_data.csv"):
    """Clean data and save it to the data/processed/ directory."""
    print("[-] Starting data preprocessing...")
    
    df_clean = df.copy()
    
    # =========================================================================
    # [TODO - TUNG NGUYEN] 
    # =========================================================================
    
    import re
    import numpy as np

    # ==================================================
    # Standardize Column Names
    # ==================================================

    df_clean.columns = [
        re.sub(
            r"_+",
            "_",
            re.sub(
                r"[^a-z0-9_]",
                "_",
                col.lower()
                .replace("(in rupees)", "")
                .strip()
            )
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
            "status",
            "description",
            "society",
            "dimensions",
            "plot_area"
        ],
        errors="ignore"
    )

    # ==================================================
    # Amount Conversion
    # ==================================================

    def convert_amount(value):

        if pd.isna(value):
            return np.nan

        value = str(value).strip()

        match = re.search(
            r"([\d.]+)",
            value
        )

        if match is None:
            return np.nan

        number = float(
            match.group(1)
        )

        value_lower = value.lower()

        if "lac" in value_lower:
            return number * 1e5

        if "cr" in value_lower:
            return number * 1e7

        return number

    df_clean["amount"] = (
        df_clean["amount"]
        .apply(convert_amount)
    )

    # ==================================================
    # Bathroom / Balcony Conversion
    # ==================================================

    def convert_count_feature(value):

        if pd.isna(value):
            return np.nan

        value = str(value).strip()

        match = re.search(
            r"(\d+)",
            value
        )

        if match is None:
            return np.nan

        count = int(
            match.group(1)
        )

        if ">" in value:
            return count + 1

        return count

    for col in [
        "bathroom",
        "balcony"
    ]:

        df_clean[col] = (
            df_clean[col]
            .apply(convert_count_feature)
            .astype("Int64")
        )

    # ==================================================
    # Car Parking Conversion
    # ==================================================

    def convert_parking(value):

        if pd.isna(value):
            return np.nan

        value = str(value).lower()

        if "covered" in value:
            return "covered"

        return "open"

    df_clean["car_parking"] = (
        df_clean["car_parking"]
        .apply(convert_parking)
    )

    # ==================================================
    # Area Processing
    # ==================================================

    for col in [
        "carpet_area",
        "super_area"
    ]:

        df_clean[col] = (
            df_clean[col]
            .astype(str)
            .str.extract(
                r"([\d.]+)"
            )
            .astype(float)
        )

    overlap_count = (
        df_clean["carpet_area"]
        .notna()
        &
        df_clean["super_area"]
        .notna()
    ).sum()

    if overlap_count > 0:

        raise ValueError(
            f"Found {overlap_count} rows "
            "containing both carpet_area "
            "and super_area."
        )

    df_clean["area_sqft"] = (
        df_clean["carpet_area"]
        .fillna(
            df_clean["super_area"]
        )
    )

    df_clean["area_type"] = pd.Series(
        np.where(
            df_clean["carpet_area"].notna(),
            "carpet",
            np.where(
                df_clean["super_area"].notna(),
                "super",
                None
            )
        ),
        dtype="object"
    )

    df_clean = df_clean.drop(
        columns=[
            "carpet_area",
            "super_area"
        ]
    )

    # ==================================================
    # Floor Processing
    # ==================================================

    floor_split = (
        df_clean["floor"]
        .astype(str)
        .str.extract(
            r"(\d+)\s+out\s+of\s+(\d+)"
        )
    )

    df_clean["room_floor"] = pd.to_numeric(
        floor_split[0],
        errors="coerce"
    )

    df_clean["total_floor"] = pd.to_numeric(
        floor_split[1],
        errors="coerce"
    )

    df_clean = df_clean.drop(
        columns=["floor"],
        errors="ignore"
    )

    # ==================================================
    # Remove Missing Target
    # ==================================================

    df_clean = df_clean.dropna(
        subset=["price"]
    )

    # ==================================================
    # Reorder Columns
    # ==================================================

    ordered_columns = [

        # Target
        "price",

        # Numeric
        "amount",
        "area_sqft",
        "bathroom",
        "balcony",
        "room_floor",
        "total_floor",

        # Categorical
        "transaction",
        "furnishing",
        "facing",
        "location",
        "overlooking",
        "car_parking",
        "ownership",
        "area_type"
    ]

    existing_columns = [
        col
        for col in ordered_columns
        if col in df_clean.columns
    ]

    df_clean = df_clean[
        existing_columns
    ]
    
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
