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


    # Drop unnecessary columns
    cols_drop = [
        "Index",
        "Title",
        "Description",
        "Status",
        "Society",
        "Floor"
    ]

    df_clean = df_clean.drop(
        columns=cols_drop,
        errors="ignore"
    )


    # Convert Area columns
    for col in ["Carpet Area", "Super Area"]:

        df_clean[col] = (
            df_clean[col]
            .astype(str)
            .str.extract(r"([\d.]+)")
            .astype(float)
        )


    # Convert Amount(in rupees)
    def convert_amount(x):

        if pd.isna(x):
            return np.nan

        x = str(x).strip()

        match = re.search(r"([\d.]+)", x)

        if match is None:
            return np.nan

        value = float(match.group(1))

        if "lac" in x.lower():
            return value * 1e5

        if "cr" in x.lower():
            return value * 1e7

        return value


    df_clean["Amount(in rupees)"] = (
        df_clean["Amount(in rupees)"]
        .apply(convert_amount)
    )


    # Convert Bathroom / Balcony / Car Parking
    def extract_numeric_feature(x):

        if pd.isna(x):
            return np.nan

        x = str(x).strip()

        match = re.search(r"(\d+)", x)

        if match is None:
            return np.nan

        value = int(match.group(1))

        if ">" in x:
            return value + 1

        return value


    for col in ["Bathroom", "Balcony", "Car Parking"]:

        df_clean[col] = (
            df_clean[col]
            .apply(extract_numeric_feature)
        )


    # Remove rows with missing target
    df_clean = df_clean.dropna(
        subset=["Price (in rupees)"]
    )


    # Median imputation
    median_cols = [
        "Amount(in rupees)",
        "Carpet Area",
        "Super Area",
        "Bathroom"
    ]

    for col in median_cols:

        df_clean[col] = (
            df_clean[col]
            .fillna(df_clean[col].median())
        )


    # Fill zero
    zero_fill_cols = [
        "Balcony",
        "Car Parking"
    ]

    for col in zero_fill_cols:

        df_clean[col] = (
            df_clean[col]
            .fillna(0)
        )


    # Fill categorical missing
    categorical_cols = [
        "location",
        "Transaction",
        "Furnishing",
        "facing",
        "overlooking",
        "Ownership"
    ]

    for col in categorical_cols:

        df_clean[col] = (
            df_clean[col]
            .fillna("Unknown")
        )


    # Clip Car Parking outliers
    car_parking_cap = (
        df_clean["Car Parking"]
        .quantile(0.99)
    )

    df_clean["Car Parking"] = (
        df_clean["Car Parking"]
        .clip(upper=car_parking_cap)
    )
    
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
