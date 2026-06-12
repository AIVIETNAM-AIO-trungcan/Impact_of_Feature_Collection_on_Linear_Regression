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
    import pandas as pd

    # Standardize column names
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

    # Drop unnecessary columns
    cols_drop = [
        "index",
        "title",
        "description",
        "status",
        "society",
        "floor",
        "dimensions",
        "plot_area"
    ]

    df_clean = df_clean.drop(
        columns=cols_drop
    )

    # Area conversion
    for col in ["carpet_area", "super_area"]:

        df_clean[col] = (
            df_clean[col]
            .astype(str)
            .str.extract(r"([\d.]+)")
            .astype(float)
        )


    # Amount conversion
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


    df_clean["amount"] = (
        df_clean["amount"]
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


    for col in ["bathroom", "balcony", "car_parking"]:

        df_clean[col] = (
            df_clean[col]
            .apply(extract_numeric_feature)
        )


    # Remove rows with missing target
    df_clean = df_clean.dropna(
        subset=["price"]
    )


    # # Median imputation
    # median_cols = [
    #     "amount",
    #     "carpet_area",
    #     "super_area",
    #     "bathroom"
    # ]

    # for col in median_cols:

    #     df_clean[col] = (
    #         df_clean[col]
    #         .fillna(df_clean[col].median())
    #     )


    # # Fill zero
    # zero_fill_cols = [
    #     "balcony",
    #     "car_parking"
    # ]

    # for col in zero_fill_cols:

    #     df_clean[col] = (
    #         df_clean[col]
    #         .fillna(0)
    #     )


    # # Fill categorical missing
    # categorical_cols = [
    #     "location",
    #     "transaction",
    #     "furnishing",
    #     "facing",
    #     "overlooking",
    #     "ownership"
    # ]

    # for col in categorical_cols:

    #     df_clean[col] = (
    #         df_clean[col]
    #         .fillna("Unknown")
    #     )


    # # Clip Car Parking outliers
    # car_parking_cap = (
    #     df_clean["car_parking"]
    #     .quantile(0.99)
    # )

    # df_clean["car_parking"] = (
    #     df_clean["car_parking"]
    #     .clip(upper=car_parking_cap)
    # )
    
    # =========================================================================
    # END OF PREPROCESSING LOGIC
    # =========================================================================

    # Reset index after cleaning
    df_clean.reset_index(drop=True, inplace=True)
    
    # Define output routing and save the file
    output_path = PROCESSED_DATA_DIR / output_file_name
    df_clean.to_csv(output_path, index=False)
    
    print(f"[-] Clean data saved at: {output_path}")

    # print("\n DANH SÁCH CÁC CỘT ĐANG CHỨA CHỮ (Gây lỗi Model):")                        #Check error 
    # cols_str = df_clean.select_dtypes(include=['object', 'string']).columns.tolist()    #Check error
    # print(cols_str)                                                                     #Check error

    # df_clean = df_clean.drop(columns=cols_str)                                          #Check error

    return df_clean
