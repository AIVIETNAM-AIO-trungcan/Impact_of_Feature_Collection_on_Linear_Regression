# =====================================================================
# MODULE: Data Splitter
# DESCRIPTION: Splits processed data into Train and Test sets to
#              prevent Data Leakage and saves them physically.
# =====================================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import PROCESSED_DATA_DIR


# ---------------------------------------------------------
# 1. DATA SPLITTING FUNCTIONS
# ---------------------------------------------------------
def split_and_save_data(
    input_file="processed_data.csv",
    train_file="train.csv",
    test_file="test.csv",
    test_size=0.2,
):
    """
    Splits data into Train and Test sets and saves them to the processed directory.

    Args:
        input_file (str): Name of the input CSV file. Defaults to "processed_data.csv".
        train_file (str): Name of the output train CSV file. Defaults to "train.csv".
        test_file (str): Name of the output test CSV file. Defaults to "test.csv".
        test_size (float): Proportion of the dataset to include in the test split. Defaults to 0.2.

    Returns:
        tuple: (train_df, test_df) containing the split pandas DataFrames.
    """
    print("      [~] Starting data physical split (Train/Test)...")

    # Load processed data
    input_path = PROCESSED_DATA_DIR / input_file
    df = pd.read_csv(input_path)

    # Perform physical split with a fixed random state for reproducibility
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

    # Define output paths
    train_path = PROCESSED_DATA_DIR / train_file
    test_path = PROCESSED_DATA_DIR / test_file

    # Save Train/Test data physically
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"      [>] Train data saved at: {train_path.name} ({len(train_df)} rows)")
    print(f"      [>] Test data saved at: {test_path.name} ({len(test_df)} rows)")

    return train_df, test_df
