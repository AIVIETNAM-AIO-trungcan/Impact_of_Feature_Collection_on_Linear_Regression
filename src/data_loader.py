# =====================================================================
# MODULE: Data Loader
# DESCRIPTION: Handles the ingestion of raw data from the configured
#              data directories into pandas DataFrames.
# =====================================================================

import pandas as pd
from src.config import RAW_DATA_DIR


# ---------------------------------------------------------
# 1. DATA INGESTION FUNCTIONS
# ---------------------------------------------------------
def load_raw_data(file_name="data.csv"):
    """
    Loads raw data from the designated raw data directory.

    Args:
        file_name (str): The name of the CSV file to load. Defaults to "data.csv".

    Returns:
        pd.DataFrame: A pandas DataFrame containing the loaded data.

    Raises:
        FileNotFoundError: If the specified file does not exist in the raw data directory.
    """
    file_path = RAW_DATA_DIR / file_name

    if not file_path.exists():
        print(f"      [!] ERROR: Data file not found at {file_path}")
        raise FileNotFoundError(f"Data file not found at: {file_path}")

    print(f"      [~] Reading CSV data from: {file_path.name}...")
    df = pd.read_csv(file_path)
    print(
        f"      [>] Successfully loaded {df.shape[0]} rows and {df.shape[1]} columns."
    )

    return df
