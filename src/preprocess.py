import pandas as pd
from src.config import PROCESSED_DATA_DIR

def preprocess_and_save(df, output_file_name="processed_data.csv"):
    """Clean data and save it to the data/processed/ directory."""
    print("[-] Starting data preprocessing...")
    
    df_clean = df.copy()
    
    # =========================================================================
    # [TODO - TUNG NGUYEN] 
    # =========================================================================
    
    # ---> CODING HERE <---
    # Write your data preprocessing logic below (drop NaN, scale, encode, etc.)
    
    
    
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
