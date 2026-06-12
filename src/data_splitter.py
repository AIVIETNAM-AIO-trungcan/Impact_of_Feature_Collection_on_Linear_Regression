import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import PROCESSED_DATA_DIR

def split_and_save_data(input_file="processed_data.csv", train_file="train.csv", test_file="test.csv", test_size=0.2):
    """Split data into Train and Test sets to prevent Data Leakage."""
    print("[-] Starting data split (Train/Test)...")
    
    # Load processed data
    df = pd.read_csv(PROCESSED_DATA_DIR / input_file)
    
    # Perform physical split with a fixed random state for reproducibility
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    
    # Define output paths
    train_path = PROCESSED_DATA_DIR / train_file
    test_path = PROCESSED_DATA_DIR / test_file
    
    # Save Train/Test data
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"[-] Train data saved at: {train_path} ({len(train_df)} rows)")
    print(f"[-] Test data saved at: {test_path} ({len(test_df)} rows)")
    
    return train_df, test_df