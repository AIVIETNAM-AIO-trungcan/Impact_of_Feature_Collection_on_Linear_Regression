from src.data_loader import load_raw_data
from src.preprocess import preprocess_and_save

def run_pipeline():
    try:
        # Step 1: Load raw data
        df_raw = load_raw_data("data.csv")
        print(f"[Info] Raw data shape: {df_raw.shape}")
        
        # Step 2: Preprocess and save data
        df_processed = preprocess_and_save(df_raw, "processed_data.csv")
        print(f"[Info] Processed data shape: {df_processed.shape}")
        
    except FileNotFoundError as e:
        print(f"[Error] {e}")

if __name__ == "__main__":
    run_pipeline()