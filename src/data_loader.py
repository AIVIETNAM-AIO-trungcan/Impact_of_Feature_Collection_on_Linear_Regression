import pandas as pd
from src.config import RAW_DATA_DIR

def load_raw_data(file_name="data.csv"):
    """Load data from the data/raw/ directory."""
    file_path = RAW_DATA_DIR / file_name
    
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found at: {file_path}")
        
    print(f"[-] Loading data from: {file_path}")
    df = pd.read_csv(file_path)
    return df
