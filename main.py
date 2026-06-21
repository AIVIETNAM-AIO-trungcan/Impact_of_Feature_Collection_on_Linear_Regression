from src.data_loader import load_raw_data
from src.preprocess import preprocess_and_save
from src.data_splitter import split_and_save_data
from src.processor import fit_transform_features
from src.model import train_baseline_model
from src.report import generate_benchmark_report
from src.config import load_pipeline_config

def run_pipeline():
    try:
        print("\n" + "="*40)
        print("🚀 STARTING ML PIPELINE")
        print("="*40)
        
        # ---------------------------------------------------------
        # STAGE 1: DATA PIPELINE (Assigned to Tung Nguyen)
        # ---------------------------------------------------------
        print("\n[STAGE 1] Data Loading & Preprocessing...")
        df_raw = load_raw_data("data.csv")
        df_processed = preprocess_and_save(df_raw, "processed_data.csv")
        
        # ---------------------------------------------------------
        # STAGE 1.5: PHYSICAL SPLIT (Train/Test)
        # ---------------------------------------------------------
        print("\n[STAGE 1.5] Train-Test Physical Split...")
        df_train, df_test = split_and_save_data(
            input_file="processed_data.csv",
            train_file="train.csv",
            test_file="test.csv"
        )

        # ---------------------------------------------------------
        # STAGE 1.8: ADVANCED FEATURE PROCESSING (Can)
        # ---------------------------------------------------------
        print("\n[STAGE 1.8] Advanced Feature Processing...")
        cfg = load_pipeline_config()
        num_cols = cfg['features']['numerical_cols']
        cat_cols = cfg['features']['categorical_cols']
        
        # Separate features (X) for processing
        X_train_raw = df_train.drop(columns=['amount'])
        X_test_raw = df_test.drop(columns=['amount'])
        
        # Clean and synchronize features (Fit on Train, Transform Test to avoid leakage)
        X_train_clean, X_test_clean = fit_transform_features(
            X_train_raw, X_test_raw, num_cols, cat_cols
        )
        
        # Recombine with target variable for model training
        df_train_final = pd.concat([X_train_clean, df_train['amount']], axis=1)
        df_test_final = pd.concat([X_test_clean, df_test['amount']], axis=1)

        # ---------------------------------------------------------
        # STAGE 2: MODEL PIPELINE (Assigned to Hoang)
        # ---------------------------------------------------------
        print("\n[STAGE 2] Baseline Model Training...")
        # Assuming 'amount' is the target column to predict
        model, metrics = train_baseline_model(df_train_final, df_test_final, target_column="amount")
        
        # ---------------------------------------------------------
        # STAGE 3: REPORTING (System Auto-generated)
        # ---------------------------------------------------------
        print("\n[STAGE 3] Generating Reports...")
        generate_benchmark_report(metrics, "baseline_report.json")
        
        print("\n" + "="*40)
        print("✅ PIPELINE EXECUTED SUCCESSFULLY!")
        print("="*40 + "\n")
        
    except Exception as e:
        print(f"\n❌ [ERROR] Pipeline failed at execution: {e}\n")

if __name__ == "__main__":
    import pandas as pd
    run_pipeline()