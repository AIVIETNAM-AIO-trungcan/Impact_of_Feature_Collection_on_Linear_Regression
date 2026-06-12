from src.data_loader import load_raw_data
from src.preprocess import preprocess_and_save
from src.data_splitter import split_and_save_data
from src.model import train_baseline_model
from src.report import generate_benchmark_report

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
        # STAGE 2: MODEL PIPELINE (Assigned to Hoang)
        # ---------------------------------------------------------
        print("\n[STAGE 2] Baseline Model Training...")
        # Assuming 'price' is the target column to predict
        model, metrics = train_baseline_model(df_train, df_test, target_column="price")
        
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
    run_pipeline()