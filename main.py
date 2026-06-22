import pandas as pd
import numpy as np
from src.data_loader import load_raw_data
from src.preprocess import preprocess_and_save
from src.data_splitter import split_and_save_data
from src.processor import fit_transform_features
from src.processor import fit_transform_features, apply_feature_transform
from src.feature_selector import select_features
from src.model import train_baseline_model
from src.report import generate_benchmark_report
from src.config import load_pipeline_config


def run_pipeline():
    try:
        print("\n" + "=" * 40)
        print("🚀 STARTING ML PIPELINE")
        print("=" * 40)

        # ---------------------------------------------------------
        # STAGE 1: DATA PIPELINE
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
            test_file="test.csv",
        )

        # ---------------------------------------------------------
        # STAGE 1.8: ADVANCED FEATURE PROCESSING
        # ---------------------------------------------------------
        print("\n[STAGE 1.8] Advanced Feature Processing...")
        cfg = load_pipeline_config()
        num_cols = cfg["features"]["numerical_cols"]
        cat_cols = cfg["features"]["categorical_cols"]

        # Separate features (X) for processing
        X_train_raw = df_train.drop(columns=["amount"])
        X_test_raw = df_test.drop(columns=["amount"])

        # Clean and synchronize features (Fit on Train, Transform Test to avoid leakage)
        X_train_clean, X_test_clean = fit_transform_features(
            X_train_raw, X_test_raw, num_cols, cat_cols
        )

        # # Recombine with target variable for model training
        # df_train_final = pd.concat([X_train_clean, df_train["amount"]], axis=1)
        # df_test_final = pd.concat([X_test_clean, df_test["amount"]], axis=1)

        # ---------------------------------------------------------
        # STAGE 1.8.5: GATEKEEPER (FEATURE SELECTION)
        # ---------------------------------------------------------
        # Purpose:
        #   - Perform univariate feature selection to mitigate
        #     the curse of dimensionality.
        # Input:
        #   - X_train_clean: Preprocessed features [n_samples, n_features]
        #   - y_train_raw: Target variable used to calculate F-scores
        # Process:
        #   - Invoke 'select_features' to retain only Top K features
        #     defined in config.yaml.
        # Output:
        #   - X_train_selected: Filtered features [n_samples, k]
        #   - X_test_selected: Filtered features [n_samples, k]
        # ---------------------------------------------------------
        use_selector = cfg.get("experiment", {}).get("use_feature_selector", False)

        if use_selector:
            print("\n[STAGE 1.8.5] Gatekeeper ENABLED: Running Feature Selection...")
            y_train_raw = df_train["amount"]

            X_train_selected, X_test_selected = select_features(
                X_train_clean, y_train_raw, X_test_clean, cfg
            )
        else:
            print(
                "\n[STAGE 1.8.5] Gatekeeper DISABLED: Running Baseline (All Features)..."
            )
            X_train_selected, X_test_selected = X_train_clean, X_test_clean

        # ---------------------------------------------------------
        # STAGE 1.9: DYNAMIC TRANSFORMATION (Log/Raw)
        # ---------------------------------------------------------
        # Purpose:
        #   - Apply log transformation to selected features if configured.
        # Input:
        #   - X_train_selected, X_test_selected: Filtered features [n_samples, k]
        # Output:
        #   - X_train_transformed, X_test_transformed: Transformed features
        # ---------------------------------------------------------
        print("\n[STAGE 1.9] Applying dynamic transformations...")

        # 1. Transform independent variables (X) based on config
        X_train_transformed = apply_feature_transform(
            X_train_selected, cfg, dataset_name="Train"
        )
        X_test_transformed = apply_feature_transform(
            X_test_selected, cfg, dataset_name="Test"
        )

        # 2. Transform target variable (y)
        y_train = df_train["amount"].copy()
        y_test = df_test["amount"].copy()

        transform_mode = cfg.get("experiment", {}).get("target_transform", "raw")

        if transform_mode == "log":
            # Cast to float and apply log1p
            y_train_transformed = np.log1p(y_train.astype(float))
            y_test_transformed = np.log1p(y_test.astype(float))
            print(
                "   -> [Processor] Target (y - Train/Test): Applied Log Transform to 'amount'."
            )
        else:
            y_train_transformed = y_train
            y_test_transformed = y_test
            print(
                "   -> [Processor] Target (y - Train/Test): RAW mode active (Keeping original actual prices)."
            )

        # Recombine with target variable for model training
        df_train_final = pd.concat([X_train_transformed, y_train_transformed], axis=1)
        df_test_final = pd.concat([X_test_transformed, y_test_transformed], axis=1)

        # ---------------------------------------------------------
        # STAGE 2: MODEL PIPELINE (Assigned to Hoang)
        # ---------------------------------------------------------
        print("\n[STAGE 2] Baseline Model Training...")
        # Assuming 'amount' is the target column to predict
        model, metrics = train_baseline_model(
            df_train_final, df_test_final, target_column="amount"
        )

        # ---------------------------------------------------------
        # STAGE 3: REPORTING (System Auto-generated)
        # ---------------------------------------------------------
        print("\n[STAGE 3] Generating Reports...")
        generate_benchmark_report(metrics, "baseline_report.json")

        print("\n" + "=" * 40)
        print("✅ PIPELINE EXECUTED SUCCESSFULLY!")
        print("=" * 40 + "\n")

    except Exception as e:
        print(f"\n❌ [ERROR] Pipeline failed at execution: {e}\n")


if __name__ == "__main__":
    run_pipeline()
