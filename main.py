import pandas as pd
import numpy as np
import yaml
import mlflow
from src.data_loader import load_raw_data
from src.preprocess import preprocess_and_save
from src.data_splitter import split_and_save_data
from src.processor import fit_transform_features
from src.processor import fit_transform_features, apply_feature_transform
from src.feature_selector_apply import run_feature_selection
from src.model import train_dynamic_model
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
        X_train_transformed = apply_feature_transform(
            X_train_clean, cfg, dataset_name="Train"
        )
        X_test_transformed = apply_feature_transform(
            X_test_clean, cfg, dataset_name="Test"
        )

        y_train = df_train["amount"].copy()
        y_test = df_test["amount"].copy()

        transform_mode = cfg.get("experiment", {}).get("target_transform", "raw")
        is_log = transform_mode == "log"

        if is_log:
            y_train_transformed = np.log1p(y_train.astype(float))
            y_test_transformed = np.log1p(y_test.astype(float))
            print("   -> [Processor] Target: Applied Log Transform (np.log1p).")
        else:
            y_train_transformed = y_train
            y_test_transformed = y_test
            print("   -> [Processor] Target: RAW mode active.")

        df_train_final = pd.concat([X_train_transformed, y_train_transformed], axis=1)
        df_test_final = pd.concat([X_test_transformed, y_test_transformed], axis=1)

        # =========================================================
        # INITIATING EXPERIMENT LOOP
        # =========================================================
        print("\n" + "=" * 60)
        print("🔬 INITIATING EXPERIMENT LOOP")
        print("=" * 60)

        mlflow.set_experiment("House_Price_Feature_Selection_Experiment")

        experiment_methods = ["filter", "forward", "backward", "lasso", "best_subset"]
        top_k = cfg.get("experiment", {}).get("top_k_features", 15)

        for method in experiment_methods:
            run_name = f"Selection_{method.upper()}"

            with mlflow.start_run(run_name=run_name):
                print(f"\n{'='*40}\n🔹 RUNNING: {run_name}\n{'='*40}")

                # ---------------------------------------------------------
                # STAGE 2: MODEL PIPELINE (Gatekeeper & Training)
                # ---------------------------------------------------------
                print(
                    f"\n[STAGE 2] Running Gatekeeper & Model Training for {method.upper()}..."
                )

                # A. Log configuration parameters to MLflow
                mlflow.log_param("selection_method", method)
                mlflow.log_param("target_transform", transform_mode)
                if method in ["filter", "best_subset"]:
                    mlflow.log_param("top_k_limit", top_k)

                # B. Gatekeeper: Execute Feature Selection logic
                selected_features = run_feature_selection(
                    df_train_final,
                    target_column="amount",
                    method=method,
                    criterion="aic",
                    top_k=top_k,
                )

                # C. Training Engine: Train model dynamically
                model_filename = f"model_{method}.pkl"
                model, metrics = train_dynamic_model(
                    df_train_final,
                    df_test_final,
                    config=cfg,
                    selected_features=selected_features,
                    target_column="amount",
                    model_name=model_filename,
                )

                # ---------------------------------------------------------
                # STAGE 3: REPORTING & METRICS VALIDATION
                # ---------------------------------------------------------
                print(
                    f"\n[STAGE 3] Generating Advanced Benchmark Report for {method.upper()}..."
                )

                # A. Setup report configuration
                selector_context = {"method": method}
                report_filename = f"report_{method}.json"

                # B. Generate JSON report matching Hoang's standard
                report_data = generate_benchmark_report(
                    metrics=metrics,
                    output_file=report_filename,
                    is_log_transformed=is_log,
                    selector_context=selector_context,
                )

                # C. Push metrics to MLflow Tracking Server
                print("   -> [MLflow] Pushing metrics to tracking server...")
                mlflow.log_metric("Test_R2", report_data["Test_R2"])
                mlflow.log_metric("Test_RMSE", report_data["Test_RMSE"])
                mlflow.log_metric("Test_MAE", report_data["Test_MAE"])

                if report_data["Train_AIC"] is not None:
                    mlflow.log_metric("Train_AIC", report_data["Train_AIC"])
                    mlflow.log_metric("Train_BIC", report_data["Train_BIC"])
                    mlflow.log_metric("Train_Adj_R2", report_data["Train_Adj_R2"])

        print("\n" + "=" * 60)
        print("✅ ALL EXPERIMENTS EXECUTED SUCCESSFULLY!")
        print(
            "📂 Check 'reports/' for JSON files and MLflow UI for experiment tracking."
        )
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n❌ [ERROR] Pipeline failed at execution: {e}\n")


if __name__ == "__main__":
    run_pipeline()
