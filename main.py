# =====================================================================
# MODULE: Main Execution Pipeline
# DESCRIPTION: Orchestrates data loading, preprocessing, feature
#              selection, model training, and reporting.
# =====================================================================

import pandas as pd
import numpy as np
import yaml
import mlflow
import time
import argparse
import sys
from pathlib import Path

from src.data_loader import load_raw_data
from src.preprocess import preprocess_and_save
from src.data_splitter import split_and_save_data
from src.processor import fit_transform_features, apply_feature_transform
from src.feature_selector_apply import run_feature_selection
from src.model import train_dynamic_model
from src.report import generate_benchmark_report, generate_leaderboard
from src.config import load_pipeline_config


def run_pipeline():
    """
    Executes the end-to-end Machine Learning pipeline.
    Includes data prep, dynamic transformations, feature selection experiments,
    and leaderboard generation.
    """
    try:
        print("\n" + "=" * 40)
        print("🚀 STARTING ML PIPELINE")
        print("=" * 40)

        # ---------------------------------------------------------
        # 1. DATA PIPELINE (Load & Preprocess)
        # ---------------------------------------------------------
        print("\n[STAGE 1] Data Loading & Preprocessing...")
        print("      [~] Loading raw data...")
        df_raw = load_raw_data("data.csv")

        print("      [~] Preprocessing and cleaning data...")
        df_processed = preprocess_and_save(df_raw, "processed_data.csv")

        print("\n[STAGE 1.5] Train-Test Physical Split...")
        df_train, df_test = split_and_save_data(
            input_file="processed_data.csv",
            train_file="train.csv",
            test_file="test.csv",
        )

        # ---------------------------------------------------------
        # 2. ADVANCED FEATURE PROCESSING
        # ---------------------------------------------------------
        print("\n[STAGE 1.8] Advanced Feature Processing...")
        cfg = load_pipeline_config()
        num_cols = cfg["features"]["numerical_cols"]
        cat_cols = cfg["features"]["categorical_cols"]

        X_train_raw = df_train.drop(columns=["amount"])
        X_test_raw = df_test.drop(columns=["amount"])

        y_train = df_train["amount"].copy()
        y_test = df_test["amount"].copy()

        print("      [~] Fitting and transforming features...")
        X_train_clean, X_test_clean = fit_transform_features(
            X_train_raw, X_test_raw, y_train, num_cols, cat_cols
        )

        # ---------------------------------------------------------
        # 3. DYNAMIC TRANSFORMATION (Log/Raw)
        # ---------------------------------------------------------
        print("\n[STAGE 1.9] Applying dynamic transformations...")
        X_train_transformed = apply_feature_transform(
            X_train_clean, cfg, dataset_name="Train"
        )
        X_test_transformed = apply_feature_transform(
            X_test_clean, cfg, dataset_name="Test"
        )

        transform_mode = cfg.get("experiment", {}).get("target_transform", "raw")
        is_log = transform_mode == "log"

        if is_log:
            y_train_transformed = np.log1p(y_train.astype(float))
            y_test_transformed = np.log1p(y_test.astype(float))
            print("      [>] Target: Applied Log Transform (np.log1p).")
        else:
            y_train_transformed = y_train
            y_test_transformed = y_test
            print("      [>] Target: RAW mode active.")

        df_train_final = pd.concat([X_train_transformed, y_train_transformed], axis=1)
        df_test_final = pd.concat([X_test_transformed, y_test_transformed], axis=1)

        # ---------------------------------------------------------
        # 4. INITIATING EXPERIMENT LOOP
        # ---------------------------------------------------------
        print("\n" + "=" * 60)
        print("🔬 INITIATING EXPERIMENT LOOP")
        print("=" * 60)

        mlflow.set_experiment("House_Price_Feature_Selection_Experiment")

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--methods",
            nargs="+",
            default=[
                "baseline",
                "filter",
                "forward",
                "backward",
                "lasso",
                "best_subset",
            ],
            help="List of feature selection methods to execute (e.g., --methods lasso forward)",
        )

        # Safely handle arguments for both Terminal and Jupyter environments
        args = (
            parser.parse_args()
            if sys.argv[0].endswith(".py")
            else parser.parse_args([])
        )
        experiment_methods = args.methods

        print(f"[*] Execution targets scheduled: {experiment_methods}")

        selection_cfg = cfg.get("experiment", {}).get("selection_settings", {})
        top_k = selection_cfg.get("top_k_features", 15)
        criterion = selection_cfg.get("criterion", "aic")

        all_reports_data = []

        metrics_dir = Path("reports") / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        for method in experiment_methods:
            # Create sepearted file name (aic/bic)
            file_suffix = f"{method}_{criterion}" if method != "baseline" else method
            run_name = f"Selection_{file_suffix.upper()}"

            with mlflow.start_run(run_name=run_name):
                print(f"\n{'='*40}\n🔹 RUNNING: {run_name}\n{'='*40}")
                start_time = time.time()

                # --- FEATURE SELECTION & TRAINING ---
                print(
                    f"\n[STAGE 2] Running Gatekeeper & Model Training for {method.upper()}..."
                )

                mlflow.log_param("selection_method", method)
                mlflow.log_param("target_transform", transform_mode)

                if method != "baseline":
                    mlflow.log_param("criterion", criterion)
                if method in ["filter", "best_subset"]:
                    mlflow.log_param("top_k_limit", top_k)

                if method == "baseline":
                    print(
                        "      [~] BASELINE active: Utilizing 100% of available features."
                    )
                    selected_features = list(
                        df_train_final.drop(columns=["amount"]).columns
                    )
                    trace = [
                        {
                            "Step": 1,
                            "Action": "Baseline: Retained all original features",
                            "Criterion": "NONE",
                            "Criterion_Score": None,
                            "MSE_Score": None,
                            "Features_Used": ", ".join(selected_features),
                        }
                    ]
                else:
                    selected_features, trace = run_feature_selection(
                        df_train_final,
                        target_column="amount",
                        method=method,
                        criterion=criterion,
                        top_k=top_k,
                    )

                # Export Trace to CSV for detailed Reporting
                trace_df = pd.DataFrame(trace)
                trace_csv_path = metrics_dir / f"{file_suffix}_step_trace.csv"
                trace_df.to_csv(trace_csv_path, index=False)
                print(f"      [>] Execution trace saved to: {trace_csv_path.name}")

                model_filename = f"model_{file_suffix}.pkl"
                model, metrics = train_dynamic_model(
                    df_train_final,
                    df_test_final,
                    config=cfg,
                    selected_features=selected_features,
                    target_column="amount",
                    model_name=model_filename,
                )

                end_time = time.time()
                execution_time = end_time - start_time

                # --- REPORTING & METRICS ---
                print(
                    f"\n[STAGE 3] Generating Advanced Benchmark Report for {method.upper()}..."
                )

                selector_context = {
                    "method": method,
                    "criterion": criterion if method != "baseline" else "none",
                }
                report_filename = f"report_{file_suffix}.json"

                report_data = generate_benchmark_report(
                    metrics=metrics,
                    output_file=report_filename,
                    is_log_transformed=is_log,
                    selector_context=selector_context,
                    execution_time=execution_time,
                )

                all_reports_data.append(report_data)

                print(
                    "      [*] Pushing all key metrics (AIC/BIC/AdjR2/MSE) to MLflow..."
                )
                mlflow.log_metric("Execution_Time_sec", execution_time)
                mlflow.log_metric("Test_R2", report_data["Test_R2"])
                mlflow.log_metric("Test_RMSE", report_data["Test_RMSE"])
                mlflow.log_metric("Test_MSE", report_data["Test_RMSE"] ** 2)
                mlflow.log_metric("Test_MAE", report_data["Test_MAE"])

                # Get all metric for reporting
                if report_data.get("Train_AIC") is not None:
                    mlflow.log_metric("Train_AIC", report_data["Train_AIC"])
                    mlflow.log_metric("Train_BIC", report_data["Train_BIC"])
                    mlflow.log_metric("Train_Adj_R2", report_data["Train_Adj_R2"])

        # ---------------------------------------------------------
        # 5. END OF LOOP: GENERATE LEADERBOARD
        # ---------------------------------------------------------
        generate_leaderboard(all_reports_data)

        print("\n" + "=" * 60)
        print("✅ ALL EXPERIMENTS EXECUTED SUCCESSFULLY!")
        print(
            "📂 Check 'reports/' for JSON, PNG Visuals, Leaderboard CSV, Step Traces, and MLflow UI."
        )
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n❌ [ERROR] Pipeline failed at execution: {e}\n")


if __name__ == "__main__":
    run_pipeline()
