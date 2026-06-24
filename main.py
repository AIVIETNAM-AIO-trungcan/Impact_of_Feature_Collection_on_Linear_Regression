import pandas as pd
import numpy as np
import yaml
import mlflow
import time  # THÊM MỚI: Thư viện đo thời gian
import argparse  # THÊM MỚI: Thư viện đọc tham số từ terminal
import sys  # THÊM MỚI: Xử lý an toàn cho Terminal/Jupyter

from src.data_loader import load_raw_data
from src.preprocess import preprocess_and_save
from src.data_splitter import split_and_save_data
from src.processor import fit_transform_features, apply_feature_transform
from src.feature_selector_apply import run_feature_selection
from src.model import train_dynamic_model
from src.report import (
    generate_benchmark_report,
    generate_leaderboard,
)  # THÊM MỚI: Import Leaderboard
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

        # ==========================================================
        # Prepare X and y
        # ==========================================================

        X_train_raw = df_train.drop(columns=["amount"])
        X_test_raw = df_test.drop(columns=["amount"])

        y_train = df_train["amount"].copy()
        y_test = df_test["amount"].copy()

        # ==========================================================
        # Feature Processing
        # ==========================================================

        X_train_clean, X_test_clean = fit_transform_features(
            X_train_raw,
            X_test_raw,
            y_train,
            num_cols,
            cat_cols,
        )

        # ---------------------------------------------------------
        # STAGE 1.9: DYNAMIC TRANSFORMATION (Log/Raw)
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

        # THÊM MỚI: Khởi tạo Parser để nhận lệnh từ Terminal
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
            help="Danh sách các phương pháp cần chạy (VD: --methods lasso forward)",
        )
        # Xử lý tham số an toàn
        args = (
            parser.parse_args()
            if sys.argv[0].endswith(".py")
            else parser.parse_args([])
        )
        experiment_methods = args.methods

        print(f"[*] Các phương pháp sẽ được chạy: {experiment_methods}")

        # Cấu hình chọn biến
        selection_cfg = cfg.get("experiment", {}).get("selection_settings", {})
        top_k = selection_cfg.get("top_k_features", 15)
        criterion = selection_cfg.get("criterion", "aic")

        # Khởi tạo mảng lưu trữ báo cáo cho Leaderboard
        all_reports_data = []

        for method in experiment_methods:
            run_name = f"Selection_{method.upper()}"

            with mlflow.start_run(run_name=run_name):
                print(f"\n{'='*40}\n🔹 RUNNING: {run_name}\n{'='*40}")

                # Bắt đầu bấm giờ
                start_time = time.time()

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
                if method == "baseline":
                    print(
                        "      [~] BASELINE active: Using 100% of available features."
                    )
                    selected_features = list(
                        df_train_final.drop(columns=["amount"]).columns
                    )
                else:
                    selected_features = run_feature_selection(
                        df_train_final,
                        target_column="amount",
                        method=method,
                        criterion=criterion,
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

                # Dừng bấm giờ
                end_time = time.time()
                execution_time = end_time - start_time

                # ---------------------------------------------------------
                # STAGE 3: REPORTING & METRICS VALIDATION
                # ---------------------------------------------------------
                print(
                    f"\n[STAGE 3] Generating Advanced Benchmark Report for {method.upper()}..."
                )

                # A. Setup report configuration
                selector_context = {"method": method}
                report_filename = f"report_{method}.json"

                # B. Generate JSON report & Visualizations
                report_data = generate_benchmark_report(
                    metrics=metrics,
                    output_file=report_filename,
                    is_log_transformed=is_log,
                    selector_context=selector_context,
                    execution_time=execution_time,
                )

                # Lưu vào mảng để tạo Leaderboard cuối chương trình
                all_reports_data.append(report_data)

                # C. Push metrics to MLflow Tracking Server
                print(
                    "   -> [MLflow] Pushing metrics & execution time to tracking server..."
                )
                mlflow.log_metric("Execution_Time_sec", execution_time)  # Log thời gian
                mlflow.log_metric("Test_R2", report_data["Test_R2"])
                mlflow.log_metric("Test_RMSE", report_data["Test_RMSE"])
                mlflow.log_metric("Test_MAE", report_data["Test_MAE"])

                if report_data["Train_AIC"] is not None:
                    mlflow.log_metric("Train_AIC", report_data["Train_AIC"])
                    mlflow.log_metric("Train_BIC", report_data["Train_BIC"])
                    mlflow.log_metric("Train_Adj_R2", report_data["Train_Adj_R2"])

        # =========================================================
        # END OF LOOP: GENERATE LEADERBOARD
        # =========================================================
        generate_leaderboard(all_reports_data)

        print("\n" + "=" * 60)
        print("✅ ALL EXPERIMENTS EXECUTED SUCCESSFULLY!")
        print(
            "📂 Check 'reports/' for JSON, PNG Visuals, Leaderboard CSV, and MLflow UI."
        )
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n❌ [ERROR] Pipeline failed at execution: {e}\n")


if __name__ == "__main__":
    run_pipeline()
