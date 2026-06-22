import json
import numpy as np
from sklearn.metrics import r2_score
from src.config import REPORTS_DIR


def generate_benchmark_report(
    metrics,
    output_file="baseline_report.json",
    is_log_transformed=False,
    selector_context=None,
):
    """
    Generates a comprehensive benchmark report including Rupee-converted metrics,
    tracks both Log-space and Rupee-space R2 scores if applicable,
    and displays the total number of features used.
    """
    print(f"[-] Generating benchmark report: {output_file}...")

    # Extract data
    y_true = np.array(metrics["y_true"])
    y_pred = np.array(metrics["y_pred"])

    # Extract features list and count them
    features_list = metrics.get("features_used", [])
    total_features = len(features_list)

    # Extract R2 from model.py (If log-transformed, this is the Log-space R2)
    r2_model = metrics.get("R2_Score", 0.0)

    # 1. Inverse transform to Rupee if model was trained in log-space
    if is_log_transformed:
        y_true = np.expm1(y_true)
        y_pred = np.expm1(y_pred)

    # 2. Calculate core regression metrics on original units (Rupee)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # Calculate R2 on the actual currency scale (Rupee)
    r2_rupee = r2_score(y_true, y_pred)

    # 3. Construct the metrics dictionary dynamically
    report_metrics = {}
    if is_log_transformed:
        report_metrics["R2_Score (Log)"] = round(float(r2_model), 4)
        report_metrics["R2_Score (Rupee)"] = round(float(r2_rupee), 4)
    else:
        report_metrics["R2_Score"] = round(float(r2_rupee), 4)

    report_metrics["RMSE (Rupee)"] = round(float(rmse), 2)
    report_metrics["MAE (Rupee)"] = round(float(mae), 2)
    report_metrics["MAPE (%)"] = f"{round(float(mape), 2)}%"

    # 4. Construct the report structure
    report_data = {
        "model_type": (
            "Linear Regression (Optimized)"
            if is_log_transformed
            else "Linear Regression (Baseline)"
        ),
        "evaluation_context": {
            "method": (
                selector_context.get("method", "None") if selector_context else "None"
            ),
            "technique_type": (
                selector_context.get("technique_type", "None")
                if selector_context
                else "None"
            ),
            "feature_selection_enabled": selector_context is not None,
        },
        "total_features": total_features,
        "features_used": features_list,
        "metrics": report_metrics,
    }

    # 5. Save to JSON
    report_path = REPORTS_DIR / output_file
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=4)

    print(f"[+] Report generated at: {report_path}")
    print(f"    -> Total Features  : {total_features}")  # <--- IN RA TERMINAL

    # Dynamic Print out
    if is_log_transformed:
        print(f"    -> R2_Score (Log)  : {report_data['metrics']['R2_Score (Log)']}")
        print(f"    -> R2_Score (Rupee): {report_data['metrics']['R2_Score (Rupee)']}")
    else:
        print(f"    -> R2_Score        : {report_data['metrics']['R2_Score']}")

    print(f"    -> RMSE: {report_data['metrics']['RMSE (Rupee)']}")
    print(f"    -> MAE:  {report_data['metrics']['MAE (Rupee)']}")
    print(f"    -> MAPE: {report_data['metrics']['MAPE (%)']}")
