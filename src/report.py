import json
import numpy as np
from src.config import REPORTS_DIR


def generate_benchmark_report(
    metrics,
    output_file="baseline_report.json",
    is_log_transformed=False,
    selector_context=None,
):
    """
    Generates a comprehensive benchmark report including Rupee-converted metrics
    and feature selection context to evaluate different techniques.

    Args:
        metrics (dict): Dict containing 'y_true' and 'y_pred' (as numpy arrays).
        output_file (str): Target filename in REPORTS_DIR.
        is_log_transformed (bool): If True, applies expm1 to revert log-scale to Rupee.
        selector_context (dict): Metadata about selection method (method, technique_type, features).
    """
    print(f"[-] Generating benchmark report: {output_file}...")

    # Extract data
    y_true = np.array(metrics["y_true"])
    y_pred = np.array(metrics["y_pred"])

    # 1. Inverse transform to Rupee if model was trained in log-space
    if is_log_transformed:
        y_true = np.expm1(y_true)
        y_pred = np.expm1(y_pred)

    # 2. Calculate core regression metrics
    # RMSE and MAE in original units (Rupee)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    # MAPE as percentage
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # 3. Construct the report structure
    # selector_context handles None gracefully to support Baseline comparison
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
        "features_used": (
            selector_context.get("features", [])
            if selector_context
            else "All original features"
        ),
        "metrics": {
            "RMSE (Rupee)": round(float(rmse), 2),
            "MAE (Rupee)": round(float(mae), 2),
            "MAPE (%)": f"{round(float(mape), 2)}%",
        },
    }

    # 4. Save to JSON
    report_path = REPORTS_DIR / output_file
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=4)

    print(f"[+] Report generated at: {report_path}")
    print(f"    -> RMSE: {report_data['metrics']['RMSE (Rupee)']}")
    print(f"    -> MAE: {report_data['metrics']['MAE (Rupee)']}")
    print(f"    -> MAPE: {report_data['metrics']['MAPE (%)']}")
