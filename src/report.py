import json
import numpy as np
from sklearn.metrics import r2_score
from src.config import REPORTS_DIR


def generate_benchmark_report(
    metrics,
    output_file="report.json",
    is_log_transformed=False,
    selector_context=None,
):
    """
    Tạo báo cáo JSON ưu tiên 100% logic thống kê của Hoàng.
    Mô phỏng chính xác cấu trúc Output từ file Jupyter Notebook gốc.
    """
    method_name = (
        selector_context.get("method", "Baseline").upper()
        if selector_context
        else "BASELINE"
    )
    print(f"      [-] Generating Hoang-style report for {method_name}...")

    # Lấy dữ liệu y thực tế và y dự đoán
    y_true = np.array(metrics["y_true"])
    y_pred = np.array(metrics["y_pred"])

    # Nếu có cấu hình log, phải đưa về Rupee để tính RMSE, MAE thực tế như Hoàng mong muốn
    if is_log_transformed:
        y_true_real = np.expm1(y_true)
        y_pred_real = np.expm1(y_pred)
    else:
        y_true_real = y_true
        y_pred_real = y_pred

    # 1. Tính toán các chỉ số trên tập TEST (Scikit-learn)
    test_rmse = np.sqrt(np.mean((y_true_real - y_pred_real) ** 2))
    test_mae = np.mean(np.abs(y_true_real - y_pred_real))
    test_r2 = r2_score(y_true_real, y_pred_real)

    # 2. Lấy các chỉ số trên tập TRAIN (Statsmodels - do model.py cung cấp)
    train_aic = metrics.get("Train_AIC")
    train_bic = metrics.get("Train_BIC")
    train_adj_r2 = metrics.get("Train_Adj_R2")

    # 3. Tạo cấu trúc Báo cáo
    report_data = {
        "Model": method_name,
        "Num_Features": metrics.get(
            "total_features", len(metrics.get("features_used", []))
        ),
        "Features": ", ".join(metrics.get("features_used", [])),
        "Train_Adj_R2": (
            round(float(train_adj_r2), 4) if train_adj_r2 is not None else None
        ),
        "Train_AIC": round(float(train_aic), 4) if train_aic is not None else None,
        "Train_BIC": round(float(train_bic), 4) if train_bic is not None else None,
        "Test_RMSE": round(float(test_rmse), 2),
        "Test_MAE": round(float(test_mae), 2),
        "Test_R2": round(float(test_r2), 4),
    }

    # 4. Xuất ra file JSON
    report_path = REPORTS_DIR / output_file
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=4)

    # 5. In ra màn hình Terminal mô phỏng bảng kết quả
    print("\n" + "=" * 50)
    print(f"📊 REPORT: {report_data['Model']}")
    print("=" * 50)
    print(f" -> Num_Features : {report_data['Num_Features']}")
    print(f" -> Train_Adj_R2 : {report_data['Train_Adj_R2']}")
    print(f" -> Train_AIC    : {report_data['Train_AIC']}")
    print(f" -> Train_BIC    : {report_data['Train_BIC']}")
    print(f" -> Test_RMSE    : {report_data['Test_RMSE']}")
    print(f" -> Test_MAE     : {report_data['Test_MAE']}")
    print(f" -> Test_R2      : {report_data['Test_R2']}")
    print("=" * 50 + "\n")

    return report_data
