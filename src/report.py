import json
import numpy as np
import pandas as pd
from pathlib import Path
import traceback
import time
import textwrap
import sys

# SAFE CONFIGURATION: Force Matplotlib to use 'Agg' backend
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import r2_score

from src.config import REPORTS_DIR


def generate_visualizations(y_true_real, y_pred_real, method_name, target_mode="Raw"):
    """
    Automatically generates an Actual vs Predicted scatter plot.
    """
    try:
        print(f"      [-] Generating scatter plot for {method_name}...")
        plt.figure(figsize=(10, 6))

        plt.scatter(
            y_true_real,
            y_pred_real,
            alpha=0.4,
            color="#1f77b4",
            s=15,
            label="Predicted Data Points",
        )

        p99_actual = np.percentile(y_true_real, 99)
        p99_pred = np.percentile(y_pred_real, 99)
        axis_limit = max(p99_actual, p99_pred) * 1.05

        plt.plot(
            [0, axis_limit],
            [0, axis_limit],
            color="#d62728",
            linestyle="--",
            linewidth=2,
            label="Ideal Fit (Actual = Predicted)",
        )

        plt.xlim(0, axis_limit)
        plt.ylim(0, axis_limit)

        plt.title(
            f"Actual vs Predicted Prices - {method_name}\n(Zoomed to 99th Percentile | Trained on {target_mode} Data)",
            fontsize=14,
            pad=15,
        )
        plt.xlabel("Actual Price (Rupees)", fontsize=12)
        plt.ylabel("Predicted Price (Rupees)", fontsize=12)

        plt.legend(loc="upper left", frameon=True, shadow=True)
        plt.grid(True, linestyle=":", alpha=0.7)

        figures_dir = Path(REPORTS_DIR) / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        fig_path = figures_dir / f"{method_name.lower()}_actual_vs_predicted.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()

    except Exception as e:
        print(
            f"      [!] Warning: Failed to generate plot for {method_name}. Error: {e}"
        )
        plt.close()


def generate_benchmark_report(
    metrics,
    output_file="report.json",
    is_log_transformed=False,
    selector_context=None,
    execution_time=None,
):
    """
    Generates JSON report and routes it to reports/metrics folder.
    """
    method_name = (
        selector_context.get("method", "Baseline").upper()
        if selector_context
        else "BASELINE"
    )
    print(f"      [-] Generating benchmark report & visuals for {method_name}...")

    target_mode = "Log" if is_log_transformed else "Raw"

    y_true = np.array(metrics["y_true"])
    y_pred = np.array(metrics["y_pred"])

    # CHÍNH XÁC: Tính R2 trên đúng thang đo mà mô hình đã học (Log scale nếu có áp dụng)
    # Điều này phản ánh trung thực khả năng giải thích phương sai của mô hình
    test_r2 = r2_score(y_true, y_pred)

    # LUÔN ÉP VỀ ĐƠN VỊ THỰC TẾ (RUPEE) ĐỂ TÍNH SAI SỐ RA TIỀN VÀ VẼ BIỂU ĐỒ
    if is_log_transformed:
        y_true_real = np.expm1(y_true)
        y_pred_real = np.expm1(y_pred)
    else:
        y_true_real = y_true
        y_pred_real = y_pred

    generate_visualizations(y_true_real, y_pred_real, method_name, target_mode)

    # TÍNH TOÁN CÁC CHỈ SỐ SAI SỐ TRÊN ĐƠN VỊ THỰC
    test_rmse = np.sqrt(np.mean((y_true_real - y_pred_real) ** 2))
    test_mae = np.mean(np.abs(y_true_real - y_pred_real))

    train_aic = metrics.get("Train_AIC")
    train_bic = metrics.get("Train_BIC")
    train_adj_r2 = metrics.get("Train_Adj_R2")

    features_list = metrics.get("features_used", [])
    num_features = len(features_list)

    # SMART TRUNCATION: Hiển thị trọn vẹn cho mô hình chọn ít biến (để lấy insight),
    # và thu gọn cho mô hình dùng quá nhiều biến (Baseline, Backward) để chống tràn CSV/Terminal.
    if num_features <= 15:
        features_display = ", ".join(features_list)
    else:
        features_display = (
            ", ".join(features_list[:10]) + f" ... (+{num_features - 10} more)"
        )

    report_data = {
        "Model": method_name,
        "Target_Mode": target_mode,
        "Execution_Time_sec": round(execution_time, 2) if execution_time else None,
        "Num_Features": metrics.get("total_features", num_features),
        "Train_Adj_R2": (
            round(float(train_adj_r2), 4) if train_adj_r2 is not None else None
        ),
        "Train_AIC": round(float(train_aic), 4) if train_aic is not None else None,
        "Train_BIC": round(float(train_bic), 4) if train_bic is not None else None,
        "Test_RMSE": round(float(test_rmse), 2),
        "Test_MAE": round(float(test_mae), 2),
        "Test_R2": round(float(test_r2), 4),
        "Features": features_display,
    }

    metrics_dir = Path(REPORTS_DIR) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    report_path = metrics_dir / output_file

    with open(report_path, "w", encoding="utf-8") as f:
        full_report = report_data.copy()
        full_report["Features"] = features_list
        json.dump(full_report, f, indent=4)

    return report_data


def generate_leaderboard_charts(df_leaderboard):
    """
    Generates Analytical Charts:
    1. RMSE Comparison (Lower is Better)
    2. Trade-off (Features vs Time)
    3. R2 Comparison (Higher is Better)
    """
    try:
        print("   [-] Generating Leaderboard Analytical Charts with Annotations...")
        figures_dir = Path(REPORTS_DIR) / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        target_mode = (
            df_leaderboard["Target_Mode"].iloc[0]
            if "Target_Mode" in df_leaderboard.columns
            else "Raw"
        )

        # =========================================================
        # CHART 1: Test RMSE Comparison (Càng thấp càng tốt)
        # =========================================================
        plt.figure(figsize=(10, 6))
        df_sorted_rmse = df_leaderboard.sort_values("Test_RMSE").reset_index(drop=True)

        highlight_colors = [
            "#e74c3c" if i == 0 else "#bdc3c7" for i in range(len(df_sorted_rmse))
        ]
        sns.barplot(
            x="Model", y="Test_RMSE", data=df_sorted_rmse, palette=highlight_colors
        )

        plt.title(
            "Model Accuracy Comparison: Test RMSE (Lower is Better)",
            fontsize=14,
            pad=15,
            fontweight="bold",
        )
        plt.xlabel("Selection Method", fontsize=12)
        plt.ylabel("Test RMSE (Rupees)", fontsize=12)
        plt.xticks(rotation=45)

        # FIX LỖI OVERLAP: Tăng độ cao (trần) của trục Y lên 20%
        max_rmse = df_sorted_rmse["Test_RMSE"].max()
        plt.ylim(0, max_rmse * 1.2)

        for index, value in enumerate(df_sorted_rmse["Test_RMSE"]):
            weight = "bold" if index == 0 else "normal"
            fontsize_val = 12 if index == 0 else 10
            plt.text(
                index,
                value,
                f"{value:,.0f}",
                ha="center",
                va="bottom",
                fontsize=fontsize_val,
                fontweight=weight,
            )

        legend_elements = [
            mpatches.Patch(color="#e74c3c", label="Winner (Lowest RMSE)"),
            mpatches.Patch(color="#bdc3c7", label="Other Models"),
        ]
        plt.legend(handles=legend_elements, loc="upper right", frameon=True)

        plt.tight_layout()
        plt.savefig(
            figures_dir / "leaderboard_01_rmse_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # =========================================================
        # CHART 2: Efficiency Trade-off
        # =========================================================
        fig, ax1 = plt.subplots(figsize=(10, 6))

        color_bar = "#3498db"
        ax1.set_xlabel("Selection Method", fontsize=12)
        ax1.set_ylabel(
            "Number of Features Used", color=color_bar, fontsize=12, fontweight="bold"
        )
        sns.barplot(
            x="Model",
            y="Num_Features",
            data=df_leaderboard,
            color=color_bar,
            alpha=0.6,
            ax=ax1,
        )
        ax1.tick_params(axis="y", labelcolor=color_bar)
        plt.xticks(rotation=45)

        for index, value in enumerate(df_leaderboard["Num_Features"]):
            ax1.text(
                index,
                value / 2,
                str(value),
                color="white",
                ha="center",
                va="center",
                fontweight="bold",
                fontsize=12,
            )

        ax2 = ax1.twinx()
        color_line = "#2c3e50"
        ax2.set_ylabel(
            "Execution Time (seconds)", color=color_line, fontsize=12, fontweight="bold"
        )
        sns.lineplot(
            x="Model",
            y="Execution_Time_sec",
            data=df_leaderboard,
            color=color_line,
            marker="o",
            markersize=8,
            linewidth=2.5,
            ax=ax2,
        )
        ax2.tick_params(axis="y", labelcolor=color_line)

        tradeoff_legend = [
            mpatches.Patch(
                color=color_bar, alpha=0.6, label="Features Used (Left Axis)"
            ),
            plt.Line2D(
                [0],
                [0],
                color=color_line,
                lw=2.5,
                marker="o",
                label="Execution Time (Right Axis)",
            ),
        ]
        ax1.legend(
            handles=tradeoff_legend,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.15),
            ncol=2,
            frameon=True,
        )

        plt.title(
            "Resource Trade-off: Features Retained vs Execution Time",
            fontsize=14,
            pad=45,
            fontweight="bold",
        )
        fig.tight_layout()
        plt.savefig(
            figures_dir / "leaderboard_02_efficiency_tradeoff.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # =========================================================
        # CHART 3: Test R2 Comparison (Càng cao càng tốt)
        # =========================================================
        plt.figure(figsize=(10, 6))
        # R2 Càng cao càng tốt nên phải Sort Ascending = False
        df_sorted_r2 = df_leaderboard.sort_values(
            "Test_R2", ascending=False
        ).reset_index(drop=True)

        # Chọn màu XANH LÁ (Green) cho Kẻ chiến thắng R2
        highlight_colors_r2 = [
            "#27ae60" if i == 0 else "#bdc3c7" for i in range(len(df_sorted_r2))
        ]
        sns.barplot(
            x="Model", y="Test_R2", data=df_sorted_r2, palette=highlight_colors_r2
        )

        plt.title(
            "Model Explanatory Power: Test R² (Higher is Better)",
            fontsize=14,
            pad=15,
            fontweight="bold",
        )
        plt.xlabel("Selection Method", fontsize=12)
        plt.ylabel(
            f"Test R² ({target_mode} Scale)", fontsize=12
        )  # Hiển thị rõ Log Scale hay Raw Scale
        plt.xticks(rotation=45)

        # Tăng trần Y lên 20% và xử lý trường hợp R2 âm
        max_r2 = df_sorted_r2["Test_R2"].max()
        min_r2 = df_sorted_r2["Test_R2"].min()
        y_bottom = min(0, min_r2 * 1.2)  # Cho phép hiện giá trị âm nếu mô hình quá tệ
        plt.ylim(y_bottom, max_r2 * 1.2 if max_r2 > 0 else max_r2 * 0.8)

        for index, value in enumerate(df_sorted_r2["Test_R2"]):
            weight = "bold" if index == 0 else "normal"
            fontsize_val = 12 if index == 0 else 10
            # Hiển thị 4 chữ số thập phân cho R2
            plt.text(
                index,
                value,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=fontsize_val,
                fontweight=weight,
            )

        legend_elements_r2 = [
            mpatches.Patch(color="#27ae60", label="Winner (Highest R²)"),
            mpatches.Patch(color="#bdc3c7", label="Other Models"),
        ]
        plt.legend(handles=legend_elements_r2, loc="upper right", frameon=True)

        plt.tight_layout()
        plt.savefig(
            figures_dir / "leaderboard_03_r2_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    except Exception as e:
        print(f"   [!] Warning: Failed to generate leaderboard charts. Error: {e}")
        plt.close()


def generate_leaderboard(
    all_reports_data, output_filename="experiment_leaderboard.csv"
):
    """
    Aggregates results and exports to a clean CSV.
    Prints a well-formatted Pandas table to the terminal.
    """
    if not all_reports_data:
        return

    df_leaderboard = pd.DataFrame(all_reports_data)

    if "Test_RMSE" in df_leaderboard.columns:
        df_leaderboard = df_leaderboard.sort_values(
            by="Test_RMSE", ascending=True
        ).reset_index(drop=True)

    generate_leaderboard_charts(df_leaderboard)

    leaderboard_path = Path(REPORTS_DIR) / output_filename

    try:
        df_leaderboard.to_csv(leaderboard_path, index=False, encoding="utf-8")
        print(f" [+] Leaderboard CSV saved at: {leaderboard_path}\n")
    except PermissionError:
        fallback_filename = output_filename.replace(
            ".csv", f"_backup_{int(time.time())}.csv"
        )
        fallback_path = Path(REPORTS_DIR) / fallback_filename
        df_leaderboard.to_csv(fallback_path, index=False, encoding="utf-8")
        print(f" [!] Warning: '{output_filename}' is locked. Saved backup instead.\n")

    display_cols = [
        "Model",
        "Target_Mode",
        "Execution_Time_sec",
        "Num_Features",
        "Test_RMSE",
        "Test_R2",
        "Features",
    ]
    existing_cols = [col for col in display_cols if col in df_leaderboard.columns]

    # Định dạng lại cách Pandas in ra terminal (cho phép tự wrap text dài mà không chèn \n vào CSV gốc)
    pd.set_option("display.max_colwidth", 70)

    print("\n" + "=" * 125)
    print("🏆 EXPERIMENT LEADERBOARD 🏆".center(125))
    print("=" * 125)
    print(df_leaderboard[existing_cols].to_string(index=False, justify="left"))
    print("=" * 125 + "\n")

    pd.reset_option("display.max_colwidth")


# =====================================================================
# CHẾ ĐỘ CHẠY ĐỘC LẬP (STANDALONE MODE)
# =====================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 STANDALONE REPORT GENERATOR ACTIVATED")
    print("=" * 60)

    # Tự động tìm kiếm tất cả các file JSON hiện có trong reports/metrics/
    metrics_dir = Path(REPORTS_DIR) / "metrics"
    json_files = list(metrics_dir.glob("report_*.json"))

    if not json_files:
        print(
            f"❌ Không tìm thấy file JSON nào trong '{metrics_dir}'. Hãy chạy main.py trước."
        )
        sys.exit(1)

    print(
        f"[*] Tìm thấy {len(json_files)} file báo cáo JSON. Đang tiến hành tổng hợp..."
    )

    all_reports_data = []
    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

                # Biến list Features trong JSON thành chuỗi (áp dụng lại rule Truncation)
                features_list = data.get("Features", [])
                if isinstance(features_list, list):
                    num_features = len(features_list)
                    if num_features <= 15:
                        data["Features"] = ", ".join(features_list)
                    else:
                        data["Features"] = (
                            ", ".join(features_list[:10])
                            + f" ... (+{num_features - 10} more)"
                        )

                all_reports_data.append(data)
        except Exception as e:
            print(f" [!] Lỗi khi đọc file {file_path.name}: {e}")

    # Gọi thẳng hàm tạo Bảng xếp hạng và vẽ biểu đồ
    generate_leaderboard(all_reports_data)
    print("✅ Đã cập nhật lại toàn bộ Leaderboard CSV & Biểu đồ PNG thành công!")
