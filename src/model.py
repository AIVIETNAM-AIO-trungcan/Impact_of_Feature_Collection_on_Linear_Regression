import joblib
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from src.config import MODELS_DIR


def train_dynamic_model(
    df_train,
    df_test,
    config,
    selected_features=None,
    target_column="amount",
    model_name="model.pkl",
):
    """
    Huấn luyện mô hình linh hoạt, kết hợp trích xuất chỉ số thống kê (AIC, BIC).

    Hàm này thực hiện hai nhiệm vụ song song:
    1. Dùng statsmodels để lấy các chỉ số chuyên sâu (AIC, BIC, Adj R2) cho báo cáo.
    2. Dùng scikit-learn để huấn luyện mô hình chính và lưu ra file .pkl phục vụ Production.
    """
    print(f"\n[-] Starting model training: {model_name}...")

    # 1. Cắt xén dữ liệu dựa trên danh sách biến đã được Feature Selector chọn lọc
    if selected_features:
        X_train = df_train[selected_features]
        X_test = df_test[selected_features]
    else:
        X_train = df_train.drop(columns=[target_column])
        X_test = df_test.drop(columns=[target_column])

    y_train = df_train[target_column]
    y_test = df_test[target_column]

    # =====================================================================
    # BƯỚC LAI GHÉP: Dùng statsmodels để lấy AIC, BIC
    # =====================================================================
    aic, bic, train_adj_r2 = None, None, None
    try:
        print("      [~] Extracting statistical metrics (AIC, BIC) via statsmodels...")

        # 1. Chỉ chọn các cột có kiểu dữ liệu là số
        X_train_numeric = X_train.select_dtypes(include=[np.number])

        # 2. Xử lý sạch sẽ NaN hoặc dữ liệu lỗi
        X_train_clean = X_train_numeric.fillna(0)

        # 3. Ép kiểu cứng sang float
        X_train_float = X_train_clean.astype(float)

        X_train_const = sm.add_constant(X_train_float, has_constant="add")
        sm_model = sm.OLS(y_train.astype(float), X_train_const).fit()

        aic = sm_model.aic
        bic = sm_model.bic
        train_adj_r2 = sm_model.rsquared_adj

    except Exception as e:
        print(f"      [!] Statistical metric extraction skipped: {e}")

    # =====================================================================
    # HUẤN LUYỆN CHÍNH: Dùng Scikit-learn để Productionize
    # =====================================================================
    model_params = config.get("model_params", {})
    algo = model_params.get("algorithm", "LinearRegression")

    if algo == "RandomForestRegressor":
        print("      -> Algorithm: RandomForestRegressor")
        model = RandomForestRegressor(
            n_estimators=model_params.get("n_estimators", 100),
            max_depth=model_params.get("max_depth", 5),
            random_state=model_params.get("random_state", 42),
        )
    else:
        print("      -> Algorithm: LinearRegression")
        model = LinearRegression()

    # Huấn luyện mô hình Scikit-learn
    model.fit(X_train, y_train)

    # Dự đoán trên tập Test
    y_pred = model.predict(X_test)

    # Đánh giá cơ bản
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    mae = mean_absolute_error(y_test, y_pred)

    # Đóng gói toàn bộ metrics (Gồm cả của Scikit-learn và Statsmodels)
    metrics = {
        "model_type": algo,
        "total_features": len(X_train.columns),
        "features_used": list(X_train.columns),
        "R2_Score": round(r2, 4),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        # Giữ lại y_true, y_pred để report.py tính Adjusted R2 trên tập Test (như đã làm ở nhánh trước)
        "y_true": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        # Thêm các chỉ số thống kê chuyên sâu trên tập Train (Dành riêng cho Hoàng)
        "Train_AIC": round(aic, 4) if aic is not None else None,
        "Train_BIC": round(bic, 4) if bic is not None else None,
        "Train_Adj_R2": round(train_adj_r2, 4) if train_adj_r2 is not None else None,
    }

    # Lưu mô hình
    model_path = MODELS_DIR / model_name
    joblib.dump(model, model_path)
    print(f"      [+] Model saved successfully at: {model_path}")

    return model, metrics
