import pandas as pd
import numpy as np


def fit_transform_features(X_train, X_test, numeric_cols, categorical_cols):
    """
    Tách phần tiền xử lý dữ liệu để đưa vào model
    Xử lý dữ liệu (Imputation & Encoding).
    """

    X_train = X_train.copy()
    X_test = X_test.copy()

    # ==================================================
    # Numeric Imputation
    # ==================================================

    for col in numeric_cols:

        if col not in X_train.columns:
            continue

        median_value = X_train[col].median()

        X_train[col] = X_train[col].fillna(median_value)
        X_test[col] = X_test[col].fillna(median_value)

    # ==================================================
    # Outlier Clipping
    # Learn From Train Only
    # ==================================================

    clip_cols = ["area", "room_floor", "total_floor", "bathroom", "balcony"]

    for col in clip_cols:

        if col not in X_train.columns:
            continue

        upper_cap = X_train[col].quantile(0.995)

        X_train[col] = X_train[col].clip(upper=upper_cap)

        X_test[col] = X_test[col].clip(upper=upper_cap)

    # ==================================================
    # Categorical Imputation
    # ==================================================

    for col in categorical_cols:

        if col not in X_train.columns:
            continue

        X_train[col] = X_train[col].fillna("Unknown")
        X_test[col] = X_test[col].fillna("Unknown")

    # --------------------------------------------------
    # Encoding & Alignment
    # --------------------------------------------------
    # Apply One-Hot Encoding
    X_train = pd.get_dummies(X_train, columns=categorical_cols)
    X_test = pd.get_dummies(X_test, columns=categorical_cols)

    # Align Test columns with Train columns
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    return X_train, X_test


# =====================================================================
# NHIỆM VỤ 2: CÔNG TẮC CHUYỂN ĐỔI TOÁN HỌC LOG/RAW (Tính năng mới)
# =====================================================================
def apply_feature_transform(df, config, dataset_name=""):
    """
    Hàm tự động áp dụng biến đổi log cho các cột đầu vào (features)
    dựa trên cấu hình trong config.yaml
    """
    df_transformed = df.copy()

    # 1. Đọc trạng thái công tắc từ config
    transform_mode = config.get("experiment", {}).get("target_transform", "raw")

    # 2. Rẽ nhánh logic
    if transform_mode == "log":
        log_cols = config.get("experiment", {}).get("log_features", [])
        for col in log_cols:
            if col in df_transformed.columns:
                # Ép kiểu về float64 để tránh lỗi khi numpy lấy log trên số nguyên (int)
                df_transformed[col] = df_transformed[col].astype(float)
                # Dùng np.log1p để lấy log an toàn
                df_transformed[col] = np.log1p(df_transformed[col])
                print(
                    f"   -> [Processor] Feature (X - {dataset_name}): Applied Log Transform to '{col}'."
                )
    else:
        print(
            f"   -> [Processor] Feature (X - {dataset_name}): RAW mode active (Keeping original values)."
        )

    return df_transformed
