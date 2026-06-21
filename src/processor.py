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

    clip_cols = [
        "area",
        "room_floor",
        "total_floor",
        "bathroom",
        "balcony"
    ]

    for col in clip_cols:

        if col not in X_train.columns:
            continue

        upper_cap = X_train[col].quantile(0.995)

        X_train[col] = X_train[col].clip(
            upper=upper_cap
        )

        X_test[col] = X_test[col].clip(
            upper=upper_cap
        )

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
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
    
    return X_train, X_test
