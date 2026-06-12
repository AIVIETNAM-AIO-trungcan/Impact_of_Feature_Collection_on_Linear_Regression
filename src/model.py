import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from src.config import MODELS_DIR

def train_baseline_model(df_train, df_test, target_column="price", model_name="baseline_model.pkl"):
    """Train a baseline Linear Regression model using Train set."""
    print("[-] Starting baseline model training with Scikit-learn Pipeline...")

    # 1. Separate Features (X) and Target (y) for both Train and Test sets
    X_train = df_train.drop(columns=[target_column])
    y_train = df_train[target_column]
    
    X_test = df_test.drop(columns=[target_column])
    y_test = df_test[target_column]
    
    # =========================================================================
    # [TUNG NGUYEN] PREPROCESSING LOGIC (ADAPTED FOR ANTI-LEAKAGE)
    # =========================================================================

    # --------------------------------------------------
    # Numeric Features
    # --------------------------------------------------

    numeric_cols = [
        "amount",
        "area_sqft",
        "bathroom",
        "balcony",
        "room_floor",
        "total_floor"
    ]

    # --------------------------------------------------
    # Categorical Features
    # --------------------------------------------------

    categorical_cols = [
        "transaction",
        "furnishing",
        "facing",
        "location",
        "overlooking",
        "car_parking",
        "ownership",
        "area_type"
    ]

    # --------------------------------------------------
    # Median Imputation
    # Learn from Train only
    # --------------------------------------------------

    for col in numeric_cols:

        median_value = (
            X_train[col]
            .median()
        )

        X_train[col] = (
            X_train[col]
            .fillna(median_value)
        )

        X_test[col] = (
            X_test[col]
            .fillna(median_value)
        )

    # --------------------------------------------------
    # Unknown Imputation
    # --------------------------------------------------

    for col in categorical_cols:

        X_train[col] = (
            X_train[col]
            .fillna("Unknown")
        )

        X_test[col] = (
            X_test[col]
            .fillna("Unknown")
        )

    # --------------------------------------------------
    # Outlier Clipping
    # Learn cap from Train only
    # --------------------------------------------------

    # outlier_cols = [
    #     "amount",
    #     "area_sqft",
    #     "room_floor",
    #     "total_floor"
    # ]

    # for col in outlier_cols:

    #     upper_cap = (
    #         X_train[col]
    #         .quantile(0.995)
    #     )

    #     X_train[col] = (
    #         X_train[col]
    #         .clip(
    #             upper=upper_cap
    #         )
    #     )

    #     X_test[col] = (
    #         X_test[col]
    #         .clip(
    #             upper=upper_cap
    #         )
    #     )

    # =========================================================================
    # [HOANG] ENCODING & MODEL TRAINING 
    # =========================================================================
    
    # Apply One-Hot Encoding using Pandas get_dummies
    X_train = pd.get_dummies(X_train, columns=categorical_cols)
    X_test = pd.get_dummies(X_test, columns=categorical_cols)

    # Align Test columns with Train columns to handle missing/extra categories
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    # 2. Train model
    model = LinearRegression()
    print(X_train[["amount"]].corrwith(y_train))
    model.fit(X_train, y_train)
    
    # 3. Predict & Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    error = pd.DataFrame({
        "actual": y_test,
        "pred": y_pred
    })

    error["abs_error"] = (
        error["actual"]
        - error["pred"]
    ).abs()

    print(
        error
        .sort_values(
            "abs_error",
            ascending=False
        )
        .head(20)
    )
    
    metrics = {
        "model_type": "Linear Regression (Baseline)",
        "features_used": list(X_train.columns),
        "R2_Score": round(r2, 4),
        "RMSE": round(rmse, 4)
    }
    
    # 4. Save the model
    model_path = MODELS_DIR / model_name
    joblib.dump(model, model_path)

    print(f"[-] Baseline model saved at: {model_path}")
    
    return model, metrics
