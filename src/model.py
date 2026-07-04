# =====================================================================
# MODULE: Model Engine
# DESCRIPTION: Trains machine learning models dynamically, extracts
#              statistical metrics, and exports production models.
# =====================================================================

import joblib
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from src.config import MODELS_DIR


# ---------------------------------------------------------
# 1. DYNAMIC MODEL TRAINING FUNCTION
# ---------------------------------------------------------
def train_dynamic_model(
    df_train,
    df_test,
    config,
    selected_features=None,
    target_column="amount",
    model_name="model.pkl",
):
    """
    Trains a machine learning model dynamically and extracts statistical metrics (AIC, BIC).

    This function performs two parallel tasks:
    1. Uses statsmodels to extract advanced statistical metrics for reporting.
    2. Uses scikit-learn to train the main model and exports it as a .pkl file for production.

    Args:
        df_train (pd.DataFrame): Training dataset.
        df_test (pd.DataFrame): Testing dataset.
        config (dict): Pipeline configuration dictionary.
        selected_features (list, optional): List of features to use. Defaults to None.
        target_column (str): Target variable name. Defaults to "amount".
        model_name (str): Output filename for the trained model. Defaults to "model.pkl".

    Returns:
        tuple: (trained_model, metrics_dictionary)
    """
    print(f"\n      [~] Starting model training: {model_name}...")

    # ---------------------------------------------------------
    # 1. DATA SUBSETTING
    # ---------------------------------------------------------
    # Filter features based on the Feature Selector's output
    if selected_features:
        X_train = df_train[selected_features]
        X_test = df_test[selected_features]
    else:
        X_train = df_train.drop(columns=[target_column])
        X_test = df_test.drop(columns=[target_column])

    y_train = df_train[target_column]
    y_test = df_test[target_column]

    # ---------------------------------------------------------
    # 2. HYBRID STEP: EXTRACT STATISTICAL METRICS (STATSMODELS)
    # ---------------------------------------------------------
    aic, bic, train_adj_r2 = None, None, None
    try:
        print("      [~] Extracting statistical metrics (AIC, BIC) via statsmodels...")

        # Select only numerical columns, handle NaNs, and force float type
        X_train_numeric = X_train.select_dtypes(include=[np.number])
        X_train_clean = X_train_numeric.fillna(0)
        X_train_float = X_train_clean.astype(float)

        X_train_const = sm.add_constant(X_train_float, has_constant="add")
        sm_model = sm.OLS(y_train.astype(float), X_train_const).fit()

        aic = sm_model.aic
        bic = sm_model.bic
        train_adj_r2 = sm_model.rsquared_adj

    except Exception as e:
        print(f"      [!] Statistical metric extraction skipped: {e}")

    # ---------------------------------------------------------
    # 3. PRODUCTION TRAINING: SKLEARN ENGINE
    # ---------------------------------------------------------
    model_params = config.get("model_params", {})
    algo = model_params.get("algorithm", "LinearRegression")

    # Ensure X_train and X_test are numeric for Scikit-learn to prevent fitting errors
    X_train_sklearn = X_train.select_dtypes(include=[np.number]).fillna(0)
    X_test_sklearn = X_test.select_dtypes(include=[np.number]).fillna(0)

    if algo == "RandomForestRegressor":
        print("      [>] Algorithm: RandomForestRegressor")
        model = RandomForestRegressor(
            n_estimators=model_params.get("n_estimators", 100),
            max_depth=model_params.get("max_depth", 5),
            random_state=model_params.get("random_state", 42),
        )
    else:
        print("      [>] Algorithm: LinearRegression")
        model = LinearRegression()

    # Train Scikit-learn model
    model.fit(X_train_sklearn, y_train)

    # Predict on Test set
    y_pred = model.predict(X_test_sklearn)

    # ---------------------------------------------------------
    # 4. EVALUATION & METRICS PACKAGING
    # ---------------------------------------------------------
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    mae = mean_absolute_error(y_test, y_pred)

    # Package all metrics (Scikit-learn + Statsmodels)
    metrics = {
        "model_type": algo,
        "total_features": len(X_train_sklearn.columns),
        "features_used": list(X_train_sklearn.columns),
        "R2_Score": round(r2, 4),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        # Keep y_true, y_pred for report.py to calculate metrics on the native scale
        "y_true": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        # Include advanced training metrics
        "Train_AIC": round(aic, 4) if aic is not None else None,
        "Train_BIC": round(bic, 4) if bic is not None else None,
        "Train_Adj_R2": round(train_adj_r2, 4) if train_adj_r2 is not None else None,
    }

    # ---------------------------------------------------------
    # 5. MODEL EXPORT
    # ---------------------------------------------------------
    model_path = MODELS_DIR / model_name
    joblib.dump(model, model_path)
    print(f"      [+] Model saved successfully at: {model_path.name}")

    return model, metrics
