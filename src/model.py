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
