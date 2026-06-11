import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from src.config import MODELS_DIR

def train_baseline_model(df, target_column="price", model_name="baseline_model.pkl"):
    """
    Train a baseline Linear Regression model using all available features.
    """
    print("[-] Starting baseline model training...")
    
    # 1. Define features and target
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")
        
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # =========================================================================
    # [TODO - HOANG] 
    # ---> CODING HERE <---
    # Implement Train-Test Split here to evaluate the model correctly.
    # Currently, it trains and evaluates on the entire dataset.
    # =========================================================================
    
    # 2. Train model
    model = LinearRegression()
    model.fit(X, y)
    
    # 3. Calculate benchmark metrics
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = mean_squared_error(y, y_pred) ** 0.5
    
    metrics = {
        "model_type": "Linear Regression (Baseline)",
        "features_used": list(X.columns),
        "R2_Score": round(r2, 4),
        "RMSE": round(rmse, 4)
    }
    
    # 4. Save the model
    model_path = MODELS_DIR / model_name
    joblib.dump(model, model_path)
    print(f"[-] Baseline model saved at: {model_path}")
    
    return model, metrics
