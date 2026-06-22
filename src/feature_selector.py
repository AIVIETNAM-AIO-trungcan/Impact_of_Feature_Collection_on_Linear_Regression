import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression


def select_features(X_train, y_train, X_test, config):
    """
    Gatekeeper Station: Selects the top K most important features
    to prevent Curse of Dimensionality and improve model training speed.
    """
    # 1. Read the VIP limit from config (Default is 20)
    top_k = config.get("experiment", {}).get("top_k_features", 20)

    total_features = X_train.shape[1]
    print(f"   -> [Selector] Original feature count: {total_features}")

    # Safety check: If we ask for more features than we actually have, just pass everything
    if top_k >= total_features:
        print(
            "   -> [Selector] 'top_k' is larger than total features. Keeping all features."
        )
        return X_train.copy(), X_test.copy()

    print(f"   -> [Selector] Selecting Top {top_k} features using f_regression...")

    # =====================================================================
    # STEP 2: INITIALIZE FEATURE SELECTOR
    # ---------------------------------------------------------------------
    # Input:
    #   - score_func = f_regression (Computes ANOVA F-value for the sample)
    #   - k = top_k (int: number of top features to select)
    # Output:
    #   - selector: Un-fitted SelectKBest instance.
    # =====================================================================
    selector = SelectKBest(score_func=f_regression, k=top_k)

    # =====================================================================
    # STEP 3: FIT ON TRAINING DATA ONLY (PREVENT DATA LEAKAGE)
    # ---------------------------------------------------------------------
    # Input:
    #   - X_train: Pandas DataFrame, shape: [n_samples, n_features]
    #   - y_train: Pandas Series, shape: [n_samples]
    # Process:
    #   - Evaluates the linear relationship using ANOVA F-value.
    #   - For each feature X_i, the F-score is calculated as:
    #       F = (R^2 / (1 - R^2)) * (n_samples - 2)
    #     where R^2 is the Pearson correlation coefficient squared between X_i and y.
    #   - High F-score implies high correlation -> Selected.
    #   - Stores F-scores and p-values in the selector's internal state.
    # Output:
    #   - Fitted selector instance.
    # =====================================================================
    selector.fit(X_train, y_train)

    # =====================================================================
    # STEP 4: EXTRACT SELECTED FEATURE NAMES
    # ---------------------------------------------------------------------
    # Flow 1: Retrieve boolean mask of selected features
    #   -> selector.get_support()
    #   -> Output: Numpy Array[bool], shape: [n_features]
    #
    # Flow 2: Apply mask to the original feature index
    #   -> X_train.columns[selector.get_support()]
    # Output (selected_cols):
    #   -> Pandas Index[str] containing the 'top_k' feature names.
    # =====================================================================
    selected_cols = X_train.columns[selector.get_support()]

    # =====================================================================
    # STEP 5: FILTER DATASETS
    # ---------------------------------------------------------------------
    # Input:
    #   - X_train, X_test: Original Pandas DataFrames
    #   - selected_cols: Pandas Index[str] of selected features
    # Process:
    #   - Subset both DataFrames using the selected feature names.
    #   - Apply .copy() to decouple memory allocation and prevent SettingWithCopyWarning.
    # Output:
    #   - X_train_selected: Pandas DataFrame, shape: [n_samples_train, k]
    #   - X_test_selected: Pandas DataFrame, shape: [n_samples_test, k]
    # =====================================================================
    X_train_selected = X_train[selected_cols].copy()
    X_test_selected = X_test[selected_cols].copy()

    print(
        f"   -> [Selector] Selection complete. Dropped {total_features - top_k} uninformative features."
    )

    return X_train_selected, X_test_selected
