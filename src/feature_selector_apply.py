# =====================================================================
# MODULE: Feature Selector
# DESCRIPTION: Implements feature selection algorithms (Best Subset,
#              Forward, Backward, Lasso, Filter) with step tracking.
# =====================================================================

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from itertools import combinations


# ---------------------------------------------------------
# 1. HELPER FUNCTIONS
# ---------------------------------------------------------
def fit_ols_for_selection(X_data, y_data, selected_features):
    """
    Fits an OLS model using statsmodels to evaluate feature subsets.

    Args:
        X_data (pd.DataFrame): The input features.
        y_data (pd.Series): The target variable.
        selected_features (list): List of feature names to include.

    Returns:
        model: The fitted OLS model or an empty model fallback if no features remain.
    """
    X_subset = (
        X_data[selected_features]
        .select_dtypes(include=[np.number])
        .fillna(0)
        .astype(float)
    )
    y_subset = y_data.astype(float)

    if X_subset.empty:

        class EmptyModel:
            aic = bic = mse_resid = np.inf
            rsquared_adj = -np.inf

        return EmptyModel()

    X_const = sm.add_constant(X_subset, has_constant="add")
    model = sm.OLS(y_subset, X_const).fit()
    return model


# ---------------------------------------------------------
# 2. BEST SUBSET SELECTION
# ---------------------------------------------------------
def best_subset_selection(X_train, y_train, criterion="aic"):
    """
    Evaluates all possible subsets of features to find the optimal combination.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target variable.
        criterion (str): Evaluation metric ('aic', 'bic', or 'adj_r2').

    Returns:
        tuple: (List of selected features, Execution trace list).
    """
    print(
        f"      [~] Running Best Subset Selection (Criterion: {criterion.upper()})..."
    )
    features = list(X_train.columns)
    n_features = len(features)

    best_overall_score = -np.inf if criterion == "adj_r2" else np.inf
    best_overall_subset = []
    trace = []

    for k in range(1, n_features + 1):
        best_score_for_k = -np.inf if criterion == "adj_r2" else np.inf
        best_mse_for_k = np.inf
        best_subset_for_k = []

        for subset in combinations(features, k):
            subset_list = list(subset)
            model = fit_ols_for_selection(X_train, y_train, subset_list)

            score = (
                model.rsquared_adj
                if criterion == "adj_r2"
                else (model.aic if criterion == "aic" else model.bic)
            )
            improved_k = (
                score > best_score_for_k
                if criterion == "adj_r2"
                else score < best_score_for_k
            )

            if improved_k:
                best_score_for_k = score
                best_subset_for_k = subset_list
                best_mse_for_k = model.mse_resid

        # Log the best performing subset for the current size k
        trace.append(
            {
                "Step": k,
                "Action": f"Best subset of size {k}",
                "Criterion": criterion.upper(),
                "Criterion_Score": best_score_for_k,
                "MSE_Score": best_mse_for_k,
                "Features_Used": ", ".join(best_subset_for_k),
            }
        )

        improved_overall = (
            best_score_for_k > best_overall_score
            if criterion == "adj_r2"
            else best_score_for_k < best_overall_score
        )
        if improved_overall:
            best_overall_score = best_score_for_k
            best_overall_subset = best_subset_for_k

    trace.append(
        {
            "Step": "FINAL",
            "Action": f"Selected optimal subset",
            "Criterion": criterion.upper(),
            "Criterion_Score": best_overall_score,
            "MSE_Score": None,
            "Features_Used": ", ".join(best_overall_subset),
        }
    )

    return best_overall_subset, trace


# ---------------------------------------------------------
# 3. FORWARD SELECTION
# ---------------------------------------------------------
def forward_selection(X_train, y_train, criterion="aic"):
    """
    Iteratively adds features that improve the model the most.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target variable.
        criterion (str): Evaluation metric ('aic', 'bic', or 'adj_r2').

    Returns:
        tuple: (List of selected features, Execution trace list).
    """
    print(f"      [~] Running Forward Selection (Criterion: {criterion.upper()})...")
    features = list(X_train.columns)
    remaining_features = set(features)
    selected_features = []
    trace = []

    best_score = -np.inf if criterion == "adj_r2" else np.inf
    step = 1

    while len(remaining_features) > 0:
        candidates = []
        for feature in sorted(remaining_features):
            candidate_features = selected_features + [feature]
            model = fit_ols_for_selection(X_train, y_train, candidate_features)

            score = (
                model.rsquared_adj
                if criterion == "adj_r2"
                else (model.aic if criterion == "aic" else model.bic)
            )
            candidates.append((score, feature, model.mse_resid))

        if criterion == "adj_r2":
            candidates.sort(reverse=True, key=lambda x: x[0])
            improved = candidates[0][0] > best_score
        else:
            candidates.sort(key=lambda x: x[0])
            improved = candidates[0][0] < best_score

        if improved:
            best_candidate = candidates[0][1]
            best_score = candidates[0][0]
            best_mse = candidates[0][2]

            selected_features.append(best_candidate)
            remaining_features.remove(best_candidate)

            trace.append(
                {
                    "Step": step,
                    "Action": f"Added feature: {best_candidate}",
                    "Criterion": criterion.upper(),
                    "Criterion_Score": best_score,
                    "MSE_Score": best_mse,
                    "Features_Used": ", ".join(selected_features),
                }
            )
            step += 1
        else:
            trace.append(
                {
                    "Step": step,
                    "Action": f"STOPPED: No further feature improved {criterion.upper()}",
                    "Criterion": criterion.upper(),
                    "Criterion_Score": best_score,
                    "MSE_Score": best_mse if step > 1 else None,
                    "Features_Used": ", ".join(selected_features),
                }
            )
            break

    return selected_features, trace


# ---------------------------------------------------------
# 4. BACKWARD ELIMINATION
# ---------------------------------------------------------
def backward_elimination(X_train, y_train, criterion="aic"):
    """
    Iteratively removes the least contributing features from a full model.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target variable.
        criterion (str): Evaluation metric ('aic', 'bic', or 'adj_r2').

    Returns:
        tuple: (List of selected features, Execution trace list).
    """
    print(f"      [~] Running Backward Elimination (Criterion: {criterion.upper()})...")
    selected_features = list(X_train.columns)
    current_model = fit_ols_for_selection(X_train, y_train, selected_features)

    current_score = (
        current_model.rsquared_adj
        if criterion == "adj_r2"
        else (current_model.aic if criterion == "aic" else current_model.bic)
    )
    current_mse = current_model.mse_resid

    trace = [
        {
            "Step": 0,
            "Action": "Initial state (All features)",
            "Criterion": criterion.upper(),
            "Criterion_Score": current_score,
            "MSE_Score": current_mse,
            "Features_Used": ", ".join(selected_features),
        }
    ]

    step = 1

    while len(selected_features) > 1:
        candidates = []
        for feature in selected_features:
            candidate_features = [f for f in selected_features if f != feature]
            model = fit_ols_for_selection(X_train, y_train, candidate_features)

            score = (
                model.rsquared_adj
                if criterion == "adj_r2"
                else (model.aic if criterion == "aic" else model.bic)
            )
            candidates.append((score, candidate_features, feature, model.mse_resid))

        if criterion == "adj_r2":
            candidates.sort(reverse=True, key=lambda x: x[0])
            improved = candidates[0][0] > current_score
        else:
            candidates.sort(key=lambda x: x[0])
            improved = candidates[0][0] < current_score

        if improved:
            removed_feature = candidates[0][2]
            selected_features = candidates[0][1]
            current_score = candidates[0][0]
            current_mse = candidates[0][3]

            trace.append(
                {
                    "Step": step,
                    "Action": f"Removed feature: {removed_feature}",
                    "Criterion": criterion.upper(),
                    "Criterion_Score": current_score,
                    "MSE_Score": current_mse,
                    "Features_Used": ", ".join(selected_features),
                }
            )
            step += 1
        else:
            trace.append(
                {
                    "Step": step,
                    "Action": f"STOPPED: Removing any feature worsens {criterion.upper()}",
                    "Criterion": criterion.upper(),
                    "Criterion_Score": current_score,
                    "MSE_Score": current_mse,
                    "Features_Used": ", ".join(selected_features),
                }
            )
            break

    return selected_features, trace


# ---------------------------------------------------------
# 5. LASSO SELECTION
# ---------------------------------------------------------
def lasso_selection(X_train, y_train):
    """
    Performs feature selection using L1 Regularization (LassoCV).

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target variable.

    Returns:
        tuple: (List of selected features, Execution trace list).
    """
    print("      [~] Running LASSO Embedded Selection...")

    # L1 penalty requires scaled features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    lasso_model = LassoCV(cv=5, random_state=42, max_iter=10000)
    lasso_model.fit(X_scaled, y_train)

    lasso_coef = pd.Series(lasso_model.coef_, index=X_train.columns)
    selected_features = list(lasso_coef[lasso_coef.abs() > 1e-6].index)

    trace = [
        {
            "Step": 1,
            "Action": "Applied L1 Penalty (Alpha determined via CV)",
            "Criterion": "ALPHA",
            "Criterion_Score": lasso_model.alpha_,
            "MSE_Score": None,
            "Features_Used": ", ".join(selected_features),
        }
    ]
    return selected_features, trace


# ---------------------------------------------------------
# 6. FILTER SELECTION
# ---------------------------------------------------------
def filter_selection(X_train, y_train, top_k=20):
    """
    Selects top K features based on univariate linear regression tests (F-test).

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target variable.
        top_k (int): Maximum number of features to select.

    Returns:
        tuple: (List of selected features, Execution trace list).
    """
    total_features = X_train.shape[1]
    print(
        f"      [~] Running Univariate Filter Selection (Top {top_k}/{total_features} features)..."
    )

    if top_k >= total_features:
        trace = [
            {
                "Step": 1,
                "Action": "Kept all features (top_k >= total features)",
                "Criterion": "F-STAT",
                "Criterion_Score": None,
                "MSE_Score": None,
                "Features_Used": ", ".join(X_train.columns),
            }
        ]
        return list(X_train.columns), trace

    selector = SelectKBest(score_func=f_regression, k=top_k)
    selector.fit(X_train, y_train)

    selected_cols = list(X_train.columns[selector.get_support()])

    trace = [
        {
            "Step": 1,
            "Action": f"Selected Top {top_k} based on F-Statistic",
            "Criterion": "F-STAT",
            "Criterion_Score": None,
            "MSE_Score": None,
            "Features_Used": ", ".join(selected_cols),
        }
    ]
    return selected_cols, trace


# ---------------------------------------------------------
# 7. PIPELINE ENTRY POINT
# ---------------------------------------------------------
def run_feature_selection(
    df_train, target_column="amount", method="lasso", criterion="aic", top_k=20
):
    """
    Master router function for initiating specific feature selection algorithms.

    Args:
        df_train (pd.DataFrame): The full training dataset including the target.
        target_column (str): The name of the target variable column.
        method (str): Algorithm to execute (e.g., 'forward', 'lasso').
        criterion (str): Evaluation metric ('aic', 'bic', or 'adj_r2').
        top_k (int): Parameter used by filter and hybrid methods.

    Returns:
        tuple: (List of selected features, Execution trace list).
    """
    X_train = df_train.drop(columns=[target_column])
    y_train = df_train[target_column]

    if method == "best_subset":
        print(f"      [!] Initiating Hybrid Strategy: KBest (Filter) + Best Subset")

        safe_k = min(top_k, 15)
        if top_k > 15:
            print(
                f"      [!] Warning: top_k={top_k} is too large for Best Subset. Capping at {safe_k} to prevent memory overflow."
            )

        filtered_cols, _ = filter_selection(X_train, y_train, top_k=safe_k)
        X_train_filtered = X_train[filtered_cols]
        return best_subset_selection(X_train_filtered, y_train, criterion)

    elif method == "forward":
        return forward_selection(X_train, y_train, criterion)

    elif method == "backward":
        return backward_elimination(X_train, y_train, criterion)

    elif method == "lasso":
        return lasso_selection(X_train, y_train)

    elif method == "filter":
        return filter_selection(X_train, y_train, top_k)

    else:
        print(
            f"      [!] Warning: Invalid method '{method}'. Defaulting to all features."
        )
        fallback_trace = [
            {
                "Step": 1,
                "Action": "Fallback to baseline",
                "Criterion": "NONE",
                "Criterion_Score": None,
                "MSE_Score": None,
                "Features_Used": ", ".join(X_train.columns),
            }
        ]
        return list(X_train.columns), fallback_trace
