import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from itertools import combinations


def fit_ols_for_selection(X_data, y_data, selected_features):
    """
    Hàm phụ trợ: Xây dựng mô hình OLS bằng statsmodels.
    Dùng để lấy các chỉ số AIC, BIC, Adj_R2 trong quá trình chọn biến.
    """
    # Ép kiểu cứng: chỉ lấy cột số, loại bỏ object/string/boolean
    X_subset = (
        X_data[selected_features]
        .select_dtypes(include=[np.number])
        .fillna(0)
        .astype(float)
    )
    y_subset = y_data.astype(float)

    # Kiểm tra nếu sau khi lọc mà trống rỗng thì trả về model rỗng (để tránh crash)
    if X_subset.empty:

        class EmptyModel:
            aic = bic = rsquared_adj = np.inf

        return EmptyModel()

    X_const = sm.add_constant(X_subset, has_constant="add")
    model = sm.OLS(y_subset, X_const).fit()
    return model


# =====================================================================
# 1. BEST SUBSET SELECTION
# =====================================================================
def best_subset_selection(X_train, y_train, criterion="aic"):
    """
    Thực hiện Best subset selection.

    Lý thuyết: Với p biến dự báo, phương pháp này sẽ xét tất cả các
    tập con biến có thể có. Tổng số mô hình khác rỗng cần xét là 2^p - 1.
    Ta chọn mô hình dựa trên Adjusted R2 lớn nhất, hoặc AIC/BIC nhỏ nhất.
    """
    print(
        f"      [~] Running Best Subset Selection (Criterion: {criterion.upper()})..."
    )
    features = list(X_train.columns)
    n_features = len(features)

    # AIC/BIC cần giá trị nhỏ nhất (khởi tạo vô cực), Adj R2 cần lớn nhất (khởi tạo âm vô cực)
    best_score = -np.inf if criterion == "adj_r2" else np.inf
    best_subset = []

    for k in range(1, n_features + 1):
        for subset in combinations(features, k):
            subset_list = list(subset)
            model = fit_ols_for_selection(X_train, y_train, subset_list)

            if criterion == "adj_r2":
                score = model.rsquared_adj
                improved = score > best_score
            elif criterion == "aic":
                score = model.aic
                improved = score < best_score
            elif criterion == "bic":
                score = model.bic
                improved = score < best_score

            if improved:
                best_score = score
                best_subset = subset_list

    return best_subset


# =====================================================================
# 2. FORWARD SELECTION
# =====================================================================
def forward_selection(X_train, y_train, criterion="aic"):
    """
    Thực hiện Forward selection.

    Lý thuyết: Phương pháp forward selection bắt đầu từ mô hình rỗng.
    Ở mỗi bước, một biến mới được thêm vào nếu việc thêm biến đó giúp
    cải thiện tiêu chí lựa chọn mô hình (như AIC, BIC, hoặc Adjusted R2).
    """
    print(f"      [~] Running Forward Selection (Criterion: {criterion.upper()})...")
    features = list(X_train.columns)
    remaining_features = set(features)
    selected_features = []

    if criterion == "adj_r2":
        best_score = -np.inf
    else:
        best_score = np.inf

    while len(remaining_features) > 0:
        candidates = []
        for feature in sorted(remaining_features):
            candidate_features = selected_features + [feature]
            model = fit_ols_for_selection(X_train, y_train, candidate_features)

            if criterion == "adj_r2":
                score = model.rsquared_adj
            elif criterion == "aic":
                score = model.aic
            elif criterion == "bic":
                score = model.bic
            else:
                raise ValueError("Criterion must be 'adj_r2', 'aic', or 'bic'")

            candidates.append((score, feature))

        # Tìm ứng viên tốt nhất trong vòng lặp (Adj R2 thì giảm dần, AIC/BIC thì tăng dần)
        if criterion == "adj_r2":
            candidates.sort(reverse=True, key=lambda x: x[0])
            improved = candidates[0][0] > best_score
        else:
            candidates.sort(key=lambda x: x[0])
            improved = candidates[0][0] < best_score

        if improved:
            selected_features.append(candidates[0][1])
            remaining_features.remove(candidates[0][1])
            best_score = candidates[0][0]
        else:
            break

    return selected_features


# =====================================================================
# 3. BACKWARD ELIMINATION
# =====================================================================
def backward_elimination(X_train, y_train, criterion="aic"):
    """
    Thực hiện Backward elimination.

    Lý thuyết: Phương pháp backward elimination bắt đầu từ mô hình đầy đủ.
    Ở mỗi bước, một biến được loại bỏ nếu việc loại bỏ biến đó giúp
    cải thiện tiêu chí lựa chọn mô hình (như AIC, BIC, hoặc Adjusted R2).
    """
    print(f"      [~] Running Backward Elimination (Criterion: {criterion.upper()})...")
    selected_features = list(X_train.columns)
    current_model = fit_ols_for_selection(X_train, y_train, selected_features)

    if criterion == "adj_r2":
        current_score = current_model.rsquared_adj
    elif criterion == "aic":
        current_score = current_model.aic
    elif criterion == "bic":
        current_score = current_model.bic

    while len(selected_features) > 1:
        candidates = []
        for feature in selected_features:
            candidate_features = [f for f in selected_features if f != feature]
            model = fit_ols_for_selection(X_train, y_train, candidate_features)

            if criterion == "adj_r2":
                score = model.rsquared_adj
            elif criterion == "aic":
                score = model.aic
            elif criterion == "bic":
                score = model.bic

            candidates.append((score, candidate_features))

        if criterion == "adj_r2":
            candidates.sort(reverse=True, key=lambda x: x[0])
            improved = candidates[0][0] > current_score
        else:
            candidates.sort(key=lambda x: x[0])
            improved = candidates[0][0] < current_score

        if improved:
            selected_features = candidates[0][1]
            current_score = candidates[0][0]
        else:
            break

    return selected_features


# =====================================================================
# 4. LASSO SELECTION
# =====================================================================
def lasso_selection(X_train, y_train):
    """
    Thực hiện LASSO Selection sử dụng LassoCV để tìm alpha tốt nhất.

    Lý thuyết : LASSO là phương pháp thuộc nhóm embedded methods.
    Nó thêm phần phạt L1 vào hàm mất mát, nhờ đó có thể đưa một số hệ số hồi quy về đúng 0.
    Các biến có hệ số bằng 0 được xem là bị loại khỏi mô hình.
    Lưu ý: Các biến dự báo cần được chuẩn hóa trước khi áp dụng.
    """
    print("      [~] Running LASSO Embedded Selection...")
    # Bước chuẩn hóa bắt buộc theo lý thuyết LASSO
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    lasso_model = LassoCV(cv=5, random_state=42, max_iter=10000)
    lasso_model.fit(X_scaled, y_train)

    lasso_coef = pd.Series(lasso_model.coef_, index=X_train.columns)
    selected_features = list(lasso_coef[lasso_coef.abs() > 1e-6].index)
    return selected_features


# =====================================================================
# 5. FILTER SELECTION (Tiền xử lý)
# =====================================================================
def filter_selection(X_train, y_train, top_k=20):
    """
    Thực hiện Filter Selection sử dụng f_regression.
    Đây là phương pháp thuộc nhóm Filter method, đánh giá mức độ tương quan
    tuyến tính độc lập của từng biến dự báo với biến mục tiêu thông qua kiểm định F.
    """
    total_features = X_train.shape[1]
    print(
        f"      [~] Running Univariate Filter Selection (Top {top_k}/{total_features} features)..."
    )

    if top_k >= total_features:
        return list(X_train.columns)

    selector = SelectKBest(score_func=f_regression, k=top_k)
    selector.fit(X_train, y_train)

    selected_cols = list(X_train.columns[selector.get_support()])
    return selected_cols


# =====================================================================
# (ENTRY POINT)
# =====================================================================
def run_feature_selection(
    df_train, target_column="amount", method="lasso", criterion="aic", top_k=20
):
    """
    Hàm giao diện (Entry point). Main.py chỉ cần gọi hàm này.
    method: "forward", "backward", "lasso", "filter", "best_subset"
    """
    X_train = df_train.drop(columns=[target_column])
    y_train = df_train[target_column]

    if method == "best_subset":

        # CHIẾN LƯỢC LAI GHÉP: Chạy KBest trước, sau đó mới chạy Best Subset
        print(f"      [!] Kích hoạt chiến lược Lai ghép (Hybrid): KBest + Best Subset")

        # Màng lọc bảo vệ: Giới hạn top_k tối đa là 15
        safe_k = min(top_k, 15)
        if top_k > 15:
            print(
                f"      [!] Cảnh báo: top_k={top_k} quá lớn. Tự động giảm xuống {safe_k}."
            )

        filtered_cols = filter_selection(X_train, y_train, top_k=safe_k)
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
            f"      [!] Warning: Method '{method}' is invalid. Returning all features."
        )
        return list(X_train.columns)
