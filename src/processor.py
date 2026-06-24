import pandas as pd
import numpy as np
from category_encoders import TargetEncoder


# =====================================================================
# HELPER
# =====================================================================

def _encode_facing(df):
    """
    Convert facing into four directional features.

    Unknown => all zeros
    North-East => north=1, east=1
    South-West => south=1, west=1
    """

    if "facing" not in df.columns:
        return df

    facing = (
        df["facing"]
        .fillna("Unknown")
        .astype(str)
        .str.lower()
    )

    df["facing_north"] = (
        facing.str.contains("north", na=False)
    ).astype(np.int8)

    df["facing_south"] = (
        facing.str.contains("south", na=False)
    ).astype(np.int8)

    df["facing_east"] = (
        facing.str.contains("east", na=False)
    ).astype(np.int8)

    df["facing_west"] = (
        facing.str.contains("west", na=False)
    ).astype(np.int8)

    df.drop(columns=["facing"], inplace=True)

    return df


# =====================================================================
# MAIN FEATURE PROCESSOR
# =====================================================================

def fit_transform_features(
    X_train,
    X_test,
    y_train,
    numeric_cols,
    categorical_cols,
):
    """
    Feature Processing Pipeline

    Steps
    -----
    1. Numeric Imputation
    2. Outlier Clipping
    3. Categorical Imputation
    4. Location Target Encoding
    5. Facing Direction Encoding
    6. One-Hot Encoding (remaining categoricals)
    7. Train/Test Alignment
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

        X_train[col] = X_train[col].fillna(
            median_value
        )

        X_test[col] = X_test[col].fillna(
            median_value
        )

    # ==================================================
    # Outlier Clipping
    # ==================================================

    clip_cols = [
        "area",
        "room_floor",
        "total_floor",
        "bathroom",
        "balcony",
    ]

    for col in clip_cols:

        if col not in X_train.columns:
            continue

        upper_cap = (
            X_train[col]
            .quantile(0.995)
        )

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

        X_train[col] = (
            X_train[col]
            .fillna("Unknown")
        )

        X_test[col] = (
            X_test[col]
            .fillna("Unknown")
        )

    # ==================================================
    # LOCATION TARGET ENCODING
    # ==================================================

    if "location" in X_train.columns:

        location_encoder = TargetEncoder(
            cols=["location"],
            smoothing=20,
        )

        X_train["median_amount_per_location"] = (
            location_encoder
            .fit_transform(
                X_train[["location"]],
                y_train,
            )["location"]
        )

        X_test["median_amount_per_location"] = (
            location_encoder
            .transform(
                X_test[["location"]]
            )["location"]
        )

        X_train.drop(
            columns=["location"],
            inplace=True,
        )

        X_test.drop(
            columns=["location"],
            inplace=True,
        )

    # ==================================================
    # FACING FEATURE ENGINEERING
    # ==================================================

    X_train = _encode_facing(X_train)
    X_test = _encode_facing(X_test)

    # ==================================================
    # REMAINING CATEGORICALS
    # ==================================================

    remaining_cat_cols = [
        col
        for col in categorical_cols
        if col not in [
            "location",
            "facing",
        ]
        and col in X_train.columns
    ]

    X_train = pd.get_dummies(
        X_train,
        columns=remaining_cat_cols,
        dtype=np.int8,
    )

    X_test = pd.get_dummies(
        X_test,
        columns=remaining_cat_cols,
        dtype=np.int8,
    )

    # ==================================================
    # ALIGN
    # ==================================================

    X_train, X_test = X_train.align(
        X_test,
        join="left",
        axis=1,
        fill_value=0,
    )

    return X_train, X_test


# =====================================================================
# FEATURE TRANSFORM (LOG / RAW)
# =====================================================================

def apply_feature_transform(
    df,
    config,
    dataset_name="",
):
    """
    Apply optional log transform
    based on config.yaml
    """

    df_transformed = df.copy()

    transform_mode = (
        config
        .get("experiment", {})
        .get(
            "target_transform",
            "raw",
        )
    )

    if transform_mode == "log":

        log_cols = (
            config
            .get("experiment", {})
            .get(
                "log_features",
                [],
            )
        )

        for col in log_cols:

            if col not in df_transformed.columns:
                continue

            df_transformed[col] = (
                df_transformed[col]
                .astype(float)
            )

            df_transformed[col] = np.log1p(
                df_transformed[col]
            )

            print(
                f"   -> [Processor] "
                f"Feature (X - {dataset_name}): "
                f"Applied Log Transform "
                f"to '{col}'."
            )

    else:

        print(
            f"   -> [Processor] "
            f"Feature (X - {dataset_name}): "
            f"RAW mode active "
            f"(Keeping original values)."
        )

    return df_transformed