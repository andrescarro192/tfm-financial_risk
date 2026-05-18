# src/data/temporal_features.py

import pandas as pd

def build_temporal_features(
    df: pd.DataFrame,
    entity_col: str = "CERT",
    period_col: str = "period",
    feature_subset: list = None,  
    lags: list = [1, 2, 3],
    windows: list = [3],
    growth_rate: bool = True
) -> pd.DataFrame:

    df = df.sort_values([entity_col, period_col]).reset_index(drop=True)

    all_feature_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    all_feature_cols = [c for c in all_feature_cols if c not in [entity_col, period_col]]

    # Si se especifica subset, aplicar temporalidad solo a esas variables
    temporal_cols = feature_subset if feature_subset else all_feature_cols

    out = df.copy()
    grouped = out.groupby(entity_col)

    for lag in lags:
        for col in temporal_cols:
            out[f"{col}_lag{lag}"] = grouped[col].shift(lag)

    for col in temporal_cols:
        out[f"{col}_diff"] = grouped[col].diff()

    if growth_rate:
        for col in temporal_cols:
            prev = grouped[col].shift(1)
            out[f"{col}_growth"] = (out[col] - prev) / prev.abs().replace(0, float("nan"))

    for w in windows:
        for col in temporal_cols:
            out[f"{col}_roll{w}"] = (
                grouped[col]
                .transform(lambda x: x.rolling(w, min_periods=1).mean())
            )

    return out
