import pandas as pd


def _copy_series_or_keep_top_10(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.map({True: "True", False: "False"})
    if len(series.unique()) > 10:
        top10 = series.value_counts().nlargest(10).index
        return series.map(lambda x: x if x in top10 else "Other values")
    return series


def _is_categorical_like(dtype):
    """Check if the dtype is categorical-like (categorical, boolean, or object)."""
    return (
        isinstance(dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(dtype)
        or pd.api.types.is_object_dtype(dtype)
    )
