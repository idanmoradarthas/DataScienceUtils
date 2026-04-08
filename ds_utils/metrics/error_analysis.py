"""Error Analysis Module.

This module provides functions to generate tabular error-analysis reports
to help identify specific feature ranges or categories where a model underperforms.
"""

from typing import List, Optional

import numpy as np
import pandas as pd


def generate_error_analysis_report(
    X: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_columns: Optional[List[str]] = None,
    bins: int = 10,
    threshold: float = 0.5,
    min_count: int = 1,
    sort_metric: str = "error_rate",
    ascending: bool = False,
) -> pd.DataFrame:
    """Generate a tabular error-analysis report grouped by feature values.

    The report groups predictions by feature values and computes error metrics per group.
    For numerical features, values are binned into equal-width bins. For categorical
    features, raw values are used as groups.

    :param X: Feature DataFrame.
    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :param feature_columns: List of columns to analyze. If None, all columns in X are used.
    :param bins: Number of bins for numerical features.
    :param threshold: Threshold for probability-based error definitions.
                      Validated but not used in the current implementation;
                      reserved for future probability-based error definitions.
    :param min_count: Minimum number of samples in a group to be included in the report.
    :param sort_metric: Metric to sort the report by. Valid options are:
                        'feature', 'group', 'count', 'error_count', 'error_rate', 'accuracy'.
    :param ascending: Whether to sort in ascending order.
    :return: DataFrame containing the error analysis report.
    :raises ValueError: If bins < 1, threshold not in [0, 1], min_count < 1, or invalid sort_metric.
    :raises KeyError: If any column in feature_columns is missing from X.
    """
    if bins < 1:
        raise ValueError("bins must be at least 1")
    if not (0 <= threshold <= 1):
        raise ValueError("threshold must be between 0 and 1 inclusive")
    if min_count < 1:
        raise ValueError("min_count must be at least 1")

    valid_columns = ["feature", "group", "count", "error_count", "error_rate", "accuracy"]
    if sort_metric not in valid_columns:
        raise ValueError(f"sort_metric must be one of {valid_columns}")

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) != len(X) or len(y_pred) != len(X):
        raise ValueError("X, y_true, and y_pred must have the same number of samples")

    if feature_columns is not None:
        missing_cols = [col for col in feature_columns if col not in X.columns]
        if missing_cols:
            raise KeyError(f"The following columns are missing from X: {missing_cols}")
        cols_to_use = feature_columns
    else:
        cols_to_use = X.columns.tolist()

    internal_df = X[cols_to_use].copy()
    internal_df["__is_error__"] = y_true != y_pred

    all_reports = []
    for col in cols_to_use:
        col_series = internal_df[col]
        if pd.api.types.is_numeric_dtype(col_series):
            groups = pd.cut(col_series, bins=bins)
        else:
            groups = col_series

        report = (
            internal_df.groupby(groups, observed=True)
            .agg(count=("__is_error__", "size"), error_count=("__is_error__", "sum"))
            .reset_index()
        )
        report.rename(columns={col: "group"}, inplace=True)
        report["feature"] = col
        report["error_rate"] = report["error_count"] / report["count"]
        report["accuracy"] = 1 - report["error_rate"]

        report = report[report["count"] >= min_count]
        all_reports.append(report[valid_columns])

    if not all_reports:
        return pd.DataFrame(columns=valid_columns)

    final_report = pd.concat(all_reports).sort_values(by=sort_metric, ascending=ascending).reset_index(drop=True)
    return final_report
