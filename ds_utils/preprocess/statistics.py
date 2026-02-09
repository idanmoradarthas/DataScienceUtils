import warnings
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from numpy.random import RandomState
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from ds_utils.math_utils import safe_percentile


def get_correlated_features(
    correlation_matrix: pd.DataFrame, features: List[str], target_feature: str, threshold: float = 0.95
) -> pd.DataFrame:
    """Calculate features correlated above a threshold with target correlations.

    Calculate features correlated above a threshold and extract a DataFrame with correlations and correlation
    to the target feature.

    :param correlation_matrix: The correlation matrix.
    :param features: List of feature names to analyze.
    :param target_feature: Name of the target feature.
    :param threshold: Correlation threshold (default 0.95).
    :return: DataFrame with correlations and correlation to the target feature.
    """
    target_corr = correlation_matrix[target_feature]
    features_corr = correlation_matrix.loc[features, features]
    corr_matrix = features_corr.where(np.triu(np.ones(features_corr.shape), k=1).astype(bool))
    corr_matrix = corr_matrix[~np.isnan(corr_matrix)].stack().reset_index()
    corr_matrix = corr_matrix[corr_matrix[0].abs() >= threshold]

    if corr_matrix.empty:
        warnings.warn(f"Correlation threshold {threshold} was too high. An empty frame was returned", UserWarning)
        return pd.DataFrame(
            columns=["level_0", "level_1", "level_0_level_1_corr", "level_0_target_corr", "level_1_target_corr"]
        )

    corr_matrix["level_0_target_corr"] = target_corr[corr_matrix["level_0"]].values
    corr_matrix["level_1_target_corr"] = target_corr[corr_matrix["level_1"]].values
    corr_matrix = corr_matrix.rename({0: "level_0_level_1_corr"}, axis=1).reset_index(drop=True)
    return corr_matrix


def extract_statistics_dataframe_per_label(df: pd.DataFrame, feature_name: str, label_name: str) -> pd.DataFrame:
    """Calculate comprehensive statistical metrics for a specified feature grouped by label.

    This method computes various statistical measures for a given numerical feature, broken down by unique
    values in the specified label column. The statistics include count, null count,
    mean, standard deviation, min/max values and multiple percentiles.

    :param df: Input pandas DataFrame containing the data
    :param feature_name: Name of the column to calculate statistics on
    :param label_name: Name of the column to group by
    :return: DataFrame with statistical metrics for each unique label value, with columns:
            - count: Number of non-null observations
            - null_count: Number of null values
            - mean: Average value
            - min: Minimum value
            - 1_percentile: 1st percentile
            - 5_percentile: 5th percentile
            - 25_percentile: 25th percentile
            - median: 50th percentile
            - 75_percentile: 75th percentile
            - 95_percentile: 95th percentile
            - 99_percentile: 99th percentile
            - max: Maximum value

    :raises KeyError: If feature_name or label_name is not found in DataFrame
    :raises TypeError: If feature_name column is not numeric
    """
    if feature_name not in df.columns:
        raise KeyError(f"Feature column '{feature_name}' not found in DataFrame")
    if label_name not in df.columns:
        raise KeyError(f"Label column '{label_name}' not found in DataFrame")
    if not pd.api.types.is_numeric_dtype(df[feature_name]):
        raise TypeError(f"Feature column '{feature_name}' must be numeric")

    # Define percentile functions with consistent naming

    def percentile_1(x):
        return safe_percentile(x, 1)

    def percentile_5(x):
        return safe_percentile(x, 5)

    def percentile_25(x):
        return safe_percentile(x, 25)

    def percentile_75(x):
        return safe_percentile(x, 75)

    def percentile_95(x):
        return safe_percentile(x, 95)

    def percentile_99(x):
        return safe_percentile(x, 99)

    return df.groupby([label_name], observed=True)[feature_name].agg(
        [
            ("count", "count"),
            ("null_count", lambda x: x.isnull().sum()),
            ("mean", "mean"),
            ("min", "min"),
            ("1_percentile", percentile_1),
            ("5_percentile", percentile_5),
            ("25_percentile", percentile_25),
            ("median", "median"),
            ("75_percentile", percentile_75),
            ("95_percentile", percentile_95),
            ("99_percentile", percentile_99),
            ("max", "max"),
        ]
    )


def compute_mutual_information(
    df: pd.DataFrame,
    features: List[str],
    label_col: str,
    *,
    n_neighbors: int = 3,
    random_state: Optional[Union[int, RandomState]] = None,
    n_jobs: Optional[int] = None,
    numerical_imputer: TransformerMixin = SimpleImputer(strategy="mean"),
    discrete_imputer: TransformerMixin = SimpleImputer(strategy="most_frequent"),
    discrete_encoder: TransformerMixin = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
) -> pd.DataFrame:
    """Compute mutual information scores between features and a target label.

    This function calculates mutual information scores for specified features with respect to a target
    label column. Features are automatically categorized as numerical or discrete (boolean/categorical)
    and preprocessed accordingly before computing mutual information.

    Any feature column that contains only null (NaN) values will be ignored and assigned a mutual
    information score of 0. A `UserWarning` will be issued listing any such columns.

    Mutual information measures the mutual dependence between two variables - higher scores indicate
    stronger relationships between the feature and the target label.

    :param df: Input pandas DataFrame containing the features and label
    :param features: List of column names to compute mutual information for
    :param label_col: Name of the target label column
    :param n_neighbors: Number of neighbors to use for MI estimation for continuous variables. Higher values
                        reduce variance of the estimation, but could introduce a bias.
    :param random_state: Random state for reproducible results. Can be int or RandomState instance
    :param n_jobs: The number of jobs to use for computing the mutual information. The parallelization is done
                   on the columns. `None` means 1 unless in a `joblib.parallel_backend` context. ``-1`` means
                   using all processors.
    :param numerical_imputer: Sklearn-compatible transformer for numerical features (default: mean imputation)
    :param discrete_imputer: Sklearn-compatible transformer for discrete features (default: most frequent imputation)
    :param discrete_encoder: Sklearn-compatible transformer for encoding discrete features (default: ordinal encoding
                            with unknown value handling)
    :return: DataFrame with columns 'feature_name' and 'mi_score', sorted by MI score (descending)

    :raises KeyError: If any feature or label_col is not found in DataFrame
    :raises ValueError: If features list is empty or label_col contains non-finite values
    :warns UserWarning: If one or more feature columns contain only null values.
    """
    # Input validation
    if not features:
        raise ValueError("features list cannot be empty")

    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not found in DataFrame")

    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise KeyError(f"Features not found in DataFrame: {missing_features}")

    if df[label_col].isnull().all():
        raise ValueError(f"Label column '{label_col}' contains only null values")

    # Identify and separate fully missing features
    fully_missing_features = [f for f in features if df[f].isnull().all()]
    if fully_missing_features:
        warnings.warn(f"Features {fully_missing_features} contain only null values and will be ignored.", UserWarning)
    features_to_process = [f for f in features if f not in fully_missing_features]

    # Create a DataFrame for missing features with MI score of 0
    missing_mi_df = pd.DataFrame({"feature_name": fully_missing_features, "mi_score": 0.0})

    # If all features were missing or no features to process, return the DataFrame of missing features
    if not features_to_process:
        return missing_mi_df.sort_values(by="feature_name").reset_index(drop=True)

    # Identify feature types for the features that will be processed
    df_processed = df[features_to_process].copy()
    numerical_features = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    boolean_features = df_processed.select_dtypes(include=["bool", "boolean"]).columns.tolist()
    categorical_features = df_processed.select_dtypes(include=["object", "category"]).columns.tolist()

    # SimpleImputer does not support boolean dtype, so convert to object
    for col in boolean_features:
        df_processed[col] = df_processed[col].astype(object)

    # Create preprocessing pipelines
    numerical_transformer = Pipeline(steps=[("imputer", numerical_imputer)], memory=None, verbose=False)
    discrete_transformer = Pipeline(
        steps=[("imputer", discrete_imputer), ("encoder", discrete_encoder)], memory=None, verbose=False
    )

    # Setup column transformer
    transformers = []
    if numerical_features:
        transformers.append(("num", numerical_transformer, numerical_features))
    if boolean_features or categorical_features:
        transformers.append(("discrete", discrete_transformer, boolean_features + categorical_features))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0,
        n_jobs=n_jobs,
        transformer_weights=None,
        verbose=False,
        verbose_feature_names_out=True,
    )

    # Create discrete features mask for mutual_info_classif
    discrete_features_mask = [False] * len(numerical_features) + [True] * (
        len(boolean_features) + len(categorical_features)
    )

    # Create ordered feature names list matching the preprocessed data
    ordered_feature_names = numerical_features + boolean_features + categorical_features

    # Apply preprocessing
    x_preprocessed = preprocessor.fit_transform(df_processed[ordered_feature_names])
    y = df[label_col]

    # Compute mutual information scores
    mi_scores = mutual_info_classif(
        X=x_preprocessed,
        y=y,
        n_neighbors=n_neighbors,
        copy=True,
        random_state=random_state,
        n_jobs=n_jobs,
        discrete_features=discrete_features_mask,
    )

    # Create results DataFrame for processed features
    processed_mi_df = pd.DataFrame({"feature_name": ordered_feature_names, "mi_score": mi_scores})

    # Combine with missing features' results
    final_mi_df = pd.concat([processed_mi_df, missing_mi_df], ignore_index=True)

    return final_mi_df.sort_values(by="mi_score", ascending=False).reset_index(drop=True)
