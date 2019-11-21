from typing import List

import numpy
import pandas


def get_correlated_features(data_frame: pandas.DataFrame, features: List[str], target_feature: str,
                            threshold: float = 0.95) -> pandas.DataFrame:
    """
    Calculate which features correlated above a threshold and extract a data frame with the correlations and correlation
    to the target feature.

    :param data_frame: the data frame.
    :param features: list of features names.
    :param target_feature: name of target feature.
    :param threshold: the threshold (default 0.95).
    :return: data frame with the correlations and correlation to the target feature.
    """
    correlations = data_frame[features + [target_feature]].corr()
    target_corr = correlations[target_feature].transpose()
    features_corr = correlations.loc[features, features]
    corr_matrix = features_corr.where(numpy.triu(numpy.ones(features_corr.shape), k=1).astype(numpy.bool))
    corr_matrix = corr_matrix[(~numpy.isnan(corr_matrix))].stack().reset_index()
    corr_matrix = corr_matrix[corr_matrix[0].abs() >= threshold]
    corr_matrix["level_0_target_corr"] = target_corr[corr_matrix["level_0"]].values.tolist()[0]
    corr_matrix["level_1_target_corr"] = target_corr[corr_matrix["level_1"]].values.tolist()[0]
    corr_matrix = corr_matrix.rename({0: "level_0_level_1_corr"}, axis=1).reset_index(drop=True)
    return corr_matrix
