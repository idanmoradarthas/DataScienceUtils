from pathlib import Path

import pandas

from ds_utils.preprocess import get_correlated_features


def test_get_correlated_features():
    data_frame = pandas.read_csv(Path(__file__).parents[0].joinpath("resources").joinpath("loan_final313_small.csv"))
    correlation = get_correlated_features(data_frame, data_frame.columns.drop("loan_condition_cat").tolist(),
                                          "loan_condition_cat", 0.95)
    correlation_expected = pandas.DataFrame([{'level_0': 'income_category_Low', 'level_1': 'income_category_Medium',
                                              'level_0_level_1_corr': -0.9999999999999999,
                                              'level_0_target_corr': -0.11821656093586508,
                                              'level_1_target_corr': 0.11821656093586504},
                                             {'level_0': 'term_ 36 months', 'level_1': 'term_ 60 months',
                                              'level_0_level_1_corr': -1.0,
                                              'level_0_target_corr': -0.11821656093586508,
                                              'level_1_target_corr': 0.11821656093586504},
                                             {'level_0': 'interest_payments_High', 'level_1': 'interest_payments_Low',
                                              'level_0_level_1_corr': -1.0, 'level_0_target_corr': -0.11821656093586508,
                                              'level_1_target_corr': 0.11821656093586504}],
                                            columns=['level_0', 'level_1', 'level_0_level_1_corr',
                                                     'level_0_target_corr',
                                                     'level_1_target_corr'])
    pandas.testing.assert_frame_equal(correlation_expected, correlation)
