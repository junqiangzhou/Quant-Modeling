import pytest

from config.config import feature_names, label_names, label_debug_columns

expected_feature_names = [
    'Open_diff', 'High_diff', 'Low_diff', 'Close_diff', 'Volume_diff',
    'MA_5_diff', 'MA_10_diff', 'MA_20_diff', 'MA_50_diff',
    'MA_5_20_Crossover_Signal_0', 'MA_5_20_Crossover_Signal_-1',
    'MA_5_20_Crossover_Signal_1', 'MA_5_10_Crossover_Signal_0',
    'MA_5_10_Crossover_Signal_-1', 'MA_5_10_Crossover_Signal_1',
    'MA_5_50_Crossover_Signal_0', 'MA_5_50_Crossover_Signal_-1',
    'MA_5_50_Crossover_Signal_1', 'MA_10_20_Crossover_Signal_0',
    'MA_10_20_Crossover_Signal_-1', 'MA_10_20_Crossover_Signal_1',
    'MA_20_50_Crossover_Signal_0', 'MA_20_50_Crossover_Signal_-1',
    'MA_20_50_Crossover_Signal_1', 'MACD_Crossover_Signal_0',
    'MACD_Crossover_Signal_-1', 'MACD_Crossover_Signal_1',
    'VWAP_Crossover_Signal_0', 'VWAP_Crossover_Signal_-1',
    'VWAP_Crossover_Signal_1', 'Price_Above_MA_5', 'Price_Below_MA_5',
    'daily_change'
]


def test_feature_names_match_expected():
    assert len(feature_names) == 33
    assert feature_names == expected_feature_names, "Feature names do not match the expected list"


expected_label_names = [
    'trend_10days', 'trend_20days', 'trend_40days', 'trend_60days'
]


def test_label_names_match_expected():
    assert len(label_names) == 4
    assert label_names == expected_label_names, "Label names do not match the expected list"


def test_label_debug_columns_match_expected():
    assert len(label_debug_columns) == 20
    # expect every 5th column to be a label
    assert label_debug_columns[
        4::
        5] == expected_label_names, "Label columns do not match the expected list"
