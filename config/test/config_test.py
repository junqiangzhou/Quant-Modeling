import pytest

from config.config import feature_names, label_names, label_columns

expected_feature_names = [
    'Open_diff', 'High_diff', 'Low_diff', 'Close_diff', 'Volume_diff',
    'MA_5_diff', 'MA_10_diff', 'MA_20_diff', 'MA_50_diff',
    'MA_5_20_Crossover_Signal_0', 'MA_5_20_Crossover_Signal_-1',
    'MA_5_20_Crossover_Signal_1', 'MA_10_50_Crossover_Signal_0',
    'MA_10_50_Crossover_Signal_-1', 'MA_10_50_Crossover_Signal_1',
    'MACD_Crossover_Signal_0', 'MACD_Crossover_Signal_-1',
    'MACD_Crossover_Signal_1', 'RSI_Over_Bought_Signal_0',
    'RSI_Over_Bought_Signal_-1', 'RSI_Over_Bought_Signal_1', 'BB_Signal_0',
    'BB_Signal_-1', 'BB_Signal_1', 'Price_Above_MA_5', 'Price_Below_MA_5',
    'MA_5_10_Crossover_Signal', 'MA_5_50_Crossover_Signal',
    'MA_10_20_Crossover_Signal', 'MA_20_50_Crossover_Signal',
    'VWAP_Crossover_Signal', 'daily_change'
]


def test_feature_names_match_expected():
    assert len(feature_names) == 32
    assert feature_names == expected_feature_names, "Feature names do not match the expected list"


expected_label_names = [
    'trend_5days', 'trend_10days', 'trend_20days', 'trend_30days'
]


def test_label_names_match_expected():
    assert len(label_names) == 4
    assert label_names == expected_label_names, "Label names do not match the expected list"


def test_label_columns_match_expected():
    assert len(label_columns) == 20
    # expect every 5th column to be a label
    assert label_columns[
        4::
        5] == expected_label_names, "Label columns do not match the expected list"
