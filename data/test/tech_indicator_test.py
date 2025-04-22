import pandas as pd
import numpy as np
import pytest

from data.tech_indicator import (add_tech_indicators, add_trading_volume,
                                 add_moving_averages, add_macd, add_kdj,
                                 add_rsi, add_obv, add_vwap,
                                 add_bollinger_bands, add_atr)


# Test trading volume calculation
@pytest.mark.parametrize("data, expected_trading_volume", [
    ({
        'High': [10, 20, 30],
        'Low': [5, 15, 25],
        'Volume': [100, 200, 300]
    }, [100 * (10 + 5) / 2.0, 200 * (20 + 15) / 2.0, 300 * (30 + 25) / 2.0]),
    ({
        'High': [50, 60],
        'Low': [40, 50],
        'Volume': [500, 600]
    }, [500 * (50 + 40) / 2.0, 600 * (60 + 50) / 2.0]),
])
def test_add_trading_volume(data, expected_trading_volume):
    # Create a sample DataFrame
    df = pd.DataFrame(data)

    # Call the function
    result_df = add_trading_volume(df)

    # Assert the 'Trading_Volume' column is correctly added
    assert 'Trading_Volume' in result_df.columns
    assert result_df['Trading_Volume'].tolist() == expected_trading_volume


# Test moving averages calculation
@pytest.mark.parametrize("data, windows, expected_moving_averages", [
    ({
        'Close': [10, 20, 30, 40, 50]
    }, [2, 3], {
        'MA_2': [np.nan, 15.0, 25.0, 35.0, 45.0],
        'MA_3': [np.nan, np.nan, 20.0, 30.0, 40.0]
    }),
    ({
        'Close': [5, 15, 25, 35]
    }, [2], {
        'MA_2': [np.nan, 10.0, 20.0, 30.0]
    }),
])
def test_add_moving_averages(data, windows, expected_moving_averages):
    # Create a sample DataFrame
    df = pd.DataFrame(data)

    # Call the function
    result_df = add_moving_averages(df, windows)

    # Assert the moving averages are correctly added
    for ma_column, expected_values in expected_moving_averages.items():
        assert ma_column in result_df.columns
        assert np.allclose(result_df[ma_column].tolist(),
                           expected_values,
                           equal_nan=True)


# Test MACD calculation
@pytest.mark.parametrize(
    "data, expected_columns, exception_expected",
    [
        # Test case: Basic MACD calculation
        ({
            'Close': [10, 20, 30, 40, 50]
        }, ['MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9'], ValueError),
        # Test case: Constant prices
        ({
            'Close': [10] * 50
        }, ['MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9'], None),
        # Test case: Increasing prices
        ({
            'Close': [i for i in range(1, 35)]
        }, ['MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9'], None),
        # Test case: Decreasing prices
        ({
            'Close': [i for i in range(40, 0, -1)]
        }, ['MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9'], None),
    ])
def test_add_macd(data, expected_columns, exception_expected):
    # Create a sample DataFrame
    df = pd.DataFrame(data)

    if exception_expected:
        with pytest.raises(exception_expected):
            add_macd(df)
    else:
        # Call the function
        result_df = add_macd(df)

        # Assert the MACD columns are correctly added
        for column in expected_columns:
            assert column in result_df.columns, f"Missing column: {column}"
            # Ensure the column contains numeric values (no NaNs for valid rows)
            assert result_df[column].notna().any(
            ), f"Column {column} contains only NaNs"


# Test KDJ calculation
@pytest.mark.parametrize(
    "data, expected_columns, exception_expected",
    [
        # Test case: Basic KDJ calculation
        (
            {
                "High": [10, 20, 30, 40, 50],
                "Low": [5, 15, 25, 35, 45],
                "Close": [7, 18, 28, 38, 48],
            },
            ["STOCHk_14_3_3", "STOCHd_14_3_3", "J"],
            ValueError,
        ),
        # Test case: Constant prices
        (
            {
                "High": [10] * 20,
                "Low": [10] * 20,
                "Close": [10] * 20,
            },
            ["STOCHk_14_3_3", "STOCHd_14_3_3", "J"],
            None,
        ),
        # Test case: Increasing prices
        (
            {
                "High": [i for i in range(1, 21)],
                "Low": [i for i in range(1, 21)],
                "Close": [i for i in range(1, 21)],
            },
            ["STOCHk_14_3_3", "STOCHd_14_3_3", "J"],
            None,
        ),
        # Test case: Missing required columns
        (
            {
                "Open": [10, 20, 30, 40],
                "High": [15, 25, 35, 45],
            },
            None,
            KeyError,
        ),
    ],
)
def test_add_kdj(data, expected_columns, exception_expected):
    df = pd.DataFrame(data)

    if exception_expected:
        with pytest.raises(exception_expected):
            add_kdj(df)
    else:
        result_df = add_kdj(df)
        for column in expected_columns:
            assert column in result_df.columns, f"{column} column not added to DataFrame"
            # Ensure the column contains numeric values (no NaNs for valid rows)
            assert result_df[column].notna().any(
            ), f"Column {column} contains only NaNs"


# Test the RSI calculation
@pytest.mark.parametrize(
    "data, expected_column, expected_condition, exception_expected",
    [
        # Test case: Basic RSI calculation
        (
            {
                "Close":
                [10, 11, 12, 11, 10, 9, 8, 9, 10, 11, 12, 13, 14, 15, 16]
            },
            "RSI_14",
            lambda df: df["RSI_14"].between(0, 100).all(),
            None,
        ),
        # Test case: Constant prices
        (
            {
                "Close": [10] * 20
            },
            "RSI_14",
            lambda df: df["RSI_14"].isna().all() or
            (df["RSI_14"] == 0).all() or (df["RSI_14"] == 50).all(),
            None,
        ),
        # Test case: Increasing prices
        (
            {
                "Close": [i for i in range(1, 21)]
            },
            "RSI_14",
            lambda df: df["RSI_14"].iloc[-1] > 70,
            None,
        ),
        # Test case: Decreasing prices
        (
            {
                "Close": [i for i in range(20, 0, -1)]
            },
            "RSI_14",
            lambda df: df["RSI_14"].iloc[-1] < 30,
            None,
        ),
        # Test case: Missing Close column
        (
            {
                "Open": [10, 11, 12, 13],
                "High": [11, 12, 13, 14],
                "Low": [9, 10, 11, 12]
            },
            None,
            None,
            KeyError,
        ),
    ],
)
def test_add_rsi(data, expected_column, expected_condition,
                 exception_expected):
    df = pd.DataFrame(data)

    if exception_expected:
        with pytest.raises(exception_expected):
            add_rsi(df)
    else:
        result_df = add_rsi(df)
        assert expected_column in result_df.columns, f"{expected_column} column not added to DataFrame"
        assert expected_condition(
            result_df), f"Condition failed for data: {data}"


# Test OBV calculation
@pytest.mark.parametrize(
    "data, expected_obv",
    [
        # Test case: Basic OBV calculation
        (
            {
                "Close": [10, 12, 11, 13, 14],
                "Volume": [100, 200, 150, 300, 250],
            },
            [0, 200, 50, 350, 600],
        ),
        # Test case: Constant prices
        (
            {
                "Close": [10, 10, 10, 10],
                "Volume": [100, 200, 300, 400],
            },
            [0, 0, 0, 0],
        ),
        # Test case: Increasing prices
        (
            {
                "Close": [10, 11, 12, 13],
                "Volume": [100, 200, 300, 400],
            },
            [0, 200, 500, 900],
        ),
        # Test case: Decreasing prices
        (
            {
                "Close": [13, 12, 11, 10],
                "Volume": [100, 200, 300, 400],
            },
            [0, -200, -500, -900],
        ),
        # Test case: Mixed price changes
        (
            {
                "Close": [10, 12, 11, 13, 12],
                "Volume": [100, 200, 150, 300, 250],
            },
            [0, 200, 50, 350, 100],
        ),
    ],
)
def test_add_obv(data, expected_obv):
    # Create a sample DataFrame
    df = pd.DataFrame(data)

    # Call the function
    result_df = add_obv(df)

    # Assert the 'OBV' column is correctly added
    assert "OBV" in result_df.columns, "'OBV' column not added to DataFrame"
    assert result_df["OBV"].tolist(
    ) == expected_obv, f"Expected {expected_obv}, but got {result_df['OBV'].tolist()}"


# Test VWAP calculation
@pytest.mark.parametrize(
    "data, expected_vwap",
    [
        # Test case: Basic VWAP calculation
        (
            {
                "Close": [10, 12, 11, 13, 14],
                "Volume": [100, 200, 150, 300, 250],
            },
            [
                10.0, 11.333333333333334, 11.222222222222221,
                11.933333333333334, 12.45
            ],
        ),
        # Test case: Constant prices
        (
            {
                "Close": [10] * 20,
                "Volume": [100] * 20,
            },
            [10.0] * 20,
        ),
    ],
)
def test_add_vwap(data, expected_vwap):
    # Create a sample DataFrame
    df = pd.DataFrame(data)

    # Call the function
    result_df = add_vwap(df)

    # Assert the 'VWAP' column is correctly added
    assert "VWAP" in result_df.columns, "'VWAP' column not added to DataFrame"
    assert np.allclose(
        result_df["VWAP"].tolist(), expected_vwap, equal_nan=True
    ), f"Expected {expected_vwap}, but got {result_df['VWAP'].tolist()}"


# Test Bollinger Bands calculation
@pytest.mark.parametrize(
    "data, expected_columns, exception_expected",
    [
        # Test case: Basic Bollinger Bands calculation
        (
            {
                "Close": [10, 12, 11, 13, 14],
                "Volume": [100, 200, 150, 300, 250],
            },
            ["BB_Upper", "BB_Lower", "BB_Mid"],
            ValueError,
        ),
        # Test case: Constant prices
        (
            {
                "Close": [10] * 20,
                "Volume": [100] * 20,
            },
            ["BB_Upper", "BB_Lower", "BB_Mid"],
            None,
        ),
        # Test case: Increasing prices
        (
            {
                "Close": [i for i in range(1, 21)],
                "Volume": [i for i in range(1, 21)],
            },
            ["BB_Upper", "BB_Lower", "BB_Mid"],
            None,
        ),
        # Test case: Decreasing prices
        (
            {
                "Close": [i for i in range(20, 0, -1)],
                "Volume": [i for i in range(1, 21)],
            },
            ["BB_Upper", "BB_Lower", "BB_Mid"],
            None,
        ),
    ],
)
def test_add_bollinger_bands(data, expected_columns, exception_expected):
    df = pd.DataFrame(data)

    if exception_expected:
        with pytest.raises(exception_expected):
            add_bollinger_bands(df)
    else:
        result_df = add_bollinger_bands(df)
        for column in expected_columns:
            assert column in result_df.columns, f"{column} column not added to DataFrame"
            # Ensure the column contains numeric values (no NaNs for valid rows)
            assert result_df[column].notna().any(
            ), f"Column {column} contains only NaNs"


# Test ATR calculation
@pytest.mark.parametrize(
    "data, expected_columns, exception_expected",
    [
        # Test case: Basic ATR calculation
        (
            {
                "High": [10, 12, 11, 13, 14],
                "Low": [5, 7, 6, 8, 9],
                "Close": [8, 10, 9, 11, 12],
            },
            ["ATR"],
            ValueError,
        ),
        # Test case: Constant prices
        (
            {
                "High": [10] * 20,
                "Low": [5] * 20,
                "Close": [8] * 20,
            },
            ["ATR"],
            None,
        ),
        # Test case: Increasing prices
        (
            {
                "High": [i for i in range(1, 21)],
                "Low": [i for i in range(1, 21)],
                "Close": [i for i in range(1, 21)],
            },
            ["ATR"],
            None,
        ),
        # Test case: Decreasing prices
        (
            {
                "High": [i for i in range(20, 0, -1)],
                "Low": [i for i in range(20, 0, -1)],
                "Close": [i for i in range(20, 0, -1)],
            },
            ["ATR"],
            None,
        ),
    ],
)
def test_add_atr(data, expected_columns, exception_expected):
    df = pd.DataFrame(data)

    if exception_expected:
        with pytest.raises(exception_expected):
            add_atr(df)
    else:
        result_df = add_atr(df)
        for column in expected_columns:
            assert column in result_df.columns, f"{column} column not added to DataFrame"
            # Ensure the column contains numeric values (no NaNs for valid rows)
            assert result_df[column].notna().any(
            ), f"Column {column} contains only NaNs"
