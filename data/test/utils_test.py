from data.utils import get_date_back, get_stock_df, perc_change, days_diff
import pandas as pd
from datetime import datetime

import pytest


@pytest.mark.parametrize(
    "date_str, delta_days, expected",
    [
        ("2023-01-10", 5, "2023-01-05"),
        ("2023-01-10", 0, "2023-01-10"),
        ("2023-01-10", 365, "2022-01-10"),
        ("2020-03-01", 1, "2020-02-29"),  # Leap year
        ("2023-01-01", 1, "2022-12-31"),  # Cross year
    ])
def test_get_date_back(date_str, delta_days, expected):
    assert get_date_back(date_str, delta_days) == expected


def test_get_stock_df():
    # Create a sample DataFrame
    data = {
        'stock': ['AAPL', 'GOOGL', 'AAPL', 'MSFT'],
        'price': [150, 2800, 155, 300]
    }
    df = pd.DataFrame(data)

    # Test for a stock that exists in the DataFrame
    result = get_stock_df(df, 'AAPL')
    expected = df[df['stock'] == 'AAPL']
    pd.testing.assert_frame_equal(result, expected)

    # Test for a stock that does not exist in the DataFrame
    with pytest.raises(ValueError):
        get_stock_df(df, 'TSLA')


def test_perc_change():
    # Test positive percentage change
    assert perc_change(100, 150) == pytest.approx(0.5)
    # Test negative percentage change
    assert perc_change(100, 50) == pytest.approx(-0.5)
    # Test no change
    assert perc_change(100, 100) == pytest.approx(0.0)
    # Test small denominator handling
    assert perc_change(0.001, 0.002) == pytest.approx(1.0)
    # Test zero denominator handling
    assert perc_change(0, 100) == pytest.approx(100000.0)


def test_days_diff():
    # Test positive difference
    date1 = datetime(2025, 4, 21)
    date2 = datetime(2025, 4, 25)
    assert days_diff(date1, date2) == 4
    # Test negative difference (absolute value)
    assert days_diff(date2, date1) == 4
    # Test same date
    assert days_diff(date1, date1) == 0
