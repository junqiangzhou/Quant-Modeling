from data.utils import get_date_back

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
