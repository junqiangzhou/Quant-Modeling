import os
import pandas as pd
import pytest
from data.data_fetcher import fetch_stocks, create_dataset

@pytest.fixture
def mock_fetch_stocks():
    # Mock fetch_stocks to return a small list of stock symbols
    return ["AAPL", "MSFT"]

def test_main_integration(mock_fetch_stocks):
    # Use only mock_fetch_stocks as the dependency
    start_date = "2023-01-01"
    end_date = "2023-03-31"

    output_df = None
    for i, stock in enumerate(mock_fetch_stocks):
        # Simulate create_dataset behavior
        df = create_dataset(stock, start_date, end_date)

        if output_df is None:
            output_df = df
        else:
            output_df = pd.concat([output_df, df], ignore_index=False)

    # Verify the content of the output file
    output_df.to_csv(f"./data/test/test_{start_date}_{end_date}.csv",
                  index=True,
                  index_label="Date")
    assert not output_df.empty
    assert len(output_df["stock"].unique()) == 2  # Two stocks: AAPL and MSFT
    assert len(output_df) == 124  # Number of days between start_date and end_date
    assert len(output_df.columns) == 68

