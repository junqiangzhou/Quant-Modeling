import pandas as pd
import pytest
from data.data_fetcher import create_dataset, add_earnings_data, preprocess_data, download_data


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
    assert len(
        output_df) == 124  # Number of days between start_date and end_date
    assert len(output_df.columns) == 74


@pytest.fixture
def sample_dataframe():
    # Create a sample dataframe to simulate stock data
    data = {
        "Open": [100, 102, 104],
        "High": [105, 107, 109],
        "Low": [95, 97, 99],
        "Close": [102, 104, 106],
        "Volume": [1000, 1500, 2000]
    }
    index = pd.date_range(start="2023-01-01", periods=3, freq="D")
    return pd.DataFrame(data, index=index)


@pytest.fixture
def mock_ticker_with_earnings(mocker):
    # Mock a Ticker object with earnings data
    mock_ticker = mocker.Mock()
    earnings_data = pd.DataFrame(
        {
            "EPS Estimate": [1.5, 1.6],
            "Reported EPS": [1.7, 1.8],
            "Surprise(%)": [13.33, 12.5]
        },
        index=pd.to_datetime(["2023-01-02", "2023-01-03"]))
    mock_ticker.earnings_dates = earnings_data
    return mock_ticker


@pytest.fixture
def mock_ticker_without_earnings(mocker):
    # Mock a Ticker object without earnings data
    mock_ticker = mocker.Mock()
    mock_ticker.earnings_dates = None
    return mock_ticker


def test_add_earnings_data_with_earnings(sample_dataframe,
                                         mock_ticker_with_earnings):
    start_date = "2023-01-01"
    end_date = "2023-01-03"

    result_df = add_earnings_data(sample_dataframe, mock_ticker_with_earnings,
                                  start_date, end_date)

    assert "EPS_Estimate" in result_df.columns
    assert "EPS_Reported" in result_df.columns
    assert "Surprise(%)" in result_df.columns
    assert "Earnings_Date" in result_df.columns

    assert result_df.loc["2023-01-02", "EPS_Reported"] == 1.7
    assert result_df.loc["2023-01-03", "EPS_Reported"] == 1.8
    assert result_df.loc["2023-01-02", "Earnings_Date"]
    assert not result_df.loc["2023-01-01", "Earnings_Date"]


def test_add_earnings_data_without_earnings(sample_dataframe,
                                            mock_ticker_without_earnings):
    start_date = "2023-01-01"
    end_date = "2023-01-03"

    result_df = add_earnings_data(sample_dataframe,
                                  mock_ticker_without_earnings, start_date,
                                  end_date)

    assert "EPS_Estimate" in result_df.columns
    assert "EPS_Reported" in result_df.columns
    assert "Surprise(%)" in result_df.columns
    assert "Earnings_Date" in result_df.columns

    assert result_df["EPS_Reported"].isna().all()
    assert result_df["Earnings_Date"].isna().sum() == 0
    assert not result_df["Earnings_Date"].all()


def test_preprocess_data_with_valid_data(mock_fetch_stocks):
    # Use only mock_fetch_stocks as the dependency
    start_date = "2023-01-01"
    end_date = "2023-03-31"

    df = download_data(mock_fetch_stocks[0], start_date, end_date)

    # Assert that the result has the expected columns
    assert len(df) == 131  # Number of days between start_date and end_date
    assert len(df.columns) == 12

    # Assert that the index is not converted and remains in its raw format
    assert isinstance(df.index, pd.DatetimeIndex)


def test_preprocess_data_with_missing_columns(sample_dataframe):
    stock_symbol = "AAPL"
    start_date = "2023-01-01"

    # Call preprocess_data without necessary columns
    with pytest.raises(ValueError):
        preprocess_data(sample_dataframe, stock_symbol, start_date)


def test_preprocess_data_with_empty_dataframe():
    stock_symbol = "AAPL"
    start_date = "2023-01-01"
    empty_df = pd.DataFrame()

    # Call preprocess_data with an empty dataframe
    with pytest.raises(ValueError):
        preprocess_data(empty_df, stock_symbol, start_date)
