from datetime import datetime, timedelta
import pandas as pd


# Calculates a shifted date from the input
def get_date_back(date_str: str, delta_days: int) -> str:
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    date_back = date_obj - timedelta(days=delta_days)
    return date_back.strftime("%Y-%m-%d")


# Extracts rows for a given stock symbol
def get_stock_df(df_in: pd.DataFrame, stock: str) -> pd.DataFrame:
    if stock not in df_in['stock'].values:
        raise ValueError(f"Stock {stock} not found in the DataFrame.")
    return df_in[df_in['stock'] == stock]


# Calculates the percentage change between two values
def perc_change(curr: float, future: float) -> float:
    return (future - curr) / max(abs(curr), 1e-3)


# Calculates the absolute number of days between two dates
def days_diff(date1: datetime, date2: datetime) -> int:
    return abs((date2 - date1).days)
