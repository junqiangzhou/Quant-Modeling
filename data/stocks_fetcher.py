import pandas as pd
from yahoo_fin import stock_info as si
import random

MAG7 = ["TSLA", "AAPL", "GOOGL", "AMZN", "MSFT", "META", "NVDA"]
ETF = ["SPY", "QQQ", "DIA", "IWM", "XLK"]
BOND = ["TLT", "IEF", "SHY", "LQD", "HYG"]
PICKS = ["NU", "HIMS", "RBLX", "UPST", "SE", "CRWD", "DDOG", "SNOW", "PLTR"]
CHINA = [
    "BABA", "TCEHY", "JD", "BIDU", "PDD", "BILI", "DOYU", "HUYA", "NTES",
    "ZIJMY", "TCOM", "LI", "NIO", "XPEV", "BEKE", "DIDIY", "LKNCY", "ZK",
    "WRD", "HSAI", "TUYA"
]
TEST_GROUP = ["NFLX", "BABA", "TSM", "JPM", "V"]


def fetch_stocks():
    # Get NASDAQ-100 symbols
    url = 'https://en.wikipedia.org/wiki/NASDAQ-100'
    # Read the tables on the webpage
    tables = pd.read_html(url)
    # The first table on the page contains the NASDAQ-100 components
    nasdaq_100_symbols = tables[4]["Ticker"].tolist()

    # print(f"NASDAQ-100 symbols ({len(nasdaq_100_symbols)}):", nasdaq_100_symbols)

    # Get S&P 500 symbols
    sp500_symbols = si.tickers_sp500()
    sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(sp500_url)
    sp500_symbols = tables[0]['Symbol'].tolist()
    # print(f"S&P 500 symbols ({len(sp500_symbols)}):", sp500_symbols[:5])

    # Combine and remove duplicates
    all_symbols = list(set(sp500_symbols + nasdaq_100_symbols))
    all_symbols = sorted(all_symbols)
    print(f"Total unique symbols: {len(all_symbols)}")
    random.seed(42)
    random.shuffle(all_symbols)

    train_size = int(0.8 * len(all_symbols))
    train_stocks, test_stocks = all_symbols[:train_size], all_symbols[
        train_size:]

    test_stocks += ETF + BOND + PICKS + CHINA

    # train_stocks, test_stocks = MAG7, TEST_GROUP
    print(f"# of stocks for training: {len(train_stocks)}")
    print(f"# of stocks for testing: {len(test_stocks)}")
    return train_stocks, test_stocks
