import pandas as pd
from yahoo_fin import stock_info as si

# Get NASDAQ-100 symbols
url = 'https://en.wikipedia.org/wiki/NASDAQ-100'
# Read the tables on the webpage
tables = pd.read_html(url)
# The first table on the page contains the NASDAQ-100 components
nasdaq_100_symbols = tables[4]["Symbol"].tolist()

print(f"NASDAQ-100 symbols ({len(nasdaq_100_symbols)}):", nasdaq_100_symbols)

# Get S&P 500 symbols
sp500_symbols = si.tickers_sp500()
# sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
# tables = pd.read_html(sp500_url)
# sp500_symbols = tables[0]['Symbol'].tolist()
print(f"S&P 500 symbols ({len(sp500_symbols)}):", sp500_symbols[:5])

# Combine and remove duplicates
all_symbols = list(set(sp500_symbols + nasdaq_100_symbols))
print(f"Total unique symbols: {len(all_symbols)}")