from config.config import label_feature
from data.stocks_fetcher import MAG7, ETF, BOND, PICKS, CHINA

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import pandas as pd
import glob

buy_names = [name + "+" for name in label_feature]
sell_names = [name + "-" for name in label_feature]
hold_names = [name + "0" for name in label_feature]

style_cell = {
    "minWidth": "20px",
    "maxWidth": "20px",
    "width": "20px",  # Adjust column widths
    "textOverflow": "ellipsis",
    "overflow": "hidden",  # Prevent text overflow
    "whiteSpace": "nowrap"  # Keep text in one line
}
style_table = {
    "width": "80%",
    "margin": "auto",
    "overflowX": "auto"
}  # Adjust table width


# Load CSV data
def load_data():
    csv_files = glob.glob("predict/data/*.csv")
    csv_files.sort()
    if not csv_files:  # Handle case where no CSV files exist
        return pd.DataFrame(), "No Data"

    last_file = csv_files[-1]

    # Extract date from filename (assuming format like "prediction_YYYY-MM-DD.csv")
    parts = last_file.split("_")
    date = parts[-1].replace(".csv", "")  # Extract the date part

    df = pd.read_csv(last_file)
    df.set_index("stock", inplace=True)

    return df, date


# Load initial data
df, date = load_data()

# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1(f"Stocks Recommendation Dashboard - {date}"),
    html.H3("Top Buys"),
    dash_table.DataTable(id="table_buy",
                         page_size=10,
                         style_cell=style_cell,
                         style_table=style_table),
    html.H3("Top Sells"),
    dash_table.DataTable(id="table_sell",
                         page_size=10,
                         style_cell=style_cell,
                         style_table=style_table),
    html.H3("Top Picks"),
    dash_table.DataTable(id="table_focus",
                         page_size=10,
                         style_cell=style_cell,
                         style_table=style_table),
    dcc.Interval(id="interval-component", interval=1e6,
                 n_intervals=0)  # Auto-refresh every 1000s
])


@app.callback([
    Output("table_buy", "data"),
    Output("table_sell", "data"),
    Output("table_focus", "data")
], [Input("interval-component", "n_intervals")])
def update_tables(n):
    # df, date = load_data()

    if df.empty:
        return [], [], []  # Return empty data if no CSV file is found

    df_buy = df.sort_values(by="BUY", ascending=False).head(100).reset_index()
    df_buy.drop(columns=sell_names + hold_names, inplace=True)

    df_sell = df.sort_values(by="SELL",
                             ascending=False).head(100).reset_index()
    df_sell.drop(columns=buy_names + hold_names, inplace=True)

    stock_picks = ETF + BOND + MAG7 + PICKS + CHINA
    df_focus = df.loc[stock_picks, ["BUY", "SELL", "HOLD"]].reset_index()

    return df_buy.to_dict("records"), df_sell.to_dict(
        "records"), df_focus.to_dict("records")


# Run the app
if __name__ == "__main__":
    app.run_server(port=8080, debug=True)
