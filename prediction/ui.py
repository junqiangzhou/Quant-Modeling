import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import pandas as pd
import glob
from data.label import label_feature


# Load CSV data
def load_data():
    csv_files = glob.glob("prediction/*.csv")
    csv_files.sort()
    if not csv_files:  # Handle case where no CSV files exist
        return pd.DataFrame(), "No Data"

    last_file = csv_files[-1]

    # Extract date from filename (assuming format like "prediction_YYYY-MM-DD.csv")
    parts = last_file.split("_")
    date = parts[-1].replace(".csv", "")  # Extract the date part

    return pd.read_csv(last_file), date


# Load initial data
df, date = load_data()

# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1(f"Buy/Sell Dashboard - {date}"),
    html.H3("Top 100 Buys"),
    dash_table.DataTable(id="table_buy", page_size=10),  # First 100 rows
    html.H3("Top 100 Sells"),
    dash_table.DataTable(id="table_sell", page_size=10),  # Last 100 rows
    dcc.Interval(id="interval-component", interval=1e6,
                 n_intervals=0)  # Auto-refresh every 1000s
])


@app.callback([Output("table_buy", "data"),
               Output("table_sell", "data")],
              [Input("interval-component", "n_intervals")])
def update_tables(n):
    # df, date = load_data()

    if df.empty:
        return [], []  # Return empty data if no CSV file is found

    df_buy = df.sort_values(by="buy_prob", ascending=False).head(100)
    df_buy.drop(columns=[label + "-" for label in label_feature], inplace=True)
    df_sell = df.sort_values(by="sell_prob", ascending=False).head(100)
    df_sell.drop(columns=[label + "+" for label in label_feature],
                 inplace=True)

    return df_buy.to_dict("records"), df_sell.to_dict("records")


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
