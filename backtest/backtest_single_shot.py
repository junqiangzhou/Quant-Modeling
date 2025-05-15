from rl.single_shot.trading_env import StockTradingEnv
from feature.feature import compute_online_feature
from config.config import Action, label_names, buy_sell_signals
from data.stocks_fetcher import MAG7

from numpy.typing import NDArray
import numpy as np
import torch


class BacktestSingleShot(StockTradingEnv):

    def __init__(self,
                 stock: str,
                 start_date: str,
                 end_date: str,
                 init_fund: float = 1.0e4):
        super().__init__(stock, start_date, end_date, init_fund)
        print(f">>>>>{stock}:")
        self.debug_mode = False

    def compute_action(self) -> Action:
        date = self.current_step
        # Must sell all shares before earnings day
        if date in self.stock_data.index and self.stock_data.loc[date][
                "Earnings_Date"]:
            if self.debug_mode:
                print(f"Earnings day must sell, {date}")
            return Action.Sell

        # Must sell to cut-loss
        if date in self.stock_data.index:
            cost_base = self.cost_base
            price = self.stock_data.loc[date]["Close"]
            if price < cost_base * 0.92:
                if self.debug_mode:
                    print(f"Must cut loss and sell, {date}")
                return Action.Sell

        features = compute_online_feature(self.stock_data, date)
        if features is None or np.isnan(features).any() or np.isinf(
                features).any():
            # print(f"NaN or INF detected in {stock} on {date}")
            return Action.Hold
        else:
            features_tensor = torch.tensor(features, dtype=torch.float32)

        buy_sell_signals_vals = self.stock_data.loc[date,
                                                    buy_sell_signals].values
        bullish_signal = self.stock_data.loc[date, "Price_Above_MA_5"]
        bearish_signal = self.stock_data.loc[date, "Price_Below_MA_5"]

        with torch.no_grad():
            logits = self.prediction_model(features_tensor)
            logits = logits.reshape(len(label_names), 3)
            probs = torch.softmax(
                logits,
                dim=1).float().numpy()  # convert logits to probabilities
            pred = np.argmax(probs, axis=1)

        def should_buy(pred: NDArray) -> bool:
            trend_up_labels = np.sum(pred == 1)
            trend_up_indicators = np.sum(buy_sell_signals_vals == 1)
            if trend_up_labels == len(
                    label_names
            ) and trend_up_indicators >= 1 and bullish_signal == 1:
                return True

            return False

        def should_sell(pred: NDArray) -> bool:
            trend_down_labels = np.sum(pred == 2)
            trend_down_indicators = np.sum(buy_sell_signals_vals == -1)
            if trend_down_labels == len(
                    label_names
            ) and trend_down_indicators >= 1 and bearish_signal == 1:
                return True

            return False

        if should_sell(pred):  # need to sell
            if self.debug_mode:
                print(
                    f"------Predicted to sell, {date}, close price {price:.2f}, prob. of trending down {probs[:, 2]}"
                )
            return Action.Sell
        elif should_buy(pred):  # good to buy
            if self.debug_mode:
                print(
                    f"++++++Predicted to buy, {date}, close price {price:.2f}, prob. of trending up {probs[:, 1]}"
                )
            return Action.Buy
        else:
            return Action.Hold

    def run(self) -> None:
        done = False
        # print("current_date: ", self.start_date, " end_date: ", self.end_date)
        try:
            start_price, end_price = self.stock_data.loc[self.start_date][
                "Close"], self.stock_data.loc[self.end_date]["Close"]
            print(
                f"Price change: {(end_price - start_price) / start_price * 100: .2f} %",
                end="      ")
        except:
            print(f"Failed to compute price change {stock}")
        while not done:
            if self.debug_mode:
                self.render()
            action = self.compute_action()
            obs, reward, done, _, _ = self.step(int(action.value))

        print(f"Quant profit: {reward * 100: .2f} %")

        return


if __name__ == "__main__":
    start_date = "2021-01-01"
    end_date = "2021-12-31"
    init_fund = 1.0e4

    for stock in MAG7:
        backtest = BacktestSingleShot(stock, start_date, end_date, init_fund)
        backtest.run()
