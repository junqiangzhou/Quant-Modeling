from rl.multi_shot.trading_env import StockTradingEnv
from feature.feature import compute_online_feature
from config.config import Action, label_names, buy_sell_signals
from data.stocks_fetcher import MAG7
from strategy.rule_based import should_buy, should_sell

from numpy.typing import NDArray
from typing import List, Tuple
import numpy as np
import torch


class BacktestSingleShot(StockTradingEnv):

    def __init__(self,
                 stocks: List[str],
                 start_date: str,
                 end_date: str,
                 init_fund: float = 1.0e4):
        super().__init__(stocks, start_date, end_date, init_fund)
        self.debug_mode = False
        self.earning_day_sell = False
        self.active_sell = False

    def compute_action(self) -> Tuple[int, Action]:
        date = self.current_step

        # Check if needed to sell current holdings
        if self.stock_holdings > 0:
            stock = self.stocks[self.stock_held]
            if self.earning_day_sell and date in self.stock_data[stock].index and self.stock_data[
                        stock].loc[date]["Earnings_Date"]:
                # Must sell all shares before earnings day
                if self.debug_mode:
                    print(f"Earnings day must sell, {date}")
                return (self.stock_held, Action.Sell)

            # Must sell to cut-loss
            if date in self.stock_data[stock].index:
                cost_base = self.cost_base
                price = self.stock_data[stock].loc[date]["Close"]
                if price < cost_base * 0.92:
                    if self.debug_mode:
                        print(f"Must cut loss and sell, {date}")
                    return (self.stock_held, Action.Sell)

        stock_actions = self.compute_stocks_action(date)
        if self.stock_holdings > 0:
            # Need to sell based on current predictions
            if stock_actions[self.stock_held][0] == 2:
                if self.debug_mode:
                    price = self.stock_data[self.stocks[
                        self.stock_held]].loc[date]["Close"]
                    print(
                        f"------Predicted to sell, {date}, close price {price:.2f}"
                    )
                return (self.stock_held, Action.Sell)

            if self.active_sell:
                # Choose to sell since other stocks have better buy signals
                for i, stock in enumerate(self.stocks):
                    if stock_actions[i][0] == 1 and stock_actions[i][
                            1] > 0.6 and stock_actions[i][1] > stock_actions[
                                self.stock_held][1]:
                        if self.debug_mode:
                            price = self.stock_data[stock].loc[date]["Close"]
                            print(
                                f"------Choose to sell with buy better stocks {date}, stock {stock}, close price {price:.2f}, prob. of trending up {stock_actions[i][1]}"
                            )
                        return (self.stock_held, Action.Sell)

        else:  # Currently holds no stocks, and check if needs to buy
            # Find the stock with highest probability of trending up
            max_probs, max_stock = 0.0, None
            for i, stock in enumerate(self.stocks):
                if stock_actions[i][0] == 1 and stock_actions[i][1] > max_probs:
                    max_probs = stock_actions[i][1]
                    max_stock = i

            if max_stock is not None:
                if self.debug_mode:
                    stock = self.stocks[max_stock]
                    price = self.stock_data[
                        self.stocks[max_stock]].loc[date]["Close"]
                    print(
                        f"++++++Predicted to buy, {date}, stock {stock}, close price {price:.2f}, prob. of trending up {max_probs}"
                    )
                return (max_stock, Action.Buy)

        return (self.stock_held, Action.Hold)

    def run(self) -> None:
        done = False
        print("current_date: ", self.start_date, " end_date: ", self.end_date)
        while not done:
            if self.debug_mode:
                self.render()
            stock_index, order = self.compute_action()
            action = (stock_index, int(order.value))
            obs, reward, done, _, _ = self.step(action)

        print(f"Quant profit: {reward * 100: .2f} %")

        return

    def compute_stocks_action(self, date):
        stock_action_data = []
        for stock in self.stocks:
            features = compute_online_feature(self.stock_data[stock], date)
            if features is None or np.isnan(features).any() or np.isinf(
                    features).any():
                stock_action_data.append(np.array([0, 0]))
            else:
                features_tensor = torch.tensor(features, dtype=torch.float32)

            with torch.no_grad():
                logits = self.prediction_model(features_tensor)
                logits = logits.reshape(len(label_names), 3)
                probs = torch.softmax(
                    logits,
                    dim=1).float().numpy()  # convert logits to probabilities
                pred = np.argmax(probs, axis=1)

            buy_sell_signals_vals = self.stock_data[stock].loc[
                date, buy_sell_signals].values
            price_above_ma = self.stock_data[stock].loc[
                date, "Price_Above_MA_5"] == 1
            price_below_ma = self.stock_data[stock].loc[
                date, "Price_Below_MA_5"] == 1

            need_buy = should_buy(pred, buy_sell_signals_vals, price_above_ma)
            need_sell = should_sell(pred, buy_sell_signals_vals,
                                    price_below_ma)
            probs_avg = np.mean(probs, axis=0)

            if need_buy:
                stock_action_data.append(np.array([1, probs_avg[1]]))
            elif need_sell:
                stock_action_data.append(np.array([2, probs_avg[2]]))
            else:
                stock_action_data.append(np.array([0, probs_avg[0]]))
        stock_action_data = np.vstack(stock_action_data)
        return stock_action_data


if __name__ == "__main__":
    stocks = MAG7
    start_date = "2021-01-01"
    end_date = "2021-12-31"
    init_fund = 1.0e4

    backtest = BacktestSingleShot(stocks, start_date, end_date, init_fund)
    backtest.run()
