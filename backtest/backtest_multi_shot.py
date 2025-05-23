from rl.multi_shot.trading_env import StockTradingEnv
from feature.feature import compute_online_feature
from config.config import Action, ActionProbability, label_names, buy_sell_signals
from data.stocks_fetcher import MAG7
from strategy.rule_based import should_buy, should_sell
from model.model import compute_model_output

from typing import List, Tuple
from datetime import datetime
import numpy as np


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
        self.use_gt_label = False
        if self.use_gt_label:
            print("+++++++++++++Using GT labels for backtest++++++++++++++")

    def compute_action(self) -> Tuple[int, Action]:
        date = self.current_step

        # Check if needed to sell current holdings
        if self.stock_holdings > 0:
            stock = self.stocks[self.stock_held]
            if self.earning_day_sell and date in self.stock_data[
                    stock].index and self.stock_data[stock].loc[date][
                        "Earnings_Date"]:
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
            if stock_actions[self.stock_held].action == Action.Sell:
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
                    if stock_actions[i].action == Action.Buy and stock_actions[
                            i].prob > 0.6 and stock_actions[
                                i].prob > stock_actions[self.stock_held].prob:
                        if self.debug_mode:
                            price = self.stock_data[stock].loc[date]["Close"]
                            print(
                                f"------Choose to sell with buy better stocks {date}, stock {stock}, close price {price:.2f}, prob. of trending up {stock_actions[i].prob}"
                            )
                        return (self.stock_held, Action.Sell)

        else:  # Currently holds no stocks, and check if needs to buy
            # Find the stock with highest probability of trending up
            max_probs, max_stock = 0.0, None
            for i, stock in enumerate(self.stocks):
                if stock_actions[i].action == Action.Buy and stock_actions[
                        i].prob > max_probs:
                    max_probs = stock_actions[i].prob
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

    def compute_stocks_action(self, date: datetime) -> List[ActionProbability]:
        stock_action_data = []
        for stock in self.stocks:
            buy_sell_signals_vals = self.stock_data[stock].loc[
                date, buy_sell_signals].values
            price_above_ma = self.stock_data[stock].loc[
                date, "Price_Above_MA_5"] == 1
            price_below_ma = self.stock_data[stock].loc[
                date, "Price_Below_MA_5"] == 1

            if self.use_gt_label:
                pred = self.stock_data[stock].loc[date, label_names].values
                probs_avg = np.ones(3)
            else:
                features = compute_online_feature(self.stock_data[stock], date)
                probs, pred = compute_model_output(self.prediction_model, features)
                # Early return if error
                if probs is None or pred is None:
                    print(f"stock {stock} feature has invalid data")
                    action_prob = ActionProbability(action=Action.Hold, prob=0)
                    stock_action_data.append(action_prob)
                    continue
                probs_avg = np.mean(probs, axis=0)

            # Choose actions and report probability
            if should_buy(pred, buy_sell_signals_vals, price_above_ma):
                action_prob = ActionProbability(action=Action.Buy,
                                                prob=probs_avg[1])
                stock_action_data.append(action_prob)
            elif should_sell(pred, buy_sell_signals_vals, price_below_ma):
                action_prob = ActionProbability(action=Action.Sell,
                                                prob=probs_avg[2])
                stock_action_data.append(action_prob)
            else:
                action_prob = ActionProbability(action=Action.Hold,
                                                prob=probs_avg[0])
                stock_action_data.append(action_prob)
        return stock_action_data


if __name__ == "__main__":
    stocks = MAG7
    start_date = "2021-01-01"
    end_date = "2021-12-31"
    init_fund = 1.0e4

    backtest = BacktestSingleShot(stocks, start_date, end_date, init_fund)
    backtest.run()
