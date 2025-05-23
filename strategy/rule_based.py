import numpy as np
from numpy.typing import NDArray

# Select which label to use and determine the buy/sell signals
predict_label = 4


def should_buy(pred: NDArray, buy_sell_signals_vals: NDArray,
               price_above_ma: bool) -> bool:
    if predict_label == 4:
        ml_pred_up = np.all(pred == 1)  # all predictions trend up
    else:
        if predict_label not in [0, 1, 2, 3]:
            raise ValueError(f"Invalid predict_label: {predict_label}")
        ml_pred_up = pred[predict_label] == 1  # selected prediction trends up
    trend_up_indicators = np.sum(buy_sell_signals_vals) > 0
    if ml_pred_up and trend_up_indicators and price_above_ma:
        return True

    return False


def should_sell(pred: NDArray, buy_sell_signals_vals: NDArray,
                price_below_ma: bool) -> bool:
    if predict_label == 4:
        ml_pred_down = np.all(pred == 2)  # all predictions trend down
    else:
        if predict_label not in [0, 1, 2, 3]:
            raise ValueError(f"Invalid predict_label: {predict_label}")
        ml_pred_down = pred[
            predict_label] == 2  # selected prediction trends down
    trend_down_indicators = np.sum(buy_sell_signals_vals) < 0
    if ml_pred_down and trend_down_indicators and price_below_ma:
        return True

    return False


def calc_pred_labels(probs: NDArray) -> NDArray:
    """
    Convert probabilities to labels.
    """
    pred = np.argmax(probs, axis=1)
    return pred
