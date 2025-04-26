import torch
from enum import Enum
from itertools import chain

# The name of the model to be exported and loaded
# The model is saved in the model directory
MODEL_EXPORT_NAME = "model_cpu_v1"

# device config
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


# Define the model type for encoder
class EncoderType(Enum):
    MLP = 0
    Transformer = 1
    LatentQueryTransformer = 2
    DualAttentionTransformer = 3


ENCODER_TYPE = EncoderType.Transformer


# Model type switches between torch (MLP) and xgboost
class ModelType(Enum):
    TORCH = 0
    XGBOOST = 1


MODEL_TYPE = ModelType.TORCH


class Action(Enum):
    Hold = 0
    Buy = 1
    Sell = 2


# random seed across code base for reproducibility
random_seed = 42

# history time windows for feature sequence length
look_back_window = 50

# future time window where the labels are calculated at training and predicted at inference time
future_time_windows = [5, 10, 20, 30]  # number of next rows to consider

# Default moving average windows
MA_WINDOWS = [5, 10, 20, 50]

# List of basic data downloaded from Yahoo Finance
base_feature = [
    'Open',
    'High',
    'Low',
    'Close',
    'Volume',
    'MA_5',
    'MA_10',
    'MA_20',
    'MA_50',  # 'Trading_Volume'
]

# List of technical indicators
macd_feature = ['MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9']
kdj_feature = ["STOCHk_14_3_3", "STOCHd_14_3_3", "J"]
rsi_feature = ["RSI_14"]
bullish_bearish_signals = ["Price_Above_MA_5", "Price_Below_MA_5"]
other_features = ["daily_change"]

# List of signals to determine buy/sell points
buy_sell_signals = [
    "MA_5_20_Crossover_Signal",  # "MA_5_10_Crossover_Signal", "MA_5_50_Crossover_Signal", 
    "MA_10_50_Crossover_Signal",  # "MA_10_20_Crossover_Signal", "MA_20_50_Crossover_Signal",
    "MACD_Crossover_Signal",
    "RSI_Over_Bought_Signal",
    "BB_Signal",  # "VWAP_Crossover_Signal"
]

# Other crossover signals that are not in buy_sell_signals but can be used as features
other_crossover_signals = [
    "MA_5_10_Crossover_Signal", "MA_5_50_Crossover_Signal",
    "MA_10_20_Crossover_Signal", "MA_20_50_Crossover_Signal",
    "VWAP_Crossover_Signal"
]

buy_sell_signals_encoded = [
    f"{signal}_{suffix}" for signal in buy_sell_signals
    for suffix in ["0", "-1", "1"]
]

feature_names = [
    name + "_diff" for name in base_feature
] + buy_sell_signals_encoded + bullish_bearish_signals + other_crossover_signals + other_features

# classification labels for model to predict
label_names = list(
    chain(*[[f"trend_{time}days"] for time in future_time_windows]))

# all columns added for debugging labeling
# [max_close, max_duration, min_close, min_duration, trend_Xdays]
label_columns = list(
    chain(*[[
        f"{time}days_max_close", f"{time}days_max_duration",
        f"{time}days_min_close", f"{time}days_min_duration",
        f"trend_{time}days"
    ] for time in future_time_windows]))
