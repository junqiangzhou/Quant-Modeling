"""
数据处理工具模块

这个模块包含了数据获取和预处理相关的函数。
"""

import os
import pandas as pd
import backtrader as bt
import tushare as ts
from datetime import datetime
import yfinance as yf
import numpy as np


def get_ts_data(ts_token, ts_code, start_date, end_date, freq='30min'):
    """
    从Tushare获取股票数据，如果本地已有则直接加载
    
    参数:
    ts_token (str): Tushare API Token
    ts_code (str): 股票代码（如：'000001.SZ'）
    start_date (str): 开始日期（如：'2020-01-01'）
    end_date (str): 结束日期（如：'2021-01-01'）
    freq (str): 数据频率，默认为30分钟
    
    返回:
    pandas.DataFrame: 包含OHLCV数据的DataFrame
    """
    # 文件路径
    file_path = f'./data/{ts_code}-{start_date}-{end_date}-{freq}.csv'

    # 检查本地是否已存在该文件
    if os.path.exists(file_path):
        print(f"从本地文件加载数据: {file_path}")
        df = pd.read_csv(file_path, parse_dates=['trade_time'])  # 读取并解析时间列
        return df

    # 设置Tushare token
    ts.set_token(ts_token)
    pro = ts.pro_api()

    # 获取数据
    df = ts.pro_bar(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date,
        freq=freq,
        asset='E',  # 股票类型
        adj='qfq',  # 前复权
    )

    if df is None or df.empty:
        print("从 Tushare 获取的数据为空，请检查权限或参数设置。")
        return None

    # 创建目录（如果不存在）
    os.makedirs('./data', exist_ok=True)

    # 保存数据到本地文件
    df.to_csv(file_path, index=False)
    print(f"数据已保存至: {file_path}")

    return df


def create_bt_data_feed(df, start_date=None, end_date=None):
    """
    将pandas DataFrame转换为backtrader的DataFeed
    
    参数:
    df (pandas DataFrame): 股票数据，包含OHLCV列
    timeframe (backtrader.TimeFrame): 时间框架
    start_date (str or datetime, optional): 回测开始日期
    end_date (str or datetime, optional): 回测结束日期
    
    返回:
    backtrader.feed.DataFeed: 回测数据
    """
    # 确保索引是日期时间
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # 如果有日期过滤
    if start_date is not None:
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        df = df[df.index >= start_date]
    
    if end_date is not None:
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        df = df[df.index <= end_date]
    
    # 将DataFrame转换为backtrader可用的DataFeed
    data = bt.feeds.PandasData(
        dataname=df,
        datetime=None,  # 使用索引作为日期
        open='Open',
        high='High',
        low='Low',
        close='Close',
        volume='Volume',
        openinterest=-1  # 不使用openinterest
    )
    
    return data 


def load_data_from_csv(file_path):
    """
    从CSV文件加载数据
    
    参数:
    file_path (str): CSV文件路径
    
    返回:
    pandas.DataFrame: 包含OHLCV数据的DataFrame
    """
    # 读取CSV文件
    df = pd.read_csv(file_path)

    # 将日期列转换为datetime类型
    if 'trade_time' in df.columns:
        df['trade_time'] = pd.to_datetime(df['trade_time'])
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    return df


def load_data_from_yahoo(tickers,
                         start_date,
                         end_date=None,
                         interval='1d',
                         save_to_csv=True,
                         data_dir='../data'):
    """
    从Yahoo Finance下载股票数据
    
    参数:
    tickers (list or str): 股票代码或代码列表
    start_date (str): 开始日期，格式为 'YYYY-MM-DD'
    end_date (str): 结束日期，格式为 'YYYY-MM-DD'，默认为当前日期
    interval (str): 数据间隔，可选值: '1d', '1wk', '1mo'等
    save_to_csv (bool): 是否保存数据到CSV文件
    data_dir (str): 保存数据的目录
    
    返回:
    pandas DataFrame: 历史价格数据
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    if isinstance(tickers, str):
        tickers = [tickers]

    all_data = {}
    for ticker in tickers:
        try:
            print(f"获取 {ticker} 的数据...")
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date,
                                 end=end_date,
                                 interval=interval)
            data = data.sort_index()

            if len(data) == 0:
                print(f"警告: 没有找到 {ticker} 的数据")
                continue

            all_data[ticker] = data

            if save_to_csv:
                if not os.path.exists(data_dir):
                    os.makedirs(data_dir)
                filename = os.path.join(
                    data_dir,
                    f"{ticker}_{start_date}_{end_date}_{interval}.csv")
                data.to_csv(filename)
                print(f"数据已保存到 {filename}")
        except Exception as e:
            print(f"获取 {ticker} 数据时出错: {e}")

    if len(tickers) == 1:
        return all_data.get(tickers[0], pd.DataFrame())
    return all_data


def calculate_indicators(df):
    """
    计算常用技术指标
    
    参数:
    df (pandas DataFrame): 包含OHLCV数据的DataFrame
    
    返回:
    pandas DataFrame: 包含原始数据和计算的指标
    """
    # 确保列名符合要求
    if not all(col in df.columns
               for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
        # 尝试将第一个字母大写
        df = df.rename(columns={col: col.capitalize() for col in df.columns})

    # 复制数据，避免修改原始数据
    result = df.copy()

    # 移动平均线
    result['SMA5'] = result['Close'].rolling(window=5).mean()
    result['SMA10'] = result['Close'].rolling(window=10).mean()
    result['SMA20'] = result['Close'].rolling(window=20).mean()
    result['SMA50'] = result['Close'].rolling(window=50).mean()
    result['SMA200'] = result['Close'].rolling(window=200).mean()

    # 指数移动平均线
    result['EMA12'] = result['Close'].ewm(span=12, adjust=False).mean()
    result['EMA26'] = result['Close'].ewm(span=26, adjust=False).mean()

    # MACD
    result['MACD'] = result['EMA12'] - result['EMA26']
    result['MACD_signal'] = result['MACD'].ewm(span=9, adjust=False).mean()
    result['MACD_hist'] = result['MACD'] - result['MACD_signal']

    # RSI (14天)
    delta = result['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    result['RSI14'] = 100 - (100 / (1 + rs))

    # 布林带 (20,2)
    result['BB_middle'] = result['Close'].rolling(window=20).mean()
    result['BB_upper'] = result['BB_middle'] + (
        result['Close'].rolling(window=20).std() * 2)
    result['BB_lower'] = result['BB_middle'] - (
        result['Close'].rolling(window=20).std() * 2)

    # 成交量变化
    result['Volume_Change'] = result['Volume'].pct_change()

    # 波动率 (20天)
    result['Volatility'] = result['Close'].pct_change().rolling(
        window=20).std() * np.sqrt(20)

    return result
