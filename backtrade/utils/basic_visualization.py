"""
基础可视化模块

这个模块提供了绘制基础图表的函数，如K线图、成交量图、技术指标等。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mplfinance as mpf


def plot_candlestick_matplotlib(df,
                                date_col='trade_time',
                                title='K线图',
                                figsize=(12, 8),
                                style='yahoo',
                                volume=True):
    """
    使用matplotlib绘制K线图
    
    Parameters
    ----------
    df : pandas.DataFrame
        包含OHLCV数据的DataFrame
    date_col : str, default 'trade_time'
        日期列名
    title : str, default 'K线图'
        图表标题
    figsize : tuple, default (12, 8)
        图表尺寸
    style : str, default 'yahoo'
        K线图样式，可选 'yahoo', 'charles', 'binance' 等
    volume : bool, default True
        是否显示成交量
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        matplotlib图形对象
    ax : matplotlib.axes.Axes
        matplotlib轴对象
    """
    # 创建副本并设置索引
    df_plot = df.copy()
    df_plot[date_col] = pd.to_datetime(df_plot[date_col])
    df_plot.set_index(date_col, inplace=True)

    # 重命名列以匹配mplfinance要求
    cols_rename = {
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close'
    }

    if 'volume' in df_plot.columns:
        cols_rename['volume'] = 'Volume'
    elif 'vol' in df_plot.columns:
        cols_rename['vol'] = 'Volume'

    # 重命名列
    for old, new in cols_rename.items():
        if old in df_plot.columns:
            df_plot.rename(columns={old: new}, inplace=True)

    # 确保所有必要的列都存在
    required_cols = ['Open', 'High', 'Low', 'Close']
    for col in required_cols:
        if col not in df_plot.columns:
            raise ValueError(f"DataFrame中缺少{col}列")

    # 绘制K线图
    if volume and 'Volume' in df_plot.columns:
        mpf.plot(df_plot,
                 type='candle',
                 style=style,
                 volume=True,
                 title=title,
                 figsize=figsize,
                 returnfig=True)
    else:
        mpf.plot(df_plot,
                 type='candle',
                 style=style,
                 volume=False,
                 title=title,
                 figsize=figsize,
                 returnfig=True)

    fig = plt.gcf()
    ax = plt.gca()

    return fig, ax


def plot_candlestick_plotly(df,
                            date_col='trade_time',
                            title='K线图',
                            indicators=None,
                            max_candles=500,
                            return_fig=False):
    """
    使用plotly绘制K线图
    
    Parameters
    ----------
    df : pandas.DataFrame
        包含OHLCV数据的DataFrame
    date_col : str, default 'trade_time'
        日期列名
    title : str, default 'K线图'
        图表标题
    indicators : dict, default None
        指标字典，格式为{name: {'line': series, 'color': color}}
    max_candles : int, default 500
        最大显示的K线数量
    return_fig : bool, default False
        是否返回图形对象
    
    Returns
    -------
    fig : plotly.graph_objects.Figure or None
        如果return_fig为True，返回plotly图形对象
    """
    # 创建副本并确保日期列是日期时间类型
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # 如果数据超过max_candles，则只取最后max_candles条
    if len(df) > max_candles:
        df = df.tail(max_candles)

    # 创建图表
    fig = make_subplots(rows=2,
                        cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.05,
                        row_heights=[0.7, 0.3],
                        subplot_titles=(title, "成交量"))

    # 添加K线图
    fig.add_trace(go.Candlestick(x=df[date_col],
                                 open=df['open'],
                                 high=df['high'],
                                 low=df['low'],
                                 close=df['close'],
                                 name='价格',
                                 increasing_line_color='red',
                                 decreasing_line_color='green'),
                  row=1,
                  col=1)

    # 添加指标
    if indicators:
        for name, indicator in indicators.items():
            fig.add_trace(go.Scatter(x=df[date_col],
                                     y=indicator['line'],
                                     name=name,
                                     line=dict(color=indicator.get(
                                         'color', 'blue'),
                                               width=1)),
                          row=1,
                          col=1)

    # 添加成交量图
    volume_col = 'volume' if 'volume' in df.columns else 'vol'
    if volume_col in df.columns:
        # 计算交易量颜色 - 涨红跌绿
        colors = []
        for i in range(len(df)):
            if i > 0:
                if df['close'].iloc[i] > df['close'].iloc[i - 1]:
                    colors.append('red')
                else:
                    colors.append('green')
            else:
                colors.append('gray')

        fig.add_trace(go.Bar(x=df[date_col],
                             y=df[volume_col],
                             name='成交量',
                             marker_color=colors),
                      row=2,
                      col=1)

    # 更新布局
    fig.update_layout(xaxis_rangeslider_visible=False,
                      height=600,
                      width=1000,
                      showlegend=True,
                      legend=dict(orientation="h",
                                  yanchor="bottom",
                                  y=1.02,
                                  xanchor="center",
                                  x=0.5))

    if return_fig:
        return fig

    fig.show()
    return None


def plot_correlation_matrix(df,
                            title='相关性矩阵',
                            figsize=(10, 8),
                            cmap='coolwarm',
                            return_fig=False):
    """
    绘制相关性矩阵热图
    
    Parameters
    ----------
    df : pandas.DataFrame
        输入数据
    title : str, default '相关性矩阵'
        图表标题
    figsize : tuple, default (10, 8)
        图表尺寸
    cmap : str, default 'coolwarm'
        热图颜色映射
    return_fig : bool, default False
        是否返回图形对象
    
    Returns
    -------
    fig : matplotlib.figure.Figure or None
        如果return_fig为True，返回matplotlib图形对象
    """
    # 计算相关性矩阵
    corr = df.corr()

    # 创建热图
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制热图
    mask = np.triu(np.ones_like(corr, dtype=bool))
    heatmap = sns.heatmap(corr,
                          mask=mask,
                          cmap=cmap,
                          vmax=1.0,
                          vmin=-1.0,
                          center=0,
                          square=True,
                          linewidths=0.5,
                          annot=True,
                          fmt=".2f",
                          cbar_kws={"shrink": 0.8},
                          ax=ax)

    # 设置标题和轴标签
    plt.title(title, fontsize=15)

    if return_fig:
        return fig

    plt.tight_layout()
    plt.show()
    return None


def plot_correlation_plotly(df, title='相关性矩阵', return_fig=False):
    """
    使用plotly绘制相关性矩阵热图
    
    Parameters
    ----------
    df : pandas.DataFrame
        输入数据
    title : str, default '相关性矩阵'
        图表标题
    return_fig : bool, default False
        是否返回图形对象
    
    Returns
    -------
    fig : plotly.graph_objects.Figure or None
        如果return_fig为True，返回plotly图形对象
    """
    # 计算相关性矩阵
    corr = df.corr()

    # 创建热图
    fig = go.Figure(data=go.Heatmap(z=corr.values,
                                    x=corr.columns,
                                    y=corr.index,
                                    colorscale='RdBu',
                                    zmin=-1.0,
                                    zmax=1.0,
                                    text=corr.round(2).values,
                                    texttemplate="%{text:.2f}",
                                    colorbar=dict(title="相关系数")))

    # 更新布局
    fig.update_layout(title=title,
                      height=600,
                      width=800,
                      xaxis_showgrid=False,
                      yaxis_showgrid=False,
                      xaxis_title='',
                      yaxis_title='')

    if return_fig:
        return fig

    fig.show()
    return None


def plot_return_distribution(returns,
                             title='收益率分布',
                             bins=50,
                             kde=True,
                             figsize=(12, 6),
                             return_fig=False):
    """
    绘制收益率分布图
    
    Parameters
    ----------
    returns : pandas.Series or numpy.ndarray
        收益率数据
    title : str, default '收益率分布'
        图表标题
    bins : int, default 50
        直方图箱数
    kde : bool, default True
        是否显示核密度估计曲线
    figsize : tuple, default (12, 6)
        图表尺寸
    return_fig : bool, default False
        是否返回图形对象
    
    Returns
    -------
    fig : matplotlib.figure.Figure or None
        如果return_fig为True，返回matplotlib图形对象
    """
    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制直方图和KDE
    sns.histplot(returns, bins=bins, kde=kde, ax=ax)

    # 添加垂直线(0处)
    plt.axvline(x=0, color='r', linestyle='--')

    # 添加统计信息
    mean = np.mean(returns)
    std = np.std(returns)
    skew = pd.Series(returns).skew()
    kurt = pd.Series(returns).kurtosis()

    stats_text = (f"均值: {mean:.4f}\n"
                  f"标准差: {std:.4f}\n"
                  f"偏度: {skew:.4f}\n"
                  f"峰度: {kurt:.4f}")

    # 在图表右上角添加统计信息
    plt.annotate(stats_text,
                 xy=(0.95, 0.95),
                 xycoords='axes fraction',
                 ha='right',
                 va='top',
                 bbox=dict(boxstyle='round', fc='white', ec='k', alpha=0.8))

    # 设置标题和轴标签
    plt.title(title, fontsize=15)
    plt.xlabel('收益率')
    plt.ylabel('频率')
    plt.grid(True, alpha=0.3)

    if return_fig:
        return fig

    plt.tight_layout()
    plt.show()
    return None


def plot_return_distribution_plotly(returns,
                                    title='收益率分布',
                                    bins=50,
                                    return_fig=False):
    """
    使用plotly绘制收益率分布图
    
    Parameters
    ----------
    returns : pandas.Series or numpy.ndarray
        收益率数据
    title : str, default '收益率分布'
        图表标题
    bins : int, default 50
        直方图箱数
    return_fig : bool, default False
        是否返回图形对象
    
    Returns
    -------
    fig : plotly.graph_objects.Figure or None
        如果return_fig为True，返回plotly图形对象
    """
    # 计算统计信息
    mean = np.mean(returns)
    std = np.std(returns)
    skew = pd.Series(returns).skew()
    kurt = pd.Series(returns).kurtosis()

    # 创建直方图
    fig = go.Figure()

    # 添加直方图
    fig.add_trace(
        go.Histogram(x=returns,
                     nbinsx=bins,
                     name='收益率',
                     marker_color='lightblue',
                     opacity=0.75))

    # 添加核密度估计曲线
    x_kde = np.linspace(min(returns), max(returns), 1000)
    kde = sns.kdeplot(returns).get_lines()[0].get_data()

    fig.add_trace(
        go.Scatter(x=kde[0],
                   y=kde[1],
                   mode='lines',
                   name='核密度估计',
                   line=dict(color='red', width=2)))

    # 添加垂直线(0处)
    fig.add_shape(type='line',
                  x0=0,
                  y0=0,
                  x1=0,
                  y1=int(max(np.histogram(returns, bins=bins)[0])),
                  line=dict(color='red', width=2, dash='dash'))

    # 添加均值线
    fig.add_shape(type='line',
                  x0=mean,
                  y0=0,
                  x1=mean,
                  y1=int(max(np.histogram(returns, bins=bins)[0]) * 0.8),
                  line=dict(color='green', width=2, dash='dash'))

    # 添加统计信息
    fig.add_annotation(x=max(returns) * 0.8,
                       y=int(max(np.histogram(returns, bins=bins)[0]) * 0.8),
                       text=(f"均值: {mean:.4f}<br>"
                             f"标准差: {std:.4f}<br>"
                             f"偏度: {skew:.4f}<br>"
                             f"峰度: {kurt:.4f}"),
                       showarrow=False,
                       align='right',
                       bgcolor='white',
                       bordercolor='black',
                       borderwidth=1,
                       borderpad=4)

    # 更新布局
    fig.update_layout(title=title,
                      xaxis_title='收益率',
                      yaxis_title='频率',
                      height=500,
                      width=800,
                      showlegend=True,
                      legend=dict(orientation="h",
                                  yanchor="bottom",
                                  y=1.02,
                                  xanchor="center",
                                  x=0.5))

    if return_fig:
        return fig

    fig.show()
    return None


def plot_rolling_statistics(time_series,
                            window=20,
                            title='滚动统计量',
                            figsize=(14, 10),
                            return_fig=False):
    """
    绘制时间序列的滚动统计量
    
    Parameters
    ----------
    time_series : pandas.Series
        时间序列数据
    window : int, default 20
        滚动窗口大小
    title : str, default '滚动统计量'
        图表标题
    figsize : tuple, default (14, 10)
        图表尺寸
    return_fig : bool, default False
        是否返回图形对象
    
    Returns
    -------
    fig : matplotlib.figure.Figure or None
        如果return_fig为True，返回matplotlib图形对象
    """
    # 计算滚动统计量
    rolling_mean = time_series.rolling(window=window).mean()
    rolling_std = time_series.rolling(window=window).std()
    rolling_min = time_series.rolling(window=window).min()
    rolling_max = time_series.rolling(window=window).max()

    # 创建子图
    fig, axs = plt.subplots(4, 1, figsize=figsize, sharex=True)

    # 绘制原始时间序列和滚动均值
    axs[0].plot(time_series, label='原始数据', color='blue', alpha=0.7)
    axs[0].plot(rolling_mean, label=f'{window}期滚动均值', color='red')
    axs[0].legend()
    axs[0].set_title(f'原始数据和{window}期滚动均值')
    axs[0].grid(True, alpha=0.3)

    # 绘制滚动标准差
    axs[1].plot(rolling_std, label=f'{window}期滚动标准差', color='orange')
    axs[1].legend()
    axs[1].set_title(f'{window}期滚动标准差')
    axs[1].grid(True, alpha=0.3)

    # 绘制滚动最大值和最小值
    axs[2].plot(rolling_max, label=f'{window}期滚动最大值', color='green')
    axs[2].plot(rolling_min, label=f'{window}期滚动最小值', color='purple')
    axs[2].legend()
    axs[2].set_title(f'{window}期滚动最大值和最小值')
    axs[2].grid(True, alpha=0.3)

    # 绘制布林带
    upper_band = rolling_mean + (rolling_std * 2)
    lower_band = rolling_mean - (rolling_std * 2)

    axs[3].plot(time_series, label='原始数据', color='blue', alpha=0.7)
    axs[3].plot(rolling_mean, label='滚动均值', color='red')
    axs[3].plot(upper_band, label='上轨(+2σ)', color='green', linestyle='--')
    axs[3].plot(lower_band, label='下轨(-2σ)', color='green', linestyle='--')
    axs[3].fill_between(time_series.index,
                        lower_band,
                        upper_band,
                        alpha=0.1,
                        color='green')
    axs[3].legend()
    axs[3].set_title('布林带 (均值 ± 2σ)')
    axs[3].grid(True, alpha=0.3)

    # 格式化日期轴
    plt.gcf().autofmt_xdate()

    # 设置总标题
    plt.suptitle(title, fontsize=16)

    if return_fig:
        return fig

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()
    return None
