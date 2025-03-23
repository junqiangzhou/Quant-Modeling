# visualization.py

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def calculate_metrics(df, signals, initial_cash=100000):
    """
    根据回测数据和信号计算性能指标（示例版本，可根据实际需求修改）。
    
    参数:
        df (pd.DataFrame): 回测时使用的行情数据（30分钟级）。
        signals (dict): 回测生成的信号字典，通常包含 'buy', 'sell', 以及其他信息。
        initial_cash (float): 初始资金。
    
    返回:
        dict: 包含主要性能指标的字典。
    """
    results = {'initial_cash': initial_cash, 'signals': signals}

    # 交易次数（假设最小值为真实交易次数）
    buy_signals = len(signals.get('buy', []))
    sell_signals = len(signals.get('sell', []))
    results['total_trades'] = min(buy_signals, sell_signals)

    # 如果有 trades 信息，可进一步计算胜率（示例）
    if 'trades' in signals:
        winning_trades = sum(1 for trade in signals['trades']
                             if trade['profit'] > 0)
        results['winning_trades'] = winning_trades
        results['losing_trades'] = len(signals['trades']) - winning_trades
        results['win_rate'] = winning_trades / len(
            signals['trades']) if signals['trades'] else 0
    else:
        results['winning_trades'] = 0
        results['losing_trades'] = 0
        results['win_rate'] = 0

    # 收益率（如果回测返回了 final_value，可直接使用）
    if 'final_value' in signals:
        results['final_value'] = signals['final_value']
        results['total_return'] = (results['final_value'] -
                                   initial_cash) / initial_cash
    else:
        results['final_value'] = initial_cash
        results['total_return'] = 0

    # 夏普比率（示例计算）
    if 'returns' in signals:
        returns = pd.Series(signals['returns'])
        risk_free_rate = 0.02  # 年化2%
        # 假设30分钟级别每年大约有3250根数据
        freq_per_year = 3250
        excess_returns = returns - (risk_free_rate / freq_per_year)
        if len(returns) > 1 and excess_returns.std() != 0:
            results['sharpe_ratio'] = np.sqrt(
                freq_per_year) * excess_returns.mean() / excess_returns.std()
        else:
            results['sharpe_ratio'] = 0
    else:
        results['sharpe_ratio'] = 0

    # 最大回撤（示例）
    if 'equity_curve' in signals:
        equity_curve = pd.Series(signals['equity_curve'])
        rolling_max = equity_curve.expanding().max()
        drawdowns = (equity_curve - rolling_max) / rolling_max
        results['max_drawdown'] = abs(
            drawdowns.min()) if len(drawdowns) > 0 else 0
    else:
        results['max_drawdown'] = 0

    return results


def find_nearest_time(df,
                      signal_time,
                      date_col='trade_time',
                      tolerance=pd.Timedelta('15min')):
    """
    在 DataFrame 中查找与 signal_time 最接近的时间戳，
    如果最小差值在 tolerance 内则返回对应时间，否则返回 None。
    """
    diffs = (df[date_col] - signal_time).abs()
    min_diff = diffs.min()
    if min_diff <= tolerance:
        return df.loc[diffs.idxmin(), date_col]
    else:
        return None


def plot_backtest_signals_30m(df,
                              signals,
                              date_col='trade_time',
                              title='交易信号分析 (30分钟级)'):
    """
    在 30 分钟级别数据上绘制 K 线、成交量以及买卖信号。
    
    参数:
        df (pd.DataFrame): 30分钟级别的行情数据 (须含 open, high, low, close, volume/vol)。
        signals (dict): 包含 'buy' 和 'sell' 信号的字典。每个信号元素可以是 datetime 或 (datetime, price) 等形式。
        date_col (str): 时间列名称，默认 'trade_time'。
        title (str): 图表标题。
    """
    # 确保时间列是 datetime 类型
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df.sort_values(by=date_col, inplace=True)

    # 创建子图：上方K线，下方成交量
    fig = make_subplots(rows=2,
                        cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.05,
                        row_heights=[0.7, 0.3],
                        subplot_titles=('价格与交易信号', '成交量'))

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

    # 添加成交量图
    volume_col = 'volume' if 'volume' in df.columns else 'vol'
    if volume_col in df.columns:
        colors = []
        close_vals = df['close'].values
        for i in range(len(df)):
            if i == 0:
                colors.append('gray')
            else:
                colors.append('red' if close_vals[i] >
                              close_vals[i - 1] else 'green')

        fig.add_trace(go.Bar(x=df[date_col],
                             y=df[volume_col],
                             name='成交量',
                             marker_color=colors),
                      row=2,
                      col=1)

    # 处理买入信号（使用最近匹配）
    buy_signals = signals.get('buy', [])
    if buy_signals:
        buy_x = []
        buy_y = []
        for sig in buy_signals:
            if isinstance(sig, tuple):
                sig_time = pd.to_datetime(sig[0])
            else:
                sig_time = pd.to_datetime(sig)

            # 查找与信号时间最接近的行情时间
            nearest_time = find_nearest_time(df,
                                             sig_time,
                                             date_col,
                                             tolerance=pd.Timedelta('15min'))
            if nearest_time is not None:
                mask = (df[date_col] == nearest_time)
                if mask.any():
                    low_price = df.loc[mask, 'low'].iloc[0]
                    buy_x.append(nearest_time)
                    buy_y.append(low_price * 0.995)

        if buy_x:
            fig.add_trace(go.Scatter(
                x=buy_x,
                y=buy_y,
                mode='markers',
                marker=dict(symbol='triangle-up',
                            size=10,
                            color='red',
                            line=dict(width=1, color='darkred')),
                name='买入信号',
                hovertemplate='买入时间: %{x}<br>价格: %{y:.2f}<extra></extra>'),
                          row=1,
                          col=1)

    # 处理卖出信号（使用最近匹配）
    sell_signals = signals.get('sell', [])
    if sell_signals:
        sell_x = []
        sell_y = []
        for sig in sell_signals:
            if isinstance(sig, tuple):
                sig_time = pd.to_datetime(sig[0])
            else:
                sig_time = pd.to_datetime(sig)

            nearest_time = find_nearest_time(df,
                                             sig_time,
                                             date_col,
                                             tolerance=pd.Timedelta('15min'))
            if nearest_time is not None:
                mask = (df[date_col] == nearest_time)
                if mask.any():
                    high_price = df.loc[mask, 'high'].iloc[0]
                    sell_x.append(nearest_time)
                    sell_y.append(high_price * 1.005)

        if sell_x:
            fig.add_trace(go.Scatter(
                x=sell_x,
                y=sell_y,
                mode='markers',
                marker=dict(symbol='triangle-down',
                            size=10,
                            color='green',
                            line=dict(width=1, color='darkgreen')),
                name='卖出信号',
                hovertemplate='卖出时间: %{x}<br>价格: %{y:.2f}<extra></extra>'),
                          row=1,
                          col=1)

    # 更新布局
    fig.update_layout(title=dict(text=title,
                                 x=0.5,
                                 xanchor='center',
                                 font=dict(size=20)),
                      xaxis_rangeslider_visible=False,
                      height=800,
                      width=1200,
                      showlegend=True,
                      legend=dict(orientation="h",
                                  yanchor="bottom",
                                  y=1.02,
                                  xanchor="center",
                                  x=0.5))

    fig.update_xaxes(showgrid=True, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridcolor='lightgray')

    fig.show()


def plot_performance_metrics(results):
    """
    绘制性能指标和交易统计图表。
    
    参数:
        results (dict): 回测结果字典，包含以下关键字段：
            - total_return (float): 总收益率(小数形式)
            - sharpe_ratio (float): 夏普比率
            - max_drawdown (float): 最大回撤(小数形式)
            - win_rate (float): 胜率(小数形式)
            - initial_cash (float)
            - final_value (float)
            - total_trades (int)
            - winning_trades (int)
            - losing_trades (int)
    """
    fig1 = make_subplots(rows=1,
                         cols=2,
                         specs=[[{
                             "type": "xy"
                         }, {
                             "type": "domain"
                         }]],
                         subplot_titles=('收益指标', '胜率分析'))

    metrics = [
        results['total_return'] * 100,  # 转为百分比
        results['sharpe_ratio'],
        results['max_drawdown'] * 100  # 转为百分比
    ]
    labels = ['总收益率(%)', '夏普比率', '最大回撤(%)']
    colors = ['green' if m > 0 else 'red' for m in metrics]
    colors[2] = 'red'  # 最大回撤始终用红色

    fig1.add_trace(go.Bar(x=labels,
                          y=metrics,
                          marker_color=colors,
                          text=[f"{x:.2f}" for x in metrics],
                          textposition="auto",
                          name="性能指标"),
                   row=1,
                   col=1)

    fig1.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)

    win_rate = results['win_rate'] * 100
    loss_rate = 100 - win_rate
    fig1.add_trace(go.Pie(
        labels=['胜率', '败率'],
        values=[win_rate, loss_rate],
        marker_colors=['green', 'red'],
        textinfo="label+percent",
        hole=0.4,
        hovertemplate="<b>%{label}</b><br>比例: %{percent}<br><extra></extra>"),
                   row=1,
                   col=2)

    fig1.update_layout(title=dict(text="策略性能分析",
                                  x=0.5,
                                  xanchor='center',
                                  font=dict(size=20)),
                       height=400,
                       width=1200,
                       showlegend=False,
                       yaxis=dict(title="数值",
                                  gridcolor='lightgray',
                                  zerolinecolor='gray',
                                  zerolinewidth=1))

    fig1.show()

    # 交易统计表格
    stats = [["初始资金", f"¥{results['initial_cash']:,.2f}"],
             ["最终资金", f"¥{results['final_value']:,.2f}"],
             ["总交易次数", str(results['total_trades'])],
             ["盈利交易", str(results['winning_trades'])],
             ["亏损交易", str(results['losing_trades'])],
             ["胜率", f"{results['win_rate']*100:.2f}%"],
             ["夏普比率", f"{results['sharpe_ratio']:.2f}"],
             ["最大回撤", f"{results['max_drawdown']*100:.2f}%"]]

    fig2 = go.Figure(data=[
        go.Table(header=dict(values=['<b>指标</b>', '<b>数值</b>'],
                             fill_color='lightgrey',
                             align='left',
                             font=dict(size=12)),
                 cells=dict(values=list(zip(*stats)),
                            fill_color=[['white', 'white']],
                            align='left',
                            font=dict(size=11),
                            height=30))
    ])

    fig2.update_layout(title=dict(text="交易统计",
                                  x=0.5,
                                  xanchor='center',
                                  font=dict(size=20)),
                       height=300,
                       width=1200)

    fig2.show()


def create_backtest_report(df,
                           signals,
                           date_col='trade_time',
                           initial_cash=100000):
    """
    生成回测报告：打印统计信息、绘制30分钟级K线信号图和性能指标图。
    
    参数:
        df (pd.DataFrame): 30分钟级别的行情数据
        signals (dict): 回测产生的信号和统计信息字典
        date_col (str): 时间列名称
        initial_cash (float): 初始资金
    """
    results = calculate_metrics(df, signals, initial_cash)

    print("=" * 50)
    print("回测结果统计")
    print("=" * 50)
    print(f"初始资金: {results['initial_cash']:,.2f}")
    print(f"最终资金: {results['final_value']:,.2f}")
    print(f"总收益率: {results['total_return']*100:.2f}%")
    print(f"夏普比率: {results['sharpe_ratio']:.2f}")
    print(f"最大回撤: {results['max_drawdown']*100:.2f}%")
    print(f"总交易次数: {results['total_trades']}")
    print(f"胜率: {results['win_rate']*100:.2f}%")
    print("=" * 50)

    # 绘制 30 分钟级别的交易信号图
    plot_backtest_signals_30m(df, signals, date_col, title='交易信号分析 (30分钟级)')

    # 绘制性能指标图
    plot_performance_metrics(results)
