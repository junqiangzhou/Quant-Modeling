import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt


class PerformanceAnalyzer(object):
    """
    性能分析工具
    """

    @staticmethod
    def analyze_returns(returns, benchmark_returns=None):
        """
        分析策略收益表现
        
        参数:
        returns (pandas Series): 策略日收益率
        benchmark_returns (pandas Series, optional): 基准日收益率
        
        返回:
        dict: 性能指标字典
        """
        # 确保returns是pandas Series
        if not isinstance(returns, pd.Series):
            returns = pd.Series(returns)

        # 如果有基准，确保它也是pandas Series
        if benchmark_returns is not None and not isinstance(
                benchmark_returns, pd.Series):
            benchmark_returns = pd.Series(benchmark_returns)

        # 计算累计收益
        cum_returns = (1 + returns).cumprod() - 1

        # 计算年化收益率 (假设252个交易日)
        n_days = len(returns)
        n_years = n_days / 252
        annual_return = ((1 + cum_returns.iloc[-1])**(1 / n_years)) - 1

        # 计算波动率 (年化)
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252)

        # 计算夏普比率 (假设无风险利率为0)
        sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0

        # 计算最大回撤
        rolling_max = cum_returns.cummax()
        drawdown = (cum_returns - rolling_max) / (1 + rolling_max)
        max_drawdown = drawdown.min()

        # 如果有基准收益率，计算相对指标
        alpha, beta = 0, 0
        if benchmark_returns is not None:
            # 确保基准收益率与策略收益率有相同的日期
            common_idx = returns.index.intersection(benchmark_returns.index)
            if len(common_idx) > 0:
                returns = returns.loc[common_idx]
                benchmark_returns = benchmark_returns.loc[common_idx]

                # 计算Beta (市场敏感度)
                covar = returns.cov(benchmark_returns)
                benchmark_var = benchmark_returns.var()
                beta = covar / benchmark_var if benchmark_var > 0 else 0

                # 计算Alpha (超额收益)
                benchmark_annual_return = (
                    (1 + (1 + benchmark_returns).cumprod().iloc[-1])
                    **(1 / n_years)) - 1
                alpha = annual_return - (beta * benchmark_annual_return)

        # 汇总结果
        results = {
            'Total Return': cum_returns.iloc[-1],
            'Annual Return': annual_return,
            'Annual Volatility': annual_vol,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown,
            'Alpha': alpha,
            'Beta': beta
        }

        return results

    @staticmethod
    def plot_returns(returns,
                     benchmark_returns=None,
                     title="Strategy Performance"):
        """
        绘制策略收益曲线
        
        参数:
        returns (pandas Series): 策略日收益率
        benchmark_returns (pandas Series, optional): 基准日收益率
        title (str): 图表标题
        """
        plt.figure(figsize=(12, 8))

        # 计算累计收益
        cum_returns = (1 + returns).cumprod() - 1

        # 绘制策略收益曲线
        plt.plot(cum_returns.index,
                 cum_returns.values,
                 label='Strategy',
                 linewidth=2)

        # 如果有基准，绘制基准收益曲线
        if benchmark_returns is not None:
            # 确保基准收益率与策略收益率有相同的日期
            common_idx = returns.index.intersection(benchmark_returns.index)
            if len(common_idx) > 0:
                benchmark_returns = benchmark_returns.loc[common_idx]
                cum_benchmark = (1 + benchmark_returns).cumprod() - 1
                plt.plot(cum_benchmark.index,
                         cum_benchmark.values,
                         label='Benchmark',
                         linewidth=2,
                         alpha=0.7)

        # 绘制零线
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        # 设置图表
        plt.title(title, fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Cumulative Returns', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)

        return plt.gcf()


def plot_performance_analysis(results):
    """  
    绘制策略性能分析图表，修复子图类型兼容性问题  
    
    参数:  
    results (dict): 回测结果字典  
    
    返回:  
    plotly.graph_objects.Figure: Plotly图形对象  
    """

    # 创建子图，指定正确的子图类型
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{
                "type": "domain"
            }, {
                "type": "xy"
            }],  # 第一行: 饼图 (domain类型), 表格  
            [{
                "type": "xy"
            }, {
                "type": "xy"
            }]  # 第二行: xy图表  
        ],
        subplot_titles=("交易胜率", f"总收益: {results.get('total_return', 0):.2f}%",
                        f"单笔交易收益", f"资金曲线"))

    # 1. 胜率饼图 (左上)
    win_trades = results.get('winning_trades', 0)
    lose_trades = results.get('losing_trades', 0)
    total_trades = results.get('total_trades', 0)

    if total_trades > 0:
        win_pct = win_trades / total_trades * 100
        lose_pct = lose_trades / total_trades * 100

        fig.add_trace(
            go.Pie(
                labels=['盈利交易', '亏损交易'],
                values=[win_trades, lose_trades],  # 使用实际交易次数而非百分比  
                textinfo='percent+label',
                marker=dict(colors=['green', 'red']),
                hole=0.4,
                hoverinfo='label+percent+value'),
            row=1,
            col=1)

        # # 添加交易次数注释
        # fig.add_annotation(
        #     text=f"总交易: {total_trades}次<br>胜率: {win_pct:.1f}%",
        #     x=0.5, y=0.5,
        #     showarrow=False,
        #     font=dict(size=12),
        #     xref="x domain",
        #     yref="y domain",
        #     row=1, col=1
        # )

    # 2. 回报统计表格 (右上)
    # 注意：table不添加到子图中，而是作为独立的图形
    summary_data = go.Table(
        header=dict(values=['<b>指标</b>', '<b>数值</b>'],
                    fill_color='royalblue',
                    align='center',
                    font=dict(color='white', size=12)),
        cells=dict(values=[
            ['初始资金', '最终资金', '总收益率', '夏普比率', '最大回撤', '交易次数', '胜率'],
            [
                f"${results.get('initial_cash', 0):,.2f}",
                f"${results.get('final_value', 0):,.2f}",
                f"{results.get('total_return', 0):.2f}%",
                f"{results.get('sharpe_ratio', 0):.2f}",
                f"{results.get('max_drawdown', 0):.2f}%", f"{total_trades}",
                f"{win_pct:.2f}%" if total_trades > 0 else 'N/A'
            ]
        ],
                   align='center'))

    # 创建一个单独的图表来显示表格
    table_fig = go.Figure(data=[summary_data])
    table_fig.update_layout(title="策略性能数据",
                            height=300,
                            width=600,
                            margin=dict(l=0, r=0, t=40, b=0))

    # 3. 单笔交易收益率 (左下)
    buy_signals = results.get('signals', {}).get('buy_signals', [])
    sell_signals = results.get('signals', {}).get('sell_signals', [])

    if buy_signals and sell_signals:
        # 计算每笔交易的收益率
        trade_returns = []
        for i in range(min(len(buy_signals), len(sell_signals))):
            buy_date, buy_price = buy_signals[i]
            sell_date, sell_price = sell_signals[i]
            if sell_date > buy_date:  # 确保卖出在买入之后
                profit_pct = (sell_price - buy_price) / buy_price * 100
                trade_returns.append((buy_date, sell_date, profit_pct))

        # 绘制每笔交易的收益率
        if trade_returns:
            dates = [tr[0] for tr in trade_returns]  # 使用买入日期
            returns = [tr[2] for tr in trade_returns]
            colors = ['green' if r >= 0 else 'red' for r in returns]

            fig.add_trace(go.Bar(x=dates,
                                 y=returns,
                                 name="单笔交易收益率",
                                 marker_color=colors),
                          row=2,
                          col=1)

            # 添加均线
            if len(returns) > 1:
                fig.add_trace(go.Scatter(x=dates,
                                         y=[sum(returns) / len(returns)] *
                                         len(dates),
                                         name="平均收益率",
                                         line=dict(color='blue',
                                                   width=2,
                                                   dash='dash')),
                              row=2,
                              col=1)

        # 4. 资金曲线 (右下)
        initial_value = results.get('initial_cash', 100000)
        final_value = results.get('final_value', initial_value)

        # 简单模拟资金曲线
        if trade_returns:
            all_dates = sorted([date for date, _ in buy_signals] +
                               [date for date, _ in sell_signals])
            equity_curve = [initial_value]
            dates = [all_dates[0]]

            for i, trade in enumerate(trade_returns):
                buy_date, sell_date, profit_pct = trade

                # 计算当前权益
                current_equity = equity_curve[-1] * (1 + profit_pct / 100)
                equity_curve.append(current_equity)
                dates.append(sell_date)

            # 确保最后一个点是最终资金
            if equity_curve[-1] != final_value and len(equity_curve) > 1:
                equity_curve[-1] = final_value

            fig.add_trace(go.Scatter(x=dates,
                                     y=equity_curve,
                                     name="资金曲线",
                                     line=dict(color='blue', width=2),
                                     fill='tozeroy'),
                          row=2,
                          col=2)

    # 更新布局
    fig.update_layout(title="策略性能分析", height=800, width=1200, showlegend=True)

    # 设置Y轴标题
    fig.update_yaxes(title_text="收益率 (%)", row=2, col=1)
    fig.update_yaxes(title_text="资金", row=2, col=2)

    # 输出表格图表和主分析图表
    return fig, table_fig


def plot_backtest_results(df, results, max_candles=200, title=None):
    """  
    绘制回测结果，包括K线图、交易信号和持仓变化  
    
    参数:  
    df (pandas.DataFrame): 包含OHLCV数据的DataFrame  
    results (dict): 回测结果字典，必须包含'signals'键，其中包含买卖信号  
    max_candles (int): 最大显示的K线数量  
    title (str): 图表标题，如果为None则使用默认标题  
    
    返回:  
    plotly.graph_objects.Figure: Plotly图形对象  
    """
    # 从结果中提取信号
    signals = results.get('signals', {})
    buy_signals = signals.get('buy_signals', [])
    sell_signals = signals.get('sell_signals', [])
    position_size = signals.get('position_size', [])

    # 复制数据防止修改原数据
    df = df.copy()

    # 确保日期在索引或者有trade_time列
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        date_col = df.columns[0]
    elif 'trade_time' in df.columns:
        date_col = 'trade_time'
        df[date_col] = pd.to_datetime(df[date_col])
    else:
        # 尝试找到日期列
        date_candidates = ['datetime', 'date', 'time']
        date_col = None
        for col in date_candidates:
            if col in df.columns:
                date_col = col
                df[date_col] = pd.to_datetime(df[date_col])
                break

        if date_col is None:
            raise ValueError("找不到日期列，请确保DataFrame包含日期列或日期索引")

    # 识别交易量列
    vol_col = None
    if 'Volume' in df.columns:
        vol_col = 'Volume'

    # 限制K线数量
    if len(df) > max_candles:
        df = df.iloc[-max_candles:]

    # 创建子图
    fig = make_subplots(rows=3,
                        cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.03,
                        row_heights=[0.6, 0.2, 0.2],
                        subplot_titles=("价格", "交易量", "持仓"))

    # 添加K线图
    fig.add_trace(
        go.Candlestick(
            x=df[date_col],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="K线",
            increasing_line_color='red',  # 中国市场习惯 - 红涨  
            decreasing_line_color='green'  # 中国市场习惯 - 绿跌  
        ),
        row=1,
        col=1)

    # 添加交易量图
    if vol_col:
        # 计算交易量颜色
        colors = [
            'red' if df['Close'].iloc[i] >= df['Open'].iloc[i] else 'green'
            for i in range(len(df))
        ]

        fig.add_trace(go.Bar(x=df[date_col],
                             y=df[vol_col],
                             name="交易量",
                             marker_color=colors,
                             opacity=0.7),
                      row=2,
                      col=1)

    # 添加买入信号
    if buy_signals:
        buy_dates = [date for date, _ in buy_signals]
        buy_prices = [price for _, price in buy_signals]

        fig.add_trace(go.Scatter(x=buy_dates,
                                 y=buy_prices,
                                 mode='markers',
                                 name='买入',
                                 marker=dict(symbol='triangle-up',
                                             size=12,
                                             color='red',
                                             line=dict(width=2, color='red'))),
                      row=1,
                      col=1)

    # 添加卖出信号
    if sell_signals:
        sell_dates = [date for date, _ in sell_signals]
        sell_prices = [price for _, price in sell_signals]

        fig.add_trace(go.Scatter(x=sell_dates,
                                 y=sell_prices,
                                 mode='markers',
                                 name='卖出',
                                 marker=dict(symbol='triangle-down',
                                             size=12,
                                             color='green',
                                             line=dict(width=2,
                                                       color='green'))),
                      row=1,
                      col=1)

    # 添加持仓变化图
    if position_size:
        pos_dates = [date for date, _ in position_size]
        pos_sizes = [size for _, size in position_size]

        fig.add_trace(go.Scatter(x=pos_dates,
                                 y=pos_sizes,
                                 name="持仓",
                                 line=dict(color='blue', width=2),
                                 fill='tozeroy'),
                      row=3,
                      col=1)

    # 设置图表标题
    if title is None:
        title = f"回测结果 - 总收益率: {results.get('total_return', 0):.2f}%"

    # 更新布局
    fig.update_layout(title=title,
                      xaxis_rangeslider_visible=False,
                      height=900,
                      width=1200,
                      showlegend=True,
                      legend=dict(orientation="h",
                                  yanchor="bottom",
                                  y=1.02,
                                  xanchor="right",
                                  x=1))

    # 设置Y轴标题
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="交易量", row=2, col=1)
    fig.update_yaxes(title_text="持仓", row=3, col=1)

    return fig
