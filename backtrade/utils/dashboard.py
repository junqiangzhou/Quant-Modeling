"""
回测结果仪表盘模块

这个模块提供了一个高度集成的可交互仪表盘，用于可视化回测结果。
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from IPython.display import display, HTML

# 导入自定义模块
from backtrade.utils.basic_visualization import plot_candlestick_plotly, plot_correlation_matrix, plot_return_distribution
from .strategy_visualization import (visualize_trading_signals,
                                     plot_strategy_performance,
                                     create_drawdown_chart,
                                     visualize_monthly_returns,
                                     plot_trade_analysis)


class BacktestDashboard:
    """
    回测结果仪表盘类，提供一个集成的界面来可视化回测结果
    """

    def __init__(self,
                 df=None,
                 results=None,
                 trades=None,
                 date_col='trade_time',
                 use_chinese=True,
                 title="回测结果仪表盘"):
        """
        初始化回测仪表盘
        
        参数:
        df (pandas.DataFrame): 包含OHLCV数据的DataFrame
        results (dict): 回测结果字典
        trades (pandas.DataFrame): 包含交易记录的DataFrame
        date_col (str): 日期列名
        use_chinese (bool): 是否使用中文标签
        title (str): 仪表盘标题
        """
        self.df = df.copy() if df is not None else None
        self.results = results
        self.trades = trades
        self.date_col = date_col
        self.use_chinese = use_chinese
        self.title = title

        # 确保日期列是日期时间类型
        if self.df is not None and date_col in self.df.columns:
            self.df[date_col] = pd.to_datetime(self.df[date_col])

    def set_data(self, df=None, results=None, trades=None):
        """
        设置或更新数据
        
        参数:
        df (pandas.DataFrame): 包含OHLCV数据的DataFrame
        results (dict): 回测结果字典
        trades (pandas.DataFrame): 包含交易记录的DataFrame
        """
        if df is not None:
            self.df = df.copy()
            # 确保日期列是日期时间类型
            if self.date_col in self.df.columns:
                self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])

        if results is not None:
            self.results = results

        if trades is not None:
            self.trades = trades

    def create_performance_dashboard(self,
                                     include_equity_curve=True,
                                     include_drawdown=True,
                                     include_monthly_returns=True,
                                     include_trade_analysis=True):
        """
        创建性能分析仪表盘
        
        参数:
        include_equity_curve (bool): 是否包含权益曲线
        include_drawdown (bool): 是否包含回撤分析
        include_monthly_returns (bool): 是否包含月度收益热图
        include_trade_analysis (bool): 是否包含交易分析
        
        返回:
        None: 直接在Notebook中显示仪表盘
        """
        if self.results is None:
            raise ValueError("需要提供回测结果")

        # 标题
        title = self.title
        if self.use_chinese:
            html = f"<h1 style='text-align:center'>{title}</h1>"
        else:
            html = f"<h1 style='text-align:center'>{title}</h1>"

        # 1. 策略性能概览
        perf_fig = plot_strategy_performance(
            self.results,
            use_chinese=self.use_chinese,
            title="策略性能概览"
            if self.use_chinese else "Strategy Performance Overview")

        # 将图表转换为HTML
        perf_html = pio.to_html(perf_fig,
                                include_plotlyjs=False,
                                full_html=False)
        html += perf_html

        # 2. 权益曲线和回撤分析
        if include_equity_curve and include_drawdown and 'equity_curve' in self.results:
            equity_df = pd.DataFrame({
                'date':
                self.results['equity_curve']['dates'],
                'equity':
                self.results['equity_curve']['values']
            })

            drawdown_fig = create_drawdown_chart(
                equity_df,
                date_col='date',
                value_col='equity',
                use_chinese=self.use_chinese,
                title="权益曲线与回撤分析"
                if self.use_chinese else "Equity Curve and Drawdown Analysis")

            # 将图表转换为HTML
            dd_html = pio.to_html(drawdown_fig,
                                  include_plotlyjs=False,
                                  full_html=False)
            html += dd_html

        # 3. 月度收益热图
        if include_monthly_returns and 'monthly_returns' in self.results:
            monthly_fig = visualize_monthly_returns(
                self.results,
                use_chinese=self.use_chinese,
                title="月度收益率热图"
                if self.use_chinese else "Monthly Returns Heatmap")

            # 将图表转换为HTML
            monthly_html = pio.to_html(monthly_fig,
                                       include_plotlyjs=False,
                                       full_html=False)
            html += monthly_html

        # 4. 交易分析
        if include_trade_analysis and self.trades is not None:
            trade_fig = plot_trade_analysis(
                self.trades,
                date_col=self.date_col
                if self.date_col in self.trades.columns else 'entry_date',
                use_chinese=self.use_chinese,
                title="交易分析" if self.use_chinese else "Trade Analysis")

            # 将图表转换为HTML
            trade_html = pio.to_html(trade_fig,
                                     include_plotlyjs=False,
                                     full_html=False)
            html += trade_html

        # 添加必要的JavaScript库
        html = """
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        """ + html

        # 显示仪表盘
        display(HTML(html))

    def create_trading_signals_dashboard(self,
                                         indicators=None,
                                         max_candles=200):
        """
        创建交易信号仪表盘
        
        参数:
        indicators (dict): 指标字典，格式为 {name: {'line': series, 'color': color}}
        max_candles (int): 最大显示的K线数量
        
        返回:
        None: 直接在Notebook中显示仪表盘
        """
        if self.df is None:
            raise ValueError("需要提供数据DataFrame")

        if self.results is None or 'signals' not in self.results:
            raise ValueError("需要提供包含交易信号的回测结果")

        # 标题
        title = self.title
        html = f"<h1 style='text-align:center'>{title}</h1>"

        # 1. K线图和交易信号
        signals_fig = visualize_trading_signals(
            self.df,
            date_col=self.date_col,
            signals=self.results['signals'],
            max_candles=max_candles,
            use_chinese=self.use_chinese,
            title="交易信号可视化"
            if self.use_chinese else "Trading Signals Visualization")

        # 将图表转换为HTML
        signals_html = pio.to_html(signals_fig,
                                   include_plotlyjs=False,
                                   full_html=False)
        html += signals_html

        # 2. 带指标的K线图
        if indicators is not None:
            indicator_fig = plot_candlestick_plotly(
                self.df.tail(max_candles),
                date_col=self.date_col,
                indicators=indicators,
                title="价格与指标" if self.use_chinese else "Price & Indicators")

            # 将图表转换为HTML
            indicator_html = pio.to_html(indicator_fig,
                                         include_plotlyjs=False,
                                         full_html=False)
            html += indicator_html

        # 添加必要的JavaScript库
        html = """
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        """ + html

        # 显示仪表盘
        display(HTML(html))

    def create_full_dashboard(self, indicators=None, max_candles=200):
        """
        创建完整的回测仪表盘，包含所有图表
        
        参数:
        indicators (dict): 指标字典，格式为 {name: {'line': series, 'color': color}}
        max_candles (int): 最大显示的K线数量
        
        返回:
        None: 直接在Notebook中显示仪表盘
        """
        if self.df is None:
            raise ValueError("需要提供数据DataFrame")

        if self.results is None:
            raise ValueError("需要提供回测结果")

        # 标题
        title = self.title
        html = f"<h1 style='text-align:center'>{title}</h1>"

        # 1. 策略性能概览
        perf_fig = plot_strategy_performance(
            self.results,
            use_chinese=self.use_chinese,
            title="策略性能概览"
            if self.use_chinese else "Strategy Performance Overview")

        # 将图表转换为HTML
        perf_html = pio.to_html(perf_fig,
                                include_plotlyjs=False,
                                full_html=False)
        html += perf_html

        # 2. K线图和交易信号
        if 'signals' in self.results:
            signals_fig = visualize_trading_signals(
                self.df,
                date_col=self.date_col,
                signals=self.results['signals'],
                max_candles=max_candles,
                use_chinese=self.use_chinese,
                title="交易信号可视化"
                if self.use_chinese else "Trading Signals Visualization")

            # 将图表转换为HTML
            signals_html = pio.to_html(signals_fig,
                                       include_plotlyjs=False,
                                       full_html=False)
            html += signals_html

        # 3. 权益曲线和回撤分析
        if 'equity_curve' in self.results:
            equity_df = pd.DataFrame({
                'date':
                self.results['equity_curve']['dates'],
                'equity':
                self.results['equity_curve']['values']
            })

            drawdown_fig = create_drawdown_chart(
                equity_df,
                date_col='date',
                value_col='equity',
                use_chinese=self.use_chinese,
                title="权益曲线与回撤分析"
                if self.use_chinese else "Equity Curve and Drawdown Analysis")

            # 将图表转换为HTML
            dd_html = pio.to_html(drawdown_fig,
                                  include_plotlyjs=False,
                                  full_html=False)
            html += dd_html

        # 4. 月度收益热图
        if 'monthly_returns' in self.results:
            monthly_fig = visualize_monthly_returns(
                self.results,
                use_chinese=self.use_chinese,
                title="月度收益率热图"
                if self.use_chinese else "Monthly Returns Heatmap")

            # 将图表转换为HTML
            monthly_html = pio.to_html(monthly_fig,
                                       include_plotlyjs=False,
                                       full_html=False)
            html += monthly_html

        # 5. 交易分析
        if self.trades is not None:
            trade_fig = plot_trade_analysis(
                self.trades,
                date_col=self.date_col
                if self.date_col in self.trades.columns else 'entry_date',
                use_chinese=self.use_chinese,
                title="交易分析" if self.use_chinese else "Trade Analysis")

            # 将图表转换为HTML
            trade_html = pio.to_html(trade_fig,
                                     include_plotlyjs=False,
                                     full_html=False)
            html += trade_html

        # 6. 带指标的K线图
        if indicators is not None:
            indicator_fig = plot_candlestick_plotly(
                self.df.tail(max_candles),
                date_col=self.date_col,
                indicators=indicators,
                title="价格与指标" if self.use_chinese else "Price & Indicators")

            # 将图表转换为HTML
            indicator_html = pio.to_html(indicator_fig,
                                         include_plotlyjs=False,
                                         full_html=False)
            html += indicator_html

        # 添加必要的JavaScript库
        html = """
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        """ + html

        # 显示仪表盘
        display(HTML(html))


# 便捷函数，用于直接创建仪表盘
def create_dashboard(df=None,
                     results=None,
                     trades=None,
                     date_col='trade_time',
                     dashboard_type='full',
                     indicators=None,
                     max_candles=200,
                     use_chinese=True,
                     title="回测结果仪表盘"):
    """
    创建回测结果仪表盘
    
    参数:
    df (pandas.DataFrame): 包含OHLCV数据的DataFrame
    results (dict): 回测结果字典
    trades (pandas.DataFrame): 包含交易记录的DataFrame
    date_col (str): 日期列名
    dashboard_type (str): 仪表盘类型，可选 'full', 'performance', 'signals'
    indicators (dict): 指标字典，格式为 {name: {'line': series, 'color': color}}
    max_candles (int): 最大显示的K线数量
    use_chinese (bool): 是否使用中文标签
    title (str): 仪表盘标题
    
    返回:
    None: 直接在Notebook中显示仪表盘
    """
    dashboard = BacktestDashboard(df=df,
                                  results=results,
                                  trades=trades,
                                  date_col=date_col,
                                  use_chinese=use_chinese,
                                  title=title)

    if dashboard_type.lower() == 'full':
        dashboard.create_full_dashboard(indicators=indicators,
                                        max_candles=max_candles)
    elif dashboard_type.lower() == 'performance':
        dashboard.create_performance_dashboard()
    elif dashboard_type.lower() == 'signals':
        dashboard.create_trading_signals_dashboard(indicators=indicators,
                                                   max_candles=max_candles)
    else:
        raise ValueError("不支持的仪表盘类型，可选 'full', 'performance', 'signals'")

    return dashboard
