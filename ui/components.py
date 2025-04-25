import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

class UIComponents:
    """
    A class for rendering UI components for the Streamlit crypto trading bot.
    """
    
    def __init__(self):
        """Initialize the UIComponents class."""
        pass
    
    def render_header(self):
        """Render the application header."""
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.image("https://img.icons8.com/fluency/96/cryptocurrency.png", width=80)
        
        with col2:
            st.title("Crypto Trading Bot")
            st.markdown("A powerful tool for cryptocurrency technical analysis and automated trading")
    
    def render_market_summary(self, market_data):
        """
        Render market summary cards.
        
        Args:
            market_data (pd.DataFrame): DataFrame with market summary data
        """
        if market_data is None or len(market_data) == 0:
            st.warning("No market data available.")
            return
        
        # Create columns for each symbol
        cols = st.columns(len(market_data))
        
        for i, (_, row) in enumerate(market_data.iterrows()):
            with cols[i]:
                symbol = row['symbol'].replace('/USDT', '')
                price = row['last_price']
                change = row['daily_change']
                
                st.metric(
                    label=symbol,
                    value=f"${self.format_number(price)}",
                    delta=f"{change:.2f}%" if change is not None else None
                )
    
    def render_price_chart(self, df, symbol, timeframe):
        """
        Render price chart with candlesticks.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            symbol (str): Trading pair symbol
            timeframe (str): Timeframe for the data
        """
        if df is None or len(df) == 0:
            st.warning("No price data available.")
            return
        
        fig = go.Figure()
        
        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        ))
        
        # Update layout
        fig.update_layout(
            title=f"{symbol} - {timeframe} Chart",
            xaxis_title="Date",
            yaxis_title="Price (USDT)",
            xaxis_rangeslider_visible=False,
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_indicator_charts(self, df, indicators):
        """
        Render technical indicator charts.
        
        Args:
            df (pd.DataFrame): DataFrame with technical indicators
            indicators (list): List of indicators to display
        """
        if df is None or len(df) == 0:
            st.warning("No indicator data available.")
            return
        
        for indicator in indicators:
            if indicator == "RSI":
                self.render_rsi_chart(df)
            elif indicator == "MACD":
                self.render_macd_chart(df)
            elif indicator == "Bollinger Bands":
                self.render_bollinger_bands_chart(df)
            elif indicator == "Moving Averages":
                self.render_moving_averages_chart(df)
            elif indicator == "Volume":
                self.render_volume_chart(df)
            elif indicator == "Stochastic":
                self.render_stochastic_chart(df)
            elif indicator == "ATR":
                self.render_atr_chart(df)
    
    def render_rsi_chart(self, df):
        """
        Render RSI chart.
        
        Args:
            df (pd.DataFrame): DataFrame with RSI data
        """
        if 'rsi_14' not in df.columns:
            st.warning("RSI data not available.")
            return
        
        fig = go.Figure()
        
        # Add RSI line
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['rsi_14'],
            name='RSI (14)',
            line=dict(color='purple')
        ))
        
        # Add overbought/oversold lines
        fig.add_shape(
            type='line',
            x0=df.index[0],
            y0=70,
            x1=df.index[-1],
            y1=70,
            line=dict(color='red', dash='dash')
        )
        
        fig.add_shape(
            type='line',
            x0=df.index[0],
            y0=30,
            x1=df.index[-1],
            y1=30,
            line=dict(color='green', dash='dash')
        )
        
        # Update layout
        fig.update_layout(
            title='Relative Strength Index (RSI)',
            xaxis_title='Date',
            yaxis_title='RSI',
            yaxis=dict(range=[0, 100]),
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_macd_chart(self, df):
        """
        Render MACD chart.
        
        Args:
            df (pd.DataFrame): DataFrame with MACD data
        """
        if 'macd_line' not in df.columns or 'macd_signal' not in df.columns:
            st.warning("MACD data not available.")
            return
        
        fig = go.Figure()
        
        # Add MACD line and signal line
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['macd_line'],
            name='MACD Line',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['macd_signal'],
            name='Signal Line',
            line=dict(color='red')
        ))
        
        # Add histogram
        colors = ['green' if val >= 0 else 'red' for val in df['macd_histogram']]
        fig.add_trace(go.Bar(
            x=df.index,
            y=df['macd_histogram'],
            name='Histogram',
            marker_color=colors
        ))
        
        # Update layout
        fig.update_layout(
            title='Moving Average Convergence Divergence (MACD)',
            xaxis_title='Date',
            yaxis_title='MACD',
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_bollinger_bands_chart(self, df):
        """
        Render Bollinger Bands chart.
        
        Args:
            df (pd.DataFrame): DataFrame with Bollinger Bands data
        """
        if 'bollinger_hband' not in df.columns or 'bollinger_lband' not in df.columns:
            st.warning("Bollinger Bands data not available.")
            return
        
        fig = go.Figure()
        
        # Add price
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['close'],
            name='Close Price',
            line=dict(color='black')
        ))
        
        # Add Bollinger Bands
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['bollinger_hband'],
            name='Upper Band',
            line=dict(color='rgba(250, 0, 0, 0.5)')
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['bollinger_mavg'],
            name='Middle Band',
            line=dict(color='rgba(0, 0, 250, 0.5)')
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['bollinger_lband'],
            name='Lower Band',
            line=dict(color='rgba(0, 250, 0, 0.5)')
        ))
        
        # Update layout
        fig.update_layout(
            title='Bollinger Bands',
            xaxis_title='Date',
            yaxis_title='Price',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_moving_averages_chart(self, df):
        """
        Render Moving Averages chart.
        
        Args:
            df (pd.DataFrame): DataFrame with Moving Averages data
        """
        if 'sma_20' not in df.columns or 'sma_50' not in df.columns or 'sma_200' not in df.columns:
            st.warning("Moving Averages data not available.")
            return
        
        fig = go.Figure()
        
        # Add price
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['close'],
            name='Close Price',
            line=dict(color='black')
        ))
        
        # Add Moving Averages
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['sma_20'],
            name='SMA 20',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['sma_50'],
            name='SMA 50',
            line=dict(color='orange')
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['sma_200'],
            name='SMA 200',
            line=dict(color='red')
        ))
        
        # Update layout
        fig.update_layout(
            title='Moving Averages',
            xaxis_title='Date',
            yaxis_title='Price',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_volume_chart(self, df):
        """
        Render Volume chart.
        
        Args:
            df (pd.DataFrame): DataFrame with volume data
        """
        if 'volume' not in df.columns:
            st.warning("Volume data not available.")
            return
        
        fig = go.Figure()
        
        # Add volume bars
        colors = ['green' if df['close'][i] >= df['open'][i] else 'red' for i in range(len(df))]
        fig.add_trace(go.Bar(
            x=df.index,
            y=df['volume'],
            name='Volume',
            marker_color=colors
        ))
        
        # Update layout
        fig.update_layout(
            title='Volume',
            xaxis_title='Date',
            yaxis_title='Volume',
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_stochastic_chart(self, df):
        """
        Render Stochastic Oscillator chart.
        
        Args:
            df (pd.DataFrame): DataFrame with Stochastic data
        """
        if 'stoch_k' not in df.columns or 'stoch_d' not in df.columns:
            st.warning("Stochastic Oscillator data not available.")
            return
        
        fig = go.Figure()
        
        # Add Stochastic lines
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['stoch_k'],
            name='%K Line',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['stoch_d'],
            name='%D Line',
            line=dict(color='red')
        ))
        
        # Add overbought/oversold lines
        fig.add_shape(
            type='line',
            x0=df.index[0],
            y0=80,
            x1=df.index[-1],
            y1=80,
            line=dict(color='red', dash='dash')
        )
        
        fig.add_shape(
            type='line',
            x0=df.index[0],
            y0=20,
            x1=df.index[-1],
            y1=20,
            line=dict(color='green', dash='dash')
        )
        
        # Update layout
        fig.update_layout(
            title='Stochastic Oscillator',
            xaxis_title='Date',
            yaxis_title='Stochastic',
            yaxis=dict(range=[0, 100]),
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_atr_chart(self, df):
        """
        Render Average True Range chart.
        
        Args:
            df (pd.DataFrame): DataFrame with ATR data
        """
        if 'atr_14' not in df.columns:
            st.warning("ATR data not available.")
            return
        
        fig = go.Figure()
        
        # Add ATR line
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['atr_14'],
            name='ATR (14)',
            line=dict(color='purple')
        ))
        
        # Update layout
        fig.update_layout(
            title='Average True Range (ATR)',
            xaxis_title='Date',
            yaxis_title='ATR',
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_backtest_results(self, backtest_df, metrics):
        """
        Render backtest results.
        
        Args:
            backtest_df (pd.DataFrame): DataFrame with backtest results
            metrics (dict): Dictionary with performance metrics
        """
        if backtest_df is None or len(backtest_df) == 0 or not metrics:
            st.warning("No backtest results available.")
            return
        
        # Display performance metrics
        st.subheader("Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Return", f"{metrics['total_return']}%")
            st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']}")
        with col2:
            st.metric("Annual Return", f"{metrics['annual_return']}%")
            st.metric("Win Rate", f"{metrics['win_rate']}%")
        with col3:
            st.metric("Max Drawdown", f"{metrics['max_drawdown']}%")
            st.metric("Profit Factor", f"{metrics['profit_factor']}")
        with col4:
            st.metric("Volatility", f"{metrics['volatility']}%")
            st.metric("Total Trades", f"{metrics['total_trades']}")
        
        # Display equity curve
        st.subheader("Equity Curve")
        
        fig = go.Figure()
        
        # Add equity curve
        fig.add_trace(go.Scatter(
            x=backtest_df.index,
            y=backtest_df['capital'],
            name='Portfolio Value',
            line=dict(color='blue')
        ))
        
        # Update layout
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Portfolio Value',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display drawdown chart
        st.subheader("Drawdown")
        
        fig = go.Figure()
        
        # Add drawdown curve
        fig.add_trace(go.Scatter(
            x=backtest_df.index,
            y=backtest_df['drawdown'] * 100,
            name='Drawdown',
            line=dict(color='red')
        ))
        
        # Update layout
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_trade_signals(self, trades_df):
        """
        Render trade signals table.
        
        Args:
            trades_df (pd.DataFrame): DataFrame with trade signals
        """
        if trades_df is None or len(trades_df) == 0:
            st.warning("No trade signals available.")
            return
        
        st.subheader("Trade Signals")
        st.dataframe(trades_df, use_container_width=True)
    
    def render_strategy_optimizer(self, param_grid, best_params, best_metrics):
        """
        Render strategy optimizer results.
        
        Args:
            param_grid (dict): Dictionary with parameter grid
            best_params (dict): Dictionary with best parameters
            best_metrics (dict): Dictionary with best metrics
        """
        st.subheader("Strategy Optimizer")
        
        if not best_params or not best_metrics:
            st.info("Run the optimizer to find the best parameters for your strategy.")
            return
        
        # Display best parameters
        st.write("Best Parameters:")
        for param, value in best_params.items():
            st.write(f"- {param}: {value}")
        
        # Display best metrics
        st.write("Performance with Best Parameters:")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Return", f"{best_metrics['total_return']}%")
        with col2:
            st.metric("Sharpe Ratio", f"{best_metrics['sharpe_ratio']}")
        with col3:
            st.metric("Max Drawdown", f"{best_metrics['max_drawdown']}%")
    
    def render_live_trading_status(self, status, last_trade=None, open_positions=None):
        """
        Render live trading status.
        
        Args:
            status (str): Trading bot status
            last_trade (dict, optional): Last trade information
            open_positions (list, optional): List of open positions
        """
        st.subheader("Trading Bot Status")
        
        # Display status with appropriate color
        if status == "Running":
            st.success("Bot Status: Running")
        elif status == "Stopped":
            st.error("Bot Status: Stopped")
        elif status == "Paused":
            st.warning("Bot Status: Paused")
        else:
            st.info(f"Bot Status: {status}")
        
        # Display last trade if available
        if last_trade:
            st.write("Last Trade:")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.write("Type:", last_trade.get("type", "N/A"))
            with col2:
                st.write("Symbol:", last_trade.get("symbol", "N/A"))
            with col3:
                st.write("Price:", last_trade.get("price", "N/A"))
            with col4:
                st.write("Time:", last_trade.get("time", "N/A"))
        
        # Display open positions if available
        if open_positions and len(open_positions) > 0:
            st.write("Open Positions:")
            positions_data = []
            for pos in open_positions:
                positions_data.append({
                    "Symbol": pos.get("symbol", "N/A"),
                    "Type": pos.get("type", "N/A"),
                    "Entry Price": pos.get("entry_price", "N/A"),
                    "Current Price": pos.get("current_price", "N/A"),
                    "Profit/Loss": pos.get("pnl", "N/A"),
                    "P/L %": pos.get("pnl_pct", "N/A")
                })
            
            st.dataframe(pd.DataFrame(positions_data), use_container_width=True)
        else:
            st.info("No open positions.")
    
    @staticmethod
    def format_number(num):
        """
        Format number for display.
        
        Args:
            num (float): Number to format
            
        Returns:
            str: Formatted number
        """
        if num is None:
            return "N/A"
        
        if abs(num) >= 1000000000:
            return f"{num / 1000000000:.2f}B"
        elif abs(num) >= 1000000:
            return f"{num / 1000000:.2f}M"
        elif abs(num) >= 1000:
            return f"{num / 1000:.2f}K"
        elif abs(num) >= 1:
            return f"{num:.2f}"
        else:
            # For small numbers like crypto prices
            if abs(num) < 0.0001:
                return f"{num:.8f}"
            else:
                return f"{num:.4f}"
