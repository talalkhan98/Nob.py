import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

# Import modules
from data.data_fetcher import CryptoDataFetcher
from analysis.indicators import TechnicalIndicators
from strategies.trading_strategies import TradingStrategies
from ui.components import UIComponents
from ui.live_trading_ui import LiveTradingUI
from utils.helpers import format_number

# Create logs directory if it doesn't exist
logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(logs_dir, exist_ok=True)

# Create models directory if it doesn't exist
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
os.makedirs(models_dir, exist_ok=True)

# Page configuration
st.set_page_config(
    page_title="Crypto Trading Bot - One Beyond All",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components
data_fetcher = CryptoDataFetcher()
indicators = TechnicalIndicators()
strategies = TradingStrategies()
ui = UIComponents()
live_trading_ui = LiveTradingUI()

# App title
ui.render_header()

# Sidebar
with st.sidebar:
    st.title("Settings")
    
    # Exchange selection
    exchange = st.selectbox(
        "Select Exchange",
        ["Binance", "Coinbase", "Kraken", "Kucoin"],
        key="selected_exchange"
    )
    
    # Symbol selection
    symbols = st.multiselect(
        "Select Trading Pairs",
        ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT"],
        default=["BTC/USDT"],
        key="selected_symbols"
    )
    
    # Timeframe selection
    timeframe = st.selectbox(
        "Select Timeframe",
        ["1m", "5m", "15m", "1h", "4h", "1d", "1w"],
        index=3,  # Default to 1h
        key="selected_timeframe"
    )
    
    # Period selection
    period = st.slider(
        "Select Period (days)",
        min_value=1,
        max_value=365,
        value=30,
        key="selected_period"
    )
    
    # Strategy selection
    strategy = st.selectbox(
        "Select Trading Strategy",
        ["RSI Strategy", "MACD Strategy", "Bollinger Bands Strategy", "Moving Average Crossover", "Custom Strategy", "One Beyond All Strategy"],
        index=5,  # Default to One Beyond All Strategy
        key="selected_strategy"
    )
    
    # Indicator selection
    indicators_list = st.multiselect(
        "Select Technical Indicators",
        ["RSI", "MACD", "Bollinger Bands", "Moving Averages", "Volume", "Stochastic", "ATR"],
        default=["RSI", "MACD", "Bollinger Bands", "Moving Averages"],
        key="selected_indicators"
    )
    
    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        fetch_button = st.button("Fetch Data", use_container_width=True)
    with col2:
        backtest_button = st.button("Run Backtest", use_container_width=True)

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Backtesting", "Live Trading", "Settings"])

with tab1:
    st.header("Market Dashboard")
    
    # Placeholder for actual implementation
    if fetch_button:
        with st.spinner("Fetching data..."):
            # This would be replaced with actual data fetching
            st.info("Fetching data for " + ", ".join(symbols))
            
            # Sample data for demonstration
            dates = pd.date_range(end=datetime.now(), periods=100).tolist()
            prices = np.random.normal(loc=50000, scale=1000, size=100).cumsum()
            volumes = np.random.normal(loc=1000, scale=100, size=100).cumsum()
            
            # Create sample dataframe
            df = pd.DataFrame({
                'date': dates,
                'price': prices,
                'volume': volumes
            })
            
            # Display sample chart
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df['date'],
                open=prices,
                high=prices * 1.01,
                low=prices * 0.99,
                close=prices,
                name='Price'
            ))
            fig.update_layout(
                title=f"{symbols[0]} Price Chart",
                xaxis_title="Date",
                yaxis_title="Price (USDT)",
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display sample metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Price", f"${format_number(prices[-1])}", f"{format_number(prices[-1] - prices[-2], prefix=True)}%")
            with col2:
                st.metric("24h Volume", f"${format_number(volumes[-1])}")
            with col3:
                st.metric("24h High", f"${format_number(prices[-1] * 1.01)}")
            with col4:
                st.metric("24h Low", f"${format_number(prices[-1] * 0.99)}")

with tab2:
    st.header("Backtesting")
    
    if backtest_button:
        with st.spinner("Running backtest..."):
            st.info(f"Backtesting {strategy} strategy on {', '.join(symbols)}")
            
            # Sample backtest results
            st.subheader("Backtest Results")
            
            # Sample metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Return", "34.2%")
            with col2:
                st.metric("Win Rate", "68%")
            with col3:
                st.metric("Profit Factor", "2.1")
            with col4:
                st.metric("Max Drawdown", "-12.3%")
            
            # Sample trades table
            st.subheader("Sample Trades")
            sample_trades = pd.DataFrame({
                'Date': pd.date_range(end=datetime.now(), periods=5).tolist(),
                'Type': ['BUY', 'SELL', 'BUY', 'SELL', 'BUY'],
                'Price': [48500, 51200, 49800, 52300, 50100],
                'Quantity': [0.1, 0.1, 0.15, 0.15, 0.2],
                'Profit/Loss': ['', '+$270', '', '+$375', '']
            })
            st.dataframe(sample_trades, use_container_width=True)

with tab3:
    # Render live trading UI
    live_trading_ui.render_live_trading_tab()

with tab4:
    # Render settings UI
    live_trading_ui.render_settings_tab()

# Footer
st.markdown("---")
st.markdown("One Beyond All Crypto Trading Bot | Created with Streamlit | Real-time signals and next candle prediction")
