import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import threading
import os

from live_trading.signal_generator import SignalGenerator
from live_trading.next_candle_predictor import NextCandlePredictor
from live_trading.trading_manager import LiveTradingManager
from live_trading.advanced_visualization import AdvancedVisualization
from live_trading.real_time_market_monitor import RealTimeMarketMonitor

class LiveTradingUI:
    """
    UI components for live trading functionality.
    """
    
    def __init__(self):
        """Initialize the LiveTradingUI class."""
        self.signal_generator = SignalGenerator()
        self.next_candle_predictor = NextCandlePredictor()
        self.trading_manager = LiveTradingManager()
        self.visualizer = AdvancedVisualization()
        self.market_monitor = RealTimeMarketMonitor()
        
        # Initialize session state variables if they don't exist
        if 'bot_running' not in st.session_state:
            st.session_state.bot_running = False
        if 'active_trades' not in st.session_state:
            st.session_state.active_trades = []
        if 'trading_history' not in st.session_state:
            st.session_state.trading_history = []
        if 'latest_signals' not in st.session_state:
            st.session_state.latest_signals = []
        if 'next_candle_predictions' not in st.session_state:
            st.session_state.next_candle_predictions = []
        if 'performance_metrics' not in st.session_state:
            st.session_state.performance_metrics = {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_profit_loss': 0,
                'max_drawdown': 0
            }
        if 'market_monitor_running' not in st.session_state:
            st.session_state.market_monitor_running = False
    
    def render_live_trading_tab(self):
        """Render the Live Trading tab."""
        st.header("One Beyond All - Live Trading")
        
        # Trading controls
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Start Bot", disabled=st.session_state.bot_running, use_container_width=True):
                st.session_state.bot_running = True
                # Start the market monitor if not already running
                if not st.session_state.market_monitor_running:
                    self._start_market_monitor()
        with col2:
            if st.button("Stop Bot", disabled=not st.session_state.bot_running, use_container_width=True):
                st.session_state.bot_running = False
        with col3:
            if st.button("Emergency Stop", disabled=not st.session_state.bot_running, use_container_width=True):
                st.session_state.bot_running = False
                st.session_state.active_trades = []
                # Stop the market monitor
                if st.session_state.market_monitor_running:
                    self._stop_market_monitor()
        
        # Trading status
        if st.session_state.bot_running:
            st.success("Trading bot is active and generating signals in real-time.")
        else:
            st.info("Trading bot is currently inactive. Configure settings and press 'Start Bot' to begin trading.")
        
        # Create tabs for different sections
        signal_tab, chart_tab, trades_tab, performance_tab = st.tabs([
            "Trading Signals", "Advanced Charts", "Active Trades", "Performance"
        ])
        
        with signal_tab:
            self._render_trading_signals()
        
        with chart_tab:
            self._render_advanced_charts()
        
        with trades_tab:
            self._render_active_trades()
        
        with performance_tab:
            self._render_performance_metrics()
    
    def _render_trading_signals(self):
        """Render the trading signals section."""
        st.subheader("Latest Trading Signals")
        
        # Generate new signals if bot is running
        if st.session_state.bot_running:
            # Get selected symbols from session state
            symbols = st.session_state.get('selected_symbols', ["BTC/USDT"])
            timeframe = st.session_state.get('selected_timeframe', "1h")
            
            # Generate signals for each symbol
            all_signals = []
            for symbol in symbols:
                # Check if we have market data for this symbol
                if st.session_state.market_monitor_running:
                    # Get signals from market monitor
                    signals = self.market_monitor.detect_trading_signals(symbol)
                else:
                    # Generate signals using signal generator
                    signal = self.signal_generator.generate_signal(symbol, timeframe)
                    signals = [signal] if signal else []
                
                all_signals.extend(signals)
            
            # Update session state with new signals
            if all_signals:
                # Add timestamp to signals
                for signal in all_signals:
                    if 'timestamp' not in signal or signal['timestamp'] is None:
                        signal['timestamp'] = datetime.now()
                
                # Add new signals to the beginning of the list
                st.session_state.latest_signals = all_signals + st.session_state.latest_signals
                
                # Keep only the latest 20 signals
                st.session_state.latest_signals = st.session_state.latest_signals[:20]
            
            # Generate next candle predictions
            for symbol in symbols:
                if st.session_state.market_monitor_running:
                    # Get data from market monitor
                    df = self.market_monitor.get_current_data(symbol)
                    if df is not None:
                        # Make prediction
                        prediction = self.next_candle_predictor.predict_next_candle(df)
                        if prediction:
                            prediction['symbol'] = symbol
                            prediction['timestamp'] = datetime.now()
                            
                            # Add to predictions
                            st.session_state.next_candle_predictions = [prediction] + st.session_state.next_candle_predictions
                            
                            # Keep only the latest 10 predictions
                            st.session_state.next_candle_predictions = st.session_state.next_candle_predictions[:10]
        
        # Display signals
        if st.session_state.latest_signals:
            # Create a DataFrame for display
            signals_data = []
            for signal in st.session_state.latest_signals:
                if signal:  # Check if signal is not None
                    signals_data.append({
                        'Symbol': signal.get('symbol', 'Unknown'),
                        'Type': signal.get('type', 'Unknown'),
                        'Entry': f"${signal.get('entry_low', 0):.6f} - ${signal.get('entry_high', 0):.6f}",
                        'Target': f"${signal.get('target_low', 0):.6f} - ${signal.get('target_high', 0):.6f}",
                        'Stop Loss': f"${signal.get('stop_loss', 0):.6f}",
                        'Ward': f"+{signal.get('ward', 0):.1f}%",
                        'Risk': f"-{signal.get('risk', 0):.1f}%",
                        'Time': signal.get('time', '00:00'),
                        'Timestamp': signal.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S')
                    })
            
            if signals_data:
                signals_df = pd.DataFrame(signals_data)
                st.dataframe(signals_df, use_container_width=True)
        else:
            st.info("No trading signals generated yet. Start the bot to begin generating signals.")
        
        # Display next candle predictions
        st.subheader("Next Candle Predictions")
        
        if st.session_state.next_candle_predictions:
            # Create a DataFrame for display
            predictions_data = []
            for prediction in st.session_state.next_candle_predictions:
                if prediction:  # Check if prediction is not None
                    predictions_data.append({
                        'Symbol': prediction.get('symbol', 'Unknown'),
                        'Next Open': f"${prediction.get('next_open', 0):.6f}",
                        'Next High': f"${prediction.get('next_high', 0):.6f}",
                        'Next Low': f"${prediction.get('next_low', 0):.6f}",
                        'Next Close': f"${prediction.get('next_close', 0):.6f}",
                        'Predicted Change': f"{prediction.get('predicted_change_pct', 0):.2f}%",
                        'Confidence': f"{prediction.get('confidence', 0):.2f}%",
                        'Timestamp': prediction.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S')
                    })
            
            if predictions_data:
                predictions_df = pd.DataFrame(predictions_data)
                st.dataframe(predictions_df, use_container_width=True)
        else:
            st.info("No next candle predictions generated yet. Start the bot to begin generating predictions.")
    
    def _render_advanced_charts(self):
        """Render the advanced charts section."""
        st.subheader("Advanced Trading Charts")
        
        # Get selected symbols from session state
        symbols = st.session_state.get('selected_symbols', ["BTC/USDT"])
        
        # Symbol selector
        selected_symbol = st.selectbox(
            "Select Symbol for Advanced Chart",
            symbols,
            key="advanced_chart_symbol"
        )
        
        # Check if market monitor is running and has data
        if st.session_state.market_monitor_running:
            # Get signals for this symbol
            signals = [s for s in st.session_state.latest_signals if s.get('symbol') == selected_symbol]
            
            # Get predictions for this symbol
            predictions = [p for p in st.session_state.next_candle_predictions if p.get('symbol') == selected_symbol]
            
            # Get advanced chart
            fig = self.market_monitor.get_advanced_chart(selected_symbol, signals, predictions)
            
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"No data available for {selected_symbol}. Start the market monitor to fetch data.")
        else:
            st.warning("Market monitor is not running. Start the bot to enable advanced charts.")
        
        # Market overview section
        st.subheader("Market Overview")
        
        # Create tabs for different market views
        heatmap_tab, correlation_tab = st.tabs(["Market Heatmap", "Correlation Matrix"])
        
        with heatmap_tab:
            if st.session_state.market_monitor_running:
                heatmap_fig = self.market_monitor.get_market_heatmap()
                if heatmap_fig is not None:
                    st.plotly_chart(heatmap_fig, use_container_width=True)
                else:
                    st.info("Market data not available yet. Please wait for data to be fetched.")
            else:
                st.info("Market monitor is not running. Start the bot to enable market heatmap.")
        
        with correlation_tab:
            if st.session_state.market_monitor_running:
                corr_fig = self.market_monitor.get_correlation_matrix()
                if corr_fig is not None:
                    st.plotly_chart(corr_fig, use_container_width=True)
                else:
                    st.info("Correlation data not available yet. Please wait for data to be fetched.")
            else:
                st.info("Market monitor is not running. Start the bot to enable correlation matrix.")
    
    def _render_active_trades(self):
        """Render the active trades section."""
        st.subheader("Active Trades")
        
        # Display active trades
        if st.session_state.active_trades:
            # Create a DataFrame for display
            trades_data = []
            for trade in st.session_state.active_trades:
                trades_data.append({
                    'Symbol': trade.get('symbol', 'Unknown'),
                    'Type': trade.get('type', 'Unknown'),
                    'Entry Price': f"${trade.get('entry_price', 0):.6f}",
                    'Current Price': f"${trade.get('current_price', 0):.6f}",
                    'Target Price': f"${trade.get('target_price', 0):.6f}",
                    'Stop Loss': f"${trade.get('stop_loss', 0):.6f}",
                    'Unrealized P/L': f"{trade.get('unrealized_pl', 0):.2f}%",
                    'Duration': trade.get('duration', '00:00:00')
                })
            
            trades_df = pd.DataFrame(trades_data)
            st.dataframe(trades_df, use_container_width=True)
            
            # Add close trade buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Close Selected Trade", disabled=not st.session_state.bot_running, use_container_width=True):
                    st.info("Trade closing functionality will be implemented in the trading_manager.py module")
            with col2:
                if st.button("Close All Trades", disabled=not st.session_state.bot_running, use_container_width=True):
                    st.session_state.active_trades = []
                    st.success("All trades closed successfully")
        else:
            st.info("No active trades. Signals will be converted to trades when the bot is running with real trading enabled.")
        
        # Trading history
        st.subheader("Trading History")
        
        if st.session_state.trading_history:
            # Create a DataFrame for display
            history_data = []
            for trade in st.session_state.trading_history:
                history_data.append({
                    'Symbol': trade.get('symbol', 'Unknown'),
                    'Type': trade.get('type', 'Unknown'),
                    'Entry Price': f"${trade.get('entry_price', 0):.6f}",
                    'Exit Price': f"${trade.get('exit_price', 0):.6f}",
                    'Profit/Loss': f"{trade.get('profit_loss', 0):.2f}%",
                    'Entry Time': trade.get('entry_time', '').strftime('%Y-%m-%d %H:%M:%S'),
                    'Exit Time': trade.get('exit_time', '').strftime('%Y-%m-%d %H:%M:%S'),
                    'Duration': trade.get('duration', '00:00:00')
                })
            
            history_df = pd.DataFrame(history_data)
            st.dataframe(history_df, use_container_width=True)
            
            # Add export button
            if st.button("Export Trading History", use_container_width=True):
                st.info("Export functionality will be implemented in the trading_manager.py module")
        else:
            st.info("No trading history available yet.")
    
    def _render_performance_metrics(self):
        """Render the performance metrics section."""
        st.subheader("Performance Metrics")
        
        # Display performance metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Trades", st.session_state.performance_metrics.get('total_trades', 0))
        with col2:
            st.metric("Win Rate", f"{st.session_state.performance_metrics.get('win_rate', 0):.2f}%")
        with col3:
            st.metric("Profit Factor", f"{st.session_state.performance_metrics.get('profit_factor', 0):.2f}")
        with col4:
            st.metric("Max Drawdown", f"{st.session_state.performance_metrics.get('max_drawdown', 0):.2f}%")
        
        # Performance chart
        if st.session_state.trading_history:
            # Get performance dashboard
            fig = self.market_monitor.get_performance_dashboard(
                st.session_state.trading_history,
                st.session_state.performance_metrics
            )
            
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No trading history available for performance analysis.")
    
    def render_settings_tab(self):
        """Render the Settings tab."""
        st.header("Bot Settings")
        
        # Create tabs for different settings
        trading_tab, signal_tab, api_tab = st.tabs([
            "Trading Parameters", "Signal Generation", "API Settings"
        ])
        
        with trading_tab:
            self._render_trading_settings()
        
        with signal_tab:
            self._render_signal_settings()
        
        with api_tab:
            self._render_api_settings()
    
    def _render_trading_settings(self):
        """Render the trading parameters settings."""
        st.subheader("Trading Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            st.number_input("Initial Capital (USDT)", min_value=10.0, value=1000.0, key="initial_capital")
            st.number_input("Position Size (%)", min_value=1.0, max_value=100.0, value=10.0, key="position_size")
            st.number_input("Take Profit (%)", min_value=0.1, value=5.0, key="take_profit")
        with col2:
            st.number_input("Max Open Positions", min_value=1, value=3, key="max_positions")
            st.number_input("Max Daily Trades", min_value=1, value=10, key="max_daily_trades")
            st.number_input("Stop Loss (%)", min_value=0.1, value=2.0, key="stop_loss")
        
        st.subheader("Risk Management")
        
        col1, col2 = st.columns(2)
        with col1:
            st.number_input("Max Daily Loss (%)", min_value=0.1, value=5.0, key="max_daily_loss")
            st.number_input("Max Drawdown (%)", min_value=1.0, value=20.0, key="max_drawdown")
        with col2:
            st.number_input("Trailing Stop (%)", min_value=0.0, value=1.0, key="trailing_stop")
            st.checkbox("Enable Trailing Stop", value=True, key="enable_trailing_stop")
        
        st.subheader("Advanced Trading Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            st.selectbox(
                "Order Type",
                ["Market", "Limit", "Stop-Limit"],
                index=0,
                key="order_type"
            )
            st.number_input("Slippage Tolerance (%)", min_value=0.0, value=0.5, key="slippage")
        with col2:
            st.checkbox("Enable Pyramiding", value=False, key="enable_pyramiding")
            st.number_input("Max Pyramid Levels", min_value=1, value=3, disabled=not st.session_state.get("enable_pyramiding", False), key="pyramid_levels")
    
    def _render_signal_settings(self):
        """Render the signal generation settings."""
        st.subheader("Signal Generation Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            st.slider("Signal Confidence Threshold (%)", min_value=50, max_value=100, value=75, key="signal_confidence")
            st.multiselect(
                "Technical Indicators for Signals",
                ["RSI", "MACD", "Bollinger Bands", "Moving Averages", "Volume", "Stochastic", "ATR"],
                default=["RSI", "MACD", "Bollinger Bands", "Moving Averages"],
                key="signal_indicators"
            )
        with col2:
            st.selectbox(
                "Market Condition Filter",
                ["All Conditions", "Trending Only", "Ranging Only", "Volatile Only"],
                index=0,
                key="market_filter"
            )
            st.checkbox("Use Machine Learning for Signal Confirmation", value=True, key="use_ml")
        
        st.subheader("Next Candle Prediction Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            st.slider("Prediction Confidence Threshold (%)", min_value=50, max_value=100, value=70, key="prediction_confidence")
            st.number_input("Historical Data Window (candles)", min_value=20, value=100, key="history_window")
        with col2:
            st.selectbox(
                "Prediction Model",
                ["Random Forest", "LSTM", "XGBoost", "Ensemble"],
                index=0,
                key="prediction_model"
            )
            st.checkbox("Auto-Retrain Model", value=True, key="auto_retrain")
        
        st.subheader("One Beyond All Strategy Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            st.slider("Strategy Aggressiveness", min_value=1, max_value=10, value=5, key="strategy_aggressiveness")
            st.multiselect(
                "Pattern Recognition",
                ["Double Top/Bottom", "Head & Shoulders", "Wedges", "Triangles", "Flags", "Channels"],
                default=["Double Top/Bottom", "Head & Shoulders", "Wedges"],
                key="pattern_recognition"
            )
        with col2:
            st.checkbox("Enable Advanced Support/Resistance", value=True, key="enable_sr")
            st.checkbox("Enable Sentiment Analysis", value=True, key="enable_sentiment")
    
    def _render_api_settings(self):
        """Render the API settings."""
        st.subheader("Exchange API Settings")
        
        # Exchange selection
        exchange = st.selectbox(
            "Select Exchange",
            ["Binance", "Coinbase", "Kraken", "Kucoin"],
            key="api_exchange"
        )
        
        # API credentials
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("API Key", type="password", key="api_key")
        with col2:
            st.text_input("API Secret", type="password", key="api_secret")
        
        # Test mode
        st.checkbox("Enable Test Mode (Paper Trading)", value=True, key="test_mode")
        
        # Save settings
        if st.button("Save API Settings", use_container_width=True):
            st.success("API settings saved successfully")
            
            # Initialize market monitor with selected exchange
            if st.session_state.market_monitor_running:
                self._stop_market_monitor()
            
            self._start_market_monitor()
    
    def _start_market_monitor(self):
        """Start the market monitor."""
        try:
            # Get selected symbols and timeframe
            symbols = st.session_state.get('selected_symbols', ["BTC/USDT"])
            timeframe = st.session_state.get('selected_timeframe', "1h")
            
            # Initialize exchange
            exchange_id = st.session_state.get('api_exchange', 'binance').lower()
            test_mode = st.session_state.get('test_mode', True)
            
            # Initialize and start market monitor
            self.market_monitor.initialize_exchange(exchange_id, test_mode)
            self.market_monitor.start_monitoring(symbols, timeframe)
            
            # Update session state
            st.session_state.market_monitor_running = True
            
            return True
        except Exception as e:
            st.error(f"Error starting market monitor: {str(e)}")
            return False
    
    def _stop_market_monitor(self):
        """Stop the market monitor."""
        try:
            # Stop market monitor
            self.market_monitor.stop_monitoring()
            
            # Update session state
            st.session_state.market_monitor_running = False
            
            return True
        except Exception as e:
            st.error(f"Error stopping market monitor: {str(e)}")
            return False
