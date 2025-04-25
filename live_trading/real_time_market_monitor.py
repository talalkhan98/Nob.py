import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import ccxt
import time
import threading
import os
import json
from live_trading.advanced_visualization import AdvancedVisualization

class RealTimeMarketMonitor:
    """
    A class for real-time market monitoring and automated chart drawing
    for the One Beyond All Crypto Trading Bot.
    """
    
    def __init__(self):
        """Initialize the RealTimeMarketMonitor class."""
        self.exchange = None
        self.symbols = []
        self.timeframes = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '30m': 1800,
            '1h': 3600,
            '4h': 14400,
            '1d': 86400
        }
        self.current_data = {}
        self.market_data = {}
        self.running = False
        self.thread = None
        self.last_update = {}
        self.visualizer = AdvancedVisualization()
        
        # Create logs directory if it doesn't exist
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        self.log_file = os.path.join(logs_dir, 'market_monitor.log')
    
    def initialize_exchange(self, exchange_id='binance', test_mode=True):
        """
        Initialize the exchange connection.
        
        Args:
            exchange_id (str): Exchange ID (e.g., 'binance', 'coinbase')
            test_mode (bool): Whether to use test mode
        """
        try:
            # Initialize exchange
            exchange_class = getattr(ccxt, exchange_id)
            self.exchange = exchange_class({
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future' if not test_mode else 'spot'
                }
            })
            
            # Load API keys if available and not in test mode
            if not test_mode:
                try:
                    # Try to load from Streamlit secrets
                    api_key = st.secrets["exchanges"][f"{exchange_id}_api_key"]
                    api_secret = st.secrets["exchanges"][f"{exchange_id}_api_secret"]
                    
                    self.exchange.apiKey = api_key
                    self.exchange.secret = api_secret
                    self.log(f"Loaded API keys for {exchange_id} from Streamlit secrets")
                except Exception as e:
                    self.log(f"Could not load API keys from Streamlit secrets: {str(e)}")
                    self.log("Running in read-only mode")
            
            self.log(f"Initialized {exchange_id} exchange connection")
            return True
        except Exception as e:
            self.log(f"Error initializing exchange: {str(e)}")
            return False
    
    def log(self, message):
        """
        Log a message to the log file.
        
        Args:
            message (str): Message to log
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        
        # Print to console
        print(log_message)
        
        # Write to log file
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')
    
    def start_monitoring(self, symbols, timeframe='1h'):
        """
        Start real-time market monitoring.
        
        Args:
            symbols (list): List of symbols to monitor
            timeframe (str): Timeframe to monitor
        """
        if self.running:
            self.log("Market monitoring is already running")
            return
        
        if not self.exchange:
            success = self.initialize_exchange()
            if not success:
                self.log("Failed to initialize exchange")
                return
        
        self.symbols = symbols
        self.timeframe = timeframe
        self.running = True
        
        # Initialize data structures
        for symbol in symbols:
            self.current_data[symbol] = None
            self.last_update[symbol] = 0
        
        # Start monitoring thread
        self.thread = threading.Thread(target=self._monitoring_loop)
        self.thread.daemon = True
        self.thread.start()
        
        self.log(f"Started market monitoring for {', '.join(symbols)} on {timeframe} timeframe")
    
    def stop_monitoring(self):
        """Stop real-time market monitoring."""
        if not self.running:
            self.log("Market monitoring is not running")
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        
        self.log("Stopped market monitoring")
    
    def _monitoring_loop(self):
        """Internal monitoring loop."""
        while self.running:
            try:
                # Update market data for all symbols
                for symbol in self.symbols:
                    self._update_market_data(symbol, self.timeframe)
                
                # Update global market data
                self._update_global_market_data()
                
                # Sleep until next update
                sleep_time = self._calculate_sleep_time()
                time.sleep(sleep_time)
            except Exception as e:
                self.log(f"Error in monitoring loop: {str(e)}")
                time.sleep(10)  # Sleep on error to avoid rapid retries
    
    def _update_market_data(self, symbol, timeframe):
        """
        Update market data for a specific symbol and timeframe.
        
        Args:
            symbol (str): Symbol to update
            timeframe (str): Timeframe to update
        """
        try:
            # Check if we need to update
            current_time = time.time()
            update_interval = self.timeframes[timeframe]
            
            if current_time - self.last_update.get(symbol, 0) < update_interval / 2:
                return  # Skip update if not enough time has passed
            
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=100)
            
            if not ohlcv or len(ohlcv) < 10:
                self.log(f"Not enough data for {symbol} on {timeframe} timeframe")
                return
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Calculate indicators
            self._calculate_indicators(df)
            
            # Store updated data
            self.current_data[symbol] = df
            self.last_update[symbol] = current_time
            
            self.log(f"Updated market data for {symbol} on {timeframe} timeframe")
        except Exception as e:
            self.log(f"Error updating market data for {symbol}: {str(e)}")
    
    def _update_global_market_data(self):
        """Update global market data."""
        try:
            # Fetch market data for top cryptocurrencies
            markets = self.exchange.fetch_tickers()
            
            # Convert to DataFrame
            data = []
            for symbol, ticker in markets.items():
                # Filter for USDT pairs
                if symbol.endswith('/USDT'):
                    data.append({
                        'symbol': symbol,
                        'last_price': ticker.get('last', 0),
                        'daily_change': ticker.get('percentage', 0),
                        'volume_24h': ticker.get('quoteVolume', 0)
                    })
            
            # Create DataFrame
            df = pd.DataFrame(data)
            
            # Sort by volume
            df = df.sort_values('volume_24h', ascending=False)
            
            # Store top 50 by volume
            self.market_data = df.head(50)
            
            self.log("Updated global market data")
        except Exception as e:
            self.log(f"Error updating global market data: {str(e)}")
    
    def _calculate_indicators(self, df):
        """
        Calculate technical indicators for a DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
        """
        # Simple Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['macd_line'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd_line'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd_line'] - df['macd_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bollinger_mid'] = df['close'].rolling(window=20).mean()
        df['bollinger_std'] = df['close'].rolling(window=20).std()
        df['bollinger_hband'] = df['bollinger_mid'] + 2 * df['bollinger_std']
        df['bollinger_lband'] = df['bollinger_mid'] - 2 * df['bollinger_std']
        
        # Average True Range (ATR)
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = true_range.rolling(window=14).mean()
    
    def _calculate_sleep_time(self):
        """
        Calculate sleep time until next update.
        
        Returns:
            float: Sleep time in seconds
        """
        # Default sleep time
        default_sleep = 5
        
        # If no timeframe specified, use default
        if not hasattr(self, 'timeframe') or self.timeframe not in self.timeframes:
            return default_sleep
        
        # Get timeframe interval
        interval = self.timeframes[self.timeframe]
        
        # Calculate time until next candle
        current_time = time.time()
        next_candle = (current_time // interval + 1) * interval
        sleep_time = next_candle - current_time
        
        # Ensure sleep time is reasonable
        if sleep_time < 1:
            sleep_time = 1
        elif sleep_time > interval / 2:
            sleep_time = interval / 4  # Check more frequently near candle close
        
        return sleep_time
    
    def get_current_data(self, symbol):
        """
        Get current market data for a symbol.
        
        Args:
            symbol (str): Symbol to get data for
            
        Returns:
            pd.DataFrame: DataFrame with market data
        """
        return self.current_data.get(symbol)
    
    def get_market_data(self):
        """
        Get global market data.
        
        Returns:
            pd.DataFrame: DataFrame with market data
        """
        return self.market_data
    
    def get_advanced_chart(self, symbol, signals=None, predictions=None):
        """
        Get advanced chart for a symbol.
        
        Args:
            symbol (str): Symbol to get chart for
            signals (list): List of trading signals
            predictions (list): List of price predictions
            
        Returns:
            go.Figure: Plotly figure with advanced chart
        """
        df = self.get_current_data(symbol)
        if df is None:
            return None
        
        return self.visualizer.create_advanced_chart(df, signals, predictions, self.timeframe)
    
    def get_market_heatmap(self):
        """
        Get market heatmap.
        
        Returns:
            go.Figure: Plotly figure with market heatmap
        """
        market_data = self.get_market_data()
        if market_data is None or market_data.empty:
            return None
        
        return self.visualizer.create_market_heatmap(market_data)
    
    def get_correlation_matrix(self):
        """
        Get correlation matrix.
        
        Returns:
            go.Figure: Plotly figure with correlation matrix
        """
        if not self.current_data:
            return None
        
        return self.visualizer.create_correlation_matrix(self.current_data)
    
    def get_performance_dashboard(self, trading_history, metrics):
        """
        Get performance dashboard.
        
        Args:
            trading_history (list): List of trade dictionaries
            metrics (dict): Dictionary with performance metrics
            
        Returns:
            go.Figure: Plotly figure with performance dashboard
        """
        return self.visualizer.create_performance_dashboard(trading_history, metrics)
    
    def detect_trading_signals(self, symbol):
        """
        Detect trading signals for a symbol.
        
        Args:
            symbol (str): Symbol to detect signals for
            
        Returns:
            list: List of trading signals
        """
        df = self.get_current_data(symbol)
        if df is None:
            return []
        
        signals = []
        
        # Check for bullish signals
        if self._is_bullish_macd_crossover(df):
            signals.append(self._create_buy_signal(df, symbol, 'MACD Crossover'))
        
        if self._is_bullish_rsi_oversold(df):
            signals.append(self._create_buy_signal(df, symbol, 'RSI Oversold'))
        
        if self._is_bullish_bollinger_bounce(df):
            signals.append(self._create_buy_signal(df, symbol, 'Bollinger Bounce'))
        
        # Check for bearish signals
        if self._is_bearish_macd_crossover(df):
            signals.append(self._create_sell_signal(df, symbol, 'MACD Crossover'))
        
        if self._is_bearish_rsi_overbought(df):
            signals.append(self._create_sell_signal(df, symbol, 'RSI Overbought'))
        
        if self._is_bearish_bollinger_squeeze(df):
            signals.append(self._create_sell_signal(df, symbol, 'Bollinger Squeeze'))
        
        return signals
    
    def _is_bullish_macd_crossover(self, df):
        """
        Check for bullish MACD crossover.
        
        Args:
            df (pd.DataFrame): DataFrame with market data
            
        Returns:
            bool: True if bullish MACD crossover detected
        """
        if len(df) < 3:
            return False
        
        # Check if MACD line crosses above signal line
        return (df['macd_line'].iloc[-2] <= df['macd_signal'].iloc[-2] and
                df['macd_line'].iloc[-1] > df['macd_signal'].iloc[-1])
    
    def _is_bearish_macd_crossover(self, df):
        """
        Check for bearish MACD crossover.
        
        Args:
            df (pd.DataFrame): DataFrame with market data
            
        Returns:
            bool: True if bearish MACD crossover detected
        """
        if len(df) < 3:
            return False
        
        # Check if MACD line crosses below signal line
        return (df['macd_line'].iloc[-2] >= df['macd_signal'].iloc[-2] and
                df['macd_line'].iloc[-1] < df['macd_signal'].iloc[-1])
    
    def _is_bullish_rsi_oversold(self, df):
        """
        Check for bullish RSI oversold condition.
        
        Args:
            df (pd.DataFrame): DataFrame with market data
            
        Returns:
            bool: True if bullish RSI oversold condition detected
        """
        if len(df) < 3:
            return False
        
        # Check if RSI crosses above 30 from below
        return (df['rsi_14'].iloc[-2] <= 30 and
                df['rsi_14'].iloc[-1] > 30)
    
    def _is_bearish_rsi_overbought(self, df):
        """
        Check for bearish RSI overbought condition.
        
        Args:
            df (pd.DataFrame): DataFrame with market data
            
        Returns:
            bool: True if bearish RSI overbought condition detected
        """
        if len(df) < 3:
            return False
        
        # Check if RSI crosses below 70 from above
        return (df['rsi_14'].iloc[-2] >= 70 and
                df['rsi_14'].iloc[-1] < 70)
    
    def _is_bullish_bollinger_bounce(self, df):
        """
        Check for bullish Bollinger Band bounce.
        
        Args:
            df (pd.DataFrame): DataFrame with market data
            
        Returns:
            bool: True if bullish Bollinger Band bounce detected
        """
        if len(df) < 3:
            return False
        
        # Check if price bounces off lower Bollinger Band
        return (df['low'].iloc[-2] <= df['bollinger_lband'].iloc[-2] and
                df['close'].iloc[-1] > df['bollinger_lband'].iloc[-1] and
                df['close'].iloc[-1] > df['close'].iloc[-2])
    
    def _is_bearish_bollinger_squeeze(self, df):
        """
        Check for bearish Bollinger Band squeeze.
        
        Args:
            df (pd.DataFrame): DataFrame with market data
            
        Returns:
            bool: True if bearish Bollinger Band squeeze detected
        """
        if len(df) < 20:
            return False
        
        # Calculate Bollinger Band width
        df['bb_width'] = (df['bollinger_hband'] - df['bollinger_lband']) / df['bollinger_mid']
        
        # Check for Bollinger Band squeeze (narrowing bands) followed by expansion
        bb_width_min = df['bb_width'].iloc[-20:-1].min()
        current_width = df['bb_width'].iloc[-1]
        
        return (current_width > bb_width_min * 1.5 and  # Width expanding
                df['close'].iloc[-1] > df['bollinger_mid'].iloc[-1] and  # Price above middle band
                df['close'].iloc[-1] > df['close'].iloc[-2])  # Price rising
    
    def _create_buy_signal(self, df, symbol, signal_type):
        """
        Create a buy signal.
        
        Args:
            df (pd.DataFrame): DataFrame with market data
            symbol (str): Symbol for the signal
            signal_type (str): Type of signal
            
        Returns:
            dict: Buy signal
        """
        current_price = df['close'].iloc[-1]
        atr = df['atr_14'].iloc[-1] if 'atr_14' in df.columns else current_price * 0.01
        
        # Calculate entry range (0.5% below and above current price)
        entry_low = round(current_price * 0.995, 6)
        entry_high = round(current_price * 1.005, 6)
        
        # Calculate target (2-3 ATR above entry)
        target_low = round(current_price + 2 * atr, 6)
        target_high = round(current_price + 3 * atr, 6)
        
        # Calculate stop loss (1 ATR below entry)
        stop_loss = round(current_price - atr, 6)
        
        # Calculate potential profit and risk percentages
        ward = round((target_low / current_price - 1) * 100, 1)  # Conservative estimate
        risk = round((1 - stop_loss / current_price) * 100, 1)
        
        # Estimate time based on timeframe
        if hasattr(self, 'timeframe'):
            if self.timeframe == '1m':
                time_estimate = "00:05"
            elif self.timeframe == '5m':
                time_estimate = "00:20"
            elif self.timeframe == '15m':
                time_estimate = "01:00"
            elif self.timeframe == '30m':
                time_estimate = "02:00"
            elif self.timeframe == '1h':
                time_estimate = "04:00"
            elif self.timeframe == '4h':
                time_estimate = "12:00"
            else:  # 1d
                time_estimate = "48:00"
        else:
            time_estimate = "01:20"  # Default
        
        return {
            'type': 'BUY',
            'symbol': symbol,
            'timestamp': df.index[-1],
            'price': current_price,
            'entry_low': entry_low,
            'entry_high': entry_high,
            'target_low': target_low,
            'target_high': target_high,
            'stop_loss': stop_loss,
            'ward': ward,
            'risk': risk,
            'time': time_estimate,
            'signal_type': signal_type
        }
    
    def _create_sell_signal(self, df, symbol, signal_type):
        """
        Create a sell signal.
        
        Args:
            df (pd.DataFrame): DataFrame with market data
            symbol (str): Symbol for the signal
            signal_type (str): Type of signal
            
        Returns:
            dict: Sell signal
        """
        current_price = df['close'].iloc[-1]
        atr = df['atr_14'].iloc[-1] if 'atr_14' in df.columns else current_price * 0.01
        
        # Calculate entry range (0.5% below and above current price)
        entry_low = round(current_price * 0.995, 6)
        entry_high = round(current_price * 1.005, 6)
        
        # Calculate target (2-3 ATR below entry)
        target_high = round(current_price - 2 * atr, 6)
        target_low = round(current_price - 3 * atr, 6)
        
        # Calculate stop loss (1 ATR above entry)
        stop_loss = round(current_price + atr, 6)
        
        # Calculate potential profit and risk percentages
        ward = round((1 - target_high / current_price) * 100, 1)  # Conservative estimate
        risk = round((stop_loss / current_price - 1) * 100, 1)
        
        # Estimate time based on timeframe
        if hasattr(self, 'timeframe'):
            if self.timeframe == '1m':
                time_estimate = "00:05"
            elif self.timeframe == '5m':
                time_estimate = "00:20"
            elif self.timeframe == '15m':
                time_estimate = "01:00"
            elif self.timeframe == '30m':
                time_estimate = "02:00"
            elif self.timeframe == '1h':
                time_estimate = "04:00"
            elif self.timeframe == '4h':
                time_estimate = "12:00"
            else:  # 1d
                time_estimate = "48:00"
        else:
            time_estimate = "01:20"  # Default
        
        return {
            'type': 'SELL',
            'symbol': symbol,
            'timestamp': df.index[-1],
            'price': current_price,
            'entry_low': entry_low,
            'entry_high': entry_high,
            'target_low': target_low,
            'target_high': target_high,
            'stop_loss': stop_loss,
            'ward': ward,
            'risk': risk,
            'time': time_estimate,
            'signal_type': signal_type
        }
