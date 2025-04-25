import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import threading
import ccxt
import json
import os
from live_trading.signal_generator import SignalGenerator
from live_trading.next_candle_predictor import NextCandlePredictor

class LiveTradingManager:
    """
    A class for managing live trading operations, including signal generation,
    next candle prediction, and trade execution.
    """
    
    def __init__(self, exchange_name=None, api_key=None, api_secret=None):
        """
        Initialize the LiveTradingManager.
        
        Args:
            exchange_name (str): Name of the exchange to connect to
            api_key (str): API key for authenticated requests
            api_secret (str): API secret for authenticated requests
        """
        self.exchange_instance = None
        self.exchange_name = exchange_name
        self.api_key = api_key
        self.api_secret = api_secret
        
        # Initialize components
        self.signal_generator = SignalGenerator()
        self.predictor = NextCandlePredictor()
        
        # Trading state
        self.is_running = False
        self.trading_thread = None
        self.active_symbols = []
        self.trading_history = []
        self.active_trades = {}
        
        # Connect to exchange if credentials provided
        if exchange_name and api_key and api_secret:
            self.connect_exchange()
    
    def connect_exchange(self):
        """
        Connect to the specified cryptocurrency exchange.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Map exchange name to CCXT class
            exchange_map = {
                'Binance': ccxt.binance,
                'Coinbase': ccxt.coinbasepro,
                'Kraken': ccxt.kraken,
                'Kucoin': ccxt.kucoin
            }
            
            if self.exchange_name not in exchange_map:
                print(f"Exchange {self.exchange_name} not supported. Available exchanges: {list(exchange_map.keys())}")
                return False
            
            # Initialize exchange with API credentials
            exchange_class = exchange_map[self.exchange_name]
            self.exchange_instance = exchange_class({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True
            })
            
            # Update signal generator with exchange instance
            self.signal_generator.exchange = self.exchange_instance
            
            print(f"Successfully connected to {self.exchange_name}.")
            return True
            
        except Exception as e:
            print(f"Error connecting to exchange: {str(e)}")
            return False
    
    def start_trading(self, symbols, timeframe='1h', update_interval=60):
        """
        Start live trading for specified symbols.
        
        Args:
            symbols (list): List of trading pair symbols
            timeframe (str): Timeframe for analysis
            update_interval (int): Interval between updates in seconds
            
        Returns:
            bool: True if started successfully, False otherwise
        """
        if self.is_running:
            print("Trading is already running.")
            return False
        
        if not symbols:
            print("No symbols specified for trading.")
            return False
        
        self.active_symbols = symbols
        self.is_running = True
        self.trading_thread = threading.Thread(
            target=self._trading_loop,
            args=(symbols, timeframe, update_interval),
            daemon=True
        )
        self.trading_thread.start()
        
        print(f"Started live trading for {', '.join(symbols)} on {timeframe} timeframe.")
        return True
    
    def stop_trading(self):
        """
        Stop live trading.
        
        Returns:
            bool: True if stopped successfully, False otherwise
        """
        if not self.is_running:
            print("Trading is not running.")
            return False
        
        self.is_running = False
        if self.trading_thread:
            self.trading_thread.join(timeout=1.0)
        
        # Stop signal generation for all symbols
        self.signal_generator.stop_signal_generation()
        
        print("Stopped live trading.")
        return True
    
    def _trading_loop(self, symbols, timeframe, update_interval):
        """
        Main trading loop that runs in a separate thread.
        
        Args:
            symbols (list): List of trading pair symbols
            timeframe (str): Timeframe for analysis
            update_interval (int): Interval between updates in seconds
        """
        while self.is_running:
            try:
                for symbol in symbols:
                    # Fetch latest data
                    df = self._fetch_latest_data(symbol, timeframe)
                    if df.empty:
                        continue
                    
                    # Generate trading signal
                    signal = self.signal_generator.generate_signal(symbol, timeframe)
                    
                    # Predict next candle
                    prediction = self.predictor.predict_next_candle(df, symbol)
                    
                    # Process signal and prediction
                    if signal and prediction:
                        self._process_trading_signal(signal, prediction)
                    
                    # Check active trades for this symbol
                    if symbol in self.active_trades:
                        self._check_trade_status(symbol, df.iloc[-1])
                
            except Exception as e:
                print(f"Error in trading loop: {str(e)}")
            
            # Sleep until next update
            time.sleep(update_interval)
    
    def _fetch_latest_data(self, symbol, timeframe, lookback=100):
        """
        Fetch the latest market data.
        
        Args:
            symbol (str): Trading pair symbol
            timeframe (str): Timeframe for analysis
            lookback (int): Number of candles to fetch
            
        Returns:
            pd.DataFrame: DataFrame with OHLCV data
        """
        try:
            if self.exchange_instance:
                # Fetch from connected exchange
                ohlcv = self.exchange_instance.fetch_ohlcv(symbol, timeframe, limit=lookback)
                
                # Convert to DataFrame
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                # Set timestamp as index
                df.set_index('timestamp', inplace=True)
                
                return df
            else:
                # Use signal generator's fetch method as fallback
                return self.signal_generator._fetch_latest_data(symbol, timeframe, lookback)
                
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            return pd.DataFrame()
    
    def _process_trading_signal(self, signal, prediction):
        """
        Process a trading signal and prediction.
        
        Args:
            signal (dict): Trading signal
            prediction (dict): Next candle prediction
        """
        symbol = signal['symbol']
        
        # Combine signal with prediction
        enhanced_signal = self._enhance_signal_with_prediction(signal, prediction)
        
        # Log the enhanced signal
        self._log_signal(enhanced_signal)
        
        # If we're connected to an exchange, execute the trade
        if self.exchange_instance and self._should_execute_trade(enhanced_signal):
            self._execute_trade(enhanced_signal)
    
    def _enhance_signal_with_prediction(self, signal, prediction):
        """
        Enhance a trading signal with next candle prediction.
        
        Args:
            signal (dict): Trading signal
            prediction (dict): Next candle prediction
            
        Returns:
            dict: Enhanced signal
        """
        # Create a copy of the signal
        enhanced = signal.copy()
        
        # Add prediction data
        enhanced['prediction'] = {
            'next_open': prediction['next_open'],
            'next_high': prediction['next_high'],
            'next_low': prediction['next_low'],
            'next_close': prediction['next_close'],
            'predicted_change_pct': prediction['predicted_change_pct']
        }
        
        # Adjust confidence based on prediction alignment
        if signal['type'] == 'BUY' and prediction['predicted_change_pct'] > 0:
            enhanced['confidence'] = self._upgrade_confidence(enhanced['confidence'])
        elif signal['type'] == 'SELL' and prediction['predicted_change_pct'] < 0:
            enhanced['confidence'] = self._upgrade_confidence(enhanced['confidence'])
        elif signal['type'] == 'BUY' and prediction['predicted_change_pct'] < -1:
            enhanced['confidence'] = self._downgrade_confidence(enhanced['confidence'])
        elif signal['type'] == 'SELL' and prediction['predicted_change_pct'] > 1:
            enhanced['confidence'] = self._downgrade_confidence(enhanced['confidence'])
        
        # Adjust target based on prediction
        if signal['type'] == 'BUY':
            # If prediction is higher than current target, increase target
            if prediction['next_high'] > enhanced['target_high']:
                enhanced['target_high'] = prediction['next_high']
                enhanced['reward_percent'] = round((enhanced['target_high'] - enhanced['current_price']) / enhanced['current_price'] * 100, 1)
        elif signal['type'] == 'SELL':
            # If prediction is lower than current target, decrease target
            if prediction['next_low'] < enhanced['target_low']:
                enhanced['target_low'] = prediction['next_low']
                enhanced['reward_percent'] = round((enhanced['current_price'] - enhanced['target_low']) / enhanced['current_price'] * 100, 1)
        
        return enhanced
    
    def _upgrade_confidence(self, confidence):
        """
        Upgrade confidence level.
        
        Args:
            confidence (str): Current confidence level
            
        Returns:
            str: Upgraded confidence level
        """
        if confidence == 'low':
            return 'medium'
        elif confidence == 'medium':
            return 'high'
        else:
            return confidence
    
    def _downgrade_confidence(self, confidence):
        """
        Downgrade confidence level.
        
        Args:
            confidence (str): Current confidence level
            
        Returns:
            str: Downgraded confidence level
        """
        if confidence == 'high':
            return 'medium'
        elif confidence == 'medium':
            return 'low'
        else:
            return confidence
    
    def _should_execute_trade(self, signal):
        """
        Determine if a trade should be executed based on signal.
        
        Args:
            signal (dict): Trading signal
            
        Returns:
            bool: Whether to execute the trade
        """
        # Only execute high confidence signals
        if signal['confidence'] != 'high':
            return False
        
        # Don't execute if we already have an active trade for this symbol
        if signal['symbol'] in self.active_trades:
            return False
        
        # Additional checks can be added here
        
        return True
    
    def _execute_trade(self, signal):
        """
        Execute a trade based on signal.
        
        Args:
            signal (dict): Trading signal
            
        Returns:
            bool: Whether the trade was executed successfully
        """
        # This is a placeholder for actual trade execution
        # In a real implementation, this would interact with the exchange API
        
        symbol = signal['symbol']
        trade_type = signal['type']
        entry_price = signal['current_price']
        stop_loss = signal['stop_loss']
        target_low = signal['target_low']
        target_high = signal['target_high']
        
        # Log the trade
        trade = {
            'symbol': symbol,
            'type': trade_type,
            'entry_time': datetime.now(),
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'target_low': target_low,
            'target_high': target_high,
            'status': 'open',
            'exit_time': None,
            'exit_price': None,
            'profit_loss': None,
            'profit_loss_pct': None
        }
        
        # Add to active trades
        self.active_trades[symbol] = trade
        
        # Add to trading history
        self.trading_history.append(trade)
        
        print(f"Executed {trade_type} trade for {symbol} at {entry_price}")
        return True
    
    def _check_trade_status(self, symbol, latest_data):
        """
        Check the status of an active trade.
        
        Args:
            symbol (str): Trading pair symbol
            latest_data (pd.Series): Latest price data
        """
        if symbol not in self.active_trades:
            return
        
        trade = self.active_trades[symbol]
        current_price = latest_data['close']
        
        # Check if stop loss or target has been hit
        if trade['type'] == 'BUY':
            # Check stop loss
            if current_price <= trade['stop_loss']:
                self._close_trade(symbol, current_price, 'stop_loss')
            # Check target
            elif current_price >= trade['target_low']:
                self._close_trade(symbol, current_price, 'target')
        
        elif trade['type'] == 'SELL':
            # Check stop loss
            if current_price >= trade['stop_loss']:
                self._close_trade(symbol, current_price, 'stop_loss')
            # Check target
            elif current_price <= trade['target_high']:
                self._close_trade(symbol, current_price, 'target')
    
    def _close_trade(self, symbol, exit_price, reason):
        """
        Close an active trade.
        
        Args:
            symbol (str): Trading pair symbol
            exit_price (float): Exit price
            reason (str): Reason for closing the trade
        """
        if symbol not in self.active_trades:
            return
        
        trade = self.active_trades[symbol]
        trade['exit_time'] = datetime.now()
        trade['exit_price'] = exit_price
        
        # Calculate profit/loss
        if trade['type'] == 'BUY':
            trade['profit_loss'] = exit_price - trade['entry_price']
            trade['profit_loss_pct'] = (exit_price - trade['entry_price']) / trade['entry_price'] * 100
        else:  # SELL
            trade['profit_loss'] = trade['entry_price'] - exit_price
            trade['profit_loss_pct'] = (trade['entry_price'] - exit_price) / trade['entry_price'] * 100
        
        trade['status'] = 'closed'
        trade['close_reason'] = reason
        
        # Update trading history
        for i, hist_trade in enumerate(self.trading_history):
            if hist_trade['symbol'] == symbol and hist_trade['status'] == 'open':
                self.trading_history[i] = trade
                break
        
        # Remove from active trades
        del self.active_trades[symbol]
        
        print(f"Closed {trade['type']} trade for {symbol} at {exit_price}. P/L: {trade['profit_loss_pct']:.2f}%")
    
    def _log_signal(self, signal):
        """
        Log a trading signal.
        
        Args:
            signal (dict): Trading signal
        """
        # Create logs directory if it doesn't exist
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        # Log file path
        log_file = os.path.join(logs_dir, 'trading_signals.json')
        
        # Convert datetime to string for JSON serialization
        signal_copy = signal.copy()
        signal_copy['timestamp'] = signal_copy['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        
        # Append to log file
        try:
            # Read existing logs
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            # Append new signal
            logs.append(signal_copy)
            
            # Write back to file
            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2)
                
        except Exception as e:
            print(f"Error logging signal: {str(e)}")
    
    def get_active_trades(self):
        """
        Get list of active trades.
        
        Returns:
            list: List of active trades
        """
        return list(self.active_trades.values())
    
    def get_trading_history(self, limit=None):
        """
        Get trading history.
        
        Args:
            limit (int): Maximum number of trades to return
            
        Returns:
            list: List of historical trades
        """
        if limit:
            return self.trading_history[-limit:]
        return self.trading_history
    
    def get_performance_metrics(self):
        """
        Calculate performance metrics for closed trades.
        
        Returns:
            dict: Performance metrics
        """
        closed_trades = [trade for trade in self.trading_history if trade['status'] == 'closed']
        
        if not closed_trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'total_profit_loss': 0
            }
        
        # Calculate metrics
        total_trades = len(closed_trades)
        winning_trades = [trade for trade in closed_trades if trade['profit_loss'] > 0]
        losing_trades = [trade for trade in closed_trades if trade['profit_loss'] <= 0]
        
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        
        avg_profit = sum(trade['profit_loss'] for trade in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(trade['profit_loss'] for trade in losing_trades) / len(losing_trades) if losing_trades else 0
        
        total_profit = sum(trade['profit_loss'] for trade in winning_trades)
        total_loss = abs(sum(trade['profit_loss'] for trade in losing_trades))
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        total_profit_loss = sum(trade['profit_loss'] for trade in closed_trades)
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_profit_loss': total_profit_loss
        }
    
    def format_signal_for_display(self, signal):
        """
        Format a signal for display in the UI.
        
        Args:
            signal (dict): Signal dictionary
            
        Returns:
            str: Formatted signal text
        """
        if not signal:
            return "No signal available."
        
        formatted = f"""
        {signal['type']} SIGNAL for {signal['symbol']}
        
        Entry: ${signal['entry_low']} - ${signal['entry_high']}
        Target: ${signal['target_low']} - ${signal['target_high']}
        Stop Loss: ${signal['stop_loss']}
        
        Ward: +{signal['reward_percent']}%
        Risk: -{signal['risk_percent']}%
        Time: {signal['time_to_hold']}
        
        Confidence: {signal['confidence'].upper()}
        Generated: {signal['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        # Add prediction if available
        if 'prediction' in signal:
            pred = signal['prediction']
            formatted += f"""
        
        NEXT CANDLE PREDICTION:
        Open: ${pred['next_open']:.6f}
        High: ${pred['next_high']:.6f}
        Low: ${pred['next_low']:.6f}
        Close: ${pred['next_close']:.6f}
        Expected Change: {pred['predicted_change_pct']:.2f}%
        """
        
        return formatted
