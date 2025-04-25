import pandas as pd
import numpy as np
from datetime import datetime
from analysis.indicators import TechnicalIndicators

class TradingStrategies:
    """
    A class for implementing various cryptocurrency trading strategies.
    """
    
    def __init__(self):
        """Initialize the TradingStrategies class."""
        self.indicators = TechnicalIndicators()
    
    def apply_strategy(self, df, strategy_name, **kwargs):
        """
        Apply a specific trading strategy to the dataframe.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            strategy_name (str): Name of the strategy to apply
            **kwargs: Additional parameters for the strategy
            
        Returns:
            pd.DataFrame: DataFrame with added strategy signals
        """
        # Add technical indicators first
        df = self.indicators.add_all_indicators(df)
        
        # Apply the selected strategy
        if strategy_name == "RSI Strategy":
            return self.rsi_strategy(df, **kwargs)
        elif strategy_name == "MACD Strategy":
            return self.macd_strategy(df, **kwargs)
        elif strategy_name == "Bollinger Bands Strategy":
            return self.bollinger_bands_strategy(df, **kwargs)
        elif strategy_name == "Moving Average Crossover":
            return self.ma_crossover_strategy(df, **kwargs)
        elif strategy_name == "Custom Strategy":
            return self.custom_strategy(df, **kwargs)
        else:
            print(f"Strategy '{strategy_name}' not found. Using RSI Strategy as default.")
            return self.rsi_strategy(df, **kwargs)
    
    def rsi_strategy(self, df, overbought=70, oversold=30, window=14):
        """
        Implement RSI trading strategy.
        
        Args:
            df (pd.DataFrame): DataFrame with technical indicators
            overbought (int): RSI level considered overbought
            oversold (int): RSI level considered oversold
            window (int): RSI window
            
        Returns:
            pd.DataFrame: DataFrame with added strategy signals
        """
        df_copy = df.copy()
        
        # Ensure RSI is calculated
        if f'rsi_{window}' not in df_copy.columns:
            df_copy = self.indicators.add_rsi(df_copy, window=window)
        
        # Initialize signal and position columns
        df_copy['signal'] = 0
        df_copy['position'] = 0
        
        # Generate signals
        # Buy signal when RSI crosses below oversold level and then back above it
        df_copy['rsi_oversold'] = (df_copy[f'rsi_{window}'] < oversold).astype(int)
        df_copy['rsi_oversold_cross'] = df_copy['rsi_oversold'].diff().fillna(0)
        df_copy.loc[(df_copy['rsi_oversold_cross'] == -1) & (df_copy[f'rsi_{window}'] > oversold), 'signal'] = 1
        
        # Sell signal when RSI crosses above overbought level and then back below it
        df_copy['rsi_overbought'] = (df_copy[f'rsi_{window}'] > overbought).astype(int)
        df_copy['rsi_overbought_cross'] = df_copy['rsi_overbought'].diff().fillna(0)
        df_copy.loc[(df_copy['rsi_overbought_cross'] == -1) & (df_copy[f'rsi_{window}'] < overbought), 'signal'] = -1
        
        # Calculate positions (1 for long, -1 for short, 0 for no position)
        df_copy['position'] = df_copy['signal'].cumsum()
        
        # Clean up temporary columns
        df_copy = df_copy.drop(['rsi_oversold', 'rsi_oversold_cross', 'rsi_overbought', 'rsi_overbought_cross'], axis=1)
        
        return df_copy
    
    def macd_strategy(self, df, window_slow=26, window_fast=12, window_sign=9):
        """
        Implement MACD trading strategy.
        
        Args:
            df (pd.DataFrame): DataFrame with technical indicators
            window_slow (int): Window for slow EMA
            window_fast (int): Window for fast EMA
            window_sign (int): Window for signal line
            
        Returns:
            pd.DataFrame: DataFrame with added strategy signals
        """
        df_copy = df.copy()
        
        # Ensure MACD is calculated
        if 'macd_line' not in df_copy.columns:
            df_copy = self.indicators.add_macd(df_copy, window_slow=window_slow, window_fast=window_fast, window_sign=window_sign)
        
        # Initialize signal and position columns
        df_copy['signal'] = 0
        df_copy['position'] = 0
        
        # Generate signals
        # Buy signal when MACD line crosses above signal line
        df_copy.loc[(df_copy['macd_line'] > df_copy['macd_signal']) & 
                   (df_copy['macd_line'].shift(1) <= df_copy['macd_signal'].shift(1)), 'signal'] = 1
        
        # Sell signal when MACD line crosses below signal line
        df_copy.loc[(df_copy['macd_line'] < df_copy['macd_signal']) & 
                   (df_copy['macd_line'].shift(1) >= df_copy['macd_signal'].shift(1)), 'signal'] = -1
        
        # Calculate positions (1 for long, -1 for short, 0 for no position)
        df_copy['position'] = df_copy['signal'].cumsum()
        
        return df_copy
    
    def bollinger_bands_strategy(self, df, window=20, window_dev=2):
        """
        Implement Bollinger Bands trading strategy.
        
        Args:
            df (pd.DataFrame): DataFrame with technical indicators
            window (int): Window for moving average
            window_dev (int): Number of standard deviations
            
        Returns:
            pd.DataFrame: DataFrame with added strategy signals
        """
        df_copy = df.copy()
        
        # Ensure Bollinger Bands are calculated
        if 'bollinger_hband' not in df_copy.columns:
            df_copy = self.indicators.add_bollinger_bands(df_copy, window=window, window_dev=window_dev)
        
        # Initialize signal and position columns
        df_copy['signal'] = 0
        df_copy['position'] = 0
        
        # Generate signals
        # Buy signal when price crosses below lower band and then back above it
        df_copy['below_lband'] = (df_copy['close'] < df_copy['bollinger_lband']).astype(int)
        df_copy['below_lband_cross'] = df_copy['below_lband'].diff().fillna(0)
        df_copy.loc[(df_copy['below_lband_cross'] == -1), 'signal'] = 1
        
        # Sell signal when price crosses above upper band and then back below it
        df_copy['above_hband'] = (df_copy['close'] > df_copy['bollinger_hband']).astype(int)
        df_copy['above_hband_cross'] = df_copy['above_hband'].diff().fillna(0)
        df_copy.loc[(df_copy['above_hband_cross'] == -1), 'signal'] = -1
        
        # Calculate positions (1 for long, -1 for short, 0 for no position)
        df_copy['position'] = df_copy['signal'].cumsum()
        
        # Clean up temporary columns
        df_copy = df_copy.drop(['below_lband', 'below_lband_cross', 'above_hband', 'above_hband_cross'], axis=1)
        
        return df_copy
    
    def ma_crossover_strategy(self, df, short_window=50, long_window=200):
        """
        Implement Moving Average Crossover trading strategy.
        
        Args:
            df (pd.DataFrame): DataFrame with technical indicators
            short_window (int): Window for short-term MA
            long_window (int): Window for long-term MA
            
        Returns:
            pd.DataFrame: DataFrame with added strategy signals
        """
        df_copy = df.copy()
        
        # Ensure Moving Averages are calculated
        if f'sma_{short_window}' not in df_copy.columns or f'sma_{long_window}' not in df_copy.columns:
            df_copy = self.indicators.add_moving_averages(df_copy, short_window=short_window, medium_window=50, long_window=long_window)
        
        # Initialize signal and position columns
        df_copy['signal'] = 0
        df_copy['position'] = 0
        
        # Generate signals
        # Buy signal when short MA crosses above long MA (Golden Cross)
        df_copy.loc[(df_copy[f'sma_{short_window}'] > df_copy[f'sma_{long_window}']) & 
                   (df_copy[f'sma_{short_window}'].shift(1) <= df_copy[f'sma_{long_window}'].shift(1)), 'signal'] = 1
        
        # Sell signal when short MA crosses below long MA (Death Cross)
        df_copy.loc[(df_copy[f'sma_{short_window}'] < df_copy[f'sma_{long_window}']) & 
                   (df_copy[f'sma_{short_window}'].shift(1) >= df_copy[f'sma_{long_window}'].shift(1)), 'signal'] = -1
        
        # Calculate positions (1 for long, -1 for short, 0 for no position)
        df_copy['position'] = df_copy['signal'].cumsum()
        
        return df_copy
    
    def custom_strategy(self, df, **kwargs):
        """
        Implement a custom trading strategy combining multiple indicators.
        
        Args:
            df (pd.DataFrame): DataFrame with technical indicators
            **kwargs: Additional parameters for the strategy
            
        Returns:
            pd.DataFrame: DataFrame with added strategy signals
        """
        df_copy = df.copy()
        
        # Ensure all necessary indicators are calculated
        df_copy = self.indicators.add_all_indicators(df_copy)
        
        # Initialize signal and position columns
        df_copy['signal'] = 0
        df_copy['position'] = 0
        
        # Generate signals based on multiple indicators
        
        # RSI conditions
        rsi_oversold = df_copy['rsi_14'] < 30
        rsi_overbought = df_copy['rsi_14'] > 70
        
        # MACD conditions
        macd_bullish = (df_copy['macd_line'] > df_copy['macd_signal']) & (df_copy['macd_line'].shift(1) <= df_copy['macd_signal'].shift(1))
        macd_bearish = (df_copy['macd_line'] < df_copy['macd_signal']) & (df_copy['macd_line'].shift(1) >= df_copy['macd_signal'].shift(1))
        
        # Bollinger Bands conditions
        bb_lower_touch = df_copy['close'] <= df_copy['bollinger_lband']
        bb_upper_touch = df_copy['close'] >= df_copy['bollinger_hband']
        
        # Moving Average conditions
        ma_bullish = (df_copy['sma_50'] > df_copy['sma_200']) & (df_copy['sma_50'].shift(1) <= df_copy['sma_200'].shift(1))
        ma_bearish = (df_copy['sma_50'] < df_copy['sma_200']) & (df_copy['sma_50'].shift(1) >= df_copy['sma_200'].shift(1))
        
        # Combined buy signals (at least 2 indicators must agree)
        buy_signals = (rsi_oversold.astype(int) + 
                       macd_bullish.astype(int) + 
                       bb_lower_touch.astype(int) + 
                       ma_bullish.astype(int))
        
        # Combined sell signals (at least 2 indicators must agree)
        sell_signals = (rsi_overbought.astype(int) + 
                        macd_bearish.astype(int) + 
                        bb_upper_touch.astype(int) + 
                        ma_bearish.astype(int))
        
        # Generate signals
        df_copy.loc[buy_signals >= 2, 'signal'] = 1
        df_copy.loc[sell_signals >= 2, 'signal'] = -1
        
        # Calculate positions (1 for long, -1 for short, 0 for no position)
        df_copy['position'] = df_copy['signal'].cumsum()
        
        return df_copy
    
    def backtest_strategy(self, df, initial_capital=10000, position_size=0.1):
        """
        Backtest a trading strategy.
        
        Args:
            df (pd.DataFrame): DataFrame with strategy signals
            initial_capital (float): Initial capital for backtesting
            position_size (float): Percentage of capital to use per trade
            
        Returns:
            tuple: (DataFrame with backtest results, dict with performance metrics)
        """
        df_copy = df.copy()
        
        # Ensure signal and position columns exist
        if 'signal' not in df_copy.columns or 'position' not in df_copy.columns:
            print("Error: DataFrame must contain 'signal' and 'position' columns.")
            return df_copy, {}
        
        # Initialize portfolio columns
        df_copy['capital'] = initial_capital
        df_copy['holdings'] = 0
        df_copy['cash'] = initial_capital
        
        # Calculate trade sizes
        df_copy['trade_size'] = df_copy['signal'] * (df_copy['cash'] * position_size)
        
        # Calculate holdings and cash
        for i in range(1, len(df_copy)):
            # Update holdings based on previous holdings and new trades
            df_copy.loc[df_copy.index[i], 'holdings'] = (
                df_copy.loc[df_copy.index[i-1], 'holdings'] + 
                df_copy.loc[df_copy.index[i], 'trade_size'] / df_copy.loc[df_copy.index[i], 'close']
            )
            
            # Update cash based on previous cash and new trades
            df_copy.loc[df_copy.index[i], 'cash'] = (
                df_copy.loc[df_copy.index[i-1], 'cash'] - 
                df_copy.loc[df_copy.index[i], 'trade_size']
            )
            
            # Update total capital
            df_copy.loc[df_copy.index[i], 'capital'] = (
                df_copy.loc[df_copy.index[i], 'cash'] + 
                df_copy.loc[df_copy.index[i], 'holdings'] * df_copy.loc[df_copy.index[i], 'close']
            )
        
        # Calculate returns
        df_copy['returns'] = df_copy['capital'].pct_change()
        df_copy['cumulative_returns'] = (1 + df_copy['returns']).cumprod() - 1
        
        # Calculate drawdowns
        df_copy['peak'] = df_copy['capital'].cummax()
        df_copy['drawdown'] = (df_copy['capital'] - df_copy['peak']) / df_copy['peak']
        
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(df_copy)
        
        return df_copy, metrics
    
    def _calculate_performance_metrics(self, df):
        """
        Calculate performance metrics for a backtest.
        
        Args:
            df (pd.DataFrame): DataFrame with backtest results
            
        Returns:
            dict: Dictionary with performance metrics
        """
        # Filter out rows with NaN returns
        returns = df['returns'].dropna()
        
        # Calculate basic metrics
        total_return = df['cumulative_returns'].iloc[-1] * 100
        annual_return = ((1 + total_return/100) ** (252 / len(returns)) - 1) * 100
        
        # Calculate volatility (annualized)
        volatility = returns.std() * np.sqrt(252) * 100
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0)
        sharpe_ratio = annual_return / volatility if volatility != 0 else 0
        
        # Calculate maximum drawdown
        max_drawdown = df['drawdown'].min() * 100
        
        # Calculate win rate
        trades = df[df['signal'] != 0]
        if len(trades) > 0:
            # Calculate profit/loss for each trade
            trades['trade_pl'] = trades['signal'] * trades['close'].pct_change().shift(-1)
            
            # Calculate win rate
            winning_trades = trades[trades['trade_pl'] > 0]
            win_rate = len(winning_trades) / len(trades) * 100
            
            # Calculate profit factor
            gross_profit = winning_trades['trade_pl'].sum()
            losing_trades = trades[trades['trade_pl'] < 0]
            gross_loss = abs(losing_trades['trade_pl'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
            
            # Calculate average win and loss
            avg_win = winning_trades['trade_pl'].mean() * 100 if len(winning_trades) > 0 else 0
            avg_loss = losing_trades['trade_pl'].mean() * 100 if len(losing_trades) > 0 else 0
        else:
            win_rate = 0
            profit_factor = 0
            avg_win = 0
            avg_loss = 0
        
        # Compile metrics
        metrics = {
            'total_return': round(total_return, 2),
            'annual_return': round(annual_return, 2),
            'volatility': round(volatility, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'max_drawdown': round(max_drawdown, 2),
            'win_rate': round(win_rate, 2),
            'profit_factor': round(profit_factor, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'total_trades': len(trades),
            'start_date': df.index[0].strftime('%Y-%m-%d'),
            'end_date': df.index[-1].strftime('%Y-%m-%d')
        }
        
        return metrics
    
    def generate_trade_signals(self, df):
        """
        Generate a list of trade signals from a DataFrame with strategy results.
        
        Args:
            df (pd.DataFrame): DataFrame with strategy signals
            
        Returns:
            pd.DataFrame: DataFrame with trade signals
        """
        # Filter rows where signal is not 0
        trades = df[df['signal'] != 0].copy()
        
        # Add trade type
        trades['type'] = trades['signal'].apply(lambda x: 'BUY' if x > 0 else 'SELL')
        
        # Calculate profit/loss for each trade
        trades['next_price'] = trades['close'].shift(-1)
        trades['profit_loss'] = trades.apply(
            lambda row: (row['next_price'] - row['close']) * row['signal'],
            axis=1
        )
        
        # Calculate profit/loss percentage
        trades['profit_loss_pct'] = trades['profit_loss'] / trades['close'] * 100
        
        # Format for display
        result = trades[['type', 'close', 'profit_loss', 'profit_loss_pct']].copy()
        result.columns = ['Type', 'Price', 'Profit/Loss', 'P/L %']
        
        # Format profit/loss columns
        result['Profit/Loss'] = result['Profit/Loss'].apply(lambda x: f"${x:.2f}")
        result['P/L %'] = result['P/L %'].apply(lambda x: f"{x:.2f}%")
        
        return result
    
    def optimize_strategy(self, df, strategy_name, param_grid, initial_capital=10000, position_size=0.1):
        """
        Optimize strategy parameters using grid search.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            strategy_name (str): Name of the strategy to optimize
            param_grid (dict): Dictionary with parameter names and values to try
            initial_capital (float): Initial capital for backtesting
            position_size (float): Percentage of capital to use per trade
            
        Returns:
            tuple: (Best parameters, Best metrics)
        """
        best_params = None
        best_metrics = None
        best_return = -float('inf')
        
        # Generate all parameter combinations
        param_combinations = self._generate_param_combinations(param_grid)
        
        for params in param_combinations:
            # Apply strategy with current parameters
            result_df = self.apply_strategy(df, strategy_name, **params)
            
            # Backtest strategy
            _, metrics = self.backtest_strategy(result_df, initial_capital, position_size)
            
            # Update best parameters if current combination is better
            if metrics['total_return'] > best_return:
                best_return = metrics['total_return']
                best_params = params
                best_metrics = metrics
        
        return best_params, best_metrics
    
    def _generate_param_combinations(self, param_grid):
        """
        Generate all combinations of parameters from a parameter grid.
        
        Args:
            param_grid (dict): Dictionary with parameter names and values to try
            
        Returns:
            list: List of parameter dictionaries
        """
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        combinations = []
        
        def generate_combinations(index, current_params):
            if index == len(keys):
                combinations.append(current_params.copy())
                return
            
            for value in values[index]:
                current_params[keys[index]] = value
                generate_combinations(index + 1, current_params)
        
        generate_combinations(0, {})
        return combinations
