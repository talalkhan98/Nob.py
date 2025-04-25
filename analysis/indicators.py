import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator

class TechnicalIndicators:
    """
    A class for calculating and visualizing technical indicators for cryptocurrency trading.
    """
    
    def __init__(self):
        """Initialize the TechnicalIndicators class."""
        pass
    
    def add_all_indicators(self, df):
        """
        Add all available technical indicators to the dataframe.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with added indicators
        """
        df = self.add_moving_averages(df)
        df = self.add_rsi(df)
        df = self.add_macd(df)
        df = self.add_bollinger_bands(df)
        df = self.add_stochastic(df)
        df = self.add_atr(df)
        df = self.add_adx(df)
        df = self.add_volume_indicators(df)
        
        return df
    
    def add_moving_averages(self, df, short_window=20, medium_window=50, long_window=200):
        """
        Add Simple Moving Average (SMA) and Exponential Moving Average (EMA) indicators.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            short_window (int): Window for short-term MA
            medium_window (int): Window for medium-term MA
            long_window (int): Window for long-term MA
            
        Returns:
            pd.DataFrame: DataFrame with added MA indicators
        """
        # Make a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Add Simple Moving Averages
        sma_short = SMAIndicator(close=df_copy['close'], window=short_window)
        sma_medium = SMAIndicator(close=df_copy['close'], window=medium_window)
        sma_long = SMAIndicator(close=df_copy['close'], window=long_window)
        
        df_copy[f'sma_{short_window}'] = sma_short.sma_indicator()
        df_copy[f'sma_{medium_window}'] = sma_medium.sma_indicator()
        df_copy[f'sma_{long_window}'] = sma_long.sma_indicator()
        
        # Add Exponential Moving Averages
        ema_short = EMAIndicator(close=df_copy['close'], window=short_window)
        ema_medium = EMAIndicator(close=df_copy['close'], window=medium_window)
        ema_long = EMAIndicator(close=df_copy['close'], window=long_window)
        
        df_copy[f'ema_{short_window}'] = ema_short.ema_indicator()
        df_copy[f'ema_{medium_window}'] = ema_medium.ema_indicator()
        df_copy[f'ema_{long_window}'] = ema_long.ema_indicator()
        
        return df_copy
    
    def add_rsi(self, df, window=14):
        """
        Add Relative Strength Index (RSI) indicator.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            window (int): Window for RSI calculation
            
        Returns:
            pd.DataFrame: DataFrame with added RSI indicator
        """
        df_copy = df.copy()
        
        rsi = RSIIndicator(close=df_copy['close'], window=window)
        df_copy[f'rsi_{window}'] = rsi.rsi()
        
        return df_copy
    
    def add_macd(self, df, window_slow=26, window_fast=12, window_sign=9):
        """
        Add Moving Average Convergence Divergence (MACD) indicator.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            window_slow (int): Window for slow EMA
            window_fast (int): Window for fast EMA
            window_sign (int): Window for signal line
            
        Returns:
            pd.DataFrame: DataFrame with added MACD indicators
        """
        df_copy = df.copy()
        
        macd = MACD(
            close=df_copy['close'],
            window_slow=window_slow,
            window_fast=window_fast,
            window_sign=window_sign
        )
        
        df_copy['macd_line'] = macd.macd()
        df_copy['macd_signal'] = macd.macd_signal()
        df_copy['macd_histogram'] = macd.macd_diff()
        
        return df_copy
    
    def add_bollinger_bands(self, df, window=20, window_dev=2):
        """
        Add Bollinger Bands indicator.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            window (int): Window for moving average
            window_dev (int): Number of standard deviations
            
        Returns:
            pd.DataFrame: DataFrame with added Bollinger Bands indicators
        """
        df_copy = df.copy()
        
        bollinger = BollingerBands(
            close=df_copy['close'],
            window=window,
            window_dev=window_dev
        )
        
        df_copy['bollinger_mavg'] = bollinger.bollinger_mavg()
        df_copy['bollinger_hband'] = bollinger.bollinger_hband()
        df_copy['bollinger_lband'] = bollinger.bollinger_lband()
        df_copy['bollinger_width'] = bollinger.bollinger_wband()
        df_copy['bollinger_pband'] = bollinger.bollinger_pband()
        
        return df_copy
    
    def add_stochastic(self, df, window=14, smooth_window=3):
        """
        Add Stochastic Oscillator indicator.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            window (int): Window for %K line
            smooth_window (int): Window for %D line
            
        Returns:
            pd.DataFrame: DataFrame with added Stochastic Oscillator indicators
        """
        df_copy = df.copy()
        
        stoch = StochasticOscillator(
            high=df_copy['high'],
            low=df_copy['low'],
            close=df_copy['close'],
            window=window,
            smooth_window=smooth_window
        )
        
        df_copy['stoch_k'] = stoch.stoch()
        df_copy['stoch_d'] = stoch.stoch_signal()
        
        return df_copy
    
    def add_atr(self, df, window=14):
        """
        Add Average True Range (ATR) indicator.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            window (int): Window for ATR calculation
            
        Returns:
            pd.DataFrame: DataFrame with added ATR indicator
        """
        df_copy = df.copy()
        
        atr = AverageTrueRange(
            high=df_copy['high'],
            low=df_copy['low'],
            close=df_copy['close'],
            window=window
        )
        
        df_copy[f'atr_{window}'] = atr.average_true_range()
        
        return df_copy
    
    def add_adx(self, df, window=14):
        """
        Add Average Directional Index (ADX) indicator.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            window (int): Window for ADX calculation
            
        Returns:
            pd.DataFrame: DataFrame with added ADX indicators
        """
        df_copy = df.copy()
        
        adx = ADXIndicator(
            high=df_copy['high'],
            low=df_copy['low'],
            close=df_copy['close'],
            window=window
        )
        
        df_copy[f'adx_{window}'] = adx.adx()
        df_copy['adx_pos'] = adx.adx_pos()
        df_copy['adx_neg'] = adx.adx_neg()
        
        return df_copy
    
    def add_volume_indicators(self, df, window=14):
        """
        Add volume-based indicators.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            window (int): Window for calculations
            
        Returns:
            pd.DataFrame: DataFrame with added volume indicators
        """
        df_copy = df.copy()
        
        # On-Balance Volume
        obv = OnBalanceVolumeIndicator(close=df_copy['close'], volume=df_copy['volume'])
        df_copy['obv'] = obv.on_balance_volume()
        
        # Volume Weighted Average Price (if we have high frequency data)
        try:
            vwap = VolumeWeightedAveragePrice(
                high=df_copy['high'],
                low=df_copy['low'],
                close=df_copy['close'],
                volume=df_copy['volume'],
                window=window
            )
            df_copy['vwap'] = vwap.volume_weighted_average_price()
        except:
            # VWAP typically requires intraday data
            pass
        
        return df_copy
    
    def plot_price_with_mas(self, df, short_window=20, medium_window=50, long_window=200, use_plotly=True):
        """
        Plot price with moving averages.
        
        Args:
            df (pd.DataFrame): DataFrame with price and MA data
            short_window (int): Window for short-term MA
            medium_window (int): Window for medium-term MA
            long_window (int): Window for long-term MA
            use_plotly (bool): Whether to use Plotly for interactive charts
            
        Returns:
            fig: Figure object (matplotlib or plotly)
        """
        if use_plotly:
            fig = go.Figure()
            
            # Add price
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price'
            ))
            
            # Add moving averages
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[f'sma_{short_window}'],
                name=f'SMA {short_window}',
                line=dict(color='blue')
            ))
            
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[f'sma_{medium_window}'],
                name=f'SMA {medium_window}',
                line=dict(color='orange')
            ))
            
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[f'sma_{long_window}'],
                name=f'SMA {long_window}',
                line=dict(color='green')
            ))
            
            # Update layout
            fig.update_layout(
                title='Price with Moving Averages',
                xaxis_title='Date',
                yaxis_title='Price',
                xaxis_rangeslider_visible=False,
                height=600
            )
            
            return fig
            
        else:
            # Matplotlib version
            plt.figure(figsize=(12, 6))
            plt.plot(df['close'], label='Close Price')
            plt.plot(df[f'sma_{short_window}'], label=f'SMA {short_window}')
            plt.plot(df[f'sma_{medium_window}'], label=f'SMA {medium_window}')
            plt.plot(df[f'sma_{long_window}'], label=f'SMA {long_window}')
            plt.title('Price with Moving Averages')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            
            return plt
    
    def plot_rsi(self, df, window=14, use_plotly=True):
        """
        Plot RSI indicator.
        
        Args:
            df (pd.DataFrame): DataFrame with RSI data
            window (int): Window for RSI calculation
            use_plotly (bool): Whether to use Plotly for interactive charts
            
        Returns:
            fig: Figure object (matplotlib or plotly)
        """
        if use_plotly:
            fig = go.Figure()
            
            # Add RSI
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[f'rsi_{window}'],
                name=f'RSI {window}',
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
                title=f'RSI ({window})',
                xaxis_title='Date',
                yaxis_title='RSI',
                yaxis=dict(range=[0, 100]),
                height=300
            )
            
            return fig
            
        else:
            # Matplotlib version
            plt.figure(figsize=(12, 3))
            plt.plot(df[f'rsi_{window}'], label=f'RSI {window}', color='purple')
            plt.axhline(y=70, color='r', linestyle='--')
            plt.axhline(y=30, color='g', linestyle='--')
            plt.title(f'RSI ({window})')
            plt.xlabel('Date')
            plt.ylabel('RSI')
            plt.ylim(0, 100)
            plt.grid(True)
            
            return plt
    
    def plot_macd(self, df, use_plotly=True):
        """
        Plot MACD indicator.
        
        Args:
            df (pd.DataFrame): DataFrame with MACD data
            use_plotly (bool): Whether to use Plotly for interactive charts
            
        Returns:
            fig: Figure object (matplotlib or plotly)
        """
        if use_plotly:
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
                title='MACD',
                xaxis_title='Date',
                yaxis_title='MACD',
                height=300
            )
            
            return fig
            
        else:
            # Matplotlib version
            plt.figure(figsize=(12, 3))
            plt.plot(df['macd_line'], label='MACD Line', color='blue')
            plt.plot(df['macd_signal'], label='Signal Line', color='red')
            
            # Plot histogram
            plt.bar(df.index, df['macd_histogram'], label='Histogram', color=['green' if val >= 0 else 'red' for val in df['macd_histogram']])
            
            plt.title('MACD')
            plt.xlabel('Date')
            plt.ylabel('MACD')
            plt.legend()
            plt.grid(True)
            
            return plt
    
    def plot_bollinger_bands(self, df, use_plotly=True):
        """
        Plot Bollinger Bands indicator.
        
        Args:
            df (pd.DataFrame): DataFrame with Bollinger Bands data
            use_plotly (bool): Whether to use Plotly for interactive charts
            
        Returns:
            fig: Figure object (matplotlib or plotly)
        """
        if use_plotly:
            fig = go.Figure()
            
            # Add price
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price'
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
                xaxis_rangeslider_visible=False,
                height=600
            )
            
            return fig
            
        else:
            # Matplotlib version
            plt.figure(figsize=(12, 6))
            plt.plot(df['close'], label='Close Price', color='black')
            plt.plot(df['bollinger_hband'], label='Upper Band', color='red', alpha=0.5)
            plt.plot(df['bollinger_mavg'], label='Middle Band', color='blue', alpha=0.5)
            plt.plot(df['bollinger_lband'], label='Lower Band', color='green', alpha=0.5)
            plt.title('Bollinger Bands')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            
            return plt
    
    def identify_signals(self, df):
        """
        Identify trading signals based on technical indicators.
        
        Args:
            df (pd.DataFrame): DataFrame with technical indicators
            
        Returns:
            pd.DataFrame: DataFrame with added signal columns
        """
        df_copy = df.copy()
        
        # Initialize signal columns
        df_copy['signal_ma_crossover'] = 0
        df_copy['signal_rsi'] = 0
        df_copy['signal_macd'] = 0
        df_copy['signal_bollinger'] = 0
        df_copy['signal_combined'] = 0
        
        # MA Crossover Signal (Golden Cross / Death Cross)
        df_copy['ma_crossover'] = np.where(
            df_copy['sma_50'] > df_copy['sma_200'],
            1,  # Bullish (Golden Cross)
            -1  # Bearish (Death Cross)
        )
        
        # Signal when crossover happens
        df_copy['signal_ma_crossover'] = df_copy['ma_crossover'].diff().fillna(0)
        
        # RSI Signal
        df_copy['signal_rsi'] = np.where(
            df_copy['rsi_14'] < 30,
            1,  # Oversold - Buy signal
            np.where(
                df_copy['rsi_14'] > 70,
                -1,  # Overbought - Sell signal
                0
            )
        )
        
        # MACD Signal
        df_copy['signal_macd'] = np.where(
            (df_copy['macd_line'] > df_copy['macd_signal']) & 
            (df_copy['macd_line'].shift(1) <= df_copy['macd_signal'].shift(1)),
            1,  # Bullish crossover
            np.where(
                (df_copy['macd_line'] < df_copy['macd_signal']) & 
                (df_copy['macd_line'].shift(1) >= df_copy['macd_signal'].shift(1)),
                -1,  # Bearish crossover
                0
            )
        )
        
        # Bollinger Bands Signal
        df_copy['signal_bollinger'] = np.where(
            df_copy['close'] <= df_copy['bollinger_lband'],
            1,  # Price at lower band - potential buy
            np.where(
                df_copy['close'] >= df_copy['bollinger_hband'],
                -1,  # Price at upper band - potential sell
                0
            )
        )
        
        # Combined Signal (simple average of all signals)
        df_copy['signal_combined'] = (
            df_copy['signal_ma_crossover'] + 
            df_copy['signal_rsi'] + 
            df_copy['signal_macd'] + 
            df_copy['signal_bollinger']
        ) / 4
        
        return df_copy
    
    def get_current_signals(self, df):
        """
        Get current trading signals based on the latest data.
        
        Args:
            df (pd.DataFrame): DataFrame with technical indicators and signals
            
        Returns:
            dict: Dictionary with current signals
        """
        # Get the latest row
        latest = df.iloc[-1]
        
        signals = {
            'ma_crossover': 'Bullish' if latest['ma_crossover'] == 1 else 'Bearish',
            'rsi': latest['rsi_14'],
            'rsi_signal': 'Oversold (Buy)' if latest['signal_rsi'] == 1 else 'Overbought (Sell)' if latest['signal_rsi'] == -1 else 'Neutral',
            'macd': 'Bullish Crossover' if latest['signal_macd'] == 1 else 'Bearish Crossover' if latest['signal_macd'] == -1 else 'Neutral',
            'bollinger': 'Lower Band (Buy)' if latest['signal_bollinger'] == 1 else 'Upper Band (Sell)' if latest['signal_bollinger'] == -1 else 'Neutral',
            'combined': latest['signal_combined'],
            'overall': 'Strong Buy' if latest['signal_combined'] > 0.5 else 
                      'Buy' if latest['signal_combined'] > 0 else
                      'Strong Sell' if latest['signal_combined'] < -0.5 else
                      'Sell' if latest['signal_combined'] < 0 else 'Neutral'
        }
        
        return signals
