import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class CryptoSignalGenerator:
    """
    A simplified signal generator for cryptocurrency trading.
    Focuses on providing clear buy/sell signals for beginners.
    """
    
    def __init__(self, data_fetcher=None):
        """Initialize the signal generator."""
        self.data_fetcher = data_fetcher
        self.loss_prevention = None
        self.profit_optimizer = None
    
    def set_loss_prevention(self, loss_prevention):
        """Set the loss prevention safeguards."""
        self.loss_prevention = loss_prevention
    
    def set_profit_optimizer(self, profit_optimizer):
        """Set the profit optimization strategies."""
        self.profit_optimizer = profit_optimizer
    
    def generate_sample_data(self, symbol, days=30):
        """
        Generate sample price data for testing or demonstration.
        
        Args:
            symbol (str): Cryptocurrency symbol (e.g., 'BTC/USDT')
            days (int): Number of days of data to generate
            
        Returns:
            pd.DataFrame: DataFrame with OHLCV data
        """
        if self.data_fetcher:
            # Use data fetcher if available
            return self.data_fetcher.fetch_historical_data(symbol, '1d', days)
        
        # Generate synthetic data if no data fetcher
        dates = pd.date_range(end=datetime.now(), periods=days)
        
        # Set base price based on symbol
        if 'BTC' in symbol:
            base_price = 50000
            volatility = 0.02
        elif 'ETH' in symbol:
            base_price = 3000
            volatility = 0.025
        elif 'SOL' in symbol:
            base_price = 100
            volatility = 0.03
        elif 'BNB' in symbol:
            base_price = 500
            volatility = 0.02
        else:
            base_price = 1
            volatility = 0.02
        
        # Generate price series with random walk
        np.random.seed(42)  # For reproducible results
        returns = np.random.normal(0.0005, volatility, days)  # Slight upward bias
        price_changes = 1 + returns
        prices = base_price * np.cumprod(price_changes)
        
        # Generate OHLCV data
        data = {
            'open': prices * np.random.uniform(0.99, 1.01, days),
            'high': prices * np.random.uniform(1.01, 1.03, days),
            'low': prices * np.random.uniform(0.97, 0.99, days),
            'close': prices,
            'volume': np.random.uniform(base_price * 1000, base_price * 5000, days)
        }
        
        # Create DataFrame
        df = pd.DataFrame(data, index=dates)
        
        # Add some technical indicators
        df['ma_fast'] = df['close'].rolling(window=5).mean()
        df['ma_slow'] = df['close'].rolling(window=20).mean()
        
        return df
    
    def generate_signals(self, price_data):
        """
        Generate trading signals based on price data.
        
        Args:
            price_data (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            dict: Signal information
        """
        # Calculate basic indicators
        df = price_data.copy()
        
        # Ensure we have enough data
        if len(df) < 20:
            return {
                'signal': 'HOLD',
                'strength': 0,
                'explanation': 'Insufficient data for analysis.',
                'entry_price': df['close'].iloc[-1],
                'stop_loss': df['close'].iloc[-1] * 0.95,
                'take_profit': df['close'].iloc[-1] * 1.05,
                'risk_level': 'Medium',
                'last_price': df['close'].iloc[-1]
            }
        
        # Calculate indicators if not already present
        if 'ma_fast' not in df.columns:
            df['ma_fast'] = df['close'].rolling(window=5).mean()
        if 'ma_slow' not in df.columns:
            df['ma_slow'] = df['close'].rolling(window=20).mean()
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate Bollinger Bands
        df['ma_20'] = df['close'].rolling(window=20).mean()
        df['std_20'] = df['close'].rolling(window=20).std()
        df['upper_band'] = df['ma_20'] + (df['std_20'] * 2)
        df['lower_band'] = df['ma_20'] - (df['std_20'] * 2)
        
        # Get latest values
        current_price = df['close'].iloc[-1]
        ma_fast = df['ma_fast'].iloc[-1]
        ma_slow = df['ma_slow'].iloc[-1]
        rsi = df['rsi'].iloc[-1]
        upper_band = df['upper_band'].iloc[-1]
        lower_band = df['lower_band'].iloc[-1]
        
        # Analyze trend
        trend_score = self._analyze_trend(df)
        
        # Analyze momentum
        momentum_score = self._analyze_momentum(df)
        
        # Analyze volatility
        volatility_score, risk_level = self._analyze_volatility(df)
        
        # Analyze support/resistance
        sr_score = self._analyze_support_resistance(df)
        
        # Calculate overall signal score (-100 to +100)
        signal_score = trend_score + momentum_score + volatility_score + sr_score
        
        # Determine signal based on score
        if signal_score > 70:
            signal = 'STRONG_BUY'
        elif signal_score > 30:
            signal = 'BUY'
        elif signal_score < -70:
            signal = 'STRONG_SELL'
        elif signal_score < -30:
            signal = 'SELL'
        else:
            signal = 'HOLD'
        
        # Calculate signal strength (0-100)
        signal_strength = min(100, abs(signal_score))
        
        # Generate explanation
        explanation = self._generate_explanation(signal, df)
        
        # Calculate entry, stop loss, and take profit levels
        entry_price = current_price
        
        # For buy signals, set stop loss below recent support
        if signal in ['STRONG_BUY', 'BUY']:
            # Find recent low as support
            recent_low = df['low'].iloc[-10:].min()
            support_level = recent_low
            
            # Set stop loss just below support
            stop_loss = support_level * 0.98
            
            # Find recent high as resistance
            recent_high = df['high'].iloc[-10:].max()
            resistance_level = recent_high
            
            # Set take profit at resistance or fixed percentage
            if resistance_level > current_price:
                take_profit = resistance_level
            else:
                take_profit = current_price * 1.05  # 5% profit target
        
        # For sell signals, set stop loss above recent resistance
        elif signal in ['STRONG_SELL', 'SELL']:
            # Find recent high as resistance
            recent_high = df['high'].iloc[-10:].max()
            resistance_level = recent_high
            
            # Set stop loss just above resistance
            stop_loss = resistance_level * 1.02
            
            # Find recent low as support
            recent_low = df['low'].iloc[-10:].min()
            support_level = recent_low
            
            # Set take profit at support or fixed percentage
            if support_level < current_price:
                take_profit = support_level
            else:
                take_profit = current_price * 0.95  # 5% profit target
        
        # For hold signals, set symmetric stop loss and take profit
        else:
            stop_loss = current_price * 0.95  # 5% stop loss
            take_profit = current_price * 1.05  # 5% take profit
        
        # Create signal dictionary
        signal_dict = {
            'signal': signal,
            'strength': signal_strength,
            'explanation': explanation,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_level': risk_level,
            'last_price': current_price
        }
        
        # Apply loss prevention safeguards if available
        if self.loss_prevention:
            signal_dict = self.loss_prevention.apply_safeguards(df, signal_dict, 1000, 0)
        
        return signal_dict
    
    def get_position_size_recommendation(self, account_balance, risk_level, signal_strength):
        """
        Get recommended position size based on account balance, risk level, and signal strength.
        
        Args:
            account_balance (float): Account balance
            risk_level (str): Risk level (Low, Medium, High)
            signal_strength (int): Signal strength (0-100)
            
        Returns:
            float: Recommended position size as percentage of account balance
        """
        # Base position size based on risk level
        if risk_level == "Low":
            base_percentage = 3.0
        elif risk_level == "Medium":
            base_percentage = 5.0
        else:  # High risk
            base_percentage = 2.0
        
        # Adjust based on signal strength
        strength_factor = signal_strength / 100  # Convert to 0-1 scale
        
        # Calculate adjusted percentage
        adjusted_percentage = base_percentage * strength_factor
        
        # Ensure within reasonable limits
        adjusted_percentage = max(1.0, min(adjusted_percentage, 10.0))
        
        return adjusted_percentage
    
    def _analyze_trend(self, df):
        """
        Analyze price trend.
        
        Args:
            df (pd.DataFrame): DataFrame with price data and indicators
            
        Returns:
            float: Trend score (-100 to +100)
        """
        # Get latest values
        ma_fast = df['ma_fast'].iloc[-1]
        ma_slow = df['ma_slow'].iloc[-1]
        current_price = df['close'].iloc[-1]
        
        # Calculate trend score
        trend_score = 0
        
        # Price above/below moving averages
        if current_price > ma_fast:
            trend_score += 20
        else:
            trend_score -= 20
        
        if current_price > ma_slow:
            trend_score += 20
        else:
            trend_score -= 20
        
        # Moving average crossover
        if ma_fast > ma_slow:
            trend_score += 30
        else:
            trend_score -= 30
        
        # Higher highs and higher lows (uptrend)
        if (df['high'].iloc[-1] > df['high'].iloc[-5]) and (df['low'].iloc[-1] > df['low'].iloc[-5]):
            trend_score += 30
        
        # Lower highs and lower lows (downtrend)
        if (df['high'].iloc[-1] < df['high'].iloc[-5]) and (df['low'].iloc[-1] < df['low'].iloc[-5]):
            trend_score -= 30
        
        return trend_score
    
    def _analyze_momentum(self, df):
        """
        Analyze price momentum.
        
        Args:
            df (pd.DataFrame): DataFrame with price data and indicators
            
        Returns:
            float: Momentum score (-100 to +100)
        """
        # Get latest values
        rsi = df['rsi'].iloc[-1]
        
        # Calculate momentum score
        momentum_score = 0
        
        # RSI
        if rsi > 70:
            momentum_score -= 40  # Overbought
        elif rsi < 30:
            momentum_score += 40  # Oversold
        elif rsi > 50:
            momentum_score += 20  # Bullish
        else:
            momentum_score -= 20  # Bearish
        
        # Price change
        price_change = (df['close'].iloc[-1] / df['close'].iloc[-5] - 1) * 100
        
        if price_change > 5:
            momentum_score -= 20  # Too much too fast
        elif price_change > 2:
            momentum_score += 20  # Good momentum
        elif price_change < -5:
            momentum_score += 20  # Oversold
        elif price_change < -2:
            momentum_score -= 20  # Losing momentum
        
        return momentum_score
    
    def _analyze_volatility(self, df):
        """
        Analyze price volatility.
        
        Args:
            df (pd.DataFrame): DataFrame with price data and indicators
            
        Returns:
            tuple: (Volatility score (-100 to +100), Risk level)
        """
        # Calculate volatility
        returns = df['close'].pct_change().dropna()
        volatility = returns.std() * 100  # Convert to percentage
        
        # Calculate volatility score
        volatility_score = 0
        
        # Bollinger Bands
        current_price = df['close'].iloc[-1]
        upper_band = df['upper_band'].iloc[-1]
        lower_band = df['lower_band'].iloc[-1]
        
        # Price near bands
        if current_price > upper_band:
            volatility_score -= 40  # Overbought
        elif current_price < lower_band:
            volatility_score += 40  # Oversold
        
        # Determine risk level
        if volatility > 5:
            risk_level = "High"
        elif volatility > 3:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        return volatility_score, risk_level
    
    def _analyze_support_resistance(self, df):
        """
        Analyze support and resistance levels.
        
        Args:
            df (pd.DataFrame): DataFrame with price data and indicators
            
        Returns:
            float: Support/resistance score (-100 to +100)
        """
        # Get latest values
        current_price = df['close'].iloc[-1]
        
        # Find recent high as resistance
        recent_high = df['high'].iloc[-10:].max()
        
        # Find recent low as support
        recent_low = df['low'].iloc[-10:].min()
        
        # Calculate support/resistance score
        sr_score = 0
        
        # Price near support
        if current_price < recent_low * 1.02:
            sr_score += 40  # Near support
        
        # Price near resistance
        if current_price > recent_high * 0.98:
            sr_score -= 40  # Near resistance
        
        return sr_score
    
    def _generate_explanation(self, signal, df):
        """
        Generate a beginner-friendly explanation for the signal.
        
        Args:
            signal (str): Signal type
            df (pd.DataFrame): DataFrame with price data and indicators
            
        Returns:
            str: Explanation text
        """
        current_price = df['close'].iloc[-1]
        ma_fast = df['ma_fast'].iloc[-1]
        ma_slow = df['ma_slow'].iloc[-1]
        rsi = df['rsi'].iloc[-1]
        
        if signal == 'STRONG_BUY':
            explanation = "Strong buy signal based on multiple positive indicators. "
            
            if current_price > ma_slow:
                explanation += "Price is above the long-term average. "
            
            if ma_fast > ma_slow:
                explanation += "Short-term average is above long-term average, indicating upward momentum. "
            
            if rsi < 50:
                explanation += "RSI indicates the price is not overbought. "
            
            explanation += "Consider buying with proper risk management."
        
        elif signal == 'BUY':
            explanation = "Buy signal based on generally positive indicators. "
            
            if current_price > ma_slow:
                explanation += "Price is above the long-term average. "
            
            if ma_fast > ma_slow:
                explanation += "Short-term average is above long-term average. "
            
            explanation += "Consider a smaller position size than with a strong buy signal."
        
        elif signal == 'STRONG_SELL':
            explanation = "Strong sell signal based on multiple negative indicators. "
            
            if current_price < ma_slow:
                explanation += "Price is below the long-term average. "
            
            if ma_fast < ma_slow:
                explanation += "Short-term average is below long-term average, indicating downward momentum. "
            
            if rsi > 50:
                explanation += "RSI indicates the price is not oversold. "
            
            explanation += "Consider selling with proper risk management."
        
        elif signal == 'SELL':
            explanation = "Sell signal based on generally negative indicators. "
            
            if current_price < ma_slow:
                explanation += "Price is below the long-term average. "
            
            if ma_fast < ma_slow:
                explanation += "Short-term average is below long-term average. "
            
            explanation += "Consider a smaller position size than with a strong sell signal."
        
        else:  # HOLD
            explanation = "No clear buy or sell signal at this time. "
            
            if abs(current_price - ma_slow) / ma_slow < 0.01:
                explanation += "Price is very close to the long-term average. "
            
            if abs(ma_fast - ma_slow) / ma_slow < 0.01:
                explanation += "Short-term and long-term averages are very close. "
            
            explanation += "Better to wait for a clearer signal before taking action."
        
        return explanation
