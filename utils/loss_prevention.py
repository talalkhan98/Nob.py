import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class LossPreventionSafeguards:
    """
    Loss prevention safeguards for the simplified crypto trading bot.
    Focuses on protecting beginners from significant losses.
    """
    
    def __init__(self, signal_generator):
        """Initialize loss prevention safeguards with a signal generator."""
        self.signal_generator = signal_generator
        
        # Default safeguard parameters
        self.auto_stop_loss_enabled = True
        self.max_daily_loss_enabled = True
        self.volatility_adjustment_enabled = True
        self.trend_confirmation_enabled = True
        self.max_position_size_enabled = True
        
        # Safeguard parameters
        self.max_daily_loss_percentage = 2.0  # Maximum 2% account loss per day
        self.max_position_size_percentage = 5.0  # Maximum 5% of account in single position
        self.min_risk_reward_ratio = 2.0  # Minimum 2:1 risk-reward ratio
        self.max_risk_per_trade_percentage = 1.0  # Maximum 1% risk per trade
    
    def set_safeguard_parameters(self, **kwargs):
        """Set safeguard parameters."""
        if 'auto_stop_loss_enabled' in kwargs:
            self.auto_stop_loss_enabled = kwargs['auto_stop_loss_enabled']
        if 'max_daily_loss_enabled' in kwargs:
            self.max_daily_loss_enabled = kwargs['max_daily_loss_enabled']
        if 'volatility_adjustment_enabled' in kwargs:
            self.volatility_adjustment_enabled = kwargs['volatility_adjustment_enabled']
        if 'trend_confirmation_enabled' in kwargs:
            self.trend_confirmation_enabled = kwargs['trend_confirmation_enabled']
        if 'max_position_size_enabled' in kwargs:
            self.max_position_size_enabled = kwargs['max_position_size_enabled']
        if 'max_daily_loss_percentage' in kwargs:
            self.max_daily_loss_percentage = kwargs['max_daily_loss_percentage']
        if 'max_position_size_percentage' in kwargs:
            self.max_position_size_percentage = kwargs['max_position_size_percentage']
        if 'min_risk_reward_ratio' in kwargs:
            self.min_risk_reward_ratio = kwargs['min_risk_reward_ratio']
        if 'max_risk_per_trade_percentage' in kwargs:
            self.max_risk_per_trade_percentage = kwargs['max_risk_per_trade_percentage']
    
    def apply_safeguards(self, price_data, signal, account_balance, daily_loss=0):
        """
        Apply loss prevention safeguards to the trading signal.
        
        Args:
            price_data (pd.DataFrame): DataFrame with OHLCV data
            signal (dict): Signal information from signal generator
            account_balance (float): Current account balance
            daily_loss (float): Current daily loss amount
            
        Returns:
            dict: Modified signal with safeguards applied
        """
        # Create a copy of the original signal
        safe_signal = signal.copy()
        
        # Apply safeguards only to buy signals
        if signal['signal'] not in ['STRONG_BUY', 'BUY']:
            return safe_signal
        
        # Track modifications made by safeguards
        modifications = []
        
        # 1. Check max daily loss
        if self.max_daily_loss_enabled and daily_loss > 0:
            daily_loss_percentage = (daily_loss / account_balance) * 100
            if daily_loss_percentage >= self.max_daily_loss_percentage:
                # Downgrade signal to HOLD if daily loss limit reached
                safe_signal['signal'] = 'HOLD'
                safe_signal['explanation'] = f"Daily loss limit of {self.max_daily_loss_percentage}% reached. Trading paused to prevent further losses."
                return safe_signal
        
        # 2. Check risk-reward ratio
        if self.auto_stop_loss_enabled:
            risk = abs(signal['entry_price'] - signal['stop_loss'])
            reward = abs(signal['take_profit'] - signal['entry_price'])
            risk_reward_ratio = reward / risk if risk > 0 else 0
            
            if risk_reward_ratio < self.min_risk_reward_ratio:
                # Adjust stop loss and take profit to achieve minimum risk-reward ratio
                if risk > 0:
                    # Calculate new take profit to achieve minimum ratio
                    new_take_profit = signal['entry_price'] + (risk * self.min_risk_reward_ratio)
                    safe_signal['take_profit'] = new_take_profit
                    modifications.append(f"Take profit adjusted to ${new_take_profit:.2f} to ensure {self.min_risk_reward_ratio}:1 risk-reward ratio")
        
        # 3. Check position size based on risk
        if self.max_position_size_enabled:
            # Calculate maximum position size based on risk per trade
            price = signal['entry_price']
            stop_loss = signal['stop_loss']
            risk_percentage = abs((price - stop_loss) / price) * 100
            
            max_risk_amount = account_balance * (self.max_risk_per_trade_percentage / 100)
            max_position_size = (max_risk_amount / risk_percentage) * 100
            
            # Ensure position size doesn't exceed maximum percentage
            max_position_by_percentage = account_balance * (self.max_position_size_percentage / 100)
            recommended_position = min(max_position_size, max_position_by_percentage)
            
            safe_signal['recommended_position_size'] = recommended_position
            modifications.append(f"Position size limited to ${recommended_position:.2f} to keep risk at {self.max_risk_per_trade_percentage}% of account")
        
        # 4. Adjust for volatility
        if self.volatility_adjustment_enabled:
            # Calculate recent volatility
            returns = price_data['close'].pct_change().dropna()
            volatility = returns.std() * 100  # Convert to percentage
            
            # Adjust stop loss based on volatility
            if volatility > 5:  # High volatility
                # Widen stop loss to avoid premature exit
                current_stop_distance = abs(signal['entry_price'] - signal['stop_loss'])
                volatility_factor = min(2.0, 1.0 + (volatility - 5) / 10)
                new_stop_distance = current_stop_distance * volatility_factor
                new_stop_loss = signal['entry_price'] - new_stop_distance
                
                safe_signal['stop_loss'] = new_stop_loss
                modifications.append(f"Stop loss widened to ${new_stop_loss:.2f} due to high volatility ({volatility:.1f}%)")
                
                # Also adjust position size down in high volatility
                if 'recommended_position_size' in safe_signal:
                    volatility_adjustment = max(0.5, 1.0 - (volatility - 5) / 20)
                    safe_signal['recommended_position_size'] *= volatility_adjustment
                    modifications.append(f"Position size reduced by {(1-volatility_adjustment)*100:.0f}% due to high volatility")
        
        # 5. Confirm trend direction
        if self.trend_confirmation_enabled:
            # Check if short and medium term trends align with signal
            short_term_trend = self._get_trend_direction(price_data, window=5)
            medium_term_trend = self._get_trend_direction(price_data, window=20)
            
            if signal['signal'] in ['STRONG_BUY', 'BUY'] and short_term_trend == 'down' and medium_term_trend == 'down':
                # Downgrade signal if trading against both trends
                safe_signal['signal'] = 'HOLD'
                safe_signal['explanation'] = "Signal downgraded to HOLD because both short and medium-term trends are down. Waiting for trend alignment."
                return safe_signal
            
            if signal['signal'] in ['STRONG_BUY', 'BUY'] and (short_term_trend == 'down' or medium_term_trend == 'down'):
                # Add warning if trading against one trend
                modifications.append(f"Caution: {'Short' if short_term_trend == 'down' else 'Medium'}-term trend is down while signal is buy")
        
        # Add modifications to explanation
        if modifications:
            safe_signal['safeguards_applied'] = modifications
            if 'explanation' in safe_signal:
                safe_signal['explanation'] += " SAFEGUARDS: " + " ".join(modifications)
        
        return safe_signal
    
    def get_max_safe_position_size(self, price_data, signal, account_balance):
        """
        Calculate the maximum safe position size based on risk parameters.
        
        Args:
            price_data (pd.DataFrame): DataFrame with OHLCV data
            signal (dict): Signal information from signal generator
            account_balance (float): Current account balance
            
        Returns:
            dict: Position size information
        """
        # Default position size based on max percentage
        default_position = account_balance * (self.max_position_size_percentage / 100)
        
        # If not a buy signal, return default
        if signal['signal'] not in ['STRONG_BUY', 'BUY']:
            return {
                'position_size': 0,
                'percentage': 0,
                'explanation': "No position size calculated for non-buy signals."
            }
        
        # Calculate position size based on risk per trade
        price = signal['entry_price']
        stop_loss = signal['stop_loss']
        
        # Calculate risk percentage
        risk_percentage = abs((price - stop_loss) / price) * 100
        
        # Calculate maximum position size based on risk
        max_risk_amount = account_balance * (self.max_risk_per_trade_percentage / 100)
        risk_based_position = (max_risk_amount / risk_percentage) * 100
        
        # Calculate position size based on volatility
        volatility_based_position = default_position
        if self.volatility_adjustment_enabled:
            # Calculate recent volatility
            returns = price_data['close'].pct_change().dropna()
            volatility = returns.std() * 100  # Convert to percentage
            
            # Adjust position size based on volatility
            if volatility > 3:  # Above normal volatility
                volatility_factor = max(0.3, 1.0 - (volatility - 3) / 20)
                volatility_based_position = default_position * volatility_factor
        
        # Calculate position size based on signal strength
        strength_based_position = default_position * (signal['strength'] / 100)
        
        # Take the minimum of all calculated position sizes
        safe_position = min(default_position, risk_based_position, volatility_based_position, strength_based_position)
        safe_percentage = (safe_position / account_balance) * 100
        
        # Round to 2 decimal places
        safe_position = round(safe_position, 2)
        safe_percentage = round(safe_percentage, 2)
        
        # Generate explanation
        explanation = f"Maximum safe position size is ${safe_position:.2f} ({safe_percentage:.2f}% of account). "
        
        if safe_position == risk_based_position:
            explanation += f"Limited by maximum risk per trade ({self.max_risk_per_trade_percentage}% of account)."
        elif safe_position == volatility_based_position:
            explanation += "Limited by current market volatility."
        elif safe_position == strength_based_position:
            explanation += f"Limited by signal strength ({signal['strength']}%)."
        else:
            explanation += f"Limited by maximum position size ({self.max_position_size_percentage}% of account)."
        
        return {
            'position_size': safe_position,
            'percentage': safe_percentage,
            'explanation': explanation
        }
    
    def get_loss_prevention_tips(self):
        """
        Get loss prevention tips for beginners.
        
        Returns:
            list: List of loss prevention tips
        """
        tips = [
            "Always use stop losses to protect your investment from significant losses.",
            "Never invest more than you can afford to lose, especially in volatile markets.",
            "Start with small position sizes (1-2% of your account) until you gain experience.",
            "Don't chase losses by increasing position sizes after losing trades.",
            "Avoid trading during major news events when volatility is highest.",
            "Don't use leverage (borrowed money) until you're consistently profitable.",
            "Take regular profits instead of waiting for the 'perfect' exit.",
            "Diversify across different cryptocurrencies to reduce risk.",
            "Be patient - not every signal requires immediate action.",
            "Follow the bot's signals rather than making emotional decisions."
        ]
        return tips
    
    def check_market_conditions(self, price_data):
        """
        Check market conditions for potential risks.
        
        Args:
            price_data (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            dict: Market condition assessment
        """
        # Calculate recent volatility
        returns = price_data['close'].pct_change().dropna()
        volatility = returns.std() * 100  # Convert to percentage
        
        # Calculate recent trend
        short_trend = self._get_trend_direction(price_data, window=5)
        medium_trend = self._get_trend_direction(price_data, window=20)
        long_trend = self._get_trend_direction(price_data, window=50)
        
        # Calculate recent volume
        recent_volume = price_data['volume'].iloc[-5:].mean()
        previous_volume = price_data['volume'].iloc[-10:-5].mean()
        volume_change = ((recent_volume - previous_volume) / previous_volume) * 100 if previous_volume > 0 else 0
        
        # Determine market condition
        if volatility > 8:
            risk_level = "High"
            risk_explanation = "Market volatility is very high, suggesting increased risk."
        elif volatility > 5:
            risk_level = "Medium"
            risk_explanation = "Market volatility is elevated, suggesting moderate risk."
        else:
            risk_level = "Low"
            risk_explanation = "Market volatility is normal, suggesting lower risk."
        
        # Check for trend alignment
        if short_trend == medium_trend == long_trend:
            trend_alignment = "Strong"
            if short_trend == "up":
                trend_explanation = "All timeframes show upward trend, suggesting strong bullish momentum."
            else:
                trend_explanation = "All timeframes show downward trend, suggesting strong bearish momentum."
        elif medium_trend == long_trend:
            trend_alignment = "Moderate"
            if medium_trend == "up":
                trend_explanation = "Medium and long-term trends are up, but short-term may be experiencing a pullback."
            else:
                trend_explanation = "Medium and long-term trends are down, but short-term may be experiencing a bounce."
        else:
            trend_alignment = "Weak"
            trend_explanation = "Trends across timeframes are not aligned, suggesting uncertainty."
        
        # Check for unusual volume
        if volume_change > 100:
            volume_alert = "Very High"
            volume_explanation = "Trading volume has more than doubled recently, suggesting potential major market moves."
        elif volume_change > 50:
            volume_alert = "High"
            volume_explanation = "Trading volume has increased significantly, suggesting increased market activity."
        elif volume_change < -50:
            volume_alert = "Very Low"
            volume_explanation = "Trading volume has decreased significantly, suggesting potential lack of interest."
        else:
            volume_alert = "Normal"
            volume_explanation = "Trading volume is within normal ranges."
        
        # Generate overall assessment
        if risk_level == "High" or volume_alert in ["Very High", "Very Low"]:
            overall_assessment = "Exercise extreme caution. Consider reducing position sizes or waiting for more stable conditions."
        elif risk_level == "Medium" or volume_alert == "High":
            overall_assessment = "Trade with caution. Use proper position sizing and ensure stop losses are in place."
        else:
            overall_assessment = "Normal trading conditions. Follow standard risk management practices."
        
        return {
            'risk_level': risk_level,
            'risk_explanation': risk_explanation,
            'volatility': volatility,
            'trend_alignment': trend_alignment,
            'trend_explanation': trend_explanation,
            'volume_alert': volume_alert,
            'volume_explanation': volume_explanation,
            'overall_assessment': overall_assessment
        }
    
    def _get_trend_direction(self, price_data, window=20):
        """Determine trend direction based on simple moving average."""
        if len(price_data) < window:
            return "neutral"
        
        df = price_data.copy()
        df['sma'] = df['close'].rolling(window=window).mean()
        
        # Get last valid SMA
        last_sma = df['sma'].dropna().iloc[-1]
        
        # Get last price
        last_price = df['close'].iloc[-1]
        
        # Determine trend direction
        if last_price > last_sma:
            return "up"
        else:
            return "down"
