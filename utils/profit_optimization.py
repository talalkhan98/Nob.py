import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class ProfitOptimizationStrategies:
    """
    Profit optimization strategies for the simplified crypto trading bot.
    Focuses on maximizing profits while keeping strategies simple for beginners.
    """
    
    def __init__(self, signal_generator):
        """Initialize profit optimization strategies with a signal generator."""
        self.signal_generator = signal_generator
        
        # Default optimization parameters
        self.trend_following_enabled = True
        self.profit_taking_enabled = True
        self.dca_enabled = False  # Dollar-Cost Averaging
        self.trailing_take_profit_enabled = True
        
        # Strategy parameters
        self.profit_taking_threshold = 5.0  # Take profit at 5% gain
        self.trailing_take_profit_distance = 2.0  # 2% trailing distance
        self.dca_intervals = [5, 10, 15]  # Buy more at 5%, 10%, and 15% drops
        self.dca_allocation = [0.3, 0.3, 0.4]  # Allocation for each DCA level
    
    def set_optimization_parameters(self, **kwargs):
        """Set optimization parameters."""
        if 'trend_following_enabled' in kwargs:
            self.trend_following_enabled = kwargs['trend_following_enabled']
        if 'profit_taking_enabled' in kwargs:
            self.profit_taking_enabled = kwargs['profit_taking_enabled']
        if 'dca_enabled' in kwargs:
            self.dca_enabled = kwargs['dca_enabled']
        if 'trailing_take_profit_enabled' in kwargs:
            self.trailing_take_profit_enabled = kwargs['trailing_take_profit_enabled']
        if 'profit_taking_threshold' in kwargs:
            self.profit_taking_threshold = kwargs['profit_taking_threshold']
        if 'trailing_take_profit_distance' in kwargs:
            self.trailing_take_profit_distance = kwargs['trailing_take_profit_distance']
        if 'dca_intervals' in kwargs:
            self.dca_intervals = kwargs['dca_intervals']
        if 'dca_allocation' in kwargs:
            self.dca_allocation = kwargs['dca_allocation']
    
    def optimize_entry_points(self, price_data, signal):
        """
        Optimize entry points to maximize profit potential.
        
        Args:
            price_data (pd.DataFrame): DataFrame with OHLCV data
            signal (dict): Signal information from signal generator
            
        Returns:
            dict: Optimized entry points
        """
        # Only optimize entry for buy signals
        if signal['signal'] not in ['STRONG_BUY', 'BUY']:
            return {'primary_entry': signal['entry_price'], 'secondary_entries': []}
        
        current_price = price_data['close'].iloc[-1]
        
        # Calculate support levels
        support_levels = self._identify_support_levels(price_data)
        
        # Find the closest support level below current price
        closest_support = None
        min_distance = float('inf')
        
        for level in support_levels:
            if level < current_price:
                distance = current_price - level
                if distance < min_distance:
                    min_distance = distance
                    closest_support = level
        
        # If a support level is found and it's not too far from current price
        if closest_support and (current_price - closest_support) / current_price < 0.05:
            primary_entry = closest_support
        else:
            primary_entry = signal['entry_price']
        
        # Calculate secondary entry points for DCA if enabled
        secondary_entries = []
        if self.dca_enabled:
            for interval in self.dca_intervals:
                dca_level = current_price * (1 - interval / 100)
                secondary_entries.append({
                    'price': dca_level,
                    'percentage_drop': interval
                })
        
        return {
            'primary_entry': primary_entry,
            'secondary_entries': secondary_entries
        }
    
    def optimize_exit_points(self, price_data, signal, entry_price):
        """
        Optimize exit points to maximize profits.
        
        Args:
            price_data (pd.DataFrame): DataFrame with OHLCV data
            signal (dict): Signal information from signal generator
            entry_price (float): Entry price
            
        Returns:
            dict: Optimized exit points
        """
        # Calculate resistance levels
        resistance_levels = self._identify_resistance_levels(price_data)
        
        # Find the closest resistance level above entry price
        closest_resistance = None
        min_distance = float('inf')
        
        for level in resistance_levels:
            if level > entry_price:
                distance = level - entry_price
                if distance < min_distance:
                    min_distance = distance
                    closest_resistance = level
        
        # Calculate take profit levels
        if closest_resistance and (closest_resistance - entry_price) / entry_price > 0.02:
            # Use resistance level if it provides at least 2% profit
            take_profit = closest_resistance
        else:
            # Otherwise use default take profit calculation
            take_profit = signal['take_profit']
        
        # Calculate trailing take profit if enabled
        trailing_take_profit = None
        if self.trailing_take_profit_enabled:
            trailing_take_profit = {
                'enabled': True,
                'distance_percentage': self.trailing_take_profit_distance
            }
        
        # Calculate profit taking levels if enabled
        profit_taking_levels = []
        if self.profit_taking_enabled:
            # Simple profit taking at threshold
            level1 = entry_price * (1 + self.profit_taking_threshold / 100)
            
            # Additional level at 1.5x threshold
            level2 = entry_price * (1 + (self.profit_taking_threshold * 1.5) / 100)
            
            # Additional level at 2x threshold
            level3 = entry_price * (1 + (self.profit_taking_threshold * 2) / 100)
            
            profit_taking_levels = [
                {'price': level1, 'percentage': 33},  # Sell 33% at first level
                {'price': level2, 'percentage': 33},  # Sell 33% at second level
                {'price': level3, 'percentage': 34}   # Sell remaining 34% at third level
            ]
        
        return {
            'take_profit': take_profit,
            'stop_loss': signal['stop_loss'],
            'trailing_take_profit': trailing_take_profit,
            'profit_taking_levels': profit_taking_levels
        }
    
    def generate_profit_optimized_plan(self, price_data, signal):
        """
        Generate a profit-optimized trading plan based on the signal.
        
        Args:
            price_data (pd.DataFrame): DataFrame with OHLCV data
            signal (dict): Signal information from signal generator
            
        Returns:
            dict: Profit-optimized trading plan
        """
        # Only generate plan for actionable signals
        if signal['signal'] not in ['STRONG_BUY', 'BUY', 'STRONG_SELL', 'SELL']:
            return {
                'action': 'HOLD',
                'explanation': 'No clear trading opportunity at this time. Better to hold current position.'
            }
        
        # Determine action based on signal
        if signal['signal'] in ['STRONG_BUY', 'BUY']:
            action = 'BUY'
        else:
            action = 'SELL'
        
        # For buy signals, optimize entry and exit points
        if action == 'BUY':
            # Optimize entry points
            entry_points = self.optimize_entry_points(price_data, signal)
            
            # Optimize exit points based on primary entry
            exit_points = self.optimize_exit_points(price_data, signal, entry_points['primary_entry'])
            
            # Calculate potential profit
            potential_profit_pct = (exit_points['take_profit'] - entry_points['primary_entry']) / entry_points['primary_entry'] * 100
            
            # Generate explanation
            explanation = f"Buy at ${entry_points['primary_entry']:.2f} with a target of ${exit_points['take_profit']:.2f} "
            explanation += f"(potential profit: {potential_profit_pct:.2f}%). "
            explanation += f"Set stop loss at ${exit_points['stop_loss']:.2f}. "
            
            if entry_points['secondary_entries']:
                explanation += "Consider additional buys if price drops: "
                for i, entry in enumerate(entry_points['secondary_entries']):
                    explanation += f"${entry['price']:.2f} ({entry['percentage_drop']}% drop)"
                    if i < len(entry_points['secondary_entries']) - 1:
                        explanation += ", "
                explanation += ". "
            
            if exit_points['profit_taking_levels']:
                explanation += "Consider taking partial profits at: "
                for i, level in enumerate(exit_points['profit_taking_levels']):
                    explanation += f"${level['price']:.2f} (sell {level['percentage']}%)"
                    if i < len(exit_points['profit_taking_levels']) - 1:
                        explanation += ", "
                explanation += "."
            
            return {
                'action': action,
                'primary_entry': entry_points['primary_entry'],
                'secondary_entries': entry_points['secondary_entries'],
                'take_profit': exit_points['take_profit'],
                'stop_loss': exit_points['stop_loss'],
                'trailing_take_profit': exit_points['trailing_take_profit'],
                'profit_taking_levels': exit_points['profit_taking_levels'],
                'potential_profit_percentage': potential_profit_pct,
                'explanation': explanation
            }
        
        # For sell signals, provide simple exit strategy
        else:
            current_price = price_data['close'].iloc[-1]
            
            # Generate explanation
            explanation = f"Sell at market price (around ${current_price:.2f}). "
            explanation += "The signal indicates negative price movement ahead. "
            
            if self.trend_following_enabled:
                explanation += "This follows the current downward trend. "
            
            return {
                'action': action,
                'current_price': current_price,
                'explanation': explanation
            }
    
    def calculate_optimal_position_size(self, account_balance, risk_level, signal_strength, profit_potential):
        """
        Calculate optimal position size based on account balance, risk level, signal strength, and profit potential.
        
        Args:
            account_balance (float): Account balance
            risk_level (str): Risk level (Low, Medium, High)
            signal_strength (int): Signal strength (0-100)
            profit_potential (float): Potential profit percentage
            
        Returns:
            dict: Optimal position size information
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
        
        # Adjust based on profit potential
        profit_factor = min(2.0, max(0.5, profit_potential / 10))  # Scale profit potential
        
        # Calculate adjusted percentage
        adjusted_percentage = base_percentage * strength_factor * profit_factor
        
        # Ensure within reasonable limits
        adjusted_percentage = max(1.0, min(adjusted_percentage, 10.0))
        
        # Calculate dollar amount
        dollar_amount = account_balance * (adjusted_percentage / 100)
        
        return {
            'percentage': adjusted_percentage,
            'dollar_amount': dollar_amount,
            'explanation': f"Recommended position size is {adjusted_percentage:.1f}% of your account (${dollar_amount:.2f}). "
                          f"This is based on {risk_level} risk level, {signal_strength}% signal strength, "
                          f"and {profit_potential:.1f}% profit potential."
        }
    
    def _identify_support_levels(self, price_data):
        """Identify support levels from price data."""
        df = price_data.copy()
        
        # Look for local minimums
        support_levels = []
        
        for i in range(2, len(df) - 2):
            if (df['low'].iloc[i] < df['low'].iloc[i-1] and 
                df['low'].iloc[i] < df['low'].iloc[i-2] and 
                df['low'].iloc[i] < df['low'].iloc[i+1] and 
                df['low'].iloc[i] < df['low'].iloc[i+2]):
                support_levels.append(df['low'].iloc[i])
        
        # If not enough support levels found, use recent lows
        if len(support_levels) < 3:
            recent_lows = df['low'].rolling(window=5).min().dropna()
            support_levels.extend(recent_lows.tolist()[-3:])
        
        # Remove duplicates and sort
        support_levels = sorted(list(set(support_levels)))
        
        return support_levels
    
    def _identify_resistance_levels(self, price_data):
        """Identify resistance levels from price data."""
        df = price_data.copy()
        
        # Look for local maximums
        resistance_levels = []
        
        for i in range(2, len(df) - 2):
            if (df['high'].iloc[i] > df['high'].iloc[i-1] and 
                df['high'].iloc[i] > df['high'].iloc[i-2] and 
                df['high'].iloc[i] > df['high'].iloc[i+1] and 
                df['high'].iloc[i] > df['high'].iloc[i+2]):
                resistance_levels.append(df['high'].iloc[i])
        
        # If not enough resistance levels found, use recent highs
        if len(resistance_levels) < 3:
            recent_highs = df['high'].rolling(window=5).max().dropna()
            resistance_levels.extend(recent_highs.tolist()[-3:])
        
        # Remove duplicates and sort
        resistance_levels = sorted(list(set(resistance_levels)))
        
        return resistance_levels
    
    def get_profit_optimization_tips(self, signal):
        """
        Get profit optimization tips based on the current signal.
        
        Args:
            signal (dict): Signal information from signal generator
            
        Returns:
            list: List of profit optimization tips
        """
        tips = []
        
        if signal['signal'] in ['STRONG_BUY', 'BUY']:
            tips.append("Consider placing a limit order slightly below the current price to get a better entry.")
            tips.append("Set up multiple take-profit orders at different levels to secure profits as price rises.")
            tips.append("Use a trailing stop loss to protect profits while allowing room for price to grow.")
            
            if self.dca_enabled:
                tips.append("Prepare additional buy orders at lower prices to average down if price dips temporarily.")
        
        elif signal['signal'] in ['STRONG_SELL', 'SELL']:
            tips.append("Consider selling in portions rather than all at once to manage risk.")
            tips.append("After selling, set buy orders at lower levels to potentially rebuy at better prices.")
            tips.append("Monitor the market for a potential trend reversal before re-entering.")
        
        else:  # HOLD
            tips.append("Use this time to review your trading plan and prepare for the next opportunity.")
            tips.append("Consider setting price alerts for both upward and downward breakouts.")
            tips.append("Review your portfolio allocation and ensure it aligns with your risk tolerance.")
        
        # General tips
        tips.append("Remember that consistent small profits compound over time to significant gains.")
        tips.append("Avoid chasing quick profits with large position sizes, as this increases risk.")
        
        return tips
