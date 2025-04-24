import unittest
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules
from signal_generator import CryptoSignalGenerator
from utils.profit_optimization import ProfitOptimizationStrategies
from utils.loss_prevention import LossPreventionSafeguards

class TestSignalGenerator(unittest.TestCase):
    """Test the CryptoSignalGenerator class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a mock data fetcher
        class MockDataFetcher:
            def fetch_historical_data(self, symbol, timeframe, limit):
                # Generate sample data
                dates = pd.date_range(end=datetime.now(), periods=limit)
                data = {
                    'open': np.random.normal(50000, 1000, limit),
                    'high': np.random.normal(51000, 1000, limit),
                    'low': np.random.normal(49000, 1000, limit),
                    'close': np.random.normal(50500, 1000, limit),
                    'volume': np.random.normal(1000000, 100000, limit)
                }
                df = pd.DataFrame(data, index=dates)
                return df
        
        self.data_fetcher = MockDataFetcher()
        self.signal_generator = CryptoSignalGenerator(self.data_fetcher)
    
    def test_generate_sample_data(self):
        """Test generate_sample_data method."""
        data = self.signal_generator.generate_sample_data("BTC/USDT", days=30)
        
        # Check data structure
        self.assertIsInstance(data, pd.DataFrame)
        self.assertTrue('open' in data.columns)
        self.assertTrue('high' in data.columns)
        self.assertTrue('low' in data.columns)
        self.assertTrue('close' in data.columns)
        self.assertTrue('volume' in data.columns)
        
        # Check data length
        self.assertEqual(len(data), 30)
    
    def test_generate_signals(self):
        """Test generate_signals method."""
        data = self.signal_generator.generate_sample_data("BTC/USDT", days=30)
        signal = self.signal_generator.generate_signals(data)
        
        # Check signal structure
        self.assertIsInstance(signal, dict)
        self.assertTrue('signal' in signal)
        self.assertTrue('strength' in signal)
        self.assertTrue('explanation' in signal)
        self.assertTrue('entry_price' in signal)
        self.assertTrue('stop_loss' in signal)
        self.assertTrue('take_profit' in signal)
        self.assertTrue('risk_level' in signal)
        self.assertTrue('last_price' in signal)
        
        # Check signal values
        self.assertIn(signal['signal'], ['STRONG_BUY', 'BUY', 'HOLD', 'SELL', 'STRONG_SELL'])
        self.assertTrue(0 <= signal['strength'] <= 100)
        self.assertIsInstance(signal['explanation'], str)
        self.assertGreater(len(signal['explanation']), 0)
        self.assertIsInstance(signal['entry_price'], float)
        self.assertIsInstance(signal['stop_loss'], float)
        self.assertIsInstance(signal['take_profit'], float)
        self.assertIn(signal['risk_level'], ['Low', 'Medium', 'High'])
        self.assertIsInstance(signal['last_price'], float)
    
    def test_get_position_size_recommendation(self):
        """Test get_position_size_recommendation method."""
        position_size = self.signal_generator.get_position_size_recommendation(1000, "Low", 80)
        
        # Check position size
        self.assertIsInstance(position_size, float)
        self.assertTrue(0 < position_size <= 10)  # Should be a reasonable percentage

class TestProfitOptimization(unittest.TestCase):
    """Test the ProfitOptimizationStrategies class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a mock signal generator
        class MockSignalGenerator:
            def generate_sample_data(self, symbol, days=30):
                # Generate sample data
                dates = pd.date_range(end=datetime.now(), periods=days)
                data = {
                    'open': np.random.normal(50000, 1000, days),
                    'high': np.random.normal(51000, 1000, days),
                    'low': np.random.normal(49000, 1000, days),
                    'close': np.random.normal(50500, 1000, days),
                    'volume': np.random.normal(1000000, 100000, days)
                }
                df = pd.DataFrame(data, index=dates)
                return df
            
            def generate_signals(self, price_data):
                return {
                    'signal': 'BUY',
                    'strength': 75,
                    'explanation': 'Test signal',
                    'entry_price': price_data['close'].iloc[-1],
                    'stop_loss': price_data['close'].iloc[-1] * 0.95,
                    'take_profit': price_data['close'].iloc[-1] * 1.05,
                    'risk_level': 'Low',
                    'last_price': price_data['close'].iloc[-1]
                }
        
        self.signal_generator = MockSignalGenerator()
        self.profit_optimizer = ProfitOptimizationStrategies(self.signal_generator)
    
    def test_optimize_entry_points(self):
        """Test optimize_entry_points method."""
        price_data = self.signal_generator.generate_sample_data("BTC/USDT", days=30)
        signal = self.signal_generator.generate_signals(price_data)
        
        entry_points = self.profit_optimizer.optimize_entry_points(price_data, signal)
        
        # Check entry points structure
        self.assertIsInstance(entry_points, dict)
        self.assertTrue('primary_entry' in entry_points)
        self.assertTrue('secondary_entries' in entry_points)
        
        # Check entry points values
        self.assertIsInstance(entry_points['primary_entry'], float)
        self.assertIsInstance(entry_points['secondary_entries'], list)
    
    def test_optimize_exit_points(self):
        """Test optimize_exit_points method."""
        price_data = self.signal_generator.generate_sample_data("BTC/USDT", days=30)
        signal = self.signal_generator.generate_signals(price_data)
        
        exit_points = self.profit_optimizer.optimize_exit_points(price_data, signal, signal['entry_price'])
        
        # Check exit points structure
        self.assertIsInstance(exit_points, dict)
        self.assertTrue('take_profit' in exit_points)
        self.assertTrue('stop_loss' in exit_points)
        self.assertTrue('trailing_take_profit' in exit_points)
        self.assertTrue('profit_taking_levels' in exit_points)
        
        # Check exit points values
        self.assertIsInstance(exit_points['take_profit'], float)
        self.assertIsInstance(exit_points['stop_loss'], float)
        if exit_points['trailing_take_profit']:
            self.assertIsInstance(exit_points['trailing_take_profit'], dict)
        self.assertIsInstance(exit_points['profit_taking_levels'], list)
    
    def test_generate_profit_optimized_plan(self):
        """Test generate_profit_optimized_plan method."""
        price_data = self.signal_generator.generate_sample_data("BTC/USDT", days=30)
        signal = self.signal_generator.generate_signals(price_data)
        
        plan = self.profit_optimizer.generate_profit_optimized_plan(price_data, signal)
        
        # Check plan structure
        self.assertIsInstance(plan, dict)
        self.assertTrue('action' in plan)
        self.assertTrue('explanation' in plan)
        
        # Check plan values
        self.assertIn(plan['action'], ['BUY', 'SELL', 'HOLD'])
        self.assertIsInstance(plan['explanation'], str)
        self.assertGreater(len(plan['explanation']), 0)

class TestLossPrevention(unittest.TestCase):
    """Test the LossPreventionSafeguards class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a mock signal generator
        class MockSignalGenerator:
            def generate_sample_data(self, symbol, days=30):
                # Generate sample data
                dates = pd.date_range(end=datetime.now(), periods=days)
                data = {
                    'open': np.random.normal(50000, 1000, days),
                    'high': np.random.normal(51000, 1000, days),
                    'low': np.random.normal(49000, 1000, days),
                    'close': np.random.normal(50500, 1000, days),
                    'volume': np.random.normal(1000000, 100000, days)
                }
                df = pd.DataFrame(data, index=dates)
                return df
            
            def generate_signals(self, price_data):
                return {
                    'signal': 'BUY',
                    'strength': 75,
                    'explanation': 'Test signal',
                    'entry_price': price_data['close'].iloc[-1],
                    'stop_loss': price_data['close'].iloc[-1] * 0.95,
                    'take_profit': price_data['close'].iloc[-1] * 1.05,
                    'risk_level': 'Low',
                    'last_price': price_data['close'].iloc[-1]
                }
        
        self.signal_generator = MockSignalGenerator()
        self.loss_prevention = LossPreventionSafeguards(self.signal_generator)
    
    def test_apply_safeguards(self):
        """Test apply_safeguards method."""
        price_data = self.signal_generator.generate_sample_data("BTC/USDT", days=30)
        signal = self.signal_generator.generate_signals(price_data)
        
        safe_signal = self.loss_prevention.apply_safeguards(price_data, signal, 1000, 0)
        
        # Check safe signal structure
        self.assertIsInstance(safe_signal, dict)
        self.assertTrue('signal' in safe_signal)
        self.assertTrue('explanation' in safe_signal)
        
        # Check safe signal values
        self.assertIn(safe_signal['signal'], ['STRONG_BUY', 'BUY', 'HOLD', 'SELL', 'STRONG_SELL'])
        self.assertIsInstance(safe_signal['explanation'], str)
    
    def test_get_max_safe_position_size(self):
        """Test get_max_safe_position_size method."""
        price_data = self.signal_generator.generate_sample_data("BTC/USDT", days=30)
        signal = self.signal_generator.generate_signals(price_data)
        
        position_size = self.loss_prevention.get_max_safe_position_size(price_data, signal, 1000)
        
        # Check position size structure
        self.assertIsInstance(position_size, dict)
        self.assertTrue('position_size' in position_size)
        self.assertTrue('percentage' in position_size)
        self.assertTrue('explanation' in position_size)
        
        # Check position size values
        self.assertIsInstance(position_size['position_size'], float)
        self.assertIsInstance(position_size['percentage'], float)
        self.assertIsInstance(position_size['explanation'], str)
        self.assertTrue(0 <= position_size['percentage'] <= 100)
    
    def test_check_market_conditions(self):
        """Test check_market_conditions method."""
        price_data = self.signal_generator.generate_sample_data("BTC/USDT", days=60)
        
        conditions = self.loss_prevention.check_market_conditions(price_data)
        
        # Check conditions structure
        self.assertIsInstance(conditions, dict)
        self.assertTrue('risk_level' in conditions)
        self.assertTrue('risk_explanation' in conditions)
        self.assertTrue('volatility' in conditions)
        self.assertTrue('trend_alignment' in conditions)
        self.assertTrue('trend_explanation' in conditions)
        self.assertTrue('volume_alert' in conditions)
        self.assertTrue('volume_explanation' in conditions)
        self.assertTrue('overall_assessment' in conditions)
        
        # Check conditions values
        self.assertIn(conditions['risk_level'], ['Low', 'Medium', 'High'])
        self.assertIsInstance(conditions['risk_explanation'], str)
        self.assertIsInstance(conditions['volatility'], float)
        self.assertIn(conditions['trend_alignment'], ['Strong', 'Moderate', 'Weak'])
        self.assertIsInstance(conditions['trend_explanation'], str)
        self.assertIn(conditions['volume_alert'], ['Normal', 'High', 'Low', 'Very High', 'Very Low'])
        self.assertIsInstance(conditions['volume_explanation'], str)
        self.assertIsInstance(conditions['overall_assessment'], str)

if __name__ == '__main__':
    unittest.main()
