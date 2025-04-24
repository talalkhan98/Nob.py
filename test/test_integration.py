import unittest
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules
from ui.dashboard import SignalDashboard
from ui.beginner_features import BeginnerFeatures

class TestUI(unittest.TestCase):
    """Test the UI components."""
    
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
            
            def get_position_size_recommendation(self, account_balance, risk_level, signal_strength):
                return 5.0  # 5% of account balance
        
        self.signal_generator = MockSignalGenerator()
        self.dashboard = SignalDashboard(self.signal_generator)
        self.beginner_features = BeginnerFeatures()
    
    def test_dashboard_initialization(self):
        """Test dashboard initialization."""
        # Check dashboard attributes
        self.assertEqual(self.dashboard.signal_generator, self.signal_generator)
        self.assertIsInstance(self.dashboard.supported_cryptos, list)
        self.assertIsInstance(self.dashboard.timeframes, list)
        self.assertIsInstance(self.dashboard.colors, dict)
    
    def test_beginner_features_initialization(self):
        """Test beginner features initialization."""
        # Check beginner features attributes
        self.assertIsInstance(self.beginner_features.colors, dict)
    
    def test_format_signal(self):
        """Test format_signal method."""
        # Test all signal types
        self.assertEqual(self.dashboard._format_signal("STRONG_BUY"), "Strong Buy")
        self.assertEqual(self.dashboard._format_signal("BUY"), "Buy")
        self.assertEqual(self.dashboard._format_signal("STRONG_SELL"), "Strong Sell")
        self.assertEqual(self.dashboard._format_signal("SELL"), "Sell")
        self.assertEqual(self.dashboard._format_signal("HOLD"), "Hold")
        self.assertEqual(self.dashboard._format_signal("UNKNOWN"), "UNKNOWN")
    
    def test_get_signal_class(self):
        """Test get_signal_class method."""
        # Test all signal types
        self.assertEqual(self.dashboard._get_signal_class("STRONG_BUY"), "strong-buy-signal")
        self.assertEqual(self.dashboard._get_signal_class("BUY"), "buy-signal")
        self.assertEqual(self.dashboard._get_signal_class("STRONG_SELL"), "strong-sell-signal")
        self.assertEqual(self.dashboard._get_signal_class("SELL"), "sell-signal")
        self.assertEqual(self.dashboard._get_signal_class("HOLD"), "hold-signal")
        self.assertEqual(self.dashboard._get_signal_class("UNKNOWN"), "")
    
    def test_render_strength_meter(self):
        """Test render_strength_meter method."""
        # Test different strength values
        self.assertEqual(self.dashboard._render_strength_meter(100), "★★★★★ (100%)")
        self.assertEqual(self.dashboard._render_strength_meter(80), "★★★★☆ (80%)")
        self.assertEqual(self.dashboard._render_strength_meter(60), "★★★☆☆ (60%)")
        self.assertEqual(self.dashboard._render_strength_meter(40), "★★☆☆☆ (40%)")
        self.assertEqual(self.dashboard._render_strength_meter(20), "★☆☆☆☆ (20%)")
        self.assertEqual(self.dashboard._render_strength_meter(0), "☆☆☆☆☆ (0%)")

class TestIntegration(unittest.TestCase):
    """Test the integration of all components."""
    
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
        
        # Import modules
        from signal_generator import CryptoSignalGenerator
        from utils.profit_optimization import ProfitOptimizationStrategies
        from utils.loss_prevention import LossPreventionSafeguards
        
        # Initialize components
        self.data_fetcher = MockDataFetcher()
        self.signal_generator = CryptoSignalGenerator(self.data_fetcher)
        self.profit_optimizer = ProfitOptimizationStrategies(self.signal_generator)
        self.loss_prevention = LossPreventionSafeguards(self.signal_generator)
        
        # Connect components
        self.signal_generator.set_loss_prevention(self.loss_prevention)
        self.signal_generator.set_profit_optimizer(self.profit_optimizer)
    
    def test_end_to_end_signal_generation(self):
        """Test end-to-end signal generation process."""
        # Generate sample data
        price_data = self.signal_generator.generate_sample_data("BTC/USDT", days=30)
        
        # Generate signal
        signal = self.signal_generator.generate_signals(price_data)
        
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
    
    def test_profit_optimization_integration(self):
        """Test profit optimization integration."""
        # Generate sample data
        price_data = self.signal_generator.generate_sample_data("BTC/USDT", days=30)
        
        # Generate signal
        signal = self.signal_generator.generate_signals(price_data)
        
        # Generate profit-optimized plan
        plan = self.profit_optimizer.generate_profit_optimized_plan(price_data, signal)
        
        # Check plan structure
        self.assertIsInstance(plan, dict)
        self.assertTrue('action' in plan)
        self.assertTrue('explanation' in plan)
        
        # Check plan values
        self.assertIn(plan['action'], ['BUY', 'SELL', 'HOLD'])
        self.assertIsInstance(plan['explanation'], str)
        self.assertGreater(len(plan['explanation']), 0)
    
    def test_loss_prevention_integration(self):
        """Test loss prevention integration."""
        # Generate sample data
        price_data = self.signal_generator.generate_sample_data("BTC/USDT", days=30)
        
        # Generate signal
        signal = self.signal_generator.generate_signals(price_data)
        
        # Apply safeguards
        safe_signal = self.loss_prevention.apply_safeguards(price_data, signal, 1000, 0)
        
        # Check safe signal structure
        self.assertIsInstance(safe_signal, dict)
        self.assertTrue('signal' in safe_signal)
        self.assertTrue('explanation' in safe_signal)
        
        # Check safe signal values
        self.assertIn(safe_signal['signal'], ['STRONG_BUY', 'BUY', 'HOLD', 'SELL', 'STRONG_SELL'])
        self.assertIsInstance(safe_signal['explanation'], str)

if __name__ == '__main__':
    unittest.main()
