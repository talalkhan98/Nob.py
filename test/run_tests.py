import sys
import os
import unittest

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import test modules
from test_components import TestSignalGenerator, TestProfitOptimization, TestLossPrevention
from test_integration import TestUI, TestIntegration

if __name__ == '__main__':
    # Create a test suite
    test_suite = unittest.TestSuite()
    
    # Add tests from test_components.py
    test_suite.addTest(unittest.makeSuite(TestSignalGenerator))
    test_suite.addTest(unittest.makeSuite(TestProfitOptimization))
    test_suite.addTest(unittest.makeSuite(TestLossPrevention))
    
    # Add tests from test_integration.py
    test_suite.addTest(unittest.makeSuite(TestUI))
    test_suite.addTest(unittest.makeSuite(TestIntegration))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(test_suite)
