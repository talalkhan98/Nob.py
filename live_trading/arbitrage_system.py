import ccxt
import pandas as pd
import numpy as np
import time
import threading
import os
import json
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor

class ArbitrageSystem:
    """
    Advanced cross-exchange arbitrage system that identifies and executes
    profitable trading opportunities across multiple exchanges.
    """
    
    def __init__(self):
        """Initialize the ArbitrageSystem."""
        self.exchanges = {}
        self.market_data = {}
        self.opportunities = []
        self.active_arbitrages = []
        self.running = False
        self.thread = None
        self.min_profit_threshold = 0.5  # Minimum profit percentage
        self.max_execution_time = 5  # Maximum execution time in seconds
        self.max_slippage = 0.2  # Maximum allowed slippage percentage
        self.balance_cache = {}
        self.balance_cache_time = {}
        self.order_book_cache = {}
        self.order_book_cache_time = {}
        
        # Create logs directory if it doesn't exist
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        self.log_file = os.path.join(logs_dir, 'arbitrage.log')
        
        # Configure logging
        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('arbitrage')
    
    def initialize_exchanges(self, exchange_configs, test_mode=True):
        """
        Initialize exchange connections.
        
        Args:
            exchange_configs (list): List of exchange configuration dictionaries
            test_mode (bool): Whether to use test mode
        
        Returns:
            bool: Success status
        """
        try:
            for config in exchange_configs:
                exchange_id = config.get('id')
                api_key = config.get('api_key')
                api_secret = config.get('api_secret')
                
                if not exchange_id:
                    self.logger.error("Exchange ID is required")
                    continue
                
                # Initialize exchange
                exchange_class = getattr(ccxt, exchange_id)
                exchange_options = {
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'spot'
                    }
                }
                
                # Add API credentials if provided and not in test mode
                if api_key and api_secret and not test_mode:
                    exchange_options['apiKey'] = api_key
                    exchange_options['secret'] = api_secret
                
                exchange = exchange_class(exchange_options)
                
                # Store exchange
                self.exchanges[exchange_id] = {
                    'instance': exchange,
                    'markets': None,
                    'fees': config.get('fees', {'maker': 0.1, 'taker': 0.1}),
                    'transfer_time': config.get('transfer_time', 60),  # seconds
                    'min_order_amount': config.get('min_order_amount', 10),  # USD
                    'enabled': True
                }
                
                self.logger.info(f"Initialized {exchange_id} exchange connection")
            
            # Load markets for all exchanges
            self._load_markets()
            
            return True
        except Exception as e:
            self.logger.error(f"Error initializing exchanges: {str(e)}")
            return False
    
    def _load_markets(self):
        """Load markets for all exchanges."""
        for exchange_id, exchange_data in self.exchanges.items():
            try:
                exchange = exchange_data['instance']
                markets = exchange.load_markets()
                self.exchanges[exchange_id]['markets'] = markets
                self.logger.info(f"Loaded {len(markets)} markets for {exchange_id}")
            except Exception as e:
                self.logger.error(f"Error loading markets for {exchange_id}: {str(e)}")
                self.exchanges[exchange_id]['enabled'] = False
    
    def start_arbitrage_scanner(self, symbols=None, scan_interval=10):
        """
        Start the arbitrage scanner.
        
        Args:
            symbols (list): List of symbols to scan (e.g., ['BTC/USDT', 'ETH/USDT'])
            scan_interval (int): Scan interval in seconds
        
        Returns:
            bool: Success status
        """
        if self.running:
            self.logger.info("Arbitrage scanner is already running")
            return False
        
        # If no symbols provided, use common ones
        if not symbols:
            symbols = ['BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'SOL/USDT', 'ADA/USDT']
        
        self.symbols = symbols
        self.scan_interval = scan_interval
        self.running = True
        
        # Start scanner thread
        self.thread = threading.Thread(target=self._scanner_loop)
        self.thread.daemon = True
        self.thread.start()
        
        self.logger.info(f"Started arbitrage scanner for {', '.join(symbols)}")
        return True
    
    def stop_arbitrage_scanner(self):
        """Stop the arbitrage scanner."""
        if not self.running:
            self.logger.info("Arbitrage scanner is not running")
            return False
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        
        self.logger.info("Stopped arbitrage scanner")
        return True
    
    def _scanner_loop(self):
        """Internal scanner loop."""
        while self.running:
            try:
                # Scan for arbitrage opportunities
                self._scan_arbitrage_opportunities()
                
                # Execute profitable opportunities if not in test mode
                self._execute_arbitrage_opportunities()
                
                # Sleep until next scan
                time.sleep(self.scan_interval)
            except Exception as e:
                self.logger.error(f"Error in scanner loop: {str(e)}")
                time.sleep(self.scan_interval)
    
    def _scan_arbitrage_opportunities(self):
        """Scan for arbitrage opportunities across exchanges."""
        # Clear previous opportunities
        self.opportunities = []
        
        # Get enabled exchanges
        enabled_exchanges = [
            exchange_id for exchange_id, exchange_data in self.exchanges.items()
            if exchange_data['enabled']
        ]
        
        if len(enabled_exchanges) < 2:
            self.logger.warning("Need at least 2 enabled exchanges for arbitrage")
            return
        
        # Scan each symbol
        for symbol in self.symbols:
            # Get ticker data from all exchanges
            ticker_data = {}
            
            for exchange_id in enabled_exchanges:
                try:
                    ticker = self._get_ticker(exchange_id, symbol)
                    if ticker:
                        ticker_data[exchange_id] = ticker
                except Exception as e:
                    self.logger.error(f"Error getting ticker for {symbol} on {exchange_id}: {str(e)}")
            
            # Need at least 2 exchanges with data
            if len(ticker_data) < 2:
                continue
            
            # Find arbitrage opportunities
            opportunities = self._find_arbitrage_opportunities(symbol, ticker_data)
            self.opportunities.extend(opportunities)
        
        # Log opportunities
        if self.opportunities:
            self.logger.info(f"Found {len(self.opportunities)} arbitrage opportunities")
            for opp in self.opportunities:
                self.logger.info(f"Opportunity: {opp['symbol']} - Buy on {opp['buy_exchange']} at {opp['buy_price']}, "
                               f"Sell on {opp['sell_exchange']} at {opp['sell_price']}, "
                               f"Profit: {opp['profit_pct']:.2f}%")
    
    def _get_ticker(self, exchange_id, symbol):
        """
        Get ticker data for a symbol on an exchange.
        
        Args:
            exchange_id (str): Exchange ID
            symbol (str): Symbol to get ticker for
        
        Returns:
            dict: Ticker data
        """
        exchange = self.exchanges[exchange_id]['instance']
        
        try:
            # Check if symbol is available on this exchange
            if symbol not in exchange.markets:
                return None
            
            # Get ticker
            ticker = exchange.fetch_ticker(symbol)
            
            # Extract relevant data
            return {
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'last': ticker['last'],
                'volume': ticker['quoteVolume'],
                'timestamp': ticker['timestamp']
            }
        except Exception as e:
            self.logger.error(f"Error fetching ticker for {symbol} on {exchange_id}: {str(e)}")
            return None
    
    def _find_arbitrage_opportunities(self, symbol, ticker_data):
        """
        Find arbitrage opportunities for a symbol across exchanges.
        
        Args:
            symbol (str): Symbol to find opportunities for
            ticker_data (dict): Ticker data for the symbol on different exchanges
        
        Returns:
            list: List of arbitrage opportunities
        """
        opportunities = []
        
        # Get exchange IDs
        exchange_ids = list(ticker_data.keys())
        
        # Compare each pair of exchanges
        for i in range(len(exchange_ids)):
            for j in range(i + 1, len(exchange_ids)):
                buy_exchange = exchange_ids[i]
                sell_exchange = exchange_ids[j]
                
                # Get ticker data
                buy_ticker = ticker_data[buy_exchange]
                sell_ticker = ticker_data[sell_exchange]
                
                # Calculate potential profits in both directions
                
                # Direction 1: Buy on exchange i, sell on exchange j
                buy_price_1 = buy_ticker['ask']
                sell_price_1 = sell_ticker['bid']
                
                # Calculate fees
                buy_fee_1 = buy_price_1 * (self.exchanges[buy_exchange]['fees']['taker'] / 100)
                sell_fee_1 = sell_price_1 * (self.exchanges[sell_exchange]['fees']['taker'] / 100)
                
                # Calculate profit
                profit_1 = sell_price_1 - buy_price_1 - buy_fee_1 - sell_fee_1
                profit_pct_1 = (profit_1 / buy_price_1) * 100
                
                # Direction 2: Buy on exchange j, sell on exchange i
                buy_price_2 = sell_ticker['ask']
                sell_price_2 = buy_ticker['bid']
                
                # Calculate fees
                buy_fee_2 = buy_price_2 * (self.exchanges[sell_exchange]['fees']['taker'] / 100)
                sell_fee_2 = sell_price_2 * (self.exchanges[buy_exchange]['fees']['taker'] / 100)
                
                # Calculate profit
                profit_2 = sell_price_2 - buy_price_2 - buy_fee_2 - sell_fee_2
                profit_pct_2 = (profit_2 / buy_price_2) * 100
                
                # Check if either direction is profitable
                if profit_pct_1 > self.min_profit_threshold:
                    opportunities.append({
                        'symbol': symbol,
                        'buy_exchange': buy_exchange,
                        'sell_exchange': sell_exchange,
                        'buy_price': buy_price_1,
                        'sell_price': sell_price_1,
                        'profit': profit_1,
                        'profit_pct': profit_pct_1,
                        'timestamp': datetime.now(),
                        'direction': 1,
                        'status': 'identified'
                    })
                
                if profit_pct_2 > self.min_profit_threshold:
                    opportunities.append({
                        'symbol': symbol,
                        'buy_exchange': sell_exchange,
                        'sell_exchange': buy_exchange,
                        'buy_price': buy_price_2,
                        'sell_price': sell_price_2,
                        'profit': profit_2,
                        'profit_pct': profit_pct_2,
                        'timestamp': datetime.now(),
                        'direction': 2,
                        'status': 'identified'
                    })
        
        return opportunities
    
    def _execute_arbitrage_opportunities(self):
        """Execute profitable arbitrage opportunities."""
        # Skip if no opportunities
        if not self.opportunities:
            return
        
        # Sort opportunities by profit percentage (descending)
        sorted_opportunities = sorted(
            self.opportunities, 
            key=lambda x: x['profit_pct'], 
            reverse=True
        )
        
        # Execute top opportunities
        for opportunity in sorted_opportunities[:3]:  # Limit to top 3
            # Skip if already being executed
            if opportunity['status'] != 'identified':
                continue
            
            # Mark as being analyzed
            opportunity['status'] = 'analyzing'
            
            # Verify opportunity with order book data
            verified = self._verify_opportunity(opportunity)
            
            if not verified:
                opportunity['status'] = 'rejected'
                continue
            
            # Execute arbitrage (in a real system, this would place actual orders)
            success = self._simulate_arbitrage_execution(opportunity)
            
            if success:
                opportunity['status'] = 'executed'
                self.active_arbitrages.append(opportunity)
                self.logger.info(f"Executed arbitrage: {opportunity['symbol']} - "
                               f"Buy on {opportunity['buy_exchange']} at {opportunity['buy_price']}, "
                               f"Sell on {opportunity['sell_exchange']} at {opportunity['sell_price']}, "
                               f"Profit: {opportunity['profit_pct']:.2f}%")
            else:
                opportunity['status'] = 'failed'
    
    def _verify_opportunity(self, opportunity):
        """
        Verify an arbitrage opportunity with order book data.
        
        Args:
            opportunity (dict): Arbitrage opportunity to verify
        
        Returns:
            bool: Whether the opportunity is still valid
        """
        symbol = opportunity['symbol']
        buy_exchange = opportunity['buy_exchange']
        sell_exchange = opportunity['sell_exchange']
        
        try:
            # Get order books
            buy_order_book = self._get_order_book(buy_exchange, symbol)
            sell_order_book = self._get_order_book(sell_exchange, symbol)
            
            if not buy_order_book or not sell_order_book:
                return False
            
            # Get best prices
            best_ask = buy_order_book['asks'][0][0]
            best_bid = sell_order_book['bids'][0][0]
            
            # Calculate fees
            buy_fee = best_ask * (self.exchanges[buy_exchange]['fees']['taker'] / 100)
            sell_fee = best_bid * (self.exchanges[sell_exchange]['fees']['taker'] / 100)
            
            # Calculate profit
            profit = best_bid - best_ask - buy_fee - sell_fee
            profit_pct = (profit / best_ask) * 100
            
            # Check if still profitable
            if profit_pct < self.min_profit_threshold:
                return False
            
            # Check liquidity
            min_order_amount = min(
                self.exchanges[buy_exchange]['min_order_amount'],
                self.exchanges[sell_exchange]['min_order_amount']
            )
            
            # Convert to base currency amount
            base_amount = min_order_amount / best_ask
            
            # Check if enough liquidity on both sides
            buy_liquidity = sum([order[1] for order in buy_order_book['asks'][:5]])
            sell_liquidity = sum([order[1] for order in sell_order_book['bids'][:5]])
            
            if base_amount > buy_liquidity or base_amount > sell_liquidity:
                return False
            
            # Update opportunity with verified data
            opportunity['verified_buy_price'] = best_ask
            opportunity['verified_sell_price'] = best_bid
            opportunity['verified_profit'] = profit
            opportunity['verified_profit_pct'] = profit_pct
            opportunity['verified_amount'] = base_amount
            
            return True
        except Exception as e:
            self.logger.error(f"Error verifying opportunity: {str(e)}")
            return False
    
    def _get_order_book(self, exchange_id, symbol, force_refresh=False):
        """
        Get order book for a symbol on an exchange.
        
        Args:
            exchange_id (str): Exchange ID
            symbol (str): Symbol to get order book for
            force_refresh (bool): Whether to force refresh the cache
        
        Returns:
            dict: Order book data
        """
        cache_key = f"{exchange_id}_{symbol}"
        current_time = time.time()
        
        # Check cache
        if not force_refresh and cache_key in self.order_book_cache:
            cache_time = self.order_book_cache_time.get(cache_key, 0)
            if current_time - cache_time < 5:  # Cache valid for 5 seconds
                return self.order_book_cache[cache_key]
        
        try:
            exchange = self.exchanges[exchange_id]['instance']
            order_book = exchange.fetch_order_book(symbol, limit=20)
            
            # Cache result
            self.order_book_cache[cache_key] = order_book
            self.order_book_cache_time[cache_key] = current_time
            
            return order_book
        except Exception as e:
            self.logger.error(f"Error fetching order book for {symbol} on {exchange_id}: {str(e)}")
            return None
    
    def _simulate_arbitrage_execution(self, opportunity):
        """
        Simulate arbitrage execution (for testing).
        
        Args:
            opportunity (dict): Arbitrage opportunity to execute
        
        Returns:
            bool: Success status
        """
        # In a real system, this would place actual orders
        # For now, just simulate execution
        
        # Add execution details
        opportunity['execution_time'] = datetime.now()
        opportunity['execution_buy_price'] = opportunity.get('verified_buy_price', opportunity['buy_price'])
        opportunity['execution_sell_price'] = opportunity.get('verified_sell_price', opportunity['sell_price'])
        opportunity['execution_amount'] = opportunity.get('verified_amount', 0.01)  # Default small amount
        
        # Calculate actual profit
        buy_cost = opportunity['execution_buy_price'] * opportunity['execution_amount']
        sell_revenue = opportunity['execution_sell_price'] * opportunity['execution_amount']
        
        # Calculate fees
        buy_fee = buy_cost * (self.exchanges[opportunity['buy_exchange']]['fees']['taker'] / 100)
        sell_fee = sell_revenue * (self.exchanges[opportunity['sell_exchange']]['fees']['taker'] / 100)
        
        # Calculate profit
        profit = sell_revenue - buy_cost - buy_fee - sell_fee
        profit_pct = (profit / buy_cost) * 100
        
        # Update opportunity with execution results
        opportunity['execution_profit'] = profit
        opportunity['execution_profit_pct'] = profit_pct
        
        # Simulate success (95% success rate)
        return np.random.random() < 0.95
    
    def get_arbitrage_opportunities(self):
        """
        Get current arbitrage opportunities.
        
        Returns:
            list: List of arbitrage opportunities
        """
        return self.opportunities
    
    def get_active_arbitrages(self):
        """
        Get active arbitrage trades.
        
        Returns:
            list: List of active arbitrage trades
        """
        return self.active_arbitrages
    
    def get_arbitrage_history(self):
        """
        Get arbitrage trade history.
        
        Returns:
            list: List of completed arbitrage trades
        """
        # Filter completed arbitrages
        return [arb for arb in self.active_arbitrages if arb['status'] == 'executed']
    
    def get_arbitrage_performance(self):
        """
        Get arbitrage performance metrics.
        
        Returns:
            dict: Performance metrics
        """
        # Get completed arbitrages
        completed = [arb for arb in self.active_arbitrages if arb['status'] == 'executed']
        
        if not completed:
            return {
                'total_trades': 0,
                'total_profit': 0,
                'avg_profit_pct': 0,
                'success_rate': 0,
                'total_volume': 0
            }
        
        # Calculate metrics
        total_trades = len(completed)
        total_profit = sum(arb.get('execution_profit', 0) for arb in completed)
        avg_profit_pct = sum(arb.get('execution_profit_pct', 0) for arb in completed) / total_trades
        success_count = sum(1 for arb in completed if arb.get('execution_profit', 0) > 0)
        success_rate = success_count / total_trades if total_trades > 0 else 0
        total_volume = sum(arb.get('execution_amount', 0) * arb.get('execution_buy_price', 0) for arb in completed)
        
        return {
            'total_trades': total_trades,
            'total_profit': total_profit,
            'avg_profit_pct': avg_profit_pct,
            'success_rate': success_rate,
            'total_volume': total_volume
        }
    
    def set_arbitrage_parameters(self, min_profit=None, max_execution_time=None, max_slippage=None):
        """
        Set arbitrage parameters.
        
        Args:
            min_profit (float): Minimum profit percentage
            max_execution_time (int): Maximum execution time in seconds
            max_slippage (float): Maximum allowed slippage percentage
        
        Returns:
            dict: Updated parameters
        """
        if min_profit is not None:
            self.min_profit_threshold = min_profit
        
        if max_execution_time is not None:
            self.max_execution_time = max_execution_time
        
        if max_slippage is not None:
            self.max_slippage = max_slippage
        
        return {
            'min_profit_threshold': self.min_profit_threshold,
            'max_execution_time': self.max_execution_time,
            'max_slippage': self.max_slippage
        }
    
    def get_exchange_status(self):
        """
        Get status of all exchanges.
        
        Returns:
            dict: Exchange status
        """
        status = {}
        
        for exchange_id, exchange_data in self.exchanges.items():
            try:
                # Check if exchange is responsive
                exchange = exchange_data['instance']
                markets_count = len(exchange_data.get('markets', {}))
                
                # Get timestamp of last successful operation
                last_operation = exchange.last_response_headers.get('Date', 'Unknown')
                
                status[exchange_id] = {
                    'enabled': exchange_data['enabled'],
                    'markets_count': markets_count,
                    'last_operation': last_operation,
                    'status': 'online' if exchange_data['enabled'] else 'offline'
                }
            except Exception as e:
                status[exchange_id] = {
                    'enabled': False,
                    'markets_count': 0,
                    'last_operation': 'Unknown',
                    'status': 'error',
                    'error': str(e)
                }
        
        return status
    
    def get_triangular_arbitrage_opportunities(self, exchange_id, base_currency='USDT'):
        """
        Find triangular arbitrage opportunities on a single exchange.
        
        Args:
            exchange_id (str): Exchange ID
            base_currency (str): Base currency for triangular arbitrage
        
        Returns:
            list: List of triangular arbitrage opportunities
        """
        if exchange_id not in self.exchanges:
            self.logger.error(f"Exchange {exchange_id} not found")
            return []
        
        exchange_data = self.exchanges[exchange_id]
        if not exchange_data['enabled']:
            self.logger.error(f"Exchange {exchange_id} is not enabled")
            return []
        
        exchange = exchange_data['instance']
        markets = exchange_data['markets']
        
        if not markets:
            self.logger.error(f"No markets loaded for {exchange_id}")
            return []
        
        # Find all trading pairs with the base currency
        base_pairs = []
        for symbol in markets:
            market = markets[symbol]
            if market['quote'] == base_currency:
                base_pairs.append(symbol)
        
        # Find all possible triangular paths
        triangular_paths = []
        
        for pair1 in base_pairs:
            currency1 = markets[pair1]['base']
            
            for pair2 in markets:
                if pair2 == pair1:
                    continue
                
                # Check if pair2 starts with currency1
                if markets[pair2]['base'] == currency1:
                    currency2 = markets[pair2]['quote']
                    
                    # Find a pair that completes the triangle
                    for pair3 in markets:
                        if pair3 == pair1 or pair3 == pair2:
                            continue
                        
                        if (markets[pair3]['base'] == currency2 and markets[pair3]['quote'] == base_currency):
                            triangular_paths.append({
                                'path': [pair1, pair2, pair3],
                                'direction': 'forward'
                            })
                
                # Check if pair2 ends with currency1
                elif markets[pair2]['quote'] == currency1:
                    currency2 = markets[pair2]['base']
                    
                    # Find a pair that completes the triangle
                    for pair3 in markets:
                        if pair3 == pair1 or pair3 == pair2:
                            continue
                        
                        if (markets[pair3]['base'] == currency2 and markets[pair3]['quote'] == base_currency):
                            triangular_paths.append({
                                'path': [pair1, pair2, pair3],
                                'direction': 'reverse'
                            })
        
        # Calculate profit for each path
        opportunities = []
        
        for path_data in triangular_paths:
            path = path_data['path']
            direction = path_data['direction']
            
            try:
                # Get tickers for all pairs in the path
                tickers = {}
                for pair in path:
                    ticker = exchange.fetch_ticker(pair)
                    tickers[pair] = ticker
                
                # Calculate profit
                if direction == 'forward':
                    # Start with 1 unit of base currency
                    amount = 1.0
                    
                    # Step 1: Convert base to currency1 using pair1
                    pair1 = path[0]
                    rate1 = tickers[pair1]['ask']
                    amount = amount / rate1  # Convert USDT to currency1
                    
                    # Step 2: Convert currency1 to currency2 using pair2
                    pair2 = path[1]
                    if markets[pair2]['base'] == markets[pair1]['base']:
                        # Selling currency1 for currency2
                        rate2 = tickers[pair2]['bid']
                        amount = amount * rate2  # Convert currency1 to currency2
                    else:
                        # Buying currency2 with currency1
                        rate2 = tickers[pair2]['ask']
                        amount = amount / rate2  # Convert currency1 to currency2
                    
                    # Step 3: Convert currency2 back to base using pair3
                    pair3 = path[2]
                    rate3 = tickers[pair3]['bid']
                    amount = amount * rate3  # Convert currency2 to USDT
                
                else:  # reverse
                    # Start with 1 unit of base currency
                    amount = 1.0
                    
                    # Step 1: Convert base to currency1 using pair1
                    pair1 = path[0]
                    rate1 = tickers[pair1]['ask']
                    amount = amount / rate1  # Convert USDT to currency1
                    
                    # Step 2: Convert currency1 to currency2 using pair2
                    pair2 = path[1]
                    if markets[pair2]['quote'] == markets[pair1]['base']:
                        # Buying currency2 with currency1
                        rate2 = tickers[pair2]['ask']
                        amount = amount / rate2  # Convert currency1 to currency2
                    else:
                        # Selling currency1 for currency2
                        rate2 = tickers[pair2]['bid']
                        amount = amount * rate2  # Convert currency1 to currency2
                    
                    # Step 3: Convert currency2 back to base using pair3
                    pair3 = path[2]
                    rate3 = tickers[pair3]['bid']
                    amount = amount * rate3  # Convert currency2 to USDT
                
                # Calculate profit percentage
                profit_pct = (amount - 1.0) * 100
                
                # Account for fees
                fee_rate = exchange_data['fees']['taker'] / 100
                fee_impact = (1 - fee_rate) ** 3 - 1  # Impact of fees on 3 trades
                adjusted_profit_pct = profit_pct + fee_impact * 100
                
                # If profitable, add to opportunities
                if adjusted_profit_pct > self.min_profit_threshold:
                    opportunities.append({
                        'exchange': exchange_id,
                        'path': path,
                        'direction': direction,
                        'profit_pct': adjusted_profit_pct,
                        'rates': [rate1, rate2, rate3],
                        'timestamp': datetime.now(),
                        'status': 'identified'
                    })
            
            except Exception as e:
                self.logger.error(f"Error calculating triangular arbitrage for path {path}: {str(e)}")
        
        # Sort by profit percentage
        opportunities.sort(key=lambda x: x['profit_pct'], reverse=True)
        
        return opportunities
