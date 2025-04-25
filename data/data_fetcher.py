import pandas as pd
import numpy as np
import ccxt
import yfinance as yf
from datetime import datetime, timedelta
import time

class CryptoDataFetcher:
    """
    A class for fetching cryptocurrency data from various exchanges.
    Supports multiple exchanges through CCXT library and additional data from Yahoo Finance.
    """
    
    def __init__(self):
        """Initialize the CryptoDataFetcher with available exchanges."""
        self.available_exchanges = {
            'Binance': ccxt.binance,
            'Coinbase': ccxt.coinbasepro,
            'Kraken': ccxt.kraken,
            'Kucoin': ccxt.kucoin
        }
        self.current_exchange = None
        self.exchange_instance = None
    
    def connect_exchange(self, exchange_name, api_key=None, api_secret=None):
        """
        Connect to a specific cryptocurrency exchange.
        
        Args:
            exchange_name (str): Name of the exchange to connect to
            api_key (str, optional): API key for authenticated requests
            api_secret (str, optional): API secret for authenticated requests
            
        Returns:
            bool: True if connection successful, False otherwise
        """
        if exchange_name not in self.available_exchanges:
            print(f"Exchange {exchange_name} not supported. Available exchanges: {list(self.available_exchanges.keys())}")
            return False
        
        try:
            exchange_class = self.available_exchanges[exchange_name]
            
            # Initialize with API credentials if provided
            if api_key and api_secret:
                self.exchange_instance = exchange_class({
                    'apiKey': api_key,
                    'secret': api_secret,
                    'enableRateLimit': True
                })
            else:
                self.exchange_instance = exchange_class({
                    'enableRateLimit': True
                })
            
            self.current_exchange = exchange_name
            return True
            
        except Exception as e:
            print(f"Error connecting to {exchange_name}: {str(e)}")
            return False
    
    def get_available_symbols(self):
        """
        Get list of available trading pairs on the connected exchange.
        
        Returns:
            list: List of available trading pairs
        """
        if not self.exchange_instance:
            print("No exchange connected. Call connect_exchange() first.")
            return []
        
        try:
            self.exchange_instance.load_markets()
            return [symbol for symbol in self.exchange_instance.symbols if '/USDT' in symbol]
        except Exception as e:
            print(f"Error fetching symbols: {str(e)}")
            return []
    
    def fetch_ohlcv(self, symbol, timeframe='1h', limit=100):
        """
        Fetch OHLCV (Open, High, Low, Close, Volume) data for a specific trading pair.
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTC/USDT')
            timeframe (str): Timeframe for the data (e.g., '1m', '5m', '1h', '1d')
            limit (int): Number of candles to fetch
            
        Returns:
            pd.DataFrame: DataFrame containing OHLCV data
        """
        if not self.exchange_instance:
            print("No exchange connected. Call connect_exchange() first.")
            return pd.DataFrame()
        
        try:
            # Fetch OHLCV data
            ohlcv = self.exchange_instance.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            print(f"Error fetching OHLCV data: {str(e)}")
            return pd.DataFrame()
    
    def fetch_ticker(self, symbol):
        """
        Fetch current ticker data for a specific trading pair.
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTC/USDT')
            
        Returns:
            dict: Dictionary containing ticker data
        """
        if not self.exchange_instance:
            print("No exchange connected. Call connect_exchange() first.")
            return {}
        
        try:
            ticker = self.exchange_instance.fetch_ticker(symbol)
            return ticker
        except Exception as e:
            print(f"Error fetching ticker data: {str(e)}")
            return {}
    
    def fetch_historical_data(self, symbol, days=30, timeframe='1d'):
        """
        Fetch historical data for a specific trading pair.
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTC/USDT')
            days (int): Number of days of historical data to fetch
            timeframe (str): Timeframe for the data
            
        Returns:
            pd.DataFrame: DataFrame containing historical data
        """
        # Calculate number of candles based on timeframe and days
        timeframe_in_minutes = self._convert_timeframe_to_minutes(timeframe)
        minutes_in_day = 24 * 60
        candles_per_day = minutes_in_day / timeframe_in_minutes
        limit = int(candles_per_day * days)
        
        # Limit to maximum allowed by exchange (usually 1000)
        limit = min(limit, 1000)
        
        return self.fetch_ohlcv(symbol, timeframe, limit)
    
    def fetch_from_yfinance(self, symbol, period="1mo", interval="1d"):
        """
        Fetch data from Yahoo Finance as a fallback or for additional data.
        
        Args:
            symbol (str): Symbol to fetch (e.g., 'BTC-USD')
            period (str): Period to fetch (e.g., '1d', '1mo', '1y')
            interval (str): Interval for the data (e.g., '1m', '1h', '1d')
            
        Returns:
            pd.DataFrame: DataFrame containing historical data
        """
        try:
            # Convert CCXT symbol format to Yahoo Finance format
            if '/' in symbol:
                base, quote = symbol.split('/')
                yf_symbol = f"{base}-{quote}"
            else:
                yf_symbol = symbol
            
            # Fetch data from Yahoo Finance
            data = yf.download(yf_symbol, period=period, interval=interval)
            return data
            
        except Exception as e:
            print(f"Error fetching data from Yahoo Finance: {str(e)}")
            return pd.DataFrame()
    
    def get_market_summary(self, symbols=None):
        """
        Get a summary of market data for multiple symbols.
        
        Args:
            symbols (list, optional): List of symbols to fetch data for. If None, uses top coins.
            
        Returns:
            pd.DataFrame: DataFrame containing market summary
        """
        if not symbols:
            symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT']
        
        if not self.exchange_instance:
            print("No exchange connected. Call connect_exchange() first.")
            return pd.DataFrame()
        
        try:
            summary_data = []
            
            for symbol in symbols:
                ticker = self.fetch_ticker(symbol)
                if ticker:
                    summary_data.append({
                        'symbol': symbol,
                        'last_price': ticker['last'],
                        'daily_change': ticker['percentage'] if 'percentage' in ticker else ticker.get('change', 0),
                        'volume_24h': ticker['quoteVolume'],
                        'high_24h': ticker['high'],
                        'low_24h': ticker['low']
                    })
            
            return pd.DataFrame(summary_data)
            
        except Exception as e:
            print(f"Error fetching market summary: {str(e)}")
            return pd.DataFrame()
    
    def _convert_timeframe_to_minutes(self, timeframe):
        """
        Convert timeframe string to minutes.
        
        Args:
            timeframe (str): Timeframe string (e.g., '1m', '5m', '1h', '1d')
            
        Returns:
            int: Timeframe in minutes
        """
        unit = timeframe[-1]
        value = int(timeframe[:-1])
        
        if unit == 'm':
            return value
        elif unit == 'h':
            return value * 60
        elif unit == 'd':
            return value * 24 * 60
        elif unit == 'w':
            return value * 7 * 24 * 60
        else:
            return 60  # Default to 1h
    
    def generate_sample_data(self, symbol='BTC/USDT', days=30, volatility=0.02):
        """
        Generate sample price data for demonstration purposes.
        
        Args:
            symbol (str): Symbol to generate data for
            days (int): Number of days of data to generate
            volatility (float): Volatility factor for price movements
            
        Returns:
            pd.DataFrame: DataFrame containing sample OHLCV data
        """
        # Set seed for reproducibility
        np.random.seed(42)
        
        # Determine starting price based on symbol
        if 'BTC' in symbol:
            start_price = 50000
        elif 'ETH' in symbol:
            start_price = 3000
        elif 'SOL' in symbol:
            start_price = 100
        elif 'XRP' in symbol:
            start_price = 0.5
        elif 'ADA' in symbol:
            start_price = 1.2
        else:
            start_price = 100
        
        # Generate timestamps
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        timestamps = pd.date_range(start=start_date, end=end_date, freq='1h')
        
        # Generate price data with random walk
        returns = np.random.normal(0, volatility, len(timestamps))
        price_changes = 1 + returns
        prices = start_price * np.cumprod(price_changes)
        
        # Generate OHLCV data
        data = []
        for i, timestamp in enumerate(timestamps):
            price = prices[i]
            high_factor = 1 + abs(np.random.normal(0, volatility/2))
            low_factor = 1 - abs(np.random.normal(0, volatility/2))
            
            # Ensure high is higher than price and low is lower than price
            high = price * high_factor
            low = price * low_factor
            
            # For the open price, use previous close or a random value for the first entry
            if i == 0:
                open_price = price * (1 + np.random.normal(0, volatility/2))
            else:
                open_price = prices[i-1]
            
            # Volume is random but correlated with price volatility
            volume = abs(np.random.normal(1000000, 500000)) * (1 + abs(returns[i]) * 10)
            
            data.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume
            })
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        return df
