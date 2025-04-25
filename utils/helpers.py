import os
import numpy as np
import pandas as pd

def format_number(num, prefix=False):
    """
    Format number for display.
    
    Args:
        num (float): Number to format
        prefix (bool): Whether to add + prefix for positive numbers
        
    Returns:
        str: Formatted number
    """
    if num is None:
        return "N/A"
    
    sign = "+" if num > 0 and prefix else ""
    
    if abs(num) >= 1000000000:
        return f"{sign}{num / 1000000000:.2f}B"
    elif abs(num) >= 1000000:
        return f"{sign}{num / 1000000:.2f}M"
    elif abs(num) >= 1000:
        return f"{sign}{num / 1000:.2f}K"
    elif abs(num) >= 1:
        return f"{sign}{num:.2f}"
    else:
        # For small numbers like crypto prices
        if abs(num) < 0.0001:
            return f"{sign}{num:.8f}"
        else:
            return f"{sign}{num:.4f}"

def calculate_risk_reward_ratio(entry_price, stop_loss, take_profit):
    """
    Calculate risk-to-reward ratio.
    
    Args:
        entry_price (float): Entry price
        stop_loss (float): Stop loss price
        take_profit (float): Take profit price
        
    Returns:
        float: Risk-to-reward ratio
    """
    if entry_price is None or stop_loss is None or take_profit is None:
        return None
    
    risk = abs(entry_price - stop_loss)
    reward = abs(take_profit - entry_price)
    
    if risk == 0:
        return float('inf')
    
    return reward / risk

def calculate_position_size(account_balance, risk_percentage, entry_price, stop_loss):
    """
    Calculate position size based on risk percentage.
    
    Args:
        account_balance (float): Account balance
        risk_percentage (float): Risk percentage (0-100)
        entry_price (float): Entry price
        stop_loss (float): Stop loss price
        
    Returns:
        float: Position size in units
    """
    if account_balance is None or risk_percentage is None or entry_price is None or stop_loss is None:
        return None
    
    risk_amount = account_balance * (risk_percentage / 100)
    risk_per_unit = abs(entry_price - stop_loss)
    
    if risk_per_unit == 0:
        return 0
    
    return risk_amount / risk_per_unit

def save_data_to_csv(df, filename, directory="data"):
    """
    Save DataFrame to CSV file.
    
    Args:
        df (pd.DataFrame): DataFrame to save
        filename (str): Filename
        directory (str): Directory to save to
        
    Returns:
        str: Path to saved file
    """
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Create full path
    path = os.path.join(directory, filename)
    
    # Save DataFrame to CSV
    df.to_csv(path)
    
    return path

def load_data_from_csv(filename, directory="data"):
    """
    Load DataFrame from CSV file.
    
    Args:
        filename (str): Filename
        directory (str): Directory to load from
        
    Returns:
        pd.DataFrame: Loaded DataFrame
    """
    # Create full path
    path = os.path.join(directory, filename)
    
    # Check if file exists
    if not os.path.exists(path):
        return None
    
    # Load DataFrame from CSV
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    
    return df

def generate_timeframes():
    """
    Generate list of common timeframes.
    
    Returns:
        list: List of timeframes
    """
    return ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]

def generate_trading_pairs():
    """
    Generate list of common trading pairs.
    
    Returns:
        list: List of trading pairs
    """
    base_currencies = ["BTC", "ETH", "SOL", "XRP", "ADA", "DOT", "DOGE", "AVAX", "MATIC", "LINK"]
    quote_currencies = ["USDT", "USDC", "BUSD"]
    
    pairs = []
    for base in base_currencies:
        for quote in quote_currencies:
            pairs.append(f"{base}/{quote}")
    
    return pairs

def calculate_correlation_matrix(price_data):
    """
    Calculate correlation matrix for multiple assets.
    
    Args:
        price_data (dict): Dictionary with asset names as keys and price DataFrames as values
        
    Returns:
        pd.DataFrame: Correlation matrix
    """
    # Extract close prices
    close_prices = {}
    for asset, df in price_data.items():
        if df is not None and 'close' in df.columns:
            close_prices[asset] = df['close']
    
    if not close_prices:
        return None
    
    # Create DataFrame with close prices
    df = pd.DataFrame(close_prices)
    
    # Calculate correlation matrix
    corr_matrix = df.corr()
    
    return corr_matrix

def calculate_volatility(df, window=14):
    """
    Calculate historical volatility.
    
    Args:
        df (pd.DataFrame): DataFrame with price data
        window (int): Window for volatility calculation
        
    Returns:
        pd.Series: Volatility series
    """
    if df is None or 'close' not in df.columns:
        return None
    
    # Calculate daily returns
    returns = df['close'].pct_change().dropna()
    
    # Calculate rolling standard deviation
    volatility = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
    
    return volatility
