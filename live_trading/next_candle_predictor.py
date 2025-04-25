import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import threading
import ccxt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os

class NextCandlePredictor:
    """
    A class for predicting the next candle's price movement using machine learning.
    """
    
    def __init__(self, model_dir='/home/ubuntu/trading_bot/crypto_trading_bot/models'):
        """
        Initialize the NextCandlePredictor.
        
        Args:
            model_dir (str): Directory to save/load prediction models
        """
        self.model_dir = model_dir
        self.models = {}
        self.scalers = {}
        self.is_training = False
        self.feature_columns = []
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
    
    def prepare_features(self, df):
        """
        Prepare features for prediction model.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV and indicator data
            
        Returns:
            pd.DataFrame: DataFrame with features for prediction
        """
        # Make a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Basic price and volume features
        df_copy['price_change'] = df_copy['close'].pct_change()
        df_copy['volume_change'] = df_copy['volume'].pct_change()
        df_copy['high_low_diff'] = (df_copy['high'] - df_copy['low']) / df_copy['low']
        df_copy['close_open_diff'] = (df_copy['close'] - df_copy['open']) / df_copy['open']
        
        # Lagged features (previous candles)
        for i in range(1, 6):  # Use 5 previous candles
            df_copy[f'price_change_lag_{i}'] = df_copy['price_change'].shift(i)
            df_copy[f'volume_change_lag_{i}'] = df_copy['volume_change'].shift(i)
            df_copy[f'high_low_diff_lag_{i}'] = df_copy['high_low_diff'].shift(i)
        
        # Technical indicator features
        # Assuming these indicators are already in the dataframe
        indicator_cols = [col for col in df_copy.columns if any(
            ind in col for ind in ['rsi', 'macd', 'bollinger', 'sma', 'ema', 'atr']
        )]
        
        # Create lagged versions of indicators
        for col in indicator_cols:
            df_copy[f'{col}_change'] = df_copy[col].pct_change()
            df_copy[f'{col}_lag_1'] = df_copy[col].shift(1)
        
        # Drop rows with NaN values
        df_copy = df_copy.dropna()
        
        # Select feature columns (excluding the original OHLCV columns)
        feature_cols = [
            'price_change', 'volume_change', 'high_low_diff', 'close_open_diff'
        ]
        
        # Add lagged features
        for i in range(1, 6):
            feature_cols.extend([
                f'price_change_lag_{i}',
                f'volume_change_lag_{i}',
                f'high_low_diff_lag_{i}'
            ])
        
        # Add indicator features
        for col in indicator_cols:
            feature_cols.extend([f'{col}_change', f'{col}_lag_1'])
        
        # Store feature columns for later use
        self.feature_columns = feature_cols
        
        return df_copy[feature_cols]
    
    def prepare_targets(self, df):
        """
        Prepare target variables for prediction model.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with target variables
        """
        # Make a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Target variables (next candle's values)
        df_copy['next_open'] = df_copy['open'].shift(-1)
        df_copy['next_high'] = df_copy['high'].shift(-1)
        df_copy['next_low'] = df_copy['low'].shift(-1)
        df_copy['next_close'] = df_copy['close'].shift(-1)
        
        # Calculate percentage changes for targets
        df_copy['next_open_pct'] = (df_copy['next_open'] - df_copy['close']) / df_copy['close']
        df_copy['next_high_pct'] = (df_copy['next_high'] - df_copy['close']) / df_copy['close']
        df_copy['next_low_pct'] = (df_copy['next_low'] - df_copy['close']) / df_copy['close']
        df_copy['next_close_pct'] = (df_copy['next_close'] - df_copy['close']) / df_copy['close']
        
        # Drop rows with NaN values
        df_copy = df_copy.dropna()
        
        # Target columns
        target_cols = ['next_open_pct', 'next_high_pct', 'next_low_pct', 'next_close_pct']
        
        return df_copy[target_cols]
    
    def train_model(self, df, symbol, force_retrain=False):
        """
        Train prediction model for a specific symbol.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV and indicator data
            symbol (str): Trading pair symbol
            force_retrain (bool): Whether to force retraining even if model exists
            
        Returns:
            bool: Whether training was successful
        """
        # Check if we're already training
        if self.is_training:
            print("Training is already in progress.")
            return False
        
        # Check if model already exists
        model_path = os.path.join(self.model_dir, f"{symbol.replace('/', '_')}_model.joblib")
        scaler_path = os.path.join(self.model_dir, f"{symbol.replace('/', '_')}_scaler.joblib")
        
        if os.path.exists(model_path) and os.path.exists(scaler_path) and not force_retrain:
            print(f"Model for {symbol} already exists. Loading existing model.")
            self.models[symbol] = joblib.load(model_path)
            self.scalers[symbol] = joblib.load(scaler_path)
            return True
        
        try:
            self.is_training = True
            print(f"Training prediction model for {symbol}...")
            
            # Prepare features and targets
            X = self.prepare_features(df)
            y = self.prepare_targets(df)
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train model
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_scaled, y)
            
            # Save model and scaler
            self.models[symbol] = model
            self.scalers[symbol] = scaler
            
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            
            print(f"Model training completed for {symbol}.")
            return True
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
            return False
            
        finally:
            self.is_training = False
    
    def predict_next_candle(self, df, symbol):
        """
        Predict the next candle's price movement.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV and indicator data
            symbol (str): Trading pair symbol
            
        Returns:
            dict: Predicted values for next candle
        """
        # Check if model exists
        if symbol not in self.models or symbol not in self.scalers:
            model_path = os.path.join(self.model_dir, f"{symbol.replace('/', '_')}_model.joblib")
            scaler_path = os.path.join(self.model_dir, f"{symbol.replace('/', '_')}_scaler.joblib")
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.models[symbol] = joblib.load(model_path)
                self.scalers[symbol] = joblib.load(scaler_path)
            else:
                print(f"No model found for {symbol}. Training new model...")
                success = self.train_model(df, symbol)
                if not success:
                    return None
        
        try:
            # Prepare features
            X = self.prepare_features(df)
            
            # Use only the last row for prediction
            X_last = X.iloc[-1:].copy()
            
            # Scale features
            X_scaled = self.scalers[symbol].transform(X_last)
            
            # Make prediction
            predictions = self.models[symbol].predict(X_scaled)[0]
            
            # Get current price
            current_price = df['close'].iloc[-1]
            
            # Calculate predicted prices
            next_open = current_price * (1 + predictions[0])
            next_high = current_price * (1 + predictions[1])
            next_low = current_price * (1 + predictions[2])
            next_close = current_price * (1 + predictions[3])
            
            # Create prediction dictionary
            prediction = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'current_price': current_price,
                'next_open': next_open,
                'next_high': next_high,
                'next_low': next_low,
                'next_close': next_close,
                'predicted_change_pct': predictions[3] * 100  # next_close_pct in percentage
            }
            
            return prediction
            
        except Exception as e:
            print(f"Error predicting next candle: {str(e)}")
            return None
    
    def evaluate_model(self, df, symbol, test_size=0.2):
        """
        Evaluate the prediction model's performance.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV and indicator data
            symbol (str): Trading pair symbol
            test_size (float): Proportion of data to use for testing
            
        Returns:
            dict: Evaluation metrics
        """
        try:
            # Prepare features and targets
            X = self.prepare_features(df)
            y = self.prepare_targets(df)
            
            # Split data into training and testing sets
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            mse = np.mean((y_test.values - y_pred) ** 2, axis=0)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_test.values - y_pred), axis=0)
            
            # Calculate direction accuracy
            direction_accuracy = np.mean((y_test['next_close_pct'] > 0) == (y_pred[:, 3] > 0))
            
            # Create metrics dictionary
            metrics = {
                'symbol': symbol,
                'mse': mse.tolist(),
                'rmse': rmse.tolist(),
                'mae': mae.tolist(),
                'direction_accuracy': direction_accuracy
            }
            
            return metrics
            
        except Exception as e:
            print(f"Error evaluating model: {str(e)}")
            return None
