import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os
import datetime
import ccxt
import time
from concurrent.futures import ThreadPoolExecutor

class AdvancedAITradingSystem:
    """
    Advanced AI-powered trading system that combines multiple machine learning models,
    deep learning, and reinforcement learning for superior trading decisions.
    """
    
    def __init__(self):
        """Initialize the AdvancedAITradingSystem."""
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.performance_metrics = {}
        self.last_training_time = {}
        self.reinforcement_agent = None
        self.ensemble_weights = {}
        
        # Create models directory if it doesn't exist
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
        os.makedirs(models_dir, exist_ok=True)
        self.models_dir = models_dir
        
        # Initialize TensorFlow for deep learning models
        self._setup_tensorflow()
    
    def _setup_tensorflow(self):
        """Set up TensorFlow with optimized configuration."""
        # Set TensorFlow logging level
        tf.get_logger().setLevel('ERROR')
        
        # Configure TensorFlow for performance
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Memory growth for better GPU utilization
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                # Use only first GPU to avoid memory issues
                tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
    
    def train_models(self, symbol, timeframe, historical_data, retrain=False):
        """
        Train multiple AI models for a specific symbol and timeframe.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTC/USDT')
            timeframe (str): Timeframe (e.g., '1h', '4h', '1d')
            historical_data (pd.DataFrame): Historical OHLCV data with indicators
            retrain (bool): Whether to retrain existing models
        
        Returns:
            bool: Success status
        """
        model_key = f"{symbol}_{timeframe}"
        
        # Check if models already exist and retrain is not requested
        if model_key in self.models and not retrain:
            print(f"Models for {model_key} already exist. Use retrain=True to retrain.")
            return True
        
        try:
            # Prepare data
            X, y_direction, y_price, y_volatility = self._prepare_training_data(historical_data)
            
            if len(X) < 100:
                print(f"Not enough data for {model_key}. Need at least 100 samples.")
                return False
            
            # Split data into training and validation sets
            train_size = int(len(X) * 0.8)
            X_train, X_val = X[:train_size], X[train_size:]
            y_dir_train, y_dir_val = y_direction[:train_size], y_direction[train_size:]
            y_price_train, y_price_val = y_price[:train_size], y_price[train_size:]
            y_vol_train, y_vol_val = y_volatility[:train_size], y_volatility[train_size:]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Save scaler
            self.scalers[model_key] = scaler
            
            # Train direction prediction model (classification)
            print(f"Training direction prediction model for {model_key}...")
            dir_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            dir_model.fit(X_train_scaled, y_dir_train)
            dir_accuracy = dir_model.score(X_val_scaled, y_dir_val)
            print(f"Direction model accuracy: {dir_accuracy:.4f}")
            
            # Train price prediction model (regression)
            print(f"Training price prediction model for {model_key}...")
            price_model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
            price_model.fit(X_train_scaled, y_price_train)
            price_mse = np.mean((price_model.predict(X_val_scaled) - y_price_val) ** 2)
            print(f"Price model MSE: {price_mse:.6f}")
            
            # Train volatility prediction model (regression)
            print(f"Training volatility prediction model for {model_key}...")
            vol_model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
            vol_model.fit(X_train_scaled, y_vol_train)
            vol_mse = np.mean((vol_model.predict(X_val_scaled) - y_vol_val) ** 2)
            print(f"Volatility model MSE: {vol_mse:.6f}")
            
            # Train deep learning model
            print(f"Training deep learning model for {model_key}...")
            dl_model = self._build_deep_learning_model(X_train.shape[1])
            dl_model.fit(
                X_train_scaled, 
                [y_dir_train, y_price_train, y_vol_train],
                epochs=50,
                batch_size=32,
                validation_data=(X_val_scaled, [y_dir_val, y_price_val, y_vol_val]),
                verbose=0
            )
            
            # Store models
            self.models[model_key] = {
                'direction': dir_model,
                'price': price_model,
                'volatility': vol_model,
                'deep_learning': dl_model
            }
            
            # Store feature importance
            self.feature_importance[model_key] = {
                'direction': dir_model.feature_importances_,
                'price': price_model.feature_importances_,
                'volatility': vol_model.feature_importances_
            }
            
            # Store performance metrics
            self.performance_metrics[model_key] = {
                'direction_accuracy': dir_accuracy,
                'price_mse': price_mse,
                'volatility_mse': vol_mse
            }
            
            # Update training time
            self.last_training_time[model_key] = datetime.datetime.now()
            
            # Save models to disk
            self._save_models(model_key)
            
            # Optimize ensemble weights
            self._optimize_ensemble_weights(model_key, X_val_scaled, y_dir_val, y_price_val)
            
            print(f"Successfully trained all models for {model_key}")
            return True
        
        except Exception as e:
            print(f"Error training models for {model_key}: {str(e)}")
            return False
    
    def _prepare_training_data(self, df):
        """
        Prepare training data from historical DataFrame.
        
        Args:
            df (pd.DataFrame): Historical OHLCV data with indicators
        
        Returns:
            tuple: (X, y_direction, y_price, y_volatility)
        """
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Drop rows with NaN values
        df.dropna(inplace=True)
        
        # Create target variables
        # 1. Direction (1 if next close > current close, 0 otherwise)
        df['next_close'] = df['close'].shift(-1)
        df['direction'] = (df['next_close'] > df['close']).astype(int)
        
        # 2. Price change percentage
        df['price_change_pct'] = (df['next_close'] - df['close']) / df['close'] * 100
        
        # 3. Volatility (high-low range of next candle)
        df['next_high'] = df['high'].shift(-1)
        df['next_low'] = df['low'].shift(-1)
        df['next_volatility'] = (df['next_high'] - df['next_low']) / df['close'] * 100
        
        # Drop the last row since we don't have next values for it
        df = df[:-1]
        
        # Select features (exclude target variables and timestamp)
        feature_cols = [col for col in df.columns if col not in [
            'next_close', 'direction', 'price_change_pct', 
            'next_high', 'next_low', 'next_volatility', 'timestamp'
        ]]
        
        X = df[feature_cols].values
        y_direction = df['direction'].values
        y_price = df['price_change_pct'].values
        y_volatility = df['next_volatility'].values
        
        return X, y_direction, y_price, y_volatility
    
    def _build_deep_learning_model(self, input_dim):
        """
        Build a deep learning model with multiple outputs.
        
        Args:
            input_dim (int): Number of input features
        
        Returns:
            tf.keras.Model: Deep learning model
        """
        # Input layer
        inputs = tf.keras.layers.Input(shape=(input_dim,))
        
        # Shared layers
        x = tf.keras.layers.Dense(128, activation='relu')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        # Direction output (classification)
        direction_output = tf.keras.layers.Dense(32, activation='relu')(x)
        direction_output = tf.keras.layers.Dense(1, activation='sigmoid', name='direction')(direction_output)
        
        # Price change output (regression)
        price_output = tf.keras.layers.Dense(32, activation='relu')(x)
        price_output = tf.keras.layers.Dense(1, name='price')(price_output)
        
        # Volatility output (regression)
        volatility_output = tf.keras.layers.Dense(32, activation='relu')(x)
        volatility_output = tf.keras.layers.Dense(1, name='volatility')(volatility_output)
        
        # Create model
        model = tf.keras.Model(
            inputs=inputs, 
            outputs=[direction_output, price_output, volatility_output]
        )
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss={
                'direction': 'binary_crossentropy',
                'price': 'mse',
                'volatility': 'mse'
            },
            metrics={
                'direction': 'accuracy',
                'price': 'mse',
                'volatility': 'mse'
            }
        )
        
        return model
    
    def _save_models(self, model_key):
        """
        Save models to disk.
        
        Args:
            model_key (str): Model key (symbol_timeframe)
        """
        # Create directory for this model if it doesn't exist
        model_dir = os.path.join(self.models_dir, model_key)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save traditional ML models
        joblib.dump(self.models[model_key]['direction'], os.path.join(model_dir, 'direction_model.joblib'))
        joblib.dump(self.models[model_key]['price'], os.path.join(model_dir, 'price_model.joblib'))
        joblib.dump(self.models[model_key]['volatility'], os.path.join(model_dir, 'volatility_model.joblib'))
        
        # Save scaler
        joblib.dump(self.scalers[model_key], os.path.join(model_dir, 'scaler.joblib'))
        
        # Save deep learning model
        self.models[model_key]['deep_learning'].save(os.path.join(model_dir, 'deep_learning_model'))
        
        # Save performance metrics
        with open(os.path.join(model_dir, 'metrics.txt'), 'w') as f:
            for metric, value in self.performance_metrics[model_key].items():
                f.write(f"{metric}: {value}\n")
        
        # Save feature importance
        np.save(os.path.join(model_dir, 'feature_importance.npy'), self.feature_importance[model_key])
        
        # Save ensemble weights
        if model_key in self.ensemble_weights:
            np.save(os.path.join(model_dir, 'ensemble_weights.npy'), self.ensemble_weights[model_key])
    
    def load_models(self, symbol, timeframe):
        """
        Load models from disk.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTC/USDT')
            timeframe (str): Timeframe (e.g., '1h', '4h', '1d')
        
        Returns:
            bool: Success status
        """
        model_key = f"{symbol}_{timeframe}"
        model_dir = os.path.join(self.models_dir, model_key)
        
        if not os.path.exists(model_dir):
            print(f"No models found for {model_key}")
            return False
        
        try:
            # Load traditional ML models
            direction_model = joblib.load(os.path.join(model_dir, 'direction_model.joblib'))
            price_model = joblib.load(os.path.join(model_dir, 'price_model.joblib'))
            volatility_model = joblib.load(os.path.join(model_dir, 'volatility_model.joblib'))
            
            # Load scaler
            scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
            
            # Load deep learning model
            dl_model = tf.keras.models.load_model(os.path.join(model_dir, 'deep_learning_model'))
            
            # Store models
            self.models[model_key] = {
                'direction': direction_model,
                'price': price_model,
                'volatility': volatility_model,
                'deep_learning': dl_model
            }
            
            # Store scaler
            self.scalers[model_key] = scaler
            
            # Load feature importance
            if os.path.exists(os.path.join(model_dir, 'feature_importance.npy')):
                self.feature_importance[model_key] = np.load(os.path.join(model_dir, 'feature_importance.npy'), allow_pickle=True).item()
            
            # Load ensemble weights
            if os.path.exists(os.path.join(model_dir, 'ensemble_weights.npy')):
                self.ensemble_weights[model_key] = np.load(os.path.join(model_dir, 'ensemble_weights.npy'), allow_pickle=True).item()
            
            # Set last training time to file modification time
            self.last_training_time[model_key] = datetime.datetime.fromtimestamp(
                os.path.getmtime(os.path.join(model_dir, 'direction_model.joblib'))
            )
            
            print(f"Successfully loaded models for {model_key}")
            return True
        
        except Exception as e:
            print(f"Error loading models for {model_key}: {str(e)}")
            return False
    
    def predict(self, symbol, timeframe, current_data):
        """
        Generate predictions using trained models.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTC/USDT')
            timeframe (str): Timeframe (e.g., '1h', '4h', '1d')
            current_data (pd.DataFrame): Current market data with indicators
        
        Returns:
            dict: Prediction results
        """
        model_key = f"{symbol}_{timeframe}"
        
        # Check if models exist
        if model_key not in self.models:
            # Try to load models
            if not self.load_models(symbol, timeframe):
                print(f"No models available for {model_key}")
                return None
        
        try:
            # Prepare input data
            X = self._prepare_prediction_data(current_data)
            
            # Scale input data
            X_scaled = self.scalers[model_key].transform(X)
            
            # Get predictions from traditional models
            direction_prob = self.models[model_key]['direction'].predict_proba(X_scaled)[0, 1]
            price_change = self.models[model_key]['price'].predict(X_scaled)[0]
            volatility = self.models[model_key]['volatility'].predict(X_scaled)[0]
            
            # Get predictions from deep learning model
            dl_predictions = self.models[model_key]['deep_learning'].predict(X_scaled)
            dl_direction_prob = dl_predictions[0][0, 0]
            dl_price_change = dl_predictions[1][0, 0]
            dl_volatility = dl_predictions[2][0, 0]
            
            # Ensemble predictions
            if model_key in self.ensemble_weights:
                weights = self.ensemble_weights[model_key]
                ensemble_direction = (weights['direction_rf'] * direction_prob + 
                                     weights['direction_dl'] * dl_direction_prob)
                ensemble_price = (weights['price_gb'] * price_change + 
                                 weights['price_dl'] * dl_price_change)
                ensemble_volatility = (weights['volatility_gb'] * volatility + 
                                      weights['volatility_dl'] * dl_volatility)
            else:
                # Default weights if not optimized
                ensemble_direction = 0.6 * direction_prob + 0.4 * dl_direction_prob
                ensemble_price = 0.6 * price_change + 0.4 * dl_price_change
                ensemble_volatility = 0.6 * volatility + 0.4 * dl_volatility
            
            # Calculate confidence based on model performance and prediction certainty
            direction_confidence = abs(ensemble_direction - 0.5) * 2 * 100  # Scale to 0-100%
            
            # Get current price
            current_price = current_data['close'].iloc[-1]
            
            # Calculate predicted prices
            predicted_change_pct = ensemble_price
            predicted_volatility_pct = ensemble_volatility
            
            # Direction: 1 for up, 0 for down
            predicted_direction = 1 if ensemble_direction > 0.5 else 0
            
            # Calculate next candle values
            next_close = current_price * (1 + predicted_change_pct / 100)
            volatility_amount = current_price * predicted_volatility_pct / 100
            
            if predicted_direction == 1:
                next_high = next_close + volatility_amount / 2
                next_low = next_close - volatility_amount / 2
                next_open = (next_low + current_price) / 2  # Somewhere between current and low
            else:
                next_high = next_close + volatility_amount / 2
                next_low = next_close - volatility_amount / 2
                next_open = (next_high + current_price) / 2  # Somewhere between current and high
            
            # Ensure logical order (low <= open, close <= high)
            next_low = min(next_low, next_open, next_close)
            next_high = max(next_high, next_open, next_close)
            
            # Return prediction results
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'current_price': current_price,
                'predicted_direction': predicted_direction,
                'predicted_change_pct': predicted_change_pct,
                'predicted_volatility_pct': predicted_volatility_pct,
                'next_open': next_open,
                'next_high': next_high,
                'next_low': next_low,
                'next_close': next_close,
                'confidence': direction_confidence,
                'timestamp': datetime.datetime.now()
            }
        
        except Exception as e:
            print(f"Error generating predictions for {model_key}: {str(e)}")
            return None
    
    def _prepare_prediction_data(self, df):
        """
        Prepare data for prediction.
        
        Args:
            df (pd.DataFrame): Current market data with indicators
        
        Returns:
            numpy.ndarray: Prepared feature array
        """
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Drop rows with NaN values
        df.dropna(inplace=True)
        
        # Select the last row for prediction
        last_row = df.iloc[-1:].copy()
        
        # Select features (exclude target variables and timestamp if present)
        exclude_cols = ['next_close', 'direction', 'price_change_pct', 
                        'next_high', 'next_low', 'next_volatility', 'timestamp']
        feature_cols = [col for col in last_row.columns if col not in exclude_cols]
        
        X = last_row[feature_cols].values
        
        return X
    
    def _optimize_ensemble_weights(self, model_key, X_val, y_dir_val, y_price_val):
        """
        Optimize weights for ensemble predictions.
        
        Args:
            model_key (str): Model key (symbol_timeframe)
            X_val (numpy.ndarray): Validation features
            y_dir_val (numpy.ndarray): Validation direction targets
            y_price_val (numpy.ndarray): Validation price targets
        """
        # Get predictions from all models
        dir_rf_pred = self.models[model_key]['direction'].predict_proba(X_val)[:, 1]
        price_gb_pred = self.models[model_key]['price'].predict(X_val)
        vol_gb_pred = self.models[model_key]['volatility'].predict(X_val)
        
        dl_pred = self.models[model_key]['deep_learning'].predict(X_val)
        dir_dl_pred = dl_pred[0].flatten()
        price_dl_pred = dl_pred[1].flatten()
        vol_dl_pred = dl_pred[2].flatten()
        
        # Simple grid search for direction weights
        best_dir_acc = 0
        best_dir_weights = {'direction_rf': 0.5, 'direction_dl': 0.5}
        
        for w_rf in np.arange(0.1, 1.0, 0.1):
            w_dl = 1 - w_rf
            ensemble_dir = w_rf * dir_rf_pred + w_dl * dir_dl_pred
            ensemble_pred = (ensemble_dir > 0.5).astype(int)
            accuracy = np.mean(ensemble_pred == y_dir_val)
            
            if accuracy > best_dir_acc:
                best_dir_acc = accuracy
                best_dir_weights = {'direction_rf': w_rf, 'direction_dl': w_dl}
        
        # Simple grid search for price weights
        best_price_mse = float('inf')
        best_price_weights = {'price_gb': 0.5, 'price_dl': 0.5}
        
        for w_gb in np.arange(0.1, 1.0, 0.1):
            w_dl = 1 - w_gb
            ensemble_price = w_gb * price_gb_pred + w_dl * price_dl_pred
            mse = np.mean((ensemble_price - y_price_val) ** 2)
            
            if mse < best_price_mse:
                best_price_mse = mse
                best_price_weights = {'price_gb': w_gb, 'price_dl': w_dl}
        
        # Simple grid search for volatility weights
        best_vol_mse = float('inf')
        best_vol_weights = {'volatility_gb': 0.5, 'volatility_dl': 0.5}
        
        for w_gb in np.arange(0.1, 1.0, 0.1):
            w_dl = 1 - w_gb
            ensemble_vol = w_gb * vol_gb_pred + w_dl * vol_dl_pred
            mse = np.mean((ensemble_vol - y_price_val) ** 2)  # Using price_val as proxy
            
            if mse < best_vol_mse:
                best_vol_mse = mse
                best_vol_weights = {'volatility_gb': w_gb, 'volatility_dl': w_dl}
        
        # Combine all weights
        self.ensemble_weights[model_key] = {**best_dir_weights, **best_price_weights, **best_vol_weights}
        print(f"Optimized ensemble weights for {model_key}: {self.ensemble_weights[model_key]}")
    
    def generate_trading_signal(self, symbol, timeframe, current_data, risk_level='medium'):
        """
        Generate a trading signal based on predictions.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTC/USDT')
            timeframe (str): Timeframe (e.g., '1h', '4h', '1d')
            current_data (pd.DataFrame): Current market data with indicators
            risk_level (str): Risk level ('low', 'medium', 'high')
        
        Returns:
            dict: Trading signal
        """
        # Get prediction
        prediction = self.predict(symbol, timeframe, current_data)
        
        if prediction is None:
            return None
        
        # Current price
        current_price = prediction['current_price']
        
        # Determine signal type based on prediction
        signal_type = 'BUY' if prediction['predicted_direction'] == 1 else 'SELL'
        
        # Set risk parameters based on risk level
        if risk_level == 'low':
            entry_spread = 0.003  # 0.3%
            target_multiplier = 1.5
            stop_loss_multiplier = 0.5
        elif risk_level == 'medium':
            entry_spread = 0.005  # 0.5%
            target_multiplier = 2.0
            stop_loss_multiplier = 0.7
        else:  # high
            entry_spread = 0.008  # 0.8%
            target_multiplier = 3.0
            stop_loss_multiplier = 1.0
        
        # Calculate entry range
        entry_low = round(current_price * (1 - entry_spread), 6)
        entry_high = round(current_price * (1 + entry_spread), 6)
        
        # Get predicted volatility
        volatility = prediction['predicted_volatility_pct'] / 100
        
        # Calculate target and stop loss based on signal type and volatility
        if signal_type == 'BUY':
            # For buy signals, target is above entry, stop loss is below
            target_low = round(current_price * (1 + volatility * target_multiplier), 6)
            target_high = round(current_price * (1 + volatility * target_multiplier * 1.2), 6)
            stop_loss = round(current_price * (1 - volatility * stop_loss_multiplier), 6)
            
            # Calculate potential profit and risk percentages
            ward = round((target_low / current_price - 1) * 100, 1)
            risk = round((1 - stop_loss / current_price) * 100, 1)
        else:
            # For sell signals, target is below entry, stop loss is above
            target_high = round(current_price * (1 - volatility * target_multiplier), 6)
            target_low = round(current_price * (1 - volatility * target_multiplier * 1.2), 6)
            stop_loss = round(current_price * (1 + volatility * stop_loss_multiplier), 6)
            
            # Calculate potential profit and risk percentages
            ward = round((1 - target_high / current_price) * 100, 1)
            risk = round((stop_loss / current_price - 1) * 100, 1)
        
        # Estimate time based on timeframe and volatility
        time_estimate = self._estimate_time_for_target(timeframe, volatility)
        
        # Create signal
        signal = {
            'type': signal_type,
            'symbol': symbol,
            'timestamp': datetime.datetime.now(),
            'price': current_price,
            'entry_low': entry_low,
            'entry_high': entry_high,
            'target_low': target_low,
            'target_high': target_high,
            'stop_loss': stop_loss,
            'ward': ward,
            'risk': risk,
            'time': time_estimate,
            'confidence': prediction['confidence'],
            'predicted_direction': prediction['predicted_direction'],
            'predicted_change_pct': prediction['predicted_change_pct'],
            'signal_source': 'Advanced AI System'
        }
        
        return signal
    
    def _estimate_time_for_target(self, timeframe, volatility):
        """
        Estimate time to reach target based on timeframe and volatility.
        
        Args:
            timeframe (str): Timeframe (e.g., '1h', '4h', '1d')
            volatility (float): Predicted volatility as decimal
        
        Returns:
            str: Time estimate in format "HH:MM"
        """
        # Base time units in hours
        base_times = {
            '1m': 1/60,
            '5m': 5/60,
            '15m': 15/60,
            '30m': 30/60,
            '1h': 1,
            '4h': 4,
            '1d': 24
        }
        
        # Get base time for timeframe
        base_time = base_times.get(timeframe, 1)
        
        # Adjust based on volatility (higher volatility = faster moves)
        volatility_factor = 1 / max(volatility, 0.005)  # Avoid division by zero
        
        # Cap volatility factor to reasonable range
        volatility_factor = min(max(volatility_factor, 0.5), 5)
        
        # Calculate estimated time in hours
        estimated_hours = base_time * volatility_factor
        
        # Convert to hours and minutes
        hours = int(estimated_hours)
        minutes = int((estimated_hours - hours) * 60)
        
        # Format as HH:MM
        return f"{hours:02d}:{minutes:02d}"
    
    def evaluate_model_performance(self, symbol, timeframe, test_data):
        """
        Evaluate model performance on test data.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTC/USDT')
            timeframe (str): Timeframe (e.g., '1h', '4h', '1d')
            test_data (pd.DataFrame): Test data with actual outcomes
        
        Returns:
            dict: Performance metrics
        """
        model_key = f"{symbol}_{timeframe}"
        
        # Check if models exist
        if model_key not in self.models:
            # Try to load models
            if not self.load_models(symbol, timeframe):
                print(f"No models available for {model_key}")
                return None
        
        try:
            # Prepare data
            X, y_direction, y_price, _ = self._prepare_training_data(test_data)
            
            # Scale features
            X_scaled = self.scalers[model_key].transform(X)
            
            # Get predictions
            dir_pred = (self.models[model_key]['direction'].predict_proba(X_scaled)[:, 1] > 0.5).astype(int)
            price_pred = self.models[model_key]['price'].predict(X_scaled)
            
            # Calculate metrics
            accuracy = np.mean(dir_pred == y_direction)
            precision = np.sum((dir_pred == 1) & (y_direction == 1)) / max(np.sum(dir_pred == 1), 1)
            recall = np.sum((dir_pred == 1) & (y_direction == 1)) / max(np.sum(y_direction == 1), 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-10)
            mse = np.mean((price_pred - y_price) ** 2)
            mae = np.mean(np.abs(price_pred - y_price))
            
            # Calculate trading performance
            pnl = []
            position = None
            entry_price = None
            
            for i in range(len(dir_pred)):
                if i == 0:
                    continue
                
                # Simple trading simulation
                if position is None and dir_pred[i-1] == 1:  # Buy signal
                    position = 'long'
                    entry_price = test_data['open'].iloc[i]
                elif position is None and dir_pred[i-1] == 0:  # Sell signal
                    position = 'short'
                    entry_price = test_data['open'].iloc[i]
                elif position == 'long' and dir_pred[i-1] == 0:  # Exit long
                    pnl.append((test_data['open'].iloc[i] - entry_price) / entry_price * 100)
                    position = None
                elif position == 'short' and dir_pred[i-1] == 1:  # Exit short
                    pnl.append((entry_price - test_data['open'].iloc[i]) / entry_price * 100)
                    position = None
            
            # Calculate trading metrics
            if pnl:
                total_pnl = sum(pnl)
                win_rate = sum(1 for p in pnl if p > 0) / len(pnl)
                avg_win = sum(p for p in pnl if p > 0) / max(sum(1 for p in pnl if p > 0), 1)
                avg_loss = sum(p for p in pnl if p <= 0) / max(sum(1 for p in pnl if p <= 0), 1)
                profit_factor = abs(sum(p for p in pnl if p > 0) / sum(p for p in pnl if p <= 0)) if sum(p for p in pnl if p <= 0) != 0 else float('inf')
            else:
                total_pnl = 0
                win_rate = 0
                avg_win = 0
                avg_loss = 0
                profit_factor = 0
            
            # Return performance metrics
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'mse': mse,
                'mae': mae,
                'total_pnl': total_pnl,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'trade_count': len(pnl)
            }
        
        except Exception as e:
            print(f"Error evaluating model performance for {model_key}: {str(e)}")
            return None
    
    def auto_optimize(self, symbol, timeframe, historical_data):
        """
        Automatically optimize models for a specific symbol and timeframe.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTC/USDT')
            timeframe (str): Timeframe (e.g., '1h', '4h', '1d')
            historical_data (pd.DataFrame): Historical OHLCV data with indicators
        
        Returns:
            dict: Optimization results
        """
        print(f"Auto-optimizing models for {symbol} on {timeframe} timeframe...")
        
        # Train models with default parameters
        self.train_models(symbol, timeframe, historical_data, retrain=True)
        
        # Evaluate performance
        model_key = f"{symbol}_{timeframe}"
        
        # Prepare data for evaluation
        test_size = int(len(historical_data) * 0.2)
        test_data = historical_data.iloc[-test_size:]
        
        # Evaluate performance
        performance = self.evaluate_model_performance(symbol, timeframe, test_data)
        
        if performance:
            print(f"Initial performance for {model_key}:")
            for metric, value in performance.items():
                print(f"  {metric}: {value}")
        
        # Return optimization results
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'performance': performance,
            'optimization_time': datetime.datetime.now()
        }
