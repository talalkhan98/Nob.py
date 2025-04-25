import os
import json
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import logging
from datetime import datetime, timedelta
import threading
import random
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns

class AutonomousSelfOptimizationSystem:
    """
    Advanced autonomous self-optimization system that continuously improves
    trading strategies, parameters, and models based on performance feedback.
    """
    
    def __init__(self):
        """Initialize the AutonomousSelfOptimizationSystem."""
        self.strategies = {}
        self.parameters = {}
        self.performance_history = {}
        self.optimization_history = {}
        self.running = False
        self.thread = None
        self.optimization_interval = 86400  # 24 hours
        self.last_optimization = {}
        self.market_conditions = {}
        self.reinforcement_models = {}
        self.genetic_population = {}
        self.evolution_stats = {}
        
        # Create directories if they don't exist
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.models_dir = os.path.join(base_dir, 'models', 'optimization')
        self.data_dir = os.path.join(base_dir, 'data', 'optimization')
        self.logs_dir = os.path.join(base_dir, 'logs')
        
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Configure logging
        self.log_file = os.path.join(self.logs_dir, 'optimization.log')
        
        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('optimization')
        
        # Initialize TensorFlow for reinforcement learning
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
                self.logger.info("GPU acceleration enabled for optimization")
            except RuntimeError as e:
                self.logger.error(f"GPU configuration error: {e}")
    
    def register_strategy(self, strategy_id, strategy_config):
        """
        Register a trading strategy for optimization.
        
        Args:
            strategy_id (str): Unique identifier for the strategy
            strategy_config (dict): Strategy configuration
        
        Returns:
            bool: Success status
        """
        try:
            # Validate strategy configuration
            required_fields = ['name', 'type', 'parameters', 'default_values']
            for field in required_fields:
                if field not in strategy_config:
                    self.logger.error(f"Missing required field '{field}' in strategy configuration")
                    return False
            
            # Store strategy
            self.strategies[strategy_id] = strategy_config
            
            # Initialize parameters with default values
            self.parameters[strategy_id] = strategy_config['default_values'].copy()
            
            # Initialize performance history
            self.performance_history[strategy_id] = []
            
            # Initialize optimization history
            self.optimization_history[strategy_id] = []
            
            # Initialize last optimization time
            self.last_optimization[strategy_id] = 0
            
            self.logger.info(f"Registered strategy '{strategy_id}' for optimization")
            return True
        except Exception as e:
            self.logger.error(f"Error registering strategy '{strategy_id}': {str(e)}")
            return False
    
    def update_strategy_performance(self, strategy_id, performance_metrics):
        """
        Update performance metrics for a strategy.
        
        Args:
            strategy_id (str): Strategy identifier
            performance_metrics (dict): Performance metrics
        
        Returns:
            bool: Success status
        """
        try:
            # Check if strategy exists
            if strategy_id not in self.strategies:
                self.logger.error(f"Strategy '{strategy_id}' not found")
                return False
            
            # Validate performance metrics
            required_metrics = ['profit_pct', 'win_rate', 'trade_count', 'timestamp']
            for metric in required_metrics:
                if metric not in performance_metrics:
                    self.logger.error(f"Missing required metric '{metric}' in performance metrics")
                    return False
            
            # Add current parameters to performance metrics
            performance_metrics['parameters'] = self.parameters[strategy_id].copy()
            
            # Store performance metrics
            self.performance_history[strategy_id].append(performance_metrics)
            
            # Keep only last 100 performance records
            if len(self.performance_history[strategy_id]) > 100:
                self.performance_history[strategy_id] = self.performance_history[strategy_id][-100:]
            
            # Check if optimization is needed
            self._check_optimization_needed(strategy_id)
            
            self.logger.info(f"Updated performance for strategy '{strategy_id}': Profit={performance_metrics['profit_pct']:.2f}%, Win Rate={performance_metrics['win_rate']:.2f}%")
            return True
        except Exception as e:
            self.logger.error(f"Error updating performance for strategy '{strategy_id}': {str(e)}")
            return False
    
    def update_market_conditions(self, symbol, market_data):
        """
        Update market conditions for optimization context.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTC/USDT')
            market_data (dict): Market condition data
        
        Returns:
            bool: Success status
        """
        try:
            # Store market conditions
            self.market_conditions[symbol] = market_data
            
            # Add timestamp if not present
            if 'timestamp' not in market_data:
                self.market_conditions[symbol]['timestamp'] = datetime.now().isoformat()
            
            self.logger.info(f"Updated market conditions for {symbol}")
            return True
        except Exception as e:
            self.logger.error(f"Error updating market conditions for {symbol}: {str(e)}")
            return False
    
    def start_optimization_service(self, interval=86400):
        """
        Start the autonomous optimization service.
        
        Args:
            interval (int): Optimization check interval in seconds
        
        Returns:
            bool: Success status
        """
        if self.running:
            self.logger.info("Optimization service is already running")
            return False
        
        self.optimization_interval = interval
        self.running = True
        
        # Start optimization thread
        self.thread = threading.Thread(target=self._optimization_loop)
        self.thread.daemon = True
        self.thread.start()
        
        self.logger.info(f"Started autonomous optimization service with interval {interval} seconds")
        return True
    
    def stop_optimization_service(self):
        """Stop the autonomous optimization service."""
        if not self.running:
            self.logger.info("Optimization service is not running")
            return False
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        
        self.logger.info("Stopped autonomous optimization service")
        return True
    
    def _optimization_loop(self):
        """Internal optimization loop."""
        while self.running:
            try:
                # Check all strategies for optimization
                for strategy_id in self.strategies:
                    self._check_optimization_needed(strategy_id, force_check=True)
                
                # Sleep until next check
                time.sleep(60)  # Check every minute
            except Exception as e:
                self.logger.error(f"Error in optimization loop: {str(e)}")
                time.sleep(60)
    
    def _check_optimization_needed(self, strategy_id, force_check=False):
        """
        Check if optimization is needed for a strategy.
        
        Args:
            strategy_id (str): Strategy identifier
            force_check (bool): Whether to force check regardless of time
        """
        try:
            # Get current time
            current_time = time.time()
            
            # Check if enough time has passed since last optimization
            time_since_last = current_time - self.last_optimization.get(strategy_id, 0)
            
            if force_check or time_since_last >= self.optimization_interval:
                # Check if enough performance data is available
                if len(self.performance_history[strategy_id]) >= 10:
                    # Run optimization in a separate thread
                    optimization_thread = threading.Thread(
                        target=self._optimize_strategy_parameters,
                        args=(strategy_id,)
                    )
                    optimization_thread.daemon = True
                    optimization_thread.start()
                    
                    # Update last optimization time
                    self.last_optimization[strategy_id] = current_time
                else:
                    self.logger.info(f"Not enough performance data for strategy '{strategy_id}' optimization")
        except Exception as e:
            self.logger.error(f"Error checking optimization for strategy '{strategy_id}': {str(e)}")
    
    def _optimize_strategy_parameters(self, strategy_id):
        """
        Optimize parameters for a strategy.
        
        Args:
            strategy_id (str): Strategy identifier
        """
        try:
            self.logger.info(f"Starting optimization for strategy '{strategy_id}'")
            
            # Get strategy configuration
            strategy_config = self.strategies[strategy_id]
            strategy_type = strategy_config['type']
            
            # Choose optimization method based on strategy type and available data
            performance_data = self.performance_history[strategy_id]
            
            if len(performance_data) >= 50:
                # Use reinforcement learning for rich data
                optimized_params = self._optimize_with_reinforcement_learning(strategy_id, performance_data)
            elif len(performance_data) >= 20:
                # Use genetic algorithm for moderate data
                optimized_params = self._optimize_with_genetic_algorithm(strategy_id, performance_data)
            else:
                # Use grid search for limited data
                optimized_params = self._optimize_with_grid_search(strategy_id, performance_data)
            
            if optimized_params:
                # Calculate expected improvement
                expected_improvement = self._calculate_expected_improvement(
                    strategy_id, self.parameters[strategy_id], optimized_params
                )
                
                # Apply optimized parameters if improvement is significant
                if expected_improvement > 0.05:  # 5% improvement threshold
                    old_params = self.parameters[strategy_id].copy()
                    self.parameters[strategy_id] = optimized_params
                    
                    # Record optimization
                    self.optimization_history[strategy_id].append({
                        'timestamp': datetime.now().isoformat(),
                        'old_parameters': old_params,
                        'new_parameters': optimized_params,
                        'expected_improvement': expected_improvement,
                        'method': 'reinforcement_learning' if len(performance_data) >= 50 else 
                                 'genetic_algorithm' if len(performance_data) >= 20 else 'grid_search'
                    })
                    
                    self.logger.info(f"Optimized parameters for strategy '{strategy_id}' with expected improvement {expected_improvement:.2f}")
                else:
                    self.logger.info(f"No significant improvement found for strategy '{strategy_id}'")
            else:
                self.logger.warning(f"Optimization failed for strategy '{strategy_id}'")
        except Exception as e:
            self.logger.error(f"Error optimizing strategy '{strategy_id}': {str(e)}")
    
    def _optimize_with_reinforcement_learning(self, strategy_id, performance_data):
        """
        Optimize parameters using reinforcement learning.
        
        Args:
            strategy_id (str): Strategy identifier
            performance_data (list): Performance history
        
        Returns:
            dict: Optimized parameters
        """
        try:
            self.logger.info(f"Using reinforcement learning to optimize strategy '{strategy_id}'")
            
            # Get strategy configuration
            strategy_config = self.strategies[strategy_id]
            parameters_config = strategy_config['parameters']
            current_params = self.parameters[strategy_id]
            
            # Check if model exists, create if not
            if strategy_id not in self.reinforcement_models:
                self._create_reinforcement_model(strategy_id, parameters_config)
            
            # Prepare training data
            states, actions, rewards = self._prepare_rl_training_data(strategy_id, performance_data)
            
            if not states or not actions or not rewards:
                self.logger.warning(f"Insufficient data for reinforcement learning optimization of '{strategy_id}'")
                return None
            
            # Train model
            model = self.reinforcement_models[strategy_id]
            
            # Convert to numpy arrays
            states_np = np.array(states)
            actions_np = np.array(actions)
            rewards_np = np.array(rewards)
            
            # Normalize rewards
            rewards_mean = np.mean(rewards_np)
            rewards_std = np.std(rewards_np) if np.std(rewards_np) > 0 else 1.0
            normalized_rewards = (rewards_np - rewards_mean) / rewards_std
            
            # Train for multiple epochs
            history = model.fit(
                states_np, 
                actions_np, 
                sample_weight=normalized_rewards,
                epochs=50,
                batch_size=min(32, len(states_np)),
                verbose=0
            )
            
            # Save model
            model_path = os.path.join(self.models_dir, f"{strategy_id}_rl_model")
            model.save(model_path)
            
            # Generate optimized parameters
            # Create a state vector from current market conditions and recent performance
            current_state = self._create_state_vector(strategy_id)
            
            if current_state is not None:
                # Predict optimal parameter adjustments
                predicted_actions = model.predict(np.array([current_state]), verbose=0)[0]
                
                # Apply actions to current parameters
                optimized_params = {}
                
                for i, (param_name, param_config) in enumerate(parameters_config.items()):
                    param_type = param_config.get('type', 'float')
                    param_min = param_config.get('min', 0.0)
                    param_max = param_config.get('max', 1.0)
                    
                    # Get current value
                    current_value = current_params.get(param_name, param_config.get('default', 0.0))
                    
                    # Calculate new value based on predicted action
                    # Actions are in range [-1, 1], scale to parameter range
                    action_scale = predicted_actions[i]
                    param_range = param_max - param_min
                    
                    # Apply scaled action to current value
                    new_value = current_value + action_scale * param_range * 0.1  # 10% max adjustment
                    
                    # Ensure value is within bounds
                    new_value = max(param_min, min(param_max, new_value))
                    
                    # Convert to appropriate type
                    if param_type == 'int':
                        new_value = int(round(new_value))
                    elif param_type == 'bool':
                        new_value = bool(new_value > 0.5)
                    
                    optimized_params[param_name] = new_value
                
                self.logger.info(f"Reinforcement learning optimization completed for '{strategy_id}'")
                return optimized_params
            else:
                self.logger.warning(f"Could not create state vector for '{strategy_id}'")
                return None
        except Exception as e:
            self.logger.error(f"Error in reinforcement learning optimization for '{strategy_id}': {str(e)}")
            return None
    
    def _create_reinforcement_model(self, strategy_id, parameters_config):
        """
        Create a reinforcement learning model for parameter optimization.
        
        Args:
            strategy_id (str): Strategy identifier
            parameters_config (dict): Parameters configuration
        """
        try:
            # Determine input and output dimensions
            state_dim = 20  # Fixed size for state representation
            action_dim = len(parameters_config)  # One output per parameter
            
            # Create model
            model = Sequential([
                Dense(64, activation='relu', input_shape=(state_dim,)),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dropout(0.2),
                Dense(action_dim, activation='tanh')  # tanh for [-1, 1] range
            ])
            
            # Compile model
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            
            # Store model
            self.reinforcement_models[strategy_id] = model
            
            self.logger.info(f"Created reinforcement learning model for strategy '{strategy_id}'")
        except Exception as e:
            self.logger.error(f"Error creating reinforcement model for '{strategy_id}': {str(e)}")
    
    def _prepare_rl_training_data(self, strategy_id, performance_data):
        """
        Prepare training data for reinforcement learning.
        
        Args:
            strategy_id (str): Strategy identifier
            performance_data (list): Performance history
        
        Returns:
            tuple: (states, actions, rewards)
        """
        try:
            states = []
            actions = []
            rewards = []
            
            # Need at least 2 data points
            if len(performance_data) < 2:
                return None, None, None
            
            # Get parameters configuration
            parameters_config = self.strategies[strategy_id]['parameters']
            
            # Process performance data
            for i in range(1, len(performance_data)):
                prev_data = performance_data[i-1]
                curr_data = performance_data[i]
                
                # Create state vector from previous performance and market conditions
                state = self._create_state_vector_from_performance(prev_data)
                
                if state is None:
                    continue
                
                # Calculate parameter changes as actions
                action = []
                
                for param_name in parameters_config:
                    prev_value = prev_data['parameters'].get(param_name, 0.0)
                    curr_value = curr_data['parameters'].get(param_name, 0.0)
                    
                    param_min = parameters_config[param_name].get('min', 0.0)
                    param_max = parameters_config[param_name].get('max', 1.0)
                    param_range = param_max - param_min
                    
                    # Normalize change to [-1, 1]
                    if param_range > 0:
                        normalized_change = (curr_value - prev_value) / (param_range * 0.1)  # 10% max adjustment
                        normalized_change = max(-1.0, min(1.0, normalized_change))
                    else:
                        normalized_change = 0.0
                    
                    action.append(normalized_change)
                
                # Use profit as reward
                reward = curr_data.get('profit_pct', 0.0)
                
                # Store data
                states.append(state)
                actions.append(action)
                rewards.append(reward)
            
            return states, actions, rewards
        except Exception as e:
            self.logger.error(f"Error preparing RL training data for '{strategy_id}': {str(e)}")
            return None, None, None
    
    def _create_state_vector_from_performance(self, performance_data):
        """
        Create a state vector from performance data.
        
        Args:
            performance_data (dict): Performance data
        
        Returns:
            list: State vector
        """
        try:
            # Extract relevant metrics
            profit_pct = performance_data.get('profit_pct', 0.0)
            win_rate = performance_data.get('win_rate', 0.0)
            trade_count = performance_data.get('trade_count', 0)
            avg_profit = performance_data.get('avg_profit', 0.0)
            avg_loss = performance_data.get('avg_loss', 0.0)
            max_drawdown = performance_data.get('max_drawdown', 0.0)
            
            # Normalize values
            normalized_profit = profit_pct / 100.0  # Assuming profit is in percentage
            normalized_win_rate = win_rate / 100.0
            normalized_trade_count = min(1.0, trade_count / 100.0)  # Cap at 100 trades
            normalized_avg_profit = avg_profit / 10.0  # Assuming average profit is in percentage
            normalized_avg_loss = avg_loss / 10.0
            normalized_drawdown = max_drawdown / 50.0  # Assuming drawdown is in percentage
            
            # Create fixed-size state vector with padding
            state = [
                normalized_profit,
                normalized_win_rate,
                normalized_trade_count,
                normalized_avg_profit,
                normalized_avg_loss,
                normalized_drawdown
            ]
            
            # Add market condition features if available
            market_features = performance_data.get('market_conditions', {})
            
            # Extract market features
            volatility = market_features.get('volatility', 0.0)
            trend_strength = market_features.get('trend_strength', 0.0)
            volume = market_features.get('volume', 0.0)
            rsi = market_features.get('rsi', 50.0)
            
            # Normalize market features
            normalized_volatility = min(1.0, volatility / 5.0)  # Cap at 5%
            normalized_trend = trend_strength / 100.0
            normalized_volume = min(1.0, volume / 1000000.0)  # Cap at 1M
            normalized_rsi = rsi / 100.0
            
            # Add to state vector
            state.extend([
                normalized_volatility,
                normalized_trend,
                normalized_volume,
                normalized_rsi
            ])
            
            # Pad to fixed size
            while len(state) < 20:
                state.append(0.0)
            
            # Truncate if too long
            if len(state) > 20:
                state = state[:20]
            
            return state
        except Exception as e:
            self.logger.error(f"Error creating state vector from performance: {str(e)}")
            return None
    
    def _create_state_vector(self, strategy_id):
        """
        Create a state vector from current conditions.
        
        Args:
            strategy_id (str): Strategy identifier
        
        Returns:
            list: State vector
        """
        try:
            # Get recent performance
            performance_history = self.performance_history[strategy_id]
            
            if not performance_history:
                return None
            
            # Use most recent performance data
            recent_performance = performance_history[-1]
            
            # Create state vector
            return self._create_state_vector_from_performance(recent_performance)
        except Exception as e:
            self.logger.error(f"Error creating state vector for '{strategy_id}': {str(e)}")
            return None
    
    def _optimize_with_genetic_algorithm(self, strategy_id, performance_data):
        """
        Optimize parameters using a genetic algorithm.
        
        Args:
            strategy_id (str): Strategy identifier
            performance_data (list): Performance history
        
        Returns:
            dict: Optimized parameters
        """
        try:
            self.logger.info(f"Using genetic algorithm to optimize strategy '{strategy_id}'")
            
            # Get strategy configuration
            strategy_config = self.strategies[strategy_id]
            parameters_config = strategy_config['parameters']
            current_params = self.parameters[strategy_id]
            
            # Initialize population if not exists
            if strategy_id not in self.genetic_population:
                self._initialize_genetic_population(strategy_id)
            
            # Get population
            population = self.genetic_population[strategy_id]
            
            # Evaluate fitness of population
            fitness_scores = self._evaluate_genetic_population(strategy_id, population, performance_data)
            
            # Run genetic algorithm for multiple generations
            for generation in range(10):
                # Select parents
                parents = self._select_genetic_parents(population, fitness_scores)
                
                # Create offspring
                offspring = self._create_genetic_offspring(parents, parameters_config)
                
                # Mutate offspring
                mutated_offspring = self._mutate_genetic_offspring(offspring, parameters_config)
                
                # Evaluate fitness of offspring
                offspring_fitness = self._evaluate_genetic_population(strategy_id, mutated_offspring, performance_data)
                
                # Combine populations
                combined_population = population + mutated_offspring
                combined_fitness = fitness_scores + offspring_fitness
                
                # Select next generation
                population, fitness_scores = self._select_next_generation(combined_population, combined_fitness)
                
                # Store updated population
                self.genetic_population[strategy_id] = population
                
                # Store evolution stats
                if strategy_id not in self.evolution_stats:
                    self.evolution_stats[strategy_id] = []
                
                self.evolution_stats[strategy_id].append({
                    'generation': generation,
                    'max_fitness': max(fitness_scores),
                    'avg_fitness': sum(fitness_scores) / len(fitness_scores),
                    'min_fitness': min(fitness_scores),
                    'timestamp': datetime.now().isoformat()
                })
            
            # Get best individual
            best_index = fitness_scores.index(max(fitness_scores))
            best_individual = population[best_index]
            
            self.logger.info(f"Genetic algorithm optimization completed for '{strategy_id}'")
            return best_individual
        except Exception as e:
            self.logger.error(f"Error in genetic algorithm optimization for '{strategy_id}': {str(e)}")
            return None
    
    def _initialize_genetic_population(self, strategy_id, population_size=20):
        """
        Initialize genetic algorithm population.
        
        Args:
            strategy_id (str): Strategy identifier
            population_size (int): Population size
        """
        try:
            # Get strategy configuration
            strategy_config = self.strategies[strategy_id]
            parameters_config = strategy_config['parameters']
            current_params = self.parameters[strategy_id]
            
            # Create population
            population = []
            
            # First individual is current parameters
            population.append(current_params.copy())
            
            # Create random individuals
            for _ in range(population_size - 1):
                individual = {}
                
                for param_name, param_config in parameters_config.items():
                    param_type = param_config.get('type', 'float')
                    param_min = param_config.get('min', 0.0)
                    param_max = param_config.get('max', 1.0)
                    
                    # Generate random value
                    if param_type == 'int':
                        value = random.randint(param_min, param_max)
                    elif param_type == 'bool':
                        value = random.choice([True, False])
                    else:  # float
                        value = param_min + random.random() * (param_max - param_min)
                    
                    individual[param_name] = value
                
                population.append(individual)
            
            # Store population
            self.genetic_population[strategy_id] = population
            
            self.logger.info(f"Initialized genetic population for strategy '{strategy_id}'")
        except Exception as e:
            self.logger.error(f"Error initializing genetic population for '{strategy_id}': {str(e)}")
    
    def _evaluate_genetic_population(self, strategy_id, population, performance_data):
        """
        Evaluate fitness of genetic population.
        
        Args:
            strategy_id (str): Strategy identifier
            population (list): Population of parameter sets
            performance_data (list): Performance history
        
        Returns:
            list: Fitness scores
        """
        try:
            fitness_scores = []
            
            for individual in population:
                # Calculate fitness based on similarity to high-performing parameter sets
                fitness = self._calculate_parameter_fitness(strategy_id, individual, performance_data)
                fitness_scores.append(fitness)
            
            return fitness_scores
        except Exception as e:
            self.logger.error(f"Error evaluating genetic population for '{strategy_id}': {str(e)}")
            return [0.0] * len(population)
    
    def _calculate_parameter_fitness(self, strategy_id, parameters, performance_data):
        """
        Calculate fitness of a parameter set.
        
        Args:
            strategy_id (str): Strategy identifier
            parameters (dict): Parameter set
            performance_data (list): Performance history
        
        Returns:
            float: Fitness score
        """
        try:
            # Get strategy configuration
            strategy_config = self.strategies[strategy_id]
            parameters_config = strategy_config['parameters']
            
            # Sort performance data by profit
            sorted_performance = sorted(
                performance_data,
                key=lambda x: x.get('profit_pct', 0.0),
                reverse=True
            )
            
            # Take top 30% of performance data
            top_count = max(1, int(len(sorted_performance) * 0.3))
            top_performance = sorted_performance[:top_count]
            
            # Calculate similarity to top-performing parameter sets
            similarity_scores = []
            
            for perf_data in top_performance:
                perf_params = perf_data.get('parameters', {})
                
                # Calculate parameter similarity
                param_similarity = 0.0
                param_count = 0
                
                for param_name, param_config in parameters_config.items():
                    if param_name in parameters and param_name in perf_params:
                        param_type = param_config.get('type', 'float')
                        param_min = param_config.get('min', 0.0)
                        param_max = param_config.get('max', 1.0)
                        
                        # Get values
                        value1 = parameters[param_name]
                        value2 = perf_params[param_name]
                        
                        # Calculate normalized similarity
                        if param_type == 'bool':
                            similarity = 1.0 if value1 == value2 else 0.0
                        else:
                            # Normalize values
                            range_size = param_max - param_min
                            if range_size > 0:
                                norm_value1 = (value1 - param_min) / range_size
                                norm_value2 = (value2 - param_min) / range_size
                                
                                # Calculate similarity (1 - normalized distance)
                                similarity = 1.0 - abs(norm_value1 - norm_value2)
                            else:
                                similarity = 1.0 if value1 == value2 else 0.0
                        
                        param_similarity += similarity
                        param_count += 1
                
                # Calculate average similarity
                avg_similarity = param_similarity / param_count if param_count > 0 else 0.0
                
                # Weight by performance
                perf_weight = perf_data.get('profit_pct', 0.0) / 100.0  # Normalize profit
                weighted_similarity = avg_similarity * (1.0 + perf_weight)
                
                similarity_scores.append(weighted_similarity)
            
            # Calculate overall fitness
            fitness = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
            
            # Add exploration bonus for diversity
            exploration_bonus = random.random() * 0.1  # 10% random bonus
            fitness += exploration_bonus
            
            return fitness
        except Exception as e:
            self.logger.error(f"Error calculating parameter fitness for '{strategy_id}': {str(e)}")
            return 0.0
    
    def _select_genetic_parents(self, population, fitness_scores, num_parents=10):
        """
        Select parents for genetic algorithm.
        
        Args:
            population (list): Population of parameter sets
            fitness_scores (list): Fitness scores
            num_parents (int): Number of parents to select
        
        Returns:
            list: Selected parents
        """
        try:
            # Ensure valid inputs
            if not population or not fitness_scores or len(population) != len(fitness_scores):
                return []
            
            # Create selection probabilities
            total_fitness = sum(fitness_scores)
            if total_fitness <= 0:
                # If all fitness scores are zero or negative, use uniform selection
                selection_probs = [1.0 / len(population)] * len(population)
            else:
                selection_probs = [score / total_fitness for score in fitness_scores]
            
            # Select parents using roulette wheel selection
            parent_indices = np.random.choice(
                len(population),
                size=min(num_parents, len(population)),
                replace=False,
                p=selection_probs
            )
            
            parents = [population[i] for i in parent_indices]
            return parents
        except Exception as e:
            self.logger.error(f"Error selecting genetic parents: {str(e)}")
            return []
    
    def _create_genetic_offspring(self, parents, parameters_config):
        """
        Create offspring through crossover.
        
        Args:
            parents (list): Parent parameter sets
            parameters_config (dict): Parameters configuration
        
        Returns:
            list: Offspring parameter sets
        """
        try:
            offspring = []
            
            # Create offspring pairs
            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):
                    parent1 = parents[i]
                    parent2 = parents[i + 1]
                    
                    # Create two children
                    child1 = {}
                    child2 = {}
                    
                    # Perform crossover for each parameter
                    for param_name in parameters_config:
                        # Randomly select which parent to inherit from
                        if random.random() < 0.5:
                            child1[param_name] = parent1.get(param_name)
                            child2[param_name] = parent2.get(param_name)
                        else:
                            child1[param_name] = parent2.get(param_name)
                            child2[param_name] = parent1.get(param_name)
                    
                    offspring.append(child1)
                    offspring.append(child2)
            
            return offspring
        except Exception as e:
            self.logger.error(f"Error creating genetic offspring: {str(e)}")
            return []
    
    def _mutate_genetic_offspring(self, offspring, parameters_config, mutation_rate=0.2):
        """
        Mutate offspring.
        
        Args:
            offspring (list): Offspring parameter sets
            parameters_config (dict): Parameters configuration
            mutation_rate (float): Mutation rate
        
        Returns:
            list: Mutated offspring
        """
        try:
            mutated_offspring = []
            
            for individual in offspring:
                mutated = individual.copy()
                
                # Mutate each parameter with probability mutation_rate
                for param_name, param_config in parameters_config.items():
                    if random.random() < mutation_rate:
                        param_type = param_config.get('type', 'float')
                        param_min = param_config.get('min', 0.0)
                        param_max = param_config.get('max', 1.0)
                        
                        # Generate mutated value
                        if param_type == 'int':
                            mutated[param_name] = random.randint(param_min, param_max)
                        elif param_type == 'bool':
                            mutated[param_name] = not mutated.get(param_name, False)
                        else:  # float
                            # Small random adjustment
                            current_value = mutated.get(param_name, 0.0)
                            range_size = param_max - param_min
                            adjustment = (random.random() - 0.5) * range_size * 0.2  # 20% max adjustment
                            new_value = current_value + adjustment
                            mutated[param_name] = max(param_min, min(param_max, new_value))
                
                mutated_offspring.append(mutated)
            
            return mutated_offspring
        except Exception as e:
            self.logger.error(f"Error mutating genetic offspring: {str(e)}")
            return offspring
    
    def _select_next_generation(self, population, fitness_scores, population_size=20):
        """
        Select next generation.
        
        Args:
            population (list): Combined population
            fitness_scores (list): Fitness scores
            population_size (int): Target population size
        
        Returns:
            tuple: (selected_population, selected_fitness)
        """
        try:
            # Ensure valid inputs
            if not population or not fitness_scores or len(population) != len(fitness_scores):
                return [], []
            
            # Sort by fitness
            sorted_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)
            
            # Select top individuals (elitism)
            elite_count = max(1, int(population_size * 0.2))  # 20% elitism
            elite_indices = sorted_indices[:elite_count]
            
            # Select remaining individuals using roulette wheel
            remaining_count = population_size - elite_count
            
            if remaining_count > 0 and len(sorted_indices) > elite_count:
                # Get remaining indices
                remaining_indices = sorted_indices[elite_count:]
                
                # Get fitness scores for remaining individuals
                remaining_fitness = [fitness_scores[i] for i in remaining_indices]
                
                # Create selection probabilities
                total_fitness = sum(remaining_fitness)
                if total_fitness <= 0:
                    # If all fitness scores are zero or negative, use uniform selection
                    selection_probs = [1.0 / len(remaining_indices)] * len(remaining_indices)
                else:
                    selection_probs = [score / total_fitness for score in remaining_fitness]
                
                # Select remaining individuals
                selected_remaining = np.random.choice(
                    remaining_indices,
                    size=min(remaining_count, len(remaining_indices)),
                    replace=False,
                    p=selection_probs
                )
                
                # Combine elite and selected
                selected_indices = list(elite_indices) + list(selected_remaining)
            else:
                selected_indices = elite_indices
            
            # Create new population and fitness scores
            selected_population = [population[i] for i in selected_indices]
            selected_fitness = [fitness_scores[i] for i in selected_indices]
            
            return selected_population, selected_fitness
        except Exception as e:
            self.logger.error(f"Error selecting next generation: {str(e)}")
            return population[:population_size], fitness_scores[:population_size]
    
    def _optimize_with_grid_search(self, strategy_id, performance_data):
        """
        Optimize parameters using grid search.
        
        Args:
            strategy_id (str): Strategy identifier
            performance_data (list): Performance history
        
        Returns:
            dict: Optimized parameters
        """
        try:
            self.logger.info(f"Using grid search to optimize strategy '{strategy_id}'")
            
            # Get strategy configuration
            strategy_config = self.strategies[strategy_id]
            parameters_config = strategy_config['parameters']
            current_params = self.parameters[strategy_id]
            
            # Create parameter grid
            param_grid = {}
            
            for param_name, param_config in parameters_config.items():
                param_type = param_config.get('type', 'float')
                param_min = param_config.get('min', 0.0)
                param_max = param_config.get('max', 1.0)
                
                current_value = current_params.get(param_name, param_config.get('default', 0.0))
                
                # Create grid values
                if param_type == 'int':
                    # For integers, create a small range around current value
                    grid_min = max(param_min, current_value - 3)
                    grid_max = min(param_max, current_value + 3)
                    param_grid[param_name] = list(range(int(grid_min), int(grid_max) + 1))
                elif param_type == 'bool':
                    # For booleans, try both values
                    param_grid[param_name] = [True, False]
                else:  # float
                    # For floats, create a small range around current value
                    range_size = param_max - param_min
                    step_size = range_size / 10.0
                    
                    grid_min = max(param_min, current_value - step_size)
                    grid_max = min(param_max, current_value + step_size)
                    
                    # Create 5 values in the range
                    param_grid[param_name] = [
                        grid_min + i * (grid_max - grid_min) / 4.0
                        for i in range(5)
                    ]
            
            # Generate grid combinations
            grid_combinations = self._generate_grid_combinations(param_grid)
            
            # Limit to 100 combinations to avoid excessive computation
            if len(grid_combinations) > 100:
                grid_combinations = random.sample(grid_combinations, 100)
            
            # Evaluate each combination
            best_fitness = -float('inf')
            best_params = None
            
            for params in grid_combinations:
                fitness = self._calculate_parameter_fitness(strategy_id, params, performance_data)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_params = params
            
            self.logger.info(f"Grid search optimization completed for '{strategy_id}'")
            return best_params
        except Exception as e:
            self.logger.error(f"Error in grid search optimization for '{strategy_id}': {str(e)}")
            return None
    
    def _generate_grid_combinations(self, param_grid):
        """
        Generate all combinations from parameter grid.
        
        Args:
            param_grid (dict): Parameter grid
        
        Returns:
            list: List of parameter combinations
        """
        try:
            # Get parameter names and values
            param_names = list(param_grid.keys())
            param_values = list(param_grid.values())
            
            # Generate all combinations
            combinations = []
            
            # Recursive helper function
            def generate_combinations(index, current_params):
                if index == len(param_names):
                    combinations.append(current_params.copy())
                    return
                
                param_name = param_names[index]
                for value in param_values[index]:
                    current_params[param_name] = value
                    generate_combinations(index + 1, current_params)
            
            # Start recursion
            generate_combinations(0, {})
            
            return combinations
        except Exception as e:
            self.logger.error(f"Error generating grid combinations: {str(e)}")
            return []
    
    def _calculate_expected_improvement(self, strategy_id, old_params, new_params):
        """
        Calculate expected improvement from parameter changes.
        
        Args:
            strategy_id (str): Strategy identifier
            old_params (dict): Old parameters
            new_params (dict): New parameters
        
        Returns:
            float: Expected improvement
        """
        try:
            # Get performance history
            performance_history = self.performance_history[strategy_id]
            
            if not performance_history:
                return 0.0
            
            # Get strategy configuration
            strategy_config = self.strategies[strategy_id]
            parameters_config = strategy_config['parameters']
            
            # Calculate parameter similarity to historical high performers
            old_fitness = self._calculate_parameter_fitness(strategy_id, old_params, performance_history)
            new_fitness = self._calculate_parameter_fitness(strategy_id, new_params, performance_history)
            
            # Calculate improvement
            improvement = new_fitness - old_fitness
            
            return max(0.0, improvement)
        except Exception as e:
            self.logger.error(f"Error calculating expected improvement for '{strategy_id}': {str(e)}")
            return 0.0
    
    def get_optimized_parameters(self, strategy_id):
        """
        Get optimized parameters for a strategy.
        
        Args:
            strategy_id (str): Strategy identifier
        
        Returns:
            dict: Optimized parameters
        """
        try:
            # Check if strategy exists
            if strategy_id not in self.strategies:
                self.logger.error(f"Strategy '{strategy_id}' not found")
                return None
            
            return self.parameters[strategy_id]
        except Exception as e:
            self.logger.error(f"Error getting optimized parameters for '{strategy_id}': {str(e)}")
            return None
    
    def get_optimization_history(self, strategy_id):
        """
        Get optimization history for a strategy.
        
        Args:
            strategy_id (str): Strategy identifier
        
        Returns:
            list: Optimization history
        """
        try:
            # Check if strategy exists
            if strategy_id not in self.strategies:
                self.logger.error(f"Strategy '{strategy_id}' not found")
                return []
            
            return self.optimization_history[strategy_id]
        except Exception as e:
            self.logger.error(f"Error getting optimization history for '{strategy_id}': {str(e)}")
            return []
    
    def get_evolution_stats(self, strategy_id):
        """
        Get evolution statistics for a strategy.
        
        Args:
            strategy_id (str): Strategy identifier
        
        Returns:
            list: Evolution statistics
        """
        try:
            # Check if strategy exists
            if strategy_id not in self.strategies:
                self.logger.error(f"Strategy '{strategy_id}' not found")
                return []
            
            return self.evolution_stats.get(strategy_id, [])
        except Exception as e:
            self.logger.error(f"Error getting evolution stats for '{strategy_id}': {str(e)}")
            return []
    
    def plot_optimization_progress(self, strategy_id):
        """
        Plot optimization progress for a strategy.
        
        Args:
            strategy_id (str): Strategy identifier
        
        Returns:
            str: Path to saved plot
        """
        try:
            # Check if strategy exists
            if strategy_id not in self.strategies:
                self.logger.error(f"Strategy '{strategy_id}' not found")
                return None
            
            # Get optimization history
            optimization_history = self.optimization_history[strategy_id]
            
            if not optimization_history:
                self.logger.warning(f"No optimization history for '{strategy_id}'")
                return None
            
            # Create DataFrame
            df = pd.DataFrame(optimization_history)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Create plot
            plt.figure(figsize=(12, 8))
            
            # Plot expected improvement
            plt.subplot(2, 1, 1)
            plt.plot(df.index, df['expected_improvement'], 'b-', marker='o')
            plt.title(f'Optimization Progress for {strategy_id}')
            plt.ylabel('Expected Improvement')
            plt.grid(True, alpha=0.3)
            
            # Plot optimization method
            plt.subplot(2, 1, 2)
            method_counts = df['method'].value_counts()
            plt.bar(method_counts.index, method_counts.values)
            plt.title('Optimization Methods Used')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            # Save plot
            plot_dir = os.path.join(self.data_dir, 'plots')
            os.makedirs(plot_dir, exist_ok=True)
            
            plot_file = os.path.join(plot_dir, f'{strategy_id}_optimization_progress.png')
            plt.savefig(plot_file)
            plt.close()
            
            return plot_file
        except Exception as e:
            self.logger.error(f"Error plotting optimization progress for '{strategy_id}': {str(e)}")
            return None
    
    def plot_evolution_progress(self, strategy_id):
        """
        Plot evolution progress for a strategy.
        
        Args:
            strategy_id (str): Strategy identifier
        
        Returns:
            str: Path to saved plot
        """
        try:
            # Check if strategy exists
            if strategy_id not in self.strategies:
                self.logger.error(f"Strategy '{strategy_id}' not found")
                return None
            
            # Get evolution stats
            evolution_stats = self.evolution_stats.get(strategy_id, [])
            
            if not evolution_stats:
                self.logger.warning(f"No evolution stats for '{strategy_id}'")
                return None
            
            # Create DataFrame
            df = pd.DataFrame(evolution_stats)
            
            # Create plot
            plt.figure(figsize=(12, 6))
            
            plt.plot(df['generation'], df['max_fitness'], 'g-', marker='o', label='Max Fitness')
            plt.plot(df['generation'], df['avg_fitness'], 'b-', marker='s', label='Avg Fitness')
            plt.plot(df['generation'], df['min_fitness'], 'r-', marker='^', label='Min Fitness')
            
            plt.title(f'Genetic Algorithm Evolution for {strategy_id}')
            plt.xlabel('Generation')
            plt.ylabel('Fitness')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save plot
            plot_dir = os.path.join(self.data_dir, 'plots')
            os.makedirs(plot_dir, exist_ok=True)
            
            plot_file = os.path.join(plot_dir, f'{strategy_id}_evolution_progress.png')
            plt.savefig(plot_file)
            plt.close()
            
            return plot_file
        except Exception as e:
            self.logger.error(f"Error plotting evolution progress for '{strategy_id}': {str(e)}")
            return None
