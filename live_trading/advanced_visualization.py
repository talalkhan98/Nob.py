import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

class AdvancedVisualization:
    """
    A class for creating advanced visualizations and automated chart drawings
    for the One Beyond All Crypto Trading Bot.
    """
    
    def __init__(self):
        """Initialize the AdvancedVisualization class."""
        self.color_scheme = {
            'background': '#0E1117',
            'text': '#FFFFFF',
            'grid': '#333333',
            'buy_signal': '#00FF7F',
            'sell_signal': '#FF4500',
            'support': '#4169E1',
            'resistance': '#FF6347',
            'trend_up': '#32CD32',
            'trend_down': '#DC143C',
            'volume_up': '#3CB371',
            'volume_down': '#CD5C5C',
            'prediction': '#FFD700'
        }
    
    def create_advanced_chart(self, df, signals=None, predictions=None, timeframe='1h'):
        """
        Create an advanced interactive chart with automated pattern recognition,
        support/resistance levels, and signal visualization.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV and indicator data
            signals (list): List of trading signals
            predictions (list): List of price predictions
            timeframe (str): Timeframe of the data
            
        Returns:
            go.Figure: Plotly figure with advanced visualization
        """
        # Create subplots with shared x-axis
        fig = make_subplots(
            rows=4, 
            cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=[0.5, 0.2, 0.15, 0.15],
            subplot_titles=("Price Chart", "Volume", "RSI", "MACD")
        )
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name="Price",
                increasing_line_color=self.color_scheme['trend_up'],
                decreasing_line_color=self.color_scheme['trend_down']
            ),
            row=1, col=1
        )
        
        # Add moving averages
        if 'sma_20' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['sma_20'],
                    name="SMA 20",
                    line=dict(color='rgba(255, 255, 255, 0.7)', width=1)
                ),
                row=1, col=1
            )
        
        if 'sma_50' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['sma_50'],
                    name="SMA 50",
                    line=dict(color='rgba(255, 165, 0, 0.7)', width=1.5)
                ),
                row=1, col=1
            )
        
        if 'sma_200' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['sma_200'],
                    name="SMA 200",
                    line=dict(color='rgba(255, 69, 0, 0.7)', width=2)
                ),
                row=1, col=1
            )
        
        # Add Bollinger Bands
        if 'bollinger_hband' in df.columns and 'bollinger_lband' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['bollinger_hband'],
                    name="Upper BB",
                    line=dict(color='rgba(250, 128, 114, 0.7)', width=1, dash='dash')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['bollinger_lband'],
                    name="Lower BB",
                    line=dict(color='rgba(135, 206, 235, 0.7)', width=1, dash='dash')
                ),
                row=1, col=1
            )
        
        # Add support and resistance levels
        support_levels, resistance_levels = self._identify_support_resistance(df)
        
        for level in support_levels:
            fig.add_shape(
                type="line",
                x0=df.index[0],
                y0=level,
                x1=df.index[-1],
                y1=level,
                line=dict(
                    color=self.color_scheme['support'],
                    width=1,
                    dash="dot",
                ),
                row=1, col=1
            )
        
        for level in resistance_levels:
            fig.add_shape(
                type="line",
                x0=df.index[0],
                y0=level,
                x1=df.index[-1],
                y1=level,
                line=dict(
                    color=self.color_scheme['resistance'],
                    width=1,
                    dash="dot",
                ),
                row=1, col=1
            )
        
        # Add chart patterns
        self._add_chart_patterns(fig, df)
        
        # Add trading signals
        if signals:
            self._add_trading_signals(fig, df, signals)
        
        # Add price predictions
        if predictions:
            self._add_price_predictions(fig, df, predictions)
        
        # Add volume chart
        colors = [self.color_scheme['volume_up'] if df['close'][i] > df['close'][i-1] 
                 else self.color_scheme['volume_down'] for i in range(1, len(df))]
        colors.insert(0, colors[0])  # Add color for first bar
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['volume'],
                name="Volume",
                marker_color=colors,
                opacity=0.8
            ),
            row=2, col=1
        )
        
        # Add RSI
        if 'rsi_14' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['rsi_14'],
                    name="RSI (14)",
                    line=dict(color='purple', width=1)
                ),
                row=3, col=1
            )
            
            # Add RSI overbought/oversold lines
            fig.add_shape(
                type="line",
                x0=df.index[0],
                y0=70,
                x1=df.index[-1],
                y1=70,
                line=dict(color='red', width=1, dash="dash"),
                row=3, col=1
            )
            
            fig.add_shape(
                type="line",
                x0=df.index[0],
                y0=30,
                x1=df.index[-1],
                y1=30,
                line=dict(color='green', width=1, dash="dash"),
                row=3, col=1
            )
        
        # Add MACD
        if all(col in df.columns for col in ['macd_line', 'macd_signal', 'macd_histogram']):
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['macd_line'],
                    name="MACD Line",
                    line=dict(color='blue', width=1)
                ),
                row=4, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['macd_signal'],
                    name="Signal Line",
                    line=dict(color='red', width=1)
                ),
                row=4, col=1
            )
            
            # Add MACD histogram
            colors = ['green' if val >= 0 else 'red' for val in df['macd_histogram']]
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['macd_histogram'],
                    name="MACD Histogram",
                    marker_color=colors,
                    opacity=0.8
                ),
                row=4, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=f"Advanced Trading Chart ({timeframe})",
            xaxis_rangeslider_visible=False,
            plot_bgcolor=self.color_scheme['background'],
            paper_bgcolor=self.color_scheme['background'],
            font=dict(color=self.color_scheme['text']),
            height=900,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=10, r=10, t=50, b=10)
        )
        
        # Update y-axis for each subplot
        fig.update_yaxes(
            gridcolor=self.color_scheme['grid'],
            zerolinecolor=self.color_scheme['grid'],
            row=1, col=1
        )
        
        fig.update_yaxes(
            gridcolor=self.color_scheme['grid'],
            zerolinecolor=self.color_scheme['grid'],
            row=2, col=1
        )
        
        fig.update_yaxes(
            gridcolor=self.color_scheme['grid'],
            zerolinecolor=self.color_scheme['grid'],
            range=[0, 100],
            row=3, col=1
        )
        
        fig.update_yaxes(
            gridcolor=self.color_scheme['grid'],
            zerolinecolor=self.color_scheme['grid'],
            row=4, col=1
        )
        
        # Update x-axis
        fig.update_xaxes(
            gridcolor=self.color_scheme['grid'],
            zerolinecolor=self.color_scheme['grid'],
            rangeslider_visible=False,
            row=4, col=1
        )
        
        return fig
    
    def _identify_support_resistance(self, df, window=10, threshold=0.02):
        """
        Identify support and resistance levels using price pivots.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            window (int): Window size for pivot detection
            threshold (float): Threshold for level clustering
            
        Returns:
            tuple: (support_levels, resistance_levels)
        """
        # Find pivot highs and lows
        pivot_highs = []
        pivot_lows = []
        
        for i in range(window, len(df) - window):
            # Check if this is a pivot high
            if all(df['high'][i] > df['high'][i-j] for j in range(1, window+1)) and \
               all(df['high'][i] > df['high'][i+j] for j in range(1, window+1)):
                pivot_highs.append(df['high'][i])
            
            # Check if this is a pivot low
            if all(df['low'][i] < df['low'][i-j] for j in range(1, window+1)) and \
               all(df['low'][i] < df['low'][i+j] for j in range(1, window+1)):
                pivot_lows.append(df['low'][i])
        
        # Cluster levels that are close to each other
        support_levels = self._cluster_levels(pivot_lows, threshold)
        resistance_levels = self._cluster_levels(pivot_highs, threshold)
        
        # Filter levels based on recent price
        current_price = df['close'].iloc[-1]
        support_levels = [level for level in support_levels if level < current_price]
        resistance_levels = [level for level in resistance_levels if level > current_price]
        
        # Limit to top 3 levels in each direction
        support_levels = sorted(support_levels, reverse=True)[:3]
        resistance_levels = sorted(resistance_levels)[:3]
        
        return support_levels, resistance_levels
    
    def _cluster_levels(self, levels, threshold):
        """
        Cluster price levels that are close to each other.
        
        Args:
            levels (list): List of price levels
            threshold (float): Threshold for clustering
            
        Returns:
            list: Clustered price levels
        """
        if not levels:
            return []
        
        # Sort levels
        sorted_levels = sorted(levels)
        
        # Initialize clusters
        clusters = [[sorted_levels[0]]]
        
        # Cluster levels
        for level in sorted_levels[1:]:
            last_cluster = clusters[-1]
            last_level = last_cluster[-1]
            
            # If this level is close to the last one, add to the same cluster
            if abs(level - last_level) / last_level < threshold:
                last_cluster.append(level)
            else:
                # Otherwise, start a new cluster
                clusters.append([level])
        
        # Calculate average level for each cluster
        clustered_levels = [sum(cluster) / len(cluster) for cluster in clusters]
        
        return clustered_levels
    
    def _add_chart_patterns(self, fig, df):
        """
        Identify and add chart patterns to the visualization.
        
        Args:
            fig (go.Figure): Plotly figure
            df (pd.DataFrame): DataFrame with OHLCV data
        """
        # Identify patterns
        patterns = self._identify_patterns(df)
        
        # Add patterns to chart
        for pattern in patterns:
            pattern_type = pattern['type']
            start_idx = pattern['start_idx']
            end_idx = pattern['end_idx']
            
            if pattern_type in ['double_top', 'head_and_shoulders', 'rising_wedge']:
                color = self.color_scheme['sell_signal']
                pattern_name = {
                    'double_top': 'Double Top',
                    'head_and_shoulders': 'Head & Shoulders',
                    'rising_wedge': 'Rising Wedge'
                }[pattern_type]
            else:
                color = self.color_scheme['buy_signal']
                pattern_name = {
                    'double_bottom': 'Double Bottom',
                    'inverse_head_and_shoulders': 'Inv. Head & Shoulders',
                    'falling_wedge': 'Falling Wedge'
                }[pattern_type]
            
            # Add pattern annotation
            fig.add_annotation(
                x=df.index[end_idx],
                y=df['high'][end_idx],
                text=pattern_name,
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=color,
                font=dict(size=10, color=color),
                align="center",
                ax=0,
                ay=-40,
                row=1, col=1
            )
            
            # Add pattern shape
            if pattern_type in ['double_top', 'double_bottom']:
                # Connect the two tops/bottoms
                y_values = df['high'] if pattern_type == 'double_top' else df['low']
                fig.add_shape(
                    type="line",
                    x0=df.index[start_idx],
                    y0=y_values[start_idx],
                    x1=df.index[end_idx],
                    y1=y_values[end_idx],
                    line=dict(color=color, width=1, dash="dot"),
                    row=1, col=1
                )
    
    def _identify_patterns(self, df, window=20):
        """
        Identify chart patterns in the price data.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            window (int): Window size for pattern detection
            
        Returns:
            list: List of identified patterns
        """
        patterns = []
        
        # Only look at the last portion of the data for pattern detection
        if len(df) > window * 3:
            df_subset = df.iloc[-window*3:]
        else:
            df_subset = df
        
        # Double Top detection
        patterns.extend(self._detect_double_top(df_subset))
        
        # Double Bottom detection
        patterns.extend(self._detect_double_bottom(df_subset))
        
        # Head and Shoulders detection (simplified)
        patterns.extend(self._detect_head_and_shoulders(df_subset))
        
        # Inverse Head and Shoulders detection (simplified)
        patterns.extend(self._detect_inverse_head_and_shoulders(df_subset))
        
        # Wedge patterns
        patterns.extend(self._detect_wedges(df_subset))
        
        return patterns
    
    def _detect_double_top(self, df, threshold=0.03):
        """
        Detect Double Top pattern.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            threshold (float): Threshold for price similarity
            
        Returns:
            list: List of Double Top patterns
        """
        patterns = []
        
        # Find local maxima
        maxima_idx = []
        for i in range(2, len(df) - 2):
            if df['high'][i] > df['high'][i-1] and df['high'][i] > df['high'][i-2] and \
               df['high'][i] > df['high'][i+1] and df['high'][i] > df['high'][i+2]:
                maxima_idx.append(i)
        
        # Check for Double Top
        for i in range(len(maxima_idx) - 1):
            idx1 = maxima_idx[i]
            idx2 = maxima_idx[i+1]
            
            # Check if the two tops are similar in price
            if abs(df['high'][idx1] - df['high'][idx2]) / df['high'][idx1] < threshold:
                # Check if there's a significant valley between the tops
                min_idx = df['low'][idx1:idx2].idxmin()
                valley = df['low'][min_idx]
                
                if (df['high'][idx1] - valley) / df['high'][idx1] > threshold * 2:
                    patterns.append({
                        'type': 'double_top',
                        'start_idx': idx1,
                        'end_idx': idx2
                    })
        
        return patterns
    
    def _detect_double_bottom(self, df, threshold=0.03):
        """
        Detect Double Bottom pattern.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            threshold (float): Threshold for price similarity
            
        Returns:
            list: List of Double Bottom patterns
        """
        patterns = []
        
        # Find local minima
        minima_idx = []
        for i in range(2, len(df) - 2):
            if df['low'][i] < df['low'][i-1] and df['low'][i] < df['low'][i-2] and \
               df['low'][i] < df['low'][i+1] and df['low'][i] < df['low'][i+2]:
                minima_idx.append(i)
        
        # Check for Double Bottom
        for i in range(len(minima_idx) - 1):
            idx1 = minima_idx[i]
            idx2 = minima_idx[i+1]
            
            # Check if the two bottoms are similar in price
            if abs(df['low'][idx1] - df['low'][idx2]) / df['low'][idx1] < threshold:
                # Check if there's a significant peak between the bottoms
                max_idx = df['high'][idx1:idx2].idxmax()
                peak = df['high'][max_idx]
                
                if (peak - df['low'][idx1]) / df['low'][idx1] > threshold * 2:
                    patterns.append({
                        'type': 'double_bottom',
                        'start_idx': idx1,
                        'end_idx': idx2
                    })
        
        return patterns
    
    def _detect_head_and_shoulders(self, df, threshold=0.03):
        """
        Detect Head and Shoulders pattern (simplified).
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            threshold (float): Threshold for pattern detection
            
        Returns:
            list: List of Head and Shoulders patterns
        """
        patterns = []
        
        # Find local maxima
        maxima_idx = []
        for i in range(2, len(df) - 2):
            if df['high'][i] > df['high'][i-1] and df['high'][i] > df['high'][i-2] and \
               df['high'][i] > df['high'][i+1] and df['high'][i] > df['high'][i+2]:
                maxima_idx.append(i)
        
        # Need at least 3 peaks for H&S
        if len(maxima_idx) < 3:
            return patterns
        
        # Check for Head and Shoulders
        for i in range(len(maxima_idx) - 2):
            left_shoulder_idx = maxima_idx[i]
            head_idx = maxima_idx[i+1]
            right_shoulder_idx = maxima_idx[i+2]
            
            # Head should be higher than shoulders
            if df['high'][head_idx] > df['high'][left_shoulder_idx] and \
               df['high'][head_idx] > df['high'][right_shoulder_idx]:
                
                # Shoulders should be at similar heights
                if abs(df['high'][left_shoulder_idx] - df['high'][right_shoulder_idx]) / df['high'][left_shoulder_idx] < threshold:
                    patterns.append({
                        'type': 'head_and_shoulders',
                        'start_idx': left_shoulder_idx,
                        'end_idx': right_shoulder_idx
                    })
        
        return patterns
    
    def _detect_inverse_head_and_shoulders(self, df, threshold=0.03):
        """
        Detect Inverse Head and Shoulders pattern (simplified).
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            threshold (float): Threshold for pattern detection
            
        Returns:
            list: List of Inverse Head and Shoulders patterns
        """
        patterns = []
        
        # Find local minima
        minima_idx = []
        for i in range(2, len(df) - 2):
            if df['low'][i] < df['low'][i-1] and df['low'][i] < df['low'][i-2] and \
               df['low'][i] < df['low'][i+1] and df['low'][i] < df['low'][i+2]:
                minima_idx.append(i)
        
        # Need at least 3 valleys for Inv H&S
        if len(minima_idx) < 3:
            return patterns
        
        # Check for Inverse Head and Shoulders
        for i in range(len(minima_idx) - 2):
            left_shoulder_idx = minima_idx[i]
            head_idx = minima_idx[i+1]
            right_shoulder_idx = minima_idx[i+2]
            
            # Head should be lower than shoulders
            if df['low'][head_idx] < df['low'][left_shoulder_idx] and \
               df['low'][head_idx] < df['low'][right_shoulder_idx]:
                
                # Shoulders should be at similar heights
                if abs(df['low'][left_shoulder_idx] - df['low'][right_shoulder_idx]) / df['low'][left_shoulder_idx] < threshold:
                    patterns.append({
                        'type': 'inverse_head_and_shoulders',
                        'start_idx': left_shoulder_idx,
                        'end_idx': right_shoulder_idx
                    })
        
        return patterns
    
    def _detect_wedges(self, df):
        """
        Detect Rising and Falling Wedge patterns.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            list: List of Wedge patterns
        """
        patterns = []
        
        # Need at least 10 candles for wedge detection
        if len(df) < 10:
            return patterns
        
        # Check for Rising Wedge (bearish)
        # Rising wedge has higher highs and higher lows, but lows rising faster than highs
        try:
            # Fit lines to highs and lows
            high_indices = list(range(len(df)))
            low_indices = list(range(len(df)))
            
            high_slope, high_intercept = np.polyfit(high_indices, df['high'].values, 1)
            low_slope, low_intercept = np.polyfit(low_indices, df['low'].values, 1)
            
            # Rising wedge: both slopes positive, low slope > high slope
            if high_slope > 0 and low_slope > 0 and low_slope > high_slope:
                # Calculate convergence point
                x_intersect = (high_intercept - low_intercept) / (low_slope - high_slope)
                
                # If convergence point is within reasonable range
                if x_intersect > len(df) and x_intersect < len(df) * 2:
                    patterns.append({
                        'type': 'rising_wedge',
                        'start_idx': 0,
                        'end_idx': len(df) - 1
                    })
            
            # Falling wedge: both slopes negative, high slope < low slope
            if high_slope < 0 and low_slope < 0 and high_slope < low_slope:
                # Calculate convergence point
                x_intersect = (high_intercept - low_intercept) / (low_slope - high_slope)
                
                # If convergence point is within reasonable range
                if x_intersect > len(df) and x_intersect < len(df) * 2:
                    patterns.append({
                        'type': 'falling_wedge',
                        'start_idx': 0,
                        'end_idx': len(df) - 1
                    })
        except:
            # Skip if polyfit fails
            pass
        
        return patterns
    
    def _add_trading_signals(self, fig, df, signals):
        """
        Add trading signals to the chart.
        
        Args:
            fig (go.Figure): Plotly figure
            df (pd.DataFrame): DataFrame with OHLCV data
            signals (list): List of trading signals
        """
        for signal in signals:
            if signal['type'] == 'BUY':
                marker_color = self.color_scheme['buy_signal']
                marker_symbol = 'triangle-up'
                y_position = df['low'].min() * 0.99  # Below the price
                text = f"BUY<br>Entry: ${signal['entry_low']:.6f} - ${signal['entry_high']:.6f}<br>Target: ${signal['target_low']:.6f} - ${signal['target_high']:.6f}<br>Stop: ${signal['stop_loss']:.6f}"
            else:  # SELL
                marker_color = self.color_scheme['sell_signal']
                marker_symbol = 'triangle-down'
                y_position = df['high'].max() * 1.01  # Above the price
                text = f"SELL<br>Entry: ${signal['entry_low']:.6f} - ${signal['entry_high']:.6f}<br>Target: ${signal['target_low']:.6f} - ${signal['target_high']:.6f}<br>Stop: ${signal['stop_loss']:.6f}"
            
            # Convert timestamp to datetime if it's not already
            if isinstance(signal['timestamp'], str):
                signal_time = datetime.datetime.strptime(signal['timestamp'], '%Y-%m-%d %H:%M:%S')
            else:
                signal_time = signal['timestamp']
            
            # Find the closest index in the dataframe
            closest_idx = None
            min_diff = float('inf')
            
            for i, idx in enumerate(df.index):
                if isinstance(idx, str):
                    idx_time = datetime.datetime.strptime(idx, '%Y-%m-%d %H:%M:%S')
                else:
                    idx_time = idx
                
                diff = abs((idx_time - signal_time).total_seconds())
                if diff < min_diff:
                    min_diff = diff
                    closest_idx = i
            
            if closest_idx is not None:
                # Add signal marker
                fig.add_trace(
                    go.Scatter(
                        x=[df.index[closest_idx]],
                        y=[y_position],
                        mode='markers+text',
                        marker=dict(
                            symbol=marker_symbol,
                            size=15,
                            color=marker_color,
                            line=dict(width=2, color='white')
                        ),
                        text=[signal['type']],
                        textposition='top center',
                        hoverinfo='text',
                        hovertext=text,
                        showlegend=False
                    ),
                    row=1, col=1
                )
                
                # Add vertical line at signal time
                fig.add_shape(
                    type="line",
                    x0=df.index[closest_idx],
                    y0=df['low'].min(),
                    x1=df.index[closest_idx],
                    y1=df['high'].max(),
                    line=dict(
                        color=marker_color,
                        width=1,
                        dash="dot",
                    ),
                    opacity=0.5,
                    row=1, col=1
                )
    
    def _add_price_predictions(self, fig, df, predictions):
        """
        Add price predictions to the chart.
        
        Args:
            fig (go.Figure): Plotly figure
            df (pd.DataFrame): DataFrame with OHLCV data
            predictions (list): List of price predictions
        """
        for prediction in predictions:
            # Convert timestamp to datetime if it's not already
            if isinstance(prediction['timestamp'], str):
                pred_time = datetime.datetime.strptime(prediction['timestamp'], '%Y-%m-%d %H:%M:%S')
            else:
                pred_time = prediction['timestamp']
            
            # Find the closest index in the dataframe
            closest_idx = None
            min_diff = float('inf')
            
            for i, idx in enumerate(df.index):
                if isinstance(idx, str):
                    idx_time = datetime.datetime.strptime(idx, '%Y-%m-%d %H:%M:%S')
                else:
                    idx_time = idx
                
                diff = abs((idx_time - pred_time).total_seconds())
                if diff < min_diff:
                    min_diff = diff
                    closest_idx = i
            
            if closest_idx is not None and closest_idx < len(df) - 1:
                # Add prediction marker
                next_idx = closest_idx + 1
                if next_idx < len(df.index):
                    # Create prediction box
                    fig.add_shape(
                        type="rect",
                        x0=df.index[closest_idx],
                        y0=prediction['next_low'],
                        x1=df.index[next_idx] if next_idx < len(df.index) else df.index[-1],
                        y1=prediction['next_high'],
                        line=dict(
                            color=self.color_scheme['prediction'],
                            width=1,
                        ),
                        fillcolor=f"rgba(255, 215, 0, 0.2)",
                        row=1, col=1
                    )
                    
                    # Add prediction annotation
                    change_color = "green" if prediction['predicted_change_pct'] > 0 else "red"
                    fig.add_annotation(
                        x=df.index[next_idx] if next_idx < len(df.index) else df.index[-1],
                        y=prediction['next_close'],
                        text=f"{prediction['predicted_change_pct']:.2f}%",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor=change_color,
                        font=dict(size=10, color=change_color),
                        align="center",
                        ax=40,
                        ay=0,
                        row=1, col=1
                    )
    
    def create_market_heatmap(self, market_data, title="Crypto Market Heatmap"):
        """
        Create a market heatmap visualization.
        
        Args:
            market_data (pd.DataFrame): DataFrame with market data
            title (str): Chart title
            
        Returns:
            go.Figure: Plotly figure with market heatmap
        """
        # Ensure market_data has required columns
        required_cols = ['symbol', 'last_price', 'daily_change', 'volume_24h']
        if not all(col in market_data.columns for col in required_cols):
            raise ValueError(f"Market data must contain columns: {required_cols}")
        
        # Sort by volume
        market_data = market_data.sort_values('volume_24h', ascending=False)
        
        # Create figure
        fig = go.Figure()
        
        # Add treemap
        fig.add_trace(go.Treemap(
            labels=market_data['symbol'],
            parents=[""] * len(market_data),
            values=market_data['volume_24h'],
            textinfo="label+value+percent",
            hovertemplate='<b>%{label}</b><br>Price: $%{customdata[0]:.6f}<br>Change: %{customdata[1]:.2f}%<br>Volume: $%{value:,.0f}<extra></extra>',
            marker=dict(
                colors=market_data['daily_change'],
                colorscale=[[0, 'red'], [0.5, 'white'], [1, 'green']],
                cmid=0,
                colorbar=dict(
                    title="Daily Change %",
                    thickness=20,
                    len=0.5
                )
            ),
            customdata=np.column_stack((
                market_data['last_price'],
                market_data['daily_change']
            ))
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            plot_bgcolor=self.color_scheme['background'],
            paper_bgcolor=self.color_scheme['background'],
            font=dict(color=self.color_scheme['text']),
            margin=dict(l=10, r=10, t=50, b=10),
            height=600
        )
        
        return fig
    
    def create_correlation_matrix(self, price_data, title="Crypto Correlation Matrix"):
        """
        Create a correlation matrix visualization.
        
        Args:
            price_data (dict): Dictionary with symbol keys and price DataFrames
            title (str): Chart title
            
        Returns:
            go.Figure: Plotly figure with correlation matrix
        """
        # Extract close prices
        close_prices = {}
        for symbol, df in price_data.items():
            if 'close' in df.columns:
                close_prices[symbol] = df['close']
        
        # Create DataFrame with all close prices
        df_prices = pd.DataFrame(close_prices)
        
        # Calculate correlation matrix
        corr_matrix = df_prices.corr()
        
        # Create figure
        fig = go.Figure()
        
        # Add heatmap
        fig.add_trace(go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu_r',
            zmid=0,
            text=corr_matrix.round(2).values,
            texttemplate="%{text:.2f}",
            hovertemplate='<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.4f}<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            plot_bgcolor=self.color_scheme['background'],
            paper_bgcolor=self.color_scheme['background'],
            font=dict(color=self.color_scheme['text']),
            height=600,
            xaxis=dict(
                tickangle=-45,
                tickfont=dict(size=12)
            ),
            yaxis=dict(
                tickfont=dict(size=12)
            )
        )
        
        return fig
    
    def create_performance_dashboard(self, trading_history, metrics, title="Trading Performance Dashboard"):
        """
        Create a performance dashboard visualization.
        
        Args:
            trading_history (list): List of trade dictionaries
            metrics (dict): Dictionary with performance metrics
            title (str): Chart title
            
        Returns:
            go.Figure: Plotly figure with performance dashboard
        """
        # Convert trading history to DataFrame
        if trading_history:
            df_trades = pd.DataFrame(trading_history)
            
            # Convert timestamps to datetime
            for col in ['entry_time', 'exit_time']:
                if col in df_trades.columns:
                    df_trades[col] = pd.to_datetime(df_trades[col])
            
            # Filter closed trades
            closed_trades = df_trades[df_trades['status'] == 'closed'].copy()
        else:
            closed_trades = pd.DataFrame()
        
        # Create figure with subplots
        fig = make_subplots(
            rows=2, 
            cols=2,
            subplot_titles=(
                "Cumulative Profit/Loss", 
                "Win/Loss Distribution",
                "Profit/Loss by Symbol",
                "Trade Duration Distribution"
            ),
            specs=[
                [{"type": "scatter"}, {"type": "pie"}],
                [{"type": "bar"}, {"type": "histogram"}]
            ]
        )
        
        if not closed_trades.empty:
            # Cumulative P/L chart
            closed_trades = closed_trades.sort_values('exit_time')
            closed_trades['cumulative_pl'] = closed_trades['profit_loss'].cumsum()
            
            fig.add_trace(
                go.Scatter(
                    x=closed_trades['exit_time'],
                    y=closed_trades['cumulative_pl'],
                    mode='lines+markers',
                    name="Cumulative P/L",
                    line=dict(color='white', width=2),
                    marker=dict(
                        size=8,
                        color=closed_trades['profit_loss'],
                        colorscale=[[0, 'red'], [0.5, 'yellow'], [1, 'green']],
                        cmid=0
                    )
                ),
                row=1, col=1
            )
            
            # Win/Loss pie chart
            win_count = len(closed_trades[closed_trades['profit_loss'] > 0])
            loss_count = len(closed_trades[closed_trades['profit_loss'] <= 0])
            
            fig.add_trace(
                go.Pie(
                    labels=['Wins', 'Losses'],
                    values=[win_count, loss_count],
                    marker=dict(colors=['green', 'red']),
                    textinfo='label+percent',
                    hole=0.4
                ),
                row=1, col=2
            )
            
            # P/L by symbol bar chart
            symbol_pl = closed_trades.groupby('symbol')['profit_loss'].sum().reset_index()
            symbol_pl = symbol_pl.sort_values('profit_loss', ascending=False)
            
            fig.add_trace(
                go.Bar(
                    x=symbol_pl['symbol'],
                    y=symbol_pl['profit_loss'],
                    marker=dict(
                        color=symbol_pl['profit_loss'],
                        colorscale=[[0, 'red'], [0.5, 'yellow'], [1, 'green']],
                        cmid=0
                    )
                ),
                row=2, col=1
            )
            
            # Trade duration histogram
            if 'entry_time' in closed_trades.columns and 'exit_time' in closed_trades.columns:
                closed_trades['duration'] = (closed_trades['exit_time'] - closed_trades['entry_time']).dt.total_seconds() / 3600  # hours
                
                fig.add_trace(
                    go.Histogram(
                        x=closed_trades['duration'],
                        marker=dict(color='lightblue'),
                        nbinsx=20
                    ),
                    row=2, col=2
                )
                
                fig.update_xaxes(title_text="Duration (hours)", row=2, col=2)
        else:
            # Add empty traces if no data
            fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name="No Data"), row=1, col=1)
            fig.add_trace(go.Pie(labels=['No Data'], values=[1]), row=1, col=2)
            fig.add_trace(go.Bar(x=[], y=[]), row=2, col=1)
            fig.add_trace(go.Histogram(x=[]), row=2, col=2)
        
        # Add metrics as annotations
        metrics_text = (
            f"Total Trades: {metrics.get('total_trades', 0)}<br>"
            f"Win Rate: {metrics.get('win_rate', 0):.2f}%<br>"
            f"Profit Factor: {metrics.get('profit_factor', 0):.2f}<br>"
            f"Total P/L: ${metrics.get('total_profit_loss', 0):.2f}"
        )
        
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.5, y=1.05,
            text=metrics_text,
            showarrow=False,
            font=dict(size=14, color="white"),
            align="center",
            bgcolor="rgba(50, 50, 50, 0.7)",
            bordercolor="white",
            borderwidth=1,
            borderpad=10
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            plot_bgcolor=self.color_scheme['background'],
            paper_bgcolor=self.color_scheme['background'],
            font=dict(color=self.color_scheme['text']),
            height=800,
            showlegend=False
        )
        
        # Update axes
        fig.update_xaxes(
            gridcolor=self.color_scheme['grid'],
            zerolinecolor=self.color_scheme['grid']
        )
        
        fig.update_yaxes(
            gridcolor=self.color_scheme['grid'],
            zerolinecolor=self.color_scheme['grid']
        )
        
        return fig
