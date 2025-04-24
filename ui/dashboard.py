import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

class SignalDashboard:
    """
    A simplified dashboard for displaying clear crypto trading signals to beginners.
    Focuses on providing straightforward buy/sell recommendations with minimal complexity.
    """
    
    def __init__(self, signal_generator):
        """Initialize the dashboard with a signal generator."""
        self.signal_generator = signal_generator
        self.supported_cryptos = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT']
        self.timeframes = ['1h', '4h', '1d']
        self.default_balance = 1000  # Default account balance in USD
        
        # Color scheme
        self.colors = {
            'buy': '#25A969',       # Green
            'strong_buy': '#1E8454', # Dark Green
            'sell': '#E74C3C',      # Red
            'strong_sell': '#B83227', # Dark Red
            'hold': '#F39C12',      # Yellow/Orange
            'background': '#F8F9FA',
            'text': '#2C3E50',
            'grid': '#E5E5E5'
        }
    
    def run(self):
        """Run the Streamlit dashboard."""
        st.set_page_config(
            page_title="Easy Crypto Signals",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Apply custom CSS
        self._apply_custom_css()
        
        # Render header
        self._render_header()
        
        # Sidebar for settings
        self._render_sidebar()
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Signal summary for selected cryptocurrencies
            self._render_signal_summary()
            
            # Detailed view for selected cryptocurrency
            self._render_detailed_view()
        
        with col2:
            # Risk management panel
            self._render_risk_management()
            
            # Educational tips
            self._render_educational_tips()
    
    def _apply_custom_css(self):
        """Apply custom CSS styling."""
        st.markdown("""
        <style>
        .main {
            background-color: #F8F9FA;
            color: #2C3E50;
        }
        .stButton button {
            width: 100%;
            height: 3em;
            font-size: 1.2em;
            font-weight: bold;
            border-radius: 10px;
        }
        .signal-card {
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .buy-signal {
            background-color: #D5F5E3;
            border-left: 10px solid #25A969;
        }
        .strong-buy-signal {
            background-color: #ABEBC6;
            border-left: 10px solid #1E8454;
        }
        .sell-signal {
            background-color: #FADBD8;
            border-left: 10px solid #E74C3C;
        }
        .strong-sell-signal {
            background-color: #F5B7B1;
            border-left: 10px solid #B83227;
        }
        .hold-signal {
            background-color: #FCF3CF;
            border-left: 10px solid #F39C12;
        }
        .risk-low {
            color: #25A969;
            font-weight: bold;
        }
        .risk-medium {
            color: #F39C12;
            font-weight: bold;
        }
        .risk-high {
            color: #E74C3C;
            font-weight: bold;
        }
        .big-number {
            font-size: 2.5em;
            font-weight: bold;
        }
        .medium-number {
            font-size: 1.8em;
            font-weight: bold;
        }
        .signal-strength {
            font-size: 1.2em;
            margin-top: 10px;
        }
        .tooltip {
            position: relative;
            display: inline-block;
            border-bottom: 1px dotted #2C3E50;
        }
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 200px;
            background-color: #555;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 5px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -100px;
            opacity: 0;
            transition: opacity 0.3s;
        }
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _render_header(self):
        """Render the dashboard header."""
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h1 style="color: #2C3E50;">Easy Crypto Signals</h1>
            <p style="font-size: 1.2em; color: #7F8C8D;">Simple Buy/Sell Signals for Beginners</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Current time and last update
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Current Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        with col2:
            st.markdown("**Market Status:** <span style='color: #25A969; font-weight: bold;'>Open</span>", unsafe_allow_html=True)
        
        st.markdown("---")
    
    def _render_sidebar(self):
        """Render the sidebar with settings."""
        st.sidebar.title("Settings")
        
        # Cryptocurrency selection
        st.sidebar.subheader("Select Cryptocurrencies")
        selected_cryptos = []
        for crypto in self.supported_cryptos:
            if st.sidebar.checkbox(crypto, value=True if crypto in ['BTC/USDT', 'ETH/USDT'] else False):
                selected_cryptos.append(crypto)
        
        # Account balance
        st.sidebar.subheader("Your Account")
        account_balance = st.sidebar.number_input(
            "Account Balance (USD)",
            min_value=10.0,
            max_value=1000000.0,
            value=self.default_balance,
            step=100.0
        )
        
        # Risk tolerance
        st.sidebar.subheader("Risk Tolerance")
        risk_tolerance = st.sidebar.select_slider(
            "Select your risk tolerance",
            options=["Very Conservative", "Conservative", "Moderate", "Aggressive", "Very Aggressive"],
            value="Moderate"
        )
        
        # Notification settings
        st.sidebar.subheader("Notifications")
        signal_alerts = st.sidebar.checkbox("Signal Change Alerts", value=True)
        price_alerts = st.sidebar.checkbox("Price Movement Alerts", value=True)
        
        # Theme selection
        st.sidebar.subheader("Theme")
        theme = st.sidebar.radio("Select Theme", ["Light", "Dark"], index=0)
        
        # About section
        st.sidebar.markdown("---")
        st.sidebar.info(
            "This bot is designed for beginners to provide clear trading signals. "
            "It simplifies complex technical analysis into straightforward buy/sell recommendations."
        )
        
        # Store settings in session state
        if 'settings' not in st.session_state:
            st.session_state.settings = {}
        
        st.session_state.settings.update({
            'selected_cryptos': selected_cryptos if selected_cryptos else ['BTC/USDT', 'ETH/USDT'],
            'account_balance': account_balance,
            'risk_tolerance': risk_tolerance,
            'signal_alerts': signal_alerts,
            'price_alerts': price_alerts,
            'theme': theme
        })
    
    def _render_signal_summary(self):
        """Render the signal summary panel for selected cryptocurrencies."""
        st.subheader("Signal Summary")
        
        # Get selected cryptocurrencies from session state
        selected_cryptos = st.session_state.settings.get('selected_cryptos', ['BTC/USDT', 'ETH/USDT'])
        
        if not selected_cryptos:
            st.warning("Please select at least one cryptocurrency in the sidebar.")
            return
        
        # Create columns for each selected cryptocurrency
        cols = st.columns(len(selected_cryptos))
        
        for i, crypto in enumerate(selected_cryptos):
            with cols[i]:
                # Generate sample data and signal for this cryptocurrency
                price_data = self.signal_generator.generate_sample_data(crypto, days=30)
                signal = self.signal_generator.generate_signals(price_data)
                
                # Determine signal class for styling
                signal_class = self._get_signal_class(signal['signal'])
                
                # Render signal card
                st.markdown(f"""
                <div class="signal-card {signal_class}">
                    <h3>{crypto.split('/')[0]}</h3>
                    <p class="big-number">${signal['last_price']:,.2f}</p>
                    <p>Signal: <strong>{self._format_signal(signal['signal'])}</strong></p>
                    <div class="signal-strength">
                        Strength: {self._render_strength_meter(signal['strength'])}
                    </div>
                    <p>{signal['explanation']}</p>
                    <p>Risk Level: <span class="risk-{signal['risk_level'].lower()}">{signal['risk_level']}</span></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Action button
                button_style = self._get_button_style(signal['signal'])
                button_text = self._get_button_text(signal['signal'])
                st.markdown(f"""
                <button style="{button_style}">
                    {button_text}
                </button>
                """, unsafe_allow_html=True)
    
    def _render_detailed_view(self):
        """Render detailed view for a selected cryptocurrency."""
        st.subheader("Detailed Analysis")
        
        # Get selected cryptocurrencies from session state
        selected_cryptos = st.session_state.settings.get('selected_cryptos', ['BTC/USDT', 'ETH/USDT'])
        
        if not selected_cryptos:
            st.warning("Please select at least one cryptocurrency in the sidebar.")
            return
        
        # Dropdown to select cryptocurrency for detailed view
        selected_crypto = st.selectbox("Select Cryptocurrency", selected_cryptos)
        
        # Generate sample data and signal for this cryptocurrency
        price_data = self.signal_generator.generate_sample_data(selected_crypto, days=30)
        signal = self.signal_generator.generate_signals(price_data)
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["Price Chart", "Signal History", "Entry/Exit Levels"])
        
        with tab1:
            # Render price chart
            self._render_price_chart(selected_crypto, price_data, signal)
        
        with tab2:
            # Render signal history
            self._render_signal_history(selected_crypto)
        
        with tab3:
            # Render entry/exit levels
            self._render_entry_exit_levels(signal)
    
    def _render_price_chart(self, crypto, price_data, signal):
        """Render a simplified price chart with key levels."""
        # Create a Plotly figure
        fig = go.Figure()
        
        # Add price candlesticks
        fig.add_trace(go.Candlestick(
            x=price_data.index,
            open=price_data['open'],
            high=price_data['high'],
            low=price_data['low'],
            close=price_data['close'],
            name="Price",
            increasing_line_color=self.colors['buy'],
            decreasing_line_color=self.colors['sell']
        ))
        
        # Add moving averages if available
        if 'ma_fast' in price_data.columns:
            fig.add_trace(go.Scatter(
                x=price_data.index,
                y=price_data['ma_fast'],
                mode='lines',
                name="Fast MA",
                line=dict(color='blue', width=1)
            ))
        
        if 'ma_slow' in price_data.columns:
            fig.add_trace(go.Scatter(
                x=price_data.index,
                y=price_data['ma_slow'],
                mode='lines',
                name="Slow MA",
                line=dict(color='orange', width=1)
            ))
        
        # Add entry, stop loss, and take profit levels if available
        last_date = price_data.index[-1]
        date_range = [last_date - timedelta(days=5), last_date + timedelta(days=5)]
        
        if signal['entry_price']:
            fig.add_trace(go.Scatter(
                x=date_range,
                y=[signal['entry_price'], signal['entry_price']],
                mode='lines',
                name="Entry",
                line=dict(color='green', width=2, dash='dash')
            ))
        
        if signal['stop_loss']:
            fig.add_trace(go.Scatter(
                x=date_range,
                y=[signal['stop_loss'], signal['stop_loss']],
                mode='lines',
                name="Stop Loss",
                line=dict(color='red', width=2, dash='dash')
            ))
        
        if signal['take_profit']:
            fig.add_trace(go.Scatter(
                x=date_range,
                y=[signal['take_profit'], signal['take_profit']],
                mode='lines',
                name="Take Profit",
                line=dict(color='purple', width=2, dash='dash')
            ))
        
        # Update layout
        fig.update_layout(
            title=f"{crypto} Price Chart",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            xaxis_rangeslider_visible=False,
            template="plotly_white",
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Show the chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanation for beginners
        st.info(
            "📈 **Chart Explanation:** Green candles show price increases, red candles show decreases. "
            "The dashed lines show suggested entry price (green), stop loss (red), and take profit (purple) levels."
        )
    
    def _render_signal_history(self, crypto):
        """Render signal history for the selected cryptocurrency."""
        # Generate sample signal history
        history = []
        np.random.seed(42)  # For reproducible results
        
        signals = ["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"]
        weights = [0.1, 0.25, 0.3, 0.25, 0.1]  # Probabilities for each signal
        
        for i in range(7):
            date = datetime.now() - timedelta(days=i)
            signal = np.random.choice(signals, p=weights)
            
            history.append({
                "date": date.strftime("%Y-%m-%d"),
                "signal": signal,
                "price": np.random.uniform(40000, 60000) if "BTC" in crypto else np.random.uniform(2000, 3000),
                "strength": np.random.randint(50, 100)
            })
        
        # Display signal history
        st.markdown("### Recent Signal History")
        st.markdown("See how signals have changed over the past week:")
        
        for entry in history:
            signal_class = self._get_signal_class(entry['signal'])
            
            st.markdown(f"""
            <div class="signal-card {signal_class}" style="padding: 10px; margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between;">
                    <div>
                        <strong>{entry['date']}</strong>: {self._format_signal(entry['signal'])}
                    </div>
                    <div>
                        ${entry['price']:,.2f}
                    </div>
                </div>
                <div class="signal-strength" style="margin-top: 5px;">
                    Strength: {self._render_strength_meter(entry['strength'])}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    def _render_entry_exit_levels(self, signal):
        """Render entry, stop loss, and take profit levels."""
        st.markdown("### Suggested Entry & Exit Levels")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style="text-align: center; padding: 20px; background-color: #D5F5E3; border-radius: 10px;">
                <h4>Entry Price</h4>
                <p class="medium-number">${:,.2f}</p>
            </div>
            """.format(signal['entry_price']), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 20px; background-color: #FADBD8; border-radius: 10px;">
                <h4>Stop Loss</h4>
                <p class="medium-number">${:,.2f}</p>
                <p>({:.2f}%)</p>
            </div>
            """.format(
                signal['stop_loss'],
                (signal['stop_loss'] - signal['entry_price']) / signal['entry_price'] * 100
            ), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="text-align: center; padding: 20px; background-color: #D6EAF8; border-radius: 10px;">
                <h4>Take Profit</h4>
                <p class="medium-number">${:,.2f}</p>
                <p>({:.2f}%)</p>
            </div>
            """.format(
                signal['take_profit'],
                (signal['take_profit'] - signal['entry_price']) / signal['entry_price'] * 100
            ), unsafe_allow_html=True)
        
        # Risk-reward ratio
        risk = abs(signal['entry_price'] - signal['stop_loss'])
        reward = abs(signal['take_profit'] - signal['entry_price'])
        risk_reward = reward / risk if risk > 0 else 0
        
        st.markdown(f"""
        <div style="margin-top: 20px; padding: 15px; background-color: #F8F9FA; border-radius: 10px; text-align: center;">
            <h4>Risk-Reward Ratio: {risk_reward:.2f}</h4>
            <p>A ratio above 2.0 is generally considered good for beginners.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Explanation for beginners
        st.info(
            "💡 **What do these levels mean?**\n\n"
            "- **Entry Price**: The suggested price to buy or sell at\n"
            "- **Stop Loss**: Set a sell order at this price to limit potential losses\n"
            "- **Take Profit**: Set a sell order at this price to secure profits\n"
            "- **Risk-Reward Ratio**: How much potential profit compared to potential loss"
        )
    
    def _render_risk_management(self):
        """Render the risk management panel."""
        st.markdown("### Risk Management")
        
        # Get account balance from session state
        account_balance = st.session_state.settings.get('account_balance', self.default_balance)
        
        # Get selected cryptocurrencies from session state
        selected_cryptos = st.session_state.settings.get('selected_cryptos', ['BTC/USDT', 'ETH/USDT'])
        
        if not selected_cryptos:
            st.warning("Please select at least one cryptocurrency in the sidebar.")
            return
        
        # Select cryptocurrency for position sizing
        selected_crypto = st.selectbox("Select Cryptocurrency for Position Sizing", selected_cryptos, key="risk_crypto")
        
        # Generate sample data and signal for this cryptocurrency
        price_data = self.signal_generator.generate_sample_data(selected_crypto, days=30)
        signal = self.signal_generator.generate_signals(price_data)
        
        # Calculate recommended position size
        risk_level = signal['risk_level']
        signal_strength = signal['strength']
        
        # Get position size recommendation
        position_size_pct = self.signal_generator.get_position_size_recommendation(
            account_balance, risk_level, signal_strength
        )
        position_size_usd = account_balance * (position_size_pct / 100)
        position_size_crypto = position_size_usd / signal['last_price']
        
        # Display position size recommendation
        st.markdown(f"""
        <div style="padding: 20px; background-color: #F8F9FA; border-radius: 10px; margin-bottom: 20px;">
            <h4>Recommended Position Size</h4>
            <p class="medium-number">${position_size_usd:.2f} ({position_size_pct:.1f}% of balance)</p>
            <p>≈ {position_size_crypto:.6f} {selected_crypto.split('/')[0]}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Risk level explanation
        risk_color = "risk-" + risk_level.lower()
        st.markdown(f"""
        <div style="padding: 15px; background-color: #F8F9FA; border-radius: 10px; margin-bottom: 20px;">
            <h4>Current Risk Level: <span class="{risk_color}">{risk_level}</span></h4>
            <p>{self._get_risk_explanation(risk_level)}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Potential profit/loss calculation
        if signal['signal'] in ['STRONG_BUY', 'BUY', 'STRONG_SELL', 'SELL']:
            potential_loss = position_size_usd * abs((signal['stop_loss'] - signal['entry_price']) / signal['entry_price'])
            potential_profit = position_size_usd * abs((signal['take_profit'] - signal['entry_price']) / signal['entry_price'])
            
            st.markdown(f"""
            <div style="display: flex; margin-bottom: 20px;">
                <div style="flex: 1; padding: 15px; background-color: #FADBD8; border-radius: 10px; margin-right: 10px; text-align: center;">
                    <h4>Potential Loss</h4>
                    <p class="medium-number">-${potential_loss:.2f}</p>
                </div>
                <div style="flex: 1; padding: 15px; background-color: #D5F5E3; border-radius: 10px; margin-left: 10px; text-align: center;">
                    <h4>Potential Profit</h4>
                    <p class="medium-number">+${potential_profit:.2f}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Risk management tips
        st.markdown("### Risk Management Tips")
        st.info(
            "🛡️ **Never risk more than you can afford to lose**\n\n"
            "🔄 **Diversify your investments across different cryptocurrencies**\n\n"
            "⏱️ **Be patient and don't make emotional decisions**"
        )
    
    def _render_educational_tips(self):
        """Render educational tips for beginners."""
        st.markdown("### Learning Corner")
        
        # Create tabs for different educational content
        tab1, tab2 = st.tabs(["Trading Basics", "Signal Explanation"])
        
        with tab1:
            st.markdown("""
            #### Key Terms for Beginners
            
            - **Buy Signal**: Indicates a good time to purchase a cryptocurrency
            - **Sell Signal**: Indicates a good time to sell a cryptocurrency
            - **Stop Loss**: A price level to sell at to limit potential losses
            - **Take Profit**: A price level to sell at to secure profits
            - **Risk-Reward Ratio**: The potential profit compared to potential loss
            - **Position Size**: How much money to invest in a particular trade
            
            #### Common Mistakes to Avoid
            
            - **FOMO (Fear Of Missing Out)**: Don't buy just because prices are rising rapidly
            - **Panic Selling**: Don't sell just because prices are falling rapidly
            - **Overtrading**: Don't make too many trades in a short period
            - **Ignoring Stop Losses**: Always use stop losses to protect your investment
            - **Investing Too Much**: Never invest more than you can afford to lose
            """)
        
        with tab2:
            st.markdown("""
            #### How Signals Are Generated
            
            This bot analyzes multiple factors to generate clear buy/sell signals:
            
            1. **Trend Analysis**: Identifies the overall price direction
            2. **Momentum Indicators**: Measures the strength of price movements
            3. **Volatility Assessment**: Evaluates market stability
            4. **Support/Resistance Levels**: Identifies key price points
            
            All these complex calculations are simplified into clear signals so you don't need to understand the technical details.
            
            #### Signal Strength Explained
            
            Signal strength indicates how confident the bot is in its recommendation:
            
            - **Strong signals (80-100%)**: Multiple indicators strongly agree
            - **Moderate signals (50-79%)**: Several indicators agree
            - **Weak signals (below 50%)**: Mixed or conflicting indicators
            """)
        
        # Daily tip (rotates randomly)
        tips = [
            "Always use stop losses to protect your investment from significant losses.",
            "Don't invest more than you can afford to lose, especially in volatile markets.",
            "Patience is key in crypto trading - don't make emotional decisions.",
            "Diversify your investments across different cryptocurrencies to reduce risk.",
            "The trend is your friend - trading with the trend is often safer for beginners."
        ]
        
        st.markdown(f"""
        <div style="padding: 15px; background-color: #D6EAF8; border-radius: 10px; margin-top: 20px;">
            <h4>💡 Tip of the Day</h4>
            <p>{np.random.choice(tips)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    def _format_signal(self, signal):
        """Format signal text for display."""
        if signal == "STRONG_BUY":
            return "Strong Buy"
        elif signal == "BUY":
            return "Buy"
        elif signal == "STRONG_SELL":
            return "Strong Sell"
        elif signal == "SELL":
            return "Sell"
        elif signal == "HOLD":
            return "Hold"
        else:
            return signal
    
    def _get_signal_class(self, signal):
        """Get CSS class for signal styling."""
        if signal == "STRONG_BUY":
            return "strong-buy-signal"
        elif signal == "BUY":
            return "buy-signal"
        elif signal == "STRONG_SELL":
            return "strong-sell-signal"
        elif signal == "SELL":
            return "sell-signal"
        elif signal == "HOLD":
            return "hold-signal"
        else:
            return ""
    
    def _render_strength_meter(self, strength):
        """Render a visual strength meter."""
        filled_stars = int(strength / 20)  # 0-5 stars based on 0-100 strength
        empty_stars = 5 - filled_stars
        
        meter = "★" * filled_stars + "☆" * empty_stars + f" ({strength}%)"
        return meter
    
    def _get_button_style(self, signal):
        """Get button style based on signal."""
        if signal == "STRONG_BUY":
            return "background-color: #1E8454; color: white; border: none; padding: 10px; border-radius: 5px; cursor: pointer; width: 100%;"
        elif signal == "BUY":
            return "background-color: #25A969; color: white; border: none; padding: 10px; border-radius: 5px; cursor: pointer; width: 100%;"
        elif signal == "STRONG_SELL":
            return "background-color: #B83227; color: white; border: none; padding: 10px; border-radius: 5px; cursor: pointer; width: 100%;"
        elif signal == "SELL":
            return "background-color: #E74C3C; color: white; border: none; padding: 10px; border-radius: 5px; cursor: pointer; width: 100%;"
        elif signal == "HOLD":
            return "background-color: #F39C12; color: white; border: none; padding: 10px; border-radius: 5px; cursor: pointer; width: 100%;"
        else:
            return "background-color: #7F8C8D; color: white; border: none; padding: 10px; border-radius: 5px; cursor: pointer; width: 100%;"
    
    def _get_button_text(self, signal):
        """Get button text based on signal."""
        if signal in ["STRONG_BUY", "BUY"]:
            return "BUY NOW"
        elif signal in ["STRONG_SELL", "SELL"]:
            return "SELL NOW"
        elif signal == "HOLD":
            return "HOLD"
        else:
            return "WAIT"
    
    def _get_risk_explanation(self, risk_level):
        """Get explanation text for risk level."""
        if risk_level == "Low":
            return "Market conditions are relatively stable. Good time for beginners to trade."
        elif risk_level == "Medium":
            return "Some market volatility present. Trade with caution and use proper position sizing."
        else:  # High risk
            return "High market volatility. Consider reducing position size or waiting for more stable conditions."
