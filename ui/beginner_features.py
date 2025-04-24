import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

class BeginnerFeatures:
    """
    Enhanced beginner-friendly features for the crypto trading bot.
    Focuses on simplifying complex concepts and providing educational guidance.
    """
    
    def __init__(self):
        """Initialize beginner features."""
        # Color scheme for consistent visual language
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
    
    def render_guided_tutorial(self):
        """Render a step-by-step guided tutorial for beginners."""
        st.markdown("## 🎓 Beginner's Guide to Crypto Trading")
        
        # Create expandable sections for each tutorial topic
        with st.expander("Step 1: Understanding Crypto Trading Basics", expanded=True):
            st.markdown("""
            ### What is Cryptocurrency Trading?
            
            Cryptocurrency trading is buying and selling digital currencies like Bitcoin (BTC) and Ethereum (ETH) 
            to make a profit from price changes.
            
            ### Key Differences from Stock Trading:
            
            - **24/7 Market**: Crypto markets never close, unlike stock markets
            - **Higher Volatility**: Prices can change dramatically in short periods
            - **Lower Entry Barrier**: You can start with small amounts of money
            - **Global Access**: Trade from anywhere with internet access
            
            ### Simple Trading Process:
            
            1. **Buy Low**: Purchase cryptocurrency when prices are low
            2. **Sell High**: Sell cryptocurrency when prices are higher
            3. **Repeat**: Continue this process to accumulate profits
            
            > **Remember**: This bot will tell you when to buy and sell with clear signals!
            """)
            
            # Add a simple illustration
            st.image("https://via.placeholder.com/800x300?text=Buy+Low,+Sell+High+Illustration", 
                     caption="The basic principle of trading: Buy Low, Sell High")
        
        with st.expander("Step 2: How to Read Trading Signals"):
            st.markdown("""
            ### Understanding the Signal Types
            
            This bot provides five simple signal types:
            
            1. **Strong Buy** (Dark Green): High confidence recommendation to purchase
            2. **Buy** (Green): General recommendation to purchase
            3. **Hold** (Yellow): Keep your current position, no action needed
            4. **Sell** (Red): General recommendation to sell
            5. **Strong Sell** (Dark Red): High confidence recommendation to sell
            
            ### Signal Strength
            
            Signal strength (shown as stars ★★★☆☆ or percentage) indicates how confident the bot is in its recommendation:
            
            - **5 stars (80-100%)**: Very high confidence
            - **4 stars (60-80%)**: High confidence
            - **3 stars (40-60%)**: Moderate confidence
            - **2 stars (20-40%)**: Low confidence
            - **1 star (0-20%)**: Very low confidence
            
            ### Timeframes
            
            Signals may work better for different timeframes:
            
            - **Short-term**: Hours to days
            - **Medium-term**: Days to weeks
            - **Long-term**: Weeks to months
            """)
            
            # Add visual examples of signals
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("""
                <div style="background-color: #ABEBC6; padding: 10px; border-radius: 5px; text-align: center;">
                    <h4>Strong Buy Example</h4>
                    <p>Signal: <strong>Strong Buy</strong></p>
                    <p>Strength: ★★★★★ (95%)</p>
                    <p>Action: Buy Now</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div style="background-color: #FCF3CF; padding: 10px; border-radius: 5px; text-align: center;">
                    <h4>Hold Example</h4>
                    <p>Signal: <strong>Hold</strong></p>
                    <p>Strength: ★★★☆☆ (55%)</p>
                    <p>Action: No Action Needed</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div style="background-color: #F5B7B1; padding: 10px; border-radius: 5px; text-align: center;">
                    <h4>Strong Sell Example</h4>
                    <p>Signal: <strong>Strong Sell</strong></p>
                    <p>Strength: ★★★★☆ (85%)</p>
                    <p>Action: Sell Now</p>
                </div>
                """, unsafe_allow_html=True)
        
        with st.expander("Step 3: Managing Risk"):
            st.markdown("""
            ### Understanding Risk in Crypto Trading
            
            Risk management is the most important skill for successful trading, especially for beginners.
            
            ### Key Risk Management Concepts:
            
            1. **Position Sizing**: How much money to invest in a single trade
               - This bot recommends position sizes based on your account balance and market conditions
               - Never invest more than the recommended amount
            
            2. **Stop Loss**: A price level where you automatically sell to limit losses
               - Always set a stop loss for every trade
               - This bot calculates appropriate stop loss levels for you
            
            3. **Take Profit**: A price level where you sell to secure profits
               - This bot calculates appropriate take profit levels for you
            
            4. **Risk-Reward Ratio**: The potential profit compared to potential loss
               - Aim for a ratio of at least 2:1 (potential profit should be twice the potential loss)
               - This bot calculates this ratio for you
            
            ### The 1% Rule (For Beginners)
            
            Never risk more than 1% of your total account balance on a single trade.
            
            Example: If you have $1,000, don't risk more than $10 on a single trade.
            """)
            
            # Add risk management visualization
            st.image("https://via.placeholder.com/800x300?text=Risk+Management+Visualization", 
                     caption="Visual representation of position sizing and stop loss")
        
        with st.expander("Step 4: Making Your First Trade"):
            st.markdown("""
            ### Step-by-Step Process for Your First Trade
            
            1. **Wait for a Strong Signal**: Look for a Strong Buy signal with high strength (4-5 stars)
            
            2. **Check Risk Level**: Ensure the risk level is Low or Medium for your first trades
            
            3. **Set Position Size**: Use the recommended position size from the Risk Management panel
            
            4. **Place Buy Order**: Use your exchange platform to place a buy order at the recommended entry price
            
            5. **Set Stop Loss**: Immediately set a stop loss order at the recommended stop loss price
            
            6. **Set Take Profit**: Set a take profit order at the recommended take profit price
            
            7. **Monitor (But Don't Obsess)**: Check the signal daily, but avoid watching prices constantly
            
            8. **Follow the Signal**: If the signal changes to Sell or Strong Sell, consider closing your position
            
            ### Example First Trade Walkthrough
            
            Let's walk through an example of a first Bitcoin trade using this bot's signals:
            """)
            
            # Create a visual walkthrough
            st.markdown("""
            <div style="background-color: #F8F9FA; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                <h4>Example: First Bitcoin Trade</h4>
                <ol>
                    <li><strong>Signal Received:</strong> Strong Buy (★★★★★ 90%)</li>
                    <li><strong>Risk Level:</strong> Low</li>
                    <li><strong>Account Balance:</strong> $1,000</li>
                    <li><strong>Recommended Position:</strong> $50 (5% of balance)</li>
                    <li><strong>Entry Price:</strong> $50,000 per BTC</li>
                    <li><strong>Stop Loss:</strong> $48,500 (-3%)</li>
                    <li><strong>Take Profit:</strong> $53,000 (+6%)</li>
                    <li><strong>Amount to Buy:</strong> 0.001 BTC ($50 ÷ $50,000)</li>
                    <li><strong>Maximum Potential Loss:</strong> $1.50 (3% of $50)</li>
                    <li><strong>Maximum Potential Profit:</strong> $3.00 (6% of $50)</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
        
        with st.expander("Step 5: Common Mistakes to Avoid"):
            st.markdown("""
            ### Top Mistakes Beginners Make in Crypto Trading
            
            1. **FOMO (Fear Of Missing Out)**
               - **Mistake**: Buying because prices are rising rapidly and you're afraid of missing profits
               - **Solution**: Only buy when the bot gives a Buy or Strong Buy signal
            
            2. **Panic Selling**
               - **Mistake**: Selling because prices are falling rapidly and you're afraid of losses
               - **Solution**: Use stop losses and only sell when the bot gives a Sell or Strong Sell signal
            
            3. **Overtrading**
               - **Mistake**: Making too many trades in a short period
               - **Solution**: Wait for strong signals and don't trade every small price movement
            
            4. **Ignoring Stop Losses**
               - **Mistake**: Not setting stop losses or removing them when prices approach them
               - **Solution**: Always set stop losses and never remove them
            
            5. **Investing Too Much**
               - **Mistake**: Investing more than you can afford to lose
               - **Solution**: Follow the position sizing recommendations from the bot
            
            6. **Emotional Trading**
               - **Mistake**: Making decisions based on emotions rather than signals
               - **Solution**: Stick to the bot's signals and avoid emotional decisions
            
            7. **Lack of Patience**
               - **Mistake**: Expecting immediate profits and giving up too soon
               - **Solution**: Be patient and give your trades time to develop
            """)
            
            # Add a visual reminder
            st.warning("Remember: The most successful traders are disciplined and patient. Follow the signals, manage your risk, and avoid these common mistakes.")
    
    def render_visual_explainers(self):
        """Render visual explanations of key trading concepts."""
        st.markdown("## 📊 Visual Trading Guides")
        
        # Create tabs for different visual explainers
        tab1, tab2, tab3 = st.tabs(["Candlestick Patterns", "Signal Types", "Risk Management"])
        
        with tab1:
            st.markdown("### Understanding Candlestick Patterns")
            st.markdown("""
            Candlestick patterns help traders understand price movements. Here are the basic patterns to know:
            """)
            
            # Create a simple candlestick explainer
            self._render_candlestick_explainer()
        
        with tab2:
            st.markdown("### How Trading Signals Work")
            st.markdown("""
            This bot analyzes multiple factors to generate clear signals. Here's a simplified view of how it works:
            """)
            
            # Create a signal generation explainer
            self._render_signal_generation_explainer()
        
        with tab3:
            st.markdown("### Visual Guide to Risk Management")
            st.markdown("""
            Proper risk management is crucial for successful trading. Here's how to manage risk effectively:
            """)
            
            # Create a risk management explainer
            self._render_risk_management_explainer()
    
    def render_simplified_glossary(self):
        """Render a simplified glossary of crypto trading terms."""
        st.markdown("## 📚 Simple Trading Dictionary")
        st.markdown("""
        This dictionary explains common trading terms in simple language for beginners.
        """)
        
        # Create a searchable glossary
        search_term = st.text_input("Search for a term:", "")
        
        # Define glossary terms
        glossary = {
            "Bitcoin (BTC)": "The first and most valuable cryptocurrency.",
            "Ethereum (ETH)": "The second-largest cryptocurrency, known for smart contracts.",
            "Altcoin": "Any cryptocurrency other than Bitcoin.",
            "Bull Market": "A market where prices are rising or expected to rise.",
            "Bear Market": "A market where prices are falling or expected to fall.",
            "Candlestick": "A chart type showing price open, high, low, and close.",
            "FOMO": "Fear Of Missing Out - buying because prices are rising rapidly.",
            "FUD": "Fear, Uncertainty, and Doubt - negative information causing selling.",
            "HODL": "Hold On for Dear Life - strategy of holding crypto long-term.",
            "Leverage": "Borrowing money to increase trading position size (risky for beginners).",
            "Liquidity": "How easily an asset can be bought or sold without affecting price.",
            "Market Cap": "Total value of a cryptocurrency (price × circulating supply).",
            "Order Book": "List of buy and sell orders for a cryptocurrency.",
            "Position": "The amount of a cryptocurrency you own.",
            "Resistance": "Price level where selling pressure may overcome buying pressure.",
            "Support": "Price level where buying pressure may overcome selling pressure.",
            "Slippage": "Difference between expected price and actual execution price.",
            "Stop Loss": "Order to sell when price reaches a specified level to limit losses.",
            "Take Profit": "Order to sell when price reaches a specified level to secure profits.",
            "Trading Volume": "Amount of a cryptocurrency traded in a given period.",
            "Volatility": "How rapidly price changes (higher volatility means higher risk).",
            "Whale": "Individual or entity holding large amounts of cryptocurrency."
        }
        
        # Filter glossary based on search term
        if search_term:
            filtered_glossary = {k: v for k, v in glossary.items() if search_term.lower() in k.lower()}
        else:
            filtered_glossary = glossary
        
        # Display glossary
        if filtered_glossary:
            for term, definition in filtered_glossary.items():
                st.markdown(f"""
                <div style="margin-bottom: 15px; padding: 10px; background-color: #F8F9FA; border-radius: 5px;">
                    <strong style="color: #2C3E50; font-size: 1.1em;">{term}</strong>
                    <p style="margin-top: 5px;">{definition}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No matching terms found. Try a different search term.")
    
    def render_interactive_demo(self):
        """Render an interactive demo for practicing with signals."""
        st.markdown("## 🎮 Practice Trading Simulator")
        st.markdown("""
        Practice making trading decisions based on signals without risking real money.
        This simulator helps you understand how to use the bot's signals effectively.
        """)
        
        # Initialize session state for the demo
        if 'demo_balance' not in st.session_state:
            st.session_state.demo_balance = 1000.0
        if 'demo_holdings' not in st.session_state:
            st.session_state.demo_holdings = 0.0
        if 'demo_history' not in st.session_state:
            st.session_state.demo_history = []
        if 'demo_day' not in st.session_state:
            st.session_state.demo_day = 1
        if 'demo_price_history' not in st.session_state:
            # Generate a price series for the demo
            np.random.seed(42)  # For reproducible results
            days = 30
            volatility = 0.03
            returns = np.random.normal(0, volatility, days)
            price_changes = 1 + returns
            start_price = 50000
            prices = start_price * np.cumprod(price_changes)
            
            st.session_state.demo_price_history = prices.tolist()
            st.session_state.demo_signals = self._generate_demo_signals(returns)
        
        # Display current status
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div style="text-align: center; padding: 10px; background-color: #F8F9FA; border-radius: 5px;">
                <h4>Day {st.session_state.demo_day}/30</h4>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="text-align: center; padding: 10px; background-color: #F8F9FA; border-radius: 5px;">
                <h4>Balance: ${st.session_state.demo_balance:.2f}</h4>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            current_price = st.session_state.demo_price_history[st.session_state.demo_day - 1]
            holdings_value = st.session_state.demo_holdings * current_price
            
            st.markdown(f"""
            <div style="text-align: center; padding: 10px; background-color: #F8F9FA; border-radius: 5px;">
                <h4>Holdings: {st.session_state.demo_holdings:.6f} BTC (${holdings_value:.2f})</h4>
            </div>
            """, unsafe_allow_html=True)
        
        # Display current price and signal
        current_price = st.session_state.demo_price_history[st.session_state.demo_day - 1]
        current_signal = st.session_state.demo_signals[st.session_state.demo_day - 1]
        
        signal_class = self._get_signal_class(current_signal)
        
        st.markdown(f"""
        <div class="signal-card {signal_class}" style="padding: 20px; border-radius: 10px; margin: 20px 0; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
            <h3>Bitcoin (BTC)</h3>
            <p style="font-size: 2em; font-weight: bold;">${current_price:.2f}</p>
            <p>Signal: <strong>{self._format_signal(current_signal)}</strong></p>
            <div style="margin-top: 10px;">
                Strength: {self._render_strength_meter(np.random.randint(60, 100))}
            </div>
            <p>{self._get_signal_explanation(current_signal)}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Trading actions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            buy_amount = st.number_input(
                "Amount to Buy (USD):",
                min_value=0.0,
                max_value=st.session_state.demo_balance,
                value=0.0,
                step=10.0
            )
            
            if st.button("Buy BTC", key="demo_buy"):
                if buy_amount > 0:
                    btc_amount = buy_amount / current_price
                    st.session_state.demo_balance -= buy_amount
                    st.session_state.demo_holdings += btc_amount
                    
                    # Record transaction
                    st.session_state.demo_history.append({
                        'day': st.session_state.demo_day,
                        'action': 'BUY',
                        'price': current_price,
                        'amount_usd': buy_amount,
                        'amount_btc': btc_amount,
                        'signal': current_signal
                    })
                    
                    st.success(f"Bought {btc_amount:.6f} BTC for ${buy_amount:.2f}")
        
        with col2:
            sell_percentage = st.slider(
                "Amount to Sell (% of holdings):",
                min_value=0,
                max_value=100,
                value=0,
                step=10
            )
            
            if st.button("Sell BTC", key="demo_sell"):
                if sell_percentage > 0 and st.session_state.demo_holdings > 0:
                    sell_ratio = sell_percentage / 100
                    btc_amount = st.session_state.demo_holdings * sell_ratio
                    usd_amount = btc_amount * current_price
                    
                    st.session_state.demo_balance += usd_amount
                    st.session_state.demo_holdings -= btc_amount
                    
                    # Record transaction
                    st.session_state.demo_history.append({
                        'day': st.session_state.demo_day,
                        'action': 'SELL',
                        'price': current_price,
                        'amount_usd': usd_amount,
                        'amount_btc': btc_amount,
                        'signal': current_signal
                    })
                    
                    st.success(f"Sold {btc_amount:.6f} BTC for ${usd_amount:.2f}")
        
        with col3:
            if st.button("Next Day", key="demo_next_day"):
                if st.session_state.demo_day < 30:
                    st.session_state.demo_day += 1
                    st.experimental_rerun()
                else:
                    st.warning("Simulation complete! Reset to start over.")
            
            if st.button("Reset Simulation", key="demo_reset"):
                st.session_state.demo_balance = 1000.0
                st.session_state.demo_holdings = 0.0
                st.session_state.demo_history = []
                st.session_state.demo_day = 1
                st.experimental_rerun()
        
        # Display price chart
        self._render_demo_price_chart()
        
        # Display transaction history
        if st.session_state.demo_history:
            st.markdown("### Your Trading History")
            
            history_df = pd.DataFrame(st.session_state.demo_history)
            
            st.markdown("""
            <style>
            .buy-row {
                background-color: rgba(37, 169, 105, 0.1);
            }
            .sell-row {
                background-color: rgba(231, 76, 60, 0.1);
            }
            </style>
            """, unsafe_allow_html=True)
            
            for i, transaction in enumerate(st.session_state.demo_history):
                row_class = "buy-row" if transaction['action'] == 'BUY' else "sell-row"
                
                st.markdown(f"""
                <div class="{row_class}" style="padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                    <strong>Day {transaction['day']}</strong>: {transaction['action']} {transaction['amount_btc']:.6f} BTC 
                    at ${transaction['price']:.2f} for ${transaction['amount_usd']:.2f}
                    (Signal: {self._format_signal(transaction['signal'])})
                </div>
                """, unsafe_allow_html=True)
            
            # Calculate performance
            initial_value = 1000.0
            final_value = st.session_state.demo_balance + (st.session_state.demo_holdings * current_price)
            profit_loss = final_value - initial_value
            profit_loss_pct = (profit_loss / initial_value) * 100
            
            profit_color = "green" if profit_loss >= 0 else "red"
            
            st.markdown(f"""
            <div style="padding: 15px; background-color: #F8F9FA; border-radius: 10px; margin-top: 20px;">
                <h4>Performance Summary</h4>
                <p>Starting Value: $1,000.00</p>
                <p>Current Value: ${final_value:.2f}</p>
                <p>Profit/Loss: <span style="color: {profit_color}; font-weight: bold;">${profit_loss:.2f} ({profit_loss_pct:.2f}%)</span></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Provide feedback on trading decisions
            correct_decisions = 0
            total_decisions = len(st.session_state.demo_history)
            
            for transaction in st.session_state.demo_history:
                if (transaction['action'] == 'BUY' and transaction['signal'] in ['STRONG_BUY', 'BUY']) or \
                   (transaction['action'] == 'SELL' and transaction['signal'] in ['STRONG_SELL', 'SELL']):
                    correct_decisions += 1
            
            if total_decisions > 0:
                decision_accuracy = (correct_decisions / total_decisions) * 100
                
                st.markdown(f"""
                <div style="padding: 15px; background-color: #F8F9FA; border-radius: 10px; margin-top: 20px;">
                    <h4>Decision Analysis</h4>
                    <p>Signal-Based Decisions: {correct_decisions}/{total_decisions} ({decision_accuracy:.1f}%)</p>
                    <p>{self._get_decision_feedback(decision_accuracy)}</p>
                </div>
                """, unsafe_allow_html=True)
    
    def render_market_context(self):
        """Render simplified market context for beginners."""
        st.markdown("## 🌍 Market Overview")
        st.markdown("""
        Understanding the broader market context helps make better trading decisions.
        Here's a simplified view of current market conditions.
        """)
        
        # Create tabs for different market views
        tab1, tab2 = st.tabs(["Market Summary", "Trend Analysis"])
        
        with tab1:
            st.markdown("### Current Market Summary")
            
            # Generate sample market data
            market_data = [
                {"coin": "Bitcoin (BTC)", "price": 50000, "change_24h": 2.5, "market_cap": 950, "volume": 30},
                {"coin": "Ethereum (ETH)", "price": 3000, "change_24h": -1.2, "market_cap": 350, "volume": 15},
                {"coin": "Binance Coin (BNB)", "price": 500, "change_24h": 0.8, "market_cap": 80, "volume": 5},
                {"coin": "Solana (SOL)", "price": 100, "change_24h": 5.2, "market_cap": 40, "volume": 3},
                {"coin": "Cardano (ADA)", "price": 1.2, "change_24h": -0.5, "market_cap": 38, "volume": 2},
                {"coin": "XRP", "price": 0.8, "change_24h": 1.1, "market_cap": 35, "volume": 2},
            ]
            
            # Create a DataFrame
            df = pd.DataFrame(market_data)
            
            # Display market data
            for i, row in df.iterrows():
                change_color = "green" if row['change_24h'] > 0 else "red"
                change_symbol = "▲" if row['change_24h'] > 0 else "▼"
                
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; padding: 10px; background-color: #F8F9FA; border-radius: 5px; margin-bottom: 10px;">
                    <div style="flex: 2;">
                        <strong>{row['coin']}</strong>
                    </div>
                    <div style="flex: 1; text-align: right;">
                        ${row['price']:,.2f}
                    </div>
                    <div style="flex: 1; text-align: right; color: {change_color};">
                        {change_symbol} {abs(row['change_24h']):,.1f}%
                    </div>
                    <div style="flex: 1; text-align: right;">
                        ${row['market_cap']}B
                    </div>
                    <div style="flex: 1; text-align: right;">
                        ${row['volume']}B
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Add column headers
            st.markdown("""
            <div style="display: flex; justify-content: space-between; padding: 5px; margin-top: -10px; font-size: 0.8em; color: #7F8C8D;">
                <div style="flex: 2;">Cryptocurrency</div>
                <div style="flex: 1; text-align: right;">Price</div>
                <div style="flex: 1; text-align: right;">24h Change</div>
                <div style="flex: 1; text-align: right;">Market Cap</div>
                <div style="flex: 1; text-align: right;">Volume (24h)</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Add market sentiment
            positive_coins = len(df[df['change_24h'] > 0])
            total_coins = len(df)
            sentiment_ratio = positive_coins / total_coins
            
            if sentiment_ratio > 0.7:
                sentiment = "Bullish (Positive)"
                sentiment_color = "green"
            elif sentiment_ratio > 0.3:
                sentiment = "Neutral"
                sentiment_color = "orange"
            else:
                sentiment = "Bearish (Negative)"
                sentiment_color = "red"
            
            st.markdown(f"""
            <div style="padding: 15px; background-color: #F8F9FA; border-radius: 10px; margin-top: 20px;">
                <h4>Market Sentiment</h4>
                <p>Current Sentiment: <span style="color: {sentiment_color}; font-weight: bold;">{sentiment}</span></p>
                <p>{positive_coins} out of {total_coins} major cryptocurrencies are showing positive price movement.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with tab2:
            st.markdown("### Simplified Trend Analysis")
            
            # Generate sample trend data
            trend_data = {
                "Bitcoin (BTC)": {
                    "short_term": "Bullish",
                    "medium_term": "Bullish",
                    "long_term": "Bullish",
                    "strength": 80
                },
                "Ethereum (ETH)": {
                    "short_term": "Bearish",
                    "medium_term": "Neutral",
                    "long_term": "Bullish",
                    "strength": 60
                },
                "Binance Coin (BNB)": {
                    "short_term": "Neutral",
                    "medium_term": "Bullish",
                    "long_term": "Bullish",
                    "strength": 70
                },
                "Solana (SOL)": {
                    "short_term": "Bullish",
                    "medium_term": "Bullish",
                    "long_term": "Neutral",
                    "strength": 75
                }
            }
            
            # Display trend analysis
            for coin, trends in trend_data.items():
                st.markdown(f"### {coin}")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    short_color = "green" if trends["short_term"] == "Bullish" else "red" if trends["short_term"] == "Bearish" else "orange"
                    st.markdown(f"""
                    <div style="text-align: center; padding: 10px; background-color: #F8F9FA; border-radius: 5px;">
                        <h5>Short Term</h5>
                        <p style="color: {short_color}; font-weight: bold;">{trends["short_term"]}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    medium_color = "green" if trends["medium_term"] == "Bullish" else "red" if trends["medium_term"] == "Bearish" else "orange"
                    st.markdown(f"""
                    <div style="text-align: center; padding: 10px; background-color: #F8F9FA; border-radius: 5px;">
                        <h5>Medium Term</h5>
                        <p style="color: {medium_color}; font-weight: bold;">{trends["medium_term"]}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    long_color = "green" if trends["long_term"] == "Bullish" else "red" if trends["long_term"] == "Bearish" else "orange"
                    st.markdown(f"""
                    <div style="text-align: center; padding: 10px; background-color: #F8F9FA; border-radius: 5px;">
                        <h5>Long Term</h5>
                        <p style="color: {long_color}; font-weight: bold;">{trends["long_term"]}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div style="text-align: center; padding: 10px; background-color: #F8F9FA; border-radius: 5px;">
                        <h5>Trend Strength</h5>
                        <p>{trends["strength"]}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Add a simple explanation
                bullish_count = sum(1 for v in [trends["short_term"], trends["medium_term"], trends["long_term"]] if v == "Bullish")
                bearish_count = sum(1 for v in [trends["short_term"], trends["medium_term"], trends["long_term"]] if v == "Bearish")
                
                if bullish_count > bearish_count:
                    explanation = f"{coin} is showing mostly bullish (positive) trends, with particular strength in "
                    if trends["short_term"] == "Bullish":
                        explanation += "short-term"
                    if trends["medium_term"] == "Bullish":
                        explanation += " and medium-term" if trends["short_term"] == "Bullish" else "medium-term"
                    if trends["long_term"] == "Bullish":
                        if trends["short_term"] == "Bullish" and trends["medium_term"] == "Bullish":
                            explanation += " and long-term"
                        elif trends["short_term"] == "Bullish" or trends["medium_term"] == "Bullish":
                            explanation += " and long-term"
                        else:
                            explanation += "long-term"
                    explanation += " timeframes."
                elif bearish_count > bullish_count:
                    explanation = f"{coin} is showing mostly bearish (negative) trends, with particular weakness in "
                    if trends["short_term"] == "Bearish":
                        explanation += "short-term"
                    if trends["medium_term"] == "Bearish":
                        explanation += " and medium-term" if trends["short_term"] == "Bearish" else "medium-term"
                    if trends["long_term"] == "Bearish":
                        if trends["short_term"] == "Bearish" and trends["medium_term"] == "Bearish":
                            explanation += " and long-term"
                        elif trends["short_term"] == "Bearish" or trends["medium_term"] == "Bearish":
                            explanation += " and long-term"
                        else:
                            explanation += "long-term"
                    explanation += " timeframes."
                else:
                    explanation = f"{coin} is showing mixed or neutral trends across different timeframes."
                
                st.markdown(f"""
                <div style="padding: 10px; background-color: #F8F9FA; border-radius: 5px; margin: 10px 0 20px 0;">
                    <p>{explanation}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Add a beginner-friendly explanation
            st.info("""
            **Understanding Timeframes:**
            
            - **Short Term**: Price movement over hours to days
            - **Medium Term**: Price movement over days to weeks
            - **Long Term**: Price movement over weeks to months
            
            **What This Means For You:**
            
            - If you're a beginner, focus on cryptocurrencies with positive trends across all timeframes
            - Avoid cryptocurrencies with negative short and medium-term trends
            - The trend strength shows how confident the analysis is (higher is better)
            """)
    
    def _render_candlestick_explainer(self):
        """Render a visual explanation of candlestick patterns."""
        # Create sample data for candlestick visualization
        dates = pd.date_range(start='2023-01-01', periods=5, freq='D')
        
        # Bullish candle
        bullish_data = {
            'open': [100],
            'high': [110],
            'low': [95],
            'close': [108]
        }
        bullish_df = pd.DataFrame(bullish_data)
        
        # Bearish candle
        bearish_data = {
            'open': [100],
            'high': [105],
            'low': [90],
            'close': [92]
        }
        bearish_df = pd.DataFrame(bearish_data)
        
        # Create candlestick visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Bullish (Green) Candle")
            st.markdown("Price closed higher than it opened")
            
            fig = go.Figure(data=[go.Candlestick(
                open=[bullish_df['open'][0]],
                high=[bullish_df['high'][0]],
                low=[bullish_df['low'][0]],
                close=[bullish_df['close'][0]],
                increasing_line_color=self.colors['buy'],
                decreasing_line_color=self.colors['sell']
            )])
            
            fig.update_layout(
                showlegend=False,
                xaxis_rangeslider_visible=False,
                height=300,
                width=200,
                margin=dict(l=0, r=0, t=0, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **What it means:**
            - Buyers were stronger than sellers
            - Price moved up during this period
            - Generally a positive sign
            """)
        
        with col2:
            st.markdown("#### Bearish (Red) Candle")
            st.markdown("Price closed lower than it opened")
            
            fig = go.Figure(data=[go.Candlestick(
                open=[bearish_df['open'][0]],
                high=[bearish_df['high'][0]],
                low=[bearish_df['low'][0]],
                close=[bearish_df['close'][0]],
                increasing_line_color=self.colors['buy'],
                decreasing_line_color=self.colors['sell']
            )])
            
            fig.update_layout(
                showlegend=False,
                xaxis_rangeslider_visible=False,
                height=300,
                width=200,
                margin=dict(l=0, r=0, t=0, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **What it means:**
            - Sellers were stronger than buyers
            - Price moved down during this period
            - Generally a negative sign
            """)
        
        # Add candlestick pattern examples
        st.markdown("### Common Candlestick Patterns")
        st.markdown("These patterns can help predict future price movements:")
        
        patterns = [
            {
                "name": "Doji",
                "description": "Opening and closing prices are very close, showing indecision in the market.",
                "interpretation": "Could signal a potential reversal of the current trend."
            },
            {
                "name": "Hammer",
                "description": "Small body at the top with a long lower wick, appearing during a downtrend.",
                "interpretation": "Potential bullish reversal signal - price tried to go lower but buyers pushed it back up."
            },
            {
                "name": "Engulfing Pattern",
                "description": "A candle that completely 'engulfs' the body of the previous candle.",
                "interpretation": "Strong signal of trend reversal, especially at the end of a trend."
            },
            {
                "name": "Morning Star",
                "description": "Three-candle pattern: large bearish candle, small candle, large bullish candle.",
                "interpretation": "Strong bullish reversal signal after a downtrend."
            }
        ]
        
        for pattern in patterns:
            st.markdown(f"""
            <div style="padding: 15px; background-color: #F8F9FA; border-radius: 10px; margin-bottom: 15px;">
                <h4>{pattern['name']}</h4>
                <p><strong>What it looks like:</strong> {pattern['description']}</p>
                <p><strong>What it means:</strong> {pattern['interpretation']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.info("""
        **Remember:** This bot analyzes these patterns (and many more indicators) automatically, 
        so you don't need to become an expert at identifying them. The bot will provide clear signals 
        based on these patterns and other factors.
        """)
    
    def _render_signal_generation_explainer(self):
        """Render a visual explanation of how trading signals are generated."""
        st.markdown("""
        The bot analyzes multiple factors to generate clear buy/sell signals. Here's a simplified view of how it works:
        """)
        
        # Create a flowchart-like visualization
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <div style="display: inline-block; padding: 15px; background-color: #F8F9FA; border-radius: 10px; margin-bottom: 10px; width: 80%;">
                <h4>Price Data Analysis</h4>
                <p>Historical prices, volume, and market data</p>
            </div>
            <div style="text-align: center; font-size: 24px;">↓</div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div style="padding: 15px; background-color: #D5F5E3; border-radius: 10px; height: 150px; text-align: center;">
                <h5>Trend Analysis</h5>
                <p>Identifies the overall price direction using moving averages</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="padding: 15px; background-color: #D6EAF8; border-radius: 10px; height: 150px; text-align: center;">
                <h5>Momentum Analysis</h5>
                <p>Measures the strength of price movements using RSI and MACD</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="padding: 15px; background-color: #FCF3CF; border-radius: 10px; height: 150px; text-align: center;">
                <h5>Volatility Analysis</h5>
                <p>Evaluates market stability using Bollinger Bands</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div style="padding: 15px; background-color: #FADBD8; border-radius: 10px; height: 150px; text-align: center;">
                <h5>Support/Resistance</h5>
                <p>Identifies key price levels where trends may change</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center;">
            <div style="font-size: 24px;">↓ ↓ ↓ ↓</div>
            <div style="display: inline-block; padding: 15px; background-color: #F8F9FA; border-radius: 10px; margin: 10px 0; width: 80%;">
                <h4>Signal Calculation</h4>
                <p>Combines all analyses with appropriate weightings</p>
            </div>
            <div style="font-size: 24px;">↓</div>
            <div style="display: inline-block; padding: 15px; background-color: #F8F9FA; border-radius: 10px; margin: 10px 0; width: 80%;">
                <h4>Clear Signal Generation</h4>
                <p>Simplifies complex analysis into straightforward recommendations</p>
            </div>
            <div style="font-size: 24px;">↓</div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown("""
            <div style="padding: 15px; background-color: #1E8454; color: white; border-radius: 10px; text-align: center;">
                <h5>Strong Buy</h5>
                <p>High confidence to purchase</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="padding: 15px; background-color: #25A969; color: white; border-radius: 10px; text-align: center;">
                <h5>Buy</h5>
                <p>General recommendation to purchase</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="padding: 15px; background-color: #F39C12; color: white; border-radius: 10px; text-align: center;">
                <h5>Hold</h5>
                <p>No action needed</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div style="padding: 15px; background-color: #E74C3C; color: white; border-radius: 10px; text-align: center;">
                <h5>Sell</h5>
                <p>General recommendation to sell</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            st.markdown("""
            <div style="padding: 15px; background-color: #B83227; color: white; border-radius: 10px; text-align: center;">
                <h5>Strong Sell</h5>
                <p>High confidence to sell</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.info("""
        **The Beauty of This System:**
        
        You don't need to understand all the complex technical indicators. The bot does all the hard work 
        and gives you clear signals that are easy to understand and act on. Just follow the signals!
        """)
    
    def _render_risk_management_explainer(self):
        """Render a visual explanation of risk management concepts."""
        st.markdown("""
        Proper risk management is the key to successful trading, especially for beginners.
        Here's how the bot helps you manage risk effectively:
        """)
        
        # Position sizing visualization
        st.markdown("### Position Sizing")
        st.markdown("How much money to invest in a single trade")
        
        account_balance = 1000  # Example account balance
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style="padding: 15px; background-color: #D5F5E3; border-radius: 10px; text-align: center;">
                <h5>Conservative (1%)</h5>
                <p class="medium-number">$10</p>
                <p>Very safe, minimal risk</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="padding: 15px; background-color: #FCF3CF; border-radius: 10px; text-align: center;">
                <h5>Moderate (2-3%)</h5>
                <p class="medium-number">$20-30</p>
                <p>Balanced approach</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="padding: 15px; background-color: #FADBD8; border-radius: 10px; text-align: center;">
                <h5>Aggressive (5%)</h5>
                <p class="medium-number">$50</p>
                <p>Higher risk, higher reward</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="padding: 15px; background-color: #F8F9FA; border-radius: 10px; margin: 15px 0;">
            <p>The bot recommends position sizes based on your account balance, market conditions, and signal strength.</p>
            <p><strong>Golden Rule:</strong> Never risk more than you can afford to lose on a single trade.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Stop loss and take profit visualization
        st.markdown("### Stop Loss & Take Profit")
        st.markdown("Automatic exit points to limit losses and secure profits")
        
        # Create a simple price chart with stop loss and take profit
        x = np.arange(0, 100)
        y = np.sin(x/10) * 10 + 100
        
        entry_point = 50
        entry_price = y[entry_point]
        stop_loss = entry_price - 5
        take_profit = entry_price + 10
        
        fig = go.Figure()
        
        # Add price line
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines',
            name='Price',
            line=dict(color='blue', width=2)
        ))
        
        # Add entry point
        fig.add_trace(go.Scatter(
            x=[entry_point],
            y=[entry_price],
            mode='markers',
            name='Entry Point',
            marker=dict(color='green', size=12, symbol='circle')
        ))
        
        # Add stop loss line
        fig.add_trace(go.Scatter(
            x=[0, 100],
            y=[stop_loss, stop_loss],
            mode='lines',
            name='Stop Loss',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        # Add take profit line
        fig.add_trace(go.Scatter(
            x=[0, 100],
            y=[take_profit, take_profit],
            mode='lines',
            name='Take Profit',
            line=dict(color='green', width=2, dash='dash')
        ))
        
        # Add annotations
        fig.add_annotation(
            x=entry_point,
            y=entry_price + 2,
            text="Entry",
            showarrow=True,
            arrowhead=1
        )
        
        fig.add_annotation(
            x=90,
            y=stop_loss,
            text="Stop Loss (-5%)",
            showarrow=True,
            arrowhead=1
        )
        
        fig.add_annotation(
            x=90,
            y=take_profit,
            text="Take Profit (+10%)",
            showarrow=True,
            arrowhead=1
        )
        
        fig.update_layout(
            title="Stop Loss & Take Profit Example",
            xaxis_title="Time",
            yaxis_title="Price",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="padding: 15px; background-color: #FADBD8; border-radius: 10px; text-align: center;">
                <h5>Stop Loss</h5>
                <p>Automatically sells when price falls to this level</p>
                <p><strong>Purpose:</strong> Limits your potential loss</p>
                <p><strong>Golden Rule:</strong> Never trade without a stop loss</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="padding: 15px; background-color: #D5F5E3; border-radius: 10px; text-align: center;">
                <h5>Take Profit</h5>
                <p>Automatically sells when price rises to this level</p>
                <p><strong>Purpose:</strong> Secures your profits</p>
                <p><strong>Golden Rule:</strong> Don't get greedy - take profits when available</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Risk-reward ratio visualization
        st.markdown("### Risk-Reward Ratio")
        st.markdown("The relationship between potential profit and potential loss")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style="padding: 15px; background-color: #FADBD8; border-radius: 10px; text-align: center;">
                <h5>Poor Ratio (1:1)</h5>
                <p>Risk $10 to make $10</p>
                <p>Not recommended</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="padding: 15px; background-color: #FCF3CF; border-radius: 10px; text-align: center;">
                <h5>Good Ratio (1:2)</h5>
                <p>Risk $10 to make $20</p>
                <p>Minimum recommended</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="padding: 15px; background-color: #D5F5E3; border-radius: 10px; text-align: center;">
                <h5>Excellent Ratio (1:3+)</h5>
                <p>Risk $10 to make $30+</p>
                <p>Ideal for beginners</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="padding: 15px; background-color: #F8F9FA; border-radius: 10px; margin: 15px 0;">
            <p><strong>Why This Matters:</strong> With a 2:1 ratio, you can be wrong half the time and still make money!</p>
            <p>Example: 10 trades with $10 risk each</p>
            <ul>
                <li>5 winning trades: 5 × $20 = $100 profit</li>
                <li>5 losing trades: 5 × $10 = $50 loss</li>
                <li>Net result: $50 profit</li>
            </ul>
            <p>The bot calculates appropriate risk-reward ratios for every signal.</p>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_demo_price_chart(self):
        """Render price chart for the demo."""
        # Get price history and current day
        prices = st.session_state.demo_price_history
        current_day = st.session_state.demo_day
        
        # Create a DataFrame for the chart
        df = pd.DataFrame({
            'day': range(1, len(prices) + 1),
            'price': prices
        })
        
        # Create the chart
        fig = go.Figure()
        
        # Add price line
        fig.add_trace(go.Scatter(
            x=df['day'],
            y=df['price'],
            mode='lines',
            name='BTC Price',
            line=dict(color='blue', width=2)
        ))
        
        # Add marker for current day
        fig.add_trace(go.Scatter(
            x=[current_day],
            y=[prices[current_day - 1]],
            mode='markers',
            name='Current Day',
            marker=dict(color='red', size=12, symbol='circle')
        ))
        
        # Add transaction markers if any
        if st.session_state.demo_history:
            buy_days = []
            buy_prices = []
            sell_days = []
            sell_prices = []
            
            for transaction in st.session_state.demo_history:
                if transaction['action'] == 'BUY':
                    buy_days.append(transaction['day'])
                    buy_prices.append(transaction['price'])
                else:  # SELL
                    sell_days.append(transaction['day'])
                    sell_prices.append(transaction['price'])
            
            if buy_days:
                fig.add_trace(go.Scatter(
                    x=buy_days,
                    y=buy_prices,
                    mode='markers',
                    name='Buy',
                    marker=dict(color='green', size=10, symbol='triangle-up')
                ))
            
            if sell_days:
                fig.add_trace(go.Scatter(
                    x=sell_days,
                    y=sell_prices,
                    mode='markers',
                    name='Sell',
                    marker=dict(color='red', size=10, symbol='triangle-down')
                ))
        
        # Update layout
        fig.update_layout(
            title="Bitcoin Price Simulation",
            xaxis_title="Day",
            yaxis_title="Price (USD)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=400
        )
        
        # Show only up to current day
        fig.update_xaxes(range=[1, current_day])
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _generate_demo_signals(self, returns):
        """Generate sample signals for the demo based on returns."""
        signals = []
        
        for i, ret in enumerate(returns):
            # Generate signal based on return and previous returns
            if i < 2:
                # For the first two days, just use the return
                if ret > 0.01:
                    signal = "STRONG_BUY"
                elif ret > 0.005:
                    signal = "BUY"
                elif ret < -0.01:
                    signal = "STRONG_SELL"
                elif ret < -0.005:
                    signal = "SELL"
                else:
                    signal = "HOLD"
            else:
                # For subsequent days, use the return and trend
                prev_returns = returns[max(0, i-3):i]
                avg_prev_return = sum(prev_returns) / len(prev_returns)
                
                if ret > 0.01 and avg_prev_return > 0:
                    signal = "STRONG_BUY"
                elif ret > 0.005 or (ret > 0 and avg_prev_return > 0.005):
                    signal = "BUY"
                elif ret < -0.01 and avg_prev_return < 0:
                    signal = "STRONG_SELL"
                elif ret < -0.005 or (ret < 0 and avg_prev_return < -0.005):
                    signal = "SELL"
                else:
                    signal = "HOLD"
            
            signals.append(signal)
        
        return signals
    
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
    
    def _get_signal_explanation(self, signal):
        """Get explanation text for a signal."""
        if signal == "STRONG_BUY":
            return "Multiple indicators show very positive conditions. Good time to buy."
        elif signal == "BUY":
            return "Overall positive outlook based on recent price action. Consider buying."
        elif signal == "STRONG_SELL":
            return "Multiple indicators show very negative conditions. Good time to sell."
        elif signal == "SELL":
            return "Overall negative outlook based on recent price action. Consider selling."
        elif signal == "HOLD":
            return "No clear direction at this time. Better to hold current position."
        else:
            return "Insufficient data to generate a reliable signal."
    
    def _get_decision_feedback(self, accuracy):
        """Get feedback text based on decision accuracy."""
        if accuracy >= 80:
            return "Excellent! You're following the signals very well. This disciplined approach is key to successful trading."
        elif accuracy >= 60:
            return "Good job! You're mostly following the signals. Try to be even more disciplined for better results."
        elif accuracy >= 40:
            return "You're sometimes following the signals, but could improve. Remember that following clear signals is usually better than guessing."
        else:
            return "You're often trading against the signals. This can be risky. Try to follow the signals more closely for better results."
