import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules
from data.data_fetcher import CryptoDataFetcher
from signal_generator import CryptoSignalGenerator
from ui.dashboard import SignalDashboard
from ui.beginner_features import BeginnerFeatures
from utils.profit_optimization import ProfitOptimizationStrategies
from utils.loss_prevention import LossPreventionSafeguards

def main():
    """Main function to run the simplified crypto trading bot."""
    # Initialize components
    data_fetcher = CryptoDataFetcher()
    signal_generator = CryptoSignalGenerator(data_fetcher)
    profit_optimizer = ProfitOptimizationStrategies(signal_generator)
    loss_prevention = LossPreventionSafeguards(signal_generator)
    
    # Apply loss prevention safeguards to signal generator
    signal_generator.set_loss_prevention(loss_prevention)
    
    # Apply profit optimization to signal generator
    signal_generator.set_profit_optimizer(profit_optimizer)
    
    # Initialize dashboard
    dashboard = SignalDashboard(signal_generator)
    
    # Initialize beginner features
    beginner_features = BeginnerFeatures()
    
    # Set page config
    st.set_page_config(
        page_title="Easy Crypto Signals",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Create tabs for main dashboard and beginner features
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Signal Dashboard", 
        "🎓 Beginner's Guide", 
        "📈 Trading Simulator",
        "🛡️ Loss Prevention",
        "💰 Profit Maximizer"
    ])
    
    with tab1:
        # Run the main dashboard
        dashboard.run()
    
    with tab2:
        # Show beginner's guide
        beginner_features.render_guided_tutorial()
        beginner_features.render_visual_explainers()
        beginner_features.render_simplified_glossary()
    
    with tab3:
        # Show trading simulator
        beginner_features.render_interactive_demo()
    
    with tab4:
        # Show loss prevention features
        st.markdown("## 🛡️ Loss Prevention System")
        st.markdown("""
        This tab helps you understand how the bot protects you from significant losses.
        Even as a beginner, these safeguards will help keep your investments safe.
        """)
        
        # Market conditions assessment
        st.subheader("Current Market Conditions")
        
        # Get sample data for demonstration
        sample_data = signal_generator.generate_sample_data("BTC/USDT", days=60)
        
        # Check market conditions
        market_conditions = loss_prevention.check_market_conditions(sample_data)
        
        # Display market conditions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            risk_color = "green" if market_conditions['risk_level'] == "Low" else "orange" if market_conditions['risk_level'] == "Medium" else "red"
            st.markdown(f"""
            <div style="padding: 15px; background-color: #F8F9FA; border-radius: 10px; text-align: center;">
                <h4>Risk Level</h4>
                <p style="color: {risk_color}; font-size: 1.5em; font-weight: bold;">{market_conditions['risk_level']}</p>
                <p>{market_conditions['risk_explanation']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            trend_color = "green" if market_conditions['trend_alignment'] == "Strong" and "up" in market_conditions['trend_explanation'] else "red" if market_conditions['trend_alignment'] == "Strong" and "down" in market_conditions['trend_explanation'] else "orange"
            st.markdown(f"""
            <div style="padding: 15px; background-color: #F8F9FA; border-radius: 10px; text-align: center;">
                <h4>Trend Alignment</h4>
                <p style="color: {trend_color}; font-size: 1.5em; font-weight: bold;">{market_conditions['trend_alignment']}</p>
                <p>{market_conditions['trend_explanation']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            volume_color = "orange" if market_conditions['volume_alert'] in ["High", "Low"] else "red" if market_conditions['volume_alert'] in ["Very High", "Very Low"] else "green"
            st.markdown(f"""
            <div style="padding: 15px; background-color: #F8F9FA; border-radius: 10px; text-align: center;">
                <h4>Volume Alert</h4>
                <p style="color: {volume_color}; font-size: 1.5em; font-weight: bold;">{market_conditions['volume_alert']}</p>
                <p>{market_conditions['volume_explanation']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="padding: 15px; background-color: #F8F9FA; border-radius: 10px; margin-top: 20px;">
            <h4>Overall Assessment</h4>
            <p>{market_conditions['overall_assessment']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Safeguards explanation
        st.subheader("Automatic Safeguards")
        st.markdown("""
        The bot applies these safeguards automatically to protect your investment:
        """)
        
        safeguards = [
            {
                "name": "Auto Stop Loss",
                "description": "Automatically sets and adjusts stop losses to limit potential losses",
                "benefit": "Prevents catastrophic losses by exiting trades when they move against you"
            },
            {
                "name": "Maximum Daily Loss Limit",
                "description": "Stops trading when daily losses reach a preset percentage of your account",
                "benefit": "Prevents multiple losses in a single day from significantly impacting your account"
            },
            {
                "name": "Volatility Adjustment",
                "description": "Adjusts position sizes and stop losses based on market volatility",
                "benefit": "Reduces risk during highly volatile periods when price movements are unpredictable"
            },
            {
                "name": "Trend Confirmation",
                "description": "Confirms signals against multiple timeframe trends",
                "benefit": "Avoids trades that go against the overall market direction"
            },
            {
                "name": "Position Size Limits",
                "description": "Limits the amount invested in any single trade",
                "benefit": "Ensures no single trade can significantly impact your overall account balance"
            }
        ]
        
        for safeguard in safeguards:
            st.markdown(f"""
            <div style="padding: 15px; background-color: #F8F9FA; border-radius: 10px; margin-bottom: 15px;">
                <h4>{safeguard['name']}</h4>
                <p><strong>What it does:</strong> {safeguard['description']}</p>
                <p><strong>How it helps you:</strong> {safeguard['benefit']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Loss prevention tips
        st.subheader("Loss Prevention Tips")
        
        tips = loss_prevention.get_loss_prevention_tips()
        
        for i, tip in enumerate(tips):
            st.markdown(f"""
            <div style="padding: 10px; background-color: #F8F9FA; border-radius: 10px; margin-bottom: 10px;">
                <p><strong>Tip {i+1}:</strong> {tip}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Position sizing calculator
        st.subheader("Safe Position Size Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            account_balance = st.number_input(
                "Your Account Balance (USD)",
                min_value=10.0,
                max_value=1000000.0,
                value=1000.0,
                step=100.0
            )
        
        with col2:
            risk_tolerance = st.select_slider(
                "Your Risk Tolerance",
                options=["Very Low", "Low", "Medium", "High", "Very High"],
                value="Low"
            )
        
        # Generate a sample signal
        sample_signal = signal_generator.generate_signals(sample_data)
        
        # Calculate safe position size
        safe_position = loss_prevention.get_max_safe_position_size(sample_data, sample_signal, account_balance)
        
        # Adjust based on risk tolerance
        risk_multipliers = {
            "Very Low": 0.5,
            "Low": 0.75,
            "Medium": 1.0,
            "High": 1.25,
            "Very High": 1.5
        }
        
        adjusted_position = safe_position['position_size'] * risk_multipliers[risk_tolerance]
        adjusted_percentage = safe_position['percentage'] * risk_multipliers[risk_tolerance]
        
        st.markdown(f"""
        <div style="padding: 15px; background-color: #F8F9FA; border-radius: 10px; margin-top: 20px;">
            <h4>Recommended Safe Position Size</h4>
            <p style="font-size: 1.5em; font-weight: bold;">${adjusted_position:.2f} ({adjusted_percentage:.2f}% of your account)</p>
            <p>This is the maximum amount you should invest in a single trade to limit potential losses.</p>
            <p><strong>Explanation:</strong> {safe_position['explanation']}</p>
            <p><strong>Adjustment:</strong> Position size was {risk_multipliers[risk_tolerance] > 1.0 ? "increased" : "decreased"} based on your {risk_tolerance.toLowerCase()} risk tolerance.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with tab5:
        # Show profit maximizer features
        st.markdown("## 💰 Profit Maximizer")
        st.markdown("""
        This tab helps you understand how the bot maximizes your profits.
        Even as a beginner, these strategies will help you make more profitable trades.
        """)
        
        # Profit optimization strategies
        st.subheader("Profit Optimization Strategies")
        
        strategies = [
            {
                "name": "Optimal Entry Points",
                "description": "Identifies the best price levels to enter trades based on support levels",
                "benefit": "Improves your entry price to maximize potential profit"
            },
            {
                "name": "Multiple Take Profit Levels",
                "description": "Sets multiple price targets to take profits at different levels",
                "benefit": "Secures profits gradually as price moves in your favor"
            },
            {
                "name": "Trailing Take Profit",
                "description": "Automatically adjusts take profit levels as price moves favorably",
                "benefit": "Captures more profit during strong price movements"
            },
            {
                "name": "Dollar-Cost Averaging (DCA)",
                "description": "Suggests additional buy points if price temporarily moves against you",
                "benefit": "Improves your average entry price during temporary dips"
            },
            {
                "name": "Position Size Optimization",
                "description": "Calculates optimal position sizes based on signal strength and profit potential",
                "benefit": "Allocates more capital to higher-probability trades with better profit potential"
            }
        ]
        
        for strategy in strategies:
            st.markdown(f"""
            <div style="padding: 15px; background-color: #F8F9FA; border-radius: 10px; margin-bottom: 15px;">
                <h4>{strategy['name']}</h4>
                <p><strong>What it does:</strong> {strategy['description']}</p>
                <p><strong>How it helps you:</strong> {strategy['benefit']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Profit optimization example
        st.subheader("Profit Optimization Example")
        
        # Generate sample data and signal
        sample_data = signal_generator.generate_sample_data("BTC/USDT", days=30)
        sample_signal = signal_generator.generate_signals(sample_data)
        
        # Generate profit-optimized plan
        optimized_plan = profit_optimizer.generate_profit_optimized_plan(sample_data, sample_signal)
        
        if optimized_plan['action'] == 'BUY':
            # Create a visualization of the optimized plan
            entry_price = optimized_plan['primary_entry']
            take_profit = optimized_plan['take_profit']
            stop_loss = optimized_plan['stop_loss']
            
            # Create price range for visualization
            price_range = np.linspace(stop_loss * 0.95, take_profit * 1.05, 100)
            
            # Create figure
            fig = go.Figure()
            
            # Add price line (horizontal)
            fig.add_shape(
                type="line",
                x0=0, y0=entry_price,
                x1=1, y1=entry_price,
                line=dict(color="blue", width=2)
            )
            
            # Add entry point
            fig.add_trace(go.Scatter(
                x=[0.5],
                y=[entry_price],
                mode="markers+text",
                name="Entry",
                marker=dict(color="blue", size=15),
                text=["Entry"],
                textposition="top center"
            ))
            
            # Add stop loss
            fig.add_shape(
                type="line",
                x0=0, y0=stop_loss,
                x1=1, y1=stop_loss,
                line=dict(color="red", width=2, dash="dash")
            )
            
            fig.add_trace(go.Scatter(
                x=[0.5],
                y=[stop_loss],
                mode="markers+text",
                name="Stop Loss",
                marker=dict(color="red", size=15),
                text=["Stop Loss"],
                textposition="bottom center"
            ))
            
            # Add take profit
            fig.add_shape(
                type="line",
                x0=0, y0=take_profit,
                x1=1, y1=take_profit,
                line=dict(color="green", width=2, dash="dash")
            )
            
            fig.add_trace(go.Scatter(
                x=[0.5],
                y=[take_profit],
                mode="markers+text",
                name="Take Profit",
                marker=dict(color="green", size=15),
                text=["Take Profit"],
                textposition="top center"
            ))
            
            # Add profit taking levels if available
            if 'profit_taking_levels' in optimized_plan and optimized_plan['profit_taking_levels']:
                for i, level in enumerate(optimized_plan['profit_taking_levels']):
                    fig.add_shape(
                        type="line",
                        x0=0, y0=level['price'],
                        x1=1, y1=level['price'],
                        line=dict(color="green", width=1, dash="dot")
                    )
                    
                    fig.add_trace(go.Scatter(
                        x=[0.7],
                        y=[level['price']],
                        mode="markers+text",
                        name=f"Take Profit {i+1}",
                        marker=dict(color="green", size=10),
                        text=[f"TP {i+1} ({level['percentage']}%)"],
                        textposition="middle right"
                    ))
            
            # Add secondary entries if available
            if 'secondary_entries' in optimized_plan and optimized_plan['secondary_entries']:
                for i, entry in enumerate(optimized_plan['secondary_entries']):
                    fig.add_shape(
                        type="line",
                        x0=0, y0=entry['price'],
                        x1=1, y1=entry['price'],
                        line=dict(color="blue", width=1, dash="dot")
                    )
                    
                    fig.add_trace(go.Scatter(
                        x=[0.3],
                        y=[entry['price']],
                        mode="markers+text",
                        name=f"DCA {i+1}",
                        marker=dict(color="blue", size=10),
                        text=[f"DCA {i+1} (-{entry['percentage_drop']}%)"],
                        textposition="middle left"
                    ))
            
            # Update layout
            fig.update_layout(
                title="Profit-Optimized Trading Plan",
                xaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False
                ),
                yaxis=dict(
                    title="Price (USD)"
                ),
                height=500,
                showlegend=False
            )
            
            # Show figure
            st.plotly_chart(fig, use_container_width=True)
            
            # Show explanation
            st.markdown(f"""
            <div style="padding: 15px; background-color: #F8F9FA; border-radius: 10px; margin-top: 20px;">
                <h4>Trading Plan Explanation</h4>
                <p>{optimized_plan['explanation']}</p>
                <p><strong>Potential Profit:</strong> {optimized_plan['potential_profit_percentage']:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info(f"Current signal is {optimized_plan['action']}. Profit optimization is most relevant for BUY signals.")
        
        # Profit optimization tips
        st.subheader("Profit Maximization Tips")
        
        tips = profit_optimizer.get_profit_optimization_tips(sample_signal)
        
        for i, tip in enumerate(tips):
            st.markdown(f"""
            <div style="padding: 10px; background-color: #F8F9FA; border-radius: 10px; margin-bottom: 10px;">
                <p><strong>Tip {i+1}:</strong> {tip}</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
