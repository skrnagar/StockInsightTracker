import streamlit as st
import plotly.graph_objects as go
from utils import get_stock_data, format_large_number
from ai_utils import predict_stock_price, analyze_sentiment, fetch_stock_news
import pandas as pd
import asyncio
from datetime import datetime
from ta import trend, momentum, volatility  # Using ta library instead of talib

# Page configuration
st.set_page_config(
    page_title="Indian Stock Data Visualization",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
    <style>
    .stApp {
        background-color: #f8f9fa;
    }
    .main {
        padding: 2rem;
    }
    .stock-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #1f2937;
    }
    .metric-label {
        font-size: 14px;
        color: #6b7280;
    }
    .news-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        border-left: 4px solid;
    }
    .news-positive {
        border-left-color: #10B981;
    }
    .news-neutral {
        border-left-color: #6B7280;
    }
    .news-negative {
        border-left-color: #EF4444;
    }
    .prediction-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-top: 1rem;
    }
    .cached-data-notice {
        padding: 8px;
        background-color: #f0f2f6;
        border-radius: 4px;
        font-size: 14px;
        color: #666;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📈 Stock Analysis")
symbol = st.sidebar.text_input("Enter NSE Stock Symbol (e.g., RELIANCE, TCS)", value="")
days_to_predict = st.sidebar.slider("Prediction Days", 7, 60, 30)

if symbol:
    # Show loading spinner
    with st.spinner(f'Analyzing {symbol.upper()}...'):
        data = get_stock_data(symbol.upper())

    if data['success']:
        # Main content area
        st.title(f"📊 {data['company_name']}")

        # Show cached data notice if applicable
        if data.get('cached'):
            st.markdown("""
                <div class='cached-data-notice'>
                    ℹ️ Showing cached data due to temporary connection issues with live data source
                </div>
            """, unsafe_allow_html=True)

        # Key metrics in a grid
        metrics = data['metrics']
        col1, col2, col3, col4 = st.columns(4)

        # Current Price and Change
        with col1:
            price_change = metrics['Current Price'] - metrics['Previous Close']
            price_change_pct = (price_change / metrics['Previous Close']) * 100
            color = "green" if price_change >= 0 else "red"
            st.markdown(f"""
                <div class='metric-container'>
                    <div class='metric-value'>₹{metrics['Current Price']:.2f}</div>
                    <div class='metric-label'>Current Price</div>
                    <div style='color: {color};'>
                        {'+' if price_change >= 0 else ''}₹{price_change:.2f} ({price_change_pct:.2f}%)
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # Market Cap
        with col2:
            st.markdown(f"""
                <div class='metric-container'>
                    <div class='metric-value'>{format_large_number(metrics['Market Cap'])}</div>
                    <div class='metric-label'>Market Cap</div>
                </div>
            """, unsafe_allow_html=True)

        # PE Ratio
        with col3:
            st.markdown(f"""
                <div class='metric-container'>
                    <div class='metric-value'>{metrics['PE Ratio']:.2f}</div>
                    <div class='metric-label'>P/E Ratio</div>
                </div>
            """, unsafe_allow_html=True)

        # Volume
        with col4:
            st.markdown(f"""
                <div class='metric-container'>
                    <div class='metric-value'>{format_large_number(metrics['Volume'])}</div>
                    <div class='metric-label'>Volume</div>
                </div>
            """, unsafe_allow_html=True)

        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Price Chart", "📊 Technical Analysis", "🔮 Predictions", "📰 News & Sentiment"])

        with tab1:
            # Price chart
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=data['historical_data'].index,
                open=data['historical_data']['Open'],
                high=data['historical_data']['High'],
                low=data['historical_data']['Low'],
                close=data['historical_data']['Close'],
                name='Price'
            ))

            fig.update_layout(
                title="Historical Price Movement",
                xaxis_title="Date",
                yaxis_title="Price (INR)",
                template="plotly_white",
                height=500,
                xaxis_rangeslider_visible=False
            )

            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("Technical Analysis")

            #Calculate technical indicators
            def calculate_technical_indicators(df):
                # Using ta library instead of talib
                df['SMA_20'] = trend.sma_indicator(df['Close'], window=20)
                df['SMA_50'] = trend.sma_indicator(df['Close'], window=50)
                df['RSI'] = momentum.rsi(df['Close'], window=14)

                # MACD calculation
                df['MACD'] = trend.macd_diff(df['Close'], 
                                           window_slow=26, 
                                           window_fast=12, 
                                           window_sign=9)
                df['MACD_Signal'] = trend.macd_signal(df['Close'],
                                                    window_slow=26,
                                                    window_fast=12,
                                                    window_sign=9)
                df['MACD_Line'] = trend.macd(df['Close'],
                                           window_slow=26,
                                           window_fast=12,
                                           window_sign=9)

                # ADX calculation
                df['ADX'] = trend.adx(df['High'], df['Low'], df['Close'], window=14)

                # Bollinger Bands
                df['BB_Upper'] = volatility.bollinger_hband(df['Close'], window=20)
                df['BB_Middle'] = volatility.bollinger_mavg(df['Close'], window=20)
                df['BB_Lower'] = volatility.bollinger_lband(df['Close'], window=20)

                # Stochastic
                df['Stoch_K'] = momentum.stoch(df['High'], df['Low'], df['Close'],
                                             window=14, smooth_window=3)
                df['Stoch_D'] = momentum.stoch_signal(df['High'], df['Low'], df['Close'],
                                                     window=14, smooth_window=3)

                return df

            def get_indicator_signals(df):
              signals = {}
              signals['RSI'] = {'value': df['RSI'].iloc[-1], 'signal': 'Neutral'}
              if df['RSI'].iloc[-1] > 70: signals['RSI']['signal'] = 'Overbought'
              if df['RSI'].iloc[-1] < 30: signals['RSI']['signal'] = 'Oversold'

              signals['MACD'] = {'value': df['MACD'].iloc[-1], 'signal': 'Neutral'}
              if df['MACD'].iloc[-1] > 0: signals['MACD']['signal'] = 'Bullish'
              if df['MACD'].iloc[-1] < 0: signals['MACD']['signal'] = 'Bearish'

              signals['ADX'] = {'value': df['ADX'].iloc[-1], 'signal': 'Neutral'}
              if df['ADX'].iloc[-1] > 25: signals['ADX']['signal'] = 'Strong Trend'

              return signals


            tech_data = calculate_technical_indicators(data['historical_data'])
            signals = get_indicator_signals(tech_data)

            # Technical Overview Card
            st.markdown("""
                <div style='background-color: white; padding: 1.5rem; border-radius: 10px; 
                         box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 1rem;'>
                    <h4>Technical Overview</h4>
                    <p>Analysis based on multiple technical indicators</p>
                </div>
            """, unsafe_allow_html=True)

            # Display indicator signals in a grid
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"""
                    <div style='background-color: white; padding: 1rem; border-radius: 8px; text-align: center; margin-bottom: 1rem;'>
                        <p style='margin: 0; color: #666;'>RSI (14)</p>
                        <p style='font-size: 1.5rem; margin: 0;'>{signals['RSI']['value']:.2f}</p>
                        <p style='color: {
                            "#EF4444" if signals['RSI']['signal'] == 'Overbought'
                            else "#10B981" if signals['RSI']['signal'] == 'Oversold'
                            else "#6B7280"
                        };'>{signals['RSI']['signal']}</p>
                    </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                    <div style='background-color: white; padding: 1rem; border-radius: 8px; text-align: center; margin-bottom: 1rem;'>
                        <p style='margin: 0; color: #666;'>MACD</p>
                        <p style='font-size: 1.5rem; margin: 0;'>{signals['MACD']['value']:.4f}</p>
                        <p style='color: {
                            "#10B981" if signals['MACD']['signal'] == 'Bullish'
                            else "#EF4444" if signals['MACD']['signal'] == 'Bearish'
                            else "#6B7280"
                        };'>{signals['MACD']['signal']}</p>
                    </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                    <div style='background-color: white; padding: 1rem; border-radius: 8px; text-align: center; margin-bottom: 1rem;'>
                        <p style='margin: 0; color: #666;'>Trend Strength (ADX)</p>
                        <p style='font-size: 1.5rem; margin: 0;'>{signals['ADX']['value']:.2f}</p>
                        <p style='color: {
                            "#10B981" if signals['ADX']['signal'] == 'Strong Trend'
                            else "#6B7280"
                        };'>{signals['ADX']['signal']}</p>
                    </div>
                """, unsafe_allow_html=True)

            # Technical Charts
            selected_indicator = st.selectbox(
                "Select Technical Indicator",
                ["Bollinger Bands", "MACD", "RSI", "Stochastic", "Moving Averages"]
            )

            if selected_indicator == "Bollinger Bands":
                fig = go.Figure()

                # Price and Bollinger Bands
                fig.add_trace(go.Scatter(
                    x=tech_data.index, y=tech_data['Close'],
                    name='Price', line=dict(color='#1f77b4')
                ))
                fig.add_trace(go.Scatter(
                    x=tech_data.index, y=tech_data['BB_Upper'],
                    name='Upper Band', line=dict(color='gray', dash='dash')
                ))
                fig.add_trace(go.Scatter(
                    x=tech_data.index, y=tech_data['BB_Lower'],
                    name='Lower Band', line=dict(color='gray', dash='dash'),
                    fill='tonexty'
                ))

            elif selected_indicator == "MACD":
                fig = go.Figure()

                # MACD Line and Signal
                fig.add_trace(go.Scatter(
                    x=tech_data.index, y=tech_data['MACD_Line'],
                    name='MACD Line', line=dict(color='#1f77b4')
                ))
                fig.add_trace(go.Scatter(
                    x=tech_data.index, y=tech_data['MACD_Signal'],
                    name='Signal Line', line=dict(color='#ff7f0e')
                ))

                # MACD Histogram
                colors = ['#10B981' if val >= 0 else '#EF4444' for val in tech_data['MACD']]
                fig.add_trace(go.Bar(
                    x=tech_data.index, y=tech_data['MACD'],
                    name='MACD Histogram', marker_color=colors
                ))

            elif selected_indicator == "RSI":
                fig = go.Figure()

                # RSI Line
                fig.add_trace(go.Scatter(
                    x=tech_data.index, y=tech_data['RSI'],
                    name='RSI', line=dict(color='#1f77b4')
                ))

                # Overbought/Oversold Lines
                fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")

            elif selected_indicator == "Stochastic":
                fig = go.Figure()

                # Stochastic K and D lines
                fig.add_trace(go.Scatter(
                    x=tech_data.index, y=tech_data['Stoch_K'],
                    name='%K', line=dict(color='#1f77b4')
                ))
                fig.add_trace(go.Scatter(
                    x=tech_data.index, y=tech_data['Stoch_D'],
                    name='%D', line=dict(color='#ff7f0e')
                ))

                # Overbought/Oversold Lines
                fig.add_hline(y=80, line_dash="dash", line_color="red")
                fig.add_hline(y=20, line_dash="dash", line_color="green")

            else:  # Moving Averages
                fig = go.Figure()

                # Price and Moving Averages
                fig.add_trace(go.Scatter(
                    x=tech_data.index, y=tech_data['Close'],
                    name='Price', line=dict(color='#1f77b4')
                ))
                fig.add_trace(go.Scatter(
                    x=tech_data.index, y=tech_data['SMA_20'],
                    name='SMA 20', line=dict(color='#ff7f0e')
                ))
                fig.add_trace(go.Scatter(
                    x=tech_data.index, y=tech_data['SMA_50'],
                    name='SMA 50', line=dict(color='#2ca02c')
                ))

            # Update layout for all charts
            fig.update_layout(
                title=f"{selected_indicator} Analysis",
                xaxis_title="Date",
                yaxis_title="Value",
                template="plotly_white",
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

            # Indicator Interpretation
            st.markdown(f"""
                <div style='background-color: white; padding: 1rem; border-radius: 8px; margin-top: 1rem;'>
                    <h4>Indicator Interpretation</h4>
                    <p>Current Signal: <strong>{signals[selected_indicator.replace(' ', '_')]['signal']}</strong></p>
                </div>
            """, unsafe_allow_html=True)


        with tab3:
            # Price predictions
            st.subheader("Trading Analysis & Predictions")
            prediction = predict_stock_price(data['historical_data'], days_to_predict)

            if prediction['success']:
                metrics = prediction['metrics']

                # Last Week's Performance Card
                st.markdown("""
                    <div style='background-color: white; padding: 1.5rem; border-radius: 10px; 
                             box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 1rem;'>
                        <h4>Last Week's Prediction Performance</h4>
                    </div>
                """, unsafe_allow_html=True)

                perf_col1, perf_col2, perf_col3 = st.columns(3)
                with perf_col1:
                    st.metric("Total Predictions", 
                             metrics['last_week_accuracy']['predictions'])
                with perf_col2:
                    st.metric("Successful Predictions", 
                             metrics['last_week_accuracy']['hits'])
                with perf_col3:
                    st.metric("Accuracy", 
                             f"{metrics['last_week_accuracy']['accuracy_pct']:.1f}%")

                # Trading Signal Card
                signal_color = {
                    'BUY': '#10B981',
                    'SELL': '#EF4444',
                    'NEUTRAL': '#6B7280'
                }[metrics['signal']]

                st.markdown(f"""
                    <div style='background-color: white; padding: 1.5rem; border-radius: 10px; 
                             box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 1rem;
                             border-left: 4px solid {signal_color};'>
                        <h3 style='margin: 0; color: {signal_color};'>{metrics['signal']} Signal</h3>
                        <div style='display: flex; justify-content: space-between; margin-top: 1rem;'>
                            <div>
                                <p style='margin: 0; color: #666;'>Current Price</p>
                                <p style='font-size: 1.5rem; margin: 0;'>₹{metrics['current_price']:.2f}</p>
                            </div>
                            <div>
                                <p style='margin: 0; color: #666;'>Stop Loss</p>
                                <p style='font-size: 1.5rem; margin: 0;'>₹{metrics['stop_loss']:.2f}</p>
                                <p style='color: #EF4444;'>{metrics['stop_loss_pct']:.2f}%</p>
                            </div>
                        </div>
                        <div style='display: flex; justify-content: space-between; margin-top: 1rem;'>
                            <div>
                                <p style='margin: 0; color: #666;'>Target 1</p>
                                <p style='font-size: 1.5rem; margin: 0;'>₹{metrics['target_1']:.2f}</p>
                                <p style='color: #10B981;'>{metrics['target_1_pct']:.2f}%</p>
                            </div>
                            <div>
                                <p style='margin: 0; color: #666;'>Target 2</p>
                                <p style='font-size: 1.5rem; margin: 0;'>₹{metrics['target_2']:.2f}</p>
                                <p style='color: #10B981;'>{metrics['target_2_pct']:.2f}%</p>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                # Support and Resistance Levels
                st.markdown("""
                    <div style='background-color: white; padding: 1.5rem; border-radius: 10px; 
                             box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 1rem;'>
                        <h4>Intraday Levels</h4>
                    </div>
                """, unsafe_allow_html=True)

                levels = metrics['support_resistance']
                level_col1, level_col2 = st.columns(2)

                with level_col1:
                    st.markdown(f"""
                        <div style='background-color: white; padding: 1rem; border-radius: 8px; text-align: center;'>
                            <p style='margin: 0; color: #666;'>Resistance 2</p>
                            <p style='font-size: 1.5rem; margin: 0; color: #EF4444;'>
                                ₹{levels['resistance_2']:.2f}
                            </p>
                        </div>
                        <div style='background-color: white; padding: 1rem; border-radius: 8px; text-align: center; margin-top: 1rem;'>
                            <p style='margin: 0; color: #666;'>Resistance 1</p>
                            <p style='font-size: 1.5rem; margin: 0; color: #EF4444;'>
                                ₹{levels['resistance_1']:.2f}
                            </p>
                        </div>
                    """, unsafe_allow_html=True)

                with level_col2:
                    st.markdown(f"""
                        <div style='background-color: white; padding: 1rem; border-radius: 8px; text-align: center;'>
                            <p style='margin: 0; color: #666;'>Support 1</p>
                            <p style='font-size: 1.5rem; margin: 0; color: #10B981;'>
                                ₹{levels['support_1']:.2f}
                            </p>
                        </div>
                        <div style='background-color: white; padding: 1rem; border-radius: 8px; text-align: center; margin-top: 1rem;'>
                            <p style='margin: 0; color: #666;'>Support 2</p>
                            <p style='font-size: 1.5rem; margin: 0; color: #10B981;'>
                                ₹{levels['support_2']:.2f}
                            </p>
                        </div>
                    """, unsafe_allow_html=True)

                # Technical Indicators
                tech_levels = metrics['technical_levels']
                tech_col1, tech_col2 = st.columns(2)

                with tech_col1:
                    st.markdown(f"""
                        <div style='background-color: white; padding: 1rem; border-radius: 8px; text-align: center;'>
                            <p style='margin: 0; color: #666;'>RSI (14)</p>
                            <p style='font-size: 1.5rem; margin: 0;'>{tech_levels['RSI']:.2f}</p>
                            <p style='color: {
                                "#EF4444" if tech_levels['RSI'] > 70
                                else "#10B981" if tech_levels['RSI'] < 30
                                else "#6B7280"
                            };'>{
                                'Overbought' if tech_levels['RSI'] > 70
                                else 'Oversold' if tech_levels['RSI'] < 30
                                else 'Neutral'
                            }</p>
                        </div>
                        <div style='background-color: white; padding: 1rem; border-radius: 8px; text-align: center; margin-top: 1rem;'>
                            <p style='margin: 0; color: #666;'>MACD</p>
                            <p style='font-size: 1.5rem; margin: 0;'>{tech_levels['MACD']:.4f}</p>
                            <p style='color: {
                                "#10B981" if tech_levels['MACD'] > 0
                                else "#EF4444"
                            };'>{
                                'Bullish' if tech_levels['MACD'] > 0
                                else 'Bearish'
                            }</p>
                        </div>
                    """, unsafe_allow_html=True)

                with tech_col2:
                    st.markdown(f"""
                        <div style='background-color: white; padding: 1rem; border-radius: 8px; text-align: center;'>
                            <p style='margin: 0; color: #666;'>Money Flow Index</p>
                            <p style='font-size: 1.5rem; margin: 0;'>{tech_levels['MFI']:.2f}</p>
                            <p style='color: {
                                "#EF4444" if tech_levels['MFI'] > 80
                                else "#10B981" if tech_levels['MFI'] < 20
                                else "#6B7280"
                            };'>{
                                'Overbought' if tech_levels['MFI'] > 80
                                else 'Oversold' if tech_levels['MFI'] < 20
                                else 'Neutral'
                            }</p>
                        </div>
                        <div style='background-color: white; padding: 1rem; border-radius: 8px; text-align: center; margin-top: 1rem;'>
                            <p style='margin: 0; color: #666;'>Williams %R</p>
                            <p style='font-size: 1.5rem; margin: 0;'>{tech_levels['Williams_R']:.2f}</p>
                            <p style='color: {
                                "#EF4444" if tech_levels['Williams_R'] > -20
                                else "#10B981" if tech_levels['Williams_R'] < -80
                                else "#6B7280"
                            };'>{
                                'Overbought' if tech_levels['Williams_R'] > -20
                                else 'Oversold' if tech_levels['Williams_R'] < -80
                                else 'Neutral'
                            }</p>
                        </div>
                    """, unsafe_allow_html=True)

                # Hourly Prediction Chart
                hourly_pred = prediction['hourly_forecast']
                pred_fig = go.Figure()

                # Add price line
                pred_fig.add_trace(go.Scatter(
                    x=hourly_pred['Time'],
                    y=hourly_pred['Price'],
                    name='Predicted Price',
                    line=dict(color='#2ca02c')
                ))

                # Add confidence interval
                pred_fig.add_trace(go.Scatter(
                    x=hourly_pred['Time'].tolist() + hourly_pred['Time'].tolist()[::-1],
                    y=hourly_pred['Upper_Bound'].tolist() + hourly_pred['Lower_Bound'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(44,160,44,0.1)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Confidence Interval'
                ))

                pred_fig.update_layout(
                    title="Hourly Price Predictions",
                    xaxis_title="Time",
                    yaxis_title="Price (INR)",
                    template="plotly_white",
                    height=500
                )

                st.plotly_chart(pred_fig, use_container_width=True)

                # Trading Guidelines
                st.markdown("""
                    <div style='background-color: white; padding: 1.5rem; border-radius: 10px; 
                             box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-top: 1rem;'>
                        <h4>Trading Guidelines</h4>
                        <ul style='margin: 0; padding-left: 1.2rem;'>
                            <li>Set stop-loss orders at the indicated level to manage risk</li>
                            <li>Consider taking partial profits at Target 1</li>
                            <li>Move stop-loss to break-even after Target 1 is reached</li>
                            <li>Hold remaining position for Target 2</li>
                            <li>Monitor technical indicators for potential trend reversals</li>
                            <li>Use support and resistance levels for entry/exit points</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)

            else:
                st.error("Failed to generate predictions. Please try again later.")

        with tab4:
            # News and sentiment
            news_col1, news_col2 = st.columns([2, 1])

            with news_col1:
                st.subheader("Latest News")
                news_items = asyncio.run(fetch_stock_news(symbol))

                for item in news_items:
                    sentiment_color = {
                        'Positive': 'news-positive',
                        'Neutral': 'news-neutral',
                        'Negative': 'news-negative'
                    }[item['sentiment_label']]

                    st.markdown(f"""
                        <div class='news-card {sentiment_color}'>
                            <h4>{item['title']}</h4>
                            <p>{item['summary'][:200]}...</p>
                            <p><small>Published: {item['time_published']}</small></p>
                            <p><small>Sentiment: {item['sentiment_label']}</small></p>
                        </div>
                    """, unsafe_allow_html=True)

            with news_col2:
                st.subheader("Sentiment Analysis")
                sentiment_result = analyze_sentiment(' '.join([item['title'] + ' ' + item['summary'] for item in news_items]))

                if isinstance(sentiment_result, str):
                    sentiment_result = eval(sentiment_result)

                sentiment_color = '#10B981' if sentiment_result['sentiment'] > 0.2 else '#EF4444' if sentiment_result['sentiment'] < -0.2 else '#6B7280'

                st.markdown(f"""
                    <div style='background-color: white; padding: 1rem; border-radius: 8px; margin-top: 1rem;'>
                        <h4>Overall Sentiment</h4>
                        <div style='font-size: 24px; color: {sentiment_color}; margin: 1rem 0;'>
                            {sentiment_result['sentiment']:.2f}
                        </div>
                        <p>Confidence: {sentiment_result['confidence']:.2f}</p>
                        <p>{sentiment_result['summary']}</p>
                    </div>
                """, unsafe_allow_html=True)

        # Additional metrics table
        with st.expander("View All Metrics"):
            metrics_df = pd.DataFrame({
                'Metric': metrics.keys(),
                'Value': [
                    f"₹{metrics['Current Price']:.2f}",
                    f"₹{metrics['Previous Close']:.2f}",
                    format_large_number(metrics['Market Cap']),
                    f"{metrics['PE Ratio']:.2f}",
                    f"{metrics['Dividend Yield']*100:.2f}%",
                    f"₹{metrics['52 Week High']:.2f}",
                    f"₹{metrics['52 Week Low']:.2f}",
                    format_large_number(metrics['Volume'])
                ]
            })

            st.dataframe(metrics_df, use_container_width=True)

            # Download button for CSV
            csv = metrics_df.to_csv(index=False)
            st.download_button(
                label="Download Metrics as CSV",
                data=csv,
                file_name=f"{symbol}_metrics.csv",
                mime="text/csv"
            )

    else:
        st.error(f"Error fetching data: {data['error']}")
else:
    # Show welcome message
    st.markdown("""
        <div style='text-align: center; padding: 2rem;'>
            <h1>📈 Indian Stock Market Analytics</h1>
            <p>Enter a stock symbol from the National Stock Exchange (NSE) to get started.</p>
            <p>Examples: RELIANCE, TCS, INFY, HDFCBANK</p>
        </div>
    """, unsafe_allow_html=True)