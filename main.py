import streamlit as st
import plotly.graph_objects as go
from utils import get_stock_data, format_large_number
from ai_utils import predict_stock_price, combine_predictions, fetch_stock_news, analyze_sentiment
import pandas as pd
import asyncio

# Page configuration with modern theme
st.set_page_config(
    page_title="Indian Stock Market Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS styling
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #1a1a1a, #2d2d2d);
        color: #ffffff;
        padding: 2rem;
    }
    .stApp {
        background: linear-gradient(135deg, #1a1a1a, #2d2d2d);
    }
    .css-1d391kg {
        background-color: #2d2d2d;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .metric-container {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 15px;
        margin: 5px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #ffffff;
    }
    .metric-label {
        font-size: 14px;
        color: #bbbbbb;
    }
    .prediction-card {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .prediction-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #ffffff;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.2);
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("📈 Stock Analysis")
    symbol = st.text_input("Enter NSE Stock Symbol (e.g., RELIANCE, TCS)", value="")
    st.markdown("""
        <div style='margin-top: 20px; padding: 10px; background: rgba(255, 255, 255, 0.1); border-radius: 5px;'>
            <h4>Popular Stocks</h4>
            <ul>
                <li>RELIANCE - Reliance Industries</li>
                <li>TCS - Tata Consultancy Services</li>
                <li>INFY - Infosys</li>
                <li>HDFCBANK - HDFC Bank</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

if symbol:
    with st.spinner(f'Analyzing {symbol.upper()}...'):
        data = get_stock_data(symbol.upper())

    if data['success']:
        # Display company name and basic metrics
        st.title(f"📊 {data['company_name']}")

        # Key metrics in modern cards
        metrics = data['metrics']
        cols = st.columns(4)

        with cols[0]:
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

        with cols[1]:
            st.markdown(f"""
                <div class='metric-container'>
                    <div class='metric-value'>{format_large_number(metrics['Market Cap'])}</div>
                    <div class='metric-label'>Market Cap</div>
                </div>
            """, unsafe_allow_html=True)

        with cols[2]:
            st.markdown(f"""
                <div class='metric-container'>
                    <div class='metric-value'>{metrics['PE Ratio']:.2f}</div>
                    <div class='metric-label'>P/E Ratio</div>
                </div>
            """, unsafe_allow_html=True)

        with cols[3]:
            st.markdown(f"""
                <div class='metric-container'>
                    <div class='metric-value'>{format_large_number(metrics['Volume'])}</div>
                    <div class='metric-label'>Volume</div>
                </div>
            """, unsafe_allow_html=True)

        # Create tabs with modern styling
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Price Chart",
            "🎯 Trading Signals",
            "🤖 AI Predictions",
            "📰 News & Sentiment"
        ])

        with tab1:
            # Enhanced price chart
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
                template="plotly_dark",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=600,
                xaxis_rangeslider_visible=False,
                title={
                    'text': "Price History",
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                }
            )

            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            # Trading signals with enhanced visualization
            prediction = predict_stock_price(data['historical_data'])

            if prediction['success']:
                metrics = prediction['metrics']
                signal_color = {
                    'BUY': '#00ff00',
                    'SELL': '#ff0000',
                    'NEUTRAL': '#888888'
                }[metrics['signal']]

                st.markdown(f"""
                    <div class='prediction-card'>
                        <div class='prediction-header' style='color: {signal_color};'>
                            {metrics['signal']} Signal (Confidence: {prediction['confidence']:.1f}%)
                        </div>
                        <table style='width: 100%;'>
                            <tr>
                                <td><strong>Stop Loss:</strong></td>
                                <td>₹{metrics['stop_loss']:.2f} ({metrics['stop_loss_pct']}%)</td>
                                <td><strong>Target 1:</strong></td>
                                <td>₹{metrics['target_1']:.2f} ({metrics['target_1_pct']}%)</td>
                            </tr>
                            <tr>
                                <td><strong>Target 2:</strong></td>
                                <td>₹{metrics['target_2']:.2f} ({metrics['target_2_pct']}%)</td>
                                <td><strong>Last Week Accuracy:</strong></td>
                                <td>{metrics['last_week_accuracy']['accuracy_pct']:.1f}%</td>
                            </tr>
                        </table>
                    </div>
                """, unsafe_allow_html=True)

        with tab3:
            # AI Predictions
            st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
            st.subheader("AI Price Predictions")

            # Get combined predictions from multiple models
            predictions = combine_predictions(data['historical_data'])

            if predictions['success']:
                pred_df = pd.DataFrame({
                    'Date': predictions['dates'],
                    'Predicted Price': predictions['predictions']
                })

                fig = go.Figure()

                # Historical prices
                fig.add_trace(go.Scatter(
                    x=data['historical_data'].index[-30:],  # Last 30 days
                    y=data['historical_data']['Close'][-30:],
                    mode='lines',
                    name='Historical',
                    line=dict(color='#888888')
                ))

                # Predictions
                fig.add_trace(go.Scatter(
                    x=pred_df['Date'],
                    y=pred_df['Predicted Price'],
                    mode='lines+markers',
                    name='AI Prediction',
                    line=dict(color='#00ff00')
                ))

                fig.update_layout(
                    template="plotly_dark",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=400,
                    title={
                        'text': "Price Forecast (Combined AI Models)",
                        'y': 0.95,
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'
                    }
                )

                st.plotly_chart(fig, use_container_width=True)

                # Prediction metrics
                st.markdown("""
                    <div style='margin-top: 20px;'>
                        <h4>Prediction Details</h4>
                        <p>Combined forecast using multiple AI models:</p>
                        <ul>
                            <li>ARIMA (Time Series Analysis)</li>
                            <li>SARIMA (Seasonal Time Series)</li>
                            <li>Prophet (Facebook's Forecasting Tool)</li>
                            <li>XGBoost (Gradient Boosting)</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)

            else:
                st.error("Failed to generate AI predictions. Please try again later.")

            st.markdown("</div>", unsafe_allow_html=True)

        with tab4:
            # News and Sentiment Analysis
            news_data = asyncio.run(fetch_stock_news(symbol))

            for news in news_data:
                sentiment_color = {
                    'Positive': '#00ff00',
                    'Negative': '#ff0000',
                    'Neutral': '#888888'
                }[news['sentiment_label']]

                st.markdown(f"""
                    <div class='prediction-card'>
                        <h4>{news['title']}</h4>
                        <p>{news['summary']}</p>
                        <p style='color: {sentiment_color};'>Sentiment: {news['sentiment_label']} 
                        (Score: {news['sentiment_score']:.2f})</p>
                        <small>Published: {news['time_published']}</small>
                    </div>
                """, unsafe_allow_html=True)

    else:
        st.error(f"Error fetching data: {data['error']}")
else:
    st.markdown("""
        <div style='text-align: center; padding: 2rem;'>
            <h1>📈 Indian Stock Market Analytics</h1>
            <p>Enter a stock symbol from the National Stock Exchange (NSE) to get started.</p>
            <div style='background: rgba(255, 255, 255, 0.1); border-radius: 10px; padding: 20px; margin-top: 20px;'>
                <h3>Features</h3>
                <ul style='list-style-type: none; padding: 0;'>
                    <li>🎯 Real-time Trading Signals</li>
                    <li>🤖 AI-powered Price Predictions</li>
                    <li>📊 Technical Analysis</li>
                    <li>📰 News Sentiment Analysis</li>
                </ul>
            </div>
        </div>
    """, unsafe_allow_html=True)