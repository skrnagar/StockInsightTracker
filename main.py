import streamlit as st
import plotly.graph_objects as go
from utils import get_stock_data, format_large_number
from ai_utils import predict_stock_price
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Indian Stock Data Visualization",
    page_icon="📈",
    layout="wide"
)

# Custom CSS (Merging relevant styles)
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to bottom, #111827, #1F2937);
    }
    .main {
        padding: 2rem;
    }
    .block-container {
        padding-top: 2rem;
    }
    .css-1d391kg {
        padding-top: 2rem;
    }
    .stProgress .st-bo {
        background-color: #0E86D4;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
    }
    .metric-label {
        font-size: 14px;
        color: #666;
    }
    .prediction-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #eee;
        margin: 1rem 0;
    }
    .prediction-header {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #eee;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📈 Stock Analysis")
symbol = st.sidebar.text_input("Enter NSE Stock Symbol (e.g., RELIANCE, TCS)", value="")

if symbol:
    with st.spinner(f'Analyzing {symbol.upper()}...'):
        data = get_stock_data(symbol.upper())

    if data['success']:
        # Display company name and basic metrics
        st.title(f"📊 {data['company_name']}")

        # Key metrics
        metrics = data['metrics']
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            price_change = metrics['Current Price'] - metrics['Previous Close']
            price_change_pct = (price_change / metrics['Previous Close']) * 100
            color = "green" if price_change >= 0 else "red"
            st.markdown(f"""
                <div style='text-align: center;'>
                    <div class='metric-value'>₹{metrics['Current Price']:.2f}</div>
                    <div class='metric-label'>Current Price</div>
                    <div style='color: {color};'>
                        {'+' if price_change >= 0 else ''}₹{price_change:.2f} ({price_change_pct:.2f}%)
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
                <div style='text-align: center;'>
                    <div class='metric-value'>{format_large_number(metrics['Market Cap'])}</div>
                    <div class='metric-label'>Market Cap</div>
                </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
                <div style='text-align: center;'>
                    <div class='metric-value'>{metrics['PE Ratio']:.2f}</div>
                    <div class='metric-label'>P/E Ratio</div>
                </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
                <div style='text-align: center;'>
                    <div class='metric-value'>{format_large_number(metrics['Volume'])}</div>
                    <div class='metric-label'>Volume</div>
                </div>
            """, unsafe_allow_html=True)

        # Create tabs
        tab1, tab2 = st.tabs(["📈 Price Chart", "🎯 Trading Signals"])

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
                title="Price History",
                xaxis_title="Date",
                yaxis_title="Price (INR)",
                template="plotly_white",
                height=600,
                xaxis_rangeslider_visible=False
            )

            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            # Trading signals and predictions
            prediction = predict_stock_price(data['historical_data'])

            if prediction['success']:
                metrics = prediction['metrics']
                signal_color = {
                    'BUY': 'green',
                    'SELL': 'red',
                    'NEUTRAL': 'gray'
                }[metrics['signal']]

                # Signal and Targets
                st.markdown(f"""
                    <div class='prediction-card'>
                        <div class='prediction-header' style='color: {signal_color};'>
                            {metrics['signal']} Signal
                        </div>
                        <table style='width: 100%;'>
                            <tr>
                                <td><strong>Current Price:</strong></td>
                                <td>₹{metrics['current_price']:.2f}</td>
                                <td><strong>Stop Loss:</strong></td>
                                <td>₹{metrics['stop_loss']:.2f} ({metrics['stop_loss_pct']}%)</td>
                            </tr>
                            <tr>
                                <td><strong>Target 1:</strong></td>
                                <td>₹{metrics['target_1']:.2f} ({metrics['target_1_pct']}%)</td>
                                <td><strong>Target 2:</strong></td>
                                <td>₹{metrics['target_2']:.2f} ({metrics['target_2_pct']}%)</td>
                            </tr>
                        </table>
                    </div>
                """, unsafe_allow_html=True)

                # Technical Indicators
                st.markdown("""
                    <div class='prediction-card'>
                        <div class='prediction-header'>Technical Indicators</div>
                        </div>
                """, unsafe_allow_html=True)

                tech_col1, tech_col2, tech_col3 = st.columns(3)

                indicators = metrics['technical_indicators']
                with tech_col1:
                    st.metric("RSI (14)", f"{indicators['RSI']:.2f}")
                    st.metric("MACD", f"{indicators['MACD']:.4f}")

                with tech_col2:
                    st.metric("SMA 20", f"₹{indicators['SMA_20']:.2f}")
                    st.metric("SMA 50", f"₹{indicators['SMA_50']:.2f}")

                with tech_col3:
                    st.metric("MFI", f"{indicators['MFI']:.2f}")
                    st.metric("ATR", f"₹{indicators['ATR']:.2f}")

                # Last Week's Performance
                accuracy = metrics['last_week_accuracy']
                st.markdown(f"""
                    <div class='prediction-card'>
                        <div class='prediction-header'>Last Week's Performance</div>
                        <p>Success Rate: {accuracy['accuracy_pct']:.1f}% ({accuracy['hits']} out of {accuracy['predictions']} predictions)</p>
                    </div>
                """, unsafe_allow_html=True)

                # Trading Guidelines
                st.markdown("""
                    <div class='prediction-card'>
                        <div class='prediction-header'>Trading Guidelines</div>
                        <ul>
                            <li>Set stop-loss orders at the indicated level to manage risk</li>
                            <li>Consider taking partial profits at Target 1</li>
                            <li>Move stop-loss to break-even after Target 1 is reached</li>
                            <li>Hold remaining position for Target 2</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.error("Failed to generate predictions. Please try again later.")

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

            # Download button
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
    st.markdown("""
        <div style='text-align: center; padding: 2rem;'>
            <h1>📈 Indian Stock Market Analytics</h1>
            <p>Enter a stock symbol from the National Stock Exchange (NSE) to get started.</p>
            <p>Examples: RELIANCE, TCS, INFY, HDFCBANK</p>
        </div>
    """, unsafe_allow_html=True)