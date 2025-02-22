import streamlit as st
import plotly.graph_objects as go
from utils import get_stock_data, format_large_number
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Indian Stock Data Visualization",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
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
    .cached-data-notice {
        padding: 8px;
        background-color: #f0f2f6;
        border-radius: 4px;
        font-size: 14px;
        color: #666;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("📈 Indian Stock Data Visualization")
st.markdown("""
Enter a stock symbol from the National Stock Exchange (NSE).
- For example: RELIANCE, TCS, INFY, HDFCBANK
- Don't include the .NS suffix, it will be added automatically
""")

# Input for stock symbol
col1, col2 = st.columns([2, 1])
with col1:
    symbol = st.text_input("Enter NSE Stock Symbol (e.g., RELIANCE, TCS)", value="")
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    search_button = st.button("Search", use_container_width=True)

if symbol and search_button:
    # Show loading spinner
    with st.spinner(f'Fetching data for {symbol.upper()}...'):
        data = get_stock_data(symbol.upper())

    if data['success']:
        # Display company name
        st.subheader(data['company_name'])

        # Show cached data notice if applicable
        if data.get('cached'):
            st.markdown("""
                <div class='cached-data-notice'>
                    ℹ️ Showing cached data due to temporary connection issues with live data source
                </div>
            """, unsafe_allow_html=True)

        # Create metrics display
        metrics = data['metrics']
        col1, col2, col3, col4 = st.columns(4)

        # Current Price and Previous Close
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

        # Market Cap
        with col2:
            st.markdown(f"""
                <div style='text-align: center;'>
                    <div class='metric-value'>{format_large_number(metrics['Market Cap'])}</div>
                    <div class='metric-label'>Market Cap</div>
                </div>
            """, unsafe_allow_html=True)

        # PE Ratio
        with col3:
            st.markdown(f"""
                <div style='text-align: center;'>
                    <div class='metric-value'>{metrics['PE Ratio']:.2f}</div>
                    <div class='metric-label'>P/E Ratio</div>
                </div>
            """, unsafe_allow_html=True)

        # Volume
        with col4:
            st.markdown(f"""
                <div style='text-align: center;'>
                    <div class='metric-value'>{format_large_number(metrics['Volume'])}</div>
                    <div class='metric-label'>Volume</div>
                </div>
            """, unsafe_allow_html=True)

        # Create price chart
        st.subheader("Price History")
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
            xaxis_title="Date",
            yaxis_title="Price (INR)",
            template="plotly_white",
            height=600,
            xaxis_rangeslider_visible=False
        )

        st.plotly_chart(fig, use_container_width=True)

        # Create metrics table
        st.subheader("Key Metrics")
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
            label="Download Data as CSV",
            data=csv,
            file_name=f"{symbol}_metrics.csv",
            mime="text/csv"
        )

    else:
        st.error(f"Error fetching data: {data['error']}")
else:
    # Show example/placeholder content
    st.info("👆 Enter an NSE stock symbol above and click 'Search' to view the data")