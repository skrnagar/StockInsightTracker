# Indian Stock Data Visualization

A Streamlit-based web application for visualizing Indian stock market data from the National Stock Exchange (NSE). This tool provides real-time stock information, interactive charts, and downloadable metrics for Indian stocks.

## Features

- **Real-time Stock Data**: Fetch current stock prices and metrics from Yahoo Finance
- **Interactive Charts**: Candlestick chart showing historical price movements
- **Key Metrics Display**: View important financial metrics including:
  - Current Price and Price Change
  - Market Capitalization
  - P/E Ratio
  - Trading Volume
  - 52-Week High/Low
  - Dividend Yield
- **Data Export**: Download stock metrics as CSV files
- **Indian Market Focus**: Specifically designed for NSE-listed stocks

## Installation

The project uses Python 3.11 and requires the following packages:
- streamlit
- yfinance
- pandas
- plotly

To run the application:

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install streamlit yfinance pandas plotly
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run main.py
   ```

## Usage

1. Enter an NSE stock symbol in the input field (e.g., RELIANCE, TCS, INFY)
2. Click the "Search" button
3. View the interactive chart and metrics
4. Download data using the "Download Data as CSV" button

## Example Stocks

Here are some popular NSE stocks you can try:
- RELIANCE (Reliance Industries)
- TCS (Tata Consultancy Services)
- INFY (Infosys)
- HDFCBANK (HDFC Bank)
- TATAMOTORS (Tata Motors)
- WIPRO (Wipro Limited)

## Data Format

- Stock prices are displayed in Indian Rupees (₹)
- Large numbers are formatted in Indian notation:
  - B for Billion
  - Cr for Crore
  - L for Lakh

## Project Structure

- `main.py`: Main Streamlit application file
- `utils.py`: Utility functions for data fetching and formatting
- `.streamlit/config.toml`: Streamlit configuration file

## Notes

- The application automatically appends '.NS' to stock symbols to fetch NSE data
- Data is sourced from Yahoo Finance API through the yfinance package
- Chart shows 1-year historical data by default
