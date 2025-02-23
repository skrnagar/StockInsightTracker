import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

def get_stock_data(symbol):
    """
    Fetch stock data from Yahoo Finance
    """
    try:
        logger.info(f"Fetching data for symbol: {symbol}")

        # Append .NS for NSE stocks if not already present
        if not symbol.endswith('.NS'):
            symbol = f"{symbol}.NS"

        stock = yf.Ticker(symbol)

        # Get historical data for the past year
        hist = stock.history(period="1y")
        logger.info(f"Historical data fetched for {symbol}")

        # Get company info
        try:
            info = stock.info
            logger.info(f"Company info fetched for {symbol}")
        except Exception as info_error:
            logger.warning(f"Failed to fetch company info: {str(info_error)}")
            info = {}

        # Create summary metrics
        metrics = {
            'Current Price': hist['Close'].iloc[-1] if not hist.empty else 0,
            'Previous Close': hist['Close'].iloc[-2] if len(hist) > 1 else 0,
            'Market Cap': info.get('marketCap', 0),
            'PE Ratio': info.get('trailingPE', 0),
            'Volume': hist['Volume'].iloc[-1] if not hist.empty else 0
        }

        return {
            'success': True,
            'historical_data': hist,
            'metrics': metrics,
            'company_name': info.get('longName', symbol.replace('.NS', ''))
        }
    except Exception as e:
        logger.error(f"Error fetching stock data: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

def format_large_number(num):
    """
    Format large numbers in Indian format (with crores and lakhs)
    """
    try:
        if num >= 1e7:  # More than 1 crore
            return f"₹{num/1e7:.2f}Cr"
        elif num >= 1e5:  # More than 1 lakh
            return f"₹{num/1e5:.2f}L"
        else:
            return f"₹{num:,.2f}"
    except Exception as e:
        logger.error(f"Error formatting number: {str(e)}")
        return str(num)