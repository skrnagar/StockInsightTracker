import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def get_stock_data(symbol):
    """
    Fetch stock data from Yahoo Finance
    """
    try:
        stock = yf.Ticker(symbol)
        
        # Get historical data for the past year
        hist = stock.history(period="1y")
        
        # Get company info
        info = stock.info
        
        # Create summary metrics
        metrics = {
            'Current Price': info.get('currentPrice', 0),
            'Previous Close': info.get('previousClose', 0),
            'Market Cap': info.get('marketCap', 0),
            'PE Ratio': info.get('trailingPE', 0),
            'Dividend Yield': info.get('dividendYield', 0) if info.get('dividendYield') else 0,
            '52 Week High': info.get('fiftyTwoWeekHigh', 0),
            '52 Week Low': info.get('fiftyTwoWeekLow', 0),
            'Volume': info.get('volume', 0)
        }
        
        return {
            'success': True,
            'historical_data': hist,
            'metrics': metrics,
            'company_name': info.get('longName', symbol)
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def format_large_number(num):
    """
    Format large numbers to human-readable format
    """
    if num >= 1e12:
        return f"${num/1e12:.2f}T"
    elif num >= 1e9:
        return f"${num/1e9:.2f}B"
    elif num >= 1e6:
        return f"${num/1e6:.2f}M"
    else:
        return f"${num:,.2f}"
