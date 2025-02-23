import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from database import cache_stock_data, get_cached_stock_data, update_user_preference

def get_stock_data(symbol):
    """
    Fetch stock data from Yahoo Finance and cache it in database
    """
    try:
        # Append .NS for NSE stocks if not already present
        if not symbol.endswith('.NS'):
            symbol = f"{symbol}.NS"

        stock = yf.Ticker(symbol)

        # Get historical data for the past year
        hist = stock.history(period="1y")

        # Cache the data in database
        try:
            cache_stock_data(symbol, hist)
            update_user_preference(symbol)
        except Exception as e:
            print(f"Warning: Failed to cache data: {str(e)}")

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
            'metrics': {
                'Current Price': hist['Close'].iloc[-1],
                'Previous Close': hist['Close'].iloc[-2],
                'Market Cap': info.get('marketCap', 0),
                'PE Ratio': info.get('trailingPE', 0),
                'Dividend Yield': info.get('dividendYield', 0),
                '52 Week High': hist['High'].max(),
                '52 Week Low': hist['Low'].min(),
                'Volume': hist['Volume'].iloc[-1]
            },
            'company_name': info.get('longName', symbol.replace('.NS', ''))
        }
    except Exception as e:
        # Try to get cached data if live fetch fails
        try:
            cached_data = get_cached_stock_data(symbol)
            if cached_data:
                hist_data = pd.DataFrame([{
                    'Open': d.open_price,
                    'High': d.high_price,
                    'Low': d.low_price,
                    'Close': d.close_price,
                    'Volume': d.volume
                } for d in cached_data], index=[d.date for d in cached_data])

                metrics = {
                    'Current Price': float(hist_data['Close'].iloc[-1]),
                    'Previous Close': float(hist_data['Close'].iloc[-2]),
                    'Market Cap': 0,  # Not available in cached data
                    'PE Ratio': 0,    # Not available in cached data
                    'Dividend Yield': 0,
                    '52 Week High': float(hist_data['High'].max()),
                    '52 Week Low': float(hist_data['Low'].min()),
                    'Volume': int(hist_data['Volume'].iloc[-1])
                }

                return {
                    'success': True,
                    'historical_data': hist_data,
                    'metrics': metrics,
                    'company_name': symbol.replace('.NS', ''),
                    'cached': True
                }
        except Exception as cache_error:
            print(f"Cache retrieval failed: {str(cache_error)}")

        return {
            'success': False,
            'error': str(e)
        }

def format_large_number(num):
    """
    Format large numbers to human-readable format (in INR)
    """
    if num >= 1e9:
        return f"₹{num/1e9:.2f}B"
    elif num >= 1e7:
        return f"₹{num/1e7:.2f}Cr"
    elif num >= 1e5:
        return f"₹{num/1e5:.2f}L"
    else:
        return f"₹{num:,.2f}"