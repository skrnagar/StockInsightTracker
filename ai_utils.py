import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
from datetime import datetime, timedelta
import openai
import os
import httpx
import ta  # Technical Analysis library

def calculate_technical_indicators(df):
    """Calculate essential technical indicators"""
    # Trend
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
    df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)

    # Momentum
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['MACD'] = ta.trend.macd_diff(df['Close'])
    df['ROC'] = ta.momentum.roc(df['Close'], window=9)

    # Volume
    df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    df['MFI'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'])

    # Volatility
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
    df['BBL'], df['BBM'], df['BBU'] = ta.volatility.bollinger_bands(df['Close'])

    return df

def prepare_intraday_prediction_data(df):
    """Prepare data for intraday prediction"""
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 
               'RSI', 'MACD', 'ROC', 'Williams_R', 'MFI']

    X = df[features].values
    y = df['Close'].values

    # Scale the features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

def generate_intraday_levels(current_price, atr, rsi, trend):
    """Generate intraday support/resistance levels"""
    volatility_factor = 1.5 if rsi > 70 or rsi < 30 else 1.0

    levels = {
        'support_1': current_price - (atr * volatility_factor),
        'support_2': current_price - (atr * volatility_factor * 1.5),
        'resistance_1': current_price + (atr * volatility_factor),
        'resistance_2': current_price + (atr * volatility_factor * 1.5)
    }

    # Adjust levels based on trend
    if trend == 'Upward':
        levels = {k: v * 1.02 for k, v in levels.items()}
    elif trend == 'Downward':
        levels = {k: v * 0.98 for k, v in levels.items()}

    return levels

def predict_stock_price(historical_data, days_to_predict=1):
    """Generate trading signals and predictions"""
    try:
        # Calculate technical indicators
        df = calculate_technical_indicators(historical_data)
        current_price = df['Close'].iloc[-1]

        # Get last week's data for backtesting
        last_week = df.last('7D')

        # Calculate base volatility for targets
        atr = df['ATR'].iloc[-1]
        avg_daily_move = df['Close'].pct_change().abs().mean() * 100  # Average daily move in percentage

        # Generate trading signal based on technical indicators
        signal = 'NEUTRAL'
        confidence = 0.0

        # Bullish conditions
        bullish_signals = [
            df['RSI'].iloc[-1] < 30,  # Oversold
            df['MACD'].iloc[-1] > 0,  # Positive MACD
            df['Close'].iloc[-1] > df['SMA_20'].iloc[-1],  # Price above SMA20
            df['MFI'].iloc[-1] < 20,  # Oversold volume
            df['Close'].iloc[-1] > df['BBM'].iloc[-1]  # Price above BB middle
        ]

        # Bearish conditions
        bearish_signals = [
            df['RSI'].iloc[-1] > 70,  # Overbought
            df['MACD'].iloc[-1] < 0,  # Negative MACD
            df['Close'].iloc[-1] < df['SMA_20'].iloc[-1],  # Price below SMA20
            df['MFI'].iloc[-1] > 80,  # Overbought volume
            df['Close'].iloc[-1] < df['BBM'].iloc[-1]  # Price below BB middle
        ]

        # Calculate signal strength
        bullish_strength = sum(bullish_signals) / len(bullish_signals)
        bearish_strength = sum(bearish_signals) / len(bearish_signals)

        if bullish_strength > 0.6:
            signal = 'BUY'
            confidence = bullish_strength
        elif bearish_strength > 0.6:
            signal = 'SELL'
            confidence = bearish_strength

        # Calculate dynamic target percentages based on volatility
        base_target = max(0.5, min(avg_daily_move * 1.5, 5.0))  # Between 0.5% and 5%

        # Generate targets and stop-loss
        if signal == 'BUY':
            stop_loss = current_price * (1 - base_target/100)
            target_1 = current_price * (1 + base_target/100)
            target_2 = current_price * (1 + base_target*2/100)
        elif signal == 'SELL':
            stop_loss = current_price * (1 + base_target/100)
            target_1 = current_price * (1 - base_target/100)
            target_2 = current_price * (1 - base_target*2/100)
        else:
            stop_loss = target_1 = target_2 = current_price

        # Calculate last week's accuracy
        last_week_accuracy = {
            'predictions': len(last_week) - 1,
            'hits': sum(1 for i in range(len(last_week)-1) 
                       if (signal == 'BUY' and last_week['Close'].iloc[i] < last_week['Close'].iloc[i+1]) or
                          (signal == 'SELL' and last_week['Close'].iloc[i] > last_week['Close'].iloc[i+1])),
            'accuracy_pct': 0.0
        }

        if last_week_accuracy['predictions'] > 0:
            last_week_accuracy['accuracy_pct'] = (
                last_week_accuracy['hits'] / last_week_accuracy['predictions'] * 100
            )

        return {
            'success': True,
            'signal': signal,
            'confidence': confidence * 100,  # Convert to percentage
            'current_price': current_price,
            'metrics': {
                'signal': signal,
                'current_price': current_price,
                'stop_loss': round(stop_loss, 2),
                'stop_loss_pct': round(((stop_loss - current_price) / current_price) * 100, 2),
                'target_1': round(target_1, 2),
                'target_1_pct': round(((target_1 - current_price) / current_price) * 100, 2),
                'target_2': round(target_2, 2),
                'target_2_pct': round(((target_2 - current_price) / current_price) * 100, 2),
                'last_week_accuracy': last_week_accuracy,
                'technical_indicators': {
                    'RSI': round(df['RSI'].iloc[-1], 2),
                    'MACD': round(df['MACD'].iloc[-1], 4),
                    'SMA_20': round(df['SMA_20'].iloc[-1], 2),
                    'SMA_50': round(df['SMA_50'].iloc[-1], 2),
                    'MFI': round(df['MFI'].iloc[-1], 2),
                    'ATR': round(atr, 2)
                }
            }
        }
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return {
            'success': False,
            'error': f"Failed to generate prediction: {str(e)}"
        }

def analyze_sentiment(text):
    """Analyze sentiment of text using OpenAI"""
    try:
        # Initialize OpenAI client
        client = openai.OpenAI()

        # Clean and truncate text if needed
        cleaned_text = text[:4000] if text else "No text available for analysis"

        response = client.chat.completions.create(
            model="gpt-4",  # Using standard GPT-4 model
            messages=[
                {
                    "role": "system",
                    "content": """You are a financial sentiment analyzer. 
                    Analyze the sentiment of the text and provide:
                    1. A sentiment score from -1 to 1 where:
                       -1 is very negative
                       0 is neutral
                       1 is very positive
                    2. A confidence score between 0 and 1
                    3. A brief explanation of the sentiment
                    Format as JSON with keys: sentiment, confidence, summary"""
                },
                {"role": "user", "content": cleaned_text}
            ],
            response_format={"type": "json_object"}
        )

        return response.choices[0].message.content
    except Exception as e:
        print(f"Sentiment analysis error: {str(e)}")
        return {
            'sentiment': 0,
            'confidence': 0,
            'summary': "Could not analyze sentiment at this time. Please try again later."
        }

async def fetch_stock_news(symbol):
    """Fetch stock news from Alpha Vantage API"""
    try:
        api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if not api_key:
            return [
                {
                    'title': 'News API Key Required',
                    'summary': 'Please configure the Alpha Vantage API key to view news.',
                    'time_published': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
                    'url': '#',
                    'sentiment_score': 0,
                    'sentiment_label': 'Neutral'
                }
            ]

        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={api_key}"
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            data = response.json()

            if 'feed' not in data:
                return [
                    {
                        'title': 'No News Available',
                        'summary': 'No recent news articles found for this stock.',
                        'time_published': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
                        'url': '#',
                        'sentiment_score': 0,
                        'sentiment_label': 'Neutral'
                    }
                ]

            news_items = []
            for item in data['feed'][:5]:  # Get latest 5 news items
                sentiment_score = float(item.get('overall_sentiment_score', 0))
                news_items.append({
                    'title': item['title'],
                    'url': item['url'],
                    'time_published': item['time_published'],
                    'summary': item.get('summary', '')[:200] + '...',
                    'sentiment_score': sentiment_score,
                    'sentiment_label': 'Positive' if sentiment_score > 0.2 
                                   else 'Negative' if sentiment_score < -0.2 
                                   else 'Neutral'
                })
            return news_items
    except Exception as e:
        print(f"News fetching error: {str(e)}")
        return [
            {
                'title': 'Error Fetching News',
                'summary': 'Unable to fetch news at this time. Please try again later.',
                'time_published': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
                'url': '#',
                'sentiment_score': 0,
                'sentiment_label': 'Neutral'
            }
        ]