import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
from datetime import datetime, timedelta
import openai
import os
import httpx
import ta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def calculate_technical_indicators(df):
    """Calculate essential technical indicators"""
    try:
        # Basic trend indicators
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
        df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)

        # Momentum indicators
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd_diff()
        df['ROC'] = ta.momentum.roc(df['Close'], window=9)

        # Volume indicators
        df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        df['MFI'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'])

        # Volatility indicators
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])

        # Calculate Bollinger Bands
        rolling_mean = df['Close'].rolling(window=20).mean()
        rolling_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = rolling_mean + (rolling_std * 2)
        df['BB_Middle'] = rolling_mean
        df['BB_Lower'] = rolling_mean - (rolling_std * 2)

        return df
    except Exception as e:
        print(f"Error calculating technical indicators: {str(e)}")
        raise e

def prepare_features(df):
    """Prepare features for ML model"""
    features = pd.DataFrame()
    features['Close'] = df['Close']
    features['Volume'] = df['Volume']
    features['RSI'] = df['RSI']
    features['MACD'] = df['MACD']
    features['SMA_20'] = df['SMA_20']
    features['SMA_50'] = df['SMA_50']
    features['ATR'] = df['ATR']

    return features.dropna()

def predict_stock_price(historical_data):
    """Generate trading signals and predictions"""
    try:
        # Calculate technical indicators
        df = calculate_technical_indicators(historical_data.copy())
        current_price = df['Close'].iloc[-1]

        # Get last week's data for backtesting
        last_week = df.last('7D')

        # Calculate base volatility for targets
        atr = df['ATR'].iloc[-1]
        avg_daily_move = df['Close'].pct_change().abs().mean() * 100

        # Generate trading signal based on technical indicators
        signal = 'NEUTRAL'
        confidence = 0.0

        # Get latest indicator values
        latest = {
            'rsi': df['RSI'].iloc[-1],
            'macd': df['MACD'].iloc[-1],
            'mfi': df['MFI'].iloc[-1],
            'sma20': df['SMA_20'].iloc[-1],
            'sma50': df['SMA_50'].iloc[-1],
            'bb_upper': df['BB_Upper'].iloc[-1],
            'bb_lower': df['BB_Lower'].iloc[-1],
            'price': current_price
        }

        # Bullish conditions
        bullish_signals = [
            latest['rsi'] < 30,  # Oversold
            latest['macd'] > 0,  # Positive MACD
            latest['price'] > latest['sma20'],  # Price above SMA20
            latest['mfi'] < 20,  # Oversold volume
            latest['price'] > latest['bb_lower']  # Price above BB lower
        ]

        # Bearish conditions
        bearish_signals = [
            latest['rsi'] > 70,  # Overbought
            latest['macd'] < 0,  # Negative MACD
            latest['price'] < latest['sma20'],  # Price below SMA20
            latest['mfi'] > 80,  # Overbought volume
            latest['price'] < latest['bb_upper']  # Price below BB upper
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
                    'RSI': round(latest['rsi'], 2),
                    'MACD': round(latest['macd'], 4),
                    'SMA_20': round(latest['sma20'], 2),
                    'SMA_50': round(latest['sma50'], 2),
                    'MFI': round(latest['mfi'], 2),
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

def predict_lstm(historical_data, prediction_days=5):
    """Generate predictions using Random Forest instead of LSTM"""
    try:
        # Prepare features
        df = calculate_technical_indicators(historical_data.copy())
        features = prepare_features(df)

        # Create target variable (next day's price)
        features['Target'] = features['Close'].shift(-1)
        features = features.dropna()

        X = features.drop(['Target', 'Close'], axis=1)
        y = features['Target']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Make predictions
        predictions = []
        last_features = X.iloc[-1:].copy()

        for _ in range(prediction_days):
            pred = model.predict(last_features)[0]
            predictions.append(pred)

            # Update features for next prediction
            last_features['Volume'] = last_features['Volume'].mean()
            last_features['RSI'] = last_features['RSI'].mean()
            last_features['MACD'] = last_features['MACD'].mean()
            last_features['SMA_20'] = pred
            last_features['SMA_50'] = last_features['SMA_50'].mean()
            last_features['ATR'] = last_features['ATR'].mean()

        return {
            'success': True,
            'predictions': predictions,
            'dates': [
                (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d')
                for i in range(1, prediction_days + 1)
            ]
        }

    except Exception as e:
        print(f"ML prediction error: {str(e)}")
        return {
            'success': False,
            'error': f"Failed to generate ML prediction: {str(e)}"
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