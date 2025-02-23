import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
from datetime import datetime, timedelta
import openai
import os
import httpx
import ta
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

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

        # Calculate Bollinger Bands manually since ta.volatility.bollinger_bands is not working
        rolling_mean = df['Close'].rolling(window=20).mean()
        rolling_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = rolling_mean + (rolling_std * 2)
        df['BB_Middle'] = rolling_mean
        df['BB_Lower'] = rolling_mean - (rolling_std * 2)

        return df
    except Exception as e:
        print(f"Error calculating technical indicators: {str(e)}")
        raise e

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

def create_lstm_model(input_shape):
    """Create and compile LSTM model"""
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def prepare_lstm_data(data, look_back=60):
    """Prepare data for LSTM model"""
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i])
        y.append(scaled_data[i, 0])

    return np.array(X), np.array(y), scaler

def predict_lstm(historical_data, prediction_days=5):
    """Generate predictions using LSTM model"""
    try:
        # Prepare data
        data = historical_data[['Close']].values
        X, y, scaler = prepare_lstm_data(data)

        # Split data
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Create and train model
        model = create_lstm_model((X.shape[1], X.shape[2]))
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=0)

        # Make predictions
        last_sequence = X[-1:]
        predictions = []

        for _ in range(prediction_days):
            next_pred = model.predict(last_sequence)
            predictions.append(next_pred[0, 0])

            # Update sequence for next prediction
            last_sequence = np.roll(last_sequence, -1)
            last_sequence[0, -1] = next_pred

        # Inverse transform predictions
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = scaler.inverse_transform(predictions)

        return {
            'success': True,
            'predictions': predictions.flatten().tolist(),
            'dates': [
                (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d')
                for i in range(1, prediction_days + 1)
            ]
        }

    except Exception as e:
        print(f"LSTM prediction error: {str(e)}")
        return {
            'success': False,
            'error': f"Failed to generate LSTM prediction: {str(e)}"
        }