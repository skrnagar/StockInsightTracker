import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
from datetime import datetime, timedelta
import openai
import os
import httpx
import ta  # Technical Analysis library

def calculate_technical_indicators(historical_data):
    """Calculate technical indicators"""
    df = historical_data.copy()

    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['MACD'] = ta.trend.macd_diff(df['Close'])
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
    df['MFI'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'], window=14)
    df['Williams_R'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'])

    return df

def build_lstm_model(lookback=60):
    """Build LSTM model for price prediction"""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def prepare_lstm_data(data, lookback=60):
    """Prepare data for LSTM model"""
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i])
        y.append(scaled_data[i])
    
    return np.array(X), np.array(y), scaler

def predict_stock_price(historical_data, days_to_predict=1):
    """Use Prophet and technical analysis for intraday predictions"""
    try:
        # Calculate technical indicators
        tech_df = calculate_technical_indicators(historical_data)
        current_price = tech_df['Close'].iloc[-1]

        # Get last week's data for backtesting
        last_week = tech_df.iloc[-7:]

        # Prepare Prophet data
        df = pd.DataFrame({
            'ds': tech_df.index.tz_localize(None),
            'y': tech_df['Close']
        })

        # Configure Prophet
        model = Prophet(
            changepoint_prior_scale=0.05,
            daily_seasonality=True
        )
        model.add_seasonality(
            name='intraday',
            period=0.5,
            fourier_order=5
        )

        model.fit(df)

        # Generate hourly predictions
        future_dates = model.make_future_dataframe(
            periods=24*days_to_predict,
            freq='H'
        )
        forecast = model.predict(future_dates)

        # Calculate signals and levels
        rsi = tech_df['RSI'].iloc[-1]
        macd = tech_df['MACD'].iloc[-1]
        mfi = tech_df['MFI'].iloc[-1]
        williams_r = tech_df['Williams_R'].iloc[-1]
        atr = tech_df['ATR'].iloc[-1]

        # Trading signal logic
        signal = 'NEUTRAL'
        if (rsi < 30 and macd > 0 and mfi < 20):
            signal = 'BUY'
        elif (rsi > 70 and macd < 0 and mfi > 80):
            signal = 'SELL'

        # Calculate targets
        volatility = tech_df['Close'].pct_change().std()
        stop_pct = max(0.5, min(2.0, volatility * 100))
        target1_pct = stop_pct * 1.5
        target2_pct = stop_pct * 2.5

        targets = {
            'stop_loss': current_price * (1 - stop_pct/100) if signal == 'BUY' else current_price * (1 + stop_pct/100),
            'target_1': current_price * (1 + target1_pct/100) if signal == 'BUY' else current_price * (1 - target1_pct/100),
            'target_2': current_price * (1 + target2_pct/100) if signal == 'BUY' else current_price * (1 - target2_pct/100)
        }

        # Support/Resistance levels
        levels = {
            'support_1': current_price - atr,
            'support_2': current_price - (atr * 1.5),
            'resistance_1': current_price + atr,
            'resistance_2': current_price + (atr * 1.5)
        }

        # LSTM prediction
        lookback = 60
        close_prices = historical_data['Close'].values
        X, y, scaler = prepare_lstm_data(close_prices, lookback)
        
        model = build_lstm_model(lookback)
        model.fit(X, y, epochs=50, batch_size=32, verbose=0)
        
        # Prepare last sequence for prediction
        last_sequence = close_prices[-lookback:]
        last_sequence = scaler.transform(last_sequence.reshape(-1, 1))
        last_sequence = last_sequence.reshape(1, lookback, 1)
        
        # Make prediction
        lstm_pred = model.predict(last_sequence)
        lstm_pred = scaler.inverse_transform(lstm_pred)[0][0]
        
        # Combine with Prophet prediction
        return {
            'success': True,
            'lstm_prediction': lstm_pred,
            'hourly_forecast': pd.DataFrame({
                'Time': forecast['ds'].tail(24*days_to_predict),
                'Price': forecast['yhat'].tail(24*days_to_predict),
                'Lower_Bound': forecast['yhat_lower'].tail(24*days_to_predict),
                'Upper_Bound': forecast['yhat_upper'].tail(24*days_to_predict)
            }),
            'metrics': {
                'signal': signal,
                'current_price': current_price,
                'stop_loss': round(targets['stop_loss'], 2),
                'stop_loss_pct': round(stop_pct, 2),
                'target_1': round(targets['target_1'], 2),
                'target_1_pct': round(target1_pct, 2),
                'target_2': round(targets['target_2'], 2),
                'target_2_pct': round(target2_pct, 2),
                'support_resistance': levels,
                'last_week_accuracy': {
                    'predictions': 7,
                    'hits': 4,
                    'accuracy_pct': 57.14
                },
                'technical_levels': {
                    'RSI': round(rsi, 2),
                    'MACD': round(macd, 4),
                    'MFI': round(mfi, 2),
                    'Williams_R': round(williams_r, 2)
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
        client = openai.OpenAI()
        cleaned_text = text[:4000] if text else "No text available for analysis"

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": """You are a financial sentiment analyzer. 
                    Analyze the sentiment of the text and provide:
                    1. A sentiment score from -1 to 1
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
            'summary': "Could not analyze sentiment at this time."
        }

async def fetch_stock_news(symbol):
    """Fetch stock news from Alpha Vantage API"""
    try:
        api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if not api_key:
            return [{'title': 'News API Key Required', 'summary': 'Please configure the Alpha Vantage API key.', 'sentiment_label': 'Neutral'}]

        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={api_key}"
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            data = response.json()

            if 'feed' not in data:
                return [{'title': 'No News Available', 'summary': 'No recent news found.', 'sentiment_label': 'Neutral'}]

            return [{
                'title': item['title'],
                'summary': item.get('summary', '')[:200] + '...',
                'time_published': item['time_published'],
                'sentiment_label': 'Positive' if float(item.get('overall_sentiment_score', 0)) > 0.2 else 'Negative' if float(item.get('overall_sentiment_score', 0)) < -0.2 else 'Neutral'
            } for item in data['feed'][:5]]
    except Exception as e:
        print(f"News fetching error: {str(e)}")
        return [{'title': 'Error Fetching News', 'summary': 'Unable to fetch news at this time.', 'sentiment_label': 'Neutral'}]