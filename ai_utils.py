import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from prophet import Prophet
from datetime import datetime, timedelta
import openai
import os
import httpx
import ta  # Technical Analysis library

def calculate_technical_indicators(historical_data):
    """Calculate comprehensive technical indicators"""
    df = historical_data.copy()

    # Trend Indicators
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
    df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)

    # MACD
    df['MACD'] = ta.trend.macd_diff(df['Close'])
    df['MACD_Line'] = ta.trend.macd(df['Close'])
    df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])

    # RSI
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])

    # Bollinger Bands
    df['BB_Upper'] = ta.volatility.bollinger_hband(df['Close'])
    df['BB_Middle'] = ta.volatility.bollinger_mavg(df['Close'])
    df['BB_Lower'] = ta.volatility.bollinger_lband(df['Close'])

    # Intraday specific indicators
    df['ROC'] = ta.momentum.roc(df['Close'], window=9)  # Rate of Change
    df['Williams_R'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'], lbp=14)
    df['MFI'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'], window=14)

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
    """Use Prophet and technical analysis for intraday predictions"""
    try:
        # Calculate technical indicators
        tech_df = calculate_technical_indicators(historical_data)
        current_price = tech_df['Close'].iloc[-1]

        # Get last week's data for backtesting
        last_week = tech_df.last('7D')

        # Prepare Prophet data
        df = pd.DataFrame({
            'ds': tech_df.index,
            'y': tech_df['Close']
        })

        # Configure Prophet for intraday predictions
        model = Prophet(
            changepoint_prior_scale=0.05,
            daily_seasonality=True
        )

        # Add custom seasonalities for intraday patterns
        model.add_seasonality(
            name='intraday',
            period=0.5,
            fourier_order=5
        )

        try:
            model.fit(df)
        except Exception as fit_error:
            print(f"Model fitting error: {str(fit_error)}")
            raise Exception("Failed to fit prediction model to the data")

        # Generate hourly predictions
        future_dates = model.make_future_dataframe(
            periods=24*days_to_predict,
            freq='H'
        )
        forecast = model.predict(future_dates)

        # Calculate intraday levels
        levels = generate_intraday_levels(
            current_price,
            tech_df['ATR'].iloc[-1],
            tech_df['RSI'].iloc[-1],
            'Upward' if forecast['trend'].diff().mean() > 0 else 'Downward'
        )

        # Get last week's prediction accuracy
        last_week_accuracy = {
            'predictions': len(last_week),
            'hits': sum(1 for i in range(len(last_week)-1) 
                       if (last_week['Close'].iloc[i] < last_week['Close'].iloc[i+1] and 
                           last_week['RSI'].iloc[i] < 70) or
                          (last_week['Close'].iloc[i] > last_week['Close'].iloc[i+1] and 
                           last_week['RSI'].iloc[i] > 30)),
            'accuracy_pct': 0.0
        }
        last_week_accuracy['accuracy_pct'] = (
            last_week_accuracy['hits'] / last_week_accuracy['predictions'] * 100
            if last_week_accuracy['predictions'] > 0 else 0
        )

        # Prepare hourly forecast data
        hourly_forecast = pd.DataFrame({
            'Time': forecast['ds'].tail(24*days_to_predict),
            'Price': forecast['yhat'].tail(24*days_to_predict),
            'Lower_Bound': forecast['yhat_lower'].tail(24*days_to_predict),
            'Upper_Bound': forecast['yhat_upper'].tail(24*days_to_predict)
        })

        # Generate trading signals
        rsi = tech_df['RSI'].iloc[-1]
        macd = tech_df['MACD'].iloc[-1]
        mfi = tech_df['MFI'].iloc[-1]
        williams_r = tech_df['Williams_R'].iloc[-1]

        # Combined signal logic
        signal = 'NEUTRAL'
        if (rsi < 30 and macd > 0 and mfi < 20 and williams_r < -80):
            signal = 'BUY'
        elif (rsi > 70 and macd < 0 and mfi > 80 and williams_r > -20):
            signal = 'SELL'

        # Calculate targets based on ATR
        atr = tech_df['ATR'].iloc[-1]
        targets = {
            'stop_loss': current_price - (atr * 2) if signal == 'BUY' else current_price + (atr * 2),
            'target_1': current_price + (atr * 3) if signal == 'BUY' else current_price - (atr * 3),
            'target_2': current_price + (atr * 5) if signal == 'BUY' else current_price - (atr * 5)
        }

        # Calculate percentages
        price_targets = {
            'stop_loss_pct': ((targets['stop_loss'] - current_price) / current_price) * 100,
            'target_1_pct': ((targets['target_1'] - current_price) / current_price) * 100,
            'target_2_pct': ((targets['target_2'] - current_price) / current_price) * 100
        }

        trading_metrics = {
            'signal': signal,
            'current_price': current_price,
            'stop_loss': round(targets['stop_loss'], 2),
            'stop_loss_pct': round(price_targets['stop_loss_pct'], 2),
            'target_1': round(targets['target_1'], 2),
            'target_1_pct': round(price_targets['target_1_pct'], 2),
            'target_2': round(targets['target_2'], 2),
            'target_2_pct': round(price_targets['target_2_pct'], 2),
            'support_resistance': levels,
            'last_week_accuracy': last_week_accuracy,
            'technical_levels': {
                'RSI': round(rsi, 2),
                'MACD': round(macd, 4),
                'MFI': round(mfi, 2),
                'Williams_R': round(williams_r, 2)
            }
        }

        return {
            'success': True,
            'hourly_forecast': hourly_forecast,
            'metrics': trading_metrics
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
            model="gpt-3.5-turbo",  # Using GPT-3.5-turbo model
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