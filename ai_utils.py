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

def calculate_trading_signals(historical_data):
    """Calculate technical indicators and trading signals"""
    df = historical_data.copy()

    # Calculate technical indicators
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['MACD'] = ta.trend.macd_diff(df['Close'])

    # Calculate volatility for stop loss
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])

    return df

def prepare_data_for_prediction(historical_data):
    """Prepare data for ML prediction"""
    df = historical_data.copy()
    df.reset_index(inplace=True)
    df = df[['Date', 'Close']]
    df.columns = ['ds', 'y']
    return df

def predict_stock_price(historical_data, days_to_predict=30):
    """Use Prophet and technical analysis to predict future stock prices and trading signals"""
    try:
        # Calculate technical indicators
        tech_df = calculate_trading_signals(historical_data)
        current_price = tech_df['Close'].iloc[-1]

        # Prepare data for Prophet
        df = prepare_data_for_prediction(historical_data)

        # Create and fit Prophet model with more conservative parameters
        model = Prophet(
            daily_seasonality=True,
            yearly_seasonality=True,
            weekly_seasonality=True,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10,
            holidays_prior_scale=10
        )

        try:
            model.fit(df)
        except Exception as fit_error:
            print(f"Model fitting error: {str(fit_error)}")
            raise Exception("Failed to fit prediction model to the data")

        # Create future dates dataframe
        future_dates = model.make_future_dataframe(periods=days_to_predict)

        # Make predictions
        forecast = model.predict(future_dates)

        # Calculate forecast data
        forecast_data = pd.DataFrame({
            'Date': forecast['ds'].tail(days_to_predict),
            'Predicted_Price': forecast['yhat'].tail(days_to_predict),
            'Lower_Bound': forecast['yhat_lower'].tail(days_to_predict),
            'Upper_Bound': forecast['yhat_upper'].tail(days_to_predict)
        })

        # Generate trading signals
        current_rsi = tech_df['RSI'].iloc[-1]
        current_macd = tech_df['MACD'].iloc[-1]
        sma20_trend = tech_df['SMA_20'].iloc[-1] > tech_df['SMA_20'].iloc[-2]
        sma50_trend = tech_df['SMA_50'].iloc[-1] > tech_df['SMA_50'].iloc[-2]

        # Calculate price targets and stop loss
        atr = tech_df['ATR'].iloc[-1]
        volatility_factor = 2.0  # Adjust risk factor

        # Trading decision logic
        is_bullish = (current_rsi < 70 and current_macd > 0 and 
                     sma20_trend and sma50_trend)
        is_bearish = (current_rsi > 30 and current_macd < 0 and 
                     not sma20_trend and not sma50_trend)

        if is_bullish:
            signal = 'BUY'
            stop_loss = current_price - (atr * volatility_factor)
            target_1 = current_price + (atr * volatility_factor * 1.5)  # 1.5:1 reward:risk
            target_2 = current_price + (atr * volatility_factor * 2.5)  # 2.5:1 reward:risk
            stop_loss_pct = ((stop_loss - current_price) / current_price) * 100
            target_1_pct = ((target_1 - current_price) / current_price) * 100
            target_2_pct = ((target_2 - current_price) / current_price) * 100
        elif is_bearish:
            signal = 'SELL'
            stop_loss = current_price + (atr * volatility_factor)
            target_1 = current_price - (atr * volatility_factor * 1.5)
            target_2 = current_price - (atr * volatility_factor * 2.5)
            stop_loss_pct = ((stop_loss - current_price) / current_price) * 100
            target_1_pct = ((target_1 - current_price) / current_price) * 100
            target_2_pct = ((target_2 - current_price) / current_price) * 100
        else:
            signal = 'NEUTRAL'
            stop_loss = target_1 = target_2 = current_price
            stop_loss_pct = target_1_pct = target_2_pct = 0.0

        # Calculate trend metrics
        recent_trend = (
            forecast['trend'].tail(days_to_predict).mean() -
            forecast['trend'].tail(days_to_predict * 2).head(days_to_predict).mean()
        )

        # Prepare trading metrics
        trading_metrics = {
            'signal': signal,
            'current_price': current_price,
            'stop_loss': round(stop_loss, 2),
            'stop_loss_pct': round(stop_loss_pct, 2),
            'target_1': round(target_1, 2),
            'target_1_pct': round(target_1_pct, 2),
            'target_2': round(target_2, 2),
            'target_2_pct': round(target_2_pct, 2),
            'rsi': round(current_rsi, 2),
            'macd': round(current_macd, 4),
            'trend': 'Upward' if recent_trend > 0 else 'Downward',
            'confidence_interval': round((forecast['yhat_upper'] - forecast['yhat_lower']).mean(), 2),
            'prediction_days': days_to_predict
        }

        return {
            'success': True,
            'forecast_data': forecast_data,
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