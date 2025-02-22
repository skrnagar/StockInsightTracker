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

    # Bollinger Bands
    df['BB_Upper'] = ta.volatility.bollinger_hband(df['Close'])
    df['BB_Middle'] = ta.volatility.bollinger_mavg(df['Close'])
    df['BB_Lower'] = ta.volatility.bollinger_lband(df['Close'])

    # Stochastic Oscillator
    df['Stoch_K'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
    df['Stoch_D'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'])

    # Volume-based indicators
    df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()

    # ATR for volatility
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])

    # Additional trend indicators
    df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'])
    df['CCI'] = ta.trend.cci(df['High'], df['Low'], df['Close'])

    return df

def get_indicator_signals(df):
    """Generate trading signals based on technical indicators"""
    current = df.iloc[-1]
    prev = df.iloc[-2]

    signals = {
        'RSI': {
            'value': round(current['RSI'], 2),
            'signal': 'Oversold' if current['RSI'] < 30 else 'Overbought' if current['RSI'] > 70 else 'Neutral',
            'interpretation': 'Indicates momentum and potential reversal points'
        },
        'MACD': {
            'value': round(current['MACD'], 4),
            'signal': 'Bullish' if current['MACD'] > 0 and current['MACD'] > prev['MACD'] 
                     else 'Bearish' if current['MACD'] < 0 and current['MACD'] < prev['MACD']
                     else 'Neutral',
            'interpretation': 'Shows trend direction and momentum'
        },
        'Bollinger': {
            'upper': round(current['BB_Upper'], 2),
            'middle': round(current['BB_Middle'], 2),
            'lower': round(current['BB_Lower'], 2),
            'signal': 'Oversold' if current['Close'] < current['BB_Lower'] 
                     else 'Overbought' if current['Close'] > current['BB_Upper']
                     else 'Neutral',
            'interpretation': 'Measures volatility and potential price levels'
        },
        'Stochastic': {
            'K': round(current['Stoch_K'], 2),
            'D': round(current['Stoch_D'], 2),
            'signal': 'Oversold' if current['Stoch_K'] < 20 
                     else 'Overbought' if current['Stoch_K'] > 80
                     else 'Neutral',
            'interpretation': 'Momentum indicator comparing closing price to price range'
        },
        'Moving_Averages': {
            'SMA20': round(current['SMA_20'], 2),
            'SMA50': round(current['SMA_50'], 2),
            'signal': 'Bullish' if current['SMA_20'] > current['SMA_50'] 
                     else 'Bearish' if current['SMA_20'] < current['SMA_50']
                     else 'Neutral',
            'interpretation': 'Trend direction and potential support/resistance levels'
        },
        'ADX': {
            'value': round(current['ADX'], 2),
            'signal': 'Strong Trend' if current['ADX'] > 25 else 'Weak Trend',
            'interpretation': 'Measures trend strength regardless of direction'
        }
    }

    return signals

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
        tech_df = calculate_technical_indicators(historical_data)
        indicator_signals = get_indicator_signals(tech_df)
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

        # Generate trading signals (using new indicator signals)
        signal = 'NEUTRAL'
        stop_loss = current_price
        target_1 = current_price
        target_2 = current_price
        stop_loss_pct = 0.0
        target_1_pct = 0.0
        target_2_pct = 0.0
        trend = 'Neutral'

        #Simplified logic -  replace with more sophisticated strategy based on indicator_signals
        if indicator_signals['RSI']['signal'] == 'Oversold' and indicator_signals['MACD']['signal'] == 'Bullish':
            signal = 'BUY'
        elif indicator_signals['RSI']['signal'] == 'Overbought' and indicator_signals['MACD']['signal'] == 'Bearish':
            signal = 'SELL'

        #Placeholder - needs improvement
        atr = tech_df['ATR'].iloc[-1]
        if signal == 'BUY':
            stop_loss = current_price - (atr * 2)
            target_1 = current_price + (atr * 3)
            target_2 = current_price + (atr * 5)
        elif signal == 'SELL':
            stop_loss = current_price + (atr * 2)
            target_1 = current_price - (atr * 3)
            target_2 = current_price - (atr * 5)
        
        stop_loss_pct = ((stop_loss - current_price) / current_price) * 100
        target_1_pct = ((target_1 - current_price) / current_price) * 100
        target_2_pct = ((target_2 - current_price) / current_price) * 100

        recent_trend = (
            forecast['trend'].tail(days_to_predict).mean() -
            forecast['trend'].tail(days_to_predict * 2).head(days_to_predict).mean()
        )
        trend = 'Upward' if recent_trend > 0 else 'Downward'

        trading_metrics = {
            'signal': signal,
            'current_price': current_price,
            'stop_loss': round(stop_loss, 2),
            'stop_loss_pct': round(stop_loss_pct, 2),
            'target_1': round(target_1, 2),
            'target_1_pct': round(target_1_pct, 2),
            'target_2': round(target_2, 2),
            'target_2_pct': round(target_2_pct, 2),
            'indicators': indicator_signals,
            'trend': trend,
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