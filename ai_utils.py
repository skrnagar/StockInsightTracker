import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from prophet import Prophet
from datetime import datetime, timedelta
import openai
import os

def prepare_data_for_prediction(historical_data):
    """Prepare data for ML prediction"""
    df = historical_data.copy()
    df.reset_index(inplace=True)
    df = df[['Date', 'Close']]
    df.columns = ['ds', 'y']
    return df

def predict_stock_price(historical_data, days_to_predict=30):
    """Use Prophet to predict future stock prices"""
    try:
        # Prepare data
        df = prepare_data_for_prediction(historical_data)
        
        # Create and fit Prophet model
        model = Prophet(daily_seasonality=True, 
                       yearly_seasonality=True, 
                       weekly_seasonality=True,
                       changepoint_prior_scale=0.05)
        model.fit(df)
        
        # Create future dates dataframe
        future_dates = model.make_future_dataframe(periods=days_to_predict)
        
        # Make predictions
        forecast = model.predict(future_dates)
        
        # Calculate confidence intervals
        forecast_data = pd.DataFrame({
            'Date': forecast['ds'].tail(days_to_predict),
            'Predicted_Price': forecast['yhat'].tail(days_to_predict),
            'Lower_Bound': forecast['yhat_lower'].tail(days_to_predict),
            'Upper_Bound': forecast['yhat_upper'].tail(days_to_predict)
        })
        
        # Calculate additional metrics
        metrics = {
            'trend': 'Upward' if forecast['trend'].tail(days_to_predict).mean() > 0 else 'Downward',
            'confidence_interval': (forecast['yhat_upper'] - forecast['yhat_lower']).mean()
        }
        
        return {
            'success': True,
            'forecast_data': forecast_data,
            'metrics': metrics
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def analyze_sentiment(text):
    """Analyze sentiment of text using OpenAI"""
    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024
            messages=[
                {
                    "role": "system",
                    "content": "You are a financial sentiment analyzer. "
                    "Analyze the sentiment of the text and provide a rating from -1 to 1 "
                    "where -1 is very negative, 0 is neutral, and 1 is very positive. "
                    "Also provide a confidence score between 0 and 1. "
                    "Format the response as JSON: "
                    "{'sentiment': number, 'confidence': number, 'summary': 'brief explanation'}"
                },
                {"role": "user", "content": text}
            ],
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
    except Exception as e:
        return {
            'sentiment': 0,
            'confidence': 0,
            'summary': f"Error analyzing sentiment: {str(e)}"
        }

async def fetch_stock_news(symbol):
    """Fetch stock news from Alpha Vantage API"""
    try:
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={os.getenv('ALPHA_VANTAGE_API_KEY')}"
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            data = response.json()
            
            if 'feed' not in data:
                return []
                
            news_items = []
            for item in data['feed'][:5]:  # Get latest 5 news items
                sentiment_score = float(item.get('overall_sentiment_score', 0))
                news_items.append({
                    'title': item['title'],
                    'url': item['url'],
                    'time_published': item['time_published'],
                    'summary': item.get('summary', ''),
                    'sentiment_score': sentiment_score,
                    'sentiment_label': 'Positive' if sentiment_score > 0.2 
                                     else 'Negative' if sentiment_score < -0.2 
                                     else 'Neutral'
                })
            return news_items
    except Exception as e:
        print(f"Error fetching news: {str(e)}")
        return []
