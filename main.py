import yfinance as yf
from transformers import pipeline
import numpy as np
import pandas as pd
import finnhub
from flask import Flask, request, render_template
from flask_cors import CORS
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv
import os

try:
    pipe = pipeline("text-classification", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
    print("[INFO] Loaded financial sentiment analysis model successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load the financial sentiment model: {e}")
    pipe = None

def fetch_news(company, from_date=None, to_date=None):
    load_dotenv()
    api_key = os.getenv("FINNHUB_API_KEY")
    finnhub_client = finnhub.Client(api_key=api_key)
    
    news = finnhub_client.company_news(symbol=company, _from=from_date, to=to_date)
    if news:
        print(f"[DEBUG] Fetched {len(news)} articles for company '{company}' from {from_date} to {to_date}.")
        return news
    else:
        print(f"[ERROR] Failed to fetch articles for '{company}'.")
        return []

# Preprocess
def preprocess_articles(articles):
    contents = [article["summary"] for article in articles if article.get("summary")]
    print(f"[DEBUG] Preprocessed {len(contents)} articles with valid content.")
    return contents

# Analyze and calculate mean score of sentiemnt (1-100 scale)
def analyze_sentiment(articles):
    if not pipe:
        print("[ERROR] Sentiment analysis pipeline is unavailable.")
        return 0

    if not articles:
        print("[DEBUG] No articles available for sentiment analysis.")
        return 0

    # sentiment analysis
    sentiments = pipe(articles)
    print(f"[DEBUG] Sentiment analysis results: {sentiments}")

    total_articles = len(articles)
    
    positive_score = 0
    neutral_score = 0
    negative_score = 0
    total_confidence = 0

    for sentiment in sentiments:
        label = sentiment['label']
        confidence = sentiment['score']  # Use confidence score by the model
    
        if label == 'positive':
            positive_score += confidence
        elif label == 'neutral':
            neutral_score += confidence
        else:
            negative_score += confidence
        
        total_confidence += confidence
    
    # calc weighted sentiment scores
    weighted_positive = (positive_score / total_confidence) * 100 if total_confidence else 0
    weighted_negative = (negative_score / total_confidence) * 100 if total_confidence else 0
    weighted_neutral = (neutral_score / total_confidence) * 100 if total_confidence else 0

    # Calculate overall sentiment confidence as weighted average
    sentiment_score = weighted_positive - weighted_negative
    sentiment_score = round(sentiment_score, 2)
    
    print(f"[DEBUG] Total articles: {total_articles}, Weighted Positive Score: {weighted_positive}, Weighted Negative Score: {weighted_negative}, Neutral Score: {weighted_neutral}")
    print(f"[DEBUG] Sentiment Score: {sentiment_score} (positive - negative)")

    return sentiment_score

def fetch_stock_data(company, from_date=None, to_date=None):
    # Download stock data for company
    stock_data = yf.download(company, start=from_date, end=to_date)
    
    if stock_data.empty:
        print(f"[ERROR] Failed to fetch stock data for {company}.")
        return None
    
    print(f"[DEBUG] Fetched stock data for {company}.")
    
    # Ensure data has the columns 'Adj Close' or 'Close' as not null
    if 'Adj Close' in stock_data.columns:
        price_data = stock_data['Adj Close']
    else:
        price_data = stock_data['Close']
        print("[WARNING] 'Adj Close' column missing, using 'Close' instead.")
    
    # 'recent_growth' = percentage change over the last 5 days
    recent_growth = price_data.pct_change(periods=5).iloc[-1] * 100
    
    # 'historical_growth' = percentage change from the first to the last day
    historical_growth = (price_data.iloc[-1] / price_data.iloc[0] - 1) * 100
    
    # Calculate 'volatility' as annual standard deviation of daily returns
    daily_returns = price_data.pct_change().dropna()
    volatility = daily_returns.std() * np.sqrt(252)  # annual volatility
    
    stock_data_dict = {
        "recent_growth": recent_growth,
        "historical_growth": historical_growth,
        "volatility": volatility
    }
    
    print(f"[DEBUG] Stock Data Calculated: {stock_data_dict}")
    
    return stock_data_dict


def calculate_confidence(sentiment_score, stock_data, investing_horizon):
    # Ensure sentiment_score is not NaN
    sentiment_score = 0 if pd.isna(sentiment_score) else sentiment_score

    # issing values in stock_data, gonna change later s othat all of it is actually corect
    stock_data = {key: (0 if pd.isna(value).any() else value) for key, value in stock_data.items()}

    if investing_horizon == "short-term":
        features = np.array([sentiment_score, stock_data.get("recent_growth", 0)]).reshape(1, -1)
        model = LinearRegression()

        # Example historical data (replace with actual historical later)
        X_train = np.array([[50, 0.5], [70, 0.7], [30, 0.2]])  # Example historical data
        y_train = np.array([80, 90, 60])  # Corresponding confidence scores

        # Normalize
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        features_scaled = scaler.transform(features)

        model.fit(X_train_scaled, y_train)

    else:
        # Long-term investing additional factors (e.g., stability)
        features = np.array([sentiment_score, stock_data.get("historical_growth", 0), stock_data.get("volatility", 0)]).reshape(1, -1)
        model = LinearRegression()

        # Example training data for long-term investing - gonan fix later
        X_train = np.array([[50, 1.0, 0.3], [70, 1.5, 0.2], [30, 0.5, 0.4]])
        y_train = np.array([85, 95, 65])  # Corresponding confidence scores

        # Normalize
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        features_scaled = scaler.transform(features)

        model.fit(X_train_scaled, y_train)

    confidence = model.predict(features_scaled)[0]
    confidence = max(0, min(100, confidence))  #within 0 & 100
    print(f"[DEBUG] Calculated confidence: {confidence} using sentiment {sentiment_score} and stock data {stock_data}")
    return round(confidence, 2)


def recommend_stocks(companies, preferences):
    recommendations = []
    total_articles = 0
    from_date = preferences.get("from_date")
    to_date = preferences.get("to_date")
    investing_horizon = preferences.get("investing_horizon")
    
    for company in companies:
        print(f"[DEBUG] Processing company: {company}")
        news = fetch_news(company, from_date=from_date, to_date=to_date)
        articles = preprocess_articles(news)
        sentiment_score = analyze_sentiment(articles)
        
        #  get the recent growth, historical growth and volatility
        stock_data = fetch_stock_data(company, from_date=from_date, to_date=to_date)
        
        if stock_data:
            confidence = calculate_confidence(sentiment_score, stock_data, investing_horizon)

            if confidence > preferences["min_confidence"]:
                recommendations.append({
                    "company": company,
                    "confidence": confidence,
                    "total_articles": len(articles)
                })
    
    return recommendations

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    if request.method == 'POST':
        companies = request.form['companies'].split(',')
        min_confidence = float(request.form['min_confidence'])
        from_date = request.form['from_date']
        to_date = request.form['to_date']
        investing_horizon = request.form['investing_horizon']

        preferences = {
            'min_confidence': min_confidence,
            'from_date': from_date,
            'to_date': to_date,
            'investing_horizon': investing_horizon
        }

        recommendations = recommend_stocks(companies, preferences)
    return render_template('index.html', recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
