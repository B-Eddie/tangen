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
from datetime import datetime, timedelta

try:
    pipe = pipeline("text-classification", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
    print("[INFO] Loaded financial sentiment analysis model successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load the financial sentiment model: {e}")
    pipe = None

# Fetch news articles for a company
def fetch_news(company, investing_horizon):
    load_dotenv()
    api_key = os.getenv("FINNHUB_API_KEY")
    finnhub_client = finnhub.Client(api_key=api_key)
    
    # Calculate the date range based on the investing horizon
    today = datetime.now()
    if investing_horizon == "short-term":
        from_date = today - timedelta(days=30)
    else:  # long-term
        from_date = today - timedelta(days=365*5)
    
    to_date = today
    
    # Convert to the format required by the Finnhub API
    from_date_str = from_date.strftime('%Y-%m-%d')
    to_date_str = to_date.strftime('%Y-%m-%d')
    
    # Fetch news with the date range
    news = finnhub_client.company_news(symbol=company, _from=from_date_str, to=to_date_str)
    if news:
        print(f"[DEBUG] Fetched {len(news)} articles for company '{company}'.")
        return news
    else:
        print(f"[ERROR] Failed to fetch articles for '{company}'.")
        return []

# Preprocess articles for sentiment analysis
def preprocess_articles(articles):
    contents = [article["summary"] for article in articles if article.get("summary")]
    print(f"[DEBUG] Preprocessed {len(contents)} articles with valid content.")
    return contents

# Analyze sentiment score from articles
def analyze_sentiment(articles):
    if not pipe:
        print("[ERROR] Sentiment analysis pipeline is unavailable.")
        return 0

    if not articles:
        print("[DEBUG] No articles available for sentiment analysis.")
        return 0

    sentiments = pipe(articles)
    print(f"[DEBUG] Sentiment analysis results: {sentiments}")

    total_articles = len(articles)
    
    positive_score = 0
    neutral_score = 0
    negative_score = 0
    total_confidence = 0

    for sentiment in sentiments:
        label = sentiment['label']
        confidence = sentiment['score']
    
        if label == 'positive':
            positive_score += confidence
        elif label == 'neutral':
            neutral_score += confidence
        else:
            negative_score += confidence
        
        total_confidence += confidence
    
    # Weighted sentiment scores
    weighted_positive = (positive_score / total_confidence) * 100 if total_confidence else 0
    weighted_negative = (negative_score / total_confidence) * 100 if total_confidence else 0
    weighted_neutral = (neutral_score / total_confidence) * 100 if total_confidence else 0

    # Calculate overall sentiment score
    sentiment_score = (weighted_positive - weighted_negative + 100) / 2
    sentiment_score = round(sentiment_score, 2)
    
    print(f"[DEBUG] Sentiment Score: {sentiment_score} (positive - negative)")

    return sentiment_score

# Fetch stock data (with automatic date range based on investing horizon)
def fetch_stock_data(company, investing_horizon):
    if investing_horizon == "short-term":
        # Fetch the last 30 days for short-term
        stock_data = yf.download(company, period="1mo")
    else:
        # Fetch the last 5 years for long-term
        stock_data = yf.download(company, period="5y")
    
    if stock_data.empty:
        print(f"[ERROR] Failed to fetch stock data for {company}.")
        return None
    
    print(f"[DEBUG] Fetched stock data for {company}.")
    
    # Select 'Adj Close' or 'Close' column
    if 'Adj Close' in stock_data.columns:
        price_data = stock_data['Adj Close']
    else:
        price_data = stock_data['Close']
    
    # Calculate recent growth
    recent_growth = price_data.pct_change(periods=5).iloc[-1] * 100 if len(price_data) > 5 else np.nan
    
    # Calculate historical growth
    historical_growth = (price_data.iloc[-1] / price_data.iloc[0] - 1) * 100 if len(price_data) > 1 else np.nan
    
    # Calculate volatility
    daily_returns = price_data.pct_change().dropna()
    volatility = daily_returns.std() * np.sqrt(252) if not daily_returns.empty else np.nan
    
    stock_data_dict = {
        "recent_growth": recent_growth,
        "historical_growth": historical_growth,
        "volatility": volatility
    }
    
    print(f"[DEBUG] Stock Data Calculated: {stock_data_dict}")
    return stock_data_dict

# Calculate confidence based on sentiment and stock data
def calculate_confidence(sentiment_score, stock_data, investing_horizon, company):
    sentiment_score = 0 if pd.isna(sentiment_score) else sentiment_score

    stock_data = {key: (0 if pd.isna(value).any() else value) for key, value in stock_data.items()}

    if investing_horizon == "short-term":
        recent_growth_val = stock_data.get("recent_growth", 0)
        if isinstance(recent_growth_val, np.ndarray) or isinstance(recent_growth_val, pd.Series):
            recent_growth_val = recent_growth_val.item()

        features = np.array([float(sentiment_score), float(recent_growth_val)]).reshape(1, -1)
        model = LinearRegression()

        ticker = company
        stock = yf.Ticker(ticker)
        hist = stock.history(period="ytd")
        hist['Daily Return'] = hist['Close'].pct_change()
        recent_growth = hist['Daily Return'].mean() * 100

        sentiment_scores = np.full(hist[['Daily Return']].dropna().shape[0], sentiment_score).reshape(-1, 1)
        daily_returns = (hist[['Daily Return']].dropna().values * 100)
        X_train = np.hstack((sentiment_scores, daily_returns))
        y_train = np.random.randint(60, 100, size=X_train.shape[0])

        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        features_scaled = scaler.transform(features)

        model.fit(X_train_scaled, y_train)

    else:
        
        historical_growth_val = stock_data.get("historical_growth", 0)
        if isinstance(historical_growth_val, np.ndarray) or isinstance(historical_growth_val, pd.Series):
            historical_growth_val = historical_growth_val.item()

        volatility_val = stock_data.get("volatility", 0)
        if isinstance(volatility_val, np.ndarray) or isinstance(volatility_val, pd.Series):
            volatility_val = volatility_val.item()

        features = np.array([sentiment_score, historical_growth_val, volatility_val]).reshape(1, -1)

        model = LinearRegression()

        ticker = company
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5y")
        hist['Annual Return'] = hist['Close'].pct_change() * 252
        hist['Volatility'] = hist['Close'].rolling(window=252).std() * np.sqrt(252)

        sentiment_scores_long = np.full(hist[['Annual Return', 'Volatility']].dropna().shape[0], sentiment_score).reshape(-1, 1)
        annual_returns = hist[['Annual Return', 'Volatility']].dropna().values
        X_train = np.hstack((sentiment_scores_long, annual_returns))
        y_train = np.random.randint(65, 100, size=X_train.shape[0])

        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        features_scaled = scaler.transform(features)

        model.fit(X_train_scaled, y_train)

    confidence = model.predict(features_scaled)[0]
    confidence = max(0, min(100, confidence))
    print(f"[DEBUG] Calculated confidence: {confidence} using sentiment {sentiment_score} and stock data {stock_data}")
    return round(confidence, 2)

# Recommend stocks based on preferences
def recommend_stocks(companies, preferences):
    recommendations = []
    investing_horizon = preferences.get("investing_horizon")

    for company in companies:
        print(f"[DEBUG] Processing company: {company}")
        news = fetch_news(company, investing_horizon)
        articles = preprocess_articles(news)
        sentiment_score = analyze_sentiment(articles)
        
        stock_data = fetch_stock_data(company, investing_horizon)
        
        if stock_data:
            confidence = calculate_confidence(sentiment_score, stock_data, investing_horizon, company)

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
        investing_horizon = request.form['investing_horizon']

        preferences = {
            'min_confidence': min_confidence,
            'investing_horizon': investing_horizon
        }

        recommendations = recommend_stocks(companies, preferences)
    return render_template('index.html', recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
