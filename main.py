import requests
from transformers import pipeline
import numpy as np
from sklearn.preprocessing import RobustScaler
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
import os

sentiment_model = pipeline("sentiment-analysis", model="yiyanghkust/finbert")

# fetch news articles
def fetch_news(company):
    load_dotenv()
    api_key = os.getenv("NEWS_API_KEY")
    url = f"https://newsapi.org/v2/everything?q={company}&apiKey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json().get("articles", [])
        print(f"[DEBUG] {len(articles)} articles for company '{company}'")
        return articles
    else:
        print(f"[DEBUG] '{company}' ded. code: {response.status_code}")
        return []

# Function to preprocess articles
def preprocess_articles(articles):
    contents = [article["content"] for article in articles if article.get("content")]
    print(f"[DEBUG] Preprocessed {len(contents)} articles with valid content")
    return contents

# analyze sentiment & calculate mean score from 1 to 100
def analyze_sentiment(articles):
    if not articles:
        print("[DEBUG] No articles available for sentiment analysis")
        return 50  # Default neutral score
    
    # sentiment of each article
    sentiments = [sentiment_model(article) for article in articles]
    print(f"[DEBUG] Sentiment analysis results: {sentiments}")
    
    scores = []
    for result in sentiments:
        label = result[0]['label']
        
        # assign sentiment labels from scale 1 to 100
        if '1 star' in label:
            sentiment_value = 1 
        elif '2 stars' in label:
            sentiment_value = 25
        elif '3 stars' in label:
            sentiment_value = 50
        elif '4 stars' in label:
            sentiment_value = 75
        elif '5 stars' in label:
            sentiment_value = 100
        else:
            sentiment_value = 50  # Default to neutral
        
        scores.append(sentiment_value)
    
    # Normalize scores woth RobustScaler (to handle outliers better)
    scaler = RobustScaler()
    normalized_scores = scaler.fit_transform(np.array(scores).reshape(-1, 1)).flatten()
    
    mean_score = np.mean(normalized_scores) if normalized_scores else 50  # Default to neutral
    print(f"[DEBUG] Mean sentiment score (normalized): {mean_score}")
    return mean_score

# calculate confidence scores
def calculate_confidence(sentiment_score, stock_data):
    features = np.array([sentiment_score, stock_data["recent_growth"]]).reshape(1, -1)
    # fix later
    model = LinearRegression()
    model.fit([[0.2, 0.1], [0.8, 0.7]], [30, 85])  # Dummy data
    confidence = model.predict(features)[0]
    print(f"[DEBUG] Calculated confidence score: {confidence} using sentiment {sentiment_score} and recent growth {stock_data['recent_growth']}")
    return round(confidence, 2)

# recommend stocks
def recommend_stocks(companies, preferences):
    recommendations = []
    for company in companies:
        print(f"[DEBUG] Processing company: {company}")
        news = fetch_news(company)
        articles = preprocess_articles(news)
        sentiment_score = analyze_sentiment(articles)
        stock_data = {"recent_growth": 0.5}  # Placeholder, fix later
        confidence = calculate_confidence(sentiment_score, stock_data)
        
        print(f"[DEBUG] Confidence score for {company}: {confidence}")
        if confidence > preferences["min_confidence"]:
            print(f"[DEBUG] Adding {company} to recommendations")
            recommendations.append({"company": company, "confidence": confidence})
        else:
            print(f"[DEBUG] {company} does not meet the minimum confidence threshold")
    return recommendations

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    data = request.json
    companies = data.get("companies", [])
    preferences = data.get("preferences", {})
    print(f"[DEBUG] Received request with companies: {companies} and preferences: {preferences}")
    recommendations = recommend_stocks(companies, preferences)
    print(f"[DEBUG] Recommendations: {recommendations}")
    return jsonify(recommendations)

if __name__ == "__main__":
    app.run(debug=True)
