from flask import Flask, render_template, request, jsonify
import re
from googleapiclient.discovery import build
import joblib
import matplotlib.pyplot as plt
import io
import base64
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator

# Initialize the Flask app
app = Flask(__name__)

# 1. Setup YouTube Data API Client
api_key = 'Enter your API key' 
youtube = build("youtube", "v3", developerKey=api_key)

def get_video_id_from_url(url): 
    """
    Extracts the video ID from a YouTube URL.
    """
    match = re.search(r'(?:v=|\/)([a-zA-Z0-9_-]{11})', url)
    if match:
        return match.group(1)
    return None

def get_youtube_comments(video_url):
    """
    Fetch YouTube comments using the YouTube Data API.
    """
    video_id = get_video_id_from_url(video_url)
    if not video_id:
        raise ValueError("Invalid YouTube URL")

    comments = []
    results = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100,  
    ).execute()
    
    for item in results.get("items", []):
        comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        comments.append(comment)
    
    return comments

def preprocess_text(text):
    """
    Preprocess the text by removing special characters and converting to lowercase.
    """
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.lower()

def predict_sentiment(comments, model, vectorizer):
    """
    Predicts sentiment for a list of comments using the trained model.
    """
    if not isinstance(model, BaseEstimator):
        raise TypeError("Loaded model is not a valid scikit-learn model")
    
    if not isinstance(vectorizer, TfidfVectorizer):
        raise TypeError("Loaded vectorizer is not a valid TfidfVectorizer instance")
    
    comments_vec = vectorizer.transform(comments)
    predictions = model.predict(comments_vec)
    return predictions

def create_sentiment_plot(sentiments):
    """
    Creates a pie chart of sentiment distribution.
    """
    sentiment_counts = Counter(sentiments)
    labels = list(sentiment_counts.keys())
    sizes = list(sentiment_counts.values())
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    
    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Sentiment Distribution')
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        video_url = request.form.get("video_url")
        try:
            comments = get_youtube_comments(video_url)
            comments_cleaned = [preprocess_text(comment) for comment in comments]
            
            # Load trained model and vectorizer
            model = joblib.load('best_model.pkl')
            vectorizer = joblib.load('tfidf_vectorizer.pkl')
            
            predictions = predict_sentiment(comments_cleaned, model, vectorizer)
            
            results = [
                {"comment": comment, "sentiment": sentiment}
                for comment, sentiment in zip(comments, predictions)
            ]

            # Create sentiment distribution plot
            plot_url = create_sentiment_plot(predictions)

            # Calculate sentiment ratios
            sentiment_counts = Counter(predictions)
            total_comments = len(predictions)
            sentiment_ratios = {sentiment: count / total_comments for sentiment, count in sentiment_counts.items()}

            return render_template("index.html", results=results, plot_url=plot_url, sentiment_ratios=sentiment_ratios)

        except Exception as e:
            return render_template("index.html", error_message=f"Error: {str(e)}")

    return render_template("index.html", results=None)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
