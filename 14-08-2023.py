#Task: Sentiment Analysis Using Lexicons in Python.

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)

    sentiment_label = None
    if sentiment_scores['compound'] >= 0.05:
        sentiment_label = 'Positive'
    elif sentiment_scores['compound'] <= -0.05:
        sentiment_label = 'Negative'
    else:
        sentiment_label = 'Neutral'

    return sentiment_label, sentiment_scores

sample_text = "I absolutely loved the movie! The acting was fantastic."

sentiment_label, sentiment_scores = analyze_sentiment(sample_text)

print("Sample Text:", sample_text)
print("Sentiment Label:", sentiment_label)
print("Sentiment Scores:", sentiment_scores)