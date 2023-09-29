import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

reviews = [
    "This product is amazing! I love it.",
    "Not worth the money. Very disappointed.",
    "Great purchase. It exceeded my expectations.",
    "The quality is poor and it broke after a few days.",
]
for review in reviews:
    sentiment_scores = sia.polarity_scores(review)
    sentiment = ""

    if sentiment_scores['compound'] >= 0.05:
        sentiment = "positive"
    elif sentiment_scores['compound'] <= -0.05:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    print(f"Review: {review}")
    print(f"Sentiment: {sentiment}")
    print()