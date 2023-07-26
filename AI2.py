import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def analyze_sentiment_vader(text):
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)

    if scores['compound'] >= 0.05:
        sentiment = 'positive'
    elif scores['compound'] <= -0.05:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'

    return sentiment, scores['compound']

sample_text = "I'm really HAPPY with the service."
sentiment, compound_score = analyze_sentiment_vader(sample_text)

print(f"Text: {sample_text}")
print(f"Sentiment: {sentiment}")
print(f"Compound Score: {compound_score}")
