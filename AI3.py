from textblob import TextBlob
import pandas as pd


def analyze_sentiment(text):
    blob = TextBlob(text)

    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    if polarity > 0:
        sentiment = 'positive'
    elif polarity < 0:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'

    return sentiment, polarity, subjectivity


df = pd.read_csv('customer_reviews.csv')

df['Sentiment'] = df['reviews.text'].apply(lambda x: analyze_sentiment(x)[0])

print(df[['reviews.text', 'Sentiment']])
