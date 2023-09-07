import pandas as pd
from textblob import TextBlob

df = pd.read_csv('twitter_data.csv')

df['tweet_text'] = df['tweet_text'].astype(str).fillna('')

df['sentiment'] = df['tweet_text'].apply(lambda tweet: TextBlob(tweet).sentiment.polarity)

df['sentiment_category'] = df['sentiment'].apply(lambda polarity: 'positive' if polarity > 0 else ('negative' if polarity < 0 else 'neutral'))

print(df[['tweet_text', 'sentiment', 'sentiment_category']])


