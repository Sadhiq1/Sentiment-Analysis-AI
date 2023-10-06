import pandas as pd
from textblob import TextBlob

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

train_df['ABSTRACT'] = train_df['ABSTRACT'].astype(str).fillna('')
test_df['ABSTRACT'] = test_df['ABSTRACT'].astype(str).fillna('')

def analyze_sentiment(text):
    sentiment = TextBlob(text).sentiment.polarity
    if sentiment > 0:
        return 'positive'
    elif sentiment < 0:
        return 'negative'
    else:
        return 'neutral'

print("Training Data Sentiment Analysis:")
for index, row in train_df.iterrows():
    sentiment = analyze_sentiment(row['ABSTRACT'])
    print(f"ID: {row['ID']}, TITLE: {row['TITLE']}, SENTIMENT: {sentiment}")

print("\nTesting Data Sentiment Analysis:")
for index, row in test_df.iterrows():
    sentiment = analyze_sentiment(row['ABSTRACT'])
    print(f"ID: {row['ID']}, TITLE: {row['TITLE']}, SENTIMENT: {sentiment}")
