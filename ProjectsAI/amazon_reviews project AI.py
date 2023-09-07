import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

df = pd.read_csv('amazon_reviews.csv')

df['text'] = df['text'].fillna('')

df['cleaned_text'] = df['text'].apply(lambda x: ' '.join([word.lower() for word in nltk.word_tokenize(str(x)) if word.isalnum()]))

analyzer = SentimentIntensityAnalyzer()

df['sentiment_scores'] = df['cleaned_text'].apply(lambda x: analyzer.polarity_scores(x))

df['sentiment'] = df['sentiment_scores'].apply(lambda x: 'positive' if x['compound'] > 0 else ('negative' if x['compound'] < 0 else 'neutral'))

print(df[['text', 'sentiment']])


