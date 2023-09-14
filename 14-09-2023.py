import nltk
from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt

data = pd.DataFrame({
    'text': [
        "I love this product!",
        "This is terrible.",
        "Great service!",
        "Not happy with the experience.",
        "Amazing customer support!"
    ]
})

def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = ''.join([char for char in text if char.isalpha() or char.isspace()])  # Remove special characters and numbers
    return text

data['text'] = data['text'].apply(preprocess_text)

def analyze_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity < 0:
        return 'negative'
    else:
        return 'neutral'

data['sentiment'] = data['text'].apply(analyze_sentiment)

sentiment_counts = data['sentiment'].value_counts()
sentiment_counts.plot(kind='bar', color=['green', 'red', 'blue'])
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Sentiment Analysis Results')
plt.xticks(rotation=0)
plt.show()

