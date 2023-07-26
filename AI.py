from textblob import TextBlob


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

sample_text = "I like that movie"
sentiment, polarity, subjectivity = analyze_sentiment(sample_text)

print(f"Text: {sample_text}")
print(f"Sentiment: {sentiment}")
print(f"Polarity: {polarity}")
print(f"Subjectivity: {subjectivity}")
