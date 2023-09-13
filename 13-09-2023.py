import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

sentences = [
    "I love this product!",
    "I don't dislike this product.",
    "The movie was surprisingly good.",
    "Sure, the weather is nice today.",
    "That concert was a blast!",
    "I can't recommend this enough.",
    "This book is amazing.",
    "The product is not bad.",
    "I'm happy with this service.",
    "I do not like this product",
    "I do not dislike this product",
    "I'm not unhappy with this service",
    "I'm really not happy with this service",
    "I'm not sure about this product",
    "This car is not as great as I expected.",
]

for sentence in sentences:
    sentiment = sia.polarity_scores(sentence)
    print(f"Sentence: '{sentence}'")
    print(f"Sentiment Score: {sentiment['compound']:.2f}")
    if sentiment['compound'] >= 0.05:
        print("Sentiment: Positive")
    elif sentiment['compound'] <= -0.05:
        print("Sentiment: Negative")
    else:
        print("Sentiment: Neutral")
    print()