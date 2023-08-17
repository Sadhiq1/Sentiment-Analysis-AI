import nltk
from nltk.tokenize import word_tokenize

# Load the AFINN-111 sentiment lexicon
afinn = {}
with open("AFINN-111.txt", "r") as afinn_file:
    for line in afinn_file:
        word, score = line.strip().split("\t")
        afinn[word] = int(score)


# Function to calculate sentiment score of a text
def calculate_sentiment(text):
    words = word_tokenize(text.lower())
    sentiment_score = sum(afinn.get(word, 0) for word in words)
    return sentiment_score


# Analyze sentiment of a text
def analyze_sentiment(text):
    sentiment_score = calculate_sentiment(text)

    if sentiment_score > 0:
        sentiment = "positive"
    elif sentiment_score < 0:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return sentiment, sentiment_score


# Example text
text = "I love this product! It's amazing."

# Analyze sentiment
sentiment, score = analyze_sentiment(text)

# Display results
print(f"Text: {text}")
print(f"Sentiment: {sentiment}")
print(f"Sentiment Score: {score}")

