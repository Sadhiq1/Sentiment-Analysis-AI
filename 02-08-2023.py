#Text Preprocessing for Sentiment Analysis
'''import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#nltk.download('stopwords')
import string
text = "This is a sample text for sentiment analysis. It includes some punctuation marks and stopwords like 'the', 'is', 'a'."
tokens = word_tokenize(text)
lowercase_tokens = [token.lower() for token in tokens]
print(lowercase_tokens)

table = str.maketrans('', '', string.punctuation)
stripped_tokens = [token.translate(table) for token in lowercase_tokens]

stop_words = set(stopwords.words('english'))
filtered_tokens = [token for token in stripped_tokens if token not in stop_words]
print(filtered_tokens)'''

#Sentiment Analysis using TextBlob
'''from textblob import TextBlob
text = "I love this product! It's amazing and works perfectly."
blob = TextBlob(text)
sentiment_score = blob.sentiment.polarity
sentiment = "positive" if sentiment_score > 0 else "negative" if sentiment_score < 0 else "neutral"

print(f"Sentiment: {sentiment}")
print(f"Sentiment Score: {sentiment_score}")'''
#Sentiment Analysis using VADER
'''from nltk.sentiment import SentimentIntensityAnalyzer
text = "This movie is terrible. I don't like it at all."
analyzer = SentimentIntensityAnalyzer()
scores = analyzer.polarity_scores(text)

sentiment_score = scores["compound"]
sentiment = "positive" if sentiment_score > 0 else "negative" if sentiment_score < 0 else "neutral"

print(f"Sentiment: {sentiment}")
print(f"Sentiment Score: {sentiment_score}")'''

#Sentiment Analysis with BERT using Transformers
from transformers import BertTokenizer, BertForSequenceClassification
import torch
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
model.eval()
text = "I am happy with the service provided. Everything was great!"
tokens = tokenizer.encode(text, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    outputs = model(tokens)
    logits = outputs.logits

predicted_class = torch.argmax(logits).item()
sentiment = "positive" if predicted_class ==1 else "negative"

print(f"Sentiment: {sentiment}")

























