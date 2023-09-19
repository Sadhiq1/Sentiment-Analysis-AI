import nltk
#nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

social_media_data = [
    "I love this product! It's amazing!",
    "Had a terrible experience with their customer service.",
    "Just received my order. It's exactly what I wanted!",
    "Feeling happy and excited today!",
    "The weather is awful. I hate rainy days.",
]

for post in social_media_data:
    sentiment = analyzer.polarity_scores(post)
    print(f"Post: {post}")
    print(f"Sentiment: {sentiment}")

import matplotlib.pyplot as plt

compound_scores = [analyzer.polarity_scores(post)['compound'] for post in social_media_data]

# Plot sentiment
plt.bar(social_media_data, compound_scores)
plt.xlabel('Social Media Posts')
plt.ylabel('Sentiment (Compound Score)')
plt.title('Sentiment Analysis of Social Media Posts')
plt.xticks(rotation=45, ha="right")
plt.show()
