#Task 1: Sentiment Analysis using AFINN
import nltk
from afinn import Afinn
afinn = Afinn()

text = "I love this product. It's amazing!"
sentiment_score = afinn.score(text)
print("Sentiment Score:", sentiment_score)

if sentiment_score > 0:
    print("Positive Sentiment")
elif sentiment_score < 0:
    print("Negative Sentiment")
else:
    print("Neutral Sentiment")

#Task 2: Sentiment Analysis using VADER

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

text = "This movie is really disappointing."
sentiment_scores = analyzer.polarity_scores(text)
print("Sentiment scores:", sentiment_scores)

if sentiment_scores['compound'] > 0:
    print("Positive Sentiment")
elif sentiment_scores['compound'] < 0:
    print("Negative Sentiment")
else:
    print("Neutral Sentiment")

#Task 3: Sentiment Analysis using SentiWordNet
import re
import os
from nltk.corpus import wordnet as wn

sentiwordnet_path = 'C:/Users/SSLTP11505/Desktop/AI/archive (1)/sentiwordnet'

def load_sentiwordnet(sentiwordnet_path):
    sentiwordnet = {}

    with open(os.path.join(sentiwordnet_path, 'SentiWordNet_3.0.0.txt'), 'r') as file:
        for line in file:
            if line.startswith("#"):
                continue
            fields = line.strip().split("\t")
            if len(fields) == 6:
                pos, offset, pos_score, neg_score, synset_terms, gloss = fields
                words = synset_terms.split(" ")
                for word in words:
                    match = re.match(r"(.*?)(#\d+)?", word)
                    if match:
                        term = match.group(1)
                        if term:
                            if pos == 'a':  # Convert to WordNet's POS tag format
                                pos_tag = wn.ADJ
                            elif pos == 'v':
                                pos_tag = wn.VERB
                            elif pos == 'r':
                                pos_tag = wn.ADV
                            elif pos == 'n':
                                pos_tag = wn.NOUN
                            else:
                                pos_tag = None

                            if pos_tag:
                                if term not in sentiwordnet:
                                    sentiwordnet[term] = []
                                sentiwordnet[term].append((float(pos_score), float(neg_score)))

    return sentiwordnet

def analyze_sentiment(word, sentiwordnet):
    if word in sentiwordnet:
        pos_score, neg_score = 0, 0
        for pos, neg in sentiwordnet[word]:
            pos_score += pos
            neg_score += neg
        if pos_score > neg_score:
            sentiment = "Positive"
        elif pos_score < neg_score:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        return sentiment
    else:
        return "Not found in SentiWordNet"

sentiwordnet = load_sentiwordnet(sentiwordnet_path)

word = "happy"
sentiment = analyze_sentiment(word, sentiwordnet)
print(f"The sentiment of '{word}' is {sentiment}")
