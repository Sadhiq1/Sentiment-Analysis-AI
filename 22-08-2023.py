#Task Name: Basic Sentiment Analysis using NLTK and Naive Bayes Classifier in Python

import nltk
import random
from nltk.corpus import movie_reviews

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())

word_features = list(all_words.keys())[:2000]

def document_features(document):
    words = set(document)
    features = {}
    for word in word_features:
        features[word] = (word in words)
    return features

featuresets = [(document_features(d), c) for (d, c) in documents]

train_set, test_set = featuresets[:1500], featuresets[1500:]

classifier = nltk.NaiveBayesClassifier.train(train_set)

accuracy = nltk.classify.accuracy(classifier, test_set)
print("Classifier Accuracy:", accuracy)

text_to_analyze = "I really loved this movie. It was amazing!"
features = document_features(text_to_analyze.split())
sentiment = classifier.classify(features)
print("Sentiment:", sentiment)

