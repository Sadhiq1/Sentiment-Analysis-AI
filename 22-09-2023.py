import nltk
import random
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize

documents = []

for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        words = movie_reviews.words(fileid)
        documents.append((words, category))

random.shuffle(documents)

all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:3000]  # Use the 3000 most common words as features

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]

training_set = featuresets[:1900]
testing_set = featuresets[1900:]

classifier = nltk.NaiveBayesClassifier.train(training_set)

accuracy = nltk.classify.accuracy(classifier, testing_set)
print("Classifier Accuracy:", accuracy)

text_to_predict = "This movie was fantastic! I loved it."
words = word_tokenize(text_to_predict)
features = find_features(words)
sentiment = classifier.classify(features)
print("Sentiment:", sentiment)