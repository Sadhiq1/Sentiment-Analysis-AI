#Implement sentiment analysis using classifiers like Naive Bayes and Support Vector Machines

import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

#nltk.download("movie_reviews")
documents = [(list(movie_reviews.words(fileid)), category) for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)
all_words = [w.lower() for w in movie_reviews.words()]
all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:3000]

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

# Prepare the feature sets
featuresets = [(find_features(rev), category) for (rev, category) in documents]

train_set, test_set = train_test_split(featuresets, test_size=0.2, random_state=42)

NB_classifier = SklearnClassifier(MultinomialNB())
NB_classifier.train(train_set)
NB_accuracy = nltk.classify.accuracy(NB_classifier, test_set)
print("Naive Bayes Accuracy:", NB_accuracy)

SVM_classifier = SklearnClassifier(SVC())
SVM_classifier.train(train_set)
SVM_accuracy = nltk.classify.accuracy(SVM_classifier, test_set)
print("SVM Accuracy:", SVM_accuracy)

NB_predictions = [NB_classifier.classify(features) for (features, label) in test_set]
SVM_predictions = [SVM_classifier.classify(features) for (features, label) in test_set]

print("Naive Bayes Classification Report:")
print(classification_report([label for (_, label) in test_set], NB_predictions))

print("SVM Classification Report:")
print(classification_report([label for (_, label) in test_set], SVM_predictions))


