"""import nltk
from nltk.corpus import movie_reviews
import random

# Download the movie_reviews dataset (only required once)
nltk.download('movie_reviews')

# Prepare the dataset
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Shuffle the documents to ensure random distribution of positive and negative samples
random.shuffle(documents)

all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(all_words.keys())[:3000]

def document_features(document):
    words = set(document)
    features = {}
    for word in word_features:
        features[f'contains({word})'] = (word in words)
    return features
featuresets = [(document_features(d), c) for (d, c) in documents]

train_set, test_set = featuresets[:1600], featuresets[1600:]

#classifier = nltk.NaiveBayesClassifier.train(train_set)
from nltk.classify import SklearnClassifier
from sklearn.linear_model import LinearRegression

regression_model = SklearnClassifier(LinearRegression())
regression_model.train(train_set)

accuracy = nltk.classify.accuracy(regression_model, test_set)
print(f'Accuracy: {accuracy: .2f}')

sample_text = "The movie was entertaining and enjoyable."
sample_features = document_features(sample_text.split())
prediction = regression_model.classify(sample_features)

print(f"Predicted sentiment intensity: {prediction: .2f}")"""

import nltk
from nltk.corpus import movie_reviews
import random
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

# Download the movie_reviews dataset (only required once)
#nltk.download('movie_reviews')

# Prepare the dataset
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Shuffle the documents to ensure random distribution of positive and negative samples
random.shuffle(documents)

all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(all_words.keys())[:3000]

def document_features(document):
    words = set(document)
    features = {}
    for word in word_features:
        features[f'contains({word})'] = (word in words)
    return features

featuresets = [(document_features(d), 1 if c == 'pos' else 0) for (d, c) in documents]

train_set, test_set = featuresets[:1600], featuresets[1600:]

# Convert features to numerical arrays
def convert_to_array(featuresets):
    X, y = [], []
    for features, label in featuresets:
        X.append(list(features.values()))
        y.append(label)
    return np.array(X), np.array(y)

X_train, y_train = convert_to_array(train_set)
X_test, y_test = convert_to_array(test_set)

# Create and train the Linear Regression model
regression_model = LinearRegression()
regression_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = regression_model.predict(X_test)

# Convert predictions to binary labels
predictions = [1 if pred > 0.5 else 0 for pred in y_pred]

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy:.2f}')

# Sample text for prediction
sample_text = "The movie was entertaining and enjoyable."
sample_features = document_features(sample_text.split())
sample_input = np.array([list(sample_features.values())])
prediction = regression_model.predict(sample_input)

# Convert prediction to binary label
sentiment = "positive" if prediction > 0.5 else "negative"
print(f"Predicted sentiment: {sentiment}")
