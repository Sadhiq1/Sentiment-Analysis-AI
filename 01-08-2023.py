import nltk
from nltk.corpus import movie_reviews
import random

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
featuresets = [(document_features(d), c) for (d, c) in documents]

train_set, test_set = featuresets[:1600], featuresets[1600:]

classifier = nltk.NaiveBayesClassifier.train(train_set)

accuracy = nltk.classify.accuracy(classifier, test_set)
print(f'Accuracy: {accuracy: .2f}')

sample_text = "The movie was entertaining and enjoyable."
sample_features = document_features(sample_text.split())
prediction = classifier.classify(sample_features)

print(f"Predicted sentiment: {prediction}")

