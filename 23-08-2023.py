#Task Name: Text Classification with Bag-of-Words (BoW) and TF-IDF Feature Extraction

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load the IMDb movie reviews dataset from CSV using Pandas
movie_reviews = 'C:/Users/SSLTP11505/Desktop/AI/IMDB Movie reviews/IMDB Dataset.csv'
data = pd.read_csv(movie_reviews)

X = data['review'].values
y = data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0).values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Text preprocessing and vectorization
count_vectorizer = CountVectorizer(max_features=5000, stop_words='english')
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

X_train_bow = count_vectorizer.fit_transform(X_train)
X_test_bow = count_vectorizer.transform(X_test)

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train Naive Bayes classifiers
nb_classifier_bow = MultinomialNB()
nb_classifier_bow.fit(X_train_bow, y_train)

nb_classifier_tfidf = MultinomialNB()
nb_classifier_tfidf.fit(X_train_tfidf, y_train)

# Make predictions and calculate accuracy
y_pred_bow = nb_classifier_bow.predict(X_test_bow)
y_pred_tfidf = nb_classifier_tfidf.predict(X_test_tfidf)

accuracy_bow = accuracy_score(y_test, y_pred_bow)
accuracy_tfidf = accuracy_score(y_test, y_pred_tfidf)

print(f"Accuracy using BoW: {accuracy_bow:.2f}")
print(f"Accuracy using TF-IDF: {accuracy_tfidf:.2f}")
