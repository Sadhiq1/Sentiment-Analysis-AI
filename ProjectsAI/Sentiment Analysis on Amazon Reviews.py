import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import re
from smart_open import open

def get_labels_and_texts(file):
    labels = []
    texts = []
    with open(file, 'rb') as f:
        for line in f:
            x = line.decode("utf-8", errors="ignore")
            labels.append(int(x[9]) - 1)
            texts.append(x[10:].strip())
    return np.array(labels), texts


train_labels, train_texts = get_labels_and_texts('train.ft.txt')
test_labels, test_texts = get_labels_and_texts('test.ft.txt')

train_labels = train_labels[:500]
train_texts = train_texts[:500]

NON_ALPHANUM = re.compile(r'[\W]')
NON_ASCII = re.compile(r'[^a-z0-1\s]')

def normalize_texts(texts):
    normalized_texts = []
    for text in texts:
        lower = text.lower()
        no_punctuation = NON_ALPHANUM.sub(r' ', lower)
        no_non_ascii = NON_ASCII.sub(r'', no_punctuation)
        normalized_texts.append(no_non_ascii)
    return normalized_texts


train_texts = normalize_texts(train_texts)
test_texts = normalize_texts(test_texts)

cv = CountVectorizer(binary=True)
cv.fit(train_texts)
X = cv.transform(train_texts)
X_test = cv.transform(test_texts)

X_train, X_val, y_train, y_val = train_test_split(X, train_labels, train_size=0.75)

for c in [0.01, 0.05, 0.25, 0.5, 1]:
    lr = LogisticRegression(C=c)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Accuracy for C={c}: {accuracy}")

y_test_pred = lr.predict(X_test)

for example_index in range(len(test_texts)):
    predicted_label = y_test_pred[example_index]
    actual_label = test_labels[example_index]
    accuracy = 1 if predicted_label == actual_label else 0
    sentiment = "positive" if predicted_label == 0 else "negative"

    print(f"\nExample {example_index + 1}:")
    print(f"Predicted Label: {predicted_label}")
    print(f"Actual Label: {actual_label}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Original Text: {test_texts[example_index]}")
    print(f"Sentiment: {sentiment}")
