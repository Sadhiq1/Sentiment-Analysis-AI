import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

reviews = [
    ("The camera quality is excellent.", "positive"),
    ("Battery life is terrible.", "negative"),
    ("Great phone overall!", "positive")
]

#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

preprocessed_reviews = [(preprocess(text), sentiment) for text, sentiment in reviews]

corpus = [text for text, sentiment in preprocessed_reviews]
labels = [sentiment for text, sentiment in preprocessed_reviews]

vectorizer = TfidfVectorizer(max_features=1000)  # You can adjust the number of features
X = vectorizer.fit_transform(corpus).toarray()

clf = MultinomialNB()
clf.fit(X, labels)

new_text = "The screen resolution is impressive."
preprocessed_new_text = preprocess(new_text)
new_text_vector = vectorizer.transform([preprocessed_new_text]).toarray()
predicted_sentiment = clf.predict(new_text_vector)
print(f"Predicted sentiment: {predicted_sentiment[0]}")

test_reviews = [
    ("The screen resolution is amazing.", "positive"),
    ("The battery drains quickly.", "negative"),
]

preprocessed_test_reviews = [(preprocess(text), sentiment) for text, sentiment in test_reviews]
test_corpus = [text for text, sentiment in preprocessed_test_reviews]
test_labels = [sentiment for text, sentiment in preprocessed_test_reviews]

test_X = vectorizer.transform(test_corpus).toarray()
predictions = clf.predict(test_X)

print(classification_report(test_labels, predictions))
