#Task Name: Text Preprocessing for Sentiment Analysis in Python

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import re

sample_text = "I absolutely loved this movie! The acting was brilliant, and the plot was engaging. Highly recommended!"

sample_text = sample_text.lower()

sample_text = re.sub(r'[^a-zA-Z\s]', '', sample_text)

tokens = word_tokenize(sample_text)

stop_words = set(stopwords.words('english'))
filterted_tokens = [token for token in tokens if token not in stop_words]

lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filterted_tokens]

stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(token) for token in lemmatized_tokens]

preprocessed_text = ' '.join(stemmed_tokens)
print(preprocessed_text)
