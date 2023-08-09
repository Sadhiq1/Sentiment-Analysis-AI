#Task: Text Preprocessing and Analysis with NLTK: Tokenization, Stopword Removal, Stemming, and Lemmatization

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')

text = "Tokenization is the process of breaking down a text into smaller units like words or sentences."

tokens = word_tokenize(text)
print("Tokens:", tokens)

stop_words = set(stopwords.words('english'))
filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
print("Filtered Tokens (After Stopword Removal):", filtered_tokens)

stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
print("Stemmed Tokens:", stemmed_tokens)

lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
print("Lemmatized Tokens:", lemmatized_tokens)
