#Text Preprocessing using NLTK in Python
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

sample_text = "Text preprocessing is a crucial step in natural language processing. It involves cleaning and preparing raw text data to make it suitable for analysis and modeling. Let's see how to perform text preprocessing using Python!"

sample_text_lower = sample_text.lower()
print("Lowercased text:")
print(sample_text_lower)

words = word_tokenize(sample_text_lower)
print("Tokenized words:")
print(words)

stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word not in stop_words]
print("Words after stopword removal:")
print(filtered_words)

import string
def remove_punctuation(text):
    translator = str.maketrans('','', string.punctuation)
    return text.translate(translator)

sample_text_no_punct = remove_punctuation(sample_text_lower)
print("Text after punctuation removal:")
print(sample_text_no_punct)

stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in filtered_words]
print("Words after stemming:")
print(stemmed_words)

lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
print("Words after lemmatization:")
print(lemmatized_words)


def preprocess_text(text):
    text_lower = text.lower()
    words = word_tokenize(text_lower)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    text_no_punct = remove_punctuation(" ".join(filtered_words))
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in filtered_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]

    return lemmatized_words


processed_text = preprocess_text(sample_text)
print("Final processed text:")
print(" ".join(processed_text))
