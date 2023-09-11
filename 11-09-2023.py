text_data = ["Text data preprocessing is important.",
             "It includes tokenization, stemming, and more.",
             "Cleaning nosiy data can improve NLP models.",
             "Dealing with abbreviations like 'NLP' is a challenge.",
             "Let's clean this unstructured text!"]
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re

tokenized_data = [word_tokenize(sentence) for sentence in text_data]
print("Tokenized Data:")
print(tokenized_data)

lowercased_data = [[word.lower() for word in sentence] for sentence in tokenized_data]
print("\nLowercased Data:")
print(lowercased_data)

cleaned_data = [[word for word in sentence if re.match('^[a-zA-Z]+$', word)] for sentence in lowercased_data]
print("\nData with Punctuation Removed:")
print(cleaned_data)

#nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
filtered_data = [[word for word in sentence if word not in stop_words] for sentence in cleaned_data]
print("\nData with Stop Words Removed:")
print(filtered_data)

stemmer = PorterStemmer()
stemmed_data = [[stemmer.stem(word) for word in sentence] for sentence in filtered_data]
print("\nStemmed Data:")
print(stemmed_data)

preprocessed_text = [' '.join(sentence) for sentence in stemmed_data]
print("\nPreprocessed Text:")
print(preprocessed_text)
