#Task : Text Preprocessing using NLTK: Cleaning and Transforming Text Data
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')

text = "Text preprocessing is an important step in natural language processing. It involves cleaning and preparing text data."

text = text.lower()

words = word_tokenize(text)

words = [word for word in words if word.isalnum()]

stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word not in stop_words]

lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]

print("Original Text:", text)
print("Tokenized Words:", words)
print("Filtered Words (without stopwords and punctuation):", filtered_words)
print("Lemmatized Words:", lemmatized_words)
