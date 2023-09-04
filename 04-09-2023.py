import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset with the specified encoding
data = pd.read_csv('test.csv', encoding='latin1')

# Clean and preprocess the text data (you can customize this step)
data['text'] = data['text'].str.lower()
data['text'] = data['text'].str.replace('[^\w\s]', '')  # Remove punctuation

# Handle missing values in the text data by filling them with an empty string ('')
data['text'].fillna('', inplace=True)

X = data['text']
y = data['sentiment']

# Handle missing values in the target labels by removing corresponding rows
# or replacing them with appropriate values
y.dropna(inplace=True)

# Ensure that X and y have the same number of samples
X = X.iloc[y.index]  # Filter X based on the indices of non-missing y values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Generate a classification report
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)

# Create a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)
