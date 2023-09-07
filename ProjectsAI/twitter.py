import pandas as pd
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv('twitter_data.csv')

df['tweet_text'] = df['tweet_text'].astype(str).fillna('')

df['sentiment'] = df['tweet_text'].apply(lambda tweet: TextBlob(tweet).sentiment.polarity)

X = df[['sentiment']]
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regression_model = LinearRegression()

regression_model.fit(X_train, y_train)

y_pred = regression_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

df['predicted_sentiment'] = regression_model.predict(X)
print(df[['tweet_text', 'sentiment', 'predicted_sentiment']])