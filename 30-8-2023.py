import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

t = np.linspace(0, 10, 500)
sine_wave = np.sin(t)

X = sine_wave[:-1]
y = sine_wave[1:]

X = X.reshape(-1, 1, 1)

model = Sequential()
model.add(LSTM(1, input_shape=(1, 1)))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X, y, epochs=10, batch_size=1, verbose=2)

seed = np.array([0.0]).reshape(1, 1, 1)  # Starting point
predicted_values = []

for _ in range(499):  # Generate predictions for the entire dataset
    prediction = model.predict(seed, verbose=0)
    predicted_values.append(prediction[0, 0])  # Extract the predicted value
    seed = prediction.reshape(1, 1, 1)  # Set the seed for the next prediction

predicted_values = np.array(predicted_values)

import matplotlib.pyplot as plt

plt.plot(t[1:], y, label='Actual Data', color='b')
plt.plot(t[1:], predicted_values, label='LSTM Predictions', color='r')
plt.legend()
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.title('LSTM Sequence Prediction')
plt.show()
