import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('sample_data.csv')

print(data.head())

print(data.info())

print(data.describe())

print(data.isnull().sum())

plt.hist(data['age'], bins=20)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

data['age'].fillna(data['age'].mean(), inplace=True)

data.drop_duplicates(inplace=True)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data[['age', 'income']] = scaler.fit_transform(data[['age', 'income']])

data = pd.get_dummies(data, columns=['gender'], drop_first=True)

data['age_squared'] = data['age'] ** 2

print(data.head())
plt.scatter(data['age'], data['income'])
plt.title('Age vs. Income')
plt.xlabel('Age')
plt.ylabel('Income')
plt.show()

correlation_matrix = data.corr()
print(correlation_matrix)

data.to_csv('cleaned_data.csv', index=False)

