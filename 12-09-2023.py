import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

X, y = make_classification(n_classes=2, weights=[0.9, 0.1], n_samples=1000, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print("Model Performance without Balancing:")
print(classification_report(y_test, y_pred))

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

classifier_balanced = RandomForestClassifier(random_state=42)
classifier_balanced.fit(X_resampled, y_resampled)

y_pred_balanced = classifier_balanced.predict(X_test)

print("Model Performance with SMOTE Balancing:")
print(classification_report(y_test, y_pred_balanced))

