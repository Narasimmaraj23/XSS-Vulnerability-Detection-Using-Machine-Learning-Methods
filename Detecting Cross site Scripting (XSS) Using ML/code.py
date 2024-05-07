import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings("ignore")
# data read
df = pd.read_csv('xss data.csv', encoding="latin-1")

X = df['Files']
y = df['Label']

cv = CountVectorizer()
X = cv.fit_transform(X)  # Fit the Data

# split the data  train and  test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Naive Bayes Classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

clf.score(X_test, y_test)
y_pred = clf.predict(X_test)

print('Accuracy of MultinomialNB: ', accuracy_score(y_pred, y_test))

print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))

# Compute confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-XSS', 'XSS'], yticklabels=['Non-XSS', 'XSS'])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

import joblib
joblib.dump(clf, 'NB_XSS_model.pkl')

NB_XSS_model = open('NB_XSS_model.pkl', 'rb')
clf = joblib.load(NB_XSS_model)

from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    df = pd.read_csv("xss data.csv", encoding="latin-1")

    X = df['Files']
    y = df['Label']

    # Extract Feature With CountVectorizer
    cv = CountVectorizer()
    X = cv.fit_transform(X)  # Fit the Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Naive Bayes Classifier
    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    if request.method == 'POST':
        message = request.form['text']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
    from waitress import serve
    app.run()
