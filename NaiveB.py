import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv(r"D:\Downloads\NaiveBdata.csv")

# Encode target variable 'label'
le = LabelEncoder()
df['class'] = le.fit_transform(df['label'])

# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X_tfidf = tfidf_vectorizer.fit_transform(df['text'])
y = df['class']

# Train Multinomial Naive Bayes
mnb = MultinomialNB()
y_pred_mnb = cross_val_predict(mnb, X_tfidf, y)
mnb.fit(X_tfidf, y)

# Print evaluation metrics
print("Multinomial Naive Bayes (TF-IDF)")
print("Accuracy:", accuracy_score(y, y_pred_mnb))
print("Recall:", recall_score(y, y_pred_mnb))
print("F1 Score:", f1_score(y, y_pred_mnb, average='macro'))
print("\nClassification Report:\n", classification_report(y, y_pred_mnb, target_names=["Ham", "Spam"], zero_division=0))

# Scatter plot of label vs message length
plt.figure(figsize=(8, 5))
plt.scatter(df['class'], df['text'].apply(len), alpha=0.5, c=df['class'], cmap='coolwarm')
plt.xlabel("Label (0 = Ham, 1 = Spam)")
plt.ylabel("Message Length")
plt.title("Scatter Plot of Label vs Message Length")
plt.grid(True)
plt.show()

# Check a sample message
print("\nCheck a message:")
sample_message = input("Enter a message to check if it's Spam or Ham: ")
sample_features = tfidf_vectorizer.transform([sample_message])
prediction = mnb.predict(sample_features)

if prediction[0] == 0:
    print("Prediction: Not Spam (Ham)")
else:
    print("Prediction: Spam")
