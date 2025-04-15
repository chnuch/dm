import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv(r"D:\Downloads\NaiveBdata.csv")

df.head()

df.columns

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['class']=le.fit_transform(df['label'])

df.head()

ve=TfidfVectorizer(stop_words='english')
x=ve.fit_transform(df['text'])
y=df['class']

mn=MultinomialNB()
mn.fit(x,y)

y_pred=cross_val_predict(mn,x,y)

from sklearn.metrics import accuracy_score,recall_score,f1_score,classification_report

print("Multinomial Naive Bayes:\n")
print("accuracy: ",accuracy_score(y,y_pred))
print("recall: ",recall_score(y,y_pred))
print("f1_score: ",f1_score(y,y_pred,average='macro'))
print("\nClassification Report:\n", classification_report(y, y_pred, target_names=["Ham","spam"], zero_division=0))

import matplotlib.pyplot as plt
plt.figure(figsize=(8,5))
plt.scatter(df['class'], df['text'].apply(len), alpha=0.5, c=df['class'], cmap='coolwarm')
plt.xlabel("Label (0 = Ham, 1 = Spam)")
plt.ylabel("Message Length")
plt.title("Scatter Plot of Label vs Message Length")
plt.grid(True)
plt.show()

from sklearn.feature_extraction.text import CountVectorizer

ve1=CountVectorizer(binary=True,stop_words='english')
x1=ve1.fit_transform(df['text'])
y1=df['class']

from sklearn.naive_bayes import BernoulliNB
bm=BernoulliNB()

y_pred1=cross_val_predict(bm,x1,y1)

print("Bernouli Naive Bayes:\n")
print("accuracy: ",accuracy_score(y,y_pred))
print("recall: ",recall_score(y,y_pred))
print("f1_score: ",f1_score(y,y_pred,average='macro'))
print("\nClassification Report:\n", classification_report(y, y_pred, target_names=["Ham","spam"], zero_division=0))

import matplotlib.pyplot as plt
plt.figure(figsize=(8,5))
plt.scatter(df['class'], df['text'].apply(len), alpha=0.5, c=df['class'], cmap='coolwarm')
plt.xlabel("Label (0 = Ham, 1 = Spam)")
plt.ylabel("Message Length")
plt.title("Scatter Plot of Label vs Message Length")
plt.grid(True)
plt.show()

sample_message = ["Congratulations! You've won a free ticket to Bahamas. Reply WIN to claim now."]
sm = ve1.transform(sample_message)
re = mn.predict(sm)
re[0]

sample_message = input("Input Text: ")
sm = ve1.transform([sample_message])
re = mn.predict(sm)
if re[0]==0:
    print("Not Spam")
else:
    print("Spam!")
