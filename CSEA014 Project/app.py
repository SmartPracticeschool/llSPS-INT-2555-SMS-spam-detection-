# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 18:53:49 2020

@author: user
"""

from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    df= pd.read_csv("SMSSpamCollection", encoding="latin-1" ,  sep='\t',names=["label", "message"])
	# Features and Labels
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    X = df['message']
    y = df['label']
	
	# Extract Feature With CountVectorizer
    cv = CountVectorizer()
    X = cv.fit_transform(X) # Fit the Data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	
    from sklearn.naive_bayes import MultinomialNB

    model = MultinomialNB()
    model.fit(X_train,y_train)
    model.score(X_test,y_test)
	
    joblib.dump(model, 'NB_spam_model.pkl')
    NB_spam_model = open('NB_spam_model.pkl','rb')
    model = joblib.load(NB_spam_model)

    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = model.predict(vect)
    return render_template('result.html',prediction = my_prediction)
   
if __name__ == '__main__':
    app.run(debug=True)
   
 


