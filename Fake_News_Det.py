from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
import pandas as pd
import contractions
from sklearn.model_selection import train_test_split
import contractions
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

app = Flask(__name__)

# Load the pre-trained model and vectorizer
tfvect = pickle.load(open(r'C:\Users\MSI GF63\Documents\UiTM Stuff\Final Year Project\Fake News Detection Model\tfidfvect.pkl', 'rb'))
loaded_model = pickle.load(open(r'C:\Users\MSI GF63\Documents\UiTM Stuff\Final Year Project\Fake News Detection Model\model.pkl', 'rb'))

def fake_news_det(news):
    # Transform the training data
    #tfid_x_train = tfvect.fit_transform(x_train)
    # Transform the test data
    #tfid_x_test = tfvect.transform(x_test)
    # Transform the input data
    input_data = [news]
    vectorized_input_data = tfvect.transform(input_data)
    # Make prediction
    prediction = loaded_model.predict(vectorized_input_data)
    return prediction

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        pred = fake_news_det(message)
        return render_template('index.html', prediction=pred)
    else:
        return render_template('index.html', prediction="Something went wrong")

if __name__ == '__main__':
    app.run(debug=True)