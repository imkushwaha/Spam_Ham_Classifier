from flask import Flask, render_template, request
import requests
import pickle
import numpy as np
import sklearn
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

app = Flask(__name__)
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


#@app.route('/about')
#def about():
#    return render_template('about.html')

#@app.route('/contact')
#def contact():
#    return render_template('contact.html')

@app.route("/predict", methods=['POST'])
def predict():
    
    if request.method == 'POST':
        text = request.form['message']
    
        # 1. preprocess
        transformed_sms = transform_text(text)
        # 2. vectorize
        vector_input = tfidf.transform([transformed_sms])
        # 3. predict
        result = model.predict(vector_input)[0]
        
        if result==1:
            prediction = "Your Message is Spam"
        else:
            prediction = "Your Message is not Spam"   
            

        return render_template('index.html', prediction_text=prediction)
    
if __name__=="__main__":
    app.run(debug=True)