import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)

classify=pickle.load(open('classify.pkl','rb'))
processdata=pickle.load(open('pipeline.pkl','rb'))

def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    data=processdata.transform(data)
    output=classify.predict(data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST']) 
def predict():
    data=[float(x) for x in request.form.values()] 
    data=processdata.transform(data)
    output=classify.predict(data)[0]
    return render_template("home.html",prediction_text="The song will {}".format(output))