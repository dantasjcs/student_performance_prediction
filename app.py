from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
import joblib

app = Flask(__name__)
model = joblib.load('model_trained.pkl')

@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        gender = int(request.form.get('gender'))
        parent = int(request.form.get('parent'))
        race = int(request.form.get('race'))
        lanche = int(request.form.get('lanche'))
        test_prep = int(request.form.get('test_prep'))
        math_score = int(request.form.get('math_score'))
        wrt_score = int(request.form.get('wrt_score'))
        X = np.array([gender, parent, race, lanche, test_prep, math_score, wrt_score]).reshape(1, -1)
        result = int(model.predict(X))
        proba = np.amax(model.predict_proba(X))*100
        return render_template("index.html", result=result, proba=proba)
    
    else:
        return render_template("index.html")