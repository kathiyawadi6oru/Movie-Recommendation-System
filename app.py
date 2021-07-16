import pickle
import numpy as np
from flask import Flask, jsonify, render_template, request
from model import recommend_movie

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = request.form.get('moviename')
    output = recommend_movie(int_features)
   # output = model(int_features)
   # output = model(movie=int_features)

    return render_template('index.html',enter='{}'.format(int_features), prediction_text='{}'.format(output),data=output,
    len = len(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    output = model(data.values())

    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
