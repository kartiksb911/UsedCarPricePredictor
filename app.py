import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the pickled model and preprocessor
pickled_model = pickle.load(open('Model.pkl', 'rb'))
pickled_preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    # Get data from the request
    data = request.json['data']
    
    # Convert the data to a DataFrame
    input_df = pd.DataFrame([data])
    
    # Preprocess the data
    new_data = pickled_preprocessor.transform(input_df)
    
    # Make prediction
    output = pickled_model.predict(new_data)
    return jsonify(output[0])

if __name__ == '__main__':
    app.run(debug=True)
