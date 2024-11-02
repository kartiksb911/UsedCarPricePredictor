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

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    data = {
        'vehicle_age': request.form['vehicle_age'],
        'km_driven': request.form['km_driven'],
        'mileage': request.form['mileage'],
        'engine': request.form['engine'],
        'max_power': request.form['max_power'],
        'seats': request.form['seats'],
        'seller_type': request.form['seller_type'],
        'fuel_type': request.form['fuel_type'],
        'transmission_type': request.form['transmission_type'],
        'brand': request.form['brand'],
        'model': request.form['model']
    }
    
    # Convert the data to a DataFrame
    input_df = pd.DataFrame([data])
    
    # Preprocess the data
    final_input = pickled_preprocessor.transform(input_df)
    
    # Make prediction
    output = pickled_model.predict(final_input)[0]
    
    # Render the home page with the prediction
    return render_template("home.html", prediction_text=f"The Vehicle Price Prediction is {output}")

if __name__ == '__main__':
    app.run(debug=True)
