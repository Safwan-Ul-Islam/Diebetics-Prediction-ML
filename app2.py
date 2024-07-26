from flask import Flask, request, render_template, render_template_string
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and scaler
model_filename = '/Users/safwan/Desktop/dataset/diebatics/diabetes_model.joblib'
scaler_filename = '/Users/safwan/Desktop/dataset/diebatics/scaler.joblib'

model = joblib.load(model_filename)
scaler = joblib.load(scaler_filename)

# Define HTML template for displaying results
result_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Diabetes Prediction Result</title>
</head>
<body>
    <h1>Diabetes Prediction Result</h1>
    <p>Prediction: {{ prediction }}</p>
    <a href="/">Go Back</a>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract input features from form
    features = [
        float(request.form['Pregnancies']),
        float(request.form['Glucose']),
        float(request.form['BloodPressure']),
        float(request.form['SkinThickness']),
        float(request.form['Insulin']),
        float(request.form['BMI']),
        float(request.form['DiabetesPedigreeFunction']),
        float(request.form['Age'])
    ]

    # Standardize input data
    features_standardized = scaler.transform([features])

    # Make prediction
    prediction = model.predict(features_standardized)
    
    # Display result
    result = "Positive" if prediction[0] == 1 else "Negative"
    return render_template_string(result_template, prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
