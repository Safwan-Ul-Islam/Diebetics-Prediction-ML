from flask import Flask, request, render_template, render_template_string
import joblib
import numpy as np

app = Flask(__name__, static_folder='/Users/safwan/diebatics/static/image')

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
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Poppins', sans-serif;
            background-color: #f2f2f2;
            margin: 0;
            padding: 0;
            background-image: url('static/image/hand-drawn-world-diabetes-day-background_23-2149099905.jpg.avif');
            
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }
        .container {
            max-width: 500px;
            padding: 30px;
            background: rgba(255, 255, 255, 0.944);
            border-radius: 12px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
            text-align: center;
        }
        h1 {
            color: #333;
            font-weight: 600;
            margin-bottom: 20px;
            font-size: 24px;
        }
        p {
            font-size: 18px;
            color: #555;
            margin: 10px 0;
        }
        .result {
            font-size: 22px;
            font-weight: 600;
            color: #4CAF50;
            margin-top: 20px;
        }
        a {
            display: inline-block;
            margin-top: 20px;
            padding: 10px 20px;
            background: linear-gradient(90deg, #4CAF50, #81C784);
            color: #fff;
            text-decoration: none;
            border-radius: 8px;
            font-size: 16px;
            transition: background 0.3s ease, transform 0.2s ease;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        }
        a:hover {
            background: linear-gradient(90deg, #45a049, #66bb6a);
            transform: scale(1.05);
        }
        a:active {
            transform: scale(1.0);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Diabetes Prediction Result</h1>
        <p>Your prediction result is:</p>
        <p class="result">{{ prediction }}</p>
        <a href="/">Go Back</a>
    </div>
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
