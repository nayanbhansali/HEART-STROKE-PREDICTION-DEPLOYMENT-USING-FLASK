from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('dt.pkl', 'rb'))

# Initialize the Flask application
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Collect the form data
        gender = int(request.form['gender'])
        age = float(request.form['age'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        ever_married = int(request.form['ever_married'])
        work_type = int(request.form['work_type'])
        Residence_type = int(request.form['Residence_type'])
        avg_glucose_level = float(request.form['avg_glucose_level'])
        bmi = float(request.form['bmi'])
        smoking_status = int(request.form['smoking_status'])

        # Create a feature array for prediction
        features = np.array([[gender, age, hypertension, heart_disease, ever_married,
                              work_type, Residence_type, avg_glucose_level, bmi, smoking_status]])
        
        # Make a prediction
        prediction = model.predict(features)

        # Render the result
        return render_template('index.html', prediction_text=f'Stroke Prediction: {"Yes" if prediction[0] == 1 else "No"}')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
