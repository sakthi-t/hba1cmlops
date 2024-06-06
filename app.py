import warnings
warnings.filterwarnings("ignore")

import mlflow
import mlflow.sklearn
import pandas as pd
from datetime import datetime
from flask import Flask, request, render_template

app = Flask(__name__, template_folder="templates")

# Load patient data
df_patients = pd.read_csv("Data/dim_patients_final_rev01.csv")
patient_ids = df_patients['patient_id'].tolist()

# Set the MLflow tracking URI to DagsHub
mlflow.set_tracking_uri("https://dagshub.com/sakthi-t/hba1cmlops.mlflow")

# Load the model from MLflow model registry
model_name = "random_forest_model"
model_version = 3
model_uri = f"models:/{model_name}/{model_version}"
loaded_model = mlflow.sklearn.load_model(model_uri)

def predict_hba1c(patient_id, visited_date, sugar):
    # Prepare input data
    visited_date = pd.to_datetime(visited_date)
    data = {
        'patient_id': [patient_id],
        'sugar': [sugar],
        'year': [visited_date.year],
        'month': [visited_date.month],
        'day': [visited_date.day]
    }
    input_df = pd.DataFrame(data)
    
    # Make prediction
    prediction = loaded_model.predict(input_df)
    return prediction[0]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        patient_id = request.form['patient_id']
        visited_date = request.form['visited_date']
        sugar = float(request.form['sugar'])
        prediction = predict_hba1c(patient_id, visited_date, sugar)
    
    return render_template('index.html', patient_ids=patient_ids, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
