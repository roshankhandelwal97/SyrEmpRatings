import pandas as pd
import joblib
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
import json

model = joblib.load('ratings/model.pkl')
scaler = joblib.load('ratings/scaler.pkl')
feature_columns = joblib.load('ratings/columns.pkl')
    
def prepare_input(data):
    """
    Prepare the input data for prediction by ensuring all features match the trained model's expectations.
    """
    # Convert dictionary to DataFrame
    input_df = pd.DataFrame([data])

    # Ensure all expected columns are present, even if not in incoming data, fill them with zeros
    for column in feature_columns:
        if column not in input_df.columns:
            input_df[column] = 0
    
    # Order columns as the model expects
    input_df = input_df[feature_columns]

    # Apply scaling
    input_df_scaled = scaler.transform(input_df)
    
    return input_df_scaled

def predict(data):
    prepared_data = prepare_input(data)
    prediction = model.predict(prepared_data)
    return prediction[0]
